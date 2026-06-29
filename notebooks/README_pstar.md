# p\* drift-budget-law experiment — Colab runbook

**Goal (one plot).** Test the closed-form prediction

> **p\* ≈ (η · ‖ḡ‖) / R**, with **R** a per-architecture constant.

Hold the architecture fixed, vary CIFAR-10-C **severity** (1→5) as the shift-magnitude
axis. If the law holds, **p\* vs (η·‖ḡ‖)** is a straight line through the origin
(slope = 1/R) — one independent line per architecture (`resnet18`, `wrn28_10`).

- **p** = `restore_prob`, the Bernoulli source-restore "tether" in HEAT.
- **p\*** = the minimum p at which the continual run does *not* collapse.
- **‖ḡ‖** = online stationary gradient norm of the adaptation loss (`grad_l2`), measured
  from a known-stable reference run.
- **η** = HEAT learning rate, fixed at `1e-3`.

You write nothing — open `notebooks/pstar_colab.ipynb` in Colab and run top-to-bottom.

---

## What to edit before running

Everything is in the **`# === EDIT ME ===`** cell (cell 2). The only fields you normally
touch:

| Variable | What it is | Default / what to do |
|---|---|---|
| `REPO_URL` | Git URL to clone | `https://github.com/octadion/heat.git`. If your fork/branch differs, change it (and set `GIT_BRANCH`). Or upload the repo manually and set `REPO_DIR`. |
| `USE_DRIVE` | Mount Drive so results survive a disconnect | `True` (**strongly recommended** — the sweep takes hours). |
| `DRIVE_RESULTS_DIR` | Where run JSONs + analysis land | `/content/drive/MyDrive/pstar_results`. Pick any Drive path. |
| `CKPT_RESNET18`, `CKPT_WRN` | Source checkpoints | See **Checkpoints** below. Leave `""` + `TRAIN_IF_MISSING=True` to train. |
| `TRAIN_IF_MISSING` | Train source models if no checkpoint given | `False`. Set `True` only if you have no checkpoints (WRN training ≈ 4× ResNet). |
| `ARCHS` | Architectures (= number of lines) | `["resnet18", "wrn28_10"]`. |
| `SEVERITIES` | Shift-magnitude axis points, **per architecture** | `{"resnet18": [1,3,5], "wrn28_10": [1,2,3,4,5]}` — WRN is the decisive line so it gets full density; ResNet uses the cheaper set. Set it to a plain list (e.g. `[1,3,5]`) to apply the same severities to every arch. |
| `P_GRID` | Coarse tether grid | `[0.0, 0.005, 0.010, 0.020]`. |
| `BISECT_STEPS` | Bisections of the stable/collapse bracket | `2` ⇒ p\* resolved to ≈ ±0.0025. |
| `SEED` | Seed | `42`. |

`GPU` choice: **Runtime → Change runtime type → GPU**. L4 or A100 recommended.

### Checkpoints (`CKPT_RESNET18`, `CKPT_WRN`)
Each accepts, transparently:
- a **direct `http(s)` URL** → downloaded with `wget`;
- a **Google-Drive file id** *or* **share URL** → downloaded with `gdown`;
- a **local path** already on disk (e.g. on Drive) → used as-is;
- **`""`** (blank) → train from scratch if `TRAIN_IF_MISSING=True`, else the cell stops.

Resolved checkpoints are normalized to `experiments/checkpoints/<arch>_final.pt` and the
final paths are echoed. The **WRN checkpoint is always required** because the mandatory
pipeline-validation gate (below) runs on WRN — even if you drop WRN from `ARCHS`.

---

## The run order (cells)

1. **GPU check** — prints `nvidia-smi`; warns if no GPU.
2. **Config** — the `# === EDIT ME ===` cell.
3. **Mount Drive** — and create `RESULTS_DIR`.
4. **Clone + install** — clones the repo, `pip install -r requirements.txt`, sets `PYTHONPATH`.
5. **CIFAR-10-C** — downloads the Zenodo tar and **verifies all 5 severities** (50000 imgs/corruption).
6. **Checkpoints** — resolves/downloads/trains; echoes paths.
7. **Pipeline validation (gate)** — reproduces the known WRN sev-5 anchor. **Stops if it fails.**
8. **Sweep** — the real experiment. Resumable.
9. **Analyze** — fits the line, shows the plot + verdict inline.

### Pipeline-validation gate (cell 7) — do not skip
Runs WRN-28-10 at severity 5 for `p ∈ {0, 0.005, 0.010}` and asserts the **known anchor**:
`p=0 → NaN ~step 603`, `p=0.005 → NaN ~step 920`, `p≥0.010 → stable` with stationary
‖ḡ‖ ≈ 13. Exact step numbers vary with hardware/seed, so the gate checks the *pattern*
(p0 collapses; p005 collapses later; p010 stable; ‖ḡ‖ in [8, 18]). **If it fails, the
cell raises** — fix the cause (wrong checkpoint, missing severities, no GPU,
`--heat-diagnostic-snapshot` not honored) before trusting any sweep output. These three
runs are part of the real sweep, so they are *reused*, not wasted.

---

## Resumability (Colab sessions die)

- Every run's JSON is written to `RESULTS_DIR` (Drive-backed) **immediately** by
  `run_tier2.py`.
- The sweep **skips any (arch, severity, p) whose JSON already exists** — re-run cell 8
  after a disconnect and it continues.
- `resnet18` (cheaper) is processed first; severities in configured order, so partial
  progress is still useful.
- A single run that errors (e.g. OOM) is recorded as a `*.run_error.txt` sidecar and the
  sweep **continues**; re-running retries it.

---

## Expected wall-clock (default: 8 cells = ResNet ×3 + WRN ×5)

One continual run = the full 15-corruption stream ≈ 2,355 steps. Per (arch, severity) the
sweep does ≈ **5–7 runs**: 1 source + the coarse grid (overlapping the p_ref ladder) + up
to `BISECT_STEPS` bisections; high-severity WRN cells may add a few upward-extension runs
when the whole coarse grid collapses.

| GPU | ResNet-18 run | WRN-28-10 run | Default sweep (ResNet [1,3,5] + WRN [1,2,3,4,5]) |
|---|---|---|---|
| A100 | ~3–5 min | ~8–12 min | ~3.5–5 h |
| L4 | ~6–9 min | ~15–22 min | ~7–9 h |
| T4 | ~10–15 min | ~25–40 min | slow — prefer L4/A100 |

WRN dominates (it is both slower per run and gets all 5 severities). For a quicker first
pass, set `SEVERITIES = [1, 3, 5]` (≈ 6 cells, ~5–7 h on L4). Training a source model from
scratch (if `TRAIN_IF_MISSING`) adds ~20–40 min (ResNet) / ~1.5–3 h (WRN) at 30 epochs —
do this once and save the checkpoint to Drive.

---

## Outputs (in `RESULTS_DIR/analysis/`)

- **`pstar_law.json`** — table: `arch, severity, source_acc, p_ref_used, grad_norm_gbar,
  eta, eta_times_gbar, p_star, p_star_bracket_low/high, collapse_criterion,
  stable_mean_acc` + per-arch fits + verdict.
- **`pstar_law.png`** — the deliverable plot: x = η·‖ḡ‖, y = p\*, one series per arch,
  least-squares fit overlaid with slope (=1/R), intercept, R².
- **`pstar_verdict.md`** — numeric verdict, one of:
  - **LAW SUPPORTED** — both archs R² ≥ ~0.9 and small intercept vs y-range.
  - **LAW PARTIAL** — linear within an arch but intercept non-trivial / one arch clean
    and one not (names the cause).
  - **LAW NOT SUPPORTED** — points scatter or p\*/‖ḡ‖ non-monotone in severity.

The plot and verdict also display inline at the bottom of the notebook.

---

## How p\* and ‖ḡ‖ are computed (so the result is auditable)

- **‖ḡ‖** (`scripts/pstar_common.grad_norm_gbar`): mean of `grad_l2` over local steps
  **50–150** within each corruption block, averaged across all 15 blocks; blocks with any
  NaN/inf are excluded. Measured from a **stable reference run** `p_ref` (0.005 ResNet /
  0.010 WRN, escalated up a fallback ladder if that itself collapses at high severity —
  the ladder rung used is reported as `p_ref_used`).
- **Collapse criterion** (`classify_run`, section 6.2, priority order): **hard** = any
  NaN/inf in `grad_l2`/`energy`/`drift_l2`, or stream `mean_accuracy ≤ 0.12`; **soft** =
  `last_accuracy` < source (no-adapt) accuracy, or stationary `drift_l2` > 5× its value
  at `p_ref`.
- **p\*** (`choose_pstar`, section 6.3): the midpoint of the final
  `[p_collapse, p_stable]` bracket after bisection. ResNet at low severity legitimately
  gives **p\*≈0** (no collapse even at p=0) — a real point consistent with the law, not
  discarded. At high severity, where the whole coarse grid collapses, the sweep folds in
  the stronger p_ref-ladder rungs already on disk and, if even those collapse, **extends
  the grid upward** (0.03→0.12, capped at ~5 runs) until a stable p is found, so the
  high-‖ḡ‖ points — which carry the slope — still get a real bracket to bisect. If no
  tested tether is stable, that cell's p\* is left **unresolved** ("law region
  exhausted") and the plot shows it as a **censored up-arrow** lower bound (excluded from
  the fit), never hidden.

**No p-hacking:** all points are reported and the fit spans all of them. The analysis
script is the single source of truth and recomputes p\* from every run on disk.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Validation gate fails: `p010_gbar_in_band` | Wrong/under-trained WRN checkpoint, or not on GPU. Confirm `wrn28_10_final.pt` is the real source model. |
| Validation gate fails: nothing collapses | `--heat-diagnostic-snapshot` not producing `drift_l2`/NaNs, or severity 5 data missing. Re-run cell 5; check 50000 imgs/corruption. |
| `FileNotFoundError ... .npy` | CIFAR-10-C not fully downloaded. Re-run cell 5. |
| Sweep restarts from scratch after disconnect | `RESULTS_DIR` not on Drive (`USE_DRIVE=False`) — results were on ephemeral local disk. Set `USE_DRIVE=True`. |
| OOM on WRN | Lower `BATCH_SIZE` (e.g. 32) in the config cell; the run is recorded as `run_error` and retried on re-run. |
| Verdict says NOT SUPPORTED / PARTIAL | That may be the honest answer. Read `pstar_verdict.md` — it names the observed cause (e.g. ‖ḡ‖ not monotone, soft-collapse dominating). Densifying `SEVERITIES` to `[1,2,3,4,5]` adds points. |
