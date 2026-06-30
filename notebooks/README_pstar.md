# p\* drift-budget-law experiment — Colab runbook (Phase A: eta-sweep)

**Goal (one plot).** Test the closed-form prediction

> **p\* ≈ (η · ‖ḡ‖) / R**, with **R** a per-architecture constant.

**Why we sweep η, not severity.** The earlier severity sweep was inconclusive: across
CIFAR-10-C severity 1→5, ‖ḡ‖ moves only ~14% (WRN) / ~21% (ResNet), so the predicted p\*
movement is **sub-grid and unobservable**. So we pull a stronger lever on the x-axis: at
fixed arch + severity, ‖ḡ‖ is ~constant near the source, so sweeping **η** moves
η·‖ḡ‖ ~20× and the law predicts **p\* scales linearly with η** — a straight line through
the origin, slope 1/R. That single plot is the decisive go/no-go.

- **Phase A (default):** `wrn28_10`, **severity 5 fixed**, `ETAS = {2e-4, 5e-4, 1e-3,
  2e-3, 4e-3}`. Predicted p\* spread ~0.0019 → ~0.038 (a ~20× range, far above grid
  resolution — that is why this works where severity did not).
- **Phase B (later, only if WRN is clean):** add `resnet18` at sev5 over the same η grid
  for a second, independent R.

- **p** = `restore_prob`, the Bernoulli source-restore "tether" in HEAT.
- **p\*** = the minimum p at which the continual run does *not* collapse.
- **‖ḡ‖** = online stationary gradient norm of the adaptation loss (`grad_l2`), measured
  from a known-stable reference run (per cell).
- **η** = HEAT learning rate — now the **swept lever** (`--heat-lr` per run), not fixed.
  A "cell" is `(arch, severity, η)`; its filename is η-tagged so etas never collide.

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
| `CKPT_DIR`, `CKPT_RESNET18`, `CKPT_WRN` | Source checkpoints | Default to `MyDrive/heat/experiments/checkpoints/<arch>_final.pt`. If yours live elsewhere, edit `CKPT_DIR`. See **Checkpoints** below; leave the paths `""` + `TRAIN_IF_MISSING=True` to train instead. |
| `TRAIN_IF_MISSING` | Train source models if no checkpoint given | `False`. Set `True` only if you have no checkpoints (WRN training ≈ 4× ResNet). |
| `ARCHS` | Architectures (= number of lines) | Phase A: `["wrn28_10"]`. Phase B: add `"resnet18"`. |
| `SEVERITIES` | Shift severity (now held fixed) | `[5]`. A plain list applies to all archs; a per-arch dict `{arch: [...]}` also works. |
| `ETAS` | **The swept lever** (x-axis = η·‖ḡ‖) | `[2e-4, 5e-4, 1e-3, 2e-3, 4e-3]` — one p\* cell per η, spanning a ~20× x-range. |
| `P_GRID` | **Base** coarse tether grid (at η=1e-3) | `[0.0, 0.005, 0.010, 0.020]`. Auto-scaled by η/1e-3 per cell (a fixed grid can't bracket p\* across 20×). |
| `BISECT_STEPS` | Bisections of the stable/collapse bracket | `3` (default for the eta-sweep; its predicted spread is large). |
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
- The sweep **skips any (arch, severity, η, p) whose JSON already exists** — re-run cell 8
  after a disconnect and it continues.
- Order is arch → severity → η, so an early η's p\* point is useful even if cut off.
- A single run that errors (e.g. OOM) is recorded as a `*.run_error.txt` sidecar and the
  sweep **continues**; re-running retries it.

---

## Expected wall-clock (Phase A: WRN sev5 × 5 η = 5 cells)

One continual run = the full 15-corruption stream ≈ 2,355 steps. Each η-cell does ≈ **5–7
runs**: 1 source + the (η-scaled) coarse grid overlapping the p_ref ladder + up to
`BISECT_STEPS` (=3) bisections; large-η cells may add a few upward-extension runs.

| GPU | WRN-28-10 run | Phase A (5 η cells, WRN sev5) |
|---|---|---|
| A100 | ~8–12 min | ~3.5–5 h |
| L4 | ~15–22 min | ~5–7 h |
| T4 | ~25–40 min | slow — prefer L4/A100 |

Phase B (adding `resnet18` over the same η grid) roughly adds another ~2–3 h on L4 (ResNet
runs are ~2–3× faster than WRN). Training a source model from scratch (if
`TRAIN_IF_MISSING`) adds ~20–40 min (ResNet) / ~1.5–3 h (WRN) at 30 epochs — do this once
and save the checkpoint to Drive.

---

## Outputs (in `RESULTS_DIR/analysis/`)

- **`pstar_law.json`** — table: `arch, severity, eta, source_acc, p_ref_used,
  grad_norm_gbar, eta_times_gbar, p_star, p_star_bracket_low/high, collapse_criterion,
  stable_mean_acc` + per-arch fits + verdict.
- **`pstar_law.png`** — the deliverable plot: x = η·‖ḡ‖ (swept via η), y = p\*, one series
  per arch (points = etas), least-squares fit overlaid with slope (=1/R), intercept, R².
- **`pstar_verdict.md`** — numeric verdict, one of:
  - **LAW SUPPORTED** — every arch R² ≥ ~0.9 and small intercept vs y-range.
  - **LAW PARTIAL** — linear within an arch but intercept non-trivial / a soft-criterion
    boundary confound / one arch clean and one not (names the cause).
  - **LAW NOT SUPPORTED** — points scatter or p\* non-monotone in η·‖ḡ‖.
- Any p\* point whose collapsing edge was a **soft** criterion (`soft:below_source` /
  `soft:drift_blowup`) is flagged in the verdict as a possible confound; for WRN sev5 the
  boundary should normally be `hard:nan_inf`.
- **`pstar_law_hardonly.{json,png,md}`** — only when `--hard-only`/`--compare-hardonly` is
  used (Step 1). Same schema, hard-criterion-only p\*.

The plot and verdict also display inline at the bottom of the notebook.

---

## How p\* and ‖ḡ‖ are computed (so the result is auditable)

- **The cell** is `(arch, severity, η)`. ‖ḡ‖, p_ref, the p-grid, and p\* are all computed
  **per cell**, and the x-axis uses the run's **own η** (`pc.heat_lr_of`), not a hardcoded
  constant.
- **‖ḡ‖** (`scripts/pstar_common.grad_norm_gbar`): mean of `grad_l2` over local steps
  **50–150** within each corruption block, averaged across all 15 blocks; blocks with any
  NaN/inf are excluded. Measured from a **stable reference run** `p_ref`, whose ladder is
  **scaled by η/1e-3** (so a stable reference exists even at η=4e-3 where p\* ~= 0.038 and
  p_ref must exceed it); the rung used is reported as `p_ref_used`.
- **Collapse criterion** (`classify_run`, unchanged): **hard** = any NaN/inf in
  `grad_l2`/`energy`/`drift_l2`, or stream `mean_accuracy ≤ 0.12`; **soft** =
  `last_accuracy` < source (no-adapt) accuracy, or stationary `drift_l2` > 5× its value at
  `p_ref`. The verdict flags any p\* whose collapsing edge was a **soft** criterion as a
  possible confound (WRN sev5 should be `hard:nan_inf`).
- **p-grid scaling** (critical): the coarse grid, p_ref ladder, and upward-extension grid
  are all **scaled by η/1e-3**, because p\* spans ~20× across the η sweep and a fixed grid
  could not bracket it.
- **p\*** (`choose_pstar`, unchanged): the midpoint of the final `[p_collapse, p_stable]`
  bracket after bisection. A cell with no collapse even at p=0 gives **p\*≈0** (a real
  point). Where the whole coarse grid collapses, the sweep folds in the stronger
  (η-scaled) p_ref-ladder rungs already on disk and, if even those collapse, **extends the
  grid upward** (capped at ~5 runs) until a stable p is found. If none is stable, p\* is
  left **unresolved** ("law region exhausted") and the plot shows it as a **censored
  up-arrow** lower bound (excluded from the fit), never hidden.

**No p-hacking:** all points are reported and the fit spans all of them. The analysis
script is the single source of truth and recomputes p\* from every run on disk.

---

## Validation: hard-only re-check (Step 1) & Phase B (second arch)

**Step 1 — hard-only re-check (zero GPU, re-analysis only).** If the verdict flags that
some p\* boundaries were set by a **soft** criterion (`soft:below_source` /
`soft:drift_blowup`) rather than `hard:nan_inf`, confirm the line isn't an artifact:

```bash
python scripts/analyze_pstar_law.py --results-dir <DIR> --archs wrn28_10 \
    --severities 5 --compare-hardonly
```

This re-reads the existing JSONs under both criteria and prints a per-η side-by-side
(`p*(soft+hard)` vs `p*(hard-only)`, with `moved?`) plus a refit slope/R/intercept/R² for
each. `--hard-only` treats only NaN/inf or chance-accuracy as collapse (soft instabilities
count as stable). Outputs go to `analysis/pstar_law_hardonly.{json,png,md}` — the default
artifacts are preserved. *Reading it:* if the hard-only line stays linear through ~origin
with a similar slope, the law is **robust** to the criterion; if low-η points move a lot,
the soft criterion was shaping them and the **hard-only fit is the clean signal**.

**Phase B — second architecture.** Add `resnet18` to `ARCHS` (sev5, same η grid). The
sweep processes ResNet with `--eta-order desc` (highest-signal points first) and pools both
archs on **one plot** (two series, two slopes 1/R) — the per-architecture-R evidence.
ResNet has a broad stable basin, so at sev5 it may not collapse at low η (**p\*≈0**, a
floor — consistent with a large R, not a refutation). If the *entire* ResNet η grid is
flat, the sweep **auto-escalates** with 2 stronger etas (`--eta-escalate-values 8e-3
1.6e-2`, capped at 2 runs) to push past the collapse boundary and reveal the slope; this
is logged as `[escalate]`.

**Phase C — ResNet hard-boundary refinement.** If ResNet's soft line is clean but it never
*hard*-collapses across the η grid (p\*_hard = 0 everywhere), push η further to try to
trigger a genuine hard boundary and test whether those hard p\* fall on the soft slope:

```bash
# extend the resnet eta-sweep upward (descending), then refine-report
python scripts/run_pstar_sweep.py --results-dir <DIR> --c10c-root data/cifar10c \
    --archs resnet18 --severities 5 --etas 8e-3 1.6e-2 3.2e-2 --eta-order desc \
    --p-grid 0.0 0.005 0.010 0.020 --bisect-steps 3 --ckpt-resnet18 <resnet.pt>
python scripts/analyze_pstar_law.py --results-dir <DIR> --archs wrn28_10 resnet18 \
    --severities 5 --hard-refine-arch resnet18
```

`--hard-refine-arch resnet18` prints, per η: `p*(soft)` vs `p*(hard)`, the hard
`collapse_criterion`, and a **confound flag**. The confound guard distinguishes a genuine
drift-collapse from optimizer/LR blow-up at high η, judged from the no-tether (p=0)
trajectory + the reference ‖ḡ‖:

- **optimizer-blowup** — first NaN within the first block (step < ~157), OR p_ref ‖ḡ‖
  > 2× its η=1e-3 value (the step size itself is unstable). **Excluded** from the slope fit.
- **genuine** — NaN appears after sustained adaptation (later block) with bounded early
  ‖ḡ‖, matching the WRN pattern. **Only these** enter the hard fit.

It then fits the genuine-drift hard p\* and prints one of three conclusions:
- **CONFIRMED** — hard slope within ~20% of the soft slope: the second R is robust under
  the hard criterion.
- **SLOPE-MISMATCH** — hard collapses but on a different slope: soft and hard measure
  different boundaries for this wide-basin arch (reports both).
- **NOT-TRIGGERABLE** — only blow-up / no hard collapse: a clean hard boundary needs a
  stronger shift than CIFAR-10-C severity provides (motivates DomainNet); the soft line
  stands as ResNet's best R estimate.

(Safety: at very high η the eta-scaled p-grid/ladder is clamped to drop values > 1, since
`restore_prob` must be in [0,1]. Unaffected for η ≤ 4e-3.)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Validation gate fails: `p010_gbar_in_band` | Wrong/under-trained WRN checkpoint, or not on GPU. Confirm `wrn28_10_final.pt` is the real source model. |
| Validation gate fails: nothing collapses | `--heat-diagnostic-snapshot` not producing `drift_l2`/NaNs, or severity 5 data missing. Re-run cell 5; check 50000 imgs/corruption. |
| `FileNotFoundError ... .npy` | CIFAR-10-C not fully downloaded. Re-run cell 5. |
| Sweep restarts from scratch after disconnect | `RESULTS_DIR` not on Drive (`USE_DRIVE=False`) — results were on ephemeral local disk. Set `USE_DRIVE=True`. |
| OOM on WRN | Lower `BATCH_SIZE` (e.g. 32) in the config cell; the run is recorded as `run_error` and retried on re-run. |
| Verdict says NOT SUPPORTED / PARTIAL | That may be the honest answer. Read `pstar_verdict.md` — it names the observed cause (e.g. p\* not monotone in η·‖ḡ‖, or a soft-criterion boundary confound). Adding more η values densifies the line. |
| Regression check: η=1e-3 WRN sev5 ≠ p\* ~0.0094 | The η plumbing broke. Confirm runs are tagged `pstar_lr0.001_...` and that `--heat-lr` reaches `run_tier2.py`. |
