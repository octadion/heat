# Stage 1 — ||g_bar||-factor test (E1) + zero-GPU analyses (E2–E5): Colab runbook

Everything reuses the eta-sweep infrastructure. **TFF** is the paper name; the code
identifier is `heat` (never label outputs "HEAT").

## What Stage 1 measures

**E1 (GPU).** Single-corruption p* cells on WRN-28-10, CIFAR-10-C severity 5:

- corruptions: `gaussian_noise`, `elastic_transform` (**calibration** — the only cells
  ever used to fit a slope) + `impulse_noise`, `contrast` (**held-out** — zero-shot
  test, never fitted);
- etas: `{5e-4, 1e-3, 2e-3}` per corruption; seed 42, batch 64;
- each (corruption, eta) cell: source run → eta-scaled p_ref ladder → eta-scaled
  coarse p-grid → upward extension → bisection — the continual sweep's exact policy.

The stream is **one corruption's full severity-5 split** (10,000 images ≈ 157 steps),
via `run_tier2.py --protocol p9 --corruptions <c>` (natively supported; no runner
change was made). Filenames are corruption-tagged:
`p9_wrn28_10_pstar_<corruption>_lr<eta>_p<p>_seed42_sev5.json` — old continual files
are never matched by the new discovery and vice versa (covered by the self-test).

**E2–E5 (zero GPU).** Convergence of the grad_l2 running mean; coupling index of
||g_bar|| with p; zero-shot p on the held-out corruptions from the calibration-only
slope; R-perturbation robustness. All computed by `scripts/analyze_stage1.py` from
JSONs on disk.

## How to run (Colab)

Use `notebooks/pstar_colab.ipynb` **cells 1–7 unchanged** (GPU check → config →
Drive mount → clone+install → CIFAR-10-C download → checkpoints → pipeline-validation
gate). The gate reuses the existing continual WRN sev5 JSONs on Drive, so it is
free — **do not skip it**. Then, instead of the sweep cell, run:

```bash
# E1 campaign (resumable; skips any JSON already on Drive)
python scripts/run_stage1_e1.py \
    --results-dir /content/drive/MyDrive/pstar_results \
    --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
    --c10c-root data/cifar10c

# Stage-1 analysis: E1 fit + E2-E5 + verdict (+ regression-anchor check)
python scripts/analyze_stage1.py \
    --results-dir /content/drive/MyDrive/pstar_results \
    --domainnet-results-dir /content/drive/MyDrive/<domainnet_results_dir>
```

If the E1 runs live in a different dir from the continual eta-sweep JSONs, pass
`--etasweep-results-dir` too (E2, the overlay line, and the anchor check read the
continual runs). `--domainnet-results-dir` is optional and only feeds the E3
coupling table's DomainNet rows.

Before anything else on a fresh clone you can sanity-check the plumbing with **zero
GPU**:

```bash
python scripts/analyze_stage1.py --self-test
```

## Cost estimate

12 cells × ~6–8 runs, each run ≈ 157 steps ≈ **1/15 of a continual run** (~40–60 s
on A100 for WRN). Whole campaign ≈ **1–2 h on A100** including source runs.

## Resumability / logging

- Every run JSON is written to `--results-dir` immediately by `run_tier2.py`;
  re-running skips existing files; failures leave `*.run_error.txt` sidecars and the
  sweep continues.
- `stage1_e1_manifest.jsonl` (in the results dir) logs one line per run: corruption,
  eta, p, seed, status, elapsed, git hash, GPU name, torch version.

## Outputs (in `<results-dir>/analysis/`)

`stage1_gbar_law.{json,png}` (points colored by corruption; per-corruption fits +
pooled fit + continual eta-sweep overlay; soft+hard and hard-only fitted separately,
never averaged), `stage1_convergence.png`, `stage1_coupling.md`,
`stage1_zeroshot.md`, `stage1_verdict.md`.

## Verdict criteria (mechanical, from the master plan)

- **E1 GREEN** = per-corruption slopes within ±20% of each other AND of the
  eta-sweep slope (same criterion family), pooled R² ≥ 0.95, x-spread of corruption
  positions ≥ 1.5×. AMBER = linear per corruption, slope spread 20–100%.
  RED = spread > 2× or non-monotone.
- **E2 GREEN** = running-mean convergence ≤ 50 steps for ≥ 90% of runs.
- **E4 GREEN** = zero-shot ≥ 95% of oracle, zero collapses, on ≥ 80% of held-out
  cells; AMBER 85–95%.
- **E5 GREEN** = no collapse at +50% R error, graceful at −25%.

**Regression anchor:** the analysis recomputes the continual WRN sev5 eta=1e-3 cell;
if p* moved from ~0.0094 it prints **MOVED-STOP** — halt and report.

**Gate:** Stage 1b (E6/E7) may run interleaved/after E1 on the same GPU. Stages 2–4
open only if E1 is GREEN or AMBER. STOP after Stage 1 and report.

## Stage-1 FOLLOW-UP — criterion resolution + pre-registered forward confirmation

After the E1 verdict (RED on pooled soft+hard, structured soft:below_source
deviations), run in this order **on the box with the Drive results**:

```bash
# PART A (zero GPU): A1 soft-boundary attribution + A2 bracket-weighted refit.
# Writes analysis/stage1_softboundary.md and FREEZES the pooled hard-only slope
# to analysis/stage1_hardslope_frozen.json (consumed by Part B).
python scripts/analyze_stage1_followup.py \
    --results-dir /content/drive/MyDrive/pstar_results

# PART B (small GPU, ~30 runs): pre-registered forward confirmation on
# {fog, shot_noise} x {5e-4, 2e-3}. Order enforced by the script:
# source+p_ref for ALL 4 cells -> analysis/stage1b_predictions.json is written
# (timestamped, never overwritten) -> only then the at-p_hat run + hard-
# criterion grid + bisection -> analysis/stage1_forward_verdict.md.
python scripts/run_stage1_forward.py \
    --results-dir /content/drive/MyDrive/pstar_results \
    --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
    --c10c-root data/cifar10c
```

- Part B refuses to start without the frozen-slope file (Part A is a hard
  prerequisite) and records which S_frozen value/file it used inside the
  predictions JSON.
- Any grid run found on disk **before** the predictions file is first written
  is recorded as a pre-registration violation in both the predictions JSON and
  the verdict — never silently ignored.
- Forward verdict (mechanical): CONFIRMED ≥3/4 cells with measured p*_hard
  within ±25% of prediction (or prediction inside the bisection bracket) AND
  no hard collapse at p_hat; PARTIAL = 2/4; FAILED ≤1/4.
- Zero-GPU plumbing checks: `python scripts/analyze_stage1_followup.py
  --self-test` (Part A) — Part B was validated end-to-end against a synthetic
  law campaign via a fake-runner harness.

**STOP after Part B; Stage 1b (E6/E7) awaits instruction and will use
hard-only as the primary law criterion with soft reported as a separate
boundary.**

## STAGE 1b — E6 cross-mechanism + E7 drive-swap (hard-primary, pre-registered)

Criterion update: the PRIMARY law criterion is **hard collapse** (nan_inf /
chance), matching the forward confirmation; soft:below_source is recorded per
run and reported as a **separate** boundary, never mixed into law fits. Slope
comparisons use the A2 bracket-weighted hard-only Bernoulli reference
`S_frozen = 1.185` (default of `--s-frozen`; recorded in every predictions
file).

One-click Colab runbook for this stage: **`notebooks/stage1b_colab.ipynb`**
(GPU check → config → Drive → clone → data → WRN checkpoint → validation gate →
optional S_frozen re-freeze → E6 → E7, with verdicts + plots shown inline). The
CLI below is what that notebook invokes.

New additive method variants (plain HEAT path is bit-identical when the flags
are at defaults): `HeatAnchor` (`--heat-anchor-lambda`, θ ← θ − ηλ(θ − θ_src)
after each SGD step, mutually exclusive with restore_prob>0) and
`HeatEntropyDrive` (`--drive entropy`, loss = mean prediction entropy of the
batch from the model's true logits). Both are dispatched by the factory under
the method name `heat`, so the runner and summary keys are untouched.

```bash
# E6 (~25-35 runs): anchor cells {gaussian_noise x 5e-4,1e-3,2e-3} +
# {elastic_transform x 1e-3}. Order enforced: source+reference-lambda for all
# cells -> analysis/e6_predictions.json ((eta*lambda)_hat = S_frozen*eta*||g||,
# timestamped, never overwritten) -> at-lambda_hat run + lambda grid +
# bisection (hard criterion) -> analysis/e6_crossmech.{json,png,md}.
python scripts/run_stage1b_e6.py --results-dir /content/drive/MyDrive/pstar_results \
    --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt --c10c-root data/cifar10c

# E7 (~18-26 runs): entropy-drive cells {gaussian_noise, elastic_transform} x
# {5e-4,1e-3,2e-3}. Order enforced: entropy reference runs -> analysis/
# e7_predictions.json (strong form p_hat = S_frozen*eta*||g||_entropy) ->
# at-p_hat + p-grid + bisection (hard) -> analysis/e7_driveswap.{json,png,md}
# incl. the calmness comparison at matched stable cells.
python scripts/run_stage1b_e7.py --results-dir /content/drive/MyDrive/pstar_results \
    --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt --c10c-root data/cifar10c
```

**PERMANENT METHODOLOGICAL-INTEGRITY RULES** (post-audit; encoded in
`stage1b_common.py`, apply to E6/E7 and every Stage-2+ analyzer/orchestrator):

- **RULE 1 (VOID guard):** if no reference points are discoverable (e.g. zero
  E1 Bernoulli hard points), or any fitted cell shows the pathological
  `p*=0` non-monotone (stable-below-collapse) boundary, the verdict is
  **VOID** — never a curve verdict.
- **RULE 2 (loud abort):** E6/E7 refuse to start when the expected E1
  reference files are absent from `--results-dir` (silent re-running of
  references is forbidden) unless `--fresh-reference` is explicitly passed
  (recorded in the manifest + predictions file).
- **RULE 3 (predictions validity):** an existing `e6/e7_predictions.json` is
  INVALID by default. Pass `--trust-existing-predictions` only if the audit
  confirmed its reference ‖ḡ‖ runs were genuine (correct checkpoint +
  correct dispatch), or `--requarantine-predictions` to move it to
  `analysis/quarantine/` (with a note, never overwritten) and re-freeze new
  predictions BEFORE any new grid run.

Forensic audit (Part A, zero GPU, read-only — run where the campaign files
live): `python scripts/audit_stage1b.py --results-dir <campaign dir>
[--e1-results-dir <E1 dir>]` → one-line root causes + `analysis/stage1b_audit.md`.

Verdicts (mechanical): E6 **SAME-CURVE** = anchor slope within ±25% of
S_frozen AND pooled (anchor + Bernoulli hard points) R² ≥ 0.95 AND ≥3/4 cells
within ±25% of the frozen prediction (or in-bracket); else **DIFFERENT-CURVE**.
E7 labels: **DRIVE-AGNOSTIC-STRONG** (hits ≥ ceil(0.75·n) AND R² ≥ 0.9) /
**DRIVE-AGNOSTIC-WEAK** (linear+monotone, own slope) / **DRIVE-SENSITIVE**
(non-monotone or >half unresolved) / **UNCLASSIFIABLE**.

**STOP after E6+E7; Stage 2 awaits instruction.**

## Known caveat (reported in the verdict's deviations section)

A single-corruption stream is only ~157 steps; the continual WRN sev5 eta=1e-3
hard collapse historically appears at step ~603. Within 157 steps some boundaries
may therefore be set by **soft** criteria — the hard-only panel of
`stage1_gbar_law.png` makes this visible rather than hidden.
