# DomainNet-126 eta-sweep — TFF on ResNet-50 (real→clipart)

Decisive single-target slice that determines the paper's shape before paying for the full
3-target sweep. **Scope (do not exceed): arch=resnet50, source=real, target=clipart,
seed=42 ONLY.** The method is `heat` in code; **all results/plots are labeled TFF**.

Open `notebooks/domainnet_clipart_sweep_colab.ipynb` (branch `formulation`), or run the two
scripts directly. ~8–12 h on L4; descending-eta order makes a partial plot informative early.

## The three questions (reported separately)
- **Q1 (LAW):** does p\* scale linearly with η·‖ḡ‖ on ResNet-50/DomainNet, giving a
  per-architecture R? (Tests the law on a 3rd arch + a semantic shift.)
- **Q2 (USEFULNESS):** does TFF's best-over-p accuracy beat **BN-adapt** at any η? →
  Branch 1 "TFF wins" vs Branch 2 "graceful fallback only". (Beating merely *source* is not
  enough — the smoke test already showed TFF only climbs *toward* source as p grows.)
- **Q3 (MECHANISM):** at each collapse boundary, **drift-collapse** (drift_l2 large/diverging
  → the law's mechanism, p\* comparable to CIFAR) or **signal-collapse** (accuracy craters to
  chance while drift_l2 stays bounded → a *different* failure axis)?

## What runs
- `scripts/run_domainnet_eta_sweep.py` — orchestrator. Baselines (source, bn_adapt) once;
  then, per η (descending), heat over the eta-scaled coarse p-grid **plus {0.1, 0.2}**
  (graceful-fallback anchors), classified by **HARD collapse only**
  (`nan_inf` OR `mean_acc ≤ chance_acc≈0.02`), bisecting the boundary. Resumable.
- `scripts/run_p9_domainnet.py` — thin per-run entry: builds the resnet50 wrapper + loads the
  AdaContrast checkpoint, then **reuses the unchanged `run_tier2.run_p9`** on the clipart
  stream. Writes the standard eta-tagged JSON. (This is why `run_tier2.py` / the runner /
  `heat.py` stay untouched — we never hit run_tier2's argparse.)
- `scripts/analyze_domainnet_clipart.py` — the source of truth: p\* (hard-only), ‖ḡ‖ from the
  smallest stable p, drift at each boundary, mechanism classification, baseline comparison,
  the confound check, and the Q1/Q2/Q3 verdict.

## Contract / wiring (small, contained — protected code untouched)
- `pstar_common.build_run_command` now takes `dataset` (`cifar10` | `domainnet126`),
  `data_root`, `target_domain`, `method`; for domainnet it routes to `run_p9_domainnet.py`.
  The CIFAR path is byte-for-byte unchanged.
- New `pstar_common.hard_collapse(data, chance_acc)` — HARD-only labels for the DomainNet
  boundary. **`classify_run` (the CIFAR collapse math) is not modified**; nor is the ‖ḡ‖
  window, the runner (`src/adapt/runner.py`), or `src/methods/heat.py`.
- A "block" = the one clipart pass (~290 batches) → `group_rows_by_block` yields ONE block
  and the 50–150 stationary window applies within it (the smoke test produced sane ‖ḡ‖≈13–18
  in BN-train mode — verify it still does in the sweep log).
- **BN stays in train mode** (the smoke test proved `frozen → NaN`).

## Boundary, mechanism, confound
- `soft:below_source` fires at ALL p (TFF never beats source here), so it **cannot** define a
  boundary. p\* = min p that avoids **hard** collapse.
- At each boundary the run's `drift_l2` is recorded and the mechanism classified:
  **drift-collapse** if `nan_inf` or drift ≫ the stable-reference drift (>5×);
  **signal-collapse** if accuracy is at chance while drift stays bounded.
- **Confound check** at the top etas (1.6e-2, 8e-3): a point is flagged **optimizer-blowup**
  (and excluded from the law fit, drawn as `×`) if the no-tether run's first NaN is in the
  first block (step < ~157) or its ‖ḡ‖ is already >2× the η=1e-3 value. If the top etas are
  *all* optimizer-blowup (no genuine collapse), the analysis says so — we'd then rethink the
  collapse definition for BN-train ResNet-50 before scaling.

## Outputs (in `RESULTS_DIR/analysis/`)
- `domainnet_clipart_law.png` — p\* vs η·‖ḡ‖, LS fit (slope=1/R, intercept, R²), each point
  colored by mechanism (red=drift, blue=signal), `×`=optimizer-blowup.
- `domainnet_clipart_usefulness.{md,json}` — per-η table: `eta, eta·‖ḡ‖, p*,
  boundary_mechanism, TFF_best_acc, bn_adapt_acc, source_acc, argmax_p, drift_at_boundary,
  confound`.
- A printed 3-line verdict: Q1 (slope/R/R²), Q2 (beats BN? which η), Q3 (drift vs signal
  counts).

## Robustness / scope
Drive-backed `RESULTS_DIR` (keep separate from the CIFAR results dir), resumable (skip
existing eta-tagged JSONs). **Do NOT** start the other domains (painting/sketch), other
archs, or multi-seed — this slice decides the paper's shape first. If the top-eta runs are
all optimizer-blowup, halt and report.
