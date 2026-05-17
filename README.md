# HEAT: Hierarchical Energy Adaptive Test-time

Working codebase for the framework paper. **HEAT** is one instantiation
of an "intrinsic adaptation" framework where:
- **What** is adapted is decided by a single principle P
- **When** to adapt emerges from ‖∇P‖ (small when consistent with input)
- **Where** to adapt emerges from the structure of ∇P across layers
- **How** to adapt is the negative gradient direction

## Method at a glance

```
E_l(x) = -logsumexp(h(P_l(pool(z_l(x)))))     for stage l = 1..L
P(x)   = sum_l E_l(x)
update : theta <- theta - eta * grad_theta P(x)
```

- `h` is the trained classifier head (frozen structure, weights adapt)
- `P_l` is a fixed orthogonal projection (channel_l -> head_in_dim).
  When channel_l == head_in_dim (final stage), `P_l = I` so the final
  stage's energy equals the actual model output energy.
- All backbone parameters are eligible for update — where-ness emerges
  from the gradient profile, not from manual layer selection
- Single SGD step per test batch (vanilla SGD, no momentum by default)

### HEAT options

- `eval_mode=False` (default): `model.train()` so BN running stats track
  the test batch — matches Tent / TEA convention.
- `eval_mode=True`: `model.eval()` — BN stats frozen. Only adaptation
  channel is the gradient step on theta. This is the **strict-monad
  variant** for defending the "single-principle" claim. Best practice:
  report both at paper time; the difference quantifies BN-stats
  contribution.
- `update_direction='-grad'` (default), `'+grad'`, `'random'`: direction
  ablation for protocol P10.

### Baseline fidelity

- **Tent**: Adam optimizer (lr=1e-3) — matches `DequanWang/tent` reference
  code. Pass `--tent-optimizer sgd` and lr=0.005 for the EATA-style
  convention.
- **TEA**: Adam (lr=1e-3), SGLD steps=20, sgld_lr=0.1, noise_std=0.01,
  persistent buffer with 5% reinit probability. Matches Tab. 7 of the
  TEA paper. Cross-check against the official repo before reporting
  paper-grade numbers.

## Installation

### Colab
```bash
!git clone <repo> heat_tta
%cd heat_tta
!bash setup_colab.sh
```

### Local
```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## End-to-end workflow

```bash
# 1. Sanity check (no GPU/data needed; ~1 min)
python scripts/smoke_test.py

# 2. Train ResNet-18 from scratch on CIFAR-10
#    Saves checkpoints at epochs 0, 5, 10, 25, 50, 100, final (for P3)
python scripts/train_source.py --epochs 200

# 3. Download CIFAR-10-C (~2.5 GB, Hendrycks & Dietterich)
python scripts/download_cifar10c.py

# 4. Hyperparameter search on HOLDOUT corruptions
python scripts/hp_search.py --checkpoint experiments/checkpoints/final.pt --method heat
python scripts/hp_search.py --checkpoint experiments/checkpoints/final.pt --method tent
python scripts/hp_search.py --checkpoint experiments/checkpoints/final.pt --method tea

# 5. Tier-1 protocols (must-have for paper)
python scripts/run_p1_benchmark.py    --checkpoint experiments/checkpoints/final.pt
python scripts/run_p2_emergent.py     --checkpoint experiments/checkpoints/final.pt
python scripts/run_p3_doseresponse.py --checkpoint-dir experiments/checkpoints
python scripts/run_p4_whereness.py    --checkpoint experiments/checkpoints/final.pt
python scripts/run_p5_ablation.py     --checkpoint experiments/checkpoints/final.pt

# 6. Tier-2 protocols (recommended)
python scripts/run_tier2.py --checkpoint experiments/checkpoints/final.pt --protocol p6
python scripts/run_tier2.py --checkpoint experiments/checkpoints/final.pt --protocol p7
python scripts/run_tier2.py --checkpoint experiments/checkpoints/final.pt --protocol p8
python scripts/run_tier2.py --checkpoint experiments/checkpoints/final.pt --protocol p9
python scripts/run_tier2.py --checkpoint experiments/checkpoints/final.pt --protocol p10

# 7. Plots
for p in p1 p2 p3 p4 p5; do
  python scripts/make_plots.py $p experiments/results/${p}_*.json
done
```

## Layout

```
heat_tta/
├── src/
│   ├── models/resnet_cifar.py       # ResNet-18 with stage outputs exposed
│   ├── data/{cifar10,cifar10c}.py   # Loaders
│   ├── methods/
│   │   ├── base.py                  # AdaptMethod ABC
│   │   ├── source.py, bn_adapt.py   # No-op baselines
│   │   ├── tent.py                  # Tent (Wang et al., 2021)
│   │   ├── tea.py                   # TEA (Yuan et al., 2024) — main competitor
│   │   └── heat.py                  # HEAT — our method
│   ├── adapt/runner.py              # Online TTA evaluation loop
│   └── utils/{seed,device}.py
├── scripts/
│   ├── train_source.py              # Train + checkpoint at multiple epochs
│   ├── download_cifar10c.py
│   ├── smoke_test.py                # No-GPU sanity check
│   ├── hp_search.py                 # LR search on holdout corruptions
│   ├── run_p1_benchmark.py          # Standard online TTA
│   ├── run_p2_emergent.py           # Pretrained vs random init
│   ├── run_p3_doseresponse.py       # Across pretraining checkpoints
│   ├── run_p4_whereness.py          # Gradient profile across layers
│   ├── run_p5_ablation.py           # Hierarchy variants
│   ├── run_tier2.py                 # P6/P7/P8/P9
│   └── make_plots.py
├── colab_quickstart.py              # Reference cells for Colab
└── experiments/
    ├── checkpoints/                 # Trained models go here
    └── results/                     # JSON logs and PNG/PDF figures
```

## Protocol → claim mapping

| Protocol | Tier | Claim tested |
|---|---|---|
| P1 | 1 | C1 (core perf), C6 (vs TEA), C7 (vs Tent) |
| P2 | 1 | C2 (emergent capacity) — pretrained vs random init |
| P3 | 1 | C2 — dose-response across pretraining stages |
| P4 | 1 | C3 (where-emergence) |
| P5 | 1 | C8 (hierarchy) — 4 hierarchy depths × 2 update sets = 8 cells |
| P6 | 2 | C4 (when-emergence vs severity) |
| P7 | 2 | C4, C9 (no degradation on clean data) |
| P8 | 2 | C4, C9 (recovery during/after shift) |
| P9 | 2 | C9 (continual stability) |
| P10 | 3 | C5 (gradient direction is the right signal) |
| HP search | — | Reproducibility / fair comparison |

## Renaming

```bash
grep -rIl "HEAT\|heat" src scripts | xargs sed -i 's/HEAT/<NEWNAME>/g; s/heat/<newname>/g'
```

## Caveats

- Hyperparameters in run scripts are initial defaults. **Always run
  `hp_search.py` first** and update the `--*-lr` flags accordingly.
- TEA's SGLD is sensitive to `--tea-sgld-steps` and `--tea-lr`. Faithful
  reproduction recommends consulting the official TEA repo for matching
  hyperparameters on each backbone.
- All experiments use seed 42 by default. **Report mean ± std over 3+
  seeds for the paper** (re-run with `--seed 0 1 2` and aggregate).
