# DomainNet-126 — HEAT integration smoke test

**Not a full experiment.** Proves the pipeline works end-to-end on ONE slice —
`arch=resnet50, source=real, target=clipart, η=1e-3` — before investing in the full
domain×η sweep. ~30–45 min on an L4. Open `notebooks/domainnet_smoke_colab.ipynb` and run
top-to-bottom, or run `scripts/run_domainnet_smoke.py` directly.

## What's new (new files only; protected code untouched)
- `src/models/resnet50_domainnet.py` — AdaContrast ResNet-50 wrapper, HEAT-compatible.
- `src/data/domainnet126.py` — DomainNet-126 loader (a "corruption" = a target domain).
- `scripts/run_domainnet_smoke.py` — the 4-check smoke slice.
- Dispatch registration only: `num_classes_for("domainnet126")=126` + a
  `get_corruption_loader` branch (`src/data/__init__.py`), and a `resnet50` factory in
  `build_arch` (`src/models/__init__.py`).
- **Untouched:** the collapse-criterion math, the ‖ḡ‖ window (`analysis_common` 50–150),
  the runner (`src/adapt/runner.py`), and `src/methods/heat.py`. The smoke script feeds
  `run_tier2.run_p9` a hand-built args namespace, so `run_tier2.py` is also unchanged (we
  never hit its argparse, which would reject the new arch/dataset).

## The architecture / HEAT integration (key design points)
The AdaContrast source net is **not** vanilla torchvision:
`x → ResNet-50 → GAP(2048) → bottleneck Linear(2048→256)+BN1d(256) → weight-normed
classifier(256→126)`. The wrapper exposes the HEAT contract:
- `model.stage_channels = [256, 512, 1024, 2048]` (layer1..4),
- `model(x, return_stages=True) → (logits, [z1,z2,z3,z4])` (maps before GAP),
- `model.linear` — a callable head whose **`.in_features == 2048`** (the bottleneck input
  dim, NOT 126 and NOT 256) that internally applies bottleneck+classifier. HEAT projects
  each stage's GAP feature to 2048, then applies this head; `identity_at_match` makes the
  final (2048-ch) stage reproduce the real logits. `heat.py` is **not** modified.
- The weight-normed classifier is **folded** into an effective plain `nn.Linear` at load
  time (identical forward; a live `weight_norm` hook makes `.weight` non-leaf and breaks
  `copy.deepcopy`, which the method factory needs). Verified the fold matches the
  weight-norm effective weight to ~1e-8.

The loader strict-loads the AdaContrast checkpoint by remapping
`encoder.0.*→backbone.*`, `encoder.1.*→bottleneck_bn.*`, `fc.*→classifier.*` (and folding
`fc.weight_g/weight_v`). On mismatch it loads non-strict and reports the exact missing/
unexpected keys.

## Resolved sources (verified, not guessed)
| What | URL | Size / note |
|---|---|---|
| Source checkpoints (AdaContrast, seeds 2020/21/22 × 4 domains) | `https://drive.google.com/drive/folders/16vTNNzzAt4M1mmeLsOxSFDRzBogaNkJw` | ~1.1 GB for all; we use **`best_real_2020.pth.tar`** (note `.pth.tar` extension; ResNet-50 ~90 MB). `gdown --folder --remaining-ok`. |
| Image lists (126-class) | `https://raw.githubusercontent.com/DianCh/AdaContrast/master/datasets/domainnet-126/<domain>_list.txt` | few MB; lines `"<domain>/cat/img.jpg <label>"` |
| clipart images (cleaned DomainNet) | `http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip` | ~1 GB (~48k imgs) — **required** |
| real images (cleaned DomainNet) | `http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip` | ~5.6 GB — **NOT needed** for this slice (note: `real.zip` is directly under `multi-source/`, not `groundtruth/`) |

**This slice needs only clipart images + the real checkpoint (~2 GB total).** Source
accuracy is measured on clipart; the real *images* are never read (the real *model* is the
checkpoint). `DOWNLOAD_REAL` stays `False` unless you want the other domains later.

Layout expected by the loader (`DOMAINNET_ROOT`):
```
domainnet-126/
  clipart_list.txt           # "clipart/cat/img.jpg <label>"  (paths relative to root)
  clipart/...                # from clipart.zip
```

## The four checks (PASS iff all sane)
1. **Checkpoint loads** — strict, or the exact mismatched-key list (so the remap can be
   fixed if a mirror differs).
2. **Source clipart acc ∈ ~[0.35, 0.55]** — proves wrapper + loader + label mapping are
   right (a wrong stage tap or class mapping shows up as chance-level acc).
3. **‖ḡ‖ finite & sane** (reported) — proves HEAT's stage taps work on ResNet-50.
4. **Collapse boundary present** — ≥1 stable AND ≥1 collapse across the coarse p-grid
   `{0, 0.005, 0.02, 0.05}`, so p* is measurable here.

The script prints an itemized PASS/FAIL and exits non-zero on failure. If check 4 shows
all-stable, clipart doesn't destabilize this source at η=1e-3 (p*≈0); if all-collapse with
chance accuracy, the source/transform/label mapping is likely wrong — fix before scaling.

## Scope
`resnet50, source=real, target=clipart, η=1e-3` ONLY. Do **not** run the other domains,
other etas, or the full sweep yet — that comes after the smoke passes.
