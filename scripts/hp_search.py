"""
Multi-criteria HP search v1.2 — with --arch and --dataset support.

Picks LR by Pareto-feasible criterion:
  maximize    holdout_p1_mean
  subject to  holdout_p9_last_acc >= chance + margin

Same logic as v1.1 but supports ResNet-18, WRN-28-10, and ViT-S/16.

NEW (vit/cifar100 patch):
  --arch vit_s — runs the HP search on a ViT-S/16 source. Optimal LRs for
                 ViT will differ from CNN values; user supplies a grid or
                 takes the default.
  --dataset {cifar10, cifar100} — selects corruption dataset + num_classes.
  Chance accuracy in the P9 feasibility constraint now follows num_classes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_corruption_loader, get_clean_loaders, num_classes_for
from src.models import build_arch
from src.methods import (
    HEAT, Tent, TEA, TEANoNoise, TEADirectEnergy, EPOTTA, ReTTA,
)
from src.adapt import evaluate_online
from src.utils import set_seed, get_device


HOLDOUT_CORRUPTIONS = ["gaussian_blur", "saturate", "spatter", "speckle_noise"]

DEFAULT_LR_GRID = {
    "heat":             [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "heat_singlestage": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "tent":             [1e-4, 5e-4, 1e-3, 5e-3],
    "tea":              [1e-4, 5e-4, 1e-3],
    "tea_nonoise":      [1e-4, 5e-4, 1e-3],
    "tea_directenergy": [1e-4, 5e-4, 1e-3, 5e-3],
    "epotta":           [1e-4, 5e-4, 1e-3, 5e-3],
    "retta":            [1e-4, 5e-4, 1e-3, 5e-3],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10", "vit_s"])
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--variant", type=str, default="heat",
                   choices=["heat", "heat_singlestage",
                            "tent", "tea", "tea_nonoise", "tea_directenergy",
                            "epotta", "retta"])
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--c100c-root", type=str, default="data/cifar100c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--cifar100-root", type=str, default="data/cifar100")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=HOLDOUT_CORRUPTIONS)
    p.add_argument("--lr-grid", type=float, nargs="+", default=None)

    # HEAT specifics
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--aggregation", type=str, default="sum",
                   choices=["sum", "self_gated", "self_gated_temperature"])
    p.add_argument("--update-all-params", action="store_true", default=True)
    p.add_argument("--bn-only", dest="update_all_params", action="store_false")
    p.add_argument("--heat-adapt-params", type=str, default="full",
                   choices=["full", "bn_affine_only"],
                   help="Explicit HEAT/TFF parameter subset. Overrides "
                        "--bn-only when set to bn_affine_only.")
    p.add_argument("--heat-bn-running-stats", type=str, default="train",
                   choices=["train", "frozen"])
    p.add_argument("--eval-modes", type=str, nargs="+", default=None)
    p.add_argument("--restore-prob", type=float, default=0.0,
                   help="HEAT-dyad restore_prob; default 0 = monad.")
    p.add_argument("--tea-sgld-steps", type=int, default=20)
    p.add_argument("--tea-sgld-lr", type=float, default=0.1)
    p.add_argument("--tea-sgld-noise", type=float, default=0.01)
    # Explicit stage selection for HEAT / heat_singlestage. Default (None)
    # means all stages (current HEAT behavior; bit-identical to pre-patch).
    p.add_argument("--heat-stages", type=int, nargs="+", default=None,
                   help="Explicit stage indices for HEAT / heat_singlestage. "
                        "Default = all stages.")

    # Constraint
    p.add_argument("--p9-margin-pt", type=float, default=15.0)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def _corruption_root(args) -> str:
    return args.c10c_root if args.dataset == "cifar10" else args.c100c_root


def _clean_root(args) -> str:
    return args.cifar10_root if args.dataset == "cifar10" else args.cifar100_root


def build_method(variant, base_model, device, lr, args, eval_mode_bool):
    m = copy.deepcopy(base_model).to(device)
    explicit_stages = list(args.heat_stages) if args.heat_stages is not None else None
    update_all_params = (
        args.update_all_params and args.heat_adapt_params == "full"
    )
    if variant == "heat":
        return HEAT(
            m, lr=lr, momentum=0.0,
            stages=explicit_stages,
            temperatures=args.temperatures,
            aggregation=args.aggregation,
            update_all_params=update_all_params,
            eval_mode=eval_mode_bool,
            bn_running_stats=args.heat_bn_running_stats,
            restore_prob=args.restore_prob,
        ).to(device)
    elif variant == "heat_singlestage":
        # HEAT restricted to a single stage. Default: the final stage.
        # An explicit --heat-stages overrides for sweeps over stage depth.
        if explicit_stages is not None:
            stages = explicit_stages
        else:
            stages = [len(m.stage_channels) - 1]
        return HEAT(
            m, lr=lr, momentum=0.0,
            stages=stages,
            temperatures=args.temperatures,
            aggregation=args.aggregation,
            update_all_params=update_all_params,
            eval_mode=eval_mode_bool,
            bn_running_stats=args.heat_bn_running_stats,
            restore_prob=args.restore_prob,
        ).to(device)
    elif variant == "tent":
        return Tent(m, lr=lr, optimizer_name="adam", momentum=0.9)
    elif variant == "tea":
        return TEA(m, lr=lr, optimizer_name="adam",
                   sgld_steps=args.tea_sgld_steps,
                   sgld_lr=args.tea_sgld_lr,
                   sgld_noise=args.tea_sgld_noise)
    elif variant == "tea_nonoise":
        return TEANoNoise(m, lr=lr, optimizer_name="adam",
                          sgld_steps=args.tea_sgld_steps,
                          sgld_lr=args.tea_sgld_lr)
    elif variant == "tea_directenergy":
        return TEADirectEnergy(m, lr=lr, optimizer_name="adam",
                               sgld_steps=args.tea_sgld_steps,
                               sgld_lr=args.tea_sgld_lr)
    elif variant == "epotta":
        train_loader, _ = get_clean_loaders(
            args.dataset, _clean_root(args),
            batch_size=args.batch_size, num_workers=args.num_workers,
            arch=args.arch,
        )
        return EPOTTA(m, source_loader=train_loader, lr=lr,
                      buffer_size=500, device=device)
    elif variant == "retta":
        return ReTTA(m, lr=lr, lambda_energy=1.0)
    raise ValueError(variant)


def evaluate_single_domain(factory, corruptions, args, device):
    accs = []
    for c in corruptions:
        loader = get_corruption_loader(
            args.dataset, _corruption_root(args), c,
            severity=args.severity, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False, arch=args.arch,
        )
        set_seed(args.seed)
        method = factory()
        acc = evaluate_online(method, loader, device,
                              progress=False).accuracy
        accs.append(acc)
    return accs


def evaluate_continual(factory, corruptions, args, device):
    set_seed(args.seed)
    method = factory()
    accs = []
    for c in corruptions:
        loader = get_corruption_loader(
            args.dataset, _corruption_root(args), c,
            severity=args.severity, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False, arch=args.arch,
        )
        acc = evaluate_online(method, loader, device,
                              progress=False).accuracy
        accs.append(acc)
    return accs


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = num_classes_for(args.dataset)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    extra = {"pretrained": False} if args.arch == "vit_s" else {}
    base_model = build_arch(args.arch, num_classes=num_classes, **extra).to(device)
    base_model.load_state_dict(ckpt["model"])

    lr_grid = args.lr_grid or DEFAULT_LR_GRID[args.variant]

    # Both `heat` and `heat_singlestage` benefit from the eval_mode sweep
    # (model.train() vs model.eval() — relevant for BN running-stat tracking).
    # Other variants don't expose eval_mode, so collapse to a single run.
    if args.variant in ("heat", "heat_singlestage"):
        if args.eval_modes is None:
            eval_modes = [True, False]
        else:
            eval_modes = [s.lower() == "true" for s in args.eval_modes]
    else:
        eval_modes = [None]

    chance_acc = 1.0 / num_classes
    p9_threshold = chance_acc + args.p9_margin_pt / 100.0

    results = []
    print(f"\n=== HP search variant={args.variant}, arch={args.arch}, "
          f"dataset={args.dataset} ===")
    print(f"  LR grid: {lr_grid}")
    print(f"  chance_acc={chance_acc:.3f}, p9_threshold={p9_threshold:.3f}")
    if args.variant in ("heat", "heat_singlestage"):
        print(f"  eval_modes: {eval_modes}, restore_prob={args.restore_prob}")

    for em in eval_modes:
        for lr in lr_grid:
            em_label = "n/a" if em is None else str(em)
            print(f"\n--- lr={lr:g}, eval_mode={em_label} ---")

            def factory(_lr=lr, _em=em):
                return build_method(args.variant, base_model, device, _lr,
                                    args, _em)

            try:
                p1_accs = evaluate_single_domain(factory, args.corruptions,
                                                 args, device)
                p1_mean = sum(p1_accs) / len(p1_accs)
                print(f"  P1 mean: {p1_mean:.4f}")

                p9_accs = evaluate_continual(factory, args.corruptions,
                                             args, device)
                p9_last = p9_accs[-1]
                p9_mean = sum(p9_accs) / len(p9_accs)
                print(f"  P9 stream: {[f'{a:.3f}' for a in p9_accs]}  "
                      f"last={p9_last:.4f}")

                feasible = p9_last >= p9_threshold
                results.append({
                    "lr": lr, "eval_mode": em,
                    "p1_mean": p1_mean,
                    "p9_stream": p9_accs,
                    "p9_last": p9_last,
                    "p9_mean": p9_mean,
                    "p9_feasible": feasible,
                })
                print(f"  feasible: {feasible}")
            except Exception as e:
                print(f"  FAIL: {e}")
                results.append({"lr": lr, "eval_mode": em, "error": str(e)})

    feasible = [r for r in results if r.get("p9_feasible", False)]
    if feasible:
        best = max(feasible, key=lambda r: r["p1_mean"])
        print(f"\n*** Pareto-feasible best: lr={best['lr']:g}, "
              f"eval_mode={best['eval_mode']}, "
              f"P1={best['p1_mean']:.4f}, P9_last={best['p9_last']:.4f}")
    else:
        print("\n!!! NO LR passes the P9 stability constraint.")
        feasible_p1 = [r for r in results if "p1_mean" in r]
        if feasible_p1:
            best = max(feasible_p1, key=lambda r: r["p1_mean"])
            print(f"    best-P1: lr={best['lr']:g}, "
                  f"P1={best['p1_mean']:.4f}, P9_last={best['p9_last']:.4f}")
        else:
            best = None

    dataset_tag = f"_{args.dataset}" if args.dataset != "cifar10" else ""
    out_path = (out_dir /
                f"hp_search_{args.variant}_{args.arch}{dataset_tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results, "best": best},
                  f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
