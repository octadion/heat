"""
Multi-criteria HP search v1.2 — with --arch support.

Picks LR by Pareto-feasible criterion:
  maximize    holdout_p1_mean
  subject to  holdout_p9_last_acc >= chance + margin

Same logic as v1.1 but supports both ResNet-18 and WRN-28-10.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_cifar10c_loader
from src.models import build_arch
from src.methods import HEAT, Tent, TEA, EPOTTA, ReTTA
from src.adapt import evaluate_online
from src.utils import set_seed, get_device


HOLDOUT_CORRUPTIONS = ["gaussian_blur", "saturate", "spatter", "speckle_noise"]

DEFAULT_LR_GRID = {
    "heat":   [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "tent":   [1e-4, 5e-4, 1e-3, 5e-3],
    "tea":    [1e-4, 5e-4, 1e-3],
    "epotta": [1e-4, 5e-4, 1e-3, 5e-3],
    "retta":  [1e-4, 5e-4, 1e-3, 5e-3],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--variant", type=str, default="heat",
                   choices=["heat", "tent", "tea", "epotta", "retta"])
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
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
    p.add_argument("--eval-modes", type=str, nargs="+", default=None)

    # Constraint
    p.add_argument("--p9-margin-pt", type=float, default=15.0)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def build_method(variant, base_model, device, lr, args, eval_mode_bool):
    m = copy.deepcopy(base_model).to(device)
    if variant == "heat":
        return HEAT(
            m, lr=lr, momentum=0.0,
            temperatures=args.temperatures,
            aggregation=args.aggregation,
            update_all_params=args.update_all_params,
            eval_mode=eval_mode_bool,
        ).to(device)
    elif variant == "tent":
        return Tent(m, lr=lr, optimizer_name="adam", momentum=0.9)
    elif variant == "tea":
        return TEA(m, lr=lr, optimizer_name="adam",
                   sgld_steps=20, sgld_lr=0.1)
    elif variant == "epotta":
        from src.data import get_cifar10_loaders
        train_loader, _ = get_cifar10_loaders(
            args.cifar10_root, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return EPOTTA(m, source_loader=train_loader, lr=lr,
                      buffer_size=500, device=device)
    elif variant == "retta":
        return ReTTA(m, lr=lr, lambda_energy=1.0)
    raise ValueError(variant)


def evaluate_single_domain(factory, corruptions, args, device):
    accs = []
    for c in corruptions:
        loader = get_cifar10c_loader(
            args.c10c_root, c, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
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
        loader = get_cifar10c_loader(
            args.c10c_root, c, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = build_arch(args.arch, num_classes=10).to(device)
    base_model.load_state_dict(ckpt["model"])

    lr_grid = args.lr_grid or DEFAULT_LR_GRID[args.variant]

    if args.variant == "heat":
        if args.eval_modes is None:
            eval_modes = [True, False]
        else:
            eval_modes = [s.lower() == "true" for s in args.eval_modes]
    else:
        eval_modes = [None]

    chance_acc = 0.10
    p9_threshold = chance_acc + args.p9_margin_pt / 100.0

    results = []
    print(f"\n=== HP search variant={args.variant}, arch={args.arch} ===")
    print(f"  LR grid: {lr_grid}")
    if args.variant == "heat":
        print(f"  eval_modes: {eval_modes}")

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

    out_path = out_dir / f"hp_search_{args.variant}_{args.arch}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results, "best": best},
                  f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
