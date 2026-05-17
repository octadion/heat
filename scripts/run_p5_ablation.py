"""
Protocol P5 — Hierarchy ablation (THREE-axis), v1.2 with --arch support.

Three axes:
  HIERARCHY DEPTH × TEMPERATURE × UPDATE SET

In v1, P5 found feature-stage hierarchy empirically flat (Δ ≤ 0.0015).
v1.1 added temperature-scale axis; v1.2 of patches confirmed temperature is
also flat at matched stability (LR amplifier, not signal).

This script keeps the full grid available for thoroughness and final paper
ablation reporting (showing negative result is honest).

For ResNet-18: stages = [0..3]  (4 stages).
For WRN-28-10: stages = [0..2]  (3 stages).
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
from src.methods import HEAT, Source
from src.adapt import evaluate_online
from src.utils import set_seed, get_device


P5_DEFAULT_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "defocus_blur", "snow", "contrast",
]

TEMPERATURE_VARIANTS = {
    "single_temp":     [1.0],
    "triple_temp":     [0.5, 1.0, 2.0],
    "quintuple_temp":  [0.25, 0.5, 1.0, 2.0, 4.0],
}

UPDATE_SET_VARIANTS = {
    "all_params": True,
    "bn_only":    False,
}


def get_hierarchy_variants(num_stages: int) -> dict[str, list[int]]:
    """Build hierarchy variants based on number of stages in the arch."""
    if num_stages == 4:
        return {
            "output_only":  [3],
            "two_level":    [1, 3],
            "three_level":  [1, 2, 3],
            "full":         [0, 1, 2, 3],
        }
    elif num_stages == 3:
        return {
            "output_only":  [2],
            "two_level":    [1, 2],
            "full":         [0, 1, 2],
        }
    else:
        # Generic fallback
        return {
            "output_only":  [num_stages - 1],
            "full":         list(range(num_stages)),
        }


QUICK_TEMPERATURE = ["single_temp", "triple_temp"]
QUICK_UPDATE_SET = ["all_params"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=P5_DEFAULT_CORRUPTIONS)
    p.add_argument("--heat-lr", type=float, default=1e-3)
    p.add_argument("--heat-momentum", type=float, default=0.0)
    p.add_argument("--heat-eval-mode", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = build_arch(args.arch, num_classes=10).to(device)
    base_model.load_state_dict(ckpt["model"])
    num_stages = len(base_model.stage_channels)

    hierarchy_variants = get_hierarchy_variants(num_stages)

    if args.quick:
        h_keys = ["output_only", "full"]
        t_keys = QUICK_TEMPERATURE
        u_keys = QUICK_UPDATE_SET
    else:
        h_keys = list(hierarchy_variants.keys())
        t_keys = list(TEMPERATURE_VARIANTS.keys())
        u_keys = list(UPDATE_SET_VARIANTS.keys())

    print(f"[ok] arch={args.arch}, num_stages={num_stages}")
    print(f"[ok] grid: {len(h_keys)} hierarchy × {len(t_keys)} temp × {len(u_keys)} update")

    results: dict = {"source": {}}
    for h in h_keys:
        for t in t_keys:
            for u in u_keys:
                results[f"{h}__{t}__{u}"] = {}

    for corruption in args.corruptions:
        print(f"\n=== corruption: {corruption} ===")
        loader = get_cifar10c_loader(
            args.c10c_root, corruption, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
        )

        # Source
        set_seed(args.seed)
        m = copy.deepcopy(base_model).to(device)
        acc = evaluate_online(Source(m), loader, device,
                              progress=False).accuracy
        results["source"][corruption] = acc
        print(f"  source: acc={acc:.4f}")

        for h in h_keys:
            for t in t_keys:
                for u in u_keys:
                    set_seed(args.seed)
                    m = copy.deepcopy(base_model).to(device)
                    heat = HEAT(
                        m,
                        lr=args.heat_lr,
                        momentum=args.heat_momentum,
                        stages=hierarchy_variants[h],
                        temperatures=TEMPERATURE_VARIANTS[t],
                        update_all_params=UPDATE_SET_VARIANTS[u],
                        eval_mode=args.heat_eval_mode,
                    ).to(device)
                    acc = evaluate_online(heat, loader, device,
                                          progress=False).accuracy
                    key = f"{h}__{t}__{u}"
                    results[key][corruption] = acc
                    print(f"  {h:13s} | {t:15s} | {u:10s}  acc={acc:.4f}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"P5 — arch={args.arch}, mean across {len(args.corruptions)} corruptions")
    print("=" * 80)
    src_mean = sum(results["source"].values()) / len(results["source"])
    print(f"  source baseline: {src_mean:.4f}\n")

    for u in u_keys:
        print(f"  --- update set: {u} ---")
        header = f"  {'temperature':17s} |" + "".join(
            f" {h:>13s} |" for h in h_keys
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in t_keys:
            row = f"  {t:17s} |"
            for h in h_keys:
                key = f"{h}__{t}__{u}"
                vals = list(results[key].values())
                mean = sum(vals) / len(vals)
                row += f" {mean:>13.4f} |"
            print(row)
        print()

    out_path = out_dir / f"p5_{args.arch}_seed{args.seed}_sev{args.severity}.json"
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "hierarchy_variants": {k: hierarchy_variants[k] for k in h_keys},
            "temperature_variants": {k: TEMPERATURE_VARIANTS[k] for k in t_keys},
            "update_set_variants": {k: UPDATE_SET_VARIANTS[k] for k in u_keys},
            "results": results,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
