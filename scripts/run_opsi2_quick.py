"""
Quick experiment: Opsi 2 — self-gating ablation.

Compares three aggregation modes for HEAT on a small subset of corruptions
to determine if self-gating provides empirical signal beyond plain sum:

  - sum                       (v1.1 default; baseline)
  - self_gated                (softmax across stages on -E_l)
  - self_gated_temperature    (softmax across temperatures on -E_τ)

Each is run at the same lr (passed as --lr; recommend the v1.2 Pareto best
of 1e-3 single-temp).

Decision rule:
  - If self_gated mean acc > sum mean acc by >= 0.5pt with comparable P9,
    self-gating shows mechanistic signal worth keeping.
  - If self_gated <= sum, self-gating drops; report negative result in paper.

Total runtime: ~10-15 minutes on L4 (3 modes × 5 corruptions).

Usage:
  python scripts/run_opsi2_quick.py \
      --checkpoint experiments/checkpoints/final.pt \
      --lr 1e-3
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
from src.models import resnet18_cifar
from src.methods import HEAT
from src.adapt import evaluate_online
from src.utils import set_seed, get_device


CORRUPTIONS = ["gaussian_noise", "shot_noise", "defocus_blur", "snow", "contrast"]

AGGREGATIONS = ["sum", "self_gated", "self_gated_temperature"]

# Test all aggregations at single-temp AND triple-temp to see which
# combination wins. self_gated only meaningful when len(stages) > 1.
CONFIGS = [
    # (label, stages, temperatures)
    ("full_singletemp", [0, 1, 2, 3], [1.0]),
    ("full_tripletemp", [0, 1, 2, 3], [0.5, 1.0, 2.0]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--corruptions", type=str, nargs="+", default=CORRUPTIONS)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = resnet18_cifar(num_classes=10).to(device)
    base_model.load_state_dict(ckpt["model"])

    # results[config_label][aggregation][corruption] = acc
    results: dict = {}
    for cfg_label, _, _ in CONFIGS:
        results[cfg_label] = {agg: {} for agg in AGGREGATIONS}

    for corruption in args.corruptions:
        print(f"\n=== corruption: {corruption} ===")
        loader = get_cifar10c_loader(
            args.c10c_root, corruption, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
        )

        for cfg_label, stages, temps in CONFIGS:
            for agg in AGGREGATIONS:
                # self_gated_temperature only meaningful when len(temps) > 1
                if agg == "self_gated_temperature" and len(temps) == 1:
                    continue
                # self_gated only meaningful when len(stages) > 1; we have 4
                # stages here so it's always valid

                set_seed(args.seed)
                m = copy.deepcopy(base_model).to(device)
                heat = HEAT(
                    m,
                    lr=args.lr,
                    momentum=0.0,
                    stages=stages,
                    temperatures=temps,
                    aggregation=agg,
                ).to(device)
                acc = evaluate_online(heat, loader, device,
                                      progress=False).accuracy
                results[cfg_label][agg][corruption] = acc
                print(f"  {cfg_label:20s} | {agg:24s}  acc={acc:.4f}")

    # Mean summary
    print("\n" + "=" * 80)
    print("MEAN ACCURACY ACROSS CORRUPTIONS")
    print("=" * 80)
    for cfg_label in results:
        print(f"\n  {cfg_label}:")
        for agg in AGGREGATIONS:
            cell = results[cfg_label].get(agg, {})
            if not cell:
                print(f"    {agg:24s}  (n/a)")
                continue
            mean = sum(cell.values()) / len(cell)
            print(f"    {agg:24s}  {mean:.4f}")

    # Decision summary
    print("\n" + "=" * 80)
    print("DECISION CHECKS")
    print("=" * 80)
    for cfg_label in results:
        sum_cell = results[cfg_label].get("sum", {})
        gated_cell = results[cfg_label].get("self_gated", {})
        if sum_cell and gated_cell:
            sum_mean = sum(sum_cell.values()) / len(sum_cell)
            gated_mean = sum(gated_cell.values()) / len(gated_cell)
            delta = gated_mean - sum_mean
            verdict = (
                "POSITIVE — self_gated shows signal worth keeping."
                if delta >= 0.005 else
                ("MARGINAL — re-check at full corruption set." if delta >= 0
                 else "NEGATIVE — self_gated does not help; drop.")
            )
            print(f"  {cfg_label}: self_gated vs sum delta = {delta:+.4f}  →  {verdict}")

    out_path = out_dir / f"opsi2_quick_lr{args.lr:g}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
