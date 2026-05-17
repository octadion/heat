"""
Cross-architecture validation of where-emergence (Opsi 3).

v1.4 BUG FIX: stage name patterns corrected for ResNet18CIFAR (uses `stage1`
prefix, not `layer1`). Same fix as run_p4_whereness.py.
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
from src.methods import HEAT
from src.utils import set_seed, get_device


CORRUPTIONS = ["gaussian_noise", "shot_noise", "defocus_blur", "snow", "contrast"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resnet-checkpoint", type=str, required=True)
    p.add_argument("--wrn-checkpoint", type=str, required=True)
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--corruptions", type=str, nargs="+", default=CORRUPTIONS)
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--aggregation", type=str, default="sum")
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def stage_index_from_param_name(name: str, arch: str) -> int | None:
    if arch == "resnet18":
        for i in range(4):
            if name.startswith(f"stage{i+1}."):
                return i
        return None
    elif arch == "wrn28_10":
        for i in range(3):
            if name.startswith(f"block{i+1}."):
                return i
        return None
    raise ValueError(arch)


def measure_gradient_profile(arch, checkpoint, args, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base_model = build_arch(arch, num_classes=10).to(device)
    base_model.load_state_dict(ckpt["model"])

    num_stages = len(base_model.stage_channels)
    profile_per_corruption = {}

    for corruption in args.corruptions:
        loader = get_cifar10c_loader(
            args.c10c_root, corruption, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
        )

        set_seed(args.seed)
        m = copy.deepcopy(base_model).to(device)
        heat = HEAT(
            m, lr=args.lr, momentum=0.0,
            stages=list(range(num_stages)),
            temperatures=args.temperatures,
            aggregation=args.aggregation,
        ).to(device)

        stage_sumsq = [0.0] * num_stages
        n_batches = 0

        for x, _ in loader:
            x = x.to(device)
            _, diags = heat.adapt_with_diagnostics(x)
            for name, gnorm in diags["param_grad_norms"].items():
                s = stage_index_from_param_name(name, arch)
                if s is not None:
                    stage_sumsq[s] += gnorm * gnorm
            n_batches += 1

        stage_norms = [(v / n_batches) ** 0.5 for v in stage_sumsq]
        profile_per_corruption[corruption] = stage_norms
        print(f"  [{arch}] {corruption}: {[f'{n:.3e}' for n in stage_norms]}")

    per_stage_norms = [0.0] * num_stages
    for prof in profile_per_corruption.values():
        for i, v in enumerate(prof):
            per_stage_norms[i] += v
    per_stage_norms = [v / len(profile_per_corruption) for v in per_stage_norms]

    return {
        "arch": arch,
        "num_stages": num_stages,
        "per_stage_norms": per_stage_norms,
        "per_corruption": profile_per_corruption,
    }


def normalize_profile(profile_list):
    m = max(profile_list)
    return [v / m for v in profile_list] if m > 0 else profile_list


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Cross-arch where-emergence study ===")
    print(f"Corruptions: {args.corruptions}")
    print(f"HEAT: lr={args.lr}, temps={args.temperatures}, agg={args.aggregation}")

    results = {}

    print(f"\n--- ResNet-18 ---")
    results["resnet18"] = measure_gradient_profile(
        "resnet18", args.resnet_checkpoint, args, device
    )

    print(f"\n--- WRN-28-10 ---")
    results["wrn28_10"] = measure_gradient_profile(
        "wrn28_10", args.wrn_checkpoint, args, device
    )

    print("\n" + "=" * 70)
    print("PER-STAGE GRADIENT PROFILE (mean across corruptions)")
    print("=" * 70)
    for arch, info in results.items():
        norms = info["per_stage_norms"]
        normed = normalize_profile(norms)
        print(f"\n  {arch} (num_stages={info['num_stages']}):")
        print(f"    raw      {[f'{n:.3e}' for n in norms]}")
        print(f"    norm-1   {[f'{n:.3f}' for n in normed]}")

    print("\n" + "=" * 70)
    print("INVARIANCE CHECK")
    print("=" * 70)
    rn_normed = normalize_profile(results["resnet18"]["per_stage_norms"])
    wrn_normed = normalize_profile(results["wrn28_10"]["per_stage_norms"])

    rn_argmax = rn_normed.index(max(rn_normed))
    wrn_argmax = wrn_normed.index(max(wrn_normed))
    print(f"  ResNet-18 dominant stage:  {rn_argmax} of {len(rn_normed)-1}")
    print(f"  WRN-28-10 dominant stage:  {wrn_argmax} of {len(wrn_normed)-1}")

    rn_relative = rn_argmax / max(1, len(rn_normed) - 1)
    wrn_relative = wrn_argmax / max(1, len(wrn_normed) - 1)
    if abs(rn_relative - wrn_relative) < 0.2:
        print("  → Profiles consistent across architectures.")
    else:
        print("  → Profiles differ; framing must address this.")

    out_path = out_dir / f"opsi3_cross_arch_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
