"""
Protocol P4 — Where-ness (gradient profile per stage).

v1.4 BUG FIX: parameter naming convention was wrong in v1.3.
ResNet18CIFAR uses `stage1`, `stage2`, `stage3`, `stage4` (1-indexed, prefix
"stage") — NOT `layer1`, `layer2`, etc. (which is torchvision convention).
Fixed mapping below for both architectures.

For each corruption: run HEAT, record per-stage gradient norms aggregated
over batches. Output mean profile per corruption + overall mean.
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
                   default=["gaussian_noise", "shot_noise", "defocus_blur",
                            "snow", "contrast", "frost", "brightness"])
    p.add_argument("--heat-lr", type=float, default=1e-3)
    p.add_argument("--heat-momentum", type=float, default=0.0)
    p.add_argument("--heat-temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--heat-aggregation", type=str, default="sum",
                   choices=["sum", "self_gated", "self_gated_temperature"])
    p.add_argument("--heat-eval-mode", action="store_true")
    p.add_argument("--out-dir", type=str, default="experiments/results")
    return p.parse_args()


def stage_index_from_param_name(name: str, arch: str) -> int | None:
    """
    Map parameter name -> stage index (0-based) or None for non-stage params.

    ResNet18CIFAR (resnet_cifar.py):
      conv1, bn1     → stem (None)
      stage1.*       → 0
      stage2.*       → 1
      stage3.*       → 2
      stage4.*       → 3
      linear         → head (None)

    WideResNet (wrn_cifar.py):
      conv1          → stem (None)
      block1.*       → 0
      block2.*       → 1
      block3.*       → 2
      bn1, linear    → head (None)
    """
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
    raise ValueError(f"unknown arch: {arch}")


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = build_arch(args.arch, num_classes=10).to(device)
    base_model.load_state_dict(ckpt["model"])

    num_stages = len(base_model.stage_channels)
    profiles = {}

    print(f"[ok] arch={args.arch}, num_stages={num_stages}")
    print(f"[ok] HEAT config: lr={args.heat_lr}, "
          f"temperatures={args.heat_temperatures}, "
          f"aggregation={args.heat_aggregation}")

    # Sanity check: print which params would be matched
    test_model = build_arch(args.arch, num_classes=10)
    matched_per_stage = [0] * num_stages
    matched_other = 0
    for name, p in test_model.named_parameters():
        s = stage_index_from_param_name(name, args.arch)
        if s is not None:
            matched_per_stage[s] += 1
        else:
            matched_other += 1
    print(f"[sanity] params per stage: {matched_per_stage}, "
          f"non-stage params: {matched_other}")

    for corruption in args.corruptions:
        print(f"\n=== {corruption} ===")
        loader = get_cifar10c_loader(
            args.c10c_root, corruption, severity=args.severity,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False,
        )

        set_seed(args.seed)
        m = copy.deepcopy(base_model).to(device)
        heat = HEAT(
            m, lr=args.heat_lr, momentum=args.heat_momentum,
            stages=list(range(num_stages)),
            temperatures=args.heat_temperatures,
            aggregation=args.heat_aggregation,
            eval_mode=args.heat_eval_mode,
        ).to(device)

        stage_sumsq = [0.0] * num_stages
        n_batches = 0
        correct = total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds, diags = heat.adapt_with_diagnostics(x)
            for name, gnorm in diags["param_grad_norms"].items():
                s = stage_index_from_param_name(name, args.arch)
                if s is not None:
                    stage_sumsq[s] += gnorm * gnorm
            n_batches += 1
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

        stage_norms = [(v / n_batches) ** 0.5 for v in stage_sumsq]
        acc = correct / total
        profiles[corruption] = {
            "per_stage_norms": stage_norms,
            "accuracy": acc,
        }
        print(f"  acc={acc:.4f}")
        print(f"  profile (raw):   {[f'{v:.3e}' for v in stage_norms]}")
        m_max = max(stage_norms)
        normed = [v / m_max for v in stage_norms] if m_max > 0 else [0.0] * num_stages
        print(f"  profile (norm):  {[f'{v:.3f}' for v in normed]}")

    # Aggregate mean profile
    print("\n=== Mean profile across corruptions ===")
    mean_profile = [0.0] * num_stages
    for c, info in profiles.items():
        for i, v in enumerate(info["per_stage_norms"]):
            mean_profile[i] += v
    mean_profile = [v / len(profiles) for v in mean_profile]
    print(f"  raw:   {[f'{v:.3e}' for v in mean_profile]}")
    m_max = max(mean_profile)
    if m_max > 0:
        print(f"  norm:  {[f'{v:.3f}' for v in [v/m_max for v in mean_profile]]}")
    else:
        print(f"  norm:  all zero — likely a bug, check stage name pattern")

    out_path = out_dir / f"p4_{args.arch}_seed{args.seed}_sev{args.severity}.json"
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "num_stages": num_stages,
            "profiles": profiles,
            "mean_profile": mean_profile,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
