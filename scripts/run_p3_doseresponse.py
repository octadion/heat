"""
Protocol P3 — Dose-response across pretraining checkpoints.

Tests how much pretraining is needed for HEAT's gain to emerge. Sweeps
checkpoints saved at epochs {0, 5, 10, 25, final} and runs HEAT on each.

If gain monotonically increases with pretraining, "emergence from learned
features" is supported as dose-response.

v1.2 additions:
  --arch, --heat-temperatures, --heat-aggregation, etc.
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
from src.adapt import evaluate_online
from src.utils import set_seed, get_device
from src.utils.method_factory import build_method, add_method_args


CHECKPOINT_LABELS = ["epoch0", "epoch5", "epoch10", "epoch25", "final"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Directory containing {arch}_epoch0.pt, ..., {arch}_final.pt")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=["gaussian_noise", "defocus_blur", "snow"])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["source", "tent", "heat"])
    p.add_argument("--checkpoints", type=str, nargs="+",
                   default=CHECKPOINT_LABELS)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    add_method_args(p)
    return p.parse_args()


def load_model(arch, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_arch(arch, num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)

    results = {ckpt_label: {m: {} for m in args.methods}
               for ckpt_label in args.checkpoints}

    for ckpt_label in args.checkpoints:
        ckpt_path = ckpt_dir / f"{args.arch}_{ckpt_label}.pt"
        if not ckpt_path.exists():
            # Fallback to v1 naming convention (without arch prefix)
            ckpt_path = ckpt_dir / f"{ckpt_label}.pt"
        if not ckpt_path.exists():
            print(f"[skip] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n=== checkpoint: {ckpt_label} ({ckpt_path}) ===")
        base_model = load_model(args.arch, str(ckpt_path), device)

        for c in args.corruptions:
            loader = get_cifar10c_loader(
                args.c10c_root, c, severity=args.severity,
                batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=False,
            )
            for m_name in args.methods:
                set_seed(args.seed)
                method = build_method(m_name, base_model, device, args,
                                      dataset_root=args.cifar10_root)
                acc = evaluate_online(method, loader, device,
                                      progress=False).accuracy
                results[ckpt_label][m_name][c] = acc
                print(f"  {m_name:10s} {c:20s}  acc={acc:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("P3 DOSE-RESPONSE")
    print("=" * 80)
    header = f"  {'method':10s}" + "".join(f"  {ck:>8s}" for ck in args.checkpoints)
    print(header)
    for m in args.methods:
        row = f"  {m:10s}"
        for ck in args.checkpoints:
            cell = results[ck][m]
            if cell:
                mean = sum(cell.values()) / len(cell)
                row += f"  {mean:>8.4f}"
            else:
                row += f"  {'n/a':>8s}"
        print(row)

    out_path = out_dir / f"p3_{args.arch}_seed{args.seed}_sev{args.severity}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
