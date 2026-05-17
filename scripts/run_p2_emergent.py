"""
Protocol P2 — Emergent capacity (pretrained vs random init).

Tests whether HEAT's gain is ENABLED by pretraining (i.e., emerges from
learned features), or works equally well from random init (i.e., generic
loss-minimization regardless of feature quality).

Compares:
  - HEAT on pretrained model (final.pt)
  - HEAT on random init (epoch 0)

For each, evaluate on 5 corruptions and report mean acc.

If pretrained >> random → emergence supported (HEAT exploits learned features).
If both similar  → not emergent (HEAT is just generic optimization).

v1.2 additions:
  --arch, --heat-temperatures, --heat-aggregation
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--checkpoint-pretrained", type=str, required=True,
                   help="Final epoch checkpoint")
    p.add_argument("--checkpoint-random", type=str, required=True,
                   help="Epoch 0 (essentially random) checkpoint")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=["gaussian_noise", "shot_noise", "defocus_blur",
                            "snow", "contrast"])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["source", "tent", "tea", "heat"])
    p.add_argument("--out-dir", type=str, default="experiments/results")
    add_method_args(p)
    return p.parse_args()


def load_model(arch, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_arch(arch, num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate_methods(args, base_model, device, label):
    accs = {m: {} for m in args.methods}
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
            accs[m_name][c] = acc
            print(f"  [{label}] {m_name:10s} {c:20s}  acc={acc:.4f}")
    return accs


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_model = load_model(args.arch, args.checkpoint_pretrained, device)
    random_model = load_model(args.arch, args.checkpoint_random, device)

    print("\n=== Pretrained ===")
    pretrained_results = evaluate_methods(args, pretrained_model, device, "pre")

    print("\n=== Random init ===")
    random_results = evaluate_methods(args, random_model, device, "rnd")

    # Summary
    print("\n" + "=" * 70)
    print("P2 EMERGENT CAPACITY")
    print("=" * 70)
    print(f"  {'method':10s}  {'pretrained':>12s}  {'random':>12s}  {'gain':>10s}")
    for m in args.methods:
        pre_mean = sum(pretrained_results[m].values()) / len(pretrained_results[m])
        rnd_mean = sum(random_results[m].values()) / len(random_results[m])
        gain = pre_mean - rnd_mean
        print(f"  {m:10s}  {pre_mean:>12.4f}  {rnd_mean:>12.4f}  {gain:>+10.4f}")

    out_path = out_dir / f"p2_{args.arch}_seed{args.seed}_sev{args.severity}.json"
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "pretrained": pretrained_results,
            "random": random_results,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
