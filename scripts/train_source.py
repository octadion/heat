"""
Train source model on CIFAR-10. Supports multiple architectures.

NEW in v1.2: --arch flag for cross-architecture validation (Opsi 3).

Saves checkpoints at epochs 0, 5, 10, 25, final (for P3 dose-response).
For WRN-28-10, training takes ~4× longer than ResNet-18; recommend running
with --epochs 30 unless full benchmarking accuracy is needed.

Usage:
  python scripts/train_source.py --arch resnet18 --epochs 30
  python scripts/train_source.py --arch wrn28_10 --epochs 30 \\
      --batch-size 128 --lr 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import get_cifar10_loaders
from src.models import build_arch
from src.utils import set_seed, get_device


CHECKPOINT_EPOCHS = [0, 5, 10, 25]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--ckpt-dir", type=str, default="experiments/checkpoints")
    return p.parse_args()


def save_checkpoint(model, optimizer, epoch, ckpt_dir, arch, label):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{arch}_{label}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "arch": arch,
    }, Path(ckpt_dir) / fname)
    print(f"  [ckpt] {fname}")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(
        args.cifar10_root, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_arch(args.arch, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Save epoch 0 (random init / immediately after first call) for P3.
    save_checkpoint(model, optimizer, 0, args.ckpt_dir, args.arch, "epoch0")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_total += y.size(0)
        scheduler.step()
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        if epoch in CHECKPOINT_EPOCHS:
            save_checkpoint(model, optimizer, epoch, args.ckpt_dir,
                            args.arch, f"epoch{epoch}")

    # Final
    save_checkpoint(model, optimizer, args.epochs, args.ckpt_dir,
                    args.arch, "final")
    print(f"\nDone. Final test acc: {evaluate(model, test_loader, device):.4f}")


if __name__ == "__main__":
    main()
