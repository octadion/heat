"""
Train source model on CIFAR-10 / CIFAR-100. Supports multiple architectures.

NEW in v1.2: --arch flag for cross-architecture validation (Opsi 3).
NEW in vit/cifar100 patch:
  --arch vit_s: finetunes timm ViT-S/16 (ImageNet pretrained backbone) with
    a CIFAR-resolution-resized + ImageNet-normalized pipeline. Uses AdamW
    + cosine schedule + mixed precision + label smoothing.
  --dataset {cifar10, cifar100}: chooses the clean dataset and num_classes.

Existing CNN behavior (resnet18, wrn28_10 on cifar10) is unchanged — same
SGD/momentum/cosine recipe, same checkpoint filenames.

Saves checkpoints at epochs 0, 5, 10, 25, final (for P3 dose-response).
For WRN-28-10, training takes ~4x longer than ResNet-18; recommend running
with --epochs 30 unless full benchmarking accuracy is needed.

Usage:
  python scripts/train_source.py --arch resnet18 --epochs 30
  python scripts/train_source.py --arch wrn28_10 --epochs 30 \\
      --batch-size 128 --lr 0.1
  python scripts/train_source.py --arch vit_s --dataset cifar10 \\
      --epochs 10 --batch-size 64 --vit-lr 1e-4 --vit-weight-decay 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_clean_loaders, num_classes_for
from src.models import build_arch
from src.utils import set_seed, get_device


CHECKPOINT_EPOCHS = [0, 5, 10, 25]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10", "vit_s"])
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--epochs", type=int, default=30,
                   help="Epochs. CNN default 30; for vit_s consider 10.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="CNN default 128; for vit_s consider 64 (224x224).")
    p.add_argument("--num-workers", type=int, default=2)

    # CNN optimizer (unchanged)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    # ViT-only optimizer (AdamW + cosine + AMP + label smoothing).
    p.add_argument("--vit-lr", type=float, default=1e-4,
                   help="AdamW learning rate for vit_s.")
    p.add_argument("--vit-weight-decay", type=float, default=0.05,
                   help="AdamW weight decay for vit_s.")
    p.add_argument("--vit-label-smoothing", type=float, default=0.1)
    p.add_argument("--vit-no-amp", action="store_true",
                   help="Disable mixed precision for vit_s.")
    p.add_argument("--vit-pretrained", action="store_true", default=True,
                   help="(default) Use ImageNet-pretrained ViT backbone.")
    p.add_argument("--vit-from-scratch", dest="vit_pretrained",
                   action="store_false",
                   help="Train ViT backbone from scratch (not recommended).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--cifar100-root", type=str, default="data/cifar100")
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


# ---------------------------------------------------------------------------
# CNN training path — unchanged from before (must reproduce existing runs).
# ---------------------------------------------------------------------------

def train_cnn(args, model, train_loader, test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

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

    save_checkpoint(model, optimizer, args.epochs, args.ckpt_dir,
                    args.arch, "final")
    print(f"\nDone. Final test acc: {evaluate(model, test_loader, device):.4f}")


# ---------------------------------------------------------------------------
# ViT finetuning path — AdamW + cosine + AMP + label smoothing.
# ---------------------------------------------------------------------------

def train_vit(args, model, train_loader, test_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=args.vit_lr,
                            weight_decay=args.vit_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.vit_label_smoothing)

    use_amp = (not args.vit_no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_checkpoint(model, optimizer, 0, args.ckpt_dir, args.arch, "epoch0")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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

    save_checkpoint(model, optimizer, args.epochs, args.ckpt_dir,
                    args.arch, "final")
    print(f"\nDone. Final test acc: {evaluate(model, test_loader, device):.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    dataset_root = args.cifar10_root if args.dataset == "cifar10" else args.cifar100_root
    num_classes = num_classes_for(args.dataset)

    train_loader, test_loader = get_clean_loaders(
        args.dataset,
        dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        arch=args.arch,
    )

    if args.arch == "vit_s":
        model = build_arch("vit_s", num_classes=num_classes,
                           pretrained=args.vit_pretrained).to(device)
        train_vit(args, model, train_loader, test_loader, device)
    else:
        model = build_arch(args.arch, num_classes=num_classes).to(device)
        train_cnn(args, model, train_loader, test_loader, device)


if __name__ == "__main__":
    main()
