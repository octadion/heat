"""CIFAR-10 loaders. Architecture-aware transforms.

Defaults (arch=None or 'resnet18' / 'wrn28_10') match the original CIFAR
recipe: 32x32 input, CIFAR-10 normalization, no resize. Use arch='vit_s'
to get a 224-resized + ImageNet-normalized pipeline (resolved from timm's
data config — NOT hardcoded — so it tracks whatever ViT-S/16 expects).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# CNN (32x32 native) transforms — unchanged, must reproduce existing behavior.
# ---------------------------------------------------------------------------

def cifar10_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def cifar10_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# ViT (224-resized) transforms — built from timm data config so the
# normalization stays in sync with whatever pretrained ViT-S/16 we load.
# CIFAR-C images are pre-corrupted at 32x32; for ViT they are upscaled to
# 224x224 at load time (standard practice for evaluating transformers on
# CIFAR-C).
# ---------------------------------------------------------------------------

def _vit_s_interpolation_mode(cfg_interp: str) -> transforms.InterpolationMode:
    name = (cfg_interp or "bicubic").lower()
    if name == "bicubic":
        return transforms.InterpolationMode.BICUBIC
    if name == "bilinear":
        return transforms.InterpolationMode.BILINEAR
    if name == "nearest":
        return transforms.InterpolationMode.NEAREST
    return transforms.InterpolationMode.BICUBIC


def vit_s_eval_transform() -> transforms.Compose:
    """Eval transform for ViT-S/16: resize to 224, ImageNet normalize."""
    from src.models.vit_cifar import get_vit_s_data_config
    cfg = get_vit_s_data_config()
    size = cfg["input_size"][-1]   # (3, 224, 224) → 224
    interp = _vit_s_interpolation_mode(cfg.get("interpolation", "bicubic"))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), interpolation=interp, antialias=True),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def vit_s_train_transform() -> transforms.Compose:
    """
    Train transform for ViT-S/16 finetuning on CIFAR.
    Resize to 224 first, then RandomCrop with padding + flip — the standard
    CIFAR augmentation recipe transposed to 224 resolution.
    """
    from src.models.vit_cifar import get_vit_s_data_config
    cfg = get_vit_s_data_config()
    size = cfg["input_size"][-1]
    interp = _vit_s_interpolation_mode(cfg.get("interpolation", "bicubic"))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), interpolation=interp, antialias=True),
        transforms.RandomCrop(size, padding=size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


# ---------------------------------------------------------------------------
# Arch-aware dispatch
# ---------------------------------------------------------------------------

def get_train_transform(arch: Optional[str] = None) -> transforms.Compose:
    if arch == "vit_s":
        return vit_s_train_transform()
    return cifar10_train_transform()


def get_eval_transform(arch: Optional[str] = None) -> transforms.Compose:
    if arch == "vit_s":
        return vit_s_eval_transform()
    return cifar10_test_transform()


def get_cifar10_loaders(
    data_root: str = "data/cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
    arch: Optional[str] = None,
):
    train_set = datasets.CIFAR10(
        data_root, train=True, download=True,
        transform=get_train_transform(arch),
    )
    test_set = datasets.CIFAR10(
        data_root, train=False, download=True,
        transform=get_eval_transform(arch),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
