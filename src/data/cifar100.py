"""CIFAR-100 loaders. Mirrors cifar10.py; uses the shared arch-aware
transform helpers so ViT-S/16 finetuning works without changes."""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets

from .cifar10 import get_train_transform, get_eval_transform


def get_cifar100_loaders(
    data_root: str = "data/cifar100",
    batch_size: int = 128,
    num_workers: int = 2,
    arch: Optional[str] = None,
):
    train_set = datasets.CIFAR100(
        data_root, train=True, download=True,
        transform=get_train_transform(arch),
    )
    test_set = datasets.CIFAR100(
        data_root, train=False, download=True,
        transform=get_eval_transform(arch),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
