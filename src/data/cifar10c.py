"""
CIFAR-10-C — Hendrycks & Dietterich corruption benchmark.

The dataset is a directory of .npy files, one per corruption type:
  data/cifar10c/
    gaussian_noise.npy           # shape (50000, 32, 32, 3) uint8
    shot_noise.npy
    ...
    labels.npy                   # shape (50000,) — same labels for all corruption files

Each corruption file has 50000 = 10000 (test set size) * 5 severities, in
severity-major order: indices 0..9999 are severity 1, 10000..19999 are
severity 2, etc.

We provide:
  - CIFAR10C — torch Dataset for one (corruption, severity) pair
  - get_cifar10c_loader — convenience wrapper

Download script: scripts/download_cifar10c.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .cifar10 import CIFAR10_MEAN, CIFAR10_STD


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
EXTRA_CORRUPTIONS = [
    "gaussian_blur", "saturate", "spatter", "speckle_noise",
]
SEVERITIES = [1, 2, 3, 4, 5]
NUM_PER_SEVERITY = 10000


class CIFAR10C(Dataset):
    """One (corruption, severity) split of CIFAR-10-C."""

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int = 5,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
        if corruption not in CORRUPTIONS + EXTRA_CORRUPTIONS:
            raise ValueError(f"unknown corruption: {corruption}")

        root = Path(root)
        data_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} not found. Run scripts/download_cifar10c.py first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"{labels_path} not found.")

        # Load and slice to the requested severity
        all_data = np.load(data_path)            # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path)        # (50000,) int

        start = (severity - 1) * NUM_PER_SEVERITY
        end = start + NUM_PER_SEVERITY
        self.data = all_data[start:end]
        self.labels = all_labels[start:end]

        # Default transform: normalize like CIFAR-10 test set
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),  # uint8 HWC -> float CHW in [0,1]
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        self.transform = transform
        self.corruption = corruption
        self.severity = severity

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]                     # uint8 (32, 32, 3)
        label = int(self.labels[idx])
        # ToTensor expects HWC uint8 numpy or PIL — we have HWC uint8, fine
        x = self.transform(img)
        return x, label


def get_cifar10c_loader(
    root: str,
    corruption: str,
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    ds = CIFAR10C(root, corruption, severity)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )
