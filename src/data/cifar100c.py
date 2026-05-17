"""
CIFAR-100-C — Hendrycks & Dietterich corruption benchmark, 100-class variant.

Mirrors cifar10c.py exactly. Same 15 corruptions + 4 extras, same 5
severities, same severity-major layout (50000 images = 10000 per severity)
per corruption .npy file.

Files expected (download from Zenodo: https://zenodo.org/records/3555552):
  data/cifar100c/
    gaussian_noise.npy          # (50000, 32, 32, 3) uint8
    ...
    labels.npy                  # (50000,) int

If the directory or any required file is missing we raise a clear
FileNotFoundError naming the expected path; we do NOT auto-download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .cifar10 import get_eval_transform
from .cifar10c import (
    CORRUPTIONS,
    EXTRA_CORRUPTIONS,
    SEVERITIES,
    NUM_PER_SEVERITY,
)


class CIFAR100C(Dataset):
    """One (corruption, severity) split of CIFAR-100-C."""

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int = 5,
        transform: Optional[transforms.Compose] = None,
        arch: Optional[str] = None,
    ):
        super().__init__()
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
        if corruption not in CORRUPTIONS + EXTRA_CORRUPTIONS:
            raise ValueError(f"unknown corruption: {corruption}")

        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(
                f"CIFAR-100-C root not found: {root_path}\n"
                f"Download CIFAR-100-C from https://zenodo.org/records/3555552 "
                f"and extract the .npy files into '{root_path}'."
            )

        data_path = root_path / f"{corruption}.npy"
        labels_path = root_path / "labels.npy"
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} not found. Download CIFAR-100-C from "
                f"https://zenodo.org/records/3555552 and extract into "
                f"'{root_path}'."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"{labels_path} not found. Download CIFAR-100-C from "
                f"https://zenodo.org/records/3555552 and extract into "
                f"'{root_path}'."
            )

        all_data = np.load(data_path)         # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path)     # (50000,) int

        start = (severity - 1) * NUM_PER_SEVERITY
        end = start + NUM_PER_SEVERITY
        self.data = all_data[start:end]
        self.labels = all_labels[start:end]

        if transform is None:
            transform = get_eval_transform(arch)
        self.transform = transform
        self.corruption = corruption
        self.severity = severity

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        label = int(self.labels[idx])
        x = self.transform(img)
        return x, label


def get_cifar100c_loader(
    root: str,
    corruption: str,
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = False,
    arch: Optional[str] = None,
) -> DataLoader:
    ds = CIFAR100C(root, corruption, severity, arch=arch)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )
