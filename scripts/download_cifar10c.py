"""
Download CIFAR-10-C (Hendrycks & Dietterich) to data/cifar10c/.

Official mirror: Zenodo. The archive is ~2.5 GB compressed and contains
.npy files for each corruption type (50000 samples each = 5 severities ×
10000 images), plus labels.npy.

Usage:
  python scripts/download_cifar10c.py
  python scripts/download_cifar10c.py --root data/cifar10c
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
ARCHIVE_NAME = "CIFAR-10-C.tar"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/cifar10c",
                   help="Final destination of the .npy files.")
    p.add_argument("--keep-archive", action="store_true",
                   help="Don't delete the .tar after extraction.")
    return p.parse_args()


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = downloaded * 100 / total_size
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} / {total_mb:.1f} MB")
        sys.stdout.flush()


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # If labels.npy already exists, assume already extracted
    if (root / "labels.npy").exists():
        print(f"[skip] {root}/labels.npy already exists. Nothing to do.")
        return

    archive_path = root / ARCHIVE_NAME
    if not archive_path.exists():
        print(f"[download] {CIFAR10C_URL}")
        print(f"[download] -> {archive_path}")
        urlretrieve(CIFAR10C_URL, archive_path, reporthook=_progress_hook)
        print()
    else:
        print(f"[skip download] {archive_path} already exists.")

    # Extract
    print(f"[extract] {archive_path}")
    with tarfile.open(archive_path, "r") as tar:
        # The tar contains a top-level CIFAR-10-C/ directory. We extract
        # there then move .npy files up one level.
        tar.extractall(path=root)

    extracted_dir = root / "CIFAR-10-C"
    if extracted_dir.exists():
        for f in extracted_dir.iterdir():
            if f.suffix == ".npy":
                target = root / f.name
                f.replace(target)
        try:
            extracted_dir.rmdir()
        except OSError:
            pass

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"[clean] removed archive {archive_path}")

    # Sanity print
    npy_files = sorted(p.name for p in root.glob("*.npy"))
    print(f"[ok] {len(npy_files)} .npy files in {root}")
    for n in npy_files:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
