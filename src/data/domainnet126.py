"""
DomainNet-126 loader (126-class, 4-domain natural-shift TTA benchmark).

NEW FILE (DomainNet-126 smoke integration). Mirrors how cifar10c.py is shaped so
the existing run_tier2 --protocol p9 machinery can consume a domain stream
unchanged: a "corruption" name here = a target DOMAIN (real/clipart/painting/
sketch), and a "block" of diagnostics = one domain pass.

Expected layout (matches the AdaContrast image lists):
    <root>/
        real_list.txt          # lines: "<relative/path.jpg> <label>"
        clipart_list.txt
        painting_list.txt
        sketch_list.txt
        real/...                # images (paths in the list are relative to <root>)
        clipart/...
        ...

Standard ImageNet eval transform (resize 256 -> center-crop 224 -> normalize).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DOMAINS = ["real", "clipart", "painting", "sketch"]
NUM_CLASSES = 126

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def domainnet_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resolve_list_file(root: Path, domain: str) -> Path:
    """Find the image-list .txt for a domain (a few common names/locations)."""
    candidates = [
        root / f"{domain}_list.txt",
        root / "domainnet-126" / f"{domain}_list.txt",
        root / "lists" / f"{domain}_list.txt",
        root / f"{domain}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"DomainNet-126 list for '{domain}' not found. Looked for: "
        f"{[str(c) for c in candidates]}. Place <domain>_list.txt under {root}.")


class DomainNet126(Dataset):
    """One DomainNet-126 domain split from an AdaContrast image list."""

    def __init__(self, root: str, domain: str,
                 transform: Optional[transforms.Compose] = None,
                 arch: Optional[str] = None):
        super().__init__()
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain '{domain}', choices: {DOMAINS}")
        self.root = Path(root)
        self.domain = domain
        list_file = _resolve_list_file(self.root, domain)

        self.samples: list[tuple[str, int]] = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                rel, label = parts[0], int(parts[1])
                self.samples.append((rel, label))
        if not self.samples:
            raise RuntimeError(f"no samples parsed from {list_file}")

        # Image paths in the list are relative to <root> (they begin with the
        # domain folder, e.g. "clipart/aircraft/..."). Fall back to <root>/<rel>.
        self.transform = transform if transform is not None else domainnet_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, rel: str) -> Path:
        p = self.root / rel
        if p.exists():
            return p
        # Some lists omit the leading domain dir; try <root>/<domain>/<rel>.
        alt = self.root / self.domain / rel
        return alt

    def __getitem__(self, idx: int):
        rel, label = self.samples[idx]
        img = Image.open(self._image_path(rel)).convert("RGB")
        return self.transform(img), label


def get_domainnet126_loader(
    root: str,
    domain: str,
    severity: int = 5,          # accepted & ignored (API parity with cifar10c)
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = False,
    arch: Optional[str] = None,
) -> DataLoader:
    """Convenience wrapper. `severity` is ignored (DomainNet has no severities);
    it is accepted so the get_corruption_loader dispatch can call this uniformly."""
    ds = DomainNet126(root, domain, arch=arch)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
