from .cifar10 import (
    get_cifar10_loaders,
    cifar10_train_transform,
    cifar10_test_transform,
    get_train_transform,
    get_eval_transform,
)
from .cifar10c import CIFAR10C, get_cifar10c_loader, CORRUPTIONS, SEVERITIES
from .cifar100 import get_cifar100_loaders
from .cifar100c import CIFAR100C, get_cifar100c_loader


def num_classes_for(dataset: str) -> int:
    """Map dataset name → number of classes."""
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    raise ValueError(f"unknown dataset: {dataset}")


def get_clean_loaders(dataset: str, data_root: str, batch_size: int,
                       num_workers: int, arch=None):
    """Dispatch to the right clean-CIFAR loader by dataset name."""
    if dataset == "cifar10":
        return get_cifar10_loaders(data_root, batch_size, num_workers, arch=arch)
    if dataset == "cifar100":
        return get_cifar100_loaders(data_root, batch_size, num_workers, arch=arch)
    raise ValueError(f"unknown dataset: {dataset}")


def get_corruption_loader(dataset: str, root: str, corruption: str,
                          severity: int = 5, batch_size: int = 64,
                          num_workers: int = 2, shuffle: bool = False,
                          arch=None):
    """Dispatch to cifar10c or cifar100c loader by dataset name."""
    if dataset == "cifar10":
        return get_cifar10c_loader(root, corruption, severity,
                                   batch_size, num_workers, shuffle, arch=arch)
    if dataset == "cifar100":
        return get_cifar100c_loader(root, corruption, severity,
                                    batch_size, num_workers, shuffle, arch=arch)
    raise ValueError(f"unknown dataset: {dataset}")


__all__ = [
    "get_cifar10_loaders",
    "cifar10_train_transform",
    "cifar10_test_transform",
    "get_train_transform",
    "get_eval_transform",
    "CIFAR10C",
    "get_cifar10c_loader",
    "CORRUPTIONS",
    "SEVERITIES",
    "get_cifar100_loaders",
    "CIFAR100C",
    "get_cifar100c_loader",
    "num_classes_for",
    "get_clean_loaders",
    "get_corruption_loader",
]
