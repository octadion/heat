"""
Architecture registry.

Adds wrn_28_10_cifar alongside resnet18_cifar.
"""

from .resnet_cifar import resnet18_cifar
from .wrn_cifar import wrn_28_10_cifar


ARCH_FACTORY = {
    "resnet18": resnet18_cifar,
    "wrn28_10": wrn_28_10_cifar,
}


def build_arch(name: str, num_classes: int = 10):
    if name not in ARCH_FACTORY:
        raise ValueError(f"unknown arch '{name}'; choices: {list(ARCH_FACTORY)}")
    return ARCH_FACTORY[name](num_classes=num_classes)


__all__ = ["resnet18_cifar", "wrn_28_10_cifar", "build_arch", "ARCH_FACTORY"]
