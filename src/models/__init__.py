"""
Architecture registry.

Adds `vit_s` alongside resnet18_cifar / wrn_28_10_cifar.

build_arch(name, num_classes=10, **kwargs):
  - resnet18 / wrn28_10 ignore extra kwargs (back-compat: they don't accept any).
  - vit_s forwards `pretrained` and friends to the timm-based wrapper.
"""

from .resnet_cifar import resnet18_cifar
from .wrn_cifar import wrn_28_10_cifar


def _resnet18_factory(num_classes: int = 10, **_):
    return resnet18_cifar(num_classes=num_classes)


def _wrn_factory(num_classes: int = 10, **_):
    return wrn_28_10_cifar(num_classes=num_classes)


def _vit_s_factory(num_classes: int = 10, pretrained: bool = True, **_):
    from .vit_cifar import vit_s_cifar  # lazy: timm only needed for ViT
    return vit_s_cifar(num_classes=num_classes, pretrained=pretrained)


def _resnet50_domainnet_factory(num_classes: int = 126, **_):
    # Lazy import: torchvision ResNet-50 only needed for the DomainNet-126 path.
    from .resnet50_domainnet import resnet50_domainnet
    return resnet50_domainnet(num_classes=num_classes)


ARCH_FACTORY = {
    "resnet18": _resnet18_factory,
    "wrn28_10": _wrn_factory,
    "vit_s": _vit_s_factory,
    "resnet50": _resnet50_domainnet_factory,
}


def build_arch(name: str, num_classes: int = 10, **kwargs):
    if name not in ARCH_FACTORY:
        raise ValueError(f"unknown arch '{name}'; choices: {list(ARCH_FACTORY)}")
    return ARCH_FACTORY[name](num_classes=num_classes, **kwargs)


__all__ = ["resnet18_cifar", "wrn_28_10_cifar", "build_arch", "ARCH_FACTORY"]
