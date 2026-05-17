"""
Base class for all test-time adaptation methods.

Each method implements `adapt(x)` which:
  1. Optionally updates model parameters (the adaptation step)
  2. Returns logits for the batch (the prediction)

Methods may be stateful (e.g., HEAT updates the model in place across
batches in online TTA). Subclasses are responsible for setting up which
parameters are trainable and which optimizer is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_LN_TYPES = (nn.LayerNorm,)


def has_batchnorm(model: nn.Module) -> bool:
    """True if the model contains any BatchNorm module."""
    return any(isinstance(m, _BN_TYPES) for m in model.modules())


def has_layernorm(model: nn.Module) -> bool:
    """True if the model contains any LayerNorm module."""
    return any(isinstance(m, _LN_TYPES) for m in model.modules())


def select_norm_affine_params(model: nn.Module):
    """
    Architecture-aware selection of affine norm parameters for TTA.

    Convention: freeze everything first, then re-enable grad on the affine
    params of:
      - all BatchNorm{1,2,3}d / SyncBatchNorm modules if the model has any BN
        (CNN case — existing behavior for resnet18 / wrn28_10);
      - all LayerNorm modules if the model has no BN but has LayerNorm
        (transformer case — vit_s).

    Returns the list of trainable parameters (the same objects that have
    .requires_grad set True on the model). Callers pass this list to their
    optimizer.

    Raises ValueError if neither BN nor LN are present (no sensible default).
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable: list[nn.Parameter] = []
    if has_batchnorm(model):
        for m in model.modules():
            if isinstance(m, _BN_TYPES):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    trainable.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    trainable.append(m.bias)
    elif has_layernorm(model):
        for m in model.modules():
            if isinstance(m, _LN_TYPES):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    trainable.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    trainable.append(m.bias)
    else:
        raise ValueError(
            "select_norm_affine_params: model contains neither BatchNorm "
            "nor LayerNorm modules — cannot choose affine TTA parameters."
        )
    return trainable


class AdaptMethod(ABC):
    """Abstract base for adaptation methods."""

    name: str = "base"

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process one test batch.

        Args:
            x: (B, C, H, W) input batch

        Returns:
            logits: (B, K) predictions for the batch
        """
        ...

    def reset(self):
        """Reset adaptation state (used between corruptions in non-continual settings)."""
        pass

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
