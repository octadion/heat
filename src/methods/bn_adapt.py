"""
BN-adapt: only updates BN running statistics on the test batch.
No gradient-based parameter update. Per Schneider et al. (2020) /
common TTA baseline.

On architectures with no BatchNorm (e.g. ViT), there is nothing for BN-adapt
to do. We raise NoBatchNormError at construction so the runner can detect
the situation and skip the method cleanly rather than silently degrading
to no-op inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AdaptMethod, has_batchnorm


class NoBatchNormError(RuntimeError):
    """Raised when bn_adapt is applied to a model without BatchNorm."""


class BNAdapt(AdaptMethod):
    name = "bn_adapt"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if not has_batchnorm(model):
            raise NoBatchNormError(
                "bn_adapt requires BatchNorm layers; this model has none "
                "(e.g. vit_s uses LayerNorm). Skip bn_adapt for this arch."
            )
        # Freeze gradients: no parameter learning
        for p in self.model.parameters():
            p.requires_grad = False
        # train() so BN tracks current batch's stats; affine still frozen
        self.model.train()

    @torch.no_grad()
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
