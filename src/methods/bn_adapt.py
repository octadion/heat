"""
BN-adapt: only updates BN running statistics on the test batch.
No gradient-based parameter update. Per Schneider et al. (2020) /
common TTA baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AdaptMethod


class BNAdapt(AdaptMethod):
    name = "bn_adapt"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # Freeze gradients: no parameter learning
        for p in self.model.parameters():
            p.requires_grad = False
        # train() so BN tracks current batch's stats; affine still frozen
        self.model.train()

    @torch.no_grad()
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
