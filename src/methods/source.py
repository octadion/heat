"""Source: no adaptation. Just inference."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AdaptMethod


class Source(AdaptMethod):
    name = "source"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
