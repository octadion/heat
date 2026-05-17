"""
Tent: entropy minimization on BN affine parameters.
Wang, Shelhamer et al., ICLR 2021 — the canonical TTA baseline.

Key implementation notes:
  - Only BN affine (gamma, beta) are trainable; everything else is frozen.
  - BN running stats are tracked on the test batch (model.train()).
  - Loss = mean entropy over softmax output.
  - One SGD step per batch.

Optimizer convention:
  - Original Tent code (DequanWang/tent): Adam, lr=1e-3.
  - EATA / common follow-up convention: SGD, momentum=0.9, lr=0.005 for
    CIFAR-10.
  We default to Adam lr=1e-3 to match Tent's official code AND to align
  with the optimizer used by TEA (so the energy-based comparison stays
  apples-to-apples). For papers that follow EATA's SGD convention, set
  optimizer_name='sgd' and lr=0.005.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod, select_norm_affine_params


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean batch entropy of softmax(logits)."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()


class Tent(AdaptMethod):
    name = "tent"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        momentum: float = 0.9,
    ):
        super().__init__(model)

        # Architecture-aware norm-affine selection (BN for CNNs, LN for ViT).
        # Same iteration order as the prior inline code on CNNs, so the
        # selected parameter set is identical for resnet18 / wrn28_10.
        select_norm_affine_params(self.model)

        self.model.train()  # BN tracks batch stats; LN is mode-invariant

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable, lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"unknown optimizer_name: {optimizer_name}")

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        predictions = logits.detach()

        loss = _softmax_entropy(logits)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return predictions
