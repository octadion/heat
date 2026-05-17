"""
ReTTA (simplified): Rethinking Entropy in TTA via Energy Duality
================================================================

Park, Won, Ro, Kim. NeurIPS 2025.

Reimplementation note
---------------------
The full ReTTA combines (a) entropy minimization, (b) energy reduction via
sliced score matching (SSM) for a sampling-free Fisher-divergence objective,
and (c) cross-entropy on predicted-class targets for discriminability. SSM
requires Hessian-vector products and Pareto resolution between objectives.

For a baseline COMPARISON to HEAT, we provide a simplified ReTTA that
captures the central conceptual move: "entropy minimization complemented by
direct energy reduction." We skip SSM (which requires careful tuning to
make stable on top of an SGD-style TTA loop) and instead use direct energy
gradient — this is the simplest sampling-free realization of the energy-
reduction objective that ReTTA argues for. The entropy term is standard.

If the paper requires apples-to-apples vs full ReTTA, run the official
codebase. This simplified version captures the central inductive bias and
gives us a representative "energy + entropy multi-objective" baseline.

  L = L_entropy + λ · L_energy
    L_entropy = mean H(softmax(f(x)))
    L_energy  = mean(-logsumexp(f(x)))

Implementation: BN-affine only (matching paper); Adam lr=1e-3 (paper conv.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod, select_norm_affine_params


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy over the batch."""
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return (-p * log_p).sum(dim=1).mean()


def _energy(logits: torch.Tensor) -> torch.Tensor:
    """Mean -logsumexp energy over the batch."""
    return (-torch.logsumexp(logits, dim=1)).mean()


class ReTTA(AdaptMethod):
    """
    Simplified ReTTA: entropy + energy multi-objective TTA.

    Args:
        model: pretrained classifier.
        lr: Adam learning rate (default 1e-3, matches paper convention).
        lambda_energy: weight on energy term. Default 1.0 (equal weighting,
            matches paper's sketch; ablate {0.1, 0.5, 1.0, 5.0} in P5).
        adapt_bn_only: BN affine only (paper default).
    """

    name = "retta"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lambda_energy: float = 1.0,
        adapt_bn_only: bool = True,
    ):
        super().__init__(model)
        self.lambda_energy = lambda_energy
        self.adapt_bn_only = adapt_bn_only

        if self.adapt_bn_only:
            # Arch-aware: BN affine on CNNs, LayerNorm affine on ViT.
            # Flag name kept (`adapt_bn_only`) for backward-compatible API.
            select_norm_affine_params(self.model)
        else:
            for p in self.model.parameters():
                p.requires_grad = True

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

        self.model.train()

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        predictions = logits.detach()

        L_ent = _entropy(logits)
        L_eng = _energy(logits)
        loss = L_ent + self.lambda_energy * L_eng

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return predictions
