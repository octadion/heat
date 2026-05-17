"""
EPOTTA: Energy-based Preference Optimization for Test-time Adaptation
======================================================================

Han, Yang, Kim. NeurIPS 2025. arXiv:2505.19607.

Reimplementation for HEAT comparison. Faithful to the paper's formulation;
not the official codebase.

Core idea
---------
Parameterize target distribution as
    p_θ(x) = (1/Z) · q_φ(x) · exp(-Ẽ_θ(x) / β)
where φ is a frozen copy of the pretrained source model and θ is the adapt
copy initialized from φ. The residual energy Ẽ captures the distribution
shift. Rearranging gives
    Ẽ_θ(x) = -β · log[p_θ(x) / q_φ(x)] + log Z.

Mathematically this matches DPO's reparameterized reward, so the marginal-
likelihood-maximization objective collapses to a DPO-style preference loss
WITHOUT explicit residual training and WITHOUT SGLD:

    L = -E_{(x_t, x_s)} [ log σ(β · ( -E_θ(x_t) + E_θ(x_s)
                                       + E_φ(x_t) - E_φ(x_s) )) ]

where E_θ(x) = -logsumexp(f_θ(x)) is the standard EBM energy of the model
output, x_t is a target test sample, x_s is a source sample drawn from a
small replay buffer.

Implementation notes
--------------------
- BN-affine-only updates on θ (matching paper's reported setup).
- φ is fully frozen, eval mode (its BN running stats are also frozen).
- Source replay buffer: small i.i.d. set from CIFAR-10 train, default 500
  images. Paper shows EPOTTA is robust to buffer size 1%-5%.
- Adam optimizer, lr=1e-3 (paper default for CIFAR-10).
- β temperature default 1.0; paper ablates {0.1, 0.5, 1.0, 5.0}, all stable.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod, select_norm_affine_params


def _build_source_buffer(
    source_loader: Iterable,
    buffer_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-load up to `buffer_size` images from the source loader."""
    chunks = []
    n = 0
    for x, _ in source_loader:
        chunks.append(x)
        n += x.size(0)
        if n >= buffer_size:
            break
    buf = torch.cat(chunks, dim=0)[:buffer_size]
    return buf.to(device)


class EPOTTA(AdaptMethod):
    """
    Energy-based Preference Optimization for TTA.

    Args:
        model: pretrained classifier; will be DEEP-COPIED into θ (adapt) and
               φ (frozen). The user's original model is left untouched.
        source_loader: an iterable yielding (x, y) batches from the source
               training set (CIFAR-10 train). Used once at init to populate
               the replay buffer.
        lr: Adam learning rate for θ's BN-affine params.
        beta: DPO temperature. Default 1.0.
        buffer_size: number of source images to retain. Default 500 (~1% of
               CIFAR-10 train).
        device: device to keep the source buffer and φ on.
        adapt_bn_only: if True (default, matches paper), update only BN affine
               params of θ. Set False to update all params (for matched-scope
               comparison vs HEAT-full).
    """

    name = "epotta"

    def __init__(
        self,
        model: nn.Module,
        source_loader: Iterable,
        lr: float = 1e-3,
        beta: float = 1.0,
        buffer_size: int = 500,
        device: torch.device | str = "cuda",
        adapt_bn_only: bool = True,
    ):
        # θ: model that will be adapted
        super().__init__(model)
        self.beta = beta
        self.adapt_bn_only = adapt_bn_only

        # φ: frozen reference copy
        self.phi = copy.deepcopy(model).to(device)
        for p in self.phi.parameters():
            p.requires_grad = False
        self.phi.eval()

        # Source replay buffer
        self.source_buffer = _build_source_buffer(source_loader, buffer_size, device)

        # Configure trainable params on θ
        self._configure_params()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

        # θ in train() so its BN stats track test batches
        # (matches Tent/TEA convention; consistent with paper)
        self.model.train()

    # ------------------------------------------------------------------

    def _configure_params(self):
        if self.adapt_bn_only:
            # Arch-aware: BN affine on CNNs, LayerNorm affine on ViT.
            # Flag name kept (`adapt_bn_only`) for backward-compatible API.
            select_norm_affine_params(self.model)
        else:
            for p in self.model.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        """E(x) = -logsumexp f(x). Returns per-sample tensor (B,)."""
        return -torch.logsumexp(logits, dim=1)

    def _sample_source_batch(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.source_buffer.size(0),
                            (batch_size,), device=self.source_buffer.device)
        return self.source_buffer[idx]

    # ------------------------------------------------------------------

    def adapt(self, x_t: torch.Tensor) -> torch.Tensor:
        # 1. predictions on x_t come from θ at the current state (pre-update),
        #    matching the standard online-TTA convention.
        logits_t_theta = self.model(x_t)
        predictions = logits_t_theta.detach()

        # 2. sample a same-size source batch from the replay buffer
        x_s = self._sample_source_batch(x_t.size(0))

        # 3. forward through θ (graph) and φ (no graph)
        # Note: a second forward of x_t through θ — we COULD reuse the one
        # above, but doing it explicitly here keeps the graph clean and
        # avoids subtle issues if BN stats change mid-step.
        logits_t_theta = self.model(x_t)
        logits_s_theta = self.model(x_s)
        with torch.no_grad():
            logits_t_phi = self.phi(x_t)
            logits_s_phi = self.phi(x_s)

        E_t_theta = self._energy(logits_t_theta)   # (B,)
        E_s_theta = self._energy(logits_s_theta)
        E_t_phi   = self._energy(logits_t_phi)
        E_s_phi   = self._energy(logits_s_phi)

        # 4. preference: target is "preferred" over source under adapted vs
        #    frozen models. Argument of σ is positive when adaptation
        #    increases relative likelihood of x_t over x_s.
        diff = self.beta * (-E_t_theta + E_s_theta + E_t_phi - E_s_phi)
        loss = -F.logsigmoid(diff).mean()

        # 5. step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return predictions
