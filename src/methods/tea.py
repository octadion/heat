"""
TEA: Test-time Energy Adaptation (Yuan et al., CVPR 2024).

Reimplementation. Reference: https://github.com/yuanyige/tea

What TEA does:
  - Reinterprets a trained classifier as an energy-based model:
        E(x) = -logsumexp_k f_theta(x)_k
  - Adapts via Contrastive Divergence:
        L_CD = E(x_test) - E(x_neg)
    where x_neg is sampled from the model's distribution via Stochastic
    Gradient Langevin Dynamics (SGLD).
  - Updates BN affine parameters only (paper's choice "for practicality").
  - Single gradient step per batch, plus extra entropy regularizer.

Default hyperparameters here MATCH the TEA paper Tab. 7 for CIFAR-10-C:
  - Optimizer: Adam (NOT SGD as in our first iteration — corrected)
  - lr = 0.001
  - SGLD steps = 20, sgld_lr = 0.1, noise_std = 0.01
  - 1 adaptation step per batch
  - Batch size 200 in their paper; we expose batch_size to the runner.

Why we need this baseline:
  HEAT's main differentiation argument is energy-based-but-better:
    (1) hierarchical, not output-only,
    (2) no SGLD / contrastive divergence machinery,
    (3) where-ness emerges (not engineered as BN-only).
  Without TEA in the comparison, the (1)-(3) story has no quantitative
  anchor.

Notes on faithfulness:
  - For paper-grade numbers, cross-check against the official repo (link
    above) by running both implementations on a single corruption and
    comparing within ~0.5 acc points.
  - The replay buffer here matches TEA's "persistent CD" pattern (mix of
    fresh random and buffered samples each step).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod


def _output_energy(logits: torch.Tensor) -> torch.Tensor:
    """E(x) = -logsumexp_k logits_k, returned per-sample (B,)."""
    return -torch.logsumexp(logits, dim=1)


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean batch entropy (used as additional regularizer per TEA paper)."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()


class TEA(AdaptMethod):
    """
    Test-time Energy Adaptation.

    Args:
        model: classifier
        lr: learning rate for the BN-affine param update (paper: 0.001)
        optimizer_name: 'adam' (paper default) or 'sgd'.
                Use 'adam' to match the paper.
        momentum: momentum if optimizer_name='sgd' (paper unused).
        sgld_steps: number of Langevin steps (paper: 20)
        sgld_lr: step size for SGLD (paper: 0.1)
        sgld_noise: standard deviation of noise added each step (paper: 0.01)
        use_buffer: if True, use a small persistent replay buffer for SGLD
                    init (matches TEA's persistent contrastive divergence).
        buffer_size: maximum number of saved SGLD samples.
        buffer_reinit_prob: probability of reinitializing a buffer slot
                            with a fresh random image (paper uses 0.05).
        entropy_coef: coefficient on the entropy regularizer term
                      (paper includes this term).
    """

    name = "tea"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        momentum: float = 0.9,
        sgld_steps: int = 20,
        sgld_lr: float = 0.1,
        sgld_noise: float = 0.01,
        use_buffer: bool = True,
        buffer_size: int = 1000,
        buffer_reinit_prob: float = 0.05,
        entropy_coef: float = 1.0,
    ):
        super().__init__(model)

        # Update only BN affine params (TEA paper's choice)
        for p in self.model.parameters():
            p.requires_grad = False
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

        self.model.train()  # so BN tracks batch stats
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable, lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"unknown optimizer_name: {optimizer_name}")

        # SGLD config
        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_noise = sgld_noise

        # Replay buffer
        self.use_buffer = use_buffer
        self.buffer_size = buffer_size
        self.buffer_reinit_prob = buffer_reinit_prob
        self._buffer: torch.Tensor | None = None  # lazily initialized

        # Reg
        self.entropy_coef = entropy_coef

    # ------------------------------------------------------------------
    # SGLD sampler
    # ------------------------------------------------------------------

    def _init_random(self, like: torch.Tensor) -> torch.Tensor:
        """Fresh uniform-noise samples in the same range as `like`."""
        return torch.rand_like(like) * 2 - 1  # ~[-1, 1] (post-normalize range)

    def _sample_sgld(self, x_real: torch.Tensor) -> torch.Tensor:
        """
        Run sgld_steps of Langevin dynamics on (B, C, H, W) tensors.

        SGLD updates a SAMPLE (not parameters), seeking low-energy regions
        of the model's input distribution.
        """
        bs = x_real.size(0)
        device = x_real.device

        # Initialize SGLD chain
        if self.use_buffer and self._buffer is not None:
            # Mix buffer + fresh.
            # Note: self._buffer lives on CPU (we store with .cpu() to save
            # GPU memory). Index it on CPU, then move the slice to `device`.
            n_fresh = max(1, int(bs * self.buffer_reinit_prob))
            n_buf = bs - n_fresh
            if n_buf > 0:
                buf_idx = torch.randint(0, self._buffer.size(0), (n_buf,))  # CPU
                x_neg = torch.cat([
                    self._buffer[buf_idx].to(device),
                    self._init_random(x_real[:n_fresh]),
                ], dim=0)
            else:
                # Batch too small for any buffer init — all fresh
                x_neg = self._init_random(x_real)
        else:
            x_neg = self._init_random(x_real)

        x_neg = x_neg.detach().clone().requires_grad_(True)

        # Do not update BN stats during SGLD inner loop
        was_training = self.model.training
        self.model.eval()

        for _ in range(self.sgld_steps):
            logits = self.model(x_neg)
            energy = _output_energy(logits).sum()
            grad = torch.autograd.grad(energy, x_neg)[0]
            x_neg = x_neg.detach() - 0.5 * self.sgld_lr * grad
            x_neg = x_neg + self.sgld_noise * torch.randn_like(x_neg)
            x_neg = x_neg.clamp(-3, 3)  # numeric safety
            x_neg.requires_grad_(True)

        if was_training:
            self.model.train()

        # Update buffer
        if self.use_buffer:
            x_save = x_neg.detach().cpu()
            if self._buffer is None:
                self._buffer = x_save
            else:
                self._buffer = torch.cat([self._buffer, x_save], dim=0)
            if self._buffer.size(0) > self.buffer_size:
                # Drop oldest
                self._buffer = self._buffer[-self.buffer_size:]

        return x_neg.detach()

    # ------------------------------------------------------------------
    # Adapt
    # ------------------------------------------------------------------

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        # Predictions for the batch (model state at arrival time)
        with torch.no_grad():
            predictions = self.model(x).detach()

        # SGLD negatives
        x_neg = self._sample_sgld(x)

        # Compute CD loss + entropy regularizer
        self.model.train()  # BN tracks batch stats during the parameter step
        logits_pos = self.model(x)
        logits_neg = self.model(x_neg)

        e_pos = _output_energy(logits_pos).mean()
        e_neg = _output_energy(logits_neg).mean()
        loss_cd = e_pos - e_neg
        loss_ent = _softmax_entropy(logits_pos)
        loss = loss_cd + self.entropy_coef * loss_ent

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return predictions