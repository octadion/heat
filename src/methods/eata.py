"""EATA: Efficient Test-Time Model Adaptation without Forgetting.

This implementation follows the official EATA structure: entropy adaptation is
applied only to reliable/non-redundant samples, and a diagonal Fisher penalty is
available for anti-forgetting. The factory requires Fisher samples by default so
`eata` is not silently reduced to Tent.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod, select_norm_affine_params


FisherDict = dict[str, tuple[torch.Tensor, torch.Tensor]]


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def default_entropy_margin(num_classes: int) -> float:
    return 0.4 * math.log(float(num_classes))


def _num_classes_from_model(model: nn.Module) -> int:
    for attr in ("linear", "fc", "classifier", "head"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    raise ValueError(
        "could not infer num_classes from model; pass an explicit entropy margin"
    )


@torch.no_grad()
def update_model_probs(
    current_model_probs: torch.Tensor | None,
    new_probs: torch.Tensor,
) -> torch.Tensor | None:
    if new_probs.numel() == 0:
        return current_model_probs
    batch_mean = new_probs.detach().mean(dim=0)
    if current_model_probs is None:
        return batch_mean
    return 0.9 * current_model_probs + 0.1 * batch_mean


def compute_fisher_diagonal(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_samples: int,
) -> FisherDict:
    """Estimate diagonal Fisher on source-like data using model predictions."""
    if max_samples <= 0:
        raise ValueError(f"max_samples must be > 0, got {max_samples}")

    params = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    fishers = {
        name: torch.zeros_like(param, device=device)
        for name, param in params.items()
    }

    was_training = model.training
    model.eval()
    seen = 0

    for x, _ in loader:
        if seen >= max_samples:
            break
        x = x.to(device, non_blocking=True)
        remaining = max_samples - seen
        if x.size(0) > remaining:
            x = x[:remaining]

        logits = model(x)
        pseudo = logits.detach().argmax(dim=1)
        loss = F.cross_entropy(logits, pseudo)

        model.zero_grad(set_to_none=True)
        loss.backward()

        batch_size = x.size(0)
        for name, param in params.items():
            if param.grad is not None:
                fishers[name] += param.grad.detach().pow(2) * batch_size
        seen += batch_size

    if seen == 0:
        raise ValueError("Fisher estimation saw zero samples")

    out = {
        name: (fisher / seen, params[name].detach().clone())
        for name, fisher in fishers.items()
    }
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return out


class EATA(AdaptMethod):
    name = "eata"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        optimizer_name: str = "sgd",
        momentum: float = 0.9,
        e_margin: float | None = None,
        d_margin: float = 0.05,
        fisher_alpha: float = 2000.0,
        fishers: FisherDict | None = None,
    ):
        super().__init__(model)
        select_norm_affine_params(self.model)
        self.model.train()

        if e_margin is None:
            e_margin = default_entropy_margin(_num_classes_from_model(self.model))
        self.e_margin = float(e_margin)
        self.d_margin = float(d_margin)
        self.fisher_alpha = float(fisher_alpha)
        self.fishers = fishers or {}
        self.current_model_probs: torch.Tensor | None = None

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)
        elif optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable, lr=lr)
        else:
            raise ValueError(f"unknown optimizer_name: {optimizer_name}")

    def _fisher_loss(self) -> torch.Tensor:
        if self.fisher_alpha <= 0.0 or not self.fishers:
            return torch.zeros((), device=next(self.model.parameters()).device)
        loss = torch.zeros((), device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            fisher = self.fishers.get(name)
            if fisher is None:
                continue
            precision, source_param = fisher
            loss = loss + (precision.to(param.device) *
                           (param - source_param.to(param.device)).pow(2)).sum()
        return self.fisher_alpha * loss

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        predictions = logits.detach()

        entropies = softmax_entropy(logits)
        reliable_ids = torch.where(entropies < self.e_margin)[0]
        if reliable_ids.numel() == 0:
            return predictions

        reliable_logits = logits[reliable_ids]
        reliable_entropies = entropies[reliable_ids]

        if self.current_model_probs is not None:
            probs = reliable_logits.softmax(dim=1)
            similarities = F.cosine_similarity(
                self.current_model_probs.to(probs.device).unsqueeze(0),
                probs,
                dim=1,
            )
            non_redundant_ids = torch.where(torch.abs(similarities) < self.d_margin)[0]
            if non_redundant_ids.numel() == 0:
                self.current_model_probs = update_model_probs(
                    self.current_model_probs,
                    probs.detach(),
                )
                return predictions
            selected_logits = reliable_logits[non_redundant_ids]
            selected_entropies = reliable_entropies[non_redundant_ids]
        else:
            selected_logits = reliable_logits
            selected_entropies = reliable_entropies

        self.current_model_probs = update_model_probs(
            self.current_model_probs,
            selected_logits.detach().softmax(dim=1),
        )

        coeff = 1.0 / torch.exp(selected_entropies.detach() - self.e_margin)
        loss = (selected_entropies * coeff).mean() + self._fisher_loss()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return predictions

