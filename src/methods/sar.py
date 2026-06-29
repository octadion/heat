"""SAR: Sharpness-aware and reliable entropy minimization for TTA."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod, select_norm_affine_params


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


class SAM(torch.optim.Optimizer):
    """Minimal SAM wrapper around a base optimizer."""

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
                del self.state[p]["e_w"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):  # pragma: no cover - SAM uses two explicit steps.
        raise NotImplementedError("SAM requires first_step and second_step")

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


class SAR(AdaptMethod):
    name = "sar"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2.5e-4,
        momentum: float = 0.9,
        rho: float = 0.05,
        e_margin: float | None = None,
        reset_ema_threshold: float | None = 0.2,
        ema_momentum: float = 0.9,
    ):
        super().__init__(model)
        select_norm_affine_params(self.model)
        self.model.train()

        if e_margin is None:
            e_margin = default_entropy_margin(_num_classes_from_model(self.model))
        self.e_margin = float(e_margin)
        self.reset_ema_threshold = reset_ema_threshold
        self.ema_momentum = float(ema_momentum)
        self.ema: float | None = None

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SAM(
            params,
            torch.optim.SGD,
            lr=lr,
            momentum=momentum,
            rho=rho,
        )
        self._source_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }
        self._optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict({
            k: v.detach().clone()
            for k, v in self._source_state.items()
        }, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))
        self.ema = None

    def _update_ema(self, value: float) -> None:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.ema_momentum * self.ema + (1.0 - self.ema_momentum) * value

    def _filtered_entropy(self, logits: torch.Tensor) -> torch.Tensor | None:
        entropies = softmax_entropy(logits)
        ids = torch.where(entropies < self.e_margin)[0]
        if ids.numel() == 0:
            return None
        return entropies[ids]

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        predictions = logits.detach()

        entropies = self._filtered_entropy(logits)
        if entropies is None:
            return predictions
        loss = entropies.mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        logits_second = self.model(x)
        entropies_second = self._filtered_entropy(logits_second)
        if entropies_second is None:
            self.optimizer.second_step(zero_grad=True)
            return predictions

        loss_second = entropies_second.mean()
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)
        self._update_ema(float(loss_second.detach().item()))

        if (
            self.reset_ema_threshold is not None
            and self.ema is not None
            and self.ema < self.reset_ema_threshold
        ):
            self.reset()

        return predictions
