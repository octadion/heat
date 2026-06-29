"""Periodic reset wrapper for online TTA methods.

This implements the non-oracle RDumb-style reset baseline used in the review
response commands: after every fixed number of adaptation batches, restore the
wrapped method to its source-time state. The wrapped method is declared
explicitly by the factory, and defaults to Tent.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn

from .base import AdaptMethod


ResetScope = Literal["full_model", "trainable_params"]


def _bn_buffer_names(model: nn.Module) -> set[str]:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, bn_types):
            continue
        prefix = f"{module_name}." if module_name else ""
        for buffer_name, _ in module.named_buffers(recurse=False):
            names.add(prefix + buffer_name)
    return names


class PeriodicReset(AdaptMethod):
    """Wrap another adaptation method and periodically reset its state."""

    name = "rdumb"

    def __init__(
        self,
        inner: AdaptMethod,
        reset_interval: int = 157,
        reset_scope: ResetScope = "full_model",
        reset_optimizer_state: bool = True,
        reset_bn_running_stats: bool = True,
        label: str = "rdumb",
    ):
        if reset_interval <= 0:
            raise ValueError(f"reset_interval must be > 0, got {reset_interval}")
        if reset_scope not in ("full_model", "trainable_params"):
            raise ValueError(f"unknown reset_scope: {reset_scope}")

        super().__init__(inner.model)
        self.inner = inner
        self.name = label
        self.reset_interval = int(reset_interval)
        self.reset_scope = reset_scope
        self.reset_optimizer_state = bool(reset_optimizer_state)
        self.reset_bn_running_stats = bool(reset_bn_running_stats)
        self.steps_seen = 0
        self.num_resets = 0

        self._source_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }
        self._bn_buffer_names = _bn_buffer_names(self.model)
        self._trainable_param_names = {
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        optimizer = getattr(self.inner, "optimizer", None)
        self._optimizer_state = (
            copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        )

    def _reset_model_state(self) -> None:
        if self.reset_scope == "full_model":
            target = {
                k: v.detach().clone()
                for k, v in self._source_state.items()
            }
            if not self.reset_bn_running_stats:
                current = self.model.state_dict()
                for name in self._bn_buffer_names:
                    if name in target and name in current:
                        target[name] = current[name].detach().clone()
            self.model.load_state_dict(target, strict=True)
            return

        current = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if name in self._trainable_param_names:
                param.data.copy_(self._source_state[name].to(param.device))
        if self.reset_bn_running_stats:
            for name in self._bn_buffer_names:
                if name in current and name in self._source_state:
                    current[name].copy_(self._source_state[name].to(current[name].device))

    def reset(self) -> None:
        self._reset_model_state()

        optimizer = getattr(self.inner, "optimizer", None)
        if (
            self.reset_optimizer_state
            and optimizer is not None
            and self._optimizer_state is not None
        ):
            optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))

        if hasattr(self.inner, "current_model_probs"):
            self.inner.current_model_probs = None
        if hasattr(self.inner, "ema"):
            self.inner.ema = None

        self.num_resets += 1

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        if self.steps_seen > 0 and self.steps_seen % self.reset_interval == 0:
            self.reset()
        self.steps_seen += 1
        return self.inner.adapt(x)

