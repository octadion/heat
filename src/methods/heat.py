"""
HEAT v1.4 — adds OPTIONAL dyad fallback via stochastic weight restore.

Backward compatibility GUARANTEED:
  HEAT(model, lr=...)                                          → v1.2 monad (default)
  HEAT(model, lr=..., restore_prob=0.0)                        → v1.2 monad (explicit)
  HEAT(model, lr=..., restore_prob=0.005)                      → v1.4 dyad

When restore_prob == 0 (default), no source snapshot is stored, no per-step
work is added, and the resulting object is BIT-IDENTICAL in behavior to v1.2.
This means all P1 / P5 / P7 / P10 results you have already collected remain
valid — the dyad path is opt-in only.

The dyad addition (worldview-explicit second principle):

  Anchor principle: stay close to source manifold
      For each parameter θ_i, with probability p:
          θ_i ← θ_source_i

  This is the CoTTA-style restore (Wang et al. 2022, citation required).
  We use ONLY the restore mechanism — no EMA teacher, no augmentation
  averaging, no confidence threshold. Single hyperparameter p.

Framework framing matches the worldview lock from project setup:
  monad target  →  HEAT-monad (restore_prob = 0.0; default)
  dyad fallback →  HEAT-dyad  (restore_prob > 0; activated for continual)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptMethod


# ----------------------------------------------------------------------------
# Fixed orthogonal projection (unchanged)
# ----------------------------------------------------------------------------

def _orthogonal_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if in_dim <= out_dim:
        A = torch.randn(out_dim, in_dim, generator=g)
        Q, _ = torch.linalg.qr(A)
        return Q[:out_dim, :in_dim].clone()
    else:
        A = torch.randn(in_dim, in_dim, generator=g)
        Q, _ = torch.linalg.qr(A)
        return Q[:out_dim, :].clone()


def _stage_energy_per_sample(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return -tau * torch.logsumexp(logits / tau, dim=1)


# ----------------------------------------------------------------------------
# HEAT v1.4
# ----------------------------------------------------------------------------

class HEAT(AdaptMethod):
    """
    Hierarchical Energy Adaptive Test-time, v1.4.

    Args (new in v1.4):
        restore_prob: probability per parameter element of restoring to source
            value at each adaptation step. Default 0.0 = HEAT-monad (no
            restore, no source snapshot stored, no extra work). Set > 0 for
            HEAT-dyad (continual scenarios). Recommended values 0.001-0.01.

    All other args identical to v1.2.
    """

    name = "heat"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        momentum: float = 0.0,
        stages: list[int] | None = None,
        temperatures: Sequence[float] | None = None,
        aggregation: str = "sum",
        projection_seed: int = 42,
        identity_at_match: bool = True,
        update_all_params: bool = True,
        eval_mode: bool = False,
        bn_running_stats: str = "train",
        update_direction: str = "-grad",
        restore_prob: float = 0.0,
        diagnostic_snapshot: bool = False,
    ):
        super().__init__(model)

        if stages is None:
            stages = list(range(len(model.stage_channels)))
        self.stages = sorted(stages)

        if temperatures is None:
            temperatures = [1.0]
        self.temperatures = tuple(float(t) for t in temperatures)
        if any(t <= 0 for t in self.temperatures):
            raise ValueError(f"temperatures must be > 0, got {self.temperatures}")

        if aggregation not in ("sum", "self_gated", "self_gated_temperature"):
            raise ValueError(f"aggregation invalid: {aggregation}")
        self.aggregation = aggregation

        if update_direction not in ("-grad", "+grad", "random"):
            raise ValueError(f"update_direction invalid: {update_direction}")
        self.update_direction = update_direction
        self.eval_mode = eval_mode
        if bn_running_stats not in ("train", "frozen"):
            raise ValueError(
                f"bn_running_stats must be 'train' or 'frozen', got {bn_running_stats}"
            )
        self.bn_running_stats = bn_running_stats

        if restore_prob < 0 or restore_prob > 1:
            raise ValueError(f"restore_prob must be in [0,1], got {restore_prob}")
        self.restore_prob = float(restore_prob)

        self.diagnostic_snapshot = bool(diagnostic_snapshot)

        # Fixed projections
        head_in_dim = model.linear.in_features
        self.projections: dict[int, torch.Tensor] = {}
        for l in self.stages:
            in_ch = model.stage_channels[l]
            if identity_at_match and in_ch == head_in_dim:
                W = torch.eye(head_in_dim)
            else:
                W = _orthogonal_projection(
                    in_ch,
                    head_in_dim,
                    seed=projection_seed + l,
                )
            W.requires_grad_(False)
            self.projections[l] = W

        self.update_all_params = update_all_params
        self._configure_params()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)

        # Source snapshot for dyad restore.
        # Used for actual stochastic restore when restore_prob > 0.
        self._source_snapshot: dict[str, torch.Tensor] = {}
        if self.restore_prob > 0:
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._source_snapshot[name] = p.detach().clone()

        # Diagnostic-only snapshot.
        # Important for p=0 monad: no restore is used, but we still want drift_l2.
        self._diagnostic_source_snapshot: dict[str, torch.Tensor] = {}
        if self.diagnostic_snapshot and self.restore_prob == 0.0:
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._diagnostic_source_snapshot[name] = p.detach().clone()

        # One-time diagnostic flag for _stochastic_restore.
        self._restore_diag_logged = False

    # ------------------------------------------------------------------

    def _configure_params(self):
        if self.update_all_params:
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True

        if self.eval_mode:
            self.model.eval()
        else:
            self.model.train()
            if self.bn_running_stats == "frozen":
                for m in self.model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()

    def to(self, device):
        self.projections = {l: W.to(device) for l, W in self.projections.items()}

        if self._source_snapshot:
            self._source_snapshot = {
                k: v.to(device) for k, v in self._source_snapshot.items()
            }

        if self._diagnostic_source_snapshot:
            self._diagnostic_source_snapshot = {
                k: v.to(device) for k, v in self._diagnostic_source_snapshot.items()
            }

        return self

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def _per_stage_temp_energies(
        self, stage_features: list[torch.Tensor]
    ) -> torch.Tensor:
        head = self.model.linear
        rows = []
        for l in self.stages:
            z_l = stage_features[l]
            z_pooled = F.adaptive_avg_pool2d(z_l, 1).flatten(1)
            W = self.projections[l]
            z_proj = z_pooled @ W.t()
            logits_l = head(z_proj)
            row = torch.stack([
                _stage_energy_per_sample(logits_l, tau)
                for tau in self.temperatures
            ], dim=0)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def hierarchical_energy(self, stage_features: list[torch.Tensor]) -> torch.Tensor:
        E_all = self._per_stage_temp_energies(stage_features)
        if self.aggregation == "sum":
            return E_all.sum(dim=(0, 1)).mean()
        elif self.aggregation == "self_gated":
            weights = F.softmax(-E_all.detach(), dim=0)
            weighted = (weights * E_all).sum(dim=0)
            return weighted.sum(dim=0).mean()
        elif self.aggregation == "self_gated_temperature":
            weights = F.softmax(-E_all.detach(), dim=1)
            weighted = (weights * E_all).sum(dim=1)
            return weighted.sum(dim=0).mean()

    # ------------------------------------------------------------------
    # Dyad anchor: stochastic weight restore (no-op when restore_prob == 0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stochastic_restore(self):
        """
        For each trainable parameter element, with probability `restore_prob`
        replace its value with the source snapshot. Bernoulli-mask style.

        This is THE dyad anchor — exactly one operation. No threshold, no
        scheduler, no detector. Single hyperparameter p.

        Cited mechanism: CoTTA (Wang et al., 2022). HEAT-dyad isolates this
        restore as the SOLE additional operation; no EMA / aug / threshold.
        """
        if self.restore_prob == 0.0 or not self._source_snapshot:
            return

        # One-time diagnostic: report whether the snapshot keys match the
        # parameter names seen at restore time. If they do not, every
        # `_source_snapshot.get(name)` returns None and the restore is a
        # silent no-op. This print fires exactly once per HEAT instance and
        # does NOT alter parameter updates or the random stream.
        if not self._restore_diag_logged:
            trainable_names = [n for n, p in self.model.named_parameters()
                               if p.requires_grad]
            n_total = len(trainable_names)
            n_found = sum(1 for n in trainable_names
                          if self._source_snapshot.get(n) is not None)
            n_missing = n_total - n_found
            print(f"[HEAT diag] _stochastic_restore: snapshot_keys="
                  f"{len(self._source_snapshot)}, trainable_params="
                  f"{n_total}, matched={n_found}, missing={n_missing}, "
                  f"restore_prob={self.restore_prob}")
            self._restore_diag_logged = True

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            src = self._source_snapshot.get(name)
            if src is None:
                continue
            mask = torch.bernoulli(
                torch.full_like(p, self.restore_prob)
            )
            # Where mask=1, restore; where mask=0, keep current value
            p.data.copy_(mask * src + (1.0 - mask) * p.data)

    # ------------------------------------------------------------------
    # Direction op (unchanged from v1.2)
    # ------------------------------------------------------------------

    def _apply_direction_to_grads(self):
        if self.update_direction == "-grad":
            return
        elif self.update_direction == "+grad":
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.neg_()
        elif self.update_direction == "random":
            for p in self.model.parameters():
                if p.grad is not None:
                    g_norm = p.grad.detach().norm()
                    rand = torch.randn_like(p.grad)
                    rand_norm = rand.norm().clamp_min(1e-12)
                    p.grad.copy_(rand * (g_norm / rand_norm))

    # ------------------------------------------------------------------
    # Adapt — adds optional restore step at the END
    # ------------------------------------------------------------------

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        logits, stages = self.model(x, return_stages=True)
        predictions = logits.detach()

        P = self.hierarchical_energy(stages)
        self.optimizer.zero_grad(set_to_none=True)
        P.backward()
        self._apply_direction_to_grads()
        self.optimizer.step()

        # Dyad anchor — no-op when restore_prob == 0
        self._stochastic_restore()

        return predictions

    # ------------------------------------------------------------------
    # Diagnostics — FIXED in v1.4: explicit gradient capture before zero_grad
    #
    # The v1.2 bug: optimizer.zero_grad(set_to_none=True) replaces .grad with
    # None, then we tried to read it later. v1.4 captures grad norms BEFORE
    # zero_grad and BEFORE step (the correct moment — grads exist, params
    # haven't moved yet).
    # ------------------------------------------------------------------

    def adapt_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits, stages = self.model(x, return_stages=True)
        predictions = logits.detach()

        # Compute energies and per-stage diagnostic values BEFORE backward
        with torch.no_grad():
            E_all = self._per_stage_temp_energies(stages)   # (S, T, B)
            stage_energy_means = E_all.mean(dim=(1, 2)).tolist()

            if self.aggregation == "self_gated":
                weights = F.softmax(-E_all, dim=0)
                stage_weight_means = weights.mean(dim=(1, 2)).tolist()
            else:
                stage_weight_means = None

        # Forward + backward
        P = self.hierarchical_energy(stages)
        self.optimizer.zero_grad(set_to_none=True)
        P.backward()

        # ------------------------------------------------------------
        # Diagnostics BEFORE:
        #   1. _apply_direction_to_grads()
        #   2. optimizer.step()
        #   3. _stochastic_restore()
        #
        # So:
        #   drift_l2 = ||theta_t - theta_source||
        #   grad_l2  = ||grad P(theta_t)||
        # ------------------------------------------------------------

        param_grad_norms: dict[str, float] = {}
        grad_sq = 0.0
        drift_sq = 0.0

        # For dyad p>0, use _source_snapshot.
        # For monad p=0, use _diagnostic_source_snapshot if enabled.
        if self._source_snapshot:
            snapshot = self._source_snapshot
        else:
            snapshot = self._diagnostic_source_snapshot

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Gradient norm
            if p.grad is not None:
                g_norm = p.grad.detach().norm().item()
                param_grad_norms[name] = g_norm
                grad_sq += g_norm * g_norm
            else:
                param_grad_norms[name] = 0.0

            # Drift from source parameter snapshot
            src = snapshot.get(name)
            if src is not None:
                drift_sq += (p.detach() - src).pow(2).sum().item()

        grad_l2 = grad_sq ** 0.5
        drift_l2 = drift_sq ** 0.5

        stage_grad_norms = {}
        for name, gn in param_grad_norms.items():
            # ResNet uses stage1..stage4; WRN uses block1..block3.
            if name.startswith("stage"):
                key = name.split(".")[0]
            elif name.startswith("block"):
                key = name.split(".")[0]
            elif name.startswith("linear"):
                key = "linear"
            else:
                key = "stem_or_other"
            stage_grad_norms[key] = stage_grad_norms.get(key, 0.0) + gn * gn
        stage_grad_norms = {k: v ** 0.5 for k, v in stage_grad_norms.items()}

        diags = {
            "energy": float(P.detach().item()),
            "param_grad_norms": param_grad_norms,
            "grad_l2": grad_l2,
            "drift_l2": drift_l2,
            "total_grad_norm": grad_l2,
            "stages": list(self.stages),
            "temperatures": list(self.temperatures),
            "aggregation": self.aggregation,
            "stage_energy_means": stage_energy_means,
            "stage_weight_means": stage_weight_means,
            "restore_prob": self.restore_prob,
            "diagnostic_snapshot": self.diagnostic_snapshot,
            "snapshot_scope": "trainable_named_parameters_only",
            "bn_buffers_in_drift": False,
            "bn_running_stats": self.bn_running_stats,
        }
        diags["stage_grad_norms"] = stage_grad_norms
        # Now safe to apply direction op and update
        self._apply_direction_to_grads()
        self.optimizer.step()

        # Dyad anchor — no-op when restore_prob == 0
        self._stochastic_restore()

        return predictions, diags
