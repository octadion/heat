"""
Stage-1b additive HEAT/TFF variants. src/methods/heat.py is NOT modified.

HeatAnchor (E6, cross-mechanism):
    Identical free-energy update and diagnostics, but INSTEAD of the Bernoulli
    restore, after each SGD step every adapted parameter is pulled toward its
    source value:

        theta <- theta - eta * lambda * (theta - theta_source)

    Implemented by overriding HEAT's `_stochastic_restore` hook — the exact
    point both `adapt` and `adapt_with_diagnostics` invoke after
    `optimizer.step()` — so the update path, the optimizer, and the diagnostic
    capture (which runs BEFORE step+restore) are inherited byte-for-byte.
    Mutually exclusive with restore_prob > 0 (asserted here AND in the method
    factory).

HeatEntropyDrive (E7, drive-swap):
    Identical optimizer, parameter set, Bernoulli restore, and diagnostics,
    but the adaptation loss is the MEAN PREDICTION ENTROPY of the batch,
    computed from the model's true logits. NOTE: the final stage feature is
    the raw block output BEFORE the model's final BN+ReLU (see
    src/models/wrn_cifar.py forward), so entropy cannot be recomputed from
    stage features. We capture the true logits with a forward hook on
    `model.linear` and override `hierarchical_energy` to return the entropy
    loss — everything else (adapt, adapt_with_diagnostics, restore) is
    inherited unchanged. Consequence, documented for the analysis: the
    per-step `energy` diagnostic field records the ENTROPY loss value for this
    variant (grad_l2 / drift_l2 semantics are unchanged: gradient of the
    adaptation loss, drift from source).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .heat import HEAT


class HeatAnchor(HEAT):
    """L2 weight-anchor tether: theta <- theta - eta*lambda*(theta - theta_src)."""

    name = "heat"  # runs under the heat pipeline; the run tag carries anchor_lam

    def __init__(self, model, lr: float = 1e-4, anchor_lambda: float = 0.0,
                 restore_prob: float = 0.0, diagnostic_snapshot: bool = False,
                 **kwargs):
        if restore_prob != 0.0:
            raise ValueError(
                "HeatAnchor is mutually exclusive with restore_prob>0 "
                f"(got restore_prob={restore_prob}); the anchor replaces the "
                "Bernoulli restore.")
        if anchor_lambda < 0:
            raise ValueError(f"anchor_lambda must be >= 0, got {anchor_lambda}")
        coef = float(lr) * float(anchor_lambda)
        if not (0.0 <= coef < 1.0):
            raise ValueError(
                f"eta*lambda must be in [0, 1) for a contraction toward the "
                f"source (got eta*lambda={coef}).")
        # diagnostic_snapshot=True forces HEAT to store the source snapshot at
        # restore_prob==0; it doubles as the anchor target.
        super().__init__(model, lr=lr, restore_prob=0.0,
                         diagnostic_snapshot=True, **kwargs)
        self.anchor_lambda = float(anchor_lambda)
        self.anchor_coef = coef  # = eta * lambda, the per-step pull fraction

    @torch.no_grad()
    def _stochastic_restore(self):
        """The anchor pull, applied at the same post-step hook where HEAT
        applies the Bernoulli restore. All adapted params, deterministic."""
        if self.anchor_coef == 0.0:
            return
        snap = self._diagnostic_source_snapshot
        if not self._restore_diag_logged:
            trainable = [n for n, p in self.model.named_parameters()
                         if p.requires_grad]
            matched = sum(1 for n in trainable if snap.get(n) is not None)
            print(f"[HeatAnchor diag] anchor pull: eta*lambda={self.anchor_coef}, "
                  f"trainable={len(trainable)}, matched_snapshot={matched}")
            self._restore_diag_logged = True
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            src = snap.get(name)
            if src is None:
                continue
            # theta <- theta - eta*lambda*(theta - theta_source)
            p.data.add_(p.data - src, alpha=-self.anchor_coef)


class HeatEntropyDrive(HEAT):
    """HEAT with loss = mean prediction entropy of the batch (true logits)."""

    name = "heat"  # runs under the heat pipeline; the run tag carries drive_entropy

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._last_logits: torch.Tensor | None = None
        # Capture the model's true logits on every forward. The hook output is
        # graph-connected, so the entropy loss backpropagates normally.
        self._logits_hook = self.model.linear.register_forward_hook(
            lambda module, inputs, output: self._stash_logits(output))

    def _stash_logits(self, logits):
        # model.linear is ALSO invoked by HEAT's per-stage head calls (e.g.
        # the no_grad diagnostic block in adapt_with_diagnostics). Stash only
        # grad-connected outputs — i.e. the true forward of the current adapt
        # step — so a detached per-stage call can never clobber the loss input.
        if logits.requires_grad:
            self._last_logits = logits

    def hierarchical_energy(self, stage_features) -> torch.Tensor:
        """Overridden drive: mean prediction entropy of the batch. Called by
        the inherited adapt/adapt_with_diagnostics at the exact point HEAT
        computes its free-energy loss; `stage_features` is ignored (the
        per-stage free-energy diagnostic means are still computed separately
        by adapt_with_diagnostics, unchanged)."""
        if self._last_logits is None:
            raise RuntimeError(
                "HeatEntropyDrive: no logits captured — hierarchical_energy "
                "called outside an adapt() forward.")
        logits, self._last_logits = self._last_logits, None  # consume once
        logp = F.log_softmax(logits, dim=1)
        return -(logp.exp() * logp).sum(dim=1).mean()
