"""
Online TTA evaluation loop — v1.5 with calibration + memory metrics.

New in v1.5:
  - ECE (Expected Calibration Error) tracking
  - Mean predictive entropy
  - Mean confidence
  - Peak GPU memory during adaptation (if CUDA)
  - Backward compatible: RunResult fields added; existing code reading
    .accuracy / .batch_accuracies / .diagnostics keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.methods.base import AdaptMethod


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: torch.Tensor,    # (N,)  max softmax probability
    correctness: torch.Tensor,    # (N,)  0/1 indicator of correct prediction
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (Naeini et al. 2015 / Guo et al. 2017).

      ECE = sum_{m} (|B_m| / N) * |acc(B_m) − conf(B_m)|

    where B_m is the m-th equal-width confidence bin.

    Returns 0.0 for empty input.
    """
    if confidences.numel() == 0:
        return 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)
    n = confidences.size(0)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = in_bin | (confidences == bin_boundaries[0])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            acc_in_bin = correctness[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece.item())


# ---------------------------------------------------------------------------
# RunResult (extended, backward compatible)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    accuracy: float
    num_correct: int
    num_total: int
    batch_accuracies: list[float]
    diagnostics: list[dict]

    # NEW in v1.5 — calibration & uncertainty
    ece: float = 0.0
    mean_confidence: float = 0.0
    mean_entropy: float = 0.0

    # NEW in v1.5 — efficiency
    peak_memory_mb: float = 0.0
    trainable_params: int = 0

    def __repr__(self) -> str:
        return (f"RunResult(acc={self.accuracy:.4f}, ece={self.ece:.4f}, "
                f"n={self.num_total})")


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _count_trainable(method: AdaptMethod) -> int:
    if not hasattr(method, "model"):
        return 0
    return sum(p.numel() for p in method.model.parameters() if p.requires_grad)


def evaluate_online(
    method: AdaptMethod,
    loader: DataLoader,
    device: torch.device,
    collect_diagnostics: bool = False,
    progress: bool = True,
    track_calibration: bool = True,
    track_memory: bool = True,
) -> RunResult:
    """
    Run online TTA on a stream and compute accuracy + calibration + memory.

    Args:
        method: AdaptMethod instance
        loader: yields (x, y) batches
        device: where the model is
        collect_diagnostics: if True and method supports it, gather grad norms
        progress: print progress bar
        track_calibration: if True, track per-sample confidence + correctness
                           and compute ECE / entropy / mean confidence
        track_memory: if True and CUDA available, track peak memory during
                      the loop (resets peak counter at start)
    """
    has_diag = (
        collect_diagnostics
        and hasattr(method, "adapt_with_diagnostics")
        and callable(getattr(method, "adapt_with_diagnostics"))
    )

    num_correct = 0
    num_total = 0
    batch_accs: list[float] = []
    diagnostics: list[dict] = []

    # Calibration tracking
    all_confidences: list[torch.Tensor] = []
    all_correctness: list[torch.Tensor] = []
    entropy_sum = 0.0
    confidence_sum = 0.0

    # Memory tracking
    is_cuda = (torch.cuda.is_available()
               and isinstance(device, torch.device) and device.type == "cuda")
    if track_memory and is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"adapt[{method.name}]", leave=False)
        except ImportError:
            iterator = loader

    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if has_diag:
            preds, diag = method.adapt_with_diagnostics(x)
            diagnostics.append(diag)
        else:
            preds = method.adapt(x)

        pred_labels = preds.argmax(dim=1)
        correct = (pred_labels == y)
        num_correct += correct.sum().item()
        bs = y.size(0)
        num_total += bs
        batch_accs.append(correct.sum().item() / bs)

        if track_calibration:
            with torch.no_grad():
                # `preds` are logits (detached). Softmax for confidence.
                probs = F.softmax(preds, dim=1)
                conf, _ = probs.max(dim=1)
                all_confidences.append(conf.detach().cpu())
                all_correctness.append(correct.detach().cpu())
                entropy_sum += -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).sum().item()
                confidence_sum += conf.sum().item()

    accuracy = num_correct / max(num_total, 1)

    # Calibration aggregate
    if track_calibration and num_total > 0:
        conf_all = torch.cat(all_confidences)
        corr_all = torch.cat(all_correctness)
        ece = compute_ece(conf_all, corr_all)
        mean_conf = confidence_sum / num_total
        mean_ent = entropy_sum / num_total
    else:
        ece = 0.0
        mean_conf = 0.0
        mean_ent = 0.0

    # Memory
    if track_memory and is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    else:
        peak_mem = 0.0

    trainable = _count_trainable(method)

    return RunResult(
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=num_total,
        batch_accuracies=batch_accs,
        diagnostics=diagnostics,
        ece=ece,
        mean_confidence=mean_conf,
        mean_entropy=mean_ent,
        peak_memory_mb=peak_mem,
        trainable_params=trainable,
    )
