"""
Shared helpers for the drift-budget-law (p*) experiment.

Used by:
  - scripts/run_pstar_sweep.py   (orchestration)
  - scripts/analyze_pstar_law.py (analysis / plot / verdict)

The scientific question (see the experiment system prompt for full detail):

    p* ~= (eta * ||g_bar||) / R

with R a per-architecture constant. Hold architecture fixed, vary CIFAR-10-C
severity (the shift-magnitude axis). If the law holds, p* vs (eta * ||g_bar||)
is a straight line through the origin with slope 1/R, one line per arch.

This module owns the pieces that BOTH the sweep and the analysis must agree on
exactly, so the two scripts can never drift apart:

  * run-output filename convention (must match scripts/run_tier2.py)
  * the collapse criterion (section 6.2 of the spec)
  * ||g_bar|| computation (section 6.1: mean grad_l2 over steps 50-150 within
    each block, averaged across the 15 corruption blocks, diverged blocks
    excluded)
  * p* bracket selection from a set of {p: collapsed?} observations (section 6.3)

We DO NOT modify anything under src/. We reuse the existing, validated helpers
in scripts/analysis_common.py for block grouping / the 50-150 stationary window
/ R^2, so the ||g_bar|| window matches the rest of the codebase byte-for-byte.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Reuse the project's existing analysis helpers (read-only). These define the
# exact stationary window (50 <= local_step <= 150) and block grouping used by
# every other diagnostic analysis in this repo, so ||g_bar|| stays consistent.
from scripts.analysis_common import (  # type: ignore
    group_rows_by_block,
    stationary_rows,
    row_metric,
    safe_float,
    compute_r2,
    load_json,
)

# Learning rate is fixed for this experiment (the x-axis is eta * ||g_bar||).
ETA = 1e-3

# Reference tether strength to measure ||g_bar|| from (section 6.1). Start here;
# escalate up the fallback ladder if the reference run itself collapses at high
# severity. The escalation ladder is the coarse grid plus a couple of stronger
# rungs, so a stable reference can almost always be found without extra runs.
PREF_DEFAULT: dict[str, float] = {"resnet18": 0.005, "wrn28_10": 0.010}
PREF_FALLBACKS: dict[str, list[float]] = {
    "resnet18": [0.005, 0.010, 0.020, 0.040],
    "wrn28_10": [0.010, 0.020, 0.040, 0.080],
}

# Coarse p-grid default (section 6.3). Always includes 0.0 so the p*~=0 case
# (small shift => small ||g_bar|| => small p*) is a measurable point.
DEFAULT_P_GRID: list[float] = [0.0, 0.005, 0.010, 0.020]

# Upward grid extension used only when even the strongest p_ref-ladder rung
# collapses (so the whole tested range is below p*). We try successively
# stronger tethers until one is stable, capped at ~5 extra runs, so the high-
# severity (high-||g_bar||) cells -- which carry the slope -- still get a real
# [collapse, stable] bracket instead of resolving only to the coarse gap.
EXTEND_P_GRID: list[float] = [0.03, 0.04, 0.06, 0.08, 0.12]

# Per-architecture default severity grids. wrn28_10 is the decisive (sharp-
# boundary) line and needs full density; resnet18's boundary is softer/cheaper
# so the reduced set suffices. The sweep CLI / notebook can still override.
DEFAULT_SEVERITIES_BY_ARCH: dict[str, list[int]] = {
    "resnet18": [1, 3, 5],
    "wrn28_10": [1, 2, 3, 4, 5],
}
# Union across archs -- the analysis default. analyze_pstar_law.py recovers only
# the cells that actually have runs on disk, so passing the union is safe.
DEFAULT_SEVERITIES: list[int] = sorted(
    {s for sevs in DEFAULT_SEVERITIES_BY_ARCH.values() for s in sevs}
)
DEFAULT_ARCHS: list[str] = ["resnet18", "wrn28_10"]

CHANCE_ACC = 0.12          # section 6.2 hard-collapse accuracy floor
DRIFT_BLOWUP_FACTOR = 5.0  # section 6.2 soft-collapse drift multiple


# ----------------------------------------------------------------------------
# Run identity: filenames must match scripts/run_tier2.py exactly.
#   out = f"{protocol}_{arch}{dataset_tag}{tag}_seed{seed}_sev{severity}.json"
# with dataset_tag = "" for cifar10 and tag = f"_{variant_tag}".
# ----------------------------------------------------------------------------

def format_p(p: float) -> str:
    """Canonical, stable string for a restore-prob value (used in tags/paths).

    Rounded to 6 dp so bisection midpoints like 0.0075 / 0.00375 stay clean and
    identical between the sweep (which writes them) and the analysis (which
    reads them back).
    """
    return f"{round(float(p), 6):g}"


def variant_tag(p: Optional[float] = None, source: bool = False,
                eta: Optional[float] = None) -> str:
    """Tag for a run. `eta` (the HEAT learning rate) is encoded so runs at
    different eta -- the swept lever in the eta-sweep -- never collide.

      heat run: pstar_lr<eta>_p<p>      source: pstar_lr<eta>_source
    """
    eta_part = f"lr{format_p(eta)}_" if eta is not None else ""
    if source:
        return f"pstar_{eta_part}source"
    if p is None:
        raise ValueError("variant_tag requires p when source=False")
    return f"pstar_{eta_part}p{format_p(p)}"


def run_output_path(
    results_dir: Path,
    arch: str,
    severity: int,
    seed: int,
    p: Optional[float] = None,
    source: bool = False,
    eta: Optional[float] = None,
) -> Path:
    tag = variant_tag(p=p, source=source, eta=eta)
    # dataset is always cifar10 here, so dataset_tag is empty (matches run_tier2).
    fname = f"p9_{arch}_{tag}_seed{seed}_sev{severity}.json"
    return Path(results_dir) / fname


def run_error_path(
    results_dir: Path,
    arch: str,
    severity: int,
    seed: int,
    p: Optional[float] = None,
    source: bool = False,
    eta: Optional[float] = None,
) -> Path:
    out = run_output_path(results_dir, arch, severity, seed, p=p, source=source, eta=eta)
    return out.with_suffix(".run_error.txt")


def discover_p_values(results_dir, arch: str, severity: int, seed: int,
                      eta: float) -> list[float]:
    """Every restore-prob whose run JSON exists on disk for (arch, severity, eta).

    Filtered to one eta by its formatted tag, so the sweep folds in the
    eta-scaled p_ref-ladder rungs but never mixes p-observations across etas
    (choose_pstar/classify_run operate at a FIXED (arch, sev, eta)).
    """
    prefix = f"p9_{arch}_pstar_lr{format_p(eta)}_p"
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(re.escape(prefix) + r"(.+?)" + re.escape(suffix) + r"$")
    ps: list[float] = []
    for path in Path(results_dir).glob(f"{prefix}*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                ps.append(float(m.group(1)))
            except ValueError:
                continue
    return sorted(set(ps))


def discover_etas(results_dir, arch: str, severity: int, seed: int) -> list[float]:
    """Every eta (learning rate) that has at least one heat run on disk for
    this (arch, severity). Lets the analysis enumerate eta-cells automatically.
    """
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(
        r"^p9_" + re.escape(arch) + r"_pstar_lr(.+?)_p.+?"
        + re.escape(suffix) + r"$")
    etas: set[float] = set()
    for path in Path(results_dir).glob(f"p9_{arch}_pstar_lr*_p*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                etas.add(float(m.group(1)))
            except ValueError:
                continue
    return sorted(etas)


# ----------------------------------------------------------------------------
# eta-scaled p-search grids (CHANGE C). The whole point of the eta-sweep is that
# p* moves ~linearly with eta over a ~20x range, so a fixed p-grid cannot
# bracket it. We scale the coarse grid, the p_ref ladder, and the upward-
# extension grid by (eta / ETA) so each eta-cell has an appropriately-placed
# search range (e.g. at eta=4e-3, p* ~= 0.038 and p_ref must exceed it).
# ----------------------------------------------------------------------------

def _scale(values: Sequence[float], eta: float) -> list[float]:
    factor = float(eta) / ETA
    return [round(v * factor, 8) for v in values]


def scaled_p_grid(eta: float, base: Optional[Sequence[float]] = None) -> list[float]:
    return _scale(base if base is not None else DEFAULT_P_GRID, eta)


def scaled_pref_ladder(arch: str, eta: float) -> list[float]:
    base = PREF_FALLBACKS.get(arch, [PREF_DEFAULT.get(arch, 0.01)])
    return _scale(base, eta)


def scaled_extend_grid(eta: float) -> list[float]:
    return _scale(EXTEND_P_GRID, eta)


def build_run_command(
    *,
    run_tier2: Path,
    arch: str,
    checkpoint: str,
    severity: int,
    seed: int,
    results_dir: Path,
    c10c_root: str,
    heat_lr: float = ETA,
    batch_size: int = 64,
    num_workers: int = 2,
    p: Optional[float] = None,
    source: bool = False,
    python: Optional[str] = None,
) -> list[str]:
    """Construct the run_tier2.py command line for one continual run."""
    python = python or sys.executable
    cmd = [
        python, str(run_tier2),
        "--protocol", "p9",
        "--arch", arch,
        "--dataset", "cifar10",
        "--checkpoint", str(checkpoint),
        "--c10c-root", str(c10c_root),
        "--severity", str(severity),
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--out-dir", str(results_dir),
    ]
    if source:
        cmd += ["--methods", "source",
                "--variant-tag", variant_tag(source=True, eta=heat_lr)]
    else:
        cmd += [
            "--methods", "heat",
            "--heat-lr", repr(float(heat_lr)),
            "--heat-restore-prob", format_p(p if p is not None else 0.0),
            "--heat-diagnostic-snapshot",  # required so drift_l2 exists even at p=0
            "--variant-tag", variant_tag(p=p, eta=heat_lr),
        ]
    return cmd


def execute_run(cmd: list[str], log_prefix: str = "") -> tuple[bool, str]:
    """Run a single continual run as a subprocess, streaming its output.

    Returns (ok, captured_tail). Output is streamed to the parent's stdout so
    Colab shows live progress; we also keep a tail for the error sidecar.
    Never raises on a non-zero exit -- the caller records run_error and moves on.
    """
    print(f"{log_prefix}$ {' '.join(cmd)}", flush=True)
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)
        proc.wait()
        ok = proc.returncode == 0
        if not ok:
            tail.append(f"[exit code {proc.returncode}]\n")
        return ok, "".join(tail)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"[execute_run exception] {type(exc).__name__}: {exc}\n"
        tail.append(msg)
        return False, "".join(tail)


# ----------------------------------------------------------------------------
# Reading runs.
# ----------------------------------------------------------------------------

def heat_summary(data: dict[str, Any]) -> dict[str, Any]:
    return (
        data.get("results", {})
        .get("summary", {})
        .get("heat", {})
    )


def source_mean_acc(data: dict[str, Any]) -> Optional[float]:
    summ = (
        data.get("results", {})
        .get("summary", {})
        .get("source", {})
    )
    return safe_float(summ.get("mean_accuracy"))


def stream_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(heat_summary(data).get("stream_diagnostics", []) or [])


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# Raw numeric keys that, if non-finite, signal a hard collapse / divergence.
_DIVERGENCE_KEYS = ("grad_l2", "total_grad_norm", "energy", "drift_l2")


def _row_nonfinite(row: dict[str, Any]) -> bool:
    for k in _DIVERGENCE_KEYS:
        if k in row and row[k] is not None and not _is_finite(row[k]):
            return True
    return False


def block_has_nonfinite(rows: Sequence[dict[str, Any]]) -> bool:
    return any(_row_nonfinite(r) for r in rows)


def first_nonfinite_step(rows: Sequence[dict[str, Any]]) -> Optional[int]:
    """First global step at which any divergence-key value is NaN/inf."""
    ordered = sorted(rows, key=lambda r: (r.get("step") if r.get("step") is not None else 1 << 60))
    for r in ordered:
        if _row_nonfinite(r):
            step = r.get("step")
            try:
                return int(step)
            except (TypeError, ValueError):
                return None
    return None


def _block_stationary_mean(rows: Sequence[dict[str, Any]], key: str) -> Optional[tuple[float, int, int]]:
    """Mean of `key` over the stationary window, averaged across non-diverged
    blocks. Returns (value, n_blocks_used, n_blocks_diverged) or None."""
    groups = group_rows_by_block(list(rows))
    if not groups:
        return None
    block_means: list[float] = []
    n_div = 0
    for _name, brows in groups.items():
        if block_has_nonfinite(brows):
            n_div += 1
            continue
        stat = stationary_rows(brows, 50, 150)
        vals = [row_metric(r, key) for r in stat]
        vals = [v for v in vals if v is not None]
        if vals:
            block_means.append(sum(vals) / len(vals))
    if not block_means:
        return None
    return (sum(block_means) / len(block_means), len(block_means), n_div)


def grad_norm_gbar(data: dict[str, Any]) -> Optional[tuple[float, int, int]]:
    """||g_bar|| per section 6.1. Returns (gbar, n_blocks_used, n_blocks_diverged)."""
    return _block_stationary_mean(stream_rows(data), "grad_l2")


def drift_stationary(data: dict[str, Any]) -> Optional[float]:
    res = _block_stationary_mean(stream_rows(data), "drift_l2")
    return res[0] if res is not None else None


def heat_lr_of(data: dict[str, Any]) -> float:
    return safe_float((data.get("args") or {}).get("heat_lr")) or ETA


# ----------------------------------------------------------------------------
# Collapse criterion (section 6.2). Priority order is enforced.
# ----------------------------------------------------------------------------

def classify_run(
    data: dict[str, Any],
    source_acc: Optional[float],
    pref_drift: Optional[float],
    use_drift_criterion: bool = True,
) -> dict[str, Any]:
    """Return {collapsed, criterion, mean_acc, last_acc, drift, gbar}.

    Priority:
      1. HARD  -- any NaN/inf in grad_l2/energy/drift (proxy for params), OR
                  stream mean_accuracy <= chance (0.12).
      2. SOFT  -- last_accuracy < source (no-adapt) accuracy for that severity,
                  OR stationary drift_l2 > 5x its value at the reference p_ref.
    A run is STABLE iff neither fires.

    `use_drift_criterion=False` disables the drift-ratio test; used when
    selecting p_ref itself (whose drift defines the ratio, so it can't be
    compared against itself).
    """
    summ = heat_summary(data)
    rows = stream_rows(data)
    mean_acc = safe_float(summ.get("mean_accuracy"))
    last_acc = safe_float(summ.get("last_accuracy"))
    gbar_res = grad_norm_gbar(data)
    gbar = gbar_res[0] if gbar_res is not None else None
    drift = drift_stationary(data)

    out: dict[str, Any] = {
        "collapsed": False,
        "criterion": "stable",
        "mean_acc": mean_acc,
        "last_acc": last_acc,
        "drift": drift,
        "gbar": gbar,
    }

    # 1. Hard collapse.
    if any(_row_nonfinite(r) for r in rows):
        out["collapsed"] = True
        out["criterion"] = "hard:nan_inf"
        return out
    # safe_float returns None for non-finite; an explicit None mean_acc with a
    # populated stream also indicates something went wrong -> treat as collapse.
    if mean_acc is None or mean_acc <= CHANCE_ACC:
        out["collapsed"] = True
        out["criterion"] = "hard:chance_acc"
        return out

    # 2. Soft instability.
    if source_acc is not None and last_acc is not None and last_acc < source_acc:
        out["collapsed"] = True
        out["criterion"] = "soft:below_source"
        return out
    if (
        use_drift_criterion
        and pref_drift is not None
        and pref_drift > 0
        and drift is not None
        and drift > DRIFT_BLOWUP_FACTOR * pref_drift
    ):
        out["collapsed"] = True
        out["criterion"] = "soft:drift_blowup"
        return out

    return out


# ----------------------------------------------------------------------------
# p* bracket selection (section 6.3). Single source of truth for both scripts.
# ----------------------------------------------------------------------------

def choose_pstar(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Given observations [{p, collapsed, valid}], find p* and its bracket.

    Threshold model: below p* collapse, above p* stable. We do NOT discard
    points; non-monotonicity (a stable p below a collapsing p) is flagged but
    p* is still computed from the threshold definition so the plot shows the
    real data.

    Returns dict with p_star, bracket_low, bracket_high, monotone, note.
    """
    valid = [pt for pt in points if pt.get("valid", True)]
    stable = sorted(pt["p"] for pt in valid if not pt["collapsed"])
    collapse = sorted(pt["p"] for pt in valid if pt["collapsed"])

    base = {"p_star": None, "bracket_low": None, "bracket_high": None,
            "monotone": True, "note": ""}

    if not valid:
        base["note"] = "no_valid_runs"
        return base

    if not stable:
        # Everything collapsed (even the strongest tether). p* is above the
        # grid; report the largest p as a lower bound, do not invent a value.
        base["p_star"] = None
        base["bracket_low"] = max(collapse) if collapse else None
        base["bracket_high"] = None
        base["note"] = "all_collapse_pstar_above_grid"
        return base

    p_stable_min = min(stable)
    collapses_below = [p for p in collapse if p < p_stable_min]
    monotone = (not collapse) or (max(collapse) < p_stable_min)

    if not collapses_below:
        # Nothing collapses at or below the smallest stable p. Since the grid
        # always includes p=0, a stable p=0 means p* ~= 0 -- a legitimate point
        # consistent with the law (small shift => small ||g_bar|| => small p*).
        base["p_star"] = 0.0 if (0.0 in stable) else float(p_stable_min)
        base["bracket_low"] = 0.0 if (0.0 in stable) else None
        base["bracket_high"] = float(p_stable_min)
        base["monotone"] = monotone
        if not monotone:
            base["note"] = "non_monotone_stable_below_collapse"
        return base

    p_collapse_max = max(collapses_below)
    base["p_star"] = round((p_collapse_max + p_stable_min) / 2.0, 6)
    base["bracket_low"] = float(p_collapse_max)
    base["bracket_high"] = float(p_stable_min)
    base["monotone"] = monotone
    if not monotone:
        base["note"] = "non_monotone_stable_below_collapse"
    return base


# ----------------------------------------------------------------------------
# Linear fit through the data (NOT forced through origin -- we want to SEE the
# intercept and judge it, per section 7c).
# ----------------------------------------------------------------------------

def least_squares_line(xs: list[float], ys: list[float]) -> Optional[dict[str, Any]]:
    """Ordinary least-squares y = slope*x + intercept, plus R^2."""
    n = len(xs)
    if n < 2 or len(ys) != n or len(set(xs)) < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    preds = [slope * x + intercept for x in xs]
    r2 = compute_r2(ys, preds)
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


__all__ = [
    "ETA",
    "PREF_DEFAULT",
    "PREF_FALLBACKS",
    "DEFAULT_P_GRID",
    "EXTEND_P_GRID",
    "DEFAULT_SEVERITIES",
    "DEFAULT_SEVERITIES_BY_ARCH",
    "DEFAULT_ARCHS",
    "format_p",
    "variant_tag",
    "run_output_path",
    "run_error_path",
    "discover_p_values",
    "discover_etas",
    "scaled_p_grid",
    "scaled_pref_ladder",
    "scaled_extend_grid",
    "build_run_command",
    "execute_run",
    "heat_summary",
    "source_mean_acc",
    "stream_rows",
    "first_nonfinite_step",
    "grad_norm_gbar",
    "drift_stationary",
    "heat_lr_of",
    "classify_run",
    "choose_pstar",
    "least_squares_line",
    "load_json",
]
