"""
Stage 1 (E1) — ||g_bar||-factor test: shared identity/helpers for
SINGLE-CORRUPTION p* cells.

ADDITIVE module. Nothing in scripts/pstar_common.py, scripts/run_tier2.py,
src/methods/heat.py, or the runner is modified. This module owns the pieces
that BOTH the E1 sweep (scripts/run_stage1_e1.py) and the Stage-1 analysis
(scripts/analyze_stage1.py) must agree on exactly:

  * the corruption-tagged variant tag:  pstar_<corruption>_lr<eta>_p<p>
    (source runs: pstar_<corruption>_lr<eta>_source)
  * the run-output filename convention (must match scripts/run_tier2.py:
    p9_<arch>_<tag>_seed<seed>_sev<severity>.json)
  * corruption-filtered discovery of p-values / etas on disk
  * the run command for a SINGLE-CORRUPTION stream (run_tier2.py --protocol p9
    --corruptions <c> — natively supported; no runner change was needed)
  * the read-only per-cell analysis (p_ref derivation, p* bracket, ||g_bar||),
    which reuses pstar_common's classify_run / choose_pstar / grad_norm_gbar
    UNCHANGED (global rule: the collapse-criterion math is never re-implemented).

Old-file compatibility (verified by tests in analyze_stage1 --self-test):
  * old continual tags are  pstar_lr<eta>_p<p>  — the new tag inserts the
    corruption between "pstar_" and "lr", so pc.discover_p_values /
    pc.discover_etas (prefix "p9_<arch>_pstar_lr") can never match a
    corruption-tagged file, and discover_p_values_sc (prefix
    "p9_<arch>_pstar_<corruption>_lr") can never match an old continual file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc


# ----------------------------------------------------------------------------
# E1 campaign definition (master plan, Stage 1).
# ----------------------------------------------------------------------------

E1_ARCH = "wrn28_10"
E1_SEVERITY = 5
E1_ETAS: list[float] = [5e-4, 1e-3, 2e-3]

# Calibration cells are the ONLY cells allowed for fitting the slope (E4).
CALIBRATION_CORRUPTIONS: list[str] = ["gaussian_noise", "elastic_transform"]
# Held-out cells are NEVER used for fitting; they are the zero-shot test set.
HELDOUT_CORRUPTIONS: list[str] = ["impulse_noise", "contrast"]
# Run order: calibration first so the slope lands early if a session dies.
E1_CORRUPTIONS: list[str] = CALIBRATION_CORRUPTIONS + HELDOUT_CORRUPTIONS


# ----------------------------------------------------------------------------
# Run identity (corruption-tagged).
# ----------------------------------------------------------------------------

def variant_tag_sc(corruption: str, p: Optional[float] = None,
                   source: bool = False, eta: Optional[float] = None) -> str:
    """Tag for a single-corruption run.

      heat run: pstar_<corruption>_lr<eta>_p<p>
      source  : pstar_<corruption>_lr<eta>_source
    """
    if not corruption:
        raise ValueError("variant_tag_sc requires a corruption name")
    eta_part = f"lr{pc.format_p(eta)}_" if eta is not None else ""
    if source:
        return f"pstar_{corruption}_{eta_part}source"
    if p is None:
        raise ValueError("variant_tag_sc requires p when source=False")
    return f"pstar_{corruption}_{eta_part}p{pc.format_p(p)}"


def run_output_path_sc(
    results_dir,
    arch: str,
    corruption: str,
    severity: int,
    seed: int,
    p: Optional[float] = None,
    source: bool = False,
    eta: Optional[float] = None,
) -> Path:
    tag = variant_tag_sc(corruption, p=p, source=source, eta=eta)
    # dataset is always cifar10 here, so dataset_tag is empty (matches run_tier2).
    fname = f"p9_{arch}_{tag}_seed{seed}_sev{severity}.json"
    return Path(results_dir) / fname


def run_error_path_sc(results_dir, arch, corruption, severity, seed,
                      p=None, source=False, eta=None) -> Path:
    out = run_output_path_sc(results_dir, arch, corruption, severity, seed,
                             p=p, source=source, eta=eta)
    return out.with_suffix(".run_error.txt")


def discover_p_values_sc(results_dir, arch: str, corruption: str,
                         severity: int, seed: int, eta: float) -> list[float]:
    """Every restore-prob whose run JSON exists on disk for this
    (arch, corruption, severity, eta) cell. Filtered by the corruption tag AND
    the eta tag, so cells never mix across corruptions or etas, and old
    continual files (no corruption in the tag) can never match."""
    prefix = f"p9_{arch}_pstar_{corruption}_lr{pc.format_p(eta)}_p"
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


def discover_etas_sc(results_dir, arch: str, corruption: str,
                     severity: int, seed: int) -> list[float]:
    """Every eta with at least one heat run on disk for (arch, corruption, sev)."""
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(
        r"^p9_" + re.escape(arch) + r"_pstar_" + re.escape(corruption)
        + r"_lr(.+?)_p.+?" + re.escape(suffix) + r"$")
    etas: set[float] = set()
    for path in Path(results_dir).glob(
            f"p9_{arch}_pstar_{corruption}_lr*_p*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                etas.add(float(m.group(1)))
            except ValueError:
                continue
    return sorted(etas)


# ----------------------------------------------------------------------------
# Run command (single-corruption stream). run_tier2.py --protocol p9 natively
# accepts --corruptions <c> (run_p9 uses `args.corruptions or P9_DEFAULT_STREAM`),
# so the stream is exactly that corruption's full severity split.
# ----------------------------------------------------------------------------

def build_run_command_sc(
    *,
    run_tier2: Path,
    arch: str,
    checkpoint: str,
    corruption: str,
    severity: int,
    seed: int,
    results_dir: Path,
    c10c_root: str,
    heat_lr: float = pc.ETA,
    batch_size: int = 64,
    num_workers: int = 2,
    p: Optional[float] = None,
    source: bool = False,
    python: Optional[str] = None,
) -> list[str]:
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
        "--corruptions", str(corruption),
    ]
    if source:
        cmd += ["--methods", "source",
                "--variant-tag", variant_tag_sc(corruption, source=True, eta=heat_lr)]
    else:
        cmd += [
            "--methods", "heat",
            "--heat-lr", repr(float(heat_lr)),
            "--heat-restore-prob", pc.format_p(p if p is not None else 0.0),
            "--heat-diagnostic-snapshot",  # required so drift_l2 exists even at p=0
            "--variant-tag", variant_tag_sc(corruption, p=p, eta=heat_lr),
        ]
    return cmd


# ----------------------------------------------------------------------------
# Environment fingerprint for the per-run manifest (§1.4 logging: git hash,
# GPU name, torch version). Collected once per process; failures degrade to
# "unknown" so a missing tool can never kill a sweep.
# ----------------------------------------------------------------------------

def environment_fingerprint(repo_root: Optional[Path] = None) -> dict[str, str]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]

    def _git(*args) -> str:
        try:
            out = subprocess.run(["git", "-C", str(root), *args],
                                 capture_output=True, text=True, timeout=15)
            return out.stdout.strip() if out.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    git_hash = _git("rev-parse", "HEAD")
    git_branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    torch_version = "unknown"
    gpu_name = "none"
    try:
        import torch  # local import: analysis boxes may not have torch
        torch_version = str(torch.__version__)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {"git_hash": git_hash, "git_branch": git_branch,
            "torch_version": torch_version, "gpu_name": gpu_name}


# ----------------------------------------------------------------------------
# Read-only per-cell analysis (mirrors analyze_pstar_law.analyze_cell, but
# corruption-tagged). Reuses pstar_common classify_run / choose_pstar /
# grad_norm_gbar byte-for-byte — the criterion math is never re-implemented.
# ----------------------------------------------------------------------------

def load_run_sc(results_dir, arch, corruption, severity, seed, p=None,
                source=False, eta=None) -> Optional[dict[str, Any]]:
    path = run_output_path_sc(results_dir, arch, corruption, severity, seed,
                              p=p, source=source, eta=eta)
    if not path.exists():
        return None
    try:
        return pc.load_json(path)
    except Exception:
        return None


def source_acc_sc(results_dir, arch, corruption, severity, seed, eta) -> Optional[float]:
    data = load_run_sc(results_dir, arch, corruption, severity, seed,
                       source=True, eta=eta)
    return pc.source_mean_acc(data) if data is not None else None


def derive_pref_sc(results_dir, arch, corruption, severity, seed, eta,
                   source_acc, hard_only: bool = False):
    """Read-only re-derivation of the reference p_ref for one cell.

    Same policy as the continual sweep: eta-SCALED ladder; stability judged by
    hard collapse + below-source only (the drift-ratio criterion is defined
    relative to p_ref itself). Returns (p_ref, pref_drift, gbar)."""
    candidates = pc.scaled_pref_ladder(arch, eta)
    last = (None, None, None)
    for cand in candidates:
        data = load_run_sc(results_dir, arch, corruption, severity, seed,
                           p=cand, eta=eta)
        if data is None:
            continue
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        drift = pc.drift_stationary(data)
        last = (cand, drift, gbar)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False, hard_only=hard_only)
        if not v["collapsed"]:
            return cand, drift, gbar
    return last  # no stable reference found; best effort


def analyze_cell_sc(results_dir, arch, corruption, severity, eta, seed,
                    hard_only: bool = False) -> dict[str, Any]:
    """Full row for one (arch, corruption, severity, eta) single-corruption cell.

    Also records per-p mean accuracy and collapse classification — E4/E5 read
    accuracies straight off the measured grid."""
    source_acc = source_acc_sc(results_dir, arch, corruption, severity, seed, eta)
    p_ref, pref_drift, gbar = derive_pref_sc(
        results_dir, arch, corruption, severity, seed, eta, source_acc,
        hard_only=hard_only)

    p_values = discover_p_values_sc(results_dir, arch, corruption, severity,
                                    seed, eta)
    points = []
    per_p: dict[float, dict[str, Any]] = {}
    eta_used = None
    for p in p_values:
        data = load_run_sc(results_dir, arch, corruption, severity, seed,
                           p=p, eta=eta)
        if data is None:
            points.append({"p": p, "collapsed": False, "valid": False})
            per_p[p] = {"valid": False}
            continue
        if eta_used is None:
            eta_used = pc.heat_lr_of(data)
        v = pc.classify_run(data, source_acc, pref_drift, hard_only=hard_only)
        points.append({"p": p, "collapsed": v["collapsed"], "valid": True})
        per_p[p] = {"valid": True, "collapsed": v["collapsed"],
                    "criterion": v["criterion"], "mean_acc": v["mean_acc"],
                    "gbar": v["gbar"], "drift": v["drift"]}

    sel = pc.choose_pstar(points)
    blo, bhi = sel["bracket_low"], sel["bracket_high"]

    if blo is not None and blo in per_p and per_p[blo].get("collapsed"):
        collapse_criterion = per_p[blo].get("criterion", "unknown")
    elif sel["p_star"] == 0.0:
        collapse_criterion = "none(p*~0)"
    else:
        collapse_criterion = sel.get("note") or "none"

    if eta_used is None:
        eta_used = eta
    return {
        "arch": arch,
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "source_acc": source_acc,
        "p_ref_used": p_ref,
        "grad_norm_gbar": gbar,
        "eta": eta_used,
        "eta_times_gbar": (gbar * eta_used) if gbar is not None else None,
        "p_star": sel["p_star"],
        "p_star_bracket_low": blo,
        "p_star_bracket_high": bhi,
        "collapse_criterion": collapse_criterion,
        "stable_mean_acc": (per_p.get(bhi, {}).get("mean_acc")
                            if bhi is not None else None),
        "monotone_bracket": sel["monotone"],
        "bracket_note": sel["note"],
        "p_values_run": p_values,
        "per_p": {pc.format_p(p): per_p[p] for p in p_values},
        "criterion_family": "hard_only" if hard_only else "soft+hard",
    }


__all__ = [
    "E1_ARCH", "E1_SEVERITY", "E1_ETAS",
    "CALIBRATION_CORRUPTIONS", "HELDOUT_CORRUPTIONS", "E1_CORRUPTIONS",
    "variant_tag_sc", "run_output_path_sc", "run_error_path_sc",
    "discover_p_values_sc", "discover_etas_sc", "build_run_command_sc",
    "environment_fingerprint",
    "load_run_sc", "source_acc_sc", "derive_pref_sc", "analyze_cell_sc",
]
