"""
Stage 1b (E6 cross-mechanism / E7 drive-swap) — shared identity/helpers.

ADDITIVE module (nothing frozen is modified). Owns what BOTH the E6/E7
orchestrators and their analyses must agree on exactly:

  * variant tags:
      E6 anchor : pstar_<corruption>_lr<eta>_anchor_lam<lambda>
      E7 entropy: pstar_<corruption>_lr<eta>_drive_<drive>_p<p>
    (source runs are REUSED from E1: pstar_<corruption>_lr<eta>_source —
    the source pass is mechanism/drive-independent.)
  * discovery, filtered so Stage-1 patterns and Stage-1b patterns can never
    cross-match (E1's p-discovery requires "_lr<eta>_p" immediately; anchor
    files have "_anchor_", drive files have "_drive_" there; E1's eta
    discovery regex captures a non-float for these files and skips them).
  * run commands (run_tier2.py --protocol p9 --corruptions <c> with
    --heat-anchor-lambda / --drive entropy; method name stays "heat", so the
    runner's diagnostics collection and the results.summary.heat key are
    untouched).
  * CRITERION (Stage-1b update): the PRIMARY law criterion is HARD collapse
    (nan_inf / chance), matching the Stage-1 forward confirmation.
    soft:below_source is recorded for every run and reported as a SEPARATE
    boundary, never mixed into law fits. Classification reuses
    pstar_common.classify_run(hard_only=True/False) UNCHANGED.
  * read-only per-cell analysis returning BOTH boundaries per cell.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import pvariance
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts.analysis_common import (
    group_rows_by_block, stationary_rows, row_metric,
)

# Bracket-weighted hard-only Bernoulli reference (wrn28_10, sev5) from A2,
# fixed by the Stage-1b GO instruction. Orchestrators take --s-frozen with
# this default and record the value used inside the predictions file.
S_FROZEN_DEFAULT = 1.185

E6_CELLS: list[tuple[str, float]] = [
    ("gaussian_noise", 5e-4), ("gaussian_noise", 1e-3),
    ("gaussian_noise", 2e-3), ("elastic_transform", 1e-3),
]
E7_CORRUPTIONS = ["gaussian_noise", "elastic_transform"]
E7_ETAS = [5e-4, 1e-3, 2e-3]


# ----------------------------------------------------------------------------
# Tags / paths / discovery — E6 anchor.
# ----------------------------------------------------------------------------

def variant_tag_anchor(corruption: str, lam: float, eta: float) -> str:
    return f"pstar_{corruption}_lr{pc.format_p(eta)}_anchor_lam{pc.format_p(lam)}"


def run_output_path_anchor(results_dir, arch, corruption, severity, seed,
                           lam, eta) -> Path:
    tag = variant_tag_anchor(corruption, lam, eta)
    return Path(results_dir) / f"p9_{arch}_{tag}_seed{seed}_sev{severity}.json"


def discover_lambdas(results_dir, arch, corruption, severity, seed,
                     eta) -> list[float]:
    prefix = f"p9_{arch}_pstar_{corruption}_lr{pc.format_p(eta)}_anchor_lam"
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(re.escape(prefix) + r"(.+?)" + re.escape(suffix) + r"$")
    out = []
    for path in Path(results_dir).glob(f"{prefix}*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
    return sorted(set(out))


# ----------------------------------------------------------------------------
# Tags / paths / discovery — E7 drive.
# ----------------------------------------------------------------------------

def variant_tag_drive(corruption: str, drive: str, p: float, eta: float) -> str:
    return (f"pstar_{corruption}_lr{pc.format_p(eta)}_drive_{drive}"
            f"_p{pc.format_p(p)}")


def run_output_path_drive(results_dir, arch, corruption, severity, seed,
                          drive, p, eta) -> Path:
    tag = variant_tag_drive(corruption, drive, p, eta)
    return Path(results_dir) / f"p9_{arch}_{tag}_seed{seed}_sev{severity}.json"


def discover_drive_p_values(results_dir, arch, corruption, severity, seed,
                            drive, eta) -> list[float]:
    prefix = (f"p9_{arch}_pstar_{corruption}_lr{pc.format_p(eta)}"
              f"_drive_{drive}_p")
    suffix = f"_seed{seed}_sev{severity}.json"
    pat = re.compile(re.escape(prefix) + r"(.+?)" + re.escape(suffix) + r"$")
    out = []
    for path in Path(results_dir).glob(f"{prefix}*{suffix}"):
        m = pat.match(path.name)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                continue
    return sorted(set(out))


# ----------------------------------------------------------------------------
# Run commands. Method name stays "heat"; the factory dispatches on the flags.
# ----------------------------------------------------------------------------

def _base_cmd(run_tier2, arch, checkpoint, corruption, severity, seed,
              results_dir, c10c_root, batch_size, num_workers, python=None):
    python = python or sys.executable
    return [
        python, str(run_tier2), "--protocol", "p9", "--arch", arch,
        "--dataset", "cifar10", "--checkpoint", str(checkpoint),
        "--c10c-root", str(c10c_root), "--severity", str(severity),
        "--seed", str(seed), "--batch-size", str(batch_size),
        "--num-workers", str(num_workers), "--out-dir", str(results_dir),
        "--corruptions", str(corruption),
    ]


def build_run_command_anchor(*, run_tier2, arch, checkpoint, corruption,
                             severity, seed, results_dir, c10c_root, heat_lr,
                             anchor_lambda, batch_size=64, num_workers=2,
                             python=None) -> list[str]:
    cmd = _base_cmd(run_tier2, arch, checkpoint, corruption, severity, seed,
                    results_dir, c10c_root, batch_size, num_workers, python)
    cmd += ["--methods", "heat",
            "--heat-lr", repr(float(heat_lr)),
            "--heat-restore-prob", "0",
            "--heat-anchor-lambda", repr(float(anchor_lambda)),
            "--heat-diagnostic-snapshot",
            "--variant-tag", variant_tag_anchor(corruption, anchor_lambda,
                                                heat_lr)]
    return cmd


def build_run_command_drive(*, run_tier2, arch, checkpoint, corruption,
                            severity, seed, results_dir, c10c_root, heat_lr,
                            drive, p, batch_size=64, num_workers=2,
                            python=None) -> list[str]:
    cmd = _base_cmd(run_tier2, arch, checkpoint, corruption, severity, seed,
                    results_dir, c10c_root, batch_size, num_workers, python)
    cmd += ["--methods", "heat",
            "--heat-lr", repr(float(heat_lr)),
            "--heat-restore-prob", pc.format_p(p),
            "--drive", str(drive),
            "--heat-diagnostic-snapshot",
            "--variant-tag", variant_tag_drive(corruption, drive, p, heat_lr)]
    return cmd


# ----------------------------------------------------------------------------
# Classification (hard primary, soft recorded) + generic boundary search
# support. Reuses classify_run / choose_pstar UNCHANGED.
# ----------------------------------------------------------------------------

def classify_both(data, source_acc):
    """Hard-primary + soft-recorded classification of one run (no drift-ratio
    criterion: p_ref-relative drift is undefined across mechanisms/drives;
    hard is primary per the Stage-1b criterion update)."""
    hard = pc.classify_run(data, source_acc, pref_drift=None,
                           use_drift_criterion=False, hard_only=True)
    soft = pc.classify_run(data, source_acc, pref_drift=None,
                           use_drift_criterion=False, hard_only=False)
    return hard, soft


def store_point(points, key_value, data, source_acc):
    """Record one boundary observation keyed by the swept value (p or lambda).
    choose_pstar consumes these dicts via their 'p' field."""
    if data is None:
        points[key_value] = {"p": key_value, "collapsed": False, "valid": False}
        return points[key_value]
    hard, soft = classify_both(data, source_acc)
    points[key_value] = {
        "p": key_value, "collapsed": hard["collapsed"], "valid": True,
        "criterion": hard["criterion"], "mean_acc": hard["mean_acc"],
        "gbar": hard["gbar"],
        "soft_collapsed": soft["collapsed"],
        "soft_criterion": soft["criterion"],
    }
    return points[key_value]


def boundary_from_points(points: dict[float, dict], use_soft=False):
    """choose_pstar over the recorded points; use_soft re-derives the SEPARATE
    soft boundary from the recorded soft classifications."""
    if not use_soft:
        return pc.choose_pstar(list(points.values()))
    soft_pts = [{"p": v["p"], "collapsed": v.get("soft_collapsed", False),
                 "valid": v.get("valid", False)} for v in points.values()]
    return pc.choose_pstar(soft_pts)


def analyze_cell_generic(results_dir, arch, corruption, severity, eta, seed,
                         values: list[float], loader) -> dict[str, Any]:
    """Read-only cell row over swept `values` (lambdas or ps); `loader(v)`
    returns the run JSON or None. Both boundaries are computed; the HARD one
    is the law boundary."""
    source_acc = sc.source_acc_sc(results_dir, arch, corruption, severity,
                                  seed, eta)
    points: dict[float, dict] = {}
    per_v: dict[str, dict] = {}
    for v in values:
        data = loader(v)
        rec = store_point(points, v, data, source_acc)
        per_v[pc.format_p(v)] = {k: rec.get(k) for k in
                                 ("valid", "collapsed", "criterion", "mean_acc",
                                  "gbar", "soft_collapsed", "soft_criterion")}
    hard_sel = boundary_from_points(points, use_soft=False)
    soft_sel = boundary_from_points(points, use_soft=True)
    return {
        "arch": arch, "corruption": corruption, "severity": severity,
        "seed": seed, "eta": eta, "source_acc": source_acc,
        "values_run": values, "per_value": per_v,
        "hard": hard_sel, "soft": soft_sel,
    }


# ----------------------------------------------------------------------------
# PERMANENT METHODOLOGICAL-INTEGRITY RULES (Stage-1b audit, F2/F3 fixes).
# Apply to ALL analyzers/orchestrators from E6/E7 onward (Stage 2+ included):
#
#   RULE 1 (VOID guard): if no reference points are discoverable (e.g. zero
#     Bernoulli hard points for a cross-mechanism fit), the verdict is VOID —
#     never a curve verdict. A boundary-resolution anomaly (p*=0 with the
#     non-monotone stable-below-collapse note) also voids the verdict:
#     instrument failure until proven otherwise.
#
#   RULE 2 (loud abort): orchestrators must ABORT before running anything if
#     the expected E1 reference files are absent from RESULTS_DIR — silent
#     re-running of references is forbidden — unless --fresh-reference is
#     explicitly passed (recorded in the manifest and predictions file).
#
#   RULE 3 (predictions validity): an existing pre-registration file is
#     INVALID by default. The orchestrator refuses to proceed until the caller
#     chooses: --trust-existing-predictions (the audit confirmed the reference
#     ||g_bar|| runs were genuine) or --requarantine-predictions (quarantine
#     the old file to analysis/quarantine/ with a note, then re-freeze new
#     predictions BEFORE any new grid run).
# ----------------------------------------------------------------------------

VOID = "VOID"


def require_e1_reference(results_dir, cells, severity, seed,
                         fresh_ok: bool) -> list[dict]:
    """RULE 2. `cells` = [(corruption, eta), ...] the campaign will use.
    Returns the missing-reference report (empty if all present). Aborts loudly
    when anything is missing and fresh_ok is False."""
    missing = []
    for corruption, eta in cells:
        src_ok = sc.run_output_path_sc(results_dir, sc.E1_ARCH, corruption,
                                       severity, seed, source=True,
                                       eta=eta).exists()
        n_pts = len(sc.discover_p_values_sc(results_dir, sc.E1_ARCH,
                                            corruption, severity, seed, eta))
        if not src_ok or n_pts == 0:
            missing.append({"corruption": corruption, "eta": eta,
                            "source_present": src_ok,
                            "n_e1_heat_runs": n_pts})
    if missing and not fresh_ok:
        lines = "\n".join(
            f"    {m['corruption']} eta={pc.format_p(m['eta'])}: "
            f"source={'ok' if m['source_present'] else 'MISSING'}, "
            f"E1 heat runs={m['n_e1_heat_runs']}" for m in missing)
        raise SystemExit(
            "[ABORT — RULE 2] expected E1 reference files are absent from "
            f"this RESULTS_DIR ({results_dir}):\n{lines}\n"
            "  A campaign here would silently re-run references and be blind "
            "to the Bernoulli baseline (the exact failure of the voided "
            "Stage-1b run). Point --results-dir at the dir that holds the E1 "
            "JSONs, or pass --fresh-reference to proceed deliberately.")
    if missing:
        print(f"[RULE 2] --fresh-reference: proceeding WITHOUT {len(missing)} "
              f"expected E1 reference cell(s); recorded.", flush=True)
    return missing


def quarantine_file(path: Path, reason: str) -> Path:
    """Move a file to <parent>/quarantine/ with a note — never overwritten."""
    qdir = path.parent / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = qdir / f"{path.name}.{stamp}"
    shutil.move(str(path), str(dest))
    (qdir / f"{path.name}.{stamp}.note.md").write_text(
        f"Quarantined {path.name} at {stamp} UTC.\nReason: {reason}\n",
        encoding="utf-8")
    return dest


def resolve_predictions_policy(pred_path: Path, trust: bool,
                               requarantine: bool) -> str:
    """RULE 3. Returns 'fresh' (no file / after quarantine) or 'use'."""
    if trust and requarantine:
        raise SystemExit("[ABORT] pass only ONE of "
                         "--trust-existing-predictions / "
                         "--requarantine-predictions.")
    if not pred_path.exists():
        return "fresh"
    if trust:
        print(f"[RULE 3] trusting existing pre-registration {pred_path.name} "
              f"(audit-confirmed).", flush=True)
        return "use"
    if requarantine:
        dest = quarantine_file(
            pred_path, "pre-registration declared INVALID (reference runs "
                       "not confirmed genuine by the Stage-1b audit); "
                       "re-freezing new predictions before any new grid.")
        print(f"[RULE 3] quarantined old pre-registration -> {dest}", flush=True)
        return "fresh"
    raise SystemExit(
        f"[ABORT — RULE 3] {pred_path} already exists and pre-registered "
        "predictions are INVALID by default after the voided campaign.\n"
        "  Pass --trust-existing-predictions ONLY if the audit confirmed the "
        "reference ||g_bar|| runs were genuine (correct checkpoint + correct "
        "dispatch), or --requarantine-predictions to quarantine it and "
        "re-freeze new predictions before any new grid run.")


def void_if_no_reference(n_reference_points: int, n_own_points: int,
                         anomalous_cells: list[str], what: str):
    """RULE 1. Returns (verdict_or_None, reason). None => not void."""
    if n_reference_points == 0:
        return VOID, (f"zero {what} reference points discoverable — no curve "
                      "verdict may be emitted (permanent rule).")
    if n_own_points == 0:
        return VOID, "zero measured boundary points — nothing to fit."
    if anomalous_cells:
        return VOID, ("boundary anomaly (p*=0 with non-monotone stable-below-"
                      f"collapse) in cells {anomalous_cells} — instrument "
                      "failure until proven otherwise.")
    return None, ""


# ----------------------------------------------------------------------------
# Calmness metrics (E7): stationary drift variance and step-to-step grad_l2
# variance over the codebase-standard stationary window (local steps 50-150),
# per block (single block for single-corruption streams).
# ----------------------------------------------------------------------------

def calmness_metrics(data) -> Optional[dict[str, float]]:
    rows = pc.stream_rows(data)
    if not rows or pc.first_nonfinite_step(rows) is not None:
        return None
    groups = group_rows_by_block(rows)
    drifts, gdiffs = [], []
    for _name, brows in groups.items():
        ordered = sorted(brows, key=lambda r: (r.get("local_step") or 0))
        stat = stationary_rows(ordered, 50, 150)
        d = [row_metric(r, "drift_l2") for r in stat]
        g = [row_metric(r, "grad_l2") for r in stat]
        d = [v for v in d if v is not None]
        g = [v for v in g if v is not None]
        drifts.extend(d)
        gdiffs.extend(b - a for a, b in zip(g, g[1:]))
    if len(drifts) < 3 or len(gdiffs) < 3:
        return None
    return {
        "drift_var_stationary": float(pvariance(drifts)),
        "grad_l2_step_var_stationary": float(pvariance(gdiffs)),
        "n_stationary_steps": len(drifts),
    }


__all__ = [
    "S_FROZEN_DEFAULT", "E6_CELLS", "E7_CORRUPTIONS", "E7_ETAS",
    "variant_tag_anchor", "run_output_path_anchor", "discover_lambdas",
    "variant_tag_drive", "run_output_path_drive", "discover_drive_p_values",
    "build_run_command_anchor", "build_run_command_drive",
    "classify_both", "store_point", "boundary_from_points",
    "analyze_cell_generic", "calmness_metrics",
]
