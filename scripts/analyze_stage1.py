"""
Stage 1 analysis — E1 fit/verdict + zero-GPU analyses E2-E5.

Single source of truth for the Stage-1 verdict. Recomputes everything from the
run JSONs on disk (never trusts a sweep's in-memory summary), reusing
pstar_common's classify_run / choose_pstar / grad_norm_gbar UNCHANGED.

  E1  ||g_bar||-factor test: per-corruption p* vs eta*||g_bar|| lines
      (WRN-28-10 sev5 single-corruption cells), soft+hard and hard-only fitted
      SEPARATELY (never averaged), pooled fit, calibration-only fit, x-spread,
      previous continual eta-sweep line overlay.
  E2  Convergence of the running mean of grad_l2 to the stationary value
      (from existing continual eta-sweep JSONs + new E1 JSONs).
  E3  Coupling index: relative change of stationary ||g_bar|| across the
      stable p range, per setting (CIFAR continual cells, E1 cells, DomainNet
      clipart cells).
  E4  Zero-shot p on held-out corruptions: p_hat = slope_calibration * eta *
      ||g_bar|| (law applied through the origin; slope fitted on CALIBRATION
      cells only), snapped to the nearest measured grid p, vs the oracle.
  E5  E4 with R perturbed by {-50,-25,+25,+50}% (p_hat -> p_hat/(1+delta)).

Artifacts (in <results-dir>/analysis/):
  stage1_gbar_law.json / stage1_gbar_law.png
  stage1_convergence.png
  stage1_coupling.md
  stage1_zeroshot.md
  stage1_verdict.md

Also prints the full cell table, fits, verdict block (exact criterion lines),
deviations, and the STOP notice.

Regression anchor: the continual WRN sev5 eta=1e-3 cell is recomputed from the
eta-sweep dir; if p* moved from ~0.0094 the script flags STOP prominently.

Usage:
  python scripts/analyze_stage1.py --results-dir <E1 dir> \
      [--etasweep-results-dir <dir>] [--domainnet-results-dir <dir>]
  python scripts/analyze_stage1.py --self-test   # no results needed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts.analysis_common import (
    group_rows_by_block, stationary_rows, row_metric, safe_float,
)
from scripts.analyze_pstar_law import (
    analyze_cell as analyze_cell_continual,
)

# --- Anchors / thresholds (exact criterion constants from the master plan) ---
ANCHOR_PSTAR = 0.0094          # WRN sev5 continual eta=1e-3
ANCHOR_TOL = 0.0005            # |p* - 0.0094| beyond this => STOP flag
E1_POOLED_R2_GREEN = 0.95
E1_SLOPE_SPREAD_GREEN = 0.20   # spread := max_slope/min_slope - 1
E1_SLOPE_SPREAD_AMBER_HI = 1.00
E1_SLOPE_SPREAD_RED = 1.00     # spread > 1.0 <=> ratio > 2x
E1_XSPREAD_GREEN = 1.5
E1_LINEAR_R2 = 0.90            # "linear per corruption" (same family as R2_GOOD)
E2_CONV_STEPS = 50
E2_GREEN_FRAC = 0.90
E2_BAND = 0.20                 # within 20% of the stationary value
E4_GREEN_RATIO = 0.95
E4_AMBER_RATIO = 0.85
E4_CELL_FRAC = 0.80
E5_DELTAS = [-0.50, -0.25, 0.25, 0.50]
E5_GRACEFUL_RATIO = 0.90       # operationalization of "graceful at -25%"

CONTINUAL_ARCHS = ["wrn28_10", "resnet18"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 analysis (E1-E5 + verdict)")
    p.add_argument("--results-dir", type=str, default="",
                   help="Dir with the E1 single-corruption run JSONs.")
    p.add_argument("--etasweep-results-dir", type=str, default="",
                   help="Dir with the continual eta-sweep JSONs "
                        "(default: same as --results-dir).")
    p.add_argument("--domainnet-results-dir", type=str, default="",
                   help="Dir with the DomainNet clipart run JSONs (optional; "
                        "E3 covers CIFAR only if omitted).")
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domainnet-chance-acc", type=float, default=0.02)
    p.add_argument("--self-test", action="store_true",
                   help="Run the built-in compatibility/pipeline tests on "
                        "synthetic JSONs (no GPU, no results dir needed).")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# ============================================================================
# E1 — cell rows + fits
# ============================================================================

def e1_all_rows(results_dir, severity, seed, hard_only):
    rows = []
    for corruption in sc.E1_CORRUPTIONS:
        etas = sorted(set(sc.E1_ETAS) | set(
            sc.discover_etas_sc(results_dir, sc.E1_ARCH, corruption, severity, seed)))
        for eta in etas:
            rows.append(sc.analyze_cell_sc(results_dir, sc.E1_ARCH, corruption,
                                           severity, eta, seed,
                                           hard_only=hard_only))
    return rows


def fit_rows(rows):
    """Least-squares p* vs eta*||g_bar|| over usable rows."""
    usable = [r for r in rows
              if r.get("eta_times_gbar") is not None and r.get("p_star") is not None]
    xs = [r["eta_times_gbar"] for r in usable]
    ys = [r["p_star"] for r in usable]
    return pc.least_squares_line(xs, ys), usable


def monotone_in_x(rows) -> bool:
    seq = sorted((r["eta_times_gbar"], r["p_star"]) for r in rows
                 if r.get("eta_times_gbar") is not None and r.get("p_star") is not None)
    vals = [y for _x, y in seq]
    return all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def e1_fits(rows):
    """Per-corruption, pooled, and calibration-only fits for ONE criterion
    family (call twice: soft+hard rows, hard-only rows; never averaged)."""
    per_corruption = {}
    for c in sc.E1_CORRUPTIONS:
        crows = [r for r in rows if r["corruption"] == c]
        fit, usable = fit_rows(crows)
        per_corruption[c] = {
            "fit": fit, "n_usable": len(usable),
            "monotone_in_x": monotone_in_x(crows),
            "mean_gbar": (sum(r["grad_norm_gbar"] for r in crows
                              if r.get("grad_norm_gbar") is not None)
                          / max(1, sum(1 for r in crows
                                       if r.get("grad_norm_gbar") is not None)))
            if any(r.get("grad_norm_gbar") is not None for r in crows) else None,
        }
    pooled_fit, pooled_usable = fit_rows(rows)
    calib_rows = [r for r in rows if r["corruption"] in sc.CALIBRATION_CORRUPTIONS]
    calib_fit, calib_usable = fit_rows(calib_rows)
    return {
        "per_corruption": per_corruption,
        "pooled": {"fit": pooled_fit, "n_usable": len(pooled_usable)},
        "calibration_only": {"fit": calib_fit, "n_usable": len(calib_usable)},
    }


def e1_xspread(fits) -> float | None:
    """x-spread of corruption positions := max/min of per-corruption mean
    ||g_bar|| (measured at p_ref). eta is common across corruptions, so the
    gbar ratio IS the x-position ratio at fixed eta."""
    gbars = [info["mean_gbar"] for info in fits["per_corruption"].values()
             if info["mean_gbar"]]
    if len(gbars) < 2 or min(gbars) <= 0:
        return None
    return max(gbars) / min(gbars)


def continual_reference(etasweep_dir, arch, severity, seed, hard_only):
    """Recompute the continual eta-sweep rows + fit from disk (read-only).
    Used for the overlay line, the reference slope, and the anchor check."""
    etas = pc.discover_etas(etasweep_dir, arch, severity, seed)
    rows = [analyze_cell_continual(etasweep_dir, arch, severity, eta, seed,
                                   hard_only=hard_only)
            for eta in etas]
    usable = [r for r in rows
              if r.get("eta_times_gbar") is not None and r.get("p_star") is not None]
    fit = pc.least_squares_line([r["eta_times_gbar"] for r in usable],
                                [r["p_star"] for r in usable])
    return rows, fit


def anchor_check(continual_rows_soft):
    """WRN sev5 continual eta=1e-3 must still give p* ~= 0.0094."""
    row = next((r for r in continual_rows_soft
                if r.get("eta") is not None and abs(r["eta"] - 1e-3) < 1e-9), None)
    if row is None or row.get("p_star") is None:
        return {"status": "NOT-CHECKABLE",
                "detail": "continual eta=1e-3 cell not found on disk "
                          "(run from a box with the eta-sweep Drive dir)."}
    moved = abs(row["p_star"] - ANCHOR_PSTAR) > ANCHOR_TOL
    return {"status": "MOVED-STOP" if moved else "OK",
            "p_star": row["p_star"], "expected": ANCHOR_PSTAR,
            "tolerance": ANCHOR_TOL,
            "detail": (f"recomputed p*={row['p_star']:.5g} vs anchor "
                       f"{ANCHOR_PSTAR} (tol {ANCHOR_TOL})")}


def e1_verdict(fits, ref_fit, family: str):
    """Mechanical E1 verdict for one criterion family.

    Definitions (stated so the verdict is reproducible):
      slope spread  := max(per-corruption slope)/min(per-corruption slope) - 1
      within +/-20% of the eta-sweep slope := |s_c/s_ref - 1| <= 0.20 per corruption
      linear per corruption := per-corruption R^2 >= 0.90 (needs >=3 points)
      x-spread := max/min of per-corruption mean ||g_bar|| at p_ref
    GREEN = spread <= 0.20 AND every corruption within +/-20% of the eta-sweep
            slope AND pooled R^2 >= 0.95 AND x-spread >= 1.5
    AMBER = linear per corruption AND 0.20 < spread <= 1.00
    RED   = spread > 1.00 (ratio > 2x) OR any corruption non-monotone in x
    """
    slopes = {c: info["fit"]["slope"]
              for c, info in fits["per_corruption"].items()
              if info["fit"] is not None}
    checks = {"family": family, "slopes": slopes}
    # PERMANENT RULE (Stage-1b audit, F2): no reference discoverable =>
    # verdict VOID, never a curve verdict.
    if ref_fit is None:
        checks["verdict"] = "VOID"
        checks["why"] = ("continual eta-sweep reference fit not discoverable "
                         "in the results dir — no curve verdict may be "
                         "emitted (permanent rule). Point the analysis at the "
                         "dir holding the eta-sweep JSONs "
                         "(--etasweep-results-dir).")
        return checks
    if len(slopes) < 2:
        checks["verdict"] = "UNRESOLVED"
        checks["why"] = (f"only {len(slopes)} corruption(s) have a fittable "
                         f"line (need >=2). Not enough usable p* points yet.")
        return checks

    vals = list(slopes.values())
    spread = (max(vals) / min(vals) - 1.0) if min(vals) > 0 else float("inf")
    pooled = fits["pooled"]["fit"]
    pooled_r2 = pooled["r2"] if pooled and pooled["r2"] is not None else None
    xspread = e1_xspread(fits)
    nonmono = [c for c, info in fits["per_corruption"].items()
               if info["monotone_in_x"] is False]
    linear_each = all(
        (info["fit"] is not None and info["fit"]["r2"] is not None
         and info["fit"]["r2"] >= E1_LINEAR_R2)
        for info in fits["per_corruption"].values() if info["n_usable"] >= 2)
    ref_slope = ref_fit["slope"] if ref_fit else None
    if ref_slope:
        ref_dev = {c: abs(s / ref_slope - 1.0) for c, s in slopes.items()}
        within_ref = all(d <= 0.20 for d in ref_dev.values())
    else:
        ref_dev, within_ref = {}, False

    checks.update({
        "slope_spread": spread,
        "pooled_r2": pooled_r2,
        "x_spread": xspread,
        "non_monotone_corruptions": nonmono,
        "linear_per_corruption": linear_each,
        "etasweep_ref_slope": ref_slope,
        "ref_slope_deviation": ref_dev,
    })

    if spread > E1_SLOPE_SPREAD_RED or nonmono:
        checks["verdict"] = "RED"
        checks["why"] = (f"spread={spread:.3g} (>1.0 means ratio >2x)"
                         + (f"; non-monotone: {nonmono}" if nonmono else ""))
    elif (spread <= E1_SLOPE_SPREAD_GREEN and within_ref
          and pooled_r2 is not None and pooled_r2 >= E1_POOLED_R2_GREEN
          and xspread is not None and xspread >= E1_XSPREAD_GREEN):
        checks["verdict"] = "GREEN"
        checks["why"] = (f"spread={spread:.3g}<=0.2, all within 20% of "
                         f"eta-sweep slope {_f(ref_slope)}, pooled "
                         f"R^2={pooled_r2:.3f}>={E1_POOLED_R2_GREEN}, "
                         f"x-spread={xspread:.2f}>={E1_XSPREAD_GREEN}")
    elif linear_each and E1_SLOPE_SPREAD_GREEN < spread <= E1_SLOPE_SPREAD_AMBER_HI:
        checks["verdict"] = "AMBER"
        checks["why"] = f"linear per corruption; spread={spread:.3g} in (0.2, 1.0]"
    else:
        checks["verdict"] = "AMBER" if linear_each and spread <= E1_SLOPE_SPREAD_GREEN \
            else "UNCLASSIFIED"
        fails = []
        if not linear_each:
            fails.append("a per-corruption fit has R^2 < 0.90")
        if not within_ref:
            fails.append("a slope deviates >20% from the eta-sweep slope")
        if pooled_r2 is None or pooled_r2 < E1_POOLED_R2_GREEN:
            fails.append(f"pooled R^2={_f(pooled_r2)} < {E1_POOLED_R2_GREEN}")
        if xspread is None or xspread < E1_XSPREAD_GREEN:
            fails.append(f"x-spread={_f(xspread)} < {E1_XSPREAD_GREEN}")
        checks["why"] = (
            ("spread <=0.2 but a GREEN side-condition failed: " + "; ".join(fails)
             + " — reported AMBER (linear, small spread).")
            if checks["verdict"] == "AMBER"
            else (f"spread={spread:.3g}; GREEN/AMBER/RED definitions not met "
                  "as written: " + "; ".join(fails)))
    return checks


# ============================================================================
# E2 — convergence of the running mean of grad_l2
# ============================================================================

def convergence_for_run(data, band=E2_BAND):
    """Per-block convergence of the running mean of grad_l2.

    Definition (documented in the artifact): within each corruption block,
    running_mean_t = mean(grad_l2[0..t]); S = the block's stationary value
    (mean over local steps 50-150, the codebase-standard window). The
    convergence step is the smallest t (1-indexed step count) such that
    |running_mean_t' - S| <= band*S for ALL t' >= t within the block. Blocks
    containing any NaN/inf are excluded (for collapsing runs this keeps
    pre-collapse blocks only). Returns {collapsed, block_steps, n_blocks_skipped}.
    """
    rows = pc.stream_rows(data)
    collapsed = pc.first_nonfinite_step(rows) is not None
    groups = group_rows_by_block(rows)
    block_steps, skipped = [], 0
    for _name, brows in groups.items():
        if pc.block_has_nonfinite(brows):
            skipped += 1
            continue
        ordered = sorted(brows, key=lambda r: (r.get("local_step") or 0))
        series = [row_metric(r, "grad_l2") for r in ordered]
        series = [v for v in series if v is not None]
        if len(series) < 10:
            skipped += 1
            continue
        stat = stationary_rows(ordered, 50, 150)
        svals = [row_metric(r, "grad_l2") for r in stat]
        svals = [v for v in svals if v is not None]
        if not svals:
            skipped += 1
            continue
        S = sum(svals) / len(svals)
        if S <= 0:
            skipped += 1
            continue
        rm, acc = [], 0.0
        for i, v in enumerate(series):
            acc += v
            rm.append(acc / (i + 1))
        violations = [i for i, v in enumerate(rm) if abs(v - S) > band * S]
        if not violations:
            conv = 1
        elif violations[-1] + 1 >= len(rm):
            conv = None  # never permanently enters the band
        else:
            conv = violations[-1] + 2  # 1-indexed first step of the final in-band run
        block_steps.append(conv)
    return {"collapsed": collapsed, "block_steps": block_steps,
            "n_blocks_skipped": skipped}


def collect_e2_runs(etasweep_dir, e1_dir, severity, seed):
    """(label, collapsing?, data) for every heat run on disk: continual
    eta-sweep runs (both archs) + E1 single-corruption runs."""
    out = []
    if etasweep_dir:
        for arch in CONTINUAL_ARCHS:
            for eta in pc.discover_etas(etasweep_dir, arch, severity, seed):
                for p in pc.discover_p_values(etasweep_dir, arch, severity, seed, eta):
                    path = pc.run_output_path(etasweep_dir, arch, severity, seed,
                                              p=p, eta=eta)
                    try:
                        data = pc.load_json(path)
                    except Exception:
                        continue
                    out.append((f"continual/{arch}/lr{pc.format_p(eta)}/p{pc.format_p(p)}",
                                data))
    if e1_dir:
        for corruption in sc.E1_CORRUPTIONS:
            for eta in sc.discover_etas_sc(e1_dir, sc.E1_ARCH, corruption,
                                           severity, seed):
                for p in sc.discover_p_values_sc(e1_dir, sc.E1_ARCH, corruption,
                                                 severity, seed, eta):
                    data = sc.load_run_sc(e1_dir, sc.E1_ARCH, corruption,
                                          severity, seed, p=p, eta=eta)
                    if data is None:
                        continue
                    out.append((f"e1/{corruption}/lr{pc.format_p(eta)}/p{pc.format_p(p)}",
                                data))
    return out


def e2_analysis(runs):
    per_run = []
    for label, data in runs:
        c = convergence_for_run(data)
        defined = [s for s in c["block_steps"] if s is not None]
        never = sum(1 for s in c["block_steps"] if s is None)
        if defined:
            med = sorted(defined)[len(defined) // 2]
        else:
            med = None
        per_run.append({"run": label, "collapsed": c["collapsed"],
                        "median_conv_step": med,
                        "n_blocks": len(c["block_steps"]),
                        "n_blocks_never_converged": never,
                        "n_blocks_skipped_nonfinite": c["n_blocks_skipped"],
                        "block_steps": c["block_steps"]})
    usable = [r for r in per_run if r["median_conv_step"] is not None
              or r["n_blocks_never_converged"] > 0]
    n_ok = sum(1 for r in usable
               if r["median_conv_step"] is not None
               and r["median_conv_step"] <= E2_CONV_STEPS)
    frac = (n_ok / len(usable)) if usable else None
    verdict = "UNRESOLVED"
    if frac is not None:
        verdict = "GREEN" if frac >= E2_GREEN_FRAC else "NOT-GREEN"
    return {"per_run": per_run, "n_runs_usable": len(usable),
            "n_runs_conv_le_50": n_ok, "fraction": frac,
            "criterion": f"GREEN iff >= {E2_GREEN_FRAC:.0%} of runs have "
                         f"median block convergence <= {E2_CONV_STEPS} steps "
                         f"(band = +/-{E2_BAND:.0%} of the stationary value)",
            "verdict": verdict}


def e2_plot(e2, out_png: Path):
    coll = [s for r in e2["per_run"] if r["collapsed"]
            for s in r["block_steps"] if s is not None]
    stab = [s for r in e2["per_run"] if not r["collapsed"]
            for s in r["block_steps"] if s is not None]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = list(range(0, 160, 5))
    if stab:
        ax.hist(stab, bins=bins, alpha=0.65, label=f"stable runs (n={len(stab)} blocks)",
                color="#1f77b4")
    if coll:
        ax.hist(coll, bins=bins, alpha=0.65,
                label=f"later-collapsing runs, pre-collapse blocks (n={len(coll)})",
                color="#d62728")
    ax.axvline(E2_CONV_STEPS, color="k", linestyle="--", linewidth=1.2,
               label=f"criterion: {E2_CONV_STEPS} steps")
    ax.set_xlabel("block-level convergence step of running-mean grad_l2 "
                  f"(within {E2_BAND:.0%} of stationary, permanently)")
    ax.set_ylabel("blocks")
    ax.set_title("E2: convergence of the ||g_bar|| running mean")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================================
# E3 — coupling index
# ============================================================================

def _coupling_from_points(stable_gbars: dict[float, float]):
    """coupling index := (gbar(p_max_stable) - gbar(p_min_stable)) /
    gbar(p_min_stable), over stable p with a finite gbar."""
    if len(stable_gbars) < 2:
        return None
    p_lo, p_hi = min(stable_gbars), max(stable_gbars)
    g_lo, g_hi = stable_gbars[p_lo], stable_gbars[p_hi]
    if not g_lo:
        return None
    return {"p_lo": p_lo, "p_hi": p_hi, "gbar_lo": g_lo, "gbar_hi": g_hi,
            "coupling_index": (g_hi - g_lo) / g_lo}


def e3_coupling(e1_dir, etasweep_dir, domainnet_dir, severity, seed,
                dn_chance):
    cells = []
    # CIFAR continual cells.
    if etasweep_dir:
        for arch in CONTINUAL_ARCHS:
            for eta in pc.discover_etas(etasweep_dir, arch, severity, seed):
                src = pc.run_output_path(etasweep_dir, arch, severity, seed,
                                         source=True, eta=eta)
                source_acc = None
                if src.exists():
                    try:
                        source_acc = pc.source_mean_acc(pc.load_json(src))
                    except Exception:
                        pass
                stable_gbars = {}
                for p in pc.discover_p_values(etasweep_dir, arch, severity, seed, eta):
                    path = pc.run_output_path(etasweep_dir, arch, severity, seed,
                                              p=p, eta=eta)
                    try:
                        data = pc.load_json(path)
                    except Exception:
                        continue
                    v = pc.classify_run(data, source_acc, pref_drift=None,
                                        use_drift_criterion=False)
                    if not v["collapsed"] and v["gbar"]:
                        stable_gbars[p] = v["gbar"]
                res = _coupling_from_points(stable_gbars)
                cells.append({"setting": f"C10C-CONT {arch}", "eta": eta,
                              "n_stable_p": len(stable_gbars), **(res or {})})
    # E1 single-corruption cells.
    if e1_dir:
        for corruption in sc.E1_CORRUPTIONS:
            for eta in sc.discover_etas_sc(e1_dir, sc.E1_ARCH, corruption,
                                           severity, seed):
                source_acc = sc.source_acc_sc(e1_dir, sc.E1_ARCH, corruption,
                                              severity, seed, eta)
                stable_gbars = {}
                for p in sc.discover_p_values_sc(e1_dir, sc.E1_ARCH, corruption,
                                                 severity, seed, eta):
                    data = sc.load_run_sc(e1_dir, sc.E1_ARCH, corruption,
                                          severity, seed, p=p, eta=eta)
                    if data is None:
                        continue
                    v = pc.classify_run(data, source_acc, pref_drift=None,
                                        use_drift_criterion=False)
                    if not v["collapsed"] and v["gbar"]:
                        stable_gbars[p] = v["gbar"]
                res = _coupling_from_points(stable_gbars)
                cells.append({"setting": f"E1 {sc.E1_ARCH} {corruption}",
                              "eta": eta, "n_stable_p": len(stable_gbars),
                              **(res or {})})
    # DomainNet clipart cells (hard-only stability; soft fires at all p there).
    if domainnet_dir:
        for eta in pc.discover_etas(domainnet_dir, "resnet50", severity, seed):
            stable_gbars = {}
            for p in pc.discover_p_values(domainnet_dir, "resnet50", severity,
                                          seed, eta):
                path = pc.run_output_path(domainnet_dir, "resnet50", severity,
                                          seed, p=p, eta=eta)
                try:
                    data = pc.load_json(path)
                except Exception:
                    continue
                v = pc.hard_collapse(data, dn_chance)
                if not v["collapsed"] and v["gbar"]:
                    stable_gbars[p] = v["gbar"]
            res = _coupling_from_points(stable_gbars)
            cells.append({"setting": "DomainNet-126 real->clipart resnet50 "
                                     "(hard-only)", "eta": eta,
                          "n_stable_p": len(stable_gbars), **(res or {})})
    return cells


def e3_markdown(cells) -> str:
    lines = ["# E3 — coupling index of stationary ||g_bar|| with p", "",
             "coupling index := (||g_bar||(p_max_stable) - ||g_bar||(p_min_stable))"
             " / ||g_bar||(p_min_stable), over the stable p range of each cell "
             "(stability: hard + below-source for CIFAR; hard-only for DomainNet).",
             "",
             "| setting | eta | n stable p | p_lo | p_hi | gbar(p_lo) | "
             "gbar(p_hi) | coupling index |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append("| " + " | ".join([
            c["setting"], _f(c.get("eta")), str(c.get("n_stable_p")),
            _f(c.get("p_lo")), _f(c.get("p_hi")),
            _f(c.get("gbar_lo"), 4), _f(c.get("gbar_hi"), 4),
            _f(c.get("coupling_index"), 3),
        ]) + " |")
    if not cells:
        lines.append("(no cells found)")
    return "\n".join(lines) + "\n"


# ============================================================================
# E4/E5 — zero-shot p on held-out cells
# ============================================================================

def _snap(p_hat, grid):
    """Nearest measured grid p; ties resolve to the LARGER p (safer tether)."""
    if not grid:
        return None
    best = min(grid, key=lambda q: (abs(q - p_hat), -q))
    return best


def e4_zeroshot(rows_soft, fits_soft, delta=0.0):
    """Zero-shot evaluation on held-out cells for one R perturbation.

    slope is fit on CALIBRATION cells only; the law is applied through the
    origin: p_hat = slope * eta * ||g_bar|| / (1 + delta), i.e. R' = R*(1+delta).
    Snapped to the nearest measured grid p of that cell. Oracle = the highest
    mean accuracy over ALL valid measured p of the cell (its p and stability
    are reported)."""
    calib_fit = fits_soft["calibration_only"]["fit"]
    if calib_fit is None:
        return {"error": "calibration fit unavailable (need >=2 usable "
                         "calibration cells)", "cells": []}
    slope = calib_fit["slope"]
    out = []
    for r in rows_soft:
        if r["corruption"] not in sc.HELDOUT_CORRUPTIONS:
            continue
        x = r.get("eta_times_gbar")
        grid = r.get("p_values_run") or []
        per_p = r.get("per_p") or {}
        if x is None or not grid:
            out.append({"corruption": r["corruption"], "eta": r["eta"],
                        "error": "missing gbar or measured grid"})
            continue
        p_hat_raw = slope * x / (1.0 + delta)
        p_hat = _snap(p_hat_raw, grid)
        info = per_p.get(pc.format_p(p_hat), {})
        acc_hat = info.get("mean_acc")
        collapsed_hat = bool(info.get("collapsed"))
        # Oracle over the measured grid.
        best_p, best_acc, best_stable = None, None, None
        for p in grid:
            q = per_p.get(pc.format_p(p), {})
            if not q.get("valid"):
                continue
            a = q.get("mean_acc")
            if a is not None and (best_acc is None or a > best_acc):
                best_p, best_acc, best_stable = p, a, (not q.get("collapsed"))
        ratio = (acc_hat / best_acc) if (acc_hat is not None and best_acc) else None
        out.append({
            "corruption": r["corruption"], "eta": r["eta"], "x_eta_gbar": x,
            "delta_R": delta, "p_hat_raw": p_hat_raw, "p_hat_snapped": p_hat,
            "acc_at_p_hat": acc_hat, "collapsed_at_p_hat": collapsed_hat,
            "criterion_at_p_hat": info.get("criterion"),
            "oracle_p": best_p, "oracle_acc": best_acc,
            "oracle_p_stable": best_stable, "acc_ratio": ratio,
        })
    return {"slope_calibration": slope, "R_calibration": (1.0 / slope) if slope else None,
            "delta_R": delta, "cells": out}


def e4_verdict(e4):
    cells = [c for c in e4.get("cells", []) if "error" not in c]
    if not cells:
        return {"verdict": "UNRESOLVED", "why": "no evaluable held-out cells"}
    green = [c for c in cells
             if c["acc_ratio"] is not None and c["acc_ratio"] >= E4_GREEN_RATIO
             and not c["collapsed_at_p_hat"]]
    amber = [c for c in cells
             if c["acc_ratio"] is not None and c["acc_ratio"] >= E4_AMBER_RATIO
             and not c["collapsed_at_p_hat"]]
    n = len(cells)
    fg, fa = len(green) / n, len(amber) / n
    if fg >= E4_CELL_FRAC:
        v = "GREEN"
    elif fa >= E4_CELL_FRAC:
        v = "AMBER"
    else:
        v = "NOT-GREEN"
    return {"verdict": v,
            "why": (f"{len(green)}/{n} cells with acc>= {E4_GREEN_RATIO:.0%} of "
                    f"oracle and no collapse; {len(amber)}/{n} at the "
                    f">={E4_AMBER_RATIO:.0%} bar. Criterion: GREEN iff "
                    f">={E4_CELL_FRAC:.0%} of held-out cells meet the 95% bar "
                    f"with zero collapses; AMBER for 85-95%.")}


def e5_verdict(e5_by_delta):
    plus50 = e5_by_delta.get(0.50, {}).get("cells", [])
    minus25 = e5_by_delta.get(-0.25, {}).get("cells", [])
    p50_ok = plus50 and all(not c.get("collapsed_at_p_hat") for c in plus50
                            if "error" not in c)
    m25 = [c for c in minus25 if "error" not in c]
    m25_nocoll = m25 and all(not c.get("collapsed_at_p_hat") for c in m25)
    ratios = sorted(c["acc_ratio"] for c in m25 if c.get("acc_ratio") is not None)
    med = ratios[len(ratios) // 2] if ratios else None
    graceful = m25_nocoll and med is not None and med >= E5_GRACEFUL_RATIO
    v = "GREEN" if (p50_ok and graceful) else ("UNRESOLVED" if not plus50 else "NOT-GREEN")
    return {"verdict": v,
            "why": (f"+50% R error: {'no collapse' if p50_ok else 'COLLAPSE or no data'}. "
                    f"-25%: {'no collapse' if m25_nocoll else 'COLLAPSE or no data'}, "
                    f"median acc ratio={_f(med, 4)} "
                    f"(graceful operationalized as no collapse AND median "
                    f"ratio >= {E5_GRACEFUL_RATIO}).")}


def zeroshot_markdown(e4, e5_by_delta) -> str:
    lines = ["# E4/E5 — zero-shot p on held-out corruptions", "",
             f"slope (calibration cells only, soft+hard family): "
             f"{_f(e4.get('slope_calibration'))}  =>  R = {_f(e4.get('R_calibration'))}",
             "p_hat = slope * eta * ||g_bar|| (through origin), snapped to the "
             "nearest measured grid p (ties -> larger p).", ""]
    hdr = ("| corruption | eta | x=eta*gbar | p_hat raw | p_hat snapped | "
           "acc(p_hat) | collapse@p_hat | oracle p | oracle acc | ratio |")
    sep = "|" + "---|" * 10
    lines += ["## E4 (delta_R = 0)", "", hdr, sep]
    for c in e4.get("cells", []):
        if "error" in c:
            lines.append(f"| {c['corruption']} | {_f(c.get('eta'))} | "
                         f"ERROR: {c['error']} |" + " |" * 7)
            continue
        lines.append("| " + " | ".join([
            c["corruption"], _f(c["eta"]), _f(c["x_eta_gbar"], 4),
            _f(c["p_hat_raw"], 4), _f(c["p_hat_snapped"]),
            _f(c["acc_at_p_hat"], 4),
            (c.get("criterion_at_p_hat") or "-") if c["collapsed_at_p_hat"] else "no",
            _f(c["oracle_p"]), _f(c["oracle_acc"], 4), _f(c["acc_ratio"], 4),
        ]) + " |")
    lines += ["", "## E5 (R perturbed; p_hat = slope*x/(1+delta))", "",
              "| delta_R | corruption | eta | p_hat snapped | acc(p_hat) | "
              "collapse@p_hat | ratio |", "|" + "---|" * 7]
    for delta in E5_DELTAS:
        for c in e5_by_delta.get(delta, {}).get("cells", []):
            if "error" in c:
                continue
            lines.append("| " + " | ".join([
                f"{delta:+.0%}", c["corruption"], _f(c["eta"]),
                _f(c["p_hat_snapped"]), _f(c["acc_at_p_hat"], 4),
                "YES" if c["collapsed_at_p_hat"] else "no", _f(c["acc_ratio"], 4),
            ]) + " |")
    return "\n".join(lines) + "\n"


# ============================================================================
# Plot (E1)
# ============================================================================

CORRUPTION_COLORS = {
    "gaussian_noise": "#1f77b4", "elastic_transform": "#2ca02c",
    "impulse_noise": "#d62728", "contrast": "#9467bd",
}


def e1_plot(rows_soft, fits_soft, rows_hard, fits_hard, ref_fit_soft,
            ref_fit_hard, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=False)
    for ax, rows, fits, ref_fit, title in (
        (axes[0], rows_soft, fits_soft, ref_fit_soft, "soft+hard criterion"),
        (axes[1], rows_hard, fits_hard, ref_fit_hard, "hard-only criterion"),
    ):
        xmax = 0.0
        for c in sc.E1_CORRUPTIONS:
            col = CORRUPTION_COLORS.get(c)
            crows = [r for r in rows if r["corruption"] == c]
            usable = [r for r in crows if r.get("eta_times_gbar") is not None
                      and r.get("p_star") is not None]
            xs = [r["eta_times_gbar"] for r in usable]
            ys = [r["p_star"] for r in usable]
            held = c in sc.HELDOUT_CORRUPTIONS
            if xs:
                xmax = max(xmax, max(xs))
                ax.scatter(xs, ys, s=70, zorder=3, color=col,
                           facecolors="none" if held else col,
                           label=f"{c}{' (held-out)' if held else ''}")
                for x, y, r in zip(xs, ys, usable):
                    ax.annotate(f"η={r['eta']:g}", (x, y),
                                textcoords="offset points", xytext=(5, 4),
                                fontsize=7, color=col)
            censored = [r for r in crows if r.get("p_star") is None
                        and r.get("eta_times_gbar") is not None]
            if censored:
                cx = [r["eta_times_gbar"] for r in censored]
                cy = [r.get("p_star_bracket_low") or 0.0 for r in censored]
                ax.scatter(cx, cy, marker="^", s=100, facecolors="none",
                           edgecolors=col, linewidths=1.5, zorder=3)
            fit = fits["per_corruption"][c]["fit"]
            if fit is not None and xs:
                xx = [0.0, max(xs) * 1.15]
                ax.plot(xx, [fit["slope"] * x + fit["intercept"] for x in xx],
                        color=col, linestyle="--", linewidth=1.1, zorder=2)
        pooled = fits["pooled"]["fit"]
        if pooled is not None and xmax > 0:
            xx = [0.0, xmax * 1.15]
            r2 = pooled["r2"]
            ax.plot(xx, [pooled["slope"] * x + pooled["intercept"] for x in xx],
                    color="k", linewidth=1.8, zorder=2,
                    label=(f"pooled: slope={pooled['slope']:.3g}, "
                           f"R²={r2:.3f}" if r2 is not None else "pooled"))
        if ref_fit is not None and xmax > 0:
            xx = [0.0, xmax * 1.15]
            ax.plot(xx, [ref_fit["slope"] * x + ref_fit["intercept"] for x in xx],
                    color="0.5", linewidth=1.4, linestyle=":", zorder=1,
                    label=f"continual eta-sweep: slope={ref_fit['slope']:.3g}")
        ax.set_xlabel(r"$\eta \cdot \|\bar{g}\|$")
        ax.set_ylabel(r"$p^*$")
        ax.set_title(f"E1 ||g_bar||-factor test — {title}")
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================================
# Table / verdict output
# ============================================================================

def print_cell_table(rows, family):
    cols = ["corruption", "eta", "source_acc", "p_ref_used", "grad_norm_gbar",
            "eta_times_gbar", "p_star", "p_star_bracket_low",
            "p_star_bracket_high", "collapse_criterion", "stable_mean_acc"]
    print(f"\n--- E1 cell table ({family}) ---")
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join(_f(r.get(c), 5) for c in cols) + " |")


def fits_lines(fits, family):
    out = [f"fits ({family}):"]
    for c, info in fits["per_corruption"].items():
        f = info["fit"]
        if f is None:
            out.append(f"  {c:18s} n={info['n_usable']}  fit=n/a")
        else:
            out.append(f"  {c:18s} n={info['n_usable']}  slope={f['slope']:.4g} "
                       f"intercept={f['intercept']:.3g} R^2={_f(f['r2'], 4)} "
                       f"mean||g_bar||={_f(info['mean_gbar'], 4)}")
    for k in ("pooled", "calibration_only"):
        f = fits[k]["fit"]
        out.append(f"  {k:18s} n={fits[k]['n_usable']}  "
                   + ("fit=n/a" if f is None else
                      f"slope={f['slope']:.4g} intercept={f['intercept']:.3g} "
                      f"R^2={_f(f['r2'], 4)}"))
    return out


def build_verdict_md(anchor, e1_soft, e1_hard, e2, e3_cells, e4, e4v,
                     e5_by_delta, e5v, deviations):
    L = ["# Stage 1 verdict", ""]
    L += ["## Regression anchor (WRN sev5 continual eta=1e-3)",
          f"- status: **{anchor['status']}** — {anchor['detail']}"]
    if anchor["status"] == "MOVED-STOP":
        L += ["", "**STOP: the regression anchor moved. Per the master plan, "
              "halt and report before trusting any Stage-1 number.**"]
    L += ["", "## E1 — ||g_bar||-factor test",
          "Criterion (verbatim): GREEN = per-corruption slopes within +/-20% of "
          "each other AND of the eta-sweep slope (same criterion family), pooled "
          "R^2 >= 0.95, x-spread of corruption positions >= 1.5x. AMBER = linear "
          "per corruption, slope spread 20-100%. RED = spread > 2x or non-monotone.",
          f"- soft+hard family: **{e1_soft.get('verdict')}** — {e1_soft.get('why')}",
          f"- hard-only family: **{e1_hard.get('verdict')}** — {e1_hard.get('why')}",
          "- (soft and hard fitted separately, never averaged; the soft+hard "
          "family is the primary verdict, matching the eta-sweep's default "
          "criterion family.)"]
    L += ["", "## E2 — convergence of the ||g_bar|| running mean",
          f"Criterion (verbatim): GREEN = convergence <= 50 steps for >= 90% of runs.",
          f"- {e2['criterion']}",
          f"- result: **{e2['verdict']}** — {e2['n_runs_conv_le_50']}/"
          f"{e2['n_runs_usable']} usable runs "
          f"({_f((e2['fraction'] or 0) * 100, 4)}%) converged <= {E2_CONV_STEPS} steps."]
    L += ["", "## E3 — coupling index",
          f"- {len(e3_cells)} cells tabulated in stage1_coupling.md "
          "(no pass/fail criterion; descriptive)."]
    L += ["", "## E4 — zero-shot p on held-out cells",
          "Criterion (verbatim): GREEN = zero-shot >= 95% of oracle, zero "
          "collapses, on >= 80% of held-out cells; AMBER 85-95%.",
          f"- result: **{e4v['verdict']}** — {e4v['why']}"]
    L += ["", "## E5 — R-perturbation robustness",
          "Criterion (verbatim): GREEN = no collapse at +50% R error, graceful "
          "at -25%.",
          f"- result: **{e5v['verdict']}** — {e5v['why']}"]
    L += ["", "## Deviations from the plan"]
    L += [f"- {d}" for d in deviations] if deviations else ["- none"]
    L += ["", "## Gate",
          "Stage 1b may run interleaved/after E1 on the same GPU. Stages 2-4 "
          "open only if E1 is GREEN or AMBER.",
          "", "**STOP — Stage 1 report complete; awaiting instruction before "
          "the next stage.**"]
    return "\n".join(L) + "\n"


# ============================================================================
# Self-test (synthetic; no GPU / no results dir)
# ============================================================================

def _synthetic_run(arch, corruption, severity, seed, eta, p, *, gbar, R,
                   n_steps=157, source_acc=0.70):
    """Synthetic run JSON matching run_tier2 p9 output. Collapses (NaN at
    step 100 + chance acc) iff p < eta*gbar/R; otherwise stable with accuracy
    declining gently in p."""
    p_star_true = eta * gbar / R
    collapsed = p < p_star_true - 1e-12
    rows = []
    for t in range(n_steps):
        g = gbar * (1.0 + (0.5 if t < 10 else 0.02 * ((t % 7) - 3) / 3.0))
        row = {"step": t, "local_step": t, "corruption": corruption,
               "restore_prob": p, "grad_l2": g, "total_grad_norm": g,
               "drift_l2": 1.0 + 0.01 * t, "energy": -5.0}
        if collapsed and t >= 100:
            row["grad_l2"] = float("nan")
            row["total_grad_norm"] = float("nan")
        rows.append(row)
    acc = 0.05 if collapsed else max(0.2, source_acc + 0.08 - 0.5 * (p - p_star_true))
    summ = {"mean_accuracy": acc, "last_accuracy": acc, "peak_accuracy": acc,
            "peak_at_corruption": corruption, "forgetting": 0.0,
            "mean_ece": 0.05, "num_diagnostic_steps": n_steps,
            "stream_diagnostics": rows}
    return {"args": {"arch": arch, "dataset": "cifar10", "severity": severity,
                     "seed": seed, "heat_lr": eta, "heat_restore_prob": p,
                     "corruptions": [corruption]},
            "results": {"per_corruption": {"heat": {corruption: {
                            "accuracy": acc, "ece": 0.05,
                            "mean_confidence": 0.8, "mean_entropy": 0.5}}},
                        "summary": {"heat": summ}}}


def _synthetic_source(corruption, source_acc=0.70):
    return {"args": {"dataset": "cifar10"},
            "results": {"summary": {"source": {"mean_accuracy": source_acc}}}}


def self_test():
    import tempfile
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("=== stage1 self-test ===")
    # 1. Tag round-trip and filename convention.
    tag = sc.variant_tag_sc("gaussian_noise", p=0.0075, eta=2e-3)
    check("variant tag", tag == "pstar_gaussian_noise_lr0.002_p0.0075", tag)
    path = sc.run_output_path_sc("/r", "wrn28_10", "gaussian_noise", 5, 42,
                                 p=0.0075, eta=2e-3)
    check("filename", path.name ==
          "p9_wrn28_10_pstar_gaussian_noise_lr0.002_p0.0075_seed42_sev5.json",
          path.name)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # 2. Old/new discovery isolation.
        (d / "p9_wrn28_10_pstar_lr0.001_p0.005_seed42_sev5.json").write_text("{}")
        (d / "p9_wrn28_10_pstar_lr0.001_source_seed42_sev5.json").write_text("{}")
        (d / "p9_wrn28_10_pstar_gaussian_noise_lr0.001_p0.0075_seed42_sev5.json"
         ).write_text("{}")
        (d / "p9_wrn28_10_pstar_gaussian_noise_lr0.001_source_seed42_sev5.json"
         ).write_text("{}")
        (d / "p9_wrn28_10_pstar_contrast_lr0.001_p0.02_seed42_sev5.json"
         ).write_text("{}")
        old = pc.discover_p_values(d, "wrn28_10", 5, 42, 1e-3)
        check("old discovery ignores corruption-tagged files", old == [0.005], old)
        new_g = sc.discover_p_values_sc(d, "wrn28_10", "gaussian_noise", 5, 42, 1e-3)
        check("new discovery (gaussian_noise)", new_g == [0.0075], new_g)
        new_c = sc.discover_p_values_sc(d, "wrn28_10", "contrast", 5, 42, 1e-3)
        check("new discovery (contrast)", new_c == [0.02], new_c)
        check("old eta discovery unaffected",
              pc.discover_etas(d, "wrn28_10", 5, 42) == [0.001])
        check("new eta discovery per corruption",
              sc.discover_etas_sc(d, "wrn28_10", "gaussian_noise", 5, 42) == [0.001])

        # 3. Command construction.
        cmd = sc.build_run_command_sc(
            run_tier2=Path("scripts/run_tier2.py"), arch="wrn28_10",
            checkpoint="ckpt.pt", corruption="gaussian_noise", severity=5,
            seed=42, results_dir=d, c10c_root="data/cifar10c", heat_lr=2e-3,
            p=0.0075)
        check("--corruptions in command", "--corruptions" in cmd and
              cmd[cmd.index("--corruptions") + 1] == "gaussian_noise")
        check("variant tag in command",
              cmd[cmd.index("--variant-tag") + 1] ==
              "pstar_gaussian_noise_lr0.002_p0.0075")
        check("diagnostic snapshot flag present",
              "--heat-diagnostic-snapshot" in cmd)

    # 4. End-to-end synthetic campaign: law with R=12, per-corruption gbar.
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for c, gbar in GBARS.items():
            for eta in sc.E1_ETAS:
                sp = sc.run_output_path_sc(d, sc.E1_ARCH, c, 5, 42,
                                           source=True, eta=eta)
                sp.write_text(json.dumps(_synthetic_source(c)))
                grid = sorted(set(pc.scaled_p_grid(eta)
                                  + pc.scaled_pref_ladder(sc.E1_ARCH, eta)))
                # add bisection-like points around the true boundary
                pst = eta * gbar / R_TRUE
                grid += [round(pst * 0.9, 6), round(pst * 1.1, 6)]
                for p in sorted(set(grid)):
                    rp = sc.run_output_path_sc(d, sc.E1_ARCH, c, 5, 42,
                                               p=p, eta=eta)
                    rp.write_text(json.dumps(_synthetic_run(
                        sc.E1_ARCH, c, 5, 42, eta, p, gbar=gbar, R=R_TRUE)))

        rows_soft = e1_all_rows(d, 5, 42, hard_only=False)
        fits_soft = e1_fits(rows_soft)
        pooled = fits_soft["pooled"]["fit"]
        check("pooled fit exists", pooled is not None)
        if pooled:
            check("pooled slope ~ 1/R",
                  abs(pooled["slope"] * R_TRUE - 1.0) < 0.15,
                  f"slope={pooled['slope']:.4g} (expect ~{1/R_TRUE:.4g})")
            check("pooled R^2 >= 0.95", pooled["r2"] >= 0.95,
                  f"R^2={pooled['r2']:.4f}")
        xs = e1_xspread(fits_soft)
        check("x-spread computed >= 1.5", xs is not None and xs >= 1.5, _f(xs))
        ref_fit = {"slope": 1.0 / R_TRUE, "intercept": 0.0, "r2": 1.0, "n": 5}
        v = e1_verdict(fits_soft, ref_fit, "soft+hard")
        check("E1 verdict GREEN on law-generated data", v["verdict"] == "GREEN",
              f"{v['verdict']} — {v.get('why')}")

        # E2 on the synthetic runs (converging series by construction).
        runs = collect_e2_runs("", d, 5, 42)
        check("E2 collected runs", len(runs) > 0, str(len(runs)))
        e2 = e2_analysis(runs)
        check("E2 verdict GREEN on synthetic", e2["verdict"] == "GREEN",
              f"frac={_f(e2['fraction'])}")

        # E4 zero-shot.
        e4 = e4_zeroshot(rows_soft, fits_soft, delta=0.0)
        cells = [c for c in e4["cells"] if "error" not in c]
        check("E4 evaluated 6 held-out cells", len(cells) == 6, str(len(cells)))
        e4v = e4_verdict(e4)
        check("E4 verdict computed", e4v["verdict"] in
              ("GREEN", "AMBER", "NOT-GREEN"), e4v["verdict"])

        # E3 coupling table renders.
        cells3 = e3_coupling(d, "", "", 5, 42, 0.02)
        check("E3 cells found", len(cells3) > 0, str(len(cells3)))

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}")
    return 0 if not failures else 1


# ============================================================================
# main
# ============================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.results_dir:
        raise SystemExit("--results-dir is required (or use --self-test)")

    results_dir = Path(args.results_dir)
    etasweep_dir = Path(args.etasweep_results_dir) if args.etasweep_results_dir \
        else results_dir
    dn_dir = Path(args.domainnet_results_dir) if args.domainnet_results_dir else ""
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    deviations = [
        "Single-corruption stream = one corruption's full severity-5 split "
        "(~157 steps at batch 64), per the plan. Note: the continual WRN "
        "sev5 eta=1e-3 hard collapse historically appears at step ~603, so "
        "single-corruption boundaries may be set by SOFT criteria within 157 "
        "steps; the hard-only panel makes this visible.",
    ]

    # --- Continual reference (overlay + anchor). ---
    cont_rows_soft, ref_fit_soft = continual_reference(
        etasweep_dir, "wrn28_10", args.severity, args.seed, hard_only=False)
    cont_rows_hard, ref_fit_hard = continual_reference(
        etasweep_dir, "wrn28_10", args.severity, args.seed, hard_only=True)
    anchor = anchor_check(cont_rows_soft)
    print(f"[anchor] {anchor['status']}: {anchor['detail']}")
    if anchor["status"] == "MOVED-STOP":
        print("[anchor] *** STOP: regression anchor moved — report before "
              "trusting anything below. ***")

    # --- E1 ---
    rows_soft = e1_all_rows(results_dir, args.severity, args.seed, hard_only=False)
    rows_hard = e1_all_rows(results_dir, args.severity, args.seed, hard_only=True)
    fits_soft = e1_fits(rows_soft)
    fits_hard = e1_fits(rows_hard)
    v_soft = e1_verdict(fits_soft, ref_fit_soft, "soft+hard")
    v_hard = e1_verdict(fits_hard, ref_fit_hard, "hard_only")

    print_cell_table(rows_soft, "soft+hard")
    print_cell_table(rows_hard, "hard-only")
    for line in fits_lines(fits_soft, "soft+hard") + fits_lines(fits_hard, "hard-only"):
        print(line)

    e1_plot(rows_soft, fits_soft, rows_hard, fits_hard, ref_fit_soft,
            ref_fit_hard, analysis_dir / "stage1_gbar_law.png")

    # --- E2 ---
    runs = collect_e2_runs(etasweep_dir, results_dir, args.severity, args.seed)
    e2 = e2_analysis(runs)
    e2_plot(e2, analysis_dir / "stage1_convergence.png")
    print(f"\n[E2] runs={len(runs)} usable={e2['n_runs_usable']} "
          f"conv<= {E2_CONV_STEPS}: {e2['n_runs_conv_le_50']} "
          f"({_f((e2['fraction'] or 0) * 100, 4)}%) -> {e2['verdict']}")

    # --- E3 ---
    e3_cells = e3_coupling(results_dir, etasweep_dir, dn_dir, args.severity,
                           args.seed, args.domainnet_chance_acc)
    md3 = e3_markdown(e3_cells)
    (analysis_dir / "stage1_coupling.md").write_text(md3, encoding="utf-8")
    print("\n" + md3)
    if not dn_dir:
        deviations.append("E3: DomainNet clipart cells not tabulated in this "
                          "invocation (--domainnet-results-dir not given); "
                          "CIFAR cells only. Re-run with the flag on the box "
                          "that has the DomainNet Drive dir.")

    # --- E4/E5 ---
    e4 = e4_zeroshot(rows_soft, fits_soft, delta=0.0)
    e4v = e4_verdict(e4)
    e5_by_delta = {dlt: e4_zeroshot(rows_soft, fits_soft, delta=dlt)
                   for dlt in E5_DELTAS}
    e5v = e5_verdict(e5_by_delta)
    md45 = zeroshot_markdown(e4, e5_by_delta)
    (analysis_dir / "stage1_zeroshot.md").write_text(md45, encoding="utf-8")
    print("\n" + md45)

    # --- artifacts ---
    law_json = {
        "anchor_check": anchor,
        "e1": {"rows_soft": rows_soft, "rows_hard": rows_hard,
               "fits_soft": fits_soft, "fits_hard": fits_hard,
               "verdict_soft": v_soft, "verdict_hard": v_hard,
               "continual_ref_fit_soft": ref_fit_soft,
               "continual_ref_fit_hard": ref_fit_hard},
        "e2": {k: v for k, v in e2.items() if k != "per_run"},
        "e2_per_run": e2["per_run"],
        "e3": e3_cells,
        "e4": e4, "e4_verdict": e4v,
        "e5": e5_by_delta, "e5_verdict": e5v,
        "deviations": deviations,
    }
    (analysis_dir / "stage1_gbar_law.json").write_text(
        json.dumps(law_json, indent=2, default=str), encoding="utf-8")

    verdict_md = build_verdict_md(anchor, v_soft, v_hard, e2, e3_cells, e4,
                                  e4v, e5_by_delta, e5v, deviations)
    (analysis_dir / "stage1_verdict.md").write_text(verdict_md, encoding="utf-8")
    print("\n================ STAGE 1 VERDICT ================\n")
    print(verdict_md)
    for name in ("stage1_gbar_law.json", "stage1_gbar_law.png",
                 "stage1_convergence.png", "stage1_coupling.md",
                 "stage1_zeroshot.md", "stage1_verdict.md"):
        print(f"[saved] {analysis_dir / name}")


if __name__ == "__main__":
    main()
