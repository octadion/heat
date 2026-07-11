"""
STAGE 1 RESTART (v2) — analyses E1-E5 + mechanical verdict on the CYCLED-x15
campaign. heat==TFF. Zero GPU; single source of truth; recomputes everything
from the run JSONs on disk.

- Fits p* vs eta*||g_bar||: per-corruption + pooled + calibration-only,
  HARD family primary and SOFT family separately (never averaged).
- Overlay/reference: the on-disk CONTINUAL law recomputed fresh
  (bracket-weighted hard-only WLS; expected ~1.152 +/- 0.022), clearly
  labeled "continual (independent reference)". Zero discoverable reference
  => the E1 verdict is VOID (permanent rule), never a curve verdict.
- E2 online ||g_bar|| convergence (cycle-aware blocks).
- E3 p -> ||g_bar|| coupling index across each cell's stable p range.
- E4 zero-shot p on held-out corruptions vs grid oracle; S is RE-FROZEN fresh
  from THIS campaign's calibration cells (hard family) into
  analysis/stage1v2_S_frozen.json — nothing from the voided runs seeds
  anything; the artifact embeds the consumed-file manifest and is never
  written null.
- E5 R-sensitivity ({-50,-25,+25,+50}%).
- VERDICT (mechanical, original spec): E1 GREEN = per-corruption slopes
  within +/-20% of each other AND of the reference slope (same criterion
  family), pooled R^2 >= 0.95, x-spread >= 1.5x; AMBER = linear per
  corruption, spread 20-100%; RED = spread > 2x or non-monotone. E2 GREEN =
  convergence <= 50 steps for >= 90% of runs. E4 GREEN = zero-shot >= 95% of
  oracle, zero collapses, on >= 80% of held-out cells (AMBER 85-95%). E5
  GREEN = no collapse at +50% R error, graceful at -25%.

Every JSON artifact embeds the manifest (paths + sha256) of the run files it
consumed (STEP 2b).

Usage:
  python scripts/analyze_stage1_v2.py --results-dir <Drive dir>
  python scripts/analyze_stage1_v2.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from scripts import pstar_common as pc
from scripts import stage1v2_common as v2
from scripts.analysis_common import stationary_rows, row_metric
from scripts.analyze_stage1 import (
    e1_fits, e1_xspread, e1_verdict, e1_plot, e2_plot, e2_analysis,
    e4_zeroshot, e4_verdict, e5_verdict, zeroshot_markdown,
    print_cell_table, fits_lines, E2_CONV_STEPS, E2_BAND, E5_DELTAS,
)
from scripts.analyze_pstar_law import analyze_cell as analyze_cell_continual
from scripts.analyze_stage1_followup import wls_line


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 RESTART analyses (v2)")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--severity", type=int, default=v2.V2_SEVERITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cycles", type=int, default=v2.CYCLES)
    p.add_argument("--etas", type=float, nargs="+", default=list(v2.V2_ETAS))
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# ----------------------------------------------------------------------------
# Rows (schema-compatible with analyze_stage1 machinery).
# ----------------------------------------------------------------------------

def v2_all_rows(results_dir, severity, seed, etas, cycles, hard_only):
    rows = []
    for corruption in v2.V2_CORRUPTIONS:
        for eta in sorted(set(etas)):
            rows.append(v2.analyze_cell_v2(results_dir, v2.V2_ARCH, corruption,
                                           severity, eta, seed,
                                           hard_only=hard_only, cycles=cycles))
    return rows


def continual_reference(results_dir, severity, seed, hard_only):
    """Bracket-weighted WLS over the on-disk continual runs — the independent
    reference slope (expected hard-only ~1.152 +/- 0.022). Returns a fit dict
    or None (=> VOID downstream)."""
    etas = pc.discover_etas(results_dir, "wrn28_10", severity, seed)
    pts = []
    for eta in etas:
        r = analyze_cell_continual(results_dir, "wrn28_10", severity, eta,
                                   seed, hard_only=hard_only)
        lo, hi = r["p_star_bracket_low"], r["p_star_bracket_high"]
        if (r["p_star"] is not None and r["eta_times_gbar"] is not None
                and lo is not None and hi is not None and hi > lo):
            pts.append((r["eta_times_gbar"], r["p_star"], hi - lo))
    if len(pts) < 2:
        return None
    w = wls_line([x for x, _, _ in pts], [y for _, y, _ in pts],
                 [1.0 / wd ** 2 for _, _, wd in pts])
    if w is None:
        return None
    return {"slope": w["slope"], "intercept": w["intercept"],
            "r2": w["r2_weighted"], "sigma_slope": w["sigma_slope"],
            "n": w["n"], "label": "continual (independent reference, "
                                  "bracket-weighted WLS)"}


# ----------------------------------------------------------------------------
# E2 — cycle-aware convergence.
# ----------------------------------------------------------------------------

def convergence_for_run_v2(data, band=E2_BAND):
    """Per CYCLE-BLOCK convergence of the running mean of grad_l2 (same
    definition as v1: smallest t s.t. |running_mean - stationary| <= band*S
    for all t' >= t within the block; NaN cycles excluded)."""
    rows = pc.stream_rows(data)
    collapsed = pc.first_nonfinite_step(rows) is not None
    block_steps, skipped = [], 0
    for cyc in v2.group_rows_by_cycle(rows):
        if pc.block_has_nonfinite(cyc):
            skipped += 1
            continue
        series = [row_metric(r, "grad_l2") for r in cyc]
        series = [x for x in series if x is not None]
        if len(series) < 10:
            skipped += 1
            continue
        stat = stationary_rows(cyc, v2.STATIONARY_LO, v2.STATIONARY_HI)
        svals = [row_metric(r, "grad_l2") for r in stat]
        svals = [x for x in svals if x is not None]
        if not svals or sum(svals) <= 0:
            skipped += 1
            continue
        S = sum(svals) / len(svals)
        rm, acc = [], 0.0
        for i, x in enumerate(series):
            acc += x
            rm.append(acc / (i + 1))
        viol = [i for i, x in enumerate(rm) if abs(x - S) > band * S]
        if not viol:
            conv = 1
        elif viol[-1] + 1 >= len(rm):
            conv = None
        else:
            conv = viol[-1] + 2
        block_steps.append(conv)
    return {"collapsed": collapsed, "block_steps": block_steps,
            "n_blocks_skipped": skipped}


def collect_e2_runs_v2(results_dir, severity, seed, etas, cycles):
    out = []
    for corruption in v2.V2_CORRUPTIONS:
        for eta in sorted(set(etas)):
            for p in v2.discover_p_values_v2(results_dir, v2.V2_ARCH,
                                             corruption, severity, seed, eta,
                                             cycles=cycles):
                data = v2.load_run_v2(results_dir, v2.V2_ARCH, corruption,
                                      severity, seed, p=p, eta=eta,
                                      cycles=cycles)
                if data is not None:
                    out.append((f"e1v2/{corruption}/lr{pc.format_p(eta)}"
                                f"/p{pc.format_p(p)}", data))
    return out


def e2_analysis_v2(runs):
    """Same aggregation/criterion as v1, but per-run convergence is computed
    with the cycle-aware splitter."""
    shim = [(label, data) for label, data in runs]
    # Reuse e2_analysis's aggregation by feeding it per-run results directly.
    per_run = []
    for label, data in shim:
        c = convergence_for_run_v2(data)
        defined = [s for s in c["block_steps"] if s is not None]
        med = sorted(defined)[len(defined) // 2] if defined else None
        per_run.append({"run": label, "collapsed": c["collapsed"],
                        "median_conv_step": med,
                        "n_blocks": len(c["block_steps"]),
                        "n_blocks_never_converged":
                            sum(1 for s in c["block_steps"] if s is None),
                        "n_blocks_skipped_nonfinite": c["n_blocks_skipped"],
                        "block_steps": c["block_steps"]})
    usable = [r for r in per_run if r["median_conv_step"] is not None
              or r["n_blocks_never_converged"] > 0]
    n_ok = sum(1 for r in usable if r["median_conv_step"] is not None
               and r["median_conv_step"] <= E2_CONV_STEPS)
    frac = (n_ok / len(usable)) if usable else None
    verdict = ("GREEN" if frac is not None and frac >= 0.90
               else ("NOT-GREEN" if frac is not None else "UNRESOLVED"))
    return {"per_run": per_run, "n_runs_usable": len(usable),
            "n_runs_conv_le_50": n_ok, "fraction": frac,
            "criterion": f"GREEN iff >= 90% of runs have median CYCLE-block "
                         f"convergence <= {E2_CONV_STEPS} steps "
                         f"(band +/-{E2_BAND:.0%})",
            "verdict": verdict}


# ----------------------------------------------------------------------------
# E3 — coupling index.
# ----------------------------------------------------------------------------

def e3_coupling_v2(results_dir, severity, seed, etas, cycles):
    cells = []
    for corruption in v2.V2_CORRUPTIONS:
        for eta in sorted(set(etas)):
            src = v2.source_acc_v2(results_dir, v2.V2_ARCH, corruption,
                                   severity, seed, eta, cycles=cycles)
            stable = {}
            for p in v2.discover_p_values_v2(results_dir, v2.V2_ARCH,
                                             corruption, severity, seed, eta,
                                             cycles=cycles):
                data = v2.load_run_v2(results_dir, v2.V2_ARCH, corruption,
                                      severity, seed, p=p, eta=eta,
                                      cycles=cycles)
                if data is None:
                    continue
                verd = pc.classify_run(data, src, pref_drift=None,
                                       use_drift_criterion=False,
                                       hard_only=True)
                if not verd["collapsed"]:
                    g = v2.grad_norm_gbar_cycled(data)
                    if g and g[0]:
                        stable[p] = g[0]
            row = {"setting": f"E1v2 {corruption}", "eta": eta,
                   "n_stable_p": len(stable)}
            if len(stable) >= 2:
                lo, hi = min(stable), max(stable)
                row.update({"p_lo": lo, "p_hi": hi, "gbar_lo": stable[lo],
                            "gbar_hi": stable[hi],
                            "coupling_index": (stable[hi] - stable[lo])
                            / stable[lo]})
            cells.append(row)
    return cells


def e3_markdown_v2(cells) -> str:
    L = ["# E3 (v2) — coupling index of stationary ||g_bar|| with p", "",
         "coupling := (gbar(p_max_stable)-gbar(p_min_stable))/gbar(p_min_stable)"
         " over each cell's HARD-stable p range (cycled window).", "",
         "| setting | eta | n stable p | p_lo | p_hi | gbar_lo | gbar_hi | "
         "coupling |", "|" + "---|" * 8]
    for c in cells:
        L.append("| " + " | ".join([
            c["setting"], _f(c.get("eta")), str(c.get("n_stable_p")),
            _f(c.get("p_lo")), _f(c.get("p_hi")), _f(c.get("gbar_lo"), 4),
            _f(c.get("gbar_hi"), 4), _f(c.get("coupling_index"), 3)]) + " |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run_analysis(results_dir: Path, severity, seed, etas, cycles):
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows_hard = v2_all_rows(results_dir, severity, seed, etas, cycles, True)
    rows_soft = v2_all_rows(results_dir, severity, seed, etas, cycles, False)
    fits_hard, fits_soft = e1_fits(rows_hard), e1_fits(rows_soft)
    ref_hard = continual_reference(results_dir, severity, seed, hard_only=True)
    ref_soft = continual_reference(results_dir, severity, seed, hard_only=False)

    verdict_hard = e1_verdict(fits_hard, ref_hard, "hard_only(primary)")
    verdict_soft = e1_verdict(fits_soft, ref_soft, "soft+hard(separate)")
    # RULE 1 extension: an anomalous non-monotone zero voids the verdict.
    anomalous = [r["corruption"] + ":" + pc.format_p(r["eta"])
                 for r in rows_hard if r.get("anomalous_zero")]
    if anomalous and verdict_hard.get("verdict") not in ("VOID",):
        verdict_hard["verdict"] = "VOID"
        verdict_hard["why"] = (f"boundary anomaly (non-monotone p*=0) in "
                               f"{anomalous} — instrument failure until "
                               f"proven otherwise (permanent rule).")

    print_cell_table(rows_hard, "HARD (primary)")
    print_cell_table(rows_soft, "soft (separate)")
    for line in (fits_lines(fits_hard, "hard") + fits_lines(fits_soft, "soft")):
        print(line)
    if ref_hard:
        print(f"continual reference (hard): slope={_f(ref_hard['slope'], 4)} "
              f"+/- {_f(ref_hard.get('sigma_slope'), 3)} "
              f"wR^2={_f(ref_hard['r2'], 4)} (expected ~1.152 +/- 0.022)")

    e1_plot(rows_soft, fits_soft, rows_hard, fits_hard, ref_soft, ref_hard,
            analysis_dir / "stage1v2_gbar_law.png")

    # E2
    runs = collect_e2_runs_v2(results_dir, severity, seed, etas, cycles)
    e2 = e2_analysis_v2(runs)
    e2_plot(e2, analysis_dir / "stage1v2_convergence.png")
    print(f"\n[E2] runs={len(runs)} usable={e2['n_runs_usable']} "
          f"conv<=50: {e2['n_runs_conv_le_50']} -> {e2['verdict']}")

    # E3
    e3_cells = e3_coupling_v2(results_dir, severity, seed, etas, cycles)
    (analysis_dir / "stage1v2_coupling.md").write_text(
        e3_markdown_v2(e3_cells), encoding="utf-8")

    # E4/E5 — S re-frozen FRESH from THIS campaign's calibration cells,
    # HARD family (primary). Refuses to freeze null.
    calib_fit = fits_hard["calibration_only"]["fit"]
    consumed = v2.consumed_manifest(
        [Path(f) for r in rows_hard for f in r.get("consumed_files", [])])
    if calib_fit is None or calib_fit.get("slope") is None:
        s_frozen_payload = None
        print("[S-freeze] VOID — calibration-only hard fit not computable; "
              "stage1v2_S_frozen.json NOT written (never null).")
    else:
        s_frozen_payload = {
            "S_frozen": calib_fit["slope"],
            "family": "hard_only (primary)",
            "definition": "calibration-only (gaussian_noise+elastic_transform)"
                          " hard-family OLS slope of p* vs eta*||g_bar|| on "
                          f"the cycled-x{cycles} campaign; intercept dropped "
                          "at prediction time",
            "intercept_not_used": calib_fit["intercept"],
            "r2": calib_fit["r2"], "n_points": calib_fit["n"],
            "provenance": "fresh campaign only — nothing seeded from voided "
                          "runs",
            "consumed_manifest": consumed,
        }
        (analysis_dir / "stage1v2_S_frozen.json").write_text(
            json.dumps(s_frozen_payload, indent=2), encoding="utf-8")
        print(f"[S-freeze] S = {_f(calib_fit['slope'], 5)} "
              f"(hard, calibration-only, n={calib_fit['n']}) -> "
              f"stage1v2_S_frozen.json")

    e4 = e4_zeroshot(rows_hard, fits_hard, delta=0.0)
    e4v = e4_verdict(e4)
    e5_by_delta = {d: e4_zeroshot(rows_hard, fits_hard, delta=d)
                   for d in E5_DELTAS}
    e5v = e5_verdict(e5_by_delta)
    (analysis_dir / "stage1v2_zeroshot.md").write_text(
        zeroshot_markdown(e4, e5_by_delta), encoding="utf-8")

    # Verdict md + JSON artifact (with consumed manifest).
    L = ["# Stage 1 (RESTART, cycled x%d) verdict" % cycles, "",
         "Protocol: single-corruption severity-5 split cycled x%d (~2355 "
         "steps), HARD criterion primary, soft separate. Reference: continual "
         "law recomputed on-disk (independent, clearly labeled)." % cycles, "",
         "## E1 — ||g_bar||-factor law",
         "Criterion: GREEN = slopes within +/-20% of each other AND of the "
         "reference slope (same family), pooled R^2 >= 0.95, x-spread >= "
         "1.5x; AMBER = linear, spread 20-100%; RED = spread > 2x or "
         "non-monotone; VOID = no reference / boundary anomaly.",
         f"- HARD (primary): **{verdict_hard.get('verdict')}** — "
         f"{verdict_hard.get('why')}",
         f"- soft (separate): **{verdict_soft.get('verdict')}** — "
         f"{verdict_soft.get('why')}", "",
         "## E2 — convergence",
         f"- {e2['criterion']}",
         f"- **{e2['verdict']}** ({e2['n_runs_conv_le_50']}/"
         f"{e2['n_runs_usable']} runs)", "",
         "## E3 — coupling", f"- {len(e3_cells)} cells in "
         "stage1v2_coupling.md (descriptive).", "",
         "## E4 — zero-shot on held-out",
         "Criterion: GREEN = >=95% of oracle, zero collapses, on >=80% of "
         "held-out cells; AMBER 85-95%.",
         f"- **{e4v['verdict']}** — {e4v['why']}", "",
         "## E5 — R-sensitivity",
         "Criterion: GREEN = no collapse at +50% R error, graceful at -25%.",
         f"- **{e5v['verdict']}** — {e5v['why']}", "",
         "**STOP — Stage-1 restart report complete; Stage 1b / Stage 2 await "
         "instruction and will use THIS campaign's frozen S as the only "
         "reference.**"]
    verdict_md = "\n".join(L) + "\n"
    (analysis_dir / "stage1v2_verdict.md").write_text(verdict_md,
                                                      encoding="utf-8")
    payload = {"rows_hard": rows_hard, "rows_soft": rows_soft,
               "fits_hard": fits_hard, "fits_soft": fits_soft,
               "continual_reference_hard": ref_hard,
               "continual_reference_soft": ref_soft,
               "verdict_hard": verdict_hard, "verdict_soft": verdict_soft,
               "e2": {k: val for k, val in e2.items() if k != "per_run"},
               "e3": e3_cells, "e4": e4, "e4_verdict": e4v,
               "e5": e5_by_delta, "e5_verdict": e5v,
               "S_frozen": s_frozen_payload,
               "consumed_manifest": consumed}
    (analysis_dir / "stage1v2_gbar_law.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n================ STAGE 1 (RESTART) VERDICT ================\n")
    print(verdict_md)
    for n in ("stage1v2_gbar_law.json", "stage1v2_gbar_law.png",
              "stage1v2_convergence.png", "stage1v2_coupling.md",
              "stage1v2_zeroshot.md", "stage1v2_verdict.md"):
        print(f"[saved] {analysis_dir / n}")
    return payload


# ----------------------------------------------------------------------------
# Self-test: synthetic cycled campaign obeying the law; NaN placed mid-stream
# for collapsing runs (step ~600 of 2355) so cycle-exclusion is exercised.
# ----------------------------------------------------------------------------

def _synth_cycled(corruption, eta, p, gbar, R, cycles=v2.CYCLES,
                  steps_per_cycle=157, source_acc=0.70):
    p_star = eta * gbar / R
    collapsed = p < p_star - 1e-12
    rows, step = [], 0
    for cyc in range(cycles):
        for t in range(steps_per_cycle):
            g = gbar * (1.0 + (0.5 if t < 10 else 0.02 * ((t % 7) - 3) / 3.0))
            row = {"step": step, "local_step": t, "corruption": corruption,
                   "restore_prob": p, "grad_l2": g, "total_grad_norm": g,
                   "drift_l2": 1.0 + 0.001 * step, "energy": -5.0}
            if collapsed and step >= 601:
                row["grad_l2"] = float("nan")
            rows.append(row)
            step += 1
    acc = 0.05 if collapsed else max(0.2, source_acc + 0.08 - 0.5 * (p - p_star))
    return {"args": {"arch": v2.V2_ARCH, "dataset": "cifar10", "severity": 5,
                     "seed": 42, "heat_lr": eta, "heat_restore_prob": p,
                     "corruptions": [corruption] * cycles},
            "results": {"summary": {"heat": {
                "mean_accuracy": acc, "last_accuracy": acc,
                "peak_accuracy": acc, "forgetting": 0.0, "mean_ece": 0.05,
                "num_diagnostic_steps": len(rows),
                "stream_diagnostics": rows}}}}


def self_test():
    import tempfile
    from scripts.analyze_stage1 import _synthetic_source, _synthetic_run
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("=== stage1 v2 self-test ===")
    # Tag isolation vs EVERY prior pattern.
    tag = v2.variant_tag_v2("gaussian_noise", p=0.0075, eta=2e-3)
    check("v2 tag", tag == "pstar_cyc15_gaussian_noise_lr0.002_p0.0075", tag)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        v2.run_output_path_v2(d, "wrn28_10", "gaussian_noise", 5, 42,
                              p=0.005, eta=1e-3).write_text("{}")
        from scripts import stage1_common as sc1
        from scripts import stage1b_common as sb
        check("old E1 discovery blind to cyc15",
              sc1.discover_p_values_sc(d, "wrn28_10", "gaussian_noise", 5, 42,
                                       1e-3) == [])
        check("continual discovery blind to cyc15",
              pc.discover_p_values(d, "wrn28_10", 5, 42, 1e-3) == [])
        check("anchor/drive discovery blind to cyc15",
              sb.discover_lambdas(d, "wrn28_10", "gaussian_noise", 5, 42, 1e-3)
              == [] and sb.discover_drive_p_values(
                  d, "wrn28_10", "gaussian_noise", 5, 42, "entropy", 1e-3) == [])
        check("v2 discovery finds its own",
              v2.discover_p_values_v2(d, "wrn28_10", "gaussian_noise", 5, 42,
                                      1e-3) == [0.005])
        check("genuine-E1 regex blind to cyc15",
              not v2.GENUINE_E1.match(
                  "p9_wrn28_10_pstar_cyc15_gaussian_noise_lr0.001_p0.005"
                  "_seed42_sev5.json"))

    # Cycle splitter + gbar with a mid-stream NaN.
    d_bad = _synth_cycled("gaussian_noise", 1e-3, 0.0, gbar=16.0, R=12.0)
    cycs = v2.group_rows_by_cycle(pc.stream_rows(d_bad))
    check("cycle splitter finds 15 cycles", len(cycs) == 15, str(len(cycs)))
    g = v2.grad_norm_gbar_cycled(d_bad)
    check("gbar excludes NaN cycles (collapse at 601 => 3 clean cycles)",
          g is not None and g[1] == 3 and g[2] == 12,
          str(g))
    check("first NaN at 601",
          pc.first_nonfinite_step(pc.stream_rows(d_bad)) == 601)

    # Guards.
    try:
        v2.require_drive_dir(Path(tempfile.gettempdir()) / "x_no_drive",
                             allow_ephemeral=False)
        check("drive guard aborts off-Drive", False)
    except SystemExit as ex:
        check("drive guard aborts off-Drive", "provenance guard" in str(ex))
    p_ok = v2.require_drive_dir(Path(tempfile.gettempdir()) / "x_no_drive",
                                allow_ephemeral=True)
    check("drive guard override works + sentinel OK", p_ok.exists())

    # Full synthetic law campaign -> analysis end-to-end.
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for c, gbar in GBARS.items():
            for eta in v2.V2_ETAS:
                v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, source=True,
                                      eta=eta).write_text(
                    json.dumps(_synthetic_source(c)))
                pst = eta * gbar / R_TRUE
                grid = sorted(set(pc.scaled_p_grid(eta)
                                  + pc.scaled_pref_ladder(v2.V2_ARCH, eta)
                                  + [round(pst * 0.9, 6), round(pst * 1.1, 6)]))
                for p in grid:
                    v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(_synth_cycled(c, eta, p, gbar, R_TRUE)))
        # continual reference files (law-generated, single-pass synthetic).
        for eta in [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]:
            pc_gbar = 13.0
            pst = eta * pc_gbar / R_TRUE
            for p in sorted({0.0, round(pst * 0.9, 6), round(pst * 1.1, 6),
                             *pc.scaled_pref_ladder("wrn28_10", eta)}):
                pc.run_output_path(d, "wrn28_10", 5, 42, p=p, eta=eta
                                   ).write_text(json.dumps(_synthetic_run(
                    "wrn28_10", "c0", 5, 42, eta, p, gbar=pc_gbar, R=R_TRUE)))
            pc.run_output_path(d, "wrn28_10", 5, 42, source=True, eta=eta
                               ).write_text(json.dumps(_synthetic_source("c0")))

        payload = run_analysis(d, 5, 42, v2.V2_ETAS, v2.CYCLES)
        check("E1 hard verdict GREEN on law data",
              payload["verdict_hard"]["verdict"] == "GREEN",
              json.dumps(payload["verdict_hard"], default=str)[:300])
        check("continual reference recomputed",
              payload["continual_reference_hard"] is not None
              and abs(payload["continual_reference_hard"]["slope"] * R_TRUE
                      - 1.0) < 0.2)
        check("S frozen fresh from calibration",
              payload["S_frozen"] is not None
              and abs(payload["S_frozen"]["S_frozen"] * R_TRUE - 1.0) < 0.2)
        check("consumed manifest embedded",
              len(payload["S_frozen"]["consumed_manifest"]) > 0)
        check("E2 verdict computed", payload["e2"]["verdict"] in
              ("GREEN", "NOT-GREEN"))
        check("E4 evaluated held-out cells",
              len([c for c in payload["e4"]["cells"] if "error" not in c]) == 6)
        check("artifacts written",
              (d / "analysis" / "stage1v2_verdict.md").exists()
              and (d / "analysis" / "stage1v2_gbar_law.png").exists())

        # VOID path: analysis with no continual reference files.
        with tempfile.TemporaryDirectory() as td2:
            d2 = Path(td2)
            for c, gbar in GBARS.items():
                eta = 1e-3
                v2.run_output_path_v2(d2, v2.V2_ARCH, c, 5, 42, source=True,
                                      eta=eta).write_text(
                    json.dumps(_synthetic_source(c)))
                for p in [0.0, 0.01, 0.02]:
                    v2.run_output_path_v2(d2, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(_synth_cycled(c, eta, p, gbar, R_TRUE)))
            payload2 = run_analysis(d2, 5, 42, [1e-3], v2.CYCLES)
            check("VOID without continual reference",
                  payload2["verdict_hard"]["verdict"] == "VOID",
                  payload2["verdict_hard"]["verdict"])

    print(f"\nself-test: {'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
    return 0 if not fails else 1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.results_dir:
        raise SystemExit("--results-dir required (or --self-test)")
    run_analysis(Path(args.results_dir), args.severity, args.seed, args.etas,
                 args.cycles)


if __name__ == "__main__":
    main()
