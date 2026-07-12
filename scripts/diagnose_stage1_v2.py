"""
STAGE-1 RESTART FOLLOW-UP — AMBER diagnosis (D1-D4). ZERO GPU; reads the
existing cycled-x15 run JSONs only. heat==TFF.

The pre-registered AMBER verdict is REPORTED ALONGSIDE every re-analysis here
and never replaced.

D1  Bracket-weighted refit (the elastic precedent from A2): per-corruption
    HARD slopes re-fit with weight 1/bracket_width^2; slope +/- bracket-
    derived uncertainty; recomputed spread; whether CONTRAST's deviation from
    the pooled slope is within its own bracket resolution (1 sigma / 1.96
    sigma, both stated).
D2  E4 per-cell table (held-out cells): corruption, eta, p_hat, snapped p,
    acc(p_hat), acc(oracle), ratio, collapse? — with explicit statements of
    whether the sub-95% cells are contrast cells and whether each miss was a
    collapse or an accuracy shortfall.
D3  E5 reframe (asymmetric margin rule): the +50%-R collapse is the law's
    sharpness (p_hat = (2/3)p* is below threshold BY CONSTRUCTION), not a
    robustness failure. Safety-margin variant p' = c * p_hat for
    c in {1.15, 1.3, 1.5}, snapped to the measured grids, evaluated on ALL
    cells under delta_R in {0, +50%}: collapse incidence + acc ratio per c;
    deliver the smallest c with zero collapses everywhere — the controller's
    margin constant for Stage 2.
D4  Contrast physics probe: within-run stationary ||g_bar|| per CYCLE-BLOCK
    at the reference p (trend across cycles, CV), contrast vs the other
    corruptions, plus ||g_bar|| across the stable-p range (reference-
    measurement stability). Stated mechanism hypothesis (not asserted):
    contrast shrinks input variance -> BN statistics shift.

Outputs: <results>/analysis/stage1v2_diagnosis.{md,json} +
stage1v2_gbar_cycles.png. The JSON embeds the consumed-file manifest.

Usage:
  python scripts/diagnose_stage1_v2.py --results-dir <Drive dir>
  python scripts/diagnose_stage1_v2.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc
from scripts import stage1v2_common as v2
from scripts.analysis_common import stationary_rows, row_metric
from scripts.analyze_stage1 import e1_fits, e4_zeroshot, _snap, CORRUPTION_COLORS
from scripts.analyze_stage1_followup import wls_line
from scripts.analyze_stage1_v2 import v2_all_rows

MARGIN_CS = [1.15, 1.30, 1.50]
DELTAS = [0.0, 0.50]   # base and the +50%-R perturbation


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1v2 AMBER diagnosis (D1-D4)")
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
# D1 — bracket-weighted refit (hard family).
# ----------------------------------------------------------------------------

def d1_refit(rows_hard):
    out = {"per_corruption": {}, "excluded": []}
    all_pts = []
    for c in v2.V2_CORRUPTIONS:
        pts = []
        for r in rows_hard:
            if r["corruption"] != c:
                continue
            lo, hi = r["p_star_bracket_low"], r["p_star_bracket_high"]
            if (r["p_star"] is None or r["eta_times_gbar"] is None
                    or lo is None or hi is None or hi <= lo):
                out["excluded"].append({"cell": f"{c}:{_f(r['eta'])}",
                                        "reason": "unresolved/one-sided bracket"})
                continue
            pts.append((r["eta_times_gbar"], r["p_star"], 1.0 / (hi - lo) ** 2))
        w = wls_line([x for x, _, _ in pts], [y for _, y, _ in pts],
                     [wt for _, _, wt in pts]) if len(pts) >= 2 else None
        out["per_corruption"][c] = {"n": len(pts), "wls": w}
        all_pts.extend(pts)
    out["pooled"] = {"n": len(all_pts),
                     "wls": wls_line([x for x, _, _ in all_pts],
                                     [y for _, y, _ in all_pts],
                                     [wt for _, _, wt in all_pts])
                     if len(all_pts) >= 2 else None}
    slopes = {c: d["wls"]["slope"] for c, d in out["per_corruption"].items()
              if d["wls"]}
    out["slopes"] = slopes
    out["spread"] = (max(slopes.values()) / min(slopes.values()) - 1.0
                     if len(slopes) >= 2 and min(slopes.values()) > 0 else None)
    pooled = out["pooled"]["wls"]
    ctr = out["per_corruption"].get("contrast", {}).get("wls")
    if pooled and ctr:
        dev = ctr["slope"] - pooled["slope"]
        out["contrast_check"] = {
            "slope_contrast": ctr["slope"], "sigma_contrast": ctr["sigma_slope"],
            "slope_pooled": pooled["slope"], "deviation": dev,
            "deviation_rel_pooled": dev / pooled["slope"] if pooled["slope"] else None,
            "within_1_sigma": abs(dev) <= ctr["sigma_slope"],
            "within_1p96_sigma": abs(dev) <= 1.96 * ctr["sigma_slope"],
        }
    else:
        out["contrast_check"] = None
    return out


# ----------------------------------------------------------------------------
# D3 — safety-margin rule over ALL cells.
# ----------------------------------------------------------------------------

def d3_margin(rows_hard, fits_hard):
    calib = fits_hard["calibration_only"]["fit"]
    if calib is None:
        return {"error": "calibration fit unavailable"}
    slope = calib["slope"]
    table = []
    for c_margin in MARGIN_CS:
        for delta in DELTAS:
            cells = []
            for r in rows_hard:
                x, grid = r.get("eta_times_gbar"), r.get("p_values_run") or []
                per_p = r.get("per_p") or {}
                if x is None or not grid:
                    continue
                p_prime_raw = c_margin * slope * x / (1.0 + delta)
                p_prime = _snap(p_prime_raw, grid)
                q = per_p.get(pc.format_p(p_prime), {})
                best = max((qq.get("mean_acc") for qq in per_p.values()
                            if qq.get("valid") and qq.get("mean_acc") is not None),
                           default=None)
                cells.append({
                    "cell": f"{r['corruption']}:{pc.format_p(r['eta'])}",
                    "p_prime_raw": p_prime_raw, "p_prime_snapped": p_prime,
                    "collapsed": bool(q.get("collapsed")),
                    "acc": q.get("mean_acc"),
                    "acc_ratio": (q.get("mean_acc") / best
                                  if q.get("mean_acc") is not None and best
                                  else None),
                })
            ratios = sorted(x["acc_ratio"] for x in cells
                            if x["acc_ratio"] is not None)
            table.append({
                "c": c_margin, "delta_R": delta, "n_cells": len(cells),
                "n_collapse": sum(1 for x in cells if x["collapsed"]),
                "median_acc_ratio": ratios[len(ratios) // 2] if ratios else None,
                "min_acc_ratio": ratios[0] if ratios else None,
                "cells": cells,
            })
    # smallest c with ZERO collapses across all cells AND both deltas.
    chosen = None
    for c_margin in MARGIN_CS:
        rows_c = [t for t in table if t["c"] == c_margin]
        if all(t["n_collapse"] == 0 for t in rows_c):
            chosen = c_margin
            break
    return {"slope_calibration": slope, "table": table,
            "margin_constant_c": chosen,
            "note": "p' = c * S_calib * eta * ||g_bar|| / (1+delta_R); snapped "
                    "to the measured grid (ties -> larger p). delta_R=+50% is "
                    "the perturbed-R case; c is chosen so even that never "
                    "collapses."}


# ----------------------------------------------------------------------------
# D4 — per-cycle ||g_bar|| stationarity at the reference p.
# ----------------------------------------------------------------------------

def per_cycle_gbar(data):
    vals = []
    for cyc in v2.group_rows_by_cycle(pc.stream_rows(data)):
        if pc.block_has_nonfinite(cyc):
            vals.append(None)
            continue
        stat = stationary_rows(cyc, v2.STATIONARY_LO, v2.STATIONARY_HI)
        xs = [row_metric(r, "grad_l2") for r in stat]
        xs = [x for x in xs if x is not None]
        vals.append(sum(xs) / len(xs) if xs else None)
    return vals


def d4_probe(results_dir, rows_hard, severity, seed, cycles):
    out = []
    for r in rows_hard:
        c, eta, p_ref = r["corruption"], r["eta"], r.get("p_ref_used")
        if p_ref is None:
            out.append({"corruption": c, "eta": eta, "note": "no p_ref"})
            continue
        data = v2.load_run_v2(results_dir, v2.V2_ARCH, c, severity, seed,
                              p=p_ref, eta=eta, cycles=cycles)
        if data is None:
            out.append({"corruption": c, "eta": eta, "note": "p_ref run missing"})
            continue
        series = per_cycle_gbar(data)
        finite = [x for x in series if x is not None]
        if len(finite) < 3:
            out.append({"corruption": c, "eta": eta, "note": "too few cycles"})
            continue
        mean = sum(finite) / len(finite)
        var = sum((x - mean) ** 2 for x in finite) / len(finite)
        cv = (var ** 0.5) / mean if mean else None
        rel_trend = (finite[-1] - finite[0]) / finite[0] if finite[0] else None
        # gbar across the STABLE p range (reference-measurement stability)
        stable_gbars = {}
        for p in r.get("p_values_run") or []:
            q = (r.get("per_p") or {}).get(pc.format_p(p), {})
            if not q.get("valid") or q.get("collapsed"):
                continue
            d2 = v2.load_run_v2(results_dir, v2.V2_ARCH, c, severity, seed,
                                p=p, eta=eta, cycles=cycles)
            if d2 is None:
                continue
            g = v2.grad_norm_gbar_cycled(d2)
            if g and g[0]:
                stable_gbars[p] = g[0]
        ref_spread = (max(stable_gbars.values()) / min(stable_gbars.values()) - 1
                      if len(stable_gbars) >= 2 else None)
        out.append({"corruption": c, "eta": eta, "p_ref": p_ref,
                    "gbar_mean": mean, "cv": cv, "rel_trend_first_to_last":
                    rel_trend, "per_cycle": series,
                    "gbar_spread_across_stable_p": ref_spread})
    return out


def d4_plot(d4, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    etas = sorted({r["eta"] for r in d4 if "per_cycle" in r})
    for ax, eta in zip(axes, etas):
        for r in d4:
            if r.get("eta") != eta or "per_cycle" not in r:
                continue
            ys = r["per_cycle"]
            xs = list(range(len(ys)))
            base = next((y for y in ys if y is not None), None)
            if base is None:
                continue
            norm = [y / base if y is not None else None for y in ys]
            ax.plot(xs, norm, marker="o", ms=3,
                    color=CORRUPTION_COLORS.get(r["corruption"], "0.5"),
                    label=r["corruption"])
        ax.set_title(f"eta={eta:g}")
        ax.set_xlabel("cycle")
        ax.set_ylabel("stationary ||g_bar|| / cycle-0")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("D4: per-cycle stationary ||g_bar|| at p_ref (normalized)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------

def build_md(amber, d1, d2, d2_notes, d3, d4) -> str:
    L = ["# Stage-1v2 AMBER diagnosis (D1-D4)", "",
         "## Pre-registered verdict (reported verbatim, never replaced)",
         f"- HARD (primary): **{amber.get('verdict', 'n/a')}** — "
         f"{amber.get('why', '(stage1v2_gbar_law.json not found)')}", "",
         "## D1 — bracket-weighted refit (weight = 1/bracket_width^2)", "",
         "| corruption | n | slope (WLS) | +/- sigma | weighted R^2 |",
         "|---|---|---|---|---|"]
    for c, d in d1["per_corruption"].items():
        w = d["wls"]
        L.append(f"| {c} | {d['n']} | {_f(w['slope'], 4) if w else 'n/a'} | "
                 f"{_f(w['sigma_slope'], 3) if w else 'n/a'} | "
                 f"{_f(w['r2_weighted'], 4) if w else 'n/a'} |")
    pw = d1["pooled"]["wls"]
    L += [f"| **pooled** | {d1['pooled']['n']} | "
          f"{_f(pw['slope'], 4) if pw else 'n/a'} | "
          f"{_f(pw['sigma_slope'], 3) if pw else 'n/a'} | "
          f"{_f(pw['r2_weighted'], 4) if pw else 'n/a'} |", "",
          f"Recomputed spread (max/min - 1): **{_f(d1['spread'], 4)}**"]
    ck = d1["contrast_check"]
    if ck:
        L += [f"- contrast deviation vs pooled: {_f(ck['deviation'], 4)} "
              f"({_f((ck['deviation_rel_pooled'] or 0) * 100, 3)}%); within "
              f"1 sigma: **{ck['within_1_sigma']}**; within 1.96 sigma: "
              f"**{ck['within_1p96_sigma']}**"]
    L += ["", "## D2 — E4 per-cell table (held-out)", "",
          "| corruption | eta | p_hat | snapped | acc(p_hat) | acc(oracle) | "
          "ratio | collapse? |", "|" + "---|" * 8]
    for c in d2.get("cells", []):
        if "error" in c:
            continue
        L.append("| " + " | ".join([
            c["corruption"], _f(c["eta"]), _f(c["p_hat_raw"], 4),
            _f(c["p_hat_snapped"]), _f(c["acc_at_p_hat"], 4),
            _f(c["oracle_acc"], 4), _f(c["acc_ratio"], 4),
            "YES" if c["collapsed_at_p_hat"] else "no"]) + " |")
    L += [""] + [f"- {n}" for n in d2_notes]
    L += ["", "## D3 — safety-margin rule (E5 reframe)",
          f"- {d3.get('note')}", "",
          "| c | delta_R | cells | collapses | median ratio | min ratio |",
          "|---|---|---|---|---|---|"]
    for t in d3.get("table", []):
        L.append(f"| {t['c']} | {t['delta_R']:+.0%} | {t['n_cells']} | "
                 f"{t['n_collapse']} | {_f(t['median_acc_ratio'], 4)} | "
                 f"{_f(t['min_acc_ratio'], 4)} |")
    L += ["", f"**Margin constant: c = {_f(d3.get('margin_constant_c'))}** "
          "(smallest tested c with zero collapses across all cells incl. "
          "+50%-R). This becomes the Stage-2 controller's margin constant."
          + ("" if d3.get("margin_constant_c") is not None else
             " — NO tested c achieved zero collapses; report honestly and "
             "extend the c grid before Stage 2.")]
    L += ["", "## D4 — contrast physics probe (per-cycle ||g_bar|| at p_ref)",
          "", "| corruption | eta | p_ref | mean ||g_bar|| | CV | trend "
          "(first->last) | spread across stable p |", "|" + "---|" * 7]
    for r in d4:
        if "per_cycle" not in r:
            L.append(f"| {r['corruption']} | {_f(r.get('eta'))} | "
                     f"n/a ({r.get('note')}) |" + " |" * 4)
            continue
        L.append("| " + " | ".join([
            r["corruption"], _f(r["eta"]), _f(r["p_ref"]),
            _f(r["gbar_mean"], 4), _f(r["cv"], 3),
            _f(r["rel_trend_first_to_last"], 3),
            _f(r["gbar_spread_across_stable_p"], 3)]) + " |")
    L += ["", "Mechanism hypothesis (stated, not asserted): contrast shrinks "
          "input variance -> BN statistics shift across cycles; a "
          "non-stationary ||g_bar|| or a large spread across stable p would "
          "make the reference-p measurement unstable and bias that "
          "corruption's slope."]
    # Honest one-paragraph summary.
    ctr = d1.get("contrast_check") or {}
    ctr_d4 = [r for r in d4 if r.get("corruption") == "contrast"
              and "per_cycle" in r]
    oth_d4 = [r for r in d4 if r.get("corruption") != "contrast"
              and "per_cycle" in r]
    ctr_cv = (sum(r["cv"] for r in ctr_d4) / len(ctr_d4)) if ctr_d4 else None
    oth_cv = (sum(r["cv"] for r in oth_d4) / len(oth_d4)) if oth_d4 else None
    L += ["", "## Summary (paper-ready, honest)", "",
          f"The Stage-1 restart verdict is the pre-registered **"
          f"{amber.get('verdict', 'n/a')}**. Bracket-weighted refit gives a "
          f"per-corruption slope spread of {_f((d1['spread'] or 0) * 100, 3)}% "
          f"(pooled slope {_f(pw['slope'], 4) if pw else 'n/a'} +/- "
          f"{_f(pw['sigma_slope'], 3) if pw else 'n/a'}). Contrast attribution: "
          f"deviation {_f((ctr.get('deviation_rel_pooled') or 0) * 100, 3)}% "
          f"vs pooled, {'within' if ctr.get('within_1p96_sigma') else 'OUTSIDE'} "
          f"its 1.96-sigma bracket resolution; per-cycle ||g_bar|| CV "
          f"{_f(ctr_cv, 3)} (contrast) vs {_f(oth_cv, 3)} (others). The +50%-R "
          f"collapse in E5 reflects the law's sharpness (p_hat=(2/3)p* by "
          f"construction); with the asymmetric margin rule the smallest safe "
          f"constant is c={_f(d3.get('margin_constant_c'))}, adopted for the "
          f"Stage-2 controller.", "",
          "**STOP — diagnosis complete. Stage 1b (E6/E7, using THIS campaign's "
          "frozen S) awaits instruction.**"]
    return "\n".join(L) + "\n"


def run_diagnosis(results_dir: Path, severity, seed, etas, cycles):
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows_hard = v2_all_rows(results_dir, severity, seed, etas, cycles, True)
    fits_hard = e1_fits(rows_hard)

    # Pre-registered verdict passthrough.
    amber = {}
    law = analysis_dir / "stage1v2_gbar_law.json"
    if law.exists():
        try:
            amber = json.loads(law.read_text(encoding="utf-8"))["verdict_hard"]
        except Exception:
            amber = {}

    d1 = d1_refit(rows_hard)
    d2 = e4_zeroshot(rows_hard, fits_hard, delta=0.0)
    cells = [c for c in d2.get("cells", []) if "error" not in c]
    sub95 = [c for c in cells if c["acc_ratio"] is not None
             and (c["acc_ratio"] < 0.95 or c["collapsed_at_p_hat"])]
    d2_notes = []
    if sub95:
        all_contrast = all(c["corruption"] == "contrast" for c in sub95)
        d2_notes.append(
            f"sub-95% / failing cells: {[c['corruption'] + ':' + _f(c['eta']) for c in sub95]} — "
            + ("ALL are contrast cells." if all_contrast
               else "NOT all contrast (attribution is not contrast-only)."))
        for c in sub95:
            kind = ("COLLAPSE at p_hat" if c["collapsed_at_p_hat"]
                    else f"accuracy shortfall (ratio {_f(c['acc_ratio'], 4)}, "
                         f"no collapse)")
            d2_notes.append(f"  - {c['corruption']}:{_f(c['eta'])}: {kind}")
    else:
        d2_notes.append("no sub-95% or collapsing held-out cells at delta_R=0.")
    d3 = d3_margin(rows_hard, fits_hard)
    d4 = d4_probe(results_dir, rows_hard, severity, seed, cycles)
    d4_plot(d4, analysis_dir / "stage1v2_gbar_cycles.png")

    md = build_md(amber, d1, d2, d2_notes, d3, d4)
    (analysis_dir / "stage1v2_diagnosis.md").write_text(md, encoding="utf-8")
    consumed = v2.consumed_manifest(
        [Path(f) for r in rows_hard for f in r.get("consumed_files", [])])
    (analysis_dir / "stage1v2_diagnosis.json").write_text(
        json.dumps({"preregistered_verdict": amber, "d1": d1,
                    "d2": d2, "d2_notes": d2_notes, "d3": d3, "d4": d4,
                    "consumed_manifest": consumed}, indent=2, default=str),
        encoding="utf-8")
    print(md)
    for n in ("stage1v2_diagnosis.md", "stage1v2_diagnosis.json",
              "stage1v2_gbar_cycles.png"):
        print(f"[saved] {analysis_dir / n}")
    return {"d1": d1, "d2": d2, "d2_notes": d2_notes, "d3": d3, "d4": d4}


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    import tempfile
    from scripts.analyze_stage1 import _synthetic_source
    from scripts.analyze_stage1_v2 import _synth_cycled
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("=== stage1v2 diagnosis self-test ===")
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0

    def synth(c, eta, p, drift=0.0):
        d = _synth_cycled(c, eta, p, GBARS[c], R_TRUE)
        if drift:
            # inflate grad_l2 per cycle (non-stationarity for D4)
            for r in d["results"]["summary"]["heat"]["stream_diagnostics"]:
                cyc = r["step"] // 157
                if r["grad_l2"] == r["grad_l2"]:  # finite
                    r["grad_l2"] *= (1.0 + drift * cyc)
                    r["total_grad_norm"] = r["grad_l2"]
        return d

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
                                  + [round(pst * f, 6)
                                     for f in (0.9, 1.1, 1.2, 1.4, 1.6)]))
                drift = 0.04 if c == "contrast" else 0.0
                for p in grid:
                    v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(synth(c, eta, p, drift)))
        res = run_diagnosis(d, 5, 42, v2.V2_ETAS, v2.CYCLES)

        pooled = res["d1"]["pooled"]["wls"]
        check("D1 pooled WLS ~ 1/R", pooled is not None
              and abs(pooled["slope"] * R_TRUE - 1.0) < 0.25,
              _f(pooled["slope"] if pooled else None))
        check("D1 contrast check computed",
              res["d1"]["contrast_check"] is not None)
        check("D2 6 held-out cells",
              len([c for c in res["d2"]["cells"] if "error" not in c]) == 6)
        check("D2 notes state contrast attribution explicitly",
              any("contrast" in n or "no sub-95%" in n
                  for n in res["d2_notes"]))
        t = res["d3"]["table"]
        check("D3 table covers 3 c x 2 deltas", len(t) == 6)
        coll_by_c = {cm: sum(x["n_collapse"] for x in t if x["c"] == cm)
                     for cm in MARGIN_CS}
        check("D3 collapse count non-increasing in c",
              coll_by_c[1.15] >= coll_by_c[1.30] >= coll_by_c[1.50],
              str(coll_by_c))
        check("D3 reports a margin constant (or honest None)",
              "margin_constant_c" in res["d3"])
        ctr = [r for r in res["d4"] if r["corruption"] == "contrast"
               and "per_cycle" in r]
        oth = [r for r in res["d4"] if r["corruption"] == "gaussian_noise"
               and "per_cycle" in r]
        check("D4 detects contrast non-stationarity (injected +4%/cycle)",
              ctr and oth and
              min(r["rel_trend_first_to_last"] for r in ctr) > 0.3
              and max(abs(r["rel_trend_first_to_last"]) for r in oth) < 0.1,
              f"ctr={[_f(r['rel_trend_first_to_last'],3) for r in ctr]} "
              f"oth={[_f(r['rel_trend_first_to_last'],3) for r in oth]}")
        check("artifacts written",
              (d / "analysis" / "stage1v2_diagnosis.md").exists()
              and (d / "analysis" / "stage1v2_gbar_cycles.png").exists())

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
    run_diagnosis(Path(args.results_dir), args.severity, args.seed, args.etas,
                  args.cycles)


if __name__ == "__main__":
    main()
