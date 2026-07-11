"""
Stage-1 follow-up, PART A (zero GPU; existing JSONs only).

A1  Soft-boundary attribution: for every E1 cell, correlate/regress p*_soft and
    p*_hard against (i) that corruption's source accuracy and (ii)
    eta*||g_bar||. Reports R^2 and Pearson r for all four (target, predictor)
    pairs — the numbers either way, no assumed pattern.
A2  Bracket-aware refit of the HARD-ONLY slopes: weighted least squares with
    per-point weight 1/bracket_width^2 (bracket_width = p*_bracket_high -
    p*_bracket_low). Reports slope +/- bracket-derived uncertainty per
    corruption, the pooled refit, the recomputed spread, and whether
    elastic_transform's deviation from the pooled slope is within its own
    bracket-resolution uncertainty (1-sigma and 1.96-sigma, both stated).

The pooled bracket-weighted hard-only slope is FROZEN to
  <results-dir>/analysis/stage1_hardslope_frozen.json
and is the S_frozen consumed by the Part-B forward-confirmation orchestrator
(scripts/run_stage1_forward.py), which records which number it used.

Definitions (so every number is reproducible):
  * WLS model: y = a + b*x with weights w_i = 1/width_i^2, i.e. the
    measurement-error model sigma_i = width_i. Same model family (with
    intercept) as every other fit in this codebase; the through-origin WLS
    slope is also reported since the forward prediction p_hat = S*eta*||g_bar||
    applies the slope through the origin.
  * slope uncertainty: sqrt(1 / sum_i w_i*(x_i - xbar_w)^2)  (with-intercept),
    sqrt(1 / sum_i w_i*x_i^2)                                 (through-origin).
  * weighted R^2: 1 - sum w*(y-yhat)^2 / sum w*(y-ybar_w)^2.
  * spread := max(per-corruption slope)/min(per-corruption slope) - 1
    (identical to the E1 verdict's definition).
  * cells with p* unresolved (censored) or a one-sided bracket are excluded
    from the WLS and counted explicitly.

Usage:
  python scripts/analyze_stage1_followup.py --results-dir <E1 dir> \
      [--etasweep-results-dir <dir>]
  python scripts/analyze_stage1_followup.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts.analyze_stage1 import e1_all_rows, e1_fits


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 follow-up Part A (A1+A2)")
    p.add_argument("--results-dir", type=str, default="",
                   help="Dir with the E1 single-corruption run JSONs.")
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# ----------------------------------------------------------------------------
# Statistics helpers (plain, dependency-free).
# ----------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ols_stats(xs, ys):
    fit = pc.least_squares_line(list(xs), list(ys))
    r = pearson(xs, ys)
    return {"n": len(xs), "slope": fit["slope"] if fit else None,
            "intercept": fit["intercept"] if fit else None,
            "r2": fit["r2"] if fit else None, "pearson_r": r}


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns slope, intercept,
    sigma_slope, weighted R^2, n."""
    n = len(xs)
    if n < 2 or len(ys) != n or len(ws) != n:
        return None
    W = sum(ws)
    if W <= 0:
        return None
    xw = sum(w * x for w, x in zip(ws, xs)) / W
    yw = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - xw) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return None
    sxy = sum(w * (x - xw) * (y - yw) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = yw - slope * xw
    sigma_slope = math.sqrt(1.0 / sxx)
    ss_res = sum(w * (y - (slope * x + intercept)) ** 2
                 for w, x, y in zip(ws, xs, ys))
    ss_tot = sum(w * (y - yw) ** 2 for w, y in zip(ws, ys))
    r2w = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"slope": slope, "intercept": intercept,
            "sigma_slope": sigma_slope, "r2_weighted": r2w, "n": n}


def wls_origin(xs, ys, ws):
    """Weighted least squares through the origin: y = b*x."""
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    if sxx <= 0:
        return None
    slope = sum(w * x * y for w, x, y in zip(ws, xs, ys)) / sxx
    return {"slope": slope, "sigma_slope": math.sqrt(1.0 / sxx),
            "n": len(xs)}


# ----------------------------------------------------------------------------
# A1 — soft-boundary attribution
# ----------------------------------------------------------------------------

def a1_attribution(rows_soft, rows_hard):
    """Correlate/regress p*_soft and p*_hard against source_acc and
    eta*||g_bar||, pooled over all E1 cells with the quantities defined."""
    def collect(rows):
        pts = [r for r in rows if r.get("p_star") is not None
               and r.get("eta_times_gbar") is not None
               and r.get("source_acc") is not None]
        return pts

    out = {"n_cells_soft": None, "n_cells_hard": None, "pairs": {},
           "per_eta_supplementary": {}}
    for tag, rows in (("p_star_soft", rows_soft), ("p_star_hard", rows_hard)):
        pts = collect(rows)
        out["n_cells_%s" % ("soft" if "soft" in tag else "hard")] = len(pts)
        ys = [r["p_star"] for r in pts]
        for pred, key in (("source_acc", "source_acc"),
                          ("eta_times_gbar", "eta_times_gbar")):
            xs = [r[key] for r in pts]
            out["pairs"][f"{tag}~{pred}"] = ols_stats(xs, ys)
        # Supplementary: per-eta correlations vs source_acc (eta pooling
        # confounds the soft target, so the fixed-eta view is shown too;
        # n per eta is small — labeled, not used for any verdict).
        per_eta = {}
        for eta in sorted({round(r["eta"], 8) for r in pts}):
            sub = [r for r in pts if abs(r["eta"] - eta) < 1e-9]
            per_eta[pc.format_p(eta)] = {
                "n": len(sub),
                "r_vs_source_acc": pearson([r["source_acc"] for r in sub],
                                           [r["p_star"] for r in sub]),
                "r_vs_eta_gbar": pearson([r["eta_times_gbar"] for r in sub],
                                         [r["p_star"] for r in sub]),
            }
        out["per_eta_supplementary"][tag] = per_eta
    return out


def a1_markdown(a1, rows_soft, rows_hard) -> str:
    L = ["# A1 — soft-boundary attribution", "",
         "For every E1 cell: p*_soft and p*_hard regressed/correlated against "
         "(i) that corruption's source accuracy and (ii) eta*||g_bar||. Pooled "
         "over all cells with both quantities defined; the expected pattern "
         "was TESTED, not assumed.", "",
         "| target | predictor | n | OLS slope | R^2 | Pearson r |",
         "|---|---|---|---|---|---|"]
    for pair, s in a1["pairs"].items():
        t, p = pair.split("~")
        L.append(f"| {t} | {p} | {s['n']} | {_f(s['slope'], 4)} | "
                 f"{_f(s['r2'], 4)} | {_f(s['pearson_r'], 4)} |")
    L += ["", "## Supplementary: fixed-eta Pearson r (n per eta is small; "
          "descriptive only)", "",
          "| target | eta | n | r vs source_acc | r vs eta*||g_bar|| |",
          "|---|---|---|---|---|"]
    for tag, per_eta in a1["per_eta_supplementary"].items():
        for eta, s in per_eta.items():
            L.append(f"| {tag} | {eta} | {s['n']} | "
                     f"{_f(s['r_vs_source_acc'], 4)} | {_f(s['r_vs_eta_gbar'], 4)} |")
    L += ["", "## Cell inventory (both families)", "",
          "| corruption | eta | source_acc | eta*gbar | p*_soft | crit_soft | "
          "p*_hard | crit_hard |", "|---|---|---|---|---|---|---|---|"]
    hmap = {(r["corruption"], round(r["eta"], 8)): r for r in rows_hard}
    for r in rows_soft:
        h = hmap.get((r["corruption"], round(r["eta"], 8)), {})
        L.append("| " + " | ".join([
            r["corruption"], _f(r["eta"]), _f(r["source_acc"], 4),
            _f(r["eta_times_gbar"], 4), _f(r["p_star"]),
            str(r["collapse_criterion"]), _f(h.get("p_star")),
            str(h.get("collapse_criterion")),
        ]) + " |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# A2 — bracket-aware refit (hard-only)
# ----------------------------------------------------------------------------

def a2_refit(rows_hard):
    """Bracket-weighted WLS of the hard-only p* points, per corruption and
    pooled. Weight = 1/bracket_width^2."""
    def usable(rows):
        out = []
        for r in rows:
            if (r.get("p_star") is None or r.get("eta_times_gbar") is None
                    or r.get("p_star_bracket_low") is None
                    or r.get("p_star_bracket_high") is None):
                continue
            width = r["p_star_bracket_high"] - r["p_star_bracket_low"]
            if width <= 0:
                continue
            out.append((r["eta_times_gbar"], r["p_star"], 1.0 / width ** 2, r))
        return out

    result = {"per_corruption": {}, "excluded_cells": []}
    all_pts = []
    calib_pts = []
    for c in sc.E1_CORRUPTIONS:
        crows = [r for r in rows_hard if r["corruption"] == c]
        pts = usable(crows)
        for r in crows:
            if r not in [p[3] for p in pts]:
                result["excluded_cells"].append(
                    {"corruption": c, "eta": r.get("eta"),
                     "reason": "unresolved p* or one-sided/zero-width bracket"})
        xs, ys, ws = ([p[0] for p in pts], [p[1] for p in pts],
                      [p[2] for p in pts])
        result["per_corruption"][c] = {
            "n": len(pts),
            "wls": wls_line(xs, ys, ws),
            "wls_origin": wls_origin(xs, ys, ws) if pts else None,
        }
        all_pts.extend(pts)
        if c in sc.CALIBRATION_CORRUPTIONS:
            calib_pts.extend(pts)

    for name, pts in (("pooled", all_pts), ("pooled_calibration_only", calib_pts)):
        xs, ys, ws = ([p[0] for p in pts], [p[1] for p in pts],
                      [p[2] for p in pts])
        result[name] = {"n": len(pts), "wls": wls_line(xs, ys, ws),
                        "wls_origin": wls_origin(xs, ys, ws) if pts else None}

    # Recomputed spread over per-corruption weighted slopes.
    slopes = {c: v["wls"]["slope"] for c, v in result["per_corruption"].items()
              if v["wls"] is not None}
    result["per_corruption_slopes"] = slopes
    if len(slopes) >= 2 and min(slopes.values()) > 0:
        result["spread"] = max(slopes.values()) / min(slopes.values()) - 1.0
    else:
        result["spread"] = None

    # elastic_transform deviation vs its own bracket-resolution uncertainty.
    pooled = result["pooled"]["wls"]
    el = result["per_corruption"].get("elastic_transform", {}).get("wls")
    if pooled and el:
        dev = el["slope"] - pooled["slope"]
        result["elastic_transform_check"] = {
            "slope_elastic": el["slope"], "sigma_elastic": el["sigma_slope"],
            "slope_pooled": pooled["slope"],
            "deviation": dev,
            "deviation_rel_pooled": dev / pooled["slope"] if pooled["slope"] else None,
            "within_1_sigma": abs(dev) <= el["sigma_slope"],
            "within_1p96_sigma": abs(dev) <= 1.96 * el["sigma_slope"],
        }
    else:
        result["elastic_transform_check"] = None
    return result


def a2_markdown(a2) -> str:
    L = ["# A2 — bracket-aware refit of the hard-only slopes", "",
         "WLS y = a + b*x with weight 1/bracket_width^2 "
         "(sigma_i := bracket width); slope uncertainty = "
         "sqrt(1/sum w*(x-xbar_w)^2). Through-origin WLS slope also shown "
         "(the forward prediction applies the slope through the origin). "
         "Cells with unresolved p* or a one-sided bracket are excluded and "
         "listed.", "",
         "| corruption | n | slope (WLS) | +/- sigma | intercept | weighted R^2 "
         "| slope (origin) | +/- sigma |", "|---|---|---|---|---|---|---|---|"]
    for c, v in a2["per_corruption"].items():
        w, o = v["wls"], v["wls_origin"]
        L.append("| " + " | ".join([
            c, str(v["n"]),
            _f(w["slope"], 4) if w else "n/a", _f(w["sigma_slope"], 3) if w else "n/a",
            _f(w["intercept"], 3) if w else "n/a",
            _f(w["r2_weighted"], 4) if w else "n/a",
            _f(o["slope"], 4) if o else "n/a", _f(o["sigma_slope"], 3) if o else "n/a",
        ]) + " |")
    for name in ("pooled", "pooled_calibration_only"):
        v = a2[name]
        w, o = v["wls"], v["wls_origin"]
        L.append("| **" + name + "** | " + " | ".join([
            str(v["n"]),
            _f(w["slope"], 4) if w else "n/a", _f(w["sigma_slope"], 3) if w else "n/a",
            _f(w["intercept"], 3) if w else "n/a",
            _f(w["r2_weighted"], 4) if w else "n/a",
            _f(o["slope"], 4) if o else "n/a", _f(o["sigma_slope"], 3) if o else "n/a",
        ]) + " |")
    L += ["", f"Recomputed spread (max/min - 1 over per-corruption WLS slopes): "
          f"**{_f(a2['spread'], 4)}**"]
    if a2["excluded_cells"]:
        L += ["", "Excluded cells:"]
        L += [f"- {e['corruption']} eta={_f(e['eta'])}: {e['reason']}"
              for e in a2["excluded_cells"]]
    ck = a2["elastic_transform_check"]
    L += ["", "## elastic_transform deviation vs bracket-resolution uncertainty"]
    if ck is None:
        L += ["- not computable (missing fit)"]
    else:
        L += [f"- slope(elastic) = {_f(ck['slope_elastic'], 4)} +/- "
              f"{_f(ck['sigma_elastic'], 3)}; slope(pooled) = "
              f"{_f(ck['slope_pooled'], 4)}",
              f"- deviation = {_f(ck['deviation'], 4)} "
              f"({_f((ck['deviation_rel_pooled'] or 0) * 100, 3)}% of pooled)",
              f"- within 1 sigma: **{ck['within_1_sigma']}**; within 1.96 sigma: "
              f"**{ck['within_1p96_sigma']}**"]
    return "\n".join(L) + "\n"


def freeze_hard_slope(a2, out_path: Path):
    """Freeze S_frozen for Part B. S_frozen = the pooled bracket-weighted
    HARD-ONLY WLS slope (with-intercept model; the intercept is dropped when
    the prediction p_hat = S*eta*||g_bar|| is formed, per the follow-up spec).
    The through-origin and calibration-only variants are recorded alongside.

    PERMANENT RULE (Stage-1b audit, F2): if the pooled refit is not
    computable (no usable E1 hard points in the dir), REFUSE to write a null
    frozen artifact — downstream campaigns gate on this file."""
    pooled = a2["pooled"]["wls"]
    if pooled is None or pooled.get("slope") is None:
        raise SystemExit(
            "[ABORT — VOID] the bracket-weighted pooled hard-only refit is "
            "not computable (no usable E1 hard points found in this results "
            "dir). Refusing to write a null stage1_hardslope_frozen.json. "
            "Point --results-dir at the dir holding the E1 JSONs.")
    payload = {
        "S_frozen": pooled["slope"] if pooled else None,
        "S_frozen_definition": "pooled bracket-weighted hard-only WLS slope "
                               "(weight=1/bracket_width^2, model y=a+b*x; "
                               "intercept dropped at prediction time)",
        "sigma_slope": pooled["sigma_slope"] if pooled else None,
        "intercept_not_used_in_prediction": pooled["intercept"] if pooled else None,
        "n_points": a2["pooled"]["n"],
        "variants": {
            "pooled_origin": a2["pooled"]["wls_origin"],
            "pooled_calibration_only": a2["pooled_calibration_only"]["wls"],
            "pooled_calibration_only_origin": a2["pooled_calibration_only"]["wls_origin"],
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def self_test():
    import tempfile
    from scripts.analyze_stage1 import _synthetic_run, _synthetic_source
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("=== stage1 follow-up Part A self-test ===")
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for c, gbar in GBARS.items():
            for eta in sc.E1_ETAS:
                sc.run_output_path_sc(d, sc.E1_ARCH, c, 5, 42, source=True,
                                      eta=eta).write_text(
                    json.dumps(_synthetic_source(c)))
                pst = eta * gbar / R_TRUE
                grid = sorted(set(pc.scaled_p_grid(eta)
                                  + pc.scaled_pref_ladder(sc.E1_ARCH, eta)
                                  + [round(pst * 0.9, 6), round(pst * 1.1, 6)]))
                for p in grid:
                    sc.run_output_path_sc(d, sc.E1_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(_synthetic_run(sc.E1_ARCH, c, 5, 42, eta, p,
                                                  gbar=gbar, R=R_TRUE)))
        rows_soft = e1_all_rows(d, 5, 42, hard_only=False)
        rows_hard = e1_all_rows(d, 5, 42, hard_only=True)

        a1 = a1_attribution(rows_soft, rows_hard)
        check("A1 produces all 4 pooled pairs", len(a1["pairs"]) == 4,
              str(list(a1["pairs"])))
        hard_x = a1["pairs"]["p_star_hard~eta_times_gbar"]
        check("A1 hard~x R^2 high on law data",
              hard_x["r2"] is not None and hard_x["r2"] > 0.9,
              _f(hard_x["r2"]))

        a2 = a2_refit(rows_hard)
        pooled = a2["pooled"]["wls"]
        check("A2 pooled WLS exists", pooled is not None)
        if pooled:
            check("A2 pooled slope ~ 1/R",
                  abs(pooled["slope"] * R_TRUE - 1.0) < 0.15,
                  f"slope={pooled['slope']:.4g} expect ~{1/R_TRUE:.4g}")
            check("A2 sigma_slope finite and positive",
                  pooled["sigma_slope"] > 0)
        check("A2 spread computed", a2["spread"] is not None, _f(a2["spread"]))
        check("A2 elastic check present",
              a2["elastic_transform_check"] is not None)

        frozen = freeze_hard_slope(a2, d / "analysis" / "stage1_hardslope_frozen.json")
        check("frozen slope file written",
              (d / "analysis" / "stage1_hardslope_frozen.json").exists())
        check("frozen S matches pooled WLS",
              frozen["S_frozen"] == pooled["slope"])

        md = a1_markdown(a1, rows_soft, rows_hard) + "\n" + a2_markdown(a2)
        check("markdown renders",
              "# A1" in md and "# A2" in md
              and "Recomputed spread" in md
              and "elastic_transform deviation" in md
              and md.count("|") > 40)

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}")
    return 0 if not failures else 1


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
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows_soft = e1_all_rows(results_dir, args.severity, args.seed, hard_only=False)
    rows_hard = e1_all_rows(results_dir, args.severity, args.seed, hard_only=True)

    a1 = a1_attribution(rows_soft, rows_hard)
    a2 = a2_refit(rows_hard)
    frozen = freeze_hard_slope(a2, analysis_dir / "stage1_hardslope_frozen.json")

    md = a1_markdown(a1, rows_soft, rows_hard) + "\n" + a2_markdown(a2)
    (analysis_dir / "stage1_softboundary.md").write_text(md, encoding="utf-8")
    (analysis_dir / "stage1_followup.json").write_text(
        json.dumps({"a1": a1, "a2": a2, "frozen": frozen}, indent=2,
                   default=str), encoding="utf-8")

    print(md)
    print(f"[frozen] S_frozen = {_f(frozen['S_frozen'], 5)} "
          f"+/- {_f(frozen['sigma_slope'], 3)} "
          f"(n={frozen['n_points']}) -> analysis/stage1_hardslope_frozen.json")
    for name in ("stage1_softboundary.md", "stage1_followup.json",
                 "stage1_hardslope_frozen.json"):
        print(f"[saved] {analysis_dir / name}")
    print("\nPart A complete. Part B (forward confirmation) consumes the "
          "frozen slope: scripts/run_stage1_forward.py")


if __name__ == "__main__":
    main()
