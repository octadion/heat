"""
STAGE-1v2 FINAL DIAGNOSTIC — D5 (decisive mechanism check) + D3-EXT + S-FREEZE.
ZERO GPU; existing cycled-x15 JSONs only. heat==TFF.

D5  The law predicts collapse when drift exceeds R. If contrast's effective R
    is ~30% smaller (slope 1.646 vs pooled 1.149 => ratio ~0.70), its
    PRE-COLLAPSE DRIFT must be proportionally smaller. For every hard-
    collapsing run (all corruptions/etas, grid points below p*): max finite
    drift_l2 before the first NaN, plus drift at ~10 steps pre-NaN
    (robustness variant). Per-corruption medians and the contrast/others
    ratio.
    VERDICT (mechanical): R-MEASURED iff contrast's median pre-collapse drift
    is lower by 20-45% (ratio in [0.55, 0.80], consistent with 1.149/1.646
    ~= 0.70) AND the other three corruptions' medians agree within ~15%
    (max/min <= 1.15). Otherwise R-NOT-EXPLANATORY, with the actual pattern
    reported honestly.

D3-EXT  Margin rule extended to c in {1.8, 2.0, 2.2, 2.5} on the existing
    grids (same snapping), per c x delta_R in {0, +50%}: collapses,
    median/min acc ratio — overall AND restricted to the compliant
    corruptions (gaussian/elastic/impulse), quantifying the over-tethering
    cost that makes open-loop margin alone a bad deal (=> feedback needed).

S-FREEZE  S = the calibration-cells slope (gaussian+elastic), BRACKET-
    WEIGHTED (1/width^2), with uncertainty -> analysis/stage1v2_S_frozen.json
    {value, sigma, source_manifest}; any previous value in that file is
    recorded inside, never silently clobbered. Reference-only numbers
    (clearly labeled, never used for prediction): pooled-ex-contrast slope
    and the continual reference ~1.15.

Appends D5/D3-EXT/S-freeze sections to analysis/stage1v2_diagnosis.md and
writes analysis/stage1v2_diagnosis_final.json (with consumed manifest).

Usage:
  python scripts/diagnose_stage1_v2_final.py --results-dir <Drive dir>
  python scripts/diagnose_stage1_v2_final.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1v2_common as v2
from scripts.analysis_common import safe_float
from scripts.analyze_stage1 import e1_fits, _snap
from scripts.analyze_stage1_followup import wls_line
from scripts.analyze_stage1_v2 import v2_all_rows
from scripts.diagnose_stage1_v2 import d3_margin as _d3_margin_base  # reuse note
from scripts.analyze_stage1_v2 import continual_reference

C_EXT = [1.8, 2.0, 2.2, 2.5]
DELTAS = [0.0, 0.50]
COMPLIANT = ["gaussian_noise", "elastic_transform", "impulse_noise"]
D5_RATIO_LO, D5_RATIO_HI = 0.55, 0.80   # "lower by 20-45%"
D5_OTHERS_AGREE = 1.15                  # max/min of the other three medians


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1v2 final diagnostic "
                                            "(D5 + D3-EXT + S-freeze)")
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
# D5 — pre-collapse drift extraction.
# ----------------------------------------------------------------------------

def pre_collapse_drift(data):
    """(first_nan_step, max finite drift before NaN, drift at ~10 steps
    pre-NaN). None fields when not extractable."""
    rows = sorted(pc.stream_rows(data),
                  key=lambda r: (r.get("step") if r.get("step") is not None
                                 else 1 << 60))
    nan_step = pc.first_nonfinite_step(rows)
    if nan_step is None:
        return None, None, None
    max_drift, drift_m10 = None, None
    target = nan_step - 10
    for r in rows:
        s = r.get("step")
        if s is None or s >= nan_step:
            break
        d = safe_float(r.get("drift_l2"))
        if d is None:
            continue
        if max_drift is None or d > max_drift:
            max_drift = d
        if s <= target:
            drift_m10 = d           # last finite drift at/before nan-10
    return nan_step, max_drift, drift_m10


def d5_collect(results_dir, rows_hard, severity, seed, cycles):
    """Every hard-collapsing (NaN) run below p*: one record per run."""
    records = []
    for r in rows_hard:
        c, eta = r["corruption"], r["eta"]
        p_star = r.get("p_star")
        for p in r.get("p_values_run") or []:
            q = (r.get("per_p") or {}).get(pc.format_p(p), {})
            if not q.get("valid") or not q.get("collapsed"):
                continue
            if q.get("first_nan_step") is None:
                continue  # chance-only collapse: no NaN trajectory to measure
            if p_star is not None and p >= p_star:
                continue  # spec: grid points BELOW p*
            data = v2.load_run_v2(results_dir, v2.V2_ARCH, c, severity, seed,
                                  p=p, eta=eta, cycles=cycles)
            if data is None:
                continue
            nan_step, dmax, dm10 = pre_collapse_drift(data)
            if dmax is None:
                continue
            records.append({"corruption": c, "eta": eta, "p": p,
                            "first_nan_step": nan_step,
                            "pre_collapse_drift_max": dmax,
                            "drift_at_nan_minus_10": dm10})
    return records


def d5_verdict(records):
    """Mechanical D5 judgement from the per-run records."""
    by_corr = {}
    for rec in records:
        by_corr.setdefault(rec["corruption"], []).append(
            rec["pre_collapse_drift_max"])
    med = {c: statistics.median(vs) for c, vs in by_corr.items() if vs}
    med_m10 = {}
    for c in by_corr:
        vs = [r["drift_at_nan_minus_10"] for r in records
              if r["corruption"] == c and r["drift_at_nan_minus_10"] is not None]
        if vs:
            med_m10[c] = statistics.median(vs)

    out = {"n_records": len(records), "median_by_corruption": med,
           "median_by_corruption_nanminus10": med_m10}
    others = [c for c in v2.V2_CORRUPTIONS if c != "contrast" and c in med]
    if "contrast" not in med or len(others) < 2:
        out["verdict"] = "UNRESOLVED"
        out["why"] = (f"insufficient collapsing runs (contrast in data: "
                      f"{'yes' if 'contrast' in med else 'NO'}; others: "
                      f"{others}).")
        return out
    pooled_others = statistics.median(
        [r["pre_collapse_drift_max"] for r in records
         if r["corruption"] != "contrast"])
    ratio = med["contrast"] / pooled_others if pooled_others else None
    o_meds = [med[c] for c in others]
    others_agree_ratio = max(o_meds) / min(o_meds) if min(o_meds) > 0 else None
    out.update({"contrast_over_others_ratio": ratio,
                "others_maxmin_ratio": others_agree_ratio,
                "criterion": f"R-MEASURED iff ratio in [{D5_RATIO_LO}, "
                             f"{D5_RATIO_HI}] (20-45% lower; slope ratio "
                             f"1.149/1.646 ~= 0.70) AND others max/min <= "
                             f"{D5_OTHERS_AGREE}"})
    if (ratio is not None and D5_RATIO_LO <= ratio <= D5_RATIO_HI
            and others_agree_ratio is not None
            and others_agree_ratio <= D5_OTHERS_AGREE):
        out["verdict"] = "R-MEASURED"
        out["why"] = (f"contrast median pre-collapse drift is "
                      f"{(1 - ratio) * 100:.0f}% lower (ratio {ratio:.3f}) and "
                      f"the other three agree (max/min "
                      f"{others_agree_ratio:.3f}).")
    else:
        out["verdict"] = "R-NOT-EXPLANATORY"
        out["why"] = (f"actual pattern: contrast/others ratio = {_f(ratio, 4)} "
                      f"(needed [{D5_RATIO_LO}, {D5_RATIO_HI}]); others "
                      f"max/min = {_f(others_agree_ratio, 4)} (needed <= "
                      f"{D5_OTHERS_AGREE}).")
    return out


# ----------------------------------------------------------------------------
# D3-EXT — extended margin sweep with over-tethering cost.
# ----------------------------------------------------------------------------

def d3_ext(rows_hard, fits_hard):
    calib = fits_hard["calibration_only"]["fit"]
    if calib is None:
        return {"error": "calibration fit unavailable"}
    slope = calib["slope"]
    table = []
    for c_margin in C_EXT:
        for delta in DELTAS:
            per_cell = []
            for r in rows_hard:
                x, grid = r.get("eta_times_gbar"), r.get("p_values_run") or []
                per_p = r.get("per_p") or {}
                if x is None or not grid:
                    continue
                p_prime = _snap(c_margin * slope * x / (1.0 + delta), grid)
                q = per_p.get(pc.format_p(p_prime), {})
                best = max((qq.get("mean_acc") for qq in per_p.values()
                            if qq.get("valid") and qq.get("mean_acc") is not None),
                           default=None)
                per_cell.append({
                    "corruption": r["corruption"], "eta": r["eta"],
                    "p_prime": p_prime, "collapsed": bool(q.get("collapsed")),
                    "acc_ratio": (q.get("mean_acc") / best
                                  if q.get("mean_acc") is not None and best
                                  else None)})

            def stats(cells):
                ratios = sorted(x["acc_ratio"] for x in cells
                                if x["acc_ratio"] is not None)
                return {"n": len(cells),
                        "n_collapse": sum(1 for x in cells if x["collapsed"]),
                        "median_ratio": ratios[len(ratios) // 2] if ratios else None,
                        "min_ratio": ratios[0] if ratios else None}
            table.append({"c": c_margin, "delta_R": delta,
                          "overall": stats(per_cell),
                          "compliant_only": stats(
                              [x for x in per_cell
                               if x["corruption"] in COMPLIANT])})
    return {"slope_calibration": slope, "table": table,
            "note": "compliant_only = gaussian/elastic/impulse; their ratio "
                    "at high c quantifies the over-tethering cost of an "
                    "open-loop margin (why feedback is needed)."}


# ----------------------------------------------------------------------------
# S-FREEZE — bracket-weighted calibration slope for Stage 1b.
# ----------------------------------------------------------------------------

def s_freeze(results_dir, rows_hard, severity, seed):
    pts, consumed_rows = [], []
    for r in rows_hard:
        if r["corruption"] not in v2.CALIBRATION_CORRUPTIONS:
            continue
        lo, hi = r["p_star_bracket_low"], r["p_star_bracket_high"]
        if (r["p_star"] is None or r["eta_times_gbar"] is None
                or lo is None or hi is None or hi <= lo):
            continue
        pts.append((r["eta_times_gbar"], r["p_star"], 1.0 / (hi - lo) ** 2))
        consumed_rows.append(r)
    w = wls_line([x for x, _, _ in pts], [y for _, y, _ in pts],
                 [wt for _, _, wt in pts]) if len(pts) >= 2 else None
    if w is None or w.get("slope") is None:
        raise SystemExit("[ABORT — VOID] bracket-weighted calibration slope "
                         "not computable; refusing to freeze a null S.")

    # Reference-only numbers (clearly labeled; never used for prediction).
    ex_pts = []
    for r in rows_hard:
        if r["corruption"] == "contrast":
            continue
        lo, hi = r["p_star_bracket_low"], r["p_star_bracket_high"]
        if (r["p_star"] is None or r["eta_times_gbar"] is None
                or lo is None or hi is None or hi <= lo):
            continue
        ex_pts.append((r["eta_times_gbar"], r["p_star"], 1.0 / (hi - lo) ** 2))
    w_ex = wls_line([x for x, _, _ in ex_pts], [y for _, y, _ in ex_pts],
                    [wt for _, _, wt in ex_pts]) if len(ex_pts) >= 2 else None
    ref_cont = continual_reference(results_dir, severity, seed, hard_only=True)

    out_path = Path(results_dir) / "analysis" / "stage1v2_S_frozen.json"
    previous = None
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None
    manifest = v2.consumed_manifest(
        [Path(f) for r in consumed_rows for f in r.get("consumed_files", [])])
    payload = {
        "value": w["slope"], "sigma": w["sigma_slope"],
        "r2_weighted": w["r2_weighted"], "n_points": w["n"],
        "definition": "calibration cells (gaussian_noise + elastic_transform),"
                      " HARD family, bracket-weighted WLS (1/width^2); the "
                      "slope actually used by E4, re-weighted as "
                      "pre-registered for Stage 1b",
        "reference_only": {
            "pooled_ex_contrast_slope": (w_ex["slope"] if w_ex else None),
            "pooled_ex_contrast_sigma": (w_ex["sigma_slope"] if w_ex else None),
            "continual_reference_slope": (ref_cont["slope"] if ref_cont else None),
            "note": "NOT for prediction; context only.",
        },
        "previous_file_value": (previous or {}).get("S_frozen",
                                                    (previous or {}).get("value")),
        "source_manifest": manifest,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, w_ex, ref_cont


# ----------------------------------------------------------------------------
# Report (appended to stage1v2_diagnosis.md).
# ----------------------------------------------------------------------------

def build_md(d5_records, d5v, d3e, sfreeze, w_ex, ref_cont) -> str:
    L = ["", "---", "", "# FINAL DIAGNOSTIC — D5 + D3-EXT + S-FREEZE", "",
         "## D5 — decisive mechanism check (pre-collapse drift ~ R)", "",
         "| corruption | eta | p | first_nan_step | max pre-collapse ||delta|| "
         "| drift @ nan-10 |", "|" + "---|" * 6]
    for rec in sorted(d5_records, key=lambda r: (r["corruption"], r["eta"],
                                                 r["p"])):
        L.append("| " + " | ".join([
            rec["corruption"], _f(rec["eta"]), _f(rec["p"]),
            str(rec["first_nan_step"]), _f(rec["pre_collapse_drift_max"], 5),
            _f(rec["drift_at_nan_minus_10"], 5)]) + " |")
    L += ["", "| corruption | median pre-collapse drift | median @ nan-10 |",
          "|---|---|---|"]
    for c in v2.V2_CORRUPTIONS:
        L.append(f"| {c} | "
                 f"{_f(d5v['median_by_corruption'].get(c), 5)} | "
                 f"{_f(d5v['median_by_corruption_nanminus10'].get(c), 5)} |")
    L += ["", f"- contrast/others ratio: **"
          f"{_f(d5v.get('contrast_over_others_ratio'), 4)}** "
          f"(expected ~0.70 if the slope gap is a real R difference); others "
          f"max/min: {_f(d5v.get('others_maxmin_ratio'), 4)}",
          f"- criterion: {d5v.get('criterion', 'n/a')}",
          f"- **D5 VERDICT: {d5v['verdict']}** — {d5v['why']}", "",
          "## D3-EXT — extended margin sweep (over-tethering cost)", "",
          f"- {d3e.get('note')}", "",
          "| c | delta_R | collapses (all) | median ratio (all) | "
          "median ratio (compliant) | min ratio (compliant) |",
          "|" + "---|" * 6]
    for t in d3e.get("table", []):
        L.append(f"| {t['c']} | {t['delta_R']:+.0%} | "
                 f"{t['overall']['n_collapse']}/{t['overall']['n']} | "
                 f"{_f(t['overall']['median_ratio'], 4)} | "
                 f"{_f(t['compliant_only']['median_ratio'], 4)} | "
                 f"{_f(t['compliant_only']['min_ratio'], 4)} |")
    L += ["", "## S-FREEZE (pre-registered, for Stage 1b)", "",
          f"- **S = {_f(sfreeze['value'], 5)} +/- {_f(sfreeze['sigma'], 3)}** "
          f"(calibration cells, bracket-weighted, wR^2="
          f"{_f(sfreeze['r2_weighted'], 4)}, n={sfreeze['n_points']}) -> "
          f"analysis/stage1v2_S_frozen.json",
          f"- previous file value (recorded, replaced): "
          f"{_f(sfreeze.get('previous_file_value'))}",
          f"- reference only (NOT for prediction): pooled-ex-contrast slope = "
          f"{_f(w_ex['slope'], 4) if w_ex else 'n/a'} +/- "
          f"{_f(w_ex['sigma_slope'], 3) if w_ex else 'n/a'}; continual "
          f"reference = {_f(ref_cont['slope'], 4) if ref_cont else 'n/a'}",
          "", "**STOP — final diagnostic complete. The Stage 1b GO decision "
          "follows the D5 verdict.**"]
    return "\n".join(L) + "\n"


def run_final(results_dir: Path, severity, seed, etas, cycles):
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows_hard = v2_all_rows(results_dir, severity, seed, etas, cycles, True)
    fits_hard = e1_fits(rows_hard)

    d5_records = d5_collect(results_dir, rows_hard, severity, seed, cycles)
    d5v = d5_verdict(d5_records)
    d3e = d3_ext(rows_hard, fits_hard)
    sfreeze, w_ex, ref_cont = s_freeze(results_dir, rows_hard, severity, seed)

    md = build_md(d5_records, d5v, d3e, sfreeze, w_ex, ref_cont)
    diag = analysis_dir / "stage1v2_diagnosis.md"
    existing = diag.read_text(encoding="utf-8") if diag.exists() else ""
    diag.write_text(existing + md, encoding="utf-8")
    (analysis_dir / "stage1v2_diagnosis_final.json").write_text(
        json.dumps({"d5_records": d5_records, "d5_verdict": d5v,
                    "d3_ext": d3e, "s_frozen": sfreeze},
                   indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"[saved] {diag} (appended)")
    print(f"[saved] {analysis_dir / 'stage1v2_diagnosis_final.json'}")
    print(f"[saved] {analysis_dir / 'stage1v2_S_frozen.json'}")
    return {"d5": d5v, "d3_ext": d3e, "s": sfreeze}


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

    print("=== stage1v2 FINAL diagnostic self-test ===")

    # Unit-test the D5 judge on fabricated medians (both branches).
    recs_yes = ([{"corruption": c, "eta": 1e-3, "p": 0.0, "first_nan_step": 601,
                  "pre_collapse_drift_max": 10.0, "drift_at_nan_minus_10": 9.5}
                 for c in COMPLIANT for _ in range(3)]
                + [{"corruption": "contrast", "eta": 1e-3, "p": 0.0,
                    "first_nan_step": 601, "pre_collapse_drift_max": 7.0,
                    "drift_at_nan_minus_10": 6.6} for _ in range(3)])
    v_yes = d5_verdict(recs_yes)
    check("D5 judge: 30%-lower contrast => R-MEASURED",
          v_yes["verdict"] == "R-MEASURED", v_yes["why"])
    recs_no = [dict(r, pre_collapse_drift_max=10.0) for r in recs_yes]
    v_no = d5_verdict(recs_no)
    check("D5 judge: equal drifts => R-NOT-EXPLANATORY",
          v_no["verdict"] == "R-NOT-EXPLANATORY", v_no["why"])

    # End-to-end on a synthetic campaign: contrast with smaller effective R
    # encoded in the DRIFT trajectory (drift ~ R_eff * step/601).
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0
    R_EFF = {"gaussian_noise": 10.0, "elastic_transform": 10.5,
             "impulse_noise": 9.8, "contrast": 7.0}

    def synth(c, eta, p):
        d = _synth_cycled(c, eta, p, GBARS[c], R_TRUE)
        for r in d["results"]["summary"]["heat"]["stream_diagnostics"]:
            r["drift_l2"] = R_EFF[c] * min(r["step"], 601) / 601.0
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
                                     for f in (0.9, 1.1, 1.3)]))
                for p in grid:
                    v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(synth(c, eta, p)))
        res = run_final(d, 5, 42, v2.V2_ETAS, v2.CYCLES)
        check("D5 e2e: contrast drift ratio detected as R-MEASURED",
              res["d5"]["verdict"] == "R-MEASURED",
              json.dumps({k: res["d5"][k] for k in
                          ("verdict", "contrast_over_others_ratio",
                           "others_maxmin_ratio")}, default=str))
        check("D3-EXT covers 4 c x 2 deltas",
              len(res["d3_ext"]["table"]) == 8)
        check("D3-EXT reports compliant-only cost",
              all("compliant_only" in t for t in res["d3_ext"]["table"]))
        check("S frozen bracket-weighted ~ 1/R",
              abs(res["s"]["value"] * R_TRUE - 1.0) < 0.25,
              _f(res["s"]["value"]))
        check("S file has sigma + source_manifest",
              res["s"]["sigma"] > 0 and len(res["s"]["source_manifest"]) > 0)
        check("diagnosis md appended with FINAL section",
              "FINAL DIAGNOSTIC" in (d / "analysis" /
                                     "stage1v2_diagnosis.md").read_text(
                                         encoding="utf-8"))

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
    run_final(Path(args.results_dir), args.severity, args.seed, args.etas,
              args.cycles)


if __name__ == "__main__":
    main()
