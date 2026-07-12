"""
D5b — ESCAPE-POINT analysis (parallel, NON-BLOCKING; never gates Stage 1b).
ZERO GPU; existing cycled-x15 JSONs only. heat==TFF.

Replaces the explosion-phase metric: for each hard-collapsing (NaN) run on
disk, estimate the drift level at ESCAPE — the departure from the stationary
plateau — defined as the FIRST step where drift_l2 exceeds 3x that cell's
stable-run stationary drift (the cell's p_ref run, cycled window). Reported
per run: escape step, drift level at escape, first-NaN step, and the
escape->NaN gap. Then per-corruption medians and the contrast/others ratio.

A secondary robustness variant is also reported: the first step of PERSISTENT
log-drift growth (5 consecutive steps with increasing drift after the
stationary window). Purely descriptive — no verdict gates on this.

Usage:
  python scripts/analyze_d5b_escape.py --results-dir <Drive dir>
  python scripts/analyze_d5b_escape.py --self-test
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
from scripts.analyze_stage1_v2 import v2_all_rows

ESCAPE_FACTOR = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="D5b escape-point analysis")
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


def escape_point(rows, drift_ref):
    """(escape_step, drift_at_escape) for the 3x-plateau rule; plus the
    persistent-growth variant (first step opening >=5 strictly-increasing
    finite drift values). None fields when not found."""
    ordered = sorted(rows, key=lambda r: (r.get("step")
                                          if r.get("step") is not None
                                          else 1 << 60))
    nan_step = pc.first_nonfinite_step(ordered)
    esc_step, esc_level = None, None
    if drift_ref and drift_ref > 0:
        thr = ESCAPE_FACTOR * drift_ref
        for r in ordered:
            s = r.get("step")
            if nan_step is not None and s is not None and s >= nan_step:
                break
            d = safe_float(r.get("drift_l2"))
            if d is not None and d > thr:
                esc_step, esc_level = s, d
                break
    # persistent-growth variant
    grow_step = None
    run_len, prev, start = 0, None, None
    for r in ordered:
        s = r.get("step")
        if nan_step is not None and s is not None and s >= nan_step:
            break
        d = safe_float(r.get("drift_l2"))
        if d is None:
            run_len, prev = 0, None
            continue
        if prev is not None and d > prev:
            if run_len == 0:
                start = s
            run_len += 1
            if run_len >= 5 and s is not None and s > v2.STATIONARY_HI:
                grow_step = start
                break
        else:
            run_len = 0
        prev = d
    return nan_step, esc_step, esc_level, grow_step


def collect(results_dir, severity, seed, etas, cycles):
    rows_hard = v2_all_rows(results_dir, severity, seed, etas, cycles, True)
    records = []
    for r in rows_hard:
        c, eta, p_ref = r["corruption"], r["eta"], r.get("p_ref_used")
        drift_ref = None
        if p_ref is not None:
            ref_run = v2.load_run_v2(results_dir, v2.V2_ARCH, c, severity,
                                     seed, p=p_ref, eta=eta, cycles=cycles)
            if ref_run is not None:
                drift_ref = v2.drift_stationary_cycled(ref_run)
        for p in r.get("p_values_run") or []:
            q = (r.get("per_p") or {}).get(pc.format_p(p), {})
            if (not q.get("valid") or not q.get("collapsed")
                    or q.get("first_nan_step") is None):
                continue
            data = v2.load_run_v2(results_dir, v2.V2_ARCH, c, severity, seed,
                                  p=p, eta=eta, cycles=cycles)
            if data is None:
                continue
            nan_step, esc_step, esc_level, grow_step = escape_point(
                pc.stream_rows(data), drift_ref)
            records.append({
                "corruption": c, "eta": eta, "p": p,
                "drift_ref_stationary": drift_ref,
                "escape_step_3x": esc_step, "escape_drift_level": esc_level,
                "persistent_growth_step": grow_step,
                "first_nan_step": nan_step,
                "escape_to_nan_gap": (nan_step - esc_step
                                      if nan_step is not None
                                      and esc_step is not None else None),
            })
    return records


def summarize(records):
    med = {}
    for c in v2.V2_CORRUPTIONS:
        vals = [r["escape_drift_level"] for r in records
                if r["corruption"] == c and r["escape_drift_level"] is not None]
        if vals:
            med[c] = statistics.median(vals)
    others_vals = [r["escape_drift_level"] for r in records
                   if r["corruption"] != "contrast"
                   and r["escape_drift_level"] is not None]
    ratio = (med.get("contrast") / statistics.median(others_vals)
             if "contrast" in med and others_vals else None)
    return {"median_escape_level_by_corruption": med,
            "contrast_over_others_ratio": ratio,
            "n_records": len(records)}


def build_md(records, summ) -> str:
    L = ["# D5b — escape-point analysis (non-gating, descriptive)", "",
         f"Escape rule: first step with drift_l2 > {ESCAPE_FACTOR}x the "
         "cell's p_ref stationary drift (cycled window). Robustness variant: "
         "first persistent log-drift growth (>=5 increasing steps past the "
         "stationary window). This REPLACES the explosion-phase metric and "
         "never gates Stage 1b.", "",
         "| corruption | eta | p | drift_ref | escape step | escape level | "
         "growth step | first NaN | gap |", "|" + "---|" * 9]
    for r in sorted(records, key=lambda x: (x["corruption"], x["eta"], x["p"])):
        L.append("| " + " | ".join([
            r["corruption"], _f(r["eta"]), _f(r["p"]),
            _f(r["drift_ref_stationary"], 4), _f(r["escape_step_3x"]),
            _f(r["escape_drift_level"], 5), _f(r["persistent_growth_step"]),
            _f(r["first_nan_step"]), _f(r["escape_to_nan_gap"])]) + " |")
    L += ["", "| corruption | median escape level |", "|---|---|"]
    for c, m in summ["median_escape_level_by_corruption"].items():
        L.append(f"| {c} | {_f(m, 5)} |")
    L += ["", f"- contrast/others escape-level ratio: "
          f"**{_f(summ['contrast_over_others_ratio'], 4)}** "
          f"(n={summ['n_records']} collapsing runs)",
          "- reported alongside Stage 1b; never gating."]
    return "\n".join(L) + "\n"


def run(results_dir: Path, severity, seed, etas, cycles):
    records = collect(results_dir, severity, seed, etas, cycles)
    summ = summarize(records)
    md = build_md(records, summ)
    adir = results_dir / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "stage1v2_escape.md").write_text(md, encoding="utf-8")
    (adir / "stage1v2_escape.json").write_text(
        json.dumps({"records": records, "summary": summ}, indent=2,
                   default=str), encoding="utf-8")
    print(md)
    print(f"[saved] {adir / 'stage1v2_escape.md'}")
    return {"records": records, "summary": summ}


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

    print("=== D5b escape self-test ===")
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0,
             "impulse_noise": 14.0, "contrast": 6.5}
    R_TRUE = 12.0
    # Stationary drift PLATEAU per corruption: the escape level tracks
    # 3x the cell's plateau by the rule's definition, so a ~0.7x contrast
    # plateau must yield a ~0.7 contrast/others escape-level ratio.
    PLATEAU = {"gaussian_noise": 1.00, "elastic_transform": 1.05,
               "impulse_noise": 0.98, "contrast": 0.69}

    def synth(c, eta, p):
        d = _synth_cycled(c, eta, p, GBARS[c], R_TRUE)
        rows = d["results"]["summary"]["heat"]["stream_diagnostics"]
        collapsed = any(r["grad_l2"] != r["grad_l2"] for r in rows)
        P = PLATEAU[c]
        for r in rows:
            s = r["step"]
            if collapsed:
                # plateau until 500, then super-linear escape to NaN@601
                r["drift_l2"] = (P if s < 500
                                 else P * (1.0 + 11.0 * ((s - 500) / 101.0) ** 2))
            else:
                r["drift_l2"] = P * (1.0 + 0.0001 * s)
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
                                  + [round(pst * f, 6) for f in (0.9, 1.1)]))
                for p in grid:
                    v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(synth(c, eta, p)))
        res = run(d, 5, 42, v2.V2_ETAS, v2.CYCLES)
        summ = res["summary"]
        check("records collected", summ["n_records"] > 0,
              str(summ["n_records"]))
        check("escape levels detected per corruption",
              len(summ["median_escape_level_by_corruption"]) == 4,
              str(summ["median_escape_level_by_corruption"]))
        ratio = summ["contrast_over_others_ratio"]
        check("contrast/others ratio ~0.7 recovered",
              ratio is not None and 0.55 <= ratio <= 0.85, _f(ratio))
        recs = res["records"]
        check("escape precedes NaN with positive gap",
              all(r["escape_to_nan_gap"] is None or r["escape_to_nan_gap"] > 0
                  for r in recs))
        check("md written", (d / "analysis" / "stage1v2_escape.md").exists())

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
    run(Path(args.results_dir), args.severity, args.seed, args.etas,
        args.cycles)


if __name__ == "__main__":
    main()
