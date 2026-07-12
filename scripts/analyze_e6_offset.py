"""
E6 POST-VERDICT ANALYSIS (O1-O3) — zero GPU; E7 untouched. heat==TFF.

The pre-registered E6 verdict (DIFFERENT-CURVE) STANDS and is reported
verbatim; this analysis tests the theory-first explanation of its structure
(slope mechanism-invariant, mechanisms differing by a one-signed constant).

O1  STOCHASTIC EXCURSION PREMIUM — derived ONLY from Bernoulli runs on disk;
    no anchor data enters the derivation. Operationalization (stated so the
    lead can swap in the closed-form Theorem-2/Lemma-6 constants, which live
    in the paper draft, not this repo):

      Near the boundary the Bernoulli tether balances drift growth in the
      MEAN: D_bar(p) ~ eta*||g_bar||/p, and collapse is triggered when
      EXCURSIONS (renewal fluctuations of the Bernoulli restore) touch the
      boundary R, i.e. D_peak = D_bar * (1 + rho) = R with rho the relative
      stationary excursion. The deterministic anchor has no excursion term
      (its contraction is exact each step), so at the same x it needs LESS
      tether by the amount the excursion consumes:

        offset_premium(cell) = p*_bern(cell) * (D_peak - D_bar) / D_peak

      with D_bar / D_peak measured over the stationary windows (cycle-aware,
      NaN cycles excluded) of the TIGHTEST STABLE Bernoulli run (bracket_high)
      of that cell, and p*_bern the cell's hard boundary. A 2-sigma variant
      (p* * 2*sigma_D / (D_bar + 2*sigma_D)) is reported for robustness.
      offset_theory = median over cells (per-cell values shown so constancy
      is checkable, not assumed).

O2  Re-evaluate all 4 anchor cells under
      (eta*lambda)_hat' = S_frozen * eta * ||g_bar|| - offset_theory
    (same +/-25%-or-in-bracket rule). Label exactly one of:
    OFFSET-EXPLAINED (>= 3/4 hits) / OFFSET-UNEXPLAINED.

O3  Anchor soft-vs-hard boundary gap per cell — one table, no interpretation.

Appends to analysis/e6_crossmech.md; writes analysis/e6_offset.json.
The identity decision is made by the lead AFTER the E7 verdict + O2.

Usage:
  python scripts/analyze_e6_offset.py --results-dir <Drive dir>
  python scripts/analyze_e6_offset.py --self-test
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
from scripts.analysis_common import stationary_rows, row_metric

E6_CELLS = [("gaussian_noise", 2e-3), ("gaussian_noise", 1e-3),
            ("gaussian_noise", 5e-4), ("elastic_transform", 1e-3)]
PRED_TOL = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="E6 O1-O3 offset analysis")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--severity", type=int, default=v2.V2_SEVERITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cycles", type=int, default=v2.CYCLES)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# ----------------------------------------------------------------------------
# O1 — excursion premium from Bernoulli runs only.
# ----------------------------------------------------------------------------

def stationary_drift_stats(data):
    """(D_bar, sigma_D, D_peak) over the cycle-aware stationary windows,
    NaN cycles excluded."""
    vals = []
    for cyc in v2.group_rows_by_cycle(pc.stream_rows(data)):
        if pc.block_has_nonfinite(cyc):
            continue
        for r in stationary_rows(cyc, v2.STATIONARY_LO, v2.STATIONARY_HI):
            d = row_metric(r, "drift_l2")
            if d is not None:
                vals.append(d)
    if len(vals) < 10:
        return None, None, None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var ** 0.5, max(vals)


def o1_premium(results_dir, severity, seed, cycles):
    """Per-cell excursion premium from the Bernoulli side of each E6 cell."""
    out = []
    for corruption, eta in E6_CELLS:
        row = v2.analyze_cell_v2(results_dir, v2.V2_ARCH, corruption,
                                 severity, eta, seed, hard_only=True,
                                 cycles=cycles)
        p_star, p_hi = row["p_star"], row["p_star_bracket_high"]
        rec = {"cell": f"{corruption}_lr{pc.format_p(eta)}",
               "corruption": corruption, "eta": eta,
               "p_star_bern": p_star, "p_edge_stable": p_hi}
        if p_star is None or p_hi is None:
            rec["note"] = "unresolved Bernoulli boundary"
            out.append(rec)
            continue
        data = v2.load_run_v2(results_dir, v2.V2_ARCH, corruption, severity,
                              seed, p=p_hi, eta=eta, cycles=cycles)
        if data is None:
            rec["note"] = "edge run missing"
            out.append(rec)
            continue
        d_bar, sigma_d, d_peak = stationary_drift_stats(data)
        if not d_bar or not d_peak:
            rec["note"] = "insufficient stationary drift data"
            out.append(rec)
            continue
        rec.update({
            "D_bar": d_bar, "sigma_D": sigma_d, "D_peak": d_peak,
            "excursion_rel": (d_peak - d_bar) / d_peak,
            "premium_peak": p_star * (d_peak - d_bar) / d_peak,
            "premium_2sigma": p_star * (2 * sigma_d)
            / (d_bar + 2 * sigma_d) if sigma_d is not None else None,
        })
        out.append(rec)
    prem = [r["premium_peak"] for r in out if r.get("premium_peak") is not None]
    prem2 = [r["premium_2sigma"] for r in out
             if r.get("premium_2sigma") is not None]
    return {"cells": out,
            "offset_theory": statistics.median(prem) if prem else None,
            "offset_theory_2sigma": statistics.median(prem2) if prem2 else None,
            "n_cells_used": len(prem)}


# ----------------------------------------------------------------------------
# O2 — re-evaluate the anchor cells with the theory offset.
# ----------------------------------------------------------------------------

def o2_reevaluate(e6_rows, S, offset):
    cells = []
    for r in e6_rows:
        x = r.get("x_eta_gbar")
        el_star = r.get("eta_lambda_star_hard")
        blo, bhi = (r.get("bracket_lambda") or [None, None])[:2]
        eta = r.get("eta")
        if x is None:
            cells.append({"cell": r.get("cell"), "note": "no x"})
            continue
        hat = S * x - offset
        within = (el_star is not None and hat > 0
                  and abs(el_star - hat) / hat <= PRED_TOL)
        in_bracket = (blo is not None and bhi is not None and eta is not None
                      and blo * eta <= hat <= bhi * eta)
        cells.append({"cell": r.get("cell"), "x_eta_gbar": x,
                      "hat_original": r.get("eta_lambda_hat"),
                      "hat_offset": hat, "eta_lambda_star_hard": el_star,
                      "within_25pct": bool(within),
                      "in_bracket": bool(in_bracket),
                      "hit": bool(within or in_bracket),
                      "hit_original": bool(r.get("cell_hit"))})
    hits = sum(1 for c in cells if c.get("hit"))
    label = "OFFSET-EXPLAINED" if hits >= 3 else "OFFSET-UNEXPLAINED"
    return {"cells": cells, "hits": hits, "n": len(cells), "label": label}


# ----------------------------------------------------------------------------
# O3 — anchor soft-vs-hard gap.
# ----------------------------------------------------------------------------

def o3_gap(e6_rows):
    rows = []
    for r in e6_rows:
        h, s = r.get("eta_lambda_star_hard"), r.get("eta_lambda_star_soft")
        rows.append({"cell": r.get("cell"), "eta_lambda_hard": h,
                     "eta_lambda_soft": s,
                     "gap": (h - s) if (h is not None and s is not None)
                     else None,
                     "gap_rel": ((h - s) / h if h else None)
                     if (h is not None and s is not None) else None})
    return rows


def build_md(preds, o1, o2, o3, S) -> str:
    L = ["", "---", "", "# E6 POST-VERDICT ANALYSIS (O1-O3)", "",
         "Pre-registered verdict **DIFFERENT-CURVE** stands (reported "
         "verbatim above). This section tests the theory-first constant-"
         "offset explanation. NOTE: the closed-form Theorem-2/Lemma-6 "
         "constants live in the paper draft; the operationalization below "
         "uses the empirically measured stationary excursion of the "
         "BERNOULLI runs only (formula in the script header) — no anchor "
         "data entered the derivation.", "",
         "## O1 — stochastic excursion premium (Bernoulli-only)", "",
         "| cell | p*_bern | edge p | D_bar | sigma_D | D_peak | "
         "excursion rel | premium(peak) | premium(2sigma) |",
         "|" + "---|" * 9]
    for r in o1["cells"]:
        L.append("| " + " | ".join([
            r["cell"], _f(r.get("p_star_bern")), _f(r.get("p_edge_stable")),
            _f(r.get("D_bar"), 4), _f(r.get("sigma_D"), 3),
            _f(r.get("D_peak"), 4), _f(r.get("excursion_rel"), 3),
            _f(r.get("premium_peak"), 4), _f(r.get("premium_2sigma"), 4)])
            + " |")
    L += ["", f"**offset_theory = {_f(o1['offset_theory'], 4)}** (median of "
          f"the peak-form premium over {o1['n_cells_used']} cells; 2-sigma "
          f"variant {_f(o1['offset_theory_2sigma'], 4)}). Fitted intercept "
          "for comparison: -0.0036 (from the pre-registered fit; NOT used in "
          "the derivation).", "",
          "## O2 — anchor cells re-evaluated with the theory offset", "",
          f"(eta*lambda)_hat' = S_frozen*eta*||g_bar|| - offset_theory, "
          f"S={_f(S, 5)}, offset={_f(o1['offset_theory'], 4)}.", "",
          "| cell | hat (original) | hat' (offset) | ηλ*_hard | within 25% | "
          "in bracket | hit' | hit (original) |", "|" + "---|" * 8]
    for c in o2["cells"]:
        L.append("| " + " | ".join([
            str(c.get("cell")), _f(c.get("hat_original")),
            _f(c.get("hat_offset")), _f(c.get("eta_lambda_star_hard")),
            str(c.get("within_25pct")), str(c.get("in_bracket")),
            "YES" if c.get("hit") else "no",
            "YES" if c.get("hit_original") else "no"]) + " |")
    L += ["", f"**O2 RESULT: {o2['label']}** ({o2['hits']}/{o2['n']} with the "
          "theory-derived offset; criterion >= 3/4).", "",
          "## O3 — anchor soft-vs-hard boundary gap (no interpretation)", "",
          "| cell | ηλ*_hard | ηλ*_soft | gap | gap/hard |",
          "|" + "---|" * 5]
    for r in o3:
        L.append("| " + " | ".join([
            str(r["cell"]), _f(r["eta_lambda_hard"]), _f(r["eta_lambda_soft"]),
            _f(r["gap"], 4), _f(r["gap_rel"], 3)]) + " |")
    L += ["", "**STOP — O1-O3 complete. The identity decision is made by the "
          "lead after the E7 verdict + O2 are both in.**"]
    return "\n".join(L) + "\n"


def run(results_dir: Path, severity, seed, cycles):
    analysis_dir = results_dir / "analysis"
    cm = analysis_dir / "e6_crossmech.json"
    if not cm.exists():
        raise SystemExit(f"[fatal] {cm} not found — run the E6 verdict first.")
    payload = json.loads(cm.read_text(encoding="utf-8"))
    e6_rows, preds = payload["rows"], payload["predictions"]
    S = float(preds["S_frozen"])

    o1 = o1_premium(results_dir, severity, seed, cycles)
    if o1["offset_theory"] is None:
        raise SystemExit("[ABORT] excursion premium not computable from the "
                         "Bernoulli runs — nothing to test.")
    o2 = o2_reevaluate(e6_rows, S, o1["offset_theory"])
    o3 = o3_gap(e6_rows)

    md = build_md(preds, o1, o2, o3, S)
    target = analysis_dir / "e6_crossmech.md"
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    target.write_text(existing + md, encoding="utf-8")
    (analysis_dir / "e6_offset.json").write_text(
        json.dumps({"o1": o1, "o2": o2, "o3": o3,
                    "S_frozen": S,
                    "fitted_intercept_reference_only": -0.0036},
                   indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"[saved] {target} (appended)")
    print(f"[saved] {analysis_dir / 'e6_offset.json'}")
    return {"o1": o1, "o2": o2, "o3": o3}


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

    print("=== E6 offset (O1-O3) self-test ===")
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0}
    R_TRUE = 12.0
    S = 1.0 / R_TRUE
    EXC = 0.15   # 15% relative excursion => premium ~= 0.15 * p*

    def synth_bern(c, eta, p):
        d = _synth_cycled(c, eta, p, GBARS[c], R_TRUE)
        rows = d["results"]["summary"]["heat"]["stream_diagnostics"]
        for i, r in enumerate(rows):
            base = 2.0
            # oscillating stationary drift with a known peak-over-mean
            r["drift_l2"] = base * (1.0 + (EXC * 2.2 if (i % 37) == 0
                                           else EXC * 0.1 * ((i % 5) - 2)))
        return d

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "analysis").mkdir()
        # Bernoulli side of the 4 E6 cells.
        for corruption, eta in E6_CELLS:
            gbar = GBARS[corruption]
            v2.run_output_path_v2(d, v2.V2_ARCH, corruption, 5, 42,
                                  source=True, eta=eta).write_text(
                json.dumps(_synthetic_source(corruption)))
            pst = eta * gbar / R_TRUE
            for p in sorted({round(pst * f, 6) for f in (0.9, 1.1)}
                            | set(pc.scaled_p_grid(eta))):
                v2.run_output_path_v2(d, v2.V2_ARCH, corruption, 5, 42, p=p,
                                      eta=eta).write_text(
                    json.dumps(synth_bern(corruption, eta, p)))
        # Fabricate the E6 verdict artifact: anchor stars = S*x - offset_true.
        # offset_true chosen to match the excursion-premium construction.
        rows_probe = o1_premium(d, 5, 42, v2.CYCLES)
        offset_true = rows_probe["offset_theory"]
        e6_rows = []
        for corruption, eta in E6_CELLS:
            x = eta * GBARS[corruption]
            star = S * x - offset_true
            e6_rows.append({"cell": f"{corruption}_lr{pc.format_p(eta)}",
                            "corruption": corruption, "eta": eta,
                            "x_eta_gbar": x, "eta_lambda_hat": S * x,
                            "eta_lambda_star_hard": star,
                            "eta_lambda_star_soft": star * 0.98,
                            "bracket_lambda": [star / eta * 0.97,
                                               star / eta * 1.03],
                            "cell_hit": False})
        (d / "analysis" / "e6_crossmech.json").write_text(json.dumps(
            {"rows": e6_rows, "verdict": {"verdict": "DIFFERENT-CURVE"},
             "predictions": {"S_frozen": S}}))
        (d / "analysis" / "e6_crossmech.md").write_text("# E6 verdict\n")

        res = run(d, 5, 42, v2.CYCLES)
        o1, o2, o3 = res["o1"], res["o2"], res["o3"]
        check("O1 premium computed on 4 cells", o1["n_cells_used"] == 4)
        check("O1 offset positive and premium-shaped",
              o1["offset_theory"] is not None and o1["offset_theory"] > 0,
              _f(o1["offset_theory"]))
        check("O2 flips to OFFSET-EXPLAINED with the derived offset",
              o2["label"] == "OFFSET-EXPLAINED" and o2["hits"] >= 3,
              json.dumps(o2, default=str)[:200])
        check("O2 originals were misses (offset was needed)",
              all(not c["hit_original"] for c in o2["cells"]))
        check("O3 table has 4 gap rows",
              len(o3) == 4 and all(r["gap"] is not None for r in o3))
        md = (d / "analysis" / "e6_crossmech.md").read_text(encoding="utf-8")
        check("md appended after the verdict", "O1 — stochastic excursion"
              in md and md.startswith("# E6 verdict"))

        # Control: with a negligible offset the label must be UNEXPLAINED.
        o2_null = o2_reevaluate(e6_rows, S, offset_true * 0.05)
        check("O2 control: tiny offset => OFFSET-UNEXPLAINED",
              o2_null["label"] == "OFFSET-UNEXPLAINED",
              json.dumps({"hits": o2_null["hits"]}))

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
    run(Path(args.results_dir), args.severity, args.seed, args.cycles)


if __name__ == "__main__":
    main()
