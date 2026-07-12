"""
STAGE-1B CLOSE-OUT (C1-C3) — zero GPU; no new runs. heat==TFF.

C1  Entropy-vs-energy ACCURACY comparison per E7 cell: entropy p=0 final-cycle
    accuracy vs source vs the energy drive at its p* (nearest stable grid p).
    DATA CAVEAT (stated in the artifact): under the cycled protocol the
    per_corruption summary key collides across cycles, so the persisted
    mean_accuracy == FINAL-CYCLE accuracy; the true stream mean is not
    persisted. Trajectory shape is therefore classified mechanically from
    final-cycle accuracy vs source (+/-2pt) plus the per-cycle drift trend:
      ADAPTING  : final >= source + 0.02 (and no NaN)
      STALLING  : |final - source| < 0.02 (and no NaN)
      DRIFTING  : final <= source - 0.02, or any NaN (collapse)
    Per-cycle stationary loss (energy field = entropy value for the entropy
    drive) first->last is reported as a supporting trend.

C2  Ensures O1-O3 (scripts/analyze_e6_offset.py) has run; reports
    OFFSET-EXPLAINED / OFFSET-UNEXPLAINED with the derived offset vs the
    fitted intercept (-0.0036).

C3  Consolidates Stage 1 + 1b into analysis/stage1_final.md: pre-registered
    verdicts VERBATIM, substantive findings beneath each, frozen constants
    with manifests. No identity language — that decision is the lead's.

Usage:
  python scripts/stage1_closeout.py --results-dir <Drive dir>
  python scripts/stage1_closeout.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1b_common as sb
from scripts import stage1v2_common as v2
from scripts.analysis_common import stationary_rows, row_metric

DRIVE = "entropy"
E7_CELLS = [(c, e) for c in sb.E7_CORRUPTIONS for e in v2.V2_ETAS]
STALL_BAND = 0.02


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1b close-out (C1-C3)")
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


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# C1 — entropy-vs-energy accuracy comparison.
# ----------------------------------------------------------------------------

def per_cycle_trend(data, key):
    """(first_cycle_mean, last_cycle_mean) of `key` over stationary windows,
    NaN cycles skipped."""
    means = []
    for cyc in v2.group_rows_by_cycle(pc.stream_rows(data)):
        if pc.block_has_nonfinite(cyc):
            continue
        vals = [row_metric(r, key) for r in
                stationary_rows(cyc, v2.STATIONARY_LO, v2.STATIONARY_HI)]
        vals = [x for x in vals if x is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    if not means:
        return None, None
    return means[0], means[-1]


def shape_of(final_acc, source_acc, has_nan):
    if has_nan:
        return "DRIFTING (collapse)"
    if final_acc is None or source_acc is None:
        return "n/a"
    if final_acc >= source_acc + STALL_BAND:
        return "ADAPTING"
    if final_acc <= source_acc - STALL_BAND:
        return "DRIFTING (below source)"
    return "STALLING (near source)"


def c1_compare(results_dir, severity, seed, cycles):
    rows = []
    for corruption, eta in E7_CELLS:
        source_acc = v2.source_acc_v2(results_dir, v2.V2_ARCH, corruption,
                                      severity, seed, eta, cycles=cycles)
        rec = {"cell": f"{corruption}_lr{pc.format_p(eta)}",
               "corruption": corruption, "eta": eta, "source_acc": source_acc}

        # Entropy p=0 (quenching drive, untethered).
        ent_path = sb.run_output_path_drive(results_dir, v2.V2_ARCH,
                                            corruption, severity, seed,
                                            DRIVE, 0.0, eta, cycles=cycles)
        ent = _load(ent_path) if ent_path.exists() else None
        if ent is not None:
            e_rows = pc.stream_rows(ent)
            has_nan = pc.first_nonfinite_step(e_rows) is not None
            final_acc = pc.heat_summary(ent).get("mean_accuracy")
            l0, l1 = per_cycle_trend(ent, "energy")
            d0, d1 = per_cycle_trend(ent, "drift_l2")
            rec.update({
                "entropy_p0_final_cycle_acc": final_acc,
                "entropy_p0_vs_source": (final_acc - source_acc
                                         if final_acc is not None
                                         and source_acc is not None else None),
                "entropy_p0_nan": has_nan,
                "entropy_p0_shape": shape_of(final_acc, source_acc, has_nan),
                "entropy_loss_first_to_last": (l0, l1),
                "entropy_drift_first_to_last": (d0, d1),
            })
        else:
            rec["entropy_p0_shape"] = "n/a (p=0 run missing)"

        # Energy drive at its p* (nearest STABLE measured grid p).
        v2row = v2.analyze_cell_v2(results_dir, v2.V2_ARCH, corruption,
                                   severity, eta, seed, hard_only=True,
                                   cycles=cycles)
        p_star = v2row.get("p_star")
        stable_ps = [p for p in (v2row.get("p_values_run") or [])
                     if (v2row.get("per_p") or {}).get(pc.format_p(p), {})
                     .get("valid")
                     and not v2row["per_p"][pc.format_p(p)].get("collapsed")]
        p_near = (min(stable_ps, key=lambda q: abs(q - p_star))
                  if p_star is not None and stable_ps else None)
        if p_near is not None:
            q = v2row["per_p"][pc.format_p(p_near)]
            rec.update({
                "energy_p_star": p_star, "energy_p_used": p_near,
                "energy_final_cycle_acc": q.get("mean_acc"),
                "energy_vs_source": (q.get("mean_acc") - source_acc
                                     if q.get("mean_acc") is not None
                                     and source_acc is not None else None),
            })
        # One mechanical sentence.
        ev = rec.get("entropy_p0_vs_source")
        gv = rec.get("energy_vs_source")
        rec["sentence"] = (
            f"{rec['cell']}: entropy p=0 finished at "
            f"{_f(rec.get('entropy_p0_final_cycle_acc'), 4)} "
            f"({'+' if (ev or 0) >= 0 else ''}{_f(ev, 3)} vs source) — "
            f"{rec.get('entropy_p0_shape')}; energy at p~p* "
            f"({_f(rec.get('energy_p_used'))}) reached "
            f"{_f(rec.get('energy_final_cycle_acc'), 4)} "
            f"({'+' if (gv or 0) >= 0 else ''}{_f(gv, 3)} vs source) — "
            + ("the tethered persistent drive adapted beyond source while "
               "the quenching drive did not."
               if (gv is not None and gv >= STALL_BAND
                   and (ev is None or ev < STALL_BAND))
               else "see numbers (pattern does not match "
                    "quench-stalls/persistent-adapts).")
        )
        rows.append(rec)
    return rows


def c1_md(rows) -> str:
    L = ["## C1 — entropy-vs-energy accuracy (per E7 cell)", "",
         "DATA CAVEAT: under the cycled protocol the persisted mean_accuracy "
         "== FINAL-CYCLE accuracy (per_corruption key collision); the true "
         "stream mean is not persisted. Shapes are classified from "
         "final-cycle accuracy vs source (+/-2pt) and NaN presence; per-cycle "
         "loss/drift trends shown as support.", "",
         "| cell | source | entropy p=0 final | Δ vs src | shape | entropy "
         "loss 1st→last | energy p used (~p*) | energy final | Δ vs src |",
         "|" + "---|" * 9]
    for r in rows:
        lt = r.get("entropy_loss_first_to_last") or (None, None)
        L.append("| " + " | ".join([
            r["cell"], _f(r.get("source_acc"), 4),
            _f(r.get("entropy_p0_final_cycle_acc"), 4),
            _f(r.get("entropy_p0_vs_source"), 3),
            str(r.get("entropy_p0_shape")),
            f"{_f(lt[0], 3)}→{_f(lt[1], 3)}",
            _f(r.get("energy_p_used")),
            _f(r.get("energy_final_cycle_acc"), 4),
            _f(r.get("energy_vs_source"), 3)]) + " |")
    L += [""]
    L += [f"- {r['sentence']}" for r in rows]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# C2 — ensure O1-O3, report.
# ----------------------------------------------------------------------------

def c2_offset(results_dir, severity, seed, cycles):
    off_path = results_dir / "analysis" / "e6_offset.json"
    if not off_path.exists():
        cm = results_dir / "analysis" / "e6_crossmech.json"
        if not cm.exists():
            return {"status": "MISSING",
                    "note": "e6_crossmech.json absent — E6 verdict not run."}
        from scripts.analyze_e6_offset import run as run_offset
        run_offset(results_dir, severity, seed, cycles)
    d = _load(off_path)
    if d is None:
        return {"status": "MISSING", "note": "e6_offset.json unreadable."}
    return {"status": "OK",
            "label": d["o2"]["label"], "hits": d["o2"]["hits"],
            "n": d["o2"]["n"],
            "offset_theory": d["o1"]["offset_theory"],
            "offset_theory_2sigma": d["o1"].get("offset_theory_2sigma"),
            "fitted_intercept": d.get("fitted_intercept_reference_only")}


# ----------------------------------------------------------------------------
# C3 — consolidated stage1_final.md.
# ----------------------------------------------------------------------------

def c3_consolidate(results_dir, c1_rows, c2) -> str:
    a = results_dir / "analysis"
    law = _load(a / "stage1v2_gbar_law.json") or {}
    diag = _load(a / "stage1v2_diagnosis.json") or {}
    diagf = _load(a / "stage1v2_diagnosis_final.json") or {}
    e6 = _load(a / "e6_crossmech.json") or {}
    e7 = _load(a / "e7_driveswap.json") or {}
    sfr = _load(a / "stage1v2_S_frozen.json") or {}
    esc = _load(a / "stage1v2_escape.json") or {}

    vh = law.get("verdict_hard") or {}
    v6 = (e6.get("verdict") or {})
    v7 = (e7.get("verdict") or {})

    miss = "(artifact missing)"
    L = ["# STAGE 1 + 1b — CONSOLIDATED VERDICT FILE (stage1_final.md)", "",
         "Protocol: single-corruption severity-5 CYCLED x15 streams; HARD "
         "criterion primary, soft separate; provenance guards active "
         "throughout. No identity language in this file — that decision is "
         "the lead's.", "",
         "## Pre-registered verdicts (VERBATIM)",
         f"- **E1 (restart, hard family): {vh.get('verdict', miss)}** — "
         f"{vh.get('why', '')}",
         f"- **E6 cross-mechanism: {v6.get('verdict', miss)}**"
         + (f" — void reason: {v6.get('void_reason')}"
            if v6.get("void_reason") else ""),
         f"- **E7 drive-swap: {v7.get('verdict', miss)}**", ""]

    # The law.
    d1 = diag.get("d1") or {}
    pooled_wls = (d1.get("pooled") or {}).get("wls") or {}
    ctr = d1.get("contrast_check") or {}
    ref = law.get("continual_reference_hard") or {}
    L += ["## The law (two-factor: p* = S * eta * ||g_bar||; cross-campaign)",
          f"- cycled campaign pooled bracket-weighted slope: "
          f"{_f(pooled_wls.get('slope'), 4)} +/- "
          f"{_f(pooled_wls.get('sigma_slope'), 3)} "
          f"(spread {_f(d1.get('spread'), 4)})",
          f"- continual (independent reference, recomputed): "
          f"{_f(ref.get('slope'), 4)} +/- {_f(ref.get('sigma_slope'), 3)}",
          f"- contrast attribution: deviation "
          f"{_f((ctr.get('deviation_rel_pooled') or 0) * 100, 3)}% vs pooled; "
          f"within 1.96 sigma: {ctr.get('within_1p96_sigma', 'n/a')} "
          f"(bracket-resolution check)", ""]

    # Mechanism (E6 + C2).
    af = v6.get("anchor_fit") or {}
    L += ["## Mechanism (E6 anchor vs Bernoulli)",
          f"- anchor fit: slope={_f(af.get('slope'), 4)} "
          f"intercept={_f(af.get('intercept'), 4)} R^2={_f(af.get('r2'), 4)}; "
          f"prediction hits {v6.get('prediction_hits', 'n/a')}/4",
          (f"- C2 offset status: **{c2.get('label', c2.get('status'))}** — "
           f"theory-derived offset {_f(c2.get('offset_theory'), 4)} "
           f"(2-sigma variant {_f(c2.get('offset_theory_2sigma'), 4)}) vs "
           f"fitted intercept {_f(c2.get('fitted_intercept'), 4)}; "
           f"hits {c2.get('hits', 'n/a')}/{c2.get('n', 'n/a')} with the "
           "offset applied (derivation used Bernoulli runs only)"), ""]

    # Drive (E7 + C1).
    ent_hard = [r for r in (e7.get("rows") or [])
                if r.get("p_star_hard") is not None]
    L += ["## Drive (E7 entropy vs energy)",
          f"- entropy hard boundaries resolved in "
          f"{len(ent_hard)}/{len(e7.get('rows') or [])} cells; strong hits "
          f"{v7.get('strong_hits', 'n/a')}/{v7.get('n_evaluable', 'n/a')}; "
          f"weak-form R^2 {_f((v7.get('entropy_fit') or {}).get('r2'), 4)}; "
          f"monotone: {v7.get('monotone_in_x', 'n/a')}",
          "- calmness (matched stable cells): see e7_driveswap.md table "
          "(drift variance + grad step-variance, energy vs entropy).", "",
          c1_md(c1_rows)]

    # Scope, margin, E2, escape.
    d3 = diag.get("d3") or {}
    e2 = law.get("e2") or {}
    L += ["## Scope & operating constants",
          f"- contrast scope: slope deviates (see law section); D4 per-cycle "
          f"||g_bar|| probe + D5/D5b drift analyses in "
          f"stage1v2_diagnosis.md / stage1v2_escape.md"
          + (f" (escape contrast/others ratio "
             f"{_f((esc.get('summary') or {}).get('contrast_over_others_ratio'), 4)})"
             if esc else ""),
          f"- controller margin constant: c = "
          f"{_f(d3.get('margin_constant_c'))} (D3; c=1.8 provisional per the "
          f"Stage-2 GO note; D3-EXT quantifies over-tethering cost)",
          f"- E2 online ||g_bar|| readability: {e2.get('verdict', 'n/a')} "
          f"({e2.get('n_runs_conv_le_50', 'n/a')}/"
          f"{e2.get('n_runs_usable', 'n/a')} runs converge <= 50 steps)", ""]

    # Frozen constants.
    L += ["## Frozen constants (with manifests)",
          f"- S_frozen = {_f(sfr.get('value'), 5)} +/- "
          f"{_f(sfr.get('sigma'), 3)} (calibration cells, bracket-weighted; "
          f"source manifest: {len(sfr.get('source_manifest') or [])} files "
          f"in stage1v2_S_frozen.json)",
          "- every analysis artifact embeds its consumed-file manifest "
          "(provenance rule STEP 2b).", "",
          "**STOP — Stage 1 + 1b closed out. Stage 2 GO and the identity "
          "call follow from C2's outcome.**"]
    return "\n".join(L) + "\n"


def run(results_dir: Path, severity, seed, cycles):
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    c1_rows = c1_compare(results_dir, severity, seed, cycles)
    c2 = c2_offset(results_dir, severity, seed, cycles)
    md = c3_consolidate(results_dir, c1_rows, c2)
    (analysis_dir / "stage1_final.md").write_text(md, encoding="utf-8")
    (analysis_dir / "stage1_closeout.json").write_text(
        json.dumps({"c1": c1_rows, "c2": c2}, indent=2, default=str),
        encoding="utf-8")
    print(md)
    print(f"[saved] {analysis_dir / 'stage1_final.md'}")
    print(f"[saved] {analysis_dir / 'stage1_closeout.json'}")
    return {"c1": c1_rows, "c2": c2}


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

    print("=== stage1 close-out self-test ===")
    GBARS = {"gaussian_noise": 16.0, "elastic_transform": 8.0}
    R_TRUE = 12.0
    SRC = 0.70

    def energy_run(c, eta, p):
        d = _synth_cycled(c, eta, p, GBARS[c], R_TRUE, source_acc=SRC)
        return d  # stable runs: acc ~ SRC+0.08 (adapting beyond source)

    def entropy_run(c, eta, p, stall=True):
        # The quench-stall story: entropy p=0 is STABLE (no NaN) but parks at
        # source accuracy. Generate with a stable p (p=1.0 >= any threshold);
        # the file PATH carries the real p — the payload restore_prob field is
        # cosmetic for C1.
        d = _synth_cycled(c, eta, 1.0, GBARS[c] * 0.6, R_TRUE, source_acc=SRC)
        rows = d["results"]["summary"]["heat"]["stream_diagnostics"]
        acc = SRC + (0.001 if stall else 0.08)       # quench: stalls at source
        d["results"]["summary"]["heat"]["mean_accuracy"] = acc
        d["results"]["summary"]["heat"]["last_accuracy"] = acc
        for r in rows:
            r["energy"] = max(0.05, 2.0 - 0.001 * r["step"])      # falling loss
        return d

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "analysis").mkdir()
        for c, gbar in GBARS.items():
            for eta in v2.V2_ETAS:
                v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, source=True,
                                      eta=eta).write_text(
                    json.dumps(_synthetic_source(c, source_acc=SRC)))
                pst = eta * gbar / R_TRUE
                for p in sorted({0.0, round(pst * 0.9, 6), round(pst * 1.1, 6)}
                                | set(pc.scaled_p_grid(eta))):
                    v2.run_output_path_v2(d, v2.V2_ARCH, c, 5, 42, p=p,
                                          eta=eta).write_text(
                        json.dumps(energy_run(c, eta, p)))
                pst_e = eta * gbar * 0.6 / R_TRUE
                for p in sorted({0.0, round(pst_e * 0.9, 6),
                                 round(pst_e * 1.1, 6)}
                                | set(pc.scaled_p_grid(eta))):
                    sb.run_output_path_drive(d, v2.V2_ARCH, c, 5, 42, DRIVE,
                                             p, eta, cycles=v2.CYCLES
                                             ).write_text(
                        json.dumps(entropy_run(c, eta, p)))
        # minimal upstream artifacts for C3
        (d / "analysis" / "stage1v2_gbar_law.json").write_text(json.dumps(
            {"verdict_hard": {"verdict": "AMBER", "why": "pre-registered"},
             "continual_reference_hard": {"slope": 1.152,
                                          "sigma_slope": 0.022},
             "e2": {"verdict": "GREEN", "n_runs_conv_le_50": 40,
                    "n_runs_usable": 42}}))
        (d / "analysis" / "stage1v2_diagnosis.json").write_text(json.dumps(
            {"d1": {"pooled": {"wls": {"slope": 1.149,
                                       "sigma_slope": 0.01}},
                    "spread": 0.43,
                    "contrast_check": {"deviation_rel_pooled": 0.43,
                                       "within_1p96_sigma": False}},
             "d3": {"margin_constant_c": 1.5}}))
        (d / "analysis" / "stage1v2_S_frozen.json").write_text(json.dumps(
            {"value": 1.1063, "sigma": 0.0095, "source_manifest": [1, 2, 3]}))
        x = {"gaussian_noise": {}, }
        e6_rows = []
        for corruption, eta in [("gaussian_noise", 2e-3),
                                ("gaussian_noise", 1e-3),
                                ("gaussian_noise", 5e-4),
                                ("elastic_transform", 1e-3)]:
            xx = eta * GBARS[corruption]
            e6_rows.append({"cell": f"{corruption}_lr{pc.format_p(eta)}",
                            "corruption": corruption, "eta": eta,
                            "x_eta_gbar": xx,
                            "eta_lambda_hat": 1.1063 * xx,
                            "eta_lambda_star_hard": 1.092 * xx - 0.0036,
                            "eta_lambda_star_soft": 1.092 * xx - 0.0037,
                            "bracket_lambda": [(1.092 * xx - 0.0036) / eta * 0.98,
                                               (1.092 * xx - 0.0036) / eta * 1.02],
                            "cell_hit": False})
        (d / "analysis" / "e6_crossmech.json").write_text(json.dumps(
            {"rows": e6_rows,
             "verdict": {"verdict": "DIFFERENT-CURVE",
                         "anchor_fit": {"slope": 1.092, "intercept": -0.0036,
                                        "r2": 0.992},
                         "prediction_hits": 1},
             "predictions": {"S_frozen": 1.1063}}))
        (d / "analysis" / "e6_crossmech.md").write_text("# E6\n")
        (d / "analysis" / "e7_driveswap.json").write_text(json.dumps(
            {"rows": [{"p_star_hard": 0.001} for _ in range(4)]
             + [{"p_star_hard": None} for _ in range(2)],
             "verdict": {"verdict": "UNCLASSIFIABLE", "strong_hits": 2,
                         "n_evaluable": 6, "monotone_in_x": False,
                         "entropy_fit": {"r2": 0.71}}}))

        res = run(d, 5, 42, v2.CYCLES)
        c1, c2 = res["c1"], res["c2"]
        check("C1 covers 6 cells", len(c1) == 6)
        stalls = [r for r in c1
                  if "STALLING" in str(r.get("entropy_p0_shape"))]
        check("C1 detects entropy stall at source", len(stalls) >= 4,
              str([r.get("entropy_p0_shape") for r in c1]))
        check("C1 energy adapts beyond source",
              all((r.get("energy_vs_source") or 0) > 0.02 for r in c1
                  if r.get("energy_vs_source") is not None))
        check("C1 sentences generated", all(r.get("sentence") for r in c1))
        check("C2 ran O1-O3 and labeled",
              c2.get("status") == "OK" and c2.get("label") in
              ("OFFSET-EXPLAINED", "OFFSET-UNEXPLAINED"),
              json.dumps(c2, default=str)[:200])
        md = (d / "analysis" / "stage1_final.md").read_text(encoding="utf-8")
        for token in ("AMBER", "DIFFERENT-CURVE", "UNCLASSIFIABLE",
                      "S_frozen = 1.1063", "C1 —", "identity"):
            check(f"stage1_final.md contains '{token}'", token in md)
        check("no identity DECISION language",
              "IDENTITY:" not in md and "same mechanism" not in md.lower())

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
