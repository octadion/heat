"""
Stage 1b / E6 — CROSS-MECHANISM (L2 weight-anchor on the same curve), with the
pre-registered-prediction amendment. heat==TFF.

Anchor cells (wrn28_10, CIFAR-10-C sev5, seed 42, single-corruption streams,
batch 64): {gaussian_noise x etas 5e-4,1e-3,2e-3} + {elastic_transform x 1e-3}.
Mechanism: HeatAnchor (additive subclass; identical free-energy update and
diagnostics; after each SGD step theta <- theta - eta*lambda*(theta - theta_src);
mutually exclusive with restore_prob>0 — asserted in the class AND the factory).

CRITERION (Stage-1b update): PRIMARY = HARD collapse (nan_inf / chance);
soft:below_source recorded for every run, reported as a SEPARATE boundary,
never mixed into law fits.

ORDER (enforced; the point of the amendment):
  Phase 1  per cell: source run (REUSED from E1 — mechanism-independent) +
           stable reference lambda from the eta-scaled ladder mapped to lambda
           units (lambda = p_ladder/eta, so eta*lambda spans the same tether
           range as E1's p_ref ladder); measure stationary ||g_bar|| with the
           SAME reference policy as E1 (hard + below-source stability).
  Phase 2  (eta*lambda)_hat = S_frozen * eta * ||g_bar|| for ALL cells ->
           analysis/e6_predictions.json, timestamped, BEFORE the first grid
           run. Never overwritten; pre-existing non-ladder lambda runs are
           recorded as violations.
  Phase 3  at-lambda_hat run first, then the lambda grid (eta-scaled coarse
           p-grid mapped to lambda = p/eta, so eta*lambda targets
           ~[5e-4, 6e-2]), upward extension, bisection — HARD criterion.
  Phase 4  Verdict.

VERDICT (mechanical): SAME-CURVE requires ALL of
  (i)   anchor slope within +/-25% of the Bernoulli hard-only reference
        S_frozen (= 1.185 default, bracket-weighted, from A2);
  (ii)  pooled R^2 >= 0.95 over Bernoulli hard points + anchor points;
  (iii) >= 3 of 4 cells with measured eta*lambda*_hard within +/-25% of the
        frozen prediction (or the prediction inside the bisection bracket).
DIFFERENT-CURVE otherwise, with every sub-result reported honestly.
Soft boundaries and the soft-side fits are reported separately.

Outputs: analysis/e6_predictions.json, analysis/e6_crossmech.{json,png,md},
stage1b_e6_manifest.jsonl. Budget ~25-35 runs (each run ~157 steps).

Usage:
  python scripts/run_stage1b_e6.py --results-dir <Drive dir> \
      --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c
  python scripts/run_stage1b_e6.py --results-dir <dir> --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts import stage1b_common as sb
from scripts.run_stage1_e1 import Manifest, ensure_run as ensure_run_e1
from scripts.analyze_stage1 import e1_all_rows, CORRUPTION_COLORS

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"
PRED_TOL = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1b / E6 cross-mechanism")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=sc.E1_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--s-frozen", type=float, default=sb.S_FROZEN_DEFAULT,
                   help="Bracket-weighted hard-only Bernoulli reference slope "
                        "(A2). Recorded in the predictions file.")
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID),
                   help="BASE coarse grid in eta*lambda units (at eta=1e-3); "
                        "scaled by eta/1e-3, then mapped to lambda = value/eta.")
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--analyze-only", action="store_true")
    # Permanent methodological-integrity rules (see stage1b_common):
    p.add_argument("--fresh-reference", action="store_true",
                   help="RULE 2 override: proceed even though expected E1 "
                        "reference files are absent from --results-dir "
                        "(recorded; the verdict will be VOID without them).")
    p.add_argument("--trust-existing-predictions", action="store_true",
                   help="RULE 3: reuse an existing e6_predictions.json — ONLY "
                        "after the audit confirmed its reference runs were "
                        "genuine.")
    p.add_argument("--requarantine-predictions", action="store_true",
                   help="RULE 3: quarantine the existing e6_predictions.json "
                        "to analysis/quarantine/ and re-freeze new "
                        "predictions before any new grid run.")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def cell_key(corruption, eta):
    return f"{corruption}_lr{pc.format_p(eta)}"


def ensure_anchor_run(args, manifest, corruption, eta, lam, log_prefix=""):
    out_path = sb.run_output_path_anchor(args.results_dir, args.arch,
                                         corruption, args.severity, args.seed,
                                         lam, eta)
    if out_path.exists():
        try:
            data = pc.load_json(out_path)
            manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                         anchor_lambda=lam, status="reused")
            return data
        except Exception:
            pass
    cmd = sb.build_run_command_anchor(
        run_tier2=RUN_TIER2, arch=args.arch, checkpoint=args.ckpt_wrn,
        corruption=corruption, severity=args.severity, seed=args.seed,
        results_dir=Path(args.results_dir), c10c_root=args.c10c_root,
        heat_lr=eta, anchor_lambda=lam, batch_size=args.batch_size,
        num_workers=args.num_workers)
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        err = out_path.with_suffix(".run_error.txt")
        try:
            err.write_text(f"run_error anchor {corruption} lr{pc.format_p(eta)} "
                           f"lam={pc.format_p(lam)}\n{tail}", encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                     anchor_lambda=lam, status="run_error",
                     elapsed_s=round(dt, 1))
        print(f"{log_prefix}[run_error] anchor lam={pc.format_p(lam)} "
              f"({dt:.1f}s) -> {err.name}, continuing.", flush=True)
        return None
    manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                 anchor_lambda=lam, status="ok", elapsed_s=round(dt, 1))
    print(f"{log_prefix}[ok] anchor {corruption} lr{pc.format_p(eta)} "
          f"lam={pc.format_p(lam)} ({dt:.1f}s) -> {out_path.name}", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def lambda_ladder(args, eta):
    """Reference-lambda ladder: the eta-scaled p_ref ladder mapped to lambda
    units, so eta*lambda visits the exact tether strengths E1's p_ref used."""
    return [round(v / eta, 6) for v in pc.scaled_pref_ladder(args.arch, eta)]


def select_ref_lambda(args, manifest, corruption, eta, source_acc,
                      log_prefix=""):
    """Stable reference lambda (E1 reference policy: hard + below-source)."""
    last = (None, None)
    for lam in lambda_ladder(args, eta):
        data = ensure_anchor_run(args, manifest, corruption, eta, lam,
                                 log_prefix=log_prefix)
        if data is None:
            continue
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        last = (lam, gbar)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False)
        if not v["collapsed"]:
            print(f"{log_prefix}  lambda_ref={pc.format_p(lam)} STABLE "
                  f"||g_bar||={_f(gbar)}", flush=True)
            return lam, gbar
        print(f"{log_prefix}  lambda_ref={pc.format_p(lam)} collapsed "
              f"({v['criterion']}); escalating.", flush=True)
    print(f"{log_prefix}  [warn] no stable reference lambda; using last.",
          flush=True)
    return last


def build_predictions(args, manifest, policy: str):
    pred_path = Path(args.results_dir) / "analysis" / "e6_predictions.json"
    if policy == "use":
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"[preregistration] reusing {pred_path.name} "
              f"({preds.get('written_at_utc')}) — audit-confirmed.", flush=True)
        return preds, pred_path

    cells, violations = {}, {}
    for corruption, eta in sb.E6_CELLS:
        key = cell_key(corruption, eta)
        print(f"\n=== E6 phase 1: {corruption} eta={pc.format_p(eta)} ===",
              flush=True)
        ladder = set(lambda_ladder(args, eta))
        pre = [l for l in sb.discover_lambdas(args.results_dir, args.arch,
                                              corruption, args.severity,
                                              args.seed, eta)
               if l not in ladder]
        if pre:
            violations[key] = pre
            print(f"  [VIOLATION] non-ladder anchor runs predate "
                  f"pre-registration: {pre} — recorded.", flush=True)
        src = ensure_run_e1(args, manifest, corruption, eta=eta, source=True,
                            log_prefix="  ")
        source_acc = pc.source_mean_acc(src) if src is not None else None
        lam_ref, gbar = select_ref_lambda(args, manifest, corruption, eta,
                                          source_acc, log_prefix="  ")
        el_hat = (args.s_frozen * eta * gbar) if gbar is not None else None
        cells[key] = {
            "corruption": corruption, "eta": eta, "source_acc": source_acc,
            "lambda_ref_used": lam_ref, "grad_norm_gbar": gbar,
            "eta_times_gbar": (eta * gbar) if gbar is not None else None,
            "eta_lambda_hat": el_hat,
            "lambda_hat": (el_hat / eta) if el_hat is not None else None,
        }
        print(f"  ||g_bar||={_f(gbar)} -> (eta*lambda)_hat = "
              f"{_f(args.s_frozen)} * {pc.format_p(eta)} * {_f(gbar)} = "
              f"{_f(el_hat)}  (lambda_hat={_f(cells[key]['lambda_hat'])})",
              flush=True)

    preds = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "S_frozen": args.s_frozen,
        "S_frozen_source": "Stage-1b GO instruction: A2 bracket-weighted "
                           "hard-only pooled Bernoulli slope (wrn28_10 sev5)",
        "prediction_formula": "(eta*lambda)_hat = S_frozen * eta * ||g_bar|| "
                              "(Remark 3 mapping: eta*lambda plays the role "
                              "of p; ||g_bar|| from the stable reference "
                              "lambda, E1 reference policy)",
        "preexisting_grid_violations": violations,
        "fresh_reference": bool(getattr(args, "fresh_reference", False)),
        "missing_e1_reference_cells": getattr(args, "_missing_e1", []),
        "cells": cells, "env": manifest.env,
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    manifest.log(event="preregistration_written", file=pred_path.name,
                 S_frozen=args.s_frozen, n_cells=len(cells),
                 violations=bool(violations))
    print(f"\n[preregistration] wrote {pred_path} — grid runs may now begin.",
          flush=True)
    return preds, pred_path


def sweep_cell(args, manifest, preds, corruption, eta):
    key = cell_key(corruption, eta)
    lam_hat = preds["cells"].get(key, {}).get("lambda_hat")
    print(f"\n=== E6 phase 3: {corruption} eta={pc.format_p(eta)} "
          f"(hard-criterion lambda grid; lambda_hat={_f(lam_hat)}) ===",
          flush=True)
    src = ensure_run_e1(args, manifest, corruption, eta=eta, source=True,
                        log_prefix="  ")
    source_acc = pc.source_mean_acc(src) if src is not None else None

    points: dict[float, dict] = {}

    def run_and_store(lam, label=""):
        data = ensure_anchor_run(args, manifest, corruption, eta, lam,
                                 log_prefix="  ")
        rec = sb.store_point(points, lam, data, source_acc)
        if rec["valid"]:
            print(f"  {label}lambda={pc.format_p(lam):>10s} "
                  f"(eta*lambda={pc.format_p(round(lam * eta, 8))}) -> "
                  f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable':13s} "
                  f"({rec['criterion']}; soft: {rec['soft_criterion']}) "
                  f"acc={_f(rec['mean_acc'], 4)}", flush=True)
        return rec

    if lam_hat is not None:
        run_and_store(round(lam_hat, 6), label="AT ")

    for v in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        lam = round(v / eta, 6)
        if lam in points:
            continue
        run_and_store(lam)

    for lam in sb.discover_lambdas(args.results_dir, args.arch, corruption,
                                   args.severity, args.seed, eta):
        if lam in points and points[lam].get("valid"):
            continue
        path = sb.run_output_path_anchor(args.results_dir, args.arch,
                                         corruption, args.severity, args.seed,
                                         lam, eta)
        try:
            data = pc.load_json(path)
        except Exception:
            continue
        sb.store_point(points, lam, data, source_acc)

    if sb.boundary_from_points(points)["bracket_high"] is None:
        print("  all tested lambda hard-collapse; extending upward (cap 5).",
              flush=True)
        extra = 0
        for v in pc.scaled_extend_grid(eta):
            if extra >= 5:
                break
            lam = round(v / eta, 6)
            if lam in points and points[lam].get("valid"):
                if not points[lam]["collapsed"]:
                    break
                continue
            rec = run_and_store(lam, label="extend ")
            extra += 1
            if rec["valid"] and not rec["collapsed"]:
                break

    for i in range(max(0, args.bisect_steps)):
        sel = sb.boundary_from_points(points)
        lo, hi = sel["bracket_low"], sel["bracket_high"]
        if lo is None or hi is None or (hi - lo) <= 1e-6:
            break
        mid = round((lo + hi) / 2.0, 6)
        if mid in points:
            break
        run_and_store(mid, label=f"bisect[{i+1}] ")

    final = sb.boundary_from_points(points)
    print(f"  => lambda*_hard~={_f(final['p_star'])} "
          f"(eta*lambda*~={_f((final['p_star'] or 0) * eta) if final['p_star'] is not None else 'n/a'}) "
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}]",
          flush=True)


# ----------------------------------------------------------------------------
# Phase 4 — analysis / verdict (read-only; recomputes from disk).
# ----------------------------------------------------------------------------

def analyze_e6(args, preds):
    rows = []
    for corruption, eta in sb.E6_CELLS:
        key = cell_key(corruption, eta)
        pred = preds["cells"].get(key, {})
        lambdas = sb.discover_lambdas(args.results_dir, args.arch, corruption,
                                      args.severity, args.seed, eta)
        row = sb.analyze_cell_generic(
            args.results_dir, args.arch, corruption, args.severity, eta,
            args.seed, lambdas,
            loader=lambda lam, c=corruption, e=eta: (
                pc.load_json(sb.run_output_path_anchor(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    lam, e))
                if sb.run_output_path_anchor(args.results_dir, args.arch, c,
                                             args.severity, args.seed, lam,
                                             e).exists() else None))
        lam_star = row["hard"]["p_star"]
        blo, bhi = row["hard"]["bracket_low"], row["hard"]["bracket_high"]
        el_hat = pred.get("eta_lambda_hat")
        el_star = (lam_star * eta) if lam_star is not None else None
        within = (el_hat is not None and el_hat > 0 and el_star is not None
                  and abs(el_star - el_hat) / el_hat <= PRED_TOL)
        in_bracket = (el_hat is not None and blo is not None and bhi is not None
                      and blo * eta <= el_hat <= bhi * eta)
        lam_star_soft = row["soft"]["p_star"]
        # F1 fix: carry the boundary-resolution disambiguators. A p*=0 with a
        # non-monotone note is a PATHOLOGICAL zero (stable-below-collapse),
        # not a real boundary — it voids the verdict (RULE 1).
        monotone = row["hard"]["monotone"]
        note = row["hard"]["note"]
        anomalous = (lam_star == 0.0 and monotone is False)
        rows.append({
            "cell": key, "corruption": corruption, "eta": eta,
            "gbar_frozen": pred.get("grad_norm_gbar"),
            "x_eta_gbar": pred.get("eta_times_gbar"),
            "eta_lambda_hat": el_hat,
            "lambda_star_hard": lam_star, "eta_lambda_star_hard": el_star,
            "bracket_lambda": [blo, bhi],
            "monotone_hard": monotone, "note_hard": note,
            "anomalous_zero": anomalous,
            "within_25pct": within, "pred_in_bracket": in_bracket,
            "cell_hit": bool(within or in_bracket),
            "lambda_star_soft": lam_star_soft,
            "eta_lambda_star_soft": (lam_star_soft * eta)
            if lam_star_soft is not None else None,
            "n_lambdas_run": len(lambdas),
        })
    return rows


def e6_verdict(args, rows, bern_rows_hard):
    anchor_pts = [(r["x_eta_gbar"], r["eta_lambda_star_hard"]) for r in rows
                  if r["x_eta_gbar"] is not None
                  and r["eta_lambda_star_hard"] is not None]
    anchor_fit = pc.least_squares_line([x for x, _ in anchor_pts],
                                       [y for _, y in anchor_pts])
    bern_pts = [(r["eta_times_gbar"], r["p_star"]) for r in bern_rows_hard
                if r.get("eta_times_gbar") is not None
                and r.get("p_star") is not None]
    pooled_pts = anchor_pts + bern_pts
    pooled_fit = pc.least_squares_line([x for x, _ in pooled_pts],
                                       [y for _, y in pooled_pts])
    hits = sum(1 for r in rows if r["cell_hit"])
    slope_ok = (anchor_fit is not None
                and abs(anchor_fit["slope"] - args.s_frozen) / args.s_frozen
                <= 0.25)
    pooled_ok = (pooled_fit is not None and pooled_fit["r2"] is not None
                 and pooled_fit["r2"] >= 0.95)
    hits_ok = hits >= 3

    # RULE 1 (permanent): no reference points / no own points / boundary
    # anomaly => VOID, never a curve verdict.
    anomalous = [r["cell"] for r in rows if r.get("anomalous_zero")]
    void, void_reason = sb.void_if_no_reference(
        len(bern_pts), len(anchor_pts), anomalous, "Bernoulli hard (E1)")
    if void:
        verdict = sb.VOID
    else:
        verdict = "SAME-CURVE" if (slope_ok and pooled_ok and hits_ok) \
            else "DIFFERENT-CURVE"
    return {
        "verdict": verdict, "void_reason": void_reason,
        "anchor_fit": anchor_fit, "pooled_fit": pooled_fit,
        "n_anchor_points": len(anchor_pts), "n_bernoulli_points": len(bern_pts),
        "anomalous_cells": anomalous,
        "S_frozen": args.s_frozen,
        "slope_within_25pct_of_S_frozen": slope_ok,
        "pooled_r2_ge_095": pooled_ok,
        "prediction_hits": hits, "prediction_hits_ok_3_of_4": hits_ok,
    }


def e6_plot(rows, bern_rows_hard, verdict, s_frozen, out_png: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xmax = 0.0
    for r in bern_rows_hard:
        x, y = r.get("eta_times_gbar"), r.get("p_star")
        if x is None or y is None:
            continue
        xmax = max(xmax, x)
        ax.scatter([x], [y], color=CORRUPTION_COLORS.get(r["corruption"], "0.4"),
                   s=45, alpha=0.7, zorder=2)
    for r in rows:
        x, y = r["x_eta_gbar"], r["eta_lambda_star_hard"]
        if x is None or y is None:
            continue
        xmax = max(xmax, x)
        ax.scatter([x], [y], marker="s", s=90, zorder=4,
                   facecolors="none", linewidths=2,
                   edgecolors=CORRUPTION_COLORS.get(r["corruption"], "k"))
        ax.annotate(f"η={r['eta']:g}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    if xmax > 0:
        xx = [0.0, xmax * 1.15]
        ax.plot(xx, [s_frozen * x for x in xx], color="0.3", linestyle=":",
                linewidth=1.6, label=f"Bernoulli reference S_frozen={s_frozen}")
        af = verdict["anchor_fit"]
        if af:
            ax.plot(xx, [af["slope"] * x + af["intercept"] for x in xx],
                    color="#d62728", linestyle="--", linewidth=1.5,
                    label=f"anchor fit slope={af['slope']:.3g} "
                          f"R²={_f(af['r2'], 3)}")
        pf = verdict["pooled_fit"]
        if pf:
            ax.plot(xx, [pf["slope"] * x + pf["intercept"] for x in xx],
                    color="k", linewidth=1.6,
                    label=f"pooled slope={pf['slope']:.3g} R²={_f(pf['r2'], 3)}")
    ax.scatter([], [], color="0.4", s=45, label="Bernoulli p*_hard (E1, circles)")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="k", s=90,
               label="anchor eta·λ*_hard (E6, squares)")
    ax.set_xlabel(r"$\eta \cdot \|\bar g\|$")
    ax.set_ylabel(r"tether strength at boundary:  $p^*$  or  $\eta\lambda^*$")
    ax.set_title(f"E6 cross-mechanism (hard-only) — {verdict['verdict']}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def e6_markdown(args, preds, rows, verdict, a2_slopes) -> str:
    L = [f"# E6 cross-mechanism verdict: **{verdict['verdict']}**", ""]
    if verdict["verdict"] == sb.VOID:
        L += [f"**VOID — {verdict['void_reason']}** (permanent rule: no "
              "reference points / boundary anomaly => no curve verdict.)", ""]
    L += ["Criterion: SAME-CURVE requires (i) anchor slope within +/-25% of the "
         f"Bernoulli hard-only reference S_frozen={args.s_frozen} "
         "(bracket-weighted, A2), (ii) pooled R^2 >= 0.95 over Bernoulli hard "
         "points + anchor points, (iii) >= 3 of 4 cells within +/-25% of the "
         "pre-registered (eta*lambda)_hat (or within the bisection bracket). "
         "PRIMARY criterion = HARD collapse; soft boundaries reported "
         "separately below, never mixed into fits.", "",
         f"Pre-registration: {preds.get('written_at_utc')} — "
         f"{preds.get('prediction_formula')}", ""]
    viol = preds.get("preexisting_grid_violations") or {}
    if viol:
        L += ["**PRE-REGISTRATION VIOLATIONS:**"]
        L += [f"- {k}: lambda={v}" for k, v in viol.items()]
        L += [""]
    L += ["| cell | ||g_bar|| | (ηλ)_hat | ηλ*_hard | bracket(λ) | monotone | "
          "note | within 25% | in bracket | hit |", "|" + "---|" * 10]
    for r in rows:
        L.append("| " + " | ".join([
            r["cell"], _f(r["gbar_frozen"], 4), _f(r["eta_lambda_hat"]),
            _f(r["eta_lambda_star_hard"]),
            f"[{_f(r['bracket_lambda'][0])},{_f(r['bracket_lambda'][1])}]",
            ("ANOMALOUS-ZERO" if r.get("anomalous_zero")
             else str(r.get("monotone_hard"))),
            str(r.get("note_hard") or "-"),
            str(r["within_25pct"]), str(r["pred_in_bracket"]),
            "YES" if r["cell_hit"] else "no"]) + " |")
    af, pf = verdict["anchor_fit"], verdict["pooled_fit"]
    L += ["", "## Fits (hard-only)",
          f"- anchor fit (n={verdict['n_anchor_points']}): "
          + ("n/a" if af is None else
             f"slope={_f(af['slope'], 4)} intercept={_f(af['intercept'], 3)} "
             f"R^2={_f(af['r2'], 4)}"),
          f"- Bernoulli reference: S_frozen={args.s_frozen} "
          f"(slope check {'PASS' if verdict['slope_within_25pct_of_S_frozen'] else 'FAIL'})",
          f"- pooled (anchor + {verdict['n_bernoulli_points']} Bernoulli hard "
          f"points): "
          + ("n/a" if pf is None else
             f"slope={_f(pf['slope'], 4)} R^2={_f(pf['r2'], 4)} "
             f"({'PASS' if verdict['pooled_r2_ge_095'] else 'FAIL'})"),
          f"- prediction hits: {verdict['prediction_hits']}/4 "
          f"({'PASS' if verdict['prediction_hits_ok_3_of_4'] else 'FAIL'})"]
    if a2_slopes:
        L += ["", "## A2 per-corruption Bernoulli hard slopes (context)"]
        L += [f"- {c}: {_f(s, 4)}" for c, s in a2_slopes.items()]
    L += ["", "## Soft boundaries (separate; usefulness boundary, not the law)",
          "| cell | λ*_soft | ηλ*_soft |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cell']} | {_f(r['lambda_star_soft'])} | "
                 f"{_f(r['eta_lambda_star_soft'])} |")
    L += ["", f"**Verdict: {verdict['verdict']}.**"]
    return "\n".join(L) + "\n"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    args.corruptions = sorted({c for c, _ in sb.E6_CELLS})  # for ensure_run_e1
    pred_path = Path(args.results_dir) / "analysis" / "e6_predictions.json"

    if not args.analyze_only:
        if not args.ckpt_wrn or not Path(args.ckpt_wrn).exists():
            raise SystemExit(f"[fatal] WRN checkpoint not found: "
                             f"{args.ckpt_wrn!r} (or pass --analyze-only)")
        # RULE 2: abort loudly if the E1 reference is absent here.
        args._missing_e1 = sb.require_e1_reference(
            Path(args.results_dir), sb.E6_CELLS, args.severity, args.seed,
            fresh_ok=args.fresh_reference)
        # RULE 3: an existing pre-registration is INVALID by default.
        policy = sb.resolve_predictions_policy(
            pred_path, args.trust_existing_predictions,
            args.requarantine_predictions)
        manifest = Manifest(Path(args.results_dir), args,
                            campaign="stage1b_e6",
                            filename="stage1b_e6_manifest.jsonl")
        manifest.log(event="rules", predictions_policy=policy,
                     fresh_reference=args.fresh_reference,
                     missing_e1_cells=len(args._missing_e1))
        print(f"[e6] env={manifest.env}  S_frozen={args.s_frozen}  "
              f"predictions_policy={policy}", flush=True)
        preds, pred_path = build_predictions(args, manifest, policy)
        for corruption, eta in sb.E6_CELLS:
            sweep_cell(args, manifest, preds, corruption, eta)

    if not pred_path.exists():
        raise SystemExit(f"[fatal] predictions file missing: {pred_path}")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))

    rows = analyze_e6(args, preds)
    bern_rows_hard = [r for r in e1_all_rows(Path(args.results_dir),
                                             args.severity, args.seed,
                                             hard_only=True)]
    verdict = e6_verdict(args, rows, bern_rows_hard)

    a2_slopes = None
    fu = Path(args.results_dir) / "analysis" / "stage1_followup.json"
    if fu.exists():
        try:
            a2_slopes = json.loads(fu.read_text(encoding="utf-8"))[
                "a2"]["per_corruption_slopes"]
        except Exception:
            a2_slopes = None

    analysis_dir = Path(args.results_dir) / "analysis"
    e6_plot(rows, bern_rows_hard, verdict, args.s_frozen,
            analysis_dir / "e6_crossmech.png")
    md = e6_markdown(args, preds, rows, verdict, a2_slopes)
    (analysis_dir / "e6_crossmech.md").write_text(md, encoding="utf-8")
    (analysis_dir / "e6_crossmech.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict,
                    "predictions": preds}, indent=2, default=str),
        encoding="utf-8")
    print("\n" + md)
    for n in ("e6_predictions.json", "e6_crossmech.json", "e6_crossmech.png",
              "e6_crossmech.md"):
        print(f"[saved] {analysis_dir / n}")


if __name__ == "__main__":
    main()
