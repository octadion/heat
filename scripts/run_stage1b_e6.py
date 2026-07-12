"""
Stage 1b / E6 — CROSS-MECHANISM (L2 weight-anchor on the same curve), on the
CORRECTED cycled protocol. heat==TFF.

PROTOCOL (binding): all streams are single-corruption severity-5 CYCLED x15
(~2355 steps). Single-pass 157-step streams are FORBIDDEN for law measurement.
HARD criterion primary; soft recorded separately.

REFERENCE (the only allowed): S_frozen read from
<results>/analysis/stage1v2_S_frozen.json (calibration cells, bracket-weighted,
from the verified restart campaign). Nothing from any voided campaign seeds
anything (RULE 3 forces trust/requarantine on any existing predictions file;
legacy single-pass anchor files carry different tags and are invisible here).

Anchor cells: {gaussian_noise x etas 5e-4,1e-3,2e-3} + {elastic_transform x
1e-3}, DESCENDING eta within each corruption. Per cell, BEFORE any lambda
grid: stable-reference lambda run -> stationary ||g_bar|| (cycled window) ->
e6_predictions.json (timestamped) with (eta*lambda)_hat = S_frozen * eta *
||g_bar|| -> only then the at-lambda_hat run + lambda grid + bisection (hard).

VERDICT: SAME-CURVE requires (i) >= 3/4 cells within +/-25% of the frozen
prediction (or within the bisection bracket), (ii) anchor slope within +/-25%
of S_frozen, (iii) pooled R^2 >= 0.95 over anchor + cycled-E1 Bernoulli hard
points; else DIFFERENT-CURVE with both slopes reported honestly. VOID rules
apply (no reference / boundary anomaly => no curve verdict).

Provenance guards: Drive-only RESULTS_DIR + sentinel, checkpoint sha256 +
per-run protocol fingerprints (with anchor_lambda), end-of-campaign
file-count check.

Usage (Colab):
  python scripts/run_stage1b_e6.py --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-wrn <wrn ckpt> --c10c-root data/cifar10c --requarantine-predictions
"""

from __future__ import annotations

import argparse
import json
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
from scripts import stage1b_common as sb
from scripts import stage1v2_common as v2
from scripts.run_stage1_e1 import Manifest
from scripts.analyze_stage1 import CORRUPTION_COLORS
from scripts.analyze_stage1_v2 import v2_all_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"
PRED_TOL = 0.25

E6_CELLS = [("gaussian_noise", 2e-3), ("gaussian_noise", 1e-3),
            ("gaussian_noise", 5e-4), ("elastic_transform", 1e-3)]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1b / E6 cross-mechanism "
                                            "(cycled x15, hard-primary)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=v2.V2_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=v2.V2_SEVERITY)
    p.add_argument("--cycles", type=int, default=v2.CYCLES)
    p.add_argument("--s-frozen-file", type=str, default="",
                   help="Default: <results-dir>/analysis/stage1v2_S_frozen.json"
                        " — the ONLY allowed reference.")
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID),
                   help="BASE coarse grid in eta*lambda units (at eta=1e-3); "
                        "scaled per eta, mapped to lambda = value/eta.")
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--allow-ephemeral", action="store_true")
    p.add_argument("--fresh-reference", action="store_true",
                   help="RULE 2 override (verdict will be VOID without the "
                        "cycled E1 reference).")
    p.add_argument("--trust-existing-predictions", action="store_true")
    p.add_argument("--requarantine-predictions", action="store_true")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def cell_key(corruption, eta):
    return f"{corruption}_lr{pc.format_p(eta)}"


def load_s_frozen(args):
    path = Path(args.s_frozen_file) if args.s_frozen_file else (
        Path(args.results_dir) / "analysis" / "stage1v2_S_frozen.json")
    if not path.exists():
        raise SystemExit(f"[ABORT] S reference not found: {path} — run the "
                         "restart campaign's S-freeze first. Nothing else may "
                         "seed predictions.")
    d = json.loads(path.read_text(encoding="utf-8"))
    val = d.get("value", d.get("S_frozen"))
    if not val:
        raise SystemExit(f"[ABORT] {path} carries no usable S value.")
    return float(val), d.get("sigma"), path


EXPECTED_FILES: set[str] = set()


def ensure_source(args, manifest, ckpt_hash, corruption, eta, log_prefix=""):
    """Cycled-protocol source (reuses the restart campaign's file)."""
    out_path = v2.run_output_path_v2(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, source=True,
                                     eta=eta, cycles=args.cycles)
    EXPECTED_FILES.add(out_path.name)
    if out_path.exists():
        try:
            data = pc.load_json(out_path)
            manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                         source=True, status="reused")
            return data
        except Exception:
            pass
    cmd = v2.build_run_command_v2(
        run_tier2=RUN_TIER2, arch=args.arch, checkpoint=args.ckpt_wrn,
        corruption=corruption, severity=args.severity, seed=args.seed,
        results_dir=Path(args.results_dir), c10c_root=args.c10c_root,
        heat_lr=eta, batch_size=args.batch_size,
        num_workers=args.num_workers, source=True, cycles=args.cycles)
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    if not ok or not out_path.exists():
        manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                     source=True, status="run_error")
        return None
    v2.inject_fingerprint(out_path, corruption=corruption, cycle_count=1,
                          eta=eta, p=None, checkpoint_hash=ckpt_hash,
                          seed=args.seed, batch_size=args.batch_size)
    manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                 source=True, status="ok")
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def ensure_anchor_run(args, manifest, ckpt_hash, corruption, eta, lam,
                      log_prefix=""):
    out_path = sb.run_output_path_anchor(args.results_dir, args.arch,
                                         corruption, args.severity, args.seed,
                                         lam, eta, cycles=args.cycles)
    EXPECTED_FILES.add(out_path.name)
    if out_path.exists():
        try:
            v2.inject_fingerprint(out_path, corruption=corruption,
                                  cycle_count=args.cycles, eta=eta, p=None,
                                  checkpoint_hash=ckpt_hash, seed=args.seed,
                                  batch_size=args.batch_size,
                                  extra={"anchor_lambda": lam,
                                         "eta_lambda": lam * eta})
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
        num_workers=args.num_workers, cycles=args.cycles)
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        err = out_path.with_suffix(".run_error.txt")
        try:
            err.write_text(f"run_error anchor cyc{args.cycles} {corruption} "
                           f"lr{pc.format_p(eta)} lam={pc.format_p(lam)}\n"
                           f"{tail}", encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                     anchor_lambda=lam, status="run_error",
                     elapsed_s=round(dt, 1))
        print(f"{log_prefix}[run_error] anchor lam={pc.format_p(lam)} "
              f"({dt:.1f}s) -> continuing.", flush=True)
        return None
    fp = v2.inject_fingerprint(out_path, corruption=corruption,
                               cycle_count=args.cycles, eta=eta, p=None,
                               checkpoint_hash=ckpt_hash, seed=args.seed,
                               batch_size=args.batch_size,
                               extra={"anchor_lambda": lam,
                                      "eta_lambda": lam * eta})
    manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                 anchor_lambda=lam, status="ok", elapsed_s=round(dt, 1),
                 n_steps=fp.get("n_steps"))
    print(f"{log_prefix}[ok] anchor cyc{args.cycles} {corruption} "
          f"lr{pc.format_p(eta)} lam={pc.format_p(lam)} ({dt:.1f}s, "
          f"n_steps={fp.get('n_steps')})", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def lambda_ladder(args, eta):
    return [round(vv / eta, 6) for vv in pc.scaled_pref_ladder(args.arch, eta)]


def select_ref_lambda(args, manifest, ckpt_hash, corruption, eta, source_acc,
                      log_prefix=""):
    last = (None, None)
    for lam in lambda_ladder(args, eta):
        data = ensure_anchor_run(args, manifest, ckpt_hash, corruption, eta,
                                 lam, log_prefix=log_prefix)
        if data is None:
            continue
        g = v2.grad_norm_gbar_cycled(data)
        gbar = g[0] if g else None
        last = (lam, gbar)
        verd = pc.classify_run(data, source_acc, pref_drift=None,
                               use_drift_criterion=False)
        if not verd["collapsed"]:
            print(f"{log_prefix}  lambda_ref={pc.format_p(lam)} STABLE "
                  f"||g_bar||={_f(gbar)} (cycled)", flush=True)
            return lam, gbar
        print(f"{log_prefix}  lambda_ref={pc.format_p(lam)} collapsed "
              f"({verd['criterion']}); escalating.", flush=True)
    print(f"{log_prefix}  [warn] no stable reference lambda; using last.",
          flush=True)
    return last


def build_predictions(args, manifest, ckpt_hash, S, S_sigma, s_path,
                      policy: str):
    pred_path = Path(args.results_dir) / "analysis" / "e6_predictions.json"
    if policy == "use":
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"[preregistration] reusing {pred_path.name} "
              f"({preds.get('written_at_utc')}) — audit-confirmed.", flush=True)
        return preds, pred_path

    cells, violations = {}, {}
    for corruption, eta in E6_CELLS:
        key = cell_key(corruption, eta)
        print(f"\n=== E6 phase 1: {corruption} eta={pc.format_p(eta)} "
              f"(cyc{args.cycles} reference) ===", flush=True)
        ladder = set(lambda_ladder(args, eta))
        pre = [l for l in sb.discover_lambdas(args.results_dir, args.arch,
                                              corruption, args.severity,
                                              args.seed, eta,
                                              cycles=args.cycles)
               if l not in ladder]
        if pre:
            violations[key] = pre
            print(f"  [VIOLATION] non-ladder anchor runs predate "
                  f"pre-registration: {pre} — recorded.", flush=True)
        src = ensure_source(args, manifest, ckpt_hash, corruption, eta,
                            log_prefix="  ")
        source_acc = pc.source_mean_acc(src) if src is not None else None
        lam_ref, gbar = select_ref_lambda(args, manifest, ckpt_hash,
                                          corruption, eta, source_acc,
                                          log_prefix="  ")
        el_hat = (S * eta * gbar) if gbar is not None else None
        cells[key] = {
            "corruption": corruption, "eta": eta, "source_acc": source_acc,
            "lambda_ref_used": lam_ref, "grad_norm_gbar": gbar,
            "eta_times_gbar": (eta * gbar) if gbar is not None else None,
            "eta_lambda_hat": el_hat,
            "lambda_hat": (el_hat / eta) if el_hat is not None else None,
        }
        print(f"  ||g_bar||={_f(gbar)} -> (eta*lambda)_hat = {_f(S, 5)} * "
              f"{pc.format_p(eta)} * {_f(gbar)} = {_f(el_hat)}", flush=True)

    preds = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": f"single-corruption sev5 cycled x{args.cycles}; "
                    "hard primary",
        "S_frozen": S, "S_frozen_sigma": S_sigma,
        "S_frozen_source_file": str(s_path),
        "prediction_formula": "(eta*lambda)_hat = S_frozen * eta * ||g_bar|| "
                              "(cycled stationary window; stable reference "
                              "lambda; Remark 3: eta*lambda plays the role "
                              "of p)",
        "preexisting_grid_violations": violations,
        "fresh_reference": bool(args.fresh_reference),
        "checkpoint_hash": ckpt_hash,
        "cells": cells, "env": manifest.env,
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    manifest.log(event="preregistration_written", file=pred_path.name,
                 S_frozen=S, n_cells=len(cells), violations=bool(violations))
    print(f"\n[preregistration] wrote {pred_path} — grid runs may now begin.",
          flush=True)
    return preds, pred_path


def sweep_cell(args, manifest, ckpt_hash, preds, corruption, eta):
    key = cell_key(corruption, eta)
    lam_hat = preds["cells"].get(key, {}).get("lambda_hat")
    print(f"\n=== E6 phase 3: {corruption} eta={pc.format_p(eta)} "
          f"(hard lambda grid; lambda_hat={_f(lam_hat)}) ===", flush=True)
    src = ensure_source(args, manifest, ckpt_hash, corruption, eta,
                        log_prefix="  ")
    source_acc = pc.source_mean_acc(src) if src is not None else None
    points: dict[float, dict] = {}

    def run_and_store(lam, label=""):
        data = ensure_anchor_run(args, manifest, ckpt_hash, corruption, eta,
                                 lam, log_prefix="  ")
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
    for vv in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        lam = round(vv / eta, 6)
        if lam in points:
            continue
        run_and_store(lam)
    for lam in sb.discover_lambdas(args.results_dir, args.arch, corruption,
                                   args.severity, args.seed, eta,
                                   cycles=args.cycles):
        if lam in points and points[lam].get("valid"):
            continue
        path = sb.run_output_path_anchor(args.results_dir, args.arch,
                                         corruption, args.severity, args.seed,
                                         lam, eta, cycles=args.cycles)
        try:
            data = pc.load_json(path)
        except Exception:
            continue
        sb.store_point(points, lam, data, source_acc)

    if sb.boundary_from_points(points)["bracket_high"] is None:
        print("  all tested lambda hard-collapse; extending upward (cap 5).",
              flush=True)
        extra = 0
        for vv in pc.scaled_extend_grid(eta):
            if extra >= 5:
                break
            lam = round(vv / eta, 6)
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
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}] "
          f"monotone={final['monotone']}", flush=True)


# ----------------------------------------------------------------------------
# Analysis / verdict (read-only).
# ----------------------------------------------------------------------------

def analyze_e6(args, preds):
    rows = []
    for corruption, eta in E6_CELLS:
        key = cell_key(corruption, eta)
        pred = preds["cells"].get(key, {})
        lambdas = sb.discover_lambdas(args.results_dir, args.arch, corruption,
                                      args.severity, args.seed, eta,
                                      cycles=args.cycles)
        row = sb.analyze_cell_generic(
            args.results_dir, args.arch, corruption, args.severity, eta,
            args.seed, lambdas,
            loader=lambda lam, c=corruption, e=eta: (
                pc.load_json(sb.run_output_path_anchor(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    lam, e, cycles=args.cycles))
                if sb.run_output_path_anchor(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    lam, e, cycles=args.cycles).exists() else None))
        lam_star = row["hard"]["p_star"]
        blo, bhi = row["hard"]["bracket_low"], row["hard"]["bracket_high"]
        el_hat = pred.get("eta_lambda_hat")
        el_star = (lam_star * eta) if lam_star is not None else None
        within = (el_hat is not None and el_hat > 0 and el_star is not None
                  and abs(el_star - el_hat) / el_hat <= PRED_TOL)
        in_bracket = (el_hat is not None and blo is not None
                      and bhi is not None and blo * eta <= el_hat <= bhi * eta)
        rows.append({
            "cell": key, "corruption": corruption, "eta": eta,
            "gbar_frozen": pred.get("grad_norm_gbar"),
            "x_eta_gbar": pred.get("eta_times_gbar"),
            "eta_lambda_hat": el_hat,
            "lambda_star_hard": lam_star, "eta_lambda_star_hard": el_star,
            "bracket_lambda": [blo, bhi],
            "monotone_hard": row["hard"]["monotone"],
            "note_hard": row["hard"]["note"],
            "anomalous_zero": (lam_star == 0.0
                               and row["hard"]["monotone"] is False),
            "within_25pct": within, "pred_in_bracket": in_bracket,
            "cell_hit": bool(within or in_bracket),
            "lambda_star_soft": row["soft"]["p_star"],
            "eta_lambda_star_soft": (row["soft"]["p_star"] * eta
                                     if row["soft"]["p_star"] is not None
                                     else None),
            "n_lambdas_run": len(lambdas),
        })
    return rows


def e6_verdict(S, rows, bern_rows_hard):
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
                and abs(anchor_fit["slope"] - S) / S <= 0.25)
    pooled_ok = (pooled_fit is not None and pooled_fit["r2"] is not None
                 and pooled_fit["r2"] >= 0.95)
    hits_ok = hits >= 3
    anomalous = [r["cell"] for r in rows if r.get("anomalous_zero")]
    void, void_reason = sb.void_if_no_reference(
        len(bern_pts), len(anchor_pts), anomalous, "cycled-E1 Bernoulli hard")
    verdict = sb.VOID if void else (
        "SAME-CURVE" if (slope_ok and pooled_ok and hits_ok)
        else "DIFFERENT-CURVE")
    return {"verdict": verdict, "void_reason": void_reason,
            "anchor_fit": anchor_fit, "pooled_fit": pooled_fit,
            "n_anchor_points": len(anchor_pts),
            "n_bernoulli_points": len(bern_pts),
            "anomalous_cells": anomalous, "S_frozen": S,
            "slope_within_25pct_of_S_frozen": slope_ok,
            "pooled_r2_ge_095": pooled_ok,
            "prediction_hits": hits, "prediction_hits_ok_3_of_4": hits_ok}


def e6_plot(rows, bern_rows_hard, verdict, S, out_png: Path):
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
        ax.scatter([x], [y], marker="s", s=90, zorder=4, facecolors="none",
                   linewidths=2,
                   edgecolors=CORRUPTION_COLORS.get(r["corruption"], "k"))
        ax.annotate(f"η={r['eta']:g}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    if xmax > 0:
        xx = [0.0, xmax * 1.15]
        ax.plot(xx, [S * x for x in xx], color="0.3", linestyle=":",
                linewidth=1.6, label=f"Bernoulli reference S_frozen={S:.4g}")
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
    ax.scatter([], [], color="0.4", s=45,
               label="cycled-E1 Bernoulli p*_hard (circles)")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="k", s=90,
               label="anchor eta·λ*_hard (squares)")
    ax.set_xlabel(r"$\eta \cdot \|\bar g\|$")
    ax.set_ylabel(r"tether at boundary:  $p^*$  or  $\eta\lambda^*$")
    ax.set_title(f"E6 cross-mechanism (cycled x15, hard) — {verdict['verdict']}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def e6_markdown(preds, rows, verdict) -> str:
    L = [f"# E6 cross-mechanism verdict: **{verdict['verdict']}**", ""]
    if verdict["verdict"] == sb.VOID:
        L += [f"**VOID — {verdict['void_reason']}**", ""]
    L += [f"Protocol: {preds.get('protocol')}. Reference: S_frozen="
          f"{_f(verdict['S_frozen'], 5)} +/- {_f(preds.get('S_frozen_sigma'), 3)} "
          f"from {preds.get('S_frozen_source_file')}.",
          f"Pre-registration: {preds.get('written_at_utc')} — "
          f"{preds.get('prediction_formula')}", ""]
    viol = preds.get("preexisting_grid_violations") or {}
    if viol:
        L += ["**PRE-REGISTRATION VIOLATIONS:**"]
        L += [f"- {k}: lambda={v}" for k, v in viol.items()]
        L += [""]
    L += ["| cell | ||g_bar|| | (ηλ)_hat | ηλ*_hard | bracket(λ) | monotone | "
          "within 25% | in bracket | hit |", "|" + "---|" * 9]
    for r in rows:
        L.append("| " + " | ".join([
            r["cell"], _f(r["gbar_frozen"], 4), _f(r["eta_lambda_hat"]),
            _f(r["eta_lambda_star_hard"]),
            f"[{_f(r['bracket_lambda'][0])},{_f(r['bracket_lambda'][1])}]",
            ("ANOMALOUS-ZERO" if r.get("anomalous_zero")
             else str(r.get("monotone_hard"))),
            str(r["within_25pct"]), str(r["pred_in_bracket"]),
            "YES" if r["cell_hit"] else "no"]) + " |")
    af, pf = verdict["anchor_fit"], verdict["pooled_fit"]
    L += ["", "## Fits (hard-only, cycled)",
          f"- anchor fit (n={verdict['n_anchor_points']}): "
          + ("n/a" if af is None else f"slope={_f(af['slope'], 4)} "
             f"intercept={_f(af['intercept'], 3)} R^2={_f(af['r2'], 4)}"),
          f"- slope-vs-S check: "
          f"{'PASS' if verdict['slope_within_25pct_of_S_frozen'] else 'FAIL'}",
          f"- pooled (anchor + {verdict['n_bernoulli_points']} Bernoulli): "
          + ("n/a" if pf is None else f"slope={_f(pf['slope'], 4)} "
             f"R^2={_f(pf['r2'], 4)} "
             f"({'PASS' if verdict['pooled_r2_ge_095'] else 'FAIL'})"),
          f"- prediction hits: {verdict['prediction_hits']}/4 "
          f"({'PASS' if verdict['prediction_hits_ok_3_of_4'] else 'FAIL'})",
          "", "## Soft boundaries (separate; never in fits)",
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
    pred_path = Path(args.results_dir) / "analysis" / "e6_predictions.json"

    if not args.analyze_only:
        v2.require_drive_dir(args.results_dir,
                             allow_ephemeral=args.allow_ephemeral)
        if not args.ckpt_wrn or not Path(args.ckpt_wrn).exists():
            raise SystemExit(f"[fatal] WRN checkpoint not found: "
                             f"{args.ckpt_wrn!r} (or pass --analyze-only)")
        ckpt_hash = v2.file_sha256(args.ckpt_wrn)
        S, S_sigma, s_path = load_s_frozen(args)
        print(f"[e6] S_frozen={S} +/- {_f(S_sigma)} from {s_path} | "
              f"ckpt sha256[:16]={ckpt_hash}", flush=True)
        sb.require_e1_reference(Path(args.results_dir), E6_CELLS,
                                args.severity, args.seed,
                                fresh_ok=args.fresh_reference,
                                cycles=args.cycles)
        policy = sb.resolve_predictions_policy(
            pred_path, args.trust_existing_predictions,
            args.requarantine_predictions)
        manifest = Manifest(Path(args.results_dir), args,
                            campaign=f"stage1b_e6_cyc{args.cycles}",
                            filename="stage1b_e6_manifest.jsonl")
        manifest.log(event="rules", predictions_policy=policy,
                     fresh_reference=args.fresh_reference,
                     checkpoint_hash=ckpt_hash, S_frozen=S)
        preds, pred_path = build_predictions(args, manifest, ckpt_hash, S,
                                             S_sigma, s_path, policy)
        for corruption, eta in E6_CELLS:
            sweep_cell(args, manifest, ckpt_hash, preds, corruption, eta)
        report = v2.drive_count_check(args.results_dir, EXPECTED_FILES)
        manifest.log(event="campaign_end", n_missing=len(report["missing"]))

    if not pred_path.exists():
        raise SystemExit(f"[fatal] predictions file missing: {pred_path}")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    S = float(preds.get("S_frozen"))

    rows = analyze_e6(args, preds)
    bern_rows_hard = v2_all_rows(Path(args.results_dir), args.severity,
                                 args.seed, v2.V2_ETAS, args.cycles,
                                 hard_only=True)
    verdict = e6_verdict(S, rows, bern_rows_hard)

    analysis_dir = Path(args.results_dir) / "analysis"
    e6_plot(rows, bern_rows_hard, verdict, S,
            analysis_dir / "e6_crossmech.png")
    md = e6_markdown(preds, rows, verdict)
    (analysis_dir / "e6_crossmech.md").write_text(md, encoding="utf-8")
    (analysis_dir / "e6_crossmech.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict, "predictions": preds},
                   indent=2, default=str), encoding="utf-8")
    print("\n" + md)
    for n in ("e6_predictions.json", "e6_crossmech.json", "e6_crossmech.png",
              "e6_crossmech.md"):
        print(f"[saved] {analysis_dir / n}")


if __name__ == "__main__":
    main()
