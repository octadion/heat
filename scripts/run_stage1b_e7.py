"""
Stage 1b / E7 — DRIVE-SWAP (entropy drive, identical Bernoulli tether) on the
CORRECTED cycled protocol. heat==TFF.

PROTOCOL (binding): single-corruption severity-5 CYCLED x15 (~2355 steps);
HARD criterion primary, soft recorded separately; single-pass streams
forbidden for law measurement.

REFERENCE (the only allowed): S_frozen from
<results>/analysis/stage1v2_S_frozen.json. Nothing from voided campaigns.

Cells: {gaussian_noise, elastic_transform} x etas {5e-4,1e-3,2e-3}, DESCENDING
eta within each corruption. Pre-register the strong-form
p_hat = S_frozen * eta * ||g_bar||_entropy (cycled stationary window, stable
entropy reference) in e7_predictions.json BEFORE any p-grid; then at-p_hat run
+ eta-scaled grid + bisection (hard). Report: (a) strong-form hit rate
(+/-25% or in-bracket), (b) weak-form own-slope linearity (R^2 >= 0.9), and
(c) the calmness comparison at matched stable cells vs the cycled-E1 energy
runs. Labels: DRIVE-AGNOSTIC-STRONG / -WEAK / DRIVE-SENSITIVE /
UNCLASSIFIABLE (strong needs hits >= ceil(0.75*n_evaluable) AND weak-form
linearity). VOID rules apply.

Provenance guards: Drive-only RESULTS_DIR + sentinel, fingerprints (with
drive=entropy), end-of-campaign count check.

Usage (Colab):
  python scripts/run_stage1b_e7.py --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-wrn <wrn ckpt> --c10c-root data/cifar10c --requarantine-predictions
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
from scripts import stage1b_common as sb
from scripts import stage1v2_common as v2
from scripts.run_stage1_e1 import Manifest
from scripts.analyze_stage1 import CORRUPTION_COLORS
from scripts.analyze_stage1_v2 import v2_all_rows
from scripts.run_stage1b_e6 import (
    load_s_frozen, ensure_source, EXPECTED_FILES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"
DRIVE = "entropy"
PRED_TOL = 0.25
WEAK_R2 = 0.90
E7_ETAS_DESC = [2e-3, 1e-3, 5e-4]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1b / E7 drive-swap "
                                            "(cycled x15, hard-primary)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=v2.V2_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=v2.V2_SEVERITY)
    p.add_argument("--cycles", type=int, default=v2.CYCLES)
    p.add_argument("--s-frozen-file", type=str, default="")
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID))
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--allow-ephemeral", action="store_true")
    p.add_argument("--fresh-reference", action="store_true")
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


def e7_cells():
    return [(c, e) for c in sb.E7_CORRUPTIONS for e in E7_ETAS_DESC]


def ensure_entropy_run(args, manifest, ckpt_hash, corruption, eta, p,
                       log_prefix=""):
    out_path = sb.run_output_path_drive(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, p, eta, cycles=args.cycles)
    EXPECTED_FILES.add(out_path.name)
    if out_path.exists():
        try:
            v2.inject_fingerprint(out_path, corruption=corruption,
                                  cycle_count=args.cycles, eta=eta, p=p,
                                  checkpoint_hash=ckpt_hash, seed=args.seed,
                                  batch_size=args.batch_size,
                                  extra={"drive": DRIVE})
            data = pc.load_json(out_path)
            manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                         p=p, drive=DRIVE, status="reused")
            return data
        except Exception:
            pass
    cmd = sb.build_run_command_drive(
        run_tier2=RUN_TIER2, arch=args.arch, checkpoint=args.ckpt_wrn,
        corruption=corruption, severity=args.severity, seed=args.seed,
        results_dir=Path(args.results_dir), c10c_root=args.c10c_root,
        heat_lr=eta, drive=DRIVE, p=p, batch_size=args.batch_size,
        num_workers=args.num_workers, cycles=args.cycles)
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        err = out_path.with_suffix(".run_error.txt")
        try:
            err.write_text(f"run_error entropy cyc{args.cycles} {corruption} "
                           f"lr{pc.format_p(eta)} p={pc.format_p(p)}\n{tail}",
                           encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta, p=p,
                     drive=DRIVE, status="run_error", elapsed_s=round(dt, 1))
        print(f"{log_prefix}[run_error] entropy p={pc.format_p(p)} "
              f"({dt:.1f}s) -> continuing.", flush=True)
        return None
    fp = v2.inject_fingerprint(out_path, corruption=corruption,
                               cycle_count=args.cycles, eta=eta, p=p,
                               checkpoint_hash=ckpt_hash, seed=args.seed,
                               batch_size=args.batch_size,
                               extra={"drive": DRIVE})
    manifest.log(file=out_path.name, corruption=corruption, eta=eta, p=p,
                 drive=DRIVE, status="ok", elapsed_s=round(dt, 1),
                 n_steps=fp.get("n_steps"))
    print(f"{log_prefix}[ok] entropy cyc{args.cycles} {corruption} "
          f"lr{pc.format_p(eta)} p={pc.format_p(p)} ({dt:.1f}s, "
          f"n_steps={fp.get('n_steps')})", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def select_ref_p(args, manifest, ckpt_hash, corruption, eta, source_acc,
                 log_prefix=""):
    last = (None, None)
    for cand in pc.scaled_pref_ladder(args.arch, eta):
        data = ensure_entropy_run(args, manifest, ckpt_hash, corruption, eta,
                                  cand, log_prefix=log_prefix)
        if data is None:
            continue
        g = v2.grad_norm_gbar_cycled(data)
        gbar = g[0] if g else None
        last = (cand, gbar)
        verd = pc.classify_run(data, source_acc, pref_drift=None,
                               use_drift_criterion=False)
        if not verd["collapsed"]:
            print(f"{log_prefix}  p_ref={pc.format_p(cand)} STABLE "
                  f"||g_bar||_entropy={_f(gbar)} (cycled)", flush=True)
            return cand, gbar
        print(f"{log_prefix}  p_ref={pc.format_p(cand)} collapsed "
              f"({verd['criterion']}); escalating.", flush=True)
    print(f"{log_prefix}  [warn] no stable entropy reference; using last.",
          flush=True)
    return last


def build_predictions(args, manifest, ckpt_hash, S, S_sigma, s_path,
                      policy: str):
    pred_path = Path(args.results_dir) / "analysis" / "e7_predictions.json"
    if policy == "use":
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"[preregistration] reusing {pred_path.name} "
              f"({preds.get('written_at_utc')}) — audit-confirmed.", flush=True)
        return preds, pred_path

    cells, violations = {}, {}
    for corruption, eta in e7_cells():
        key = cell_key(corruption, eta)
        print(f"\n=== E7 phase 1: {corruption} eta={pc.format_p(eta)} "
              f"(cyc{args.cycles} entropy reference) ===", flush=True)
        ladder = set(pc.scaled_pref_ladder(args.arch, eta))
        pre = [q for q in sb.discover_drive_p_values(
                   args.results_dir, args.arch, corruption, args.severity,
                   args.seed, DRIVE, eta, cycles=args.cycles)
               if q not in ladder]
        if pre:
            violations[key] = pre
            print(f"  [VIOLATION] non-ladder entropy runs predate "
                  f"pre-registration: {pre} — recorded.", flush=True)
        src = ensure_source(args, manifest, ckpt_hash, corruption, eta,
                            log_prefix="  ")
        source_acc = pc.source_mean_acc(src) if src is not None else None
        p_ref, gbar = select_ref_p(args, manifest, ckpt_hash, corruption, eta,
                                   source_acc, log_prefix="  ")
        p_hat = (S * eta * gbar) if gbar is not None else None
        if p_hat is not None:
            p_hat = min(max(p_hat, 0.0), 1.0)
        cells[key] = {
            "corruption": corruption, "eta": eta, "source_acc": source_acc,
            "p_ref_used": p_ref, "grad_norm_gbar_entropy": gbar,
            "eta_times_gbar_entropy": (eta * gbar) if gbar is not None else None,
            "p_hat_strong": p_hat,
        }
        print(f"  ||g_bar||_entropy={_f(gbar)} -> p_hat = {_f(S, 5)} * "
              f"{pc.format_p(eta)} * {_f(gbar)} = {_f(p_hat)}", flush=True)

    preds = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": f"single-corruption sev5 cycled x{args.cycles}; "
                    "hard primary",
        "S_frozen": S, "S_frozen_sigma": S_sigma,
        "S_frozen_source_file": str(s_path),
        "prediction_formula": "strong form: p_hat = S_frozen * eta * "
                              "||g_bar||_entropy (cycled stationary window; "
                              "same R — drive-independent)",
        "preexisting_grid_violations": violations,
        "fresh_reference": bool(args.fresh_reference),
        "checkpoint_hash": ckpt_hash,
        "cells": cells, "env": manifest.env,
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    manifest.log(event="preregistration_written", file=pred_path.name,
                 S_frozen=S, n_cells=len(cells), violations=bool(violations))
    print(f"\n[preregistration] wrote {pred_path} — p-grid may now begin.",
          flush=True)
    return preds, pred_path


def sweep_cell(args, manifest, ckpt_hash, preds, corruption, eta):
    key = cell_key(corruption, eta)
    p_hat = preds["cells"].get(key, {}).get("p_hat_strong")
    print(f"\n=== E7 phase 3: {corruption} eta={pc.format_p(eta)} "
          f"(hard entropy p-grid; p_hat={_f(p_hat)}) ===", flush=True)
    src = ensure_source(args, manifest, ckpt_hash, corruption, eta,
                        log_prefix="  ")
    source_acc = pc.source_mean_acc(src) if src is not None else None
    points: dict[float, dict] = {}

    def run_and_store(p, label=""):
        data = ensure_entropy_run(args, manifest, ckpt_hash, corruption, eta,
                                  p, log_prefix="  ")
        rec = sb.store_point(points, p, data, source_acc)
        if rec["valid"]:
            print(f"  {label}p={pc.format_p(p):>9s} -> "
                  f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable':13s} "
                  f"({rec['criterion']}; soft: {rec['soft_criterion']}) "
                  f"acc={_f(rec['mean_acc'], 4)}", flush=True)
        return rec

    if p_hat is not None:
        run_and_store(round(p_hat, 6), label="AT ")
    for p in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        if p in points:
            continue
        run_and_store(p)
    for p in sb.discover_drive_p_values(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, eta, cycles=args.cycles):
        if p in points and points[p].get("valid"):
            continue
        path = sb.run_output_path_drive(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, p, eta, cycles=args.cycles)
        try:
            data = pc.load_json(path)
        except Exception:
            continue
        sb.store_point(points, p, data, source_acc)

    if sb.boundary_from_points(points)["bracket_high"] is None:
        print("  all tested p hard-collapse; extending upward (cap 5).",
              flush=True)
        extra = 0
        for p in pc.scaled_extend_grid(eta):
            if extra >= 5:
                break
            if p in points and points[p].get("valid"):
                if not points[p]["collapsed"]:
                    break
                continue
            rec = run_and_store(p, label="extend ")
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
    print(f"  => p*_hard(entropy)~={_f(final['p_star'])} "
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}] "
          f"monotone={final['monotone']}", flush=True)


# ----------------------------------------------------------------------------
# Analysis / verdict.
# ----------------------------------------------------------------------------

def analyze_e7(args, preds):
    rows = []
    for corruption, eta in e7_cells():
        key = cell_key(corruption, eta)
        pred = preds["cells"].get(key, {})
        ps = sb.discover_drive_p_values(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, eta, cycles=args.cycles)
        row = sb.analyze_cell_generic(
            args.results_dir, args.arch, corruption, args.severity, eta,
            args.seed, ps,
            loader=lambda p, c=corruption, e=eta: (
                pc.load_json(sb.run_output_path_drive(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    DRIVE, p, e, cycles=args.cycles))
                if sb.run_output_path_drive(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    DRIVE, p, e, cycles=args.cycles).exists() else None))
        p_star = row["hard"]["p_star"]
        blo, bhi = row["hard"]["bracket_low"], row["hard"]["bracket_high"]
        p_hat = pred.get("p_hat_strong")
        within = (p_hat is not None and p_hat > 0 and p_star is not None
                  and abs(p_star - p_hat) / p_hat <= PRED_TOL)
        in_bracket = (p_hat is not None and blo is not None and bhi is not None
                      and blo <= p_hat <= bhi)
        rows.append({
            "cell": key, "corruption": corruption, "eta": eta,
            "gbar_entropy_frozen": pred.get("grad_norm_gbar_entropy"),
            "x_eta_gbar_entropy": pred.get("eta_times_gbar_entropy"),
            "p_hat_strong": p_hat, "p_star_hard": p_star,
            "bracket": [blo, bhi],
            "monotone_hard": row["hard"]["monotone"],
            "note_hard": row["hard"]["note"],
            "anomalous_zero": (p_star == 0.0
                               and row["hard"]["monotone"] is False),
            "within_25pct": within, "pred_in_bracket": in_bracket,
            "cell_hit": bool(within or in_bracket),
            "p_star_soft": row["soft"]["p_star"],
            "n_p_run": len(ps),
        })
    return rows


def matched_calmness(args, corruption, eta=1e-3):
    """Largest common HARD-stable p between the cycled-E1 energy cell and the
    cycled E7 entropy cell; cycle-aware calmness metrics for both."""
    src_acc = v2.source_acc_v2(args.results_dir, args.arch, corruption,
                               args.severity, args.seed, eta,
                               cycles=args.cycles)

    def stable_ps(ps, loader):
        out = {}
        for p in ps:
            data = loader(p)
            if data is None:
                continue
            hard, _ = sb.classify_both(data, src_acc)
            if not hard["collapsed"]:
                out[p] = data
        return out

    energy = stable_ps(
        v2.discover_p_values_v2(args.results_dir, args.arch, corruption,
                                args.severity, args.seed, eta,
                                cycles=args.cycles),
        lambda p: v2.load_run_v2(args.results_dir, args.arch, corruption,
                                 args.severity, args.seed, p=p, eta=eta,
                                 cycles=args.cycles))
    entropy = stable_ps(
        sb.discover_drive_p_values(args.results_dir, args.arch, corruption,
                                   args.severity, args.seed, DRIVE, eta,
                                   cycles=args.cycles),
        lambda p: (pc.load_json(sb.run_output_path_drive(
            args.results_dir, args.arch, corruption, args.severity, args.seed,
            DRIVE, p, eta, cycles=args.cycles))
            if sb.run_output_path_drive(
                args.results_dir, args.arch, corruption, args.severity,
                args.seed, DRIVE, p, eta, cycles=args.cycles).exists()
            else None))
    common = sorted(set(energy) & set(entropy))
    if not common:
        return {"corruption": corruption, "eta": eta, "matched_p": None,
                "note": "no common stable p between drives"}
    p = common[-1]
    return {"corruption": corruption, "eta": eta, "matched_p": p,
            "energy": sb.calmness_metrics(energy[p], cycles=args.cycles),
            "entropy": sb.calmness_metrics(entropy[p], cycles=args.cycles)}


def e7_verdict(rows, n_energy_reference_points):
    pts = [(r["x_eta_gbar_entropy"], r["p_star_hard"]) for r in rows
           if r["x_eta_gbar_entropy"] is not None
           and r["p_star_hard"] is not None]
    fit = pc.least_squares_line([x for x, _ in pts], [y for _, y in pts])
    seq = sorted(pts)
    monotone = all(b[1] >= a[1] - 1e-9 for a, b in zip(seq, seq[1:]))
    evaluable = [r for r in rows if r["p_hat_strong"] is not None]
    hits = sum(1 for r in evaluable if r["cell_hit"])
    n_ev = len(evaluable)
    strong_need = math.ceil(0.75 * n_ev) if n_ev else None
    unresolved = sum(1 for r in rows if r["p_star_hard"] is None)
    linear = fit is not None and fit["r2"] is not None and fit["r2"] >= WEAK_R2
    strong = (n_ev > 0 and hits >= strong_need and linear)
    anomalous = [r["cell"] for r in rows if r.get("anomalous_zero")]
    void, void_reason = sb.void_if_no_reference(
        n_energy_reference_points, len(pts), anomalous,
        "cycled-E1 energy hard")
    if void:
        label = sb.VOID
    elif strong:
        label = "DRIVE-AGNOSTIC-STRONG"
    elif linear and monotone:
        label = "DRIVE-AGNOSTIC-WEAK"
    elif (pts and not monotone) or unresolved > len(rows) / 2:
        label = "DRIVE-SENSITIVE"
    else:
        label = "UNCLASSIFIABLE"
    return {"verdict": label, "void_reason": void_reason,
            "n_energy_reference_points": n_energy_reference_points,
            "anomalous_cells": anomalous, "entropy_fit": fit,
            "n_points": len(pts), "monotone_in_x": monotone,
            "strong_hits": hits, "n_evaluable": n_ev,
            "strong_hits_needed": strong_need,
            "weak_form_linear_r2_ge_09": linear,
            "n_unresolved_cells": unresolved}


def e7_plot(S, rows, verdict, energy_rows_hard, out_png: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xmax = 0.0
    for r in energy_rows_hard:
        if r["corruption"] not in sb.E7_CORRUPTIONS:
            continue
        x, y = r.get("eta_times_gbar"), r.get("p_star")
        if x is None or y is None:
            continue
        xmax = max(xmax, x)
        ax.scatter([x], [y], s=45, alpha=0.7, zorder=2,
                   color=CORRUPTION_COLORS.get(r["corruption"], "0.4"))
    for r in rows:
        x, y = r["x_eta_gbar_entropy"], r["p_star_hard"]
        if x is None or y is None:
            continue
        xmax = max(xmax, x)
        ax.scatter([x], [y], marker="D", s=80, zorder=4, facecolors="none",
                   linewidths=2,
                   edgecolors=CORRUPTION_COLORS.get(r["corruption"], "k"))
        ax.annotate(f"η={r['eta']:g}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    if xmax > 0:
        xx = [0.0, xmax * 1.15]
        ax.plot(xx, [S * x for x in xx], color="0.3", linestyle=":",
                linewidth=1.6, label=f"energy reference S_frozen={S:.4g}")
        f = verdict["entropy_fit"]
        if f:
            ax.plot(xx, [f["slope"] * x + f["intercept"] for x in xx],
                    color="#2ca02c", linestyle="--", linewidth=1.5,
                    label=f"entropy fit slope={f['slope']:.3g} "
                          f"R²={_f(f['r2'], 3)}")
    ax.scatter([], [], color="0.4", s=45,
               label="cycled-E1 energy p*_hard (circles)")
    ax.scatter([], [], marker="D", facecolors="none", edgecolors="k", s=80,
               label="entropy p*_hard (diamonds)")
    ax.set_xlabel(r"$\eta \cdot \|\bar g\|$ (per-drive)")
    ax.set_ylabel(r"$p^*$ (hard)")
    ax.set_title(f"E7 drive-swap (cycled x15, hard) — {verdict['verdict']}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def e7_markdown(preds, rows, verdict, calm) -> str:
    L = [f"# E7 drive-swap verdict: **{verdict['verdict']}**", ""]
    if verdict["verdict"] == sb.VOID:
        L += [f"**VOID — {verdict['void_reason']}**", ""]
    L += [f"Protocol: {preds.get('protocol')}. Reference: S_frozen="
          f"{_f(preds.get('S_frozen'), 5)} +/- "
          f"{_f(preds.get('S_frozen_sigma'), 3)} from "
          f"{preds.get('S_frozen_source_file')}.",
          f"Pre-registration: {preds.get('written_at_utc')} — "
          f"{preds.get('prediction_formula')}. Strong label needs hits >= "
          f"ceil(0.75*n_evaluable) = {verdict['strong_hits_needed']} AND "
          f"weak-form R^2 >= {WEAK_R2}.", ""]
    viol = preds.get("preexisting_grid_violations") or {}
    if viol:
        L += ["**PRE-REGISTRATION VIOLATIONS:**"]
        L += [f"- {k}: p={v}" for k, v in viol.items()]
        L += [""]
    L += ["| cell | ||g_bar||_entropy | p_hat | p*_hard | bracket | monotone "
          "| within 25% | in bracket | hit | p*_soft |", "|" + "---|" * 10]
    for r in rows:
        L.append("| " + " | ".join([
            r["cell"], _f(r["gbar_entropy_frozen"], 4), _f(r["p_hat_strong"]),
            _f(r["p_star_hard"]),
            f"[{_f(r['bracket'][0])},{_f(r['bracket'][1])}]",
            ("ANOMALOUS-ZERO" if r.get("anomalous_zero")
             else str(r.get("monotone_hard"))),
            str(r["within_25pct"]), str(r["pred_in_bracket"]),
            "YES" if r["cell_hit"] else "no", _f(r["p_star_soft"])]) + " |")
    f = verdict["entropy_fit"]
    L += ["", "## Tiers",
          f"- (a) strong-form hit rate: {verdict['strong_hits']}/"
          f"{verdict['n_evaluable']} (need {verdict['strong_hits_needed']})",
          f"- (b) weak-form entropy fit (n={verdict['n_points']}): "
          + ("n/a" if f is None else
             f"slope={_f(f['slope'], 4)} intercept={_f(f['intercept'], 3)} "
             f"R^2={_f(f['r2'], 4)} "
             f"({'PASS' if verdict['weak_form_linear_r2_ge_09'] else 'FAIL'})"),
          f"- monotone in x: {verdict['monotone_in_x']}; unresolved: "
          f"{verdict['n_unresolved_cells']}",
          "", "## Calmness at matched stable cells (eta=1e-3, cycle-aware)",
          "| corruption | matched p | drive | drift var | grad_l2 step var | "
          "n steps |", "|" + "---|" * 6]
    for c in calm:
        if c.get("matched_p") is None:
            L.append(f"| {c['corruption']} | n/a ({c.get('note')}) |"
                     + " |" * 4)
            continue
        for drive in ("energy", "entropy"):
            m = c.get(drive)
            L.append("| " + " | ".join([
                c["corruption"], _f(c["matched_p"]), drive,
                _f(m["drift_var_stationary"], 4) if m else "n/a",
                _f(m["grad_l2_step_var_stationary"], 4) if m else "n/a",
                str(m["n_stationary_steps"]) if m else "n/a"]) + " |")
    L += ["", f"**Verdict: {verdict['verdict']}.**", "",
          "**STOP — Stage 1b complete (E6 + E7 on the corrected protocol); "
          "Stage 2 (controller, margin c=1.8 provisional) awaits "
          "instruction.**"]
    return "\n".join(L) + "\n"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    pred_path = Path(args.results_dir) / "analysis" / "e7_predictions.json"

    if not args.analyze_only:
        v2.require_drive_dir(args.results_dir,
                             allow_ephemeral=args.allow_ephemeral)
        if not args.ckpt_wrn or not Path(args.ckpt_wrn).exists():
            raise SystemExit(f"[fatal] WRN checkpoint not found: "
                             f"{args.ckpt_wrn!r} (or pass --analyze-only)")
        ckpt_hash = v2.file_sha256(args.ckpt_wrn)
        S, S_sigma, s_path = load_s_frozen(args)
        print(f"[e7] S_frozen={S} +/- {_f(S_sigma)} from {s_path} | "
              f"ckpt sha256[:16]={ckpt_hash}", flush=True)
        sb.require_e1_reference(Path(args.results_dir), e7_cells(),
                                args.severity, args.seed,
                                fresh_ok=args.fresh_reference,
                                cycles=args.cycles)
        policy = sb.resolve_predictions_policy(
            pred_path, args.trust_existing_predictions,
            args.requarantine_predictions)
        manifest = Manifest(Path(args.results_dir), args,
                            campaign=f"stage1b_e7_cyc{args.cycles}",
                            filename="stage1b_e7_manifest.jsonl")
        manifest.log(event="rules", predictions_policy=policy,
                     fresh_reference=args.fresh_reference,
                     checkpoint_hash=ckpt_hash, S_frozen=S)
        preds, pred_path = build_predictions(args, manifest, ckpt_hash, S,
                                             S_sigma, s_path, policy)
        for corruption, eta in e7_cells():
            sweep_cell(args, manifest, ckpt_hash, preds, corruption, eta)
        report = v2.drive_count_check(args.results_dir, EXPECTED_FILES)
        manifest.log(event="campaign_end", n_missing=len(report["missing"]))

    if not pred_path.exists():
        raise SystemExit(f"[fatal] predictions file missing: {pred_path}")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    S = float(preds.get("S_frozen"))

    rows = analyze_e7(args, preds)
    energy_rows_hard = v2_all_rows(Path(args.results_dir), args.severity,
                                   args.seed, v2.V2_ETAS, args.cycles,
                                   hard_only=True)
    n_energy_ref = sum(1 for r in energy_rows_hard
                       if r["corruption"] in sb.E7_CORRUPTIONS
                       and r.get("eta_times_gbar") is not None
                       and r.get("p_star") is not None)
    verdict = e7_verdict(rows, n_energy_ref)
    calm = [matched_calmness(args, c) for c in sb.E7_CORRUPTIONS]

    analysis_dir = Path(args.results_dir) / "analysis"
    e7_plot(S, rows, verdict, energy_rows_hard,
            analysis_dir / "e7_driveswap.png")
    md = e7_markdown(preds, rows, verdict, calm)
    (analysis_dir / "e7_driveswap.md").write_text(md, encoding="utf-8")
    (analysis_dir / "e7_driveswap.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict, "calmness": calm,
                    "predictions": preds}, indent=2, default=str),
        encoding="utf-8")
    print("\n" + md)
    for n in ("e7_predictions.json", "e7_driveswap.json", "e7_driveswap.png",
              "e7_driveswap.md"):
        print(f"[saved] {analysis_dir / n}")


if __name__ == "__main__":
    main()
