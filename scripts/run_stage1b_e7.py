"""
Stage 1b / E7 — DRIVE-SWAP (entropy drive under the identical tether), with
the pre-registered two-tier-hypothesis amendment. heat==TFF.

Cells (wrn28_10, CIFAR-10-C sev5, seed 42, single-corruption streams, batch
64): {gaussian_noise, elastic_transform} x etas {5e-4, 1e-3, 2e-3}, ENTROPY
drive only (the energy counterparts exist from E1). Mechanism:
HeatEntropyDrive (additive subclass; identical optimizer, params, Bernoulli
restore, diagnostics; loss = mean prediction entropy of the batch).

CRITERION (Stage-1b update): PRIMARY = HARD collapse; soft:below_source
recorded per run and reported as a SEPARATE boundary, never mixed into fits.

ORDER (enforced):
  Phase 1  per cell: source (REUSED from E1 — drive-independent) + stable
           reference p from the eta-scaled ladder UNDER THE ENTROPY DRIVE;
           measure stationary ||g_bar||_entropy (E1 reference policy).
  Phase 2  strong-form predictions p_hat = S_frozen * eta * ||g_bar||_entropy
           (same R — drive-independent) for ALL cells ->
           analysis/e7_predictions.json, timestamped, BEFORE the p-grid.
           Never overwritten; pre-existing non-ladder p runs recorded as
           violations.
  Phase 3  at-p_hat run first, then eta-scaled p-grid + upward extension +
           bisection — HARD criterion primary.
  Phase 4  Verdict + calmness comparison.

REPORTED (both, mechanically):
  (a) strong-form hit rate: measured p*_hard within +/-25% of the frozen
      prediction (or within the bisection bracket);
  (b) weak-form: entropy points linear on their own (R^2 >= 0.9), own slope.
Calmness: at one matched stable cell per corruption (eta=1e-3, largest common
stable p between the E1 energy cell and the E7 entropy cell), stationary
drift variance and step-to-step variance of grad_l2, energy vs entropy.

VERDICT labels (operationalization stated so it is reproducible):
  DRIVE-AGNOSTIC-STRONG: hit rate >= ceil(0.75 * n_evaluable) AND weak-form
      linear (R^2 >= 0.9). (75% matches the E6/forward 3-of-4 standard.)
  DRIVE-AGNOSTIC-WEAK:   not strong, but entropy points linear (R^2 >= 0.9)
      AND p*_hard monotone in eta*||g_bar||_entropy (own slope reported).
  DRIVE-SENSITIVE:       non-monotone entropy boundaries OR more than half of
      the cells unresolved ("unstable").
  UNCLASSIFIABLE:        anything else (raw curves reported, no forced
      conclusion).

Outputs: analysis/e7_predictions.json, analysis/e7_driveswap.{json,png,md},
stage1b_e7_manifest.jsonl. Budget ~18-26 runs (each run ~157 steps).

Usage:
  python scripts/run_stage1b_e7.py --results-dir <Drive dir> \
      --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c
  python scripts/run_stage1b_e7.py --results-dir <dir> --analyze-only
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
DRIVE = "entropy"
PRED_TOL = 0.25
WEAK_R2 = 0.90


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1b / E7 drive-swap")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=sc.E1_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--s-frozen", type=float, default=sb.S_FROZEN_DEFAULT)
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID))
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--analyze-only", action="store_true")
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
    return [(c, e) for c in sb.E7_CORRUPTIONS for e in sorted(sb.E7_ETAS)]


def ensure_entropy_run(args, manifest, corruption, eta, p, log_prefix=""):
    out_path = sb.run_output_path_drive(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, p, eta)
    if out_path.exists():
        try:
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
        num_workers=args.num_workers)
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        err = out_path.with_suffix(".run_error.txt")
        try:
            err.write_text(f"run_error entropy {corruption} lr{pc.format_p(eta)} "
                           f"p={pc.format_p(p)}\n{tail}", encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta, p=p,
                     drive=DRIVE, status="run_error", elapsed_s=round(dt, 1))
        print(f"{log_prefix}[run_error] entropy p={pc.format_p(p)} ({dt:.1f}s) "
              f"-> {err.name}, continuing.", flush=True)
        return None
    manifest.log(file=out_path.name, corruption=corruption, eta=eta, p=p,
                 drive=DRIVE, status="ok", elapsed_s=round(dt, 1))
    print(f"{log_prefix}[ok] entropy {corruption} lr{pc.format_p(eta)} "
          f"p={pc.format_p(p)} ({dt:.1f}s) -> {out_path.name}", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def select_ref_p(args, manifest, corruption, eta, source_acc, log_prefix=""):
    """Stable reference p under the ENTROPY drive (E1 reference policy)."""
    last = (None, None)
    for cand in pc.scaled_pref_ladder(args.arch, eta):
        data = ensure_entropy_run(args, manifest, corruption, eta, cand,
                                  log_prefix=log_prefix)
        if data is None:
            continue
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        last = (cand, gbar)
        v = pc.classify_run(data, source_acc, pref_drift=None,
                            use_drift_criterion=False)
        if not v["collapsed"]:
            print(f"{log_prefix}  p_ref={pc.format_p(cand)} STABLE "
                  f"||g_bar||_entropy={_f(gbar)}", flush=True)
            return cand, gbar
        print(f"{log_prefix}  p_ref={pc.format_p(cand)} collapsed "
              f"({v['criterion']}); escalating.", flush=True)
    print(f"{log_prefix}  [warn] no stable entropy reference; using last.",
          flush=True)
    return last


def build_predictions(args, manifest):
    pred_path = Path(args.results_dir) / "analysis" / "e7_predictions.json"
    if pred_path.exists():
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"[preregistration] {pred_path} exists "
              f"({preds.get('written_at_utc')}); never overwritten — resuming.",
              flush=True)
        return preds, pred_path

    cells, violations = {}, {}
    for corruption, eta in e7_cells():
        key = cell_key(corruption, eta)
        print(f"\n=== E7 phase 1: {corruption} eta={pc.format_p(eta)} "
              f"(entropy reference) ===", flush=True)
        ladder = set(pc.scaled_pref_ladder(args.arch, eta))
        pre = [q for q in sb.discover_drive_p_values(
                   args.results_dir, args.arch, corruption, args.severity,
                   args.seed, DRIVE, eta) if q not in ladder]
        if pre:
            violations[key] = pre
            print(f"  [VIOLATION] non-ladder entropy grid runs predate "
                  f"pre-registration: {pre} — recorded.", flush=True)
        src = ensure_run_e1(args, manifest, corruption, eta=eta, source=True,
                            log_prefix="  ")
        source_acc = pc.source_mean_acc(src) if src is not None else None
        p_ref, gbar = select_ref_p(args, manifest, corruption, eta, source_acc,
                                   log_prefix="  ")
        p_hat = (args.s_frozen * eta * gbar) if gbar is not None else None
        if p_hat is not None:
            p_hat = min(max(p_hat, 0.0), 1.0)
        cells[key] = {
            "corruption": corruption, "eta": eta, "source_acc": source_acc,
            "p_ref_used": p_ref, "grad_norm_gbar_entropy": gbar,
            "eta_times_gbar_entropy": (eta * gbar) if gbar is not None else None,
            "p_hat_strong": p_hat,
        }
        print(f"  ||g_bar||_entropy={_f(gbar)} -> p_hat = "
              f"{_f(args.s_frozen)} * {pc.format_p(eta)} * {_f(gbar)} = "
              f"{_f(p_hat)}", flush=True)

    preds = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "S_frozen": args.s_frozen,
        "S_frozen_source": "Stage-1b GO instruction: A2 bracket-weighted "
                           "hard-only pooled Bernoulli slope (wrn28_10 sev5)",
        "prediction_formula": "strong form: p_hat = S_frozen * eta * "
                              "||g_bar||_entropy (same R — drive-independent; "
                              "||g_bar||_entropy from the stable entropy "
                              "reference run, E1 reference policy)",
        "preexisting_grid_violations": violations,
        "cells": cells, "env": manifest.env,
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    manifest.log(event="preregistration_written", file=pred_path.name,
                 S_frozen=args.s_frozen, n_cells=len(cells),
                 violations=bool(violations))
    print(f"\n[preregistration] wrote {pred_path} — p-grid runs may now begin.",
          flush=True)
    return preds, pred_path


def sweep_cell(args, manifest, preds, corruption, eta):
    key = cell_key(corruption, eta)
    p_hat = preds["cells"].get(key, {}).get("p_hat_strong")
    print(f"\n=== E7 phase 3: {corruption} eta={pc.format_p(eta)} "
          f"(hard-criterion entropy p-grid; p_hat={_f(p_hat)}) ===", flush=True)
    src = ensure_run_e1(args, manifest, corruption, eta=eta, source=True,
                        log_prefix="  ")
    source_acc = pc.source_mean_acc(src) if src is not None else None

    points: dict[float, dict] = {}

    def run_and_store(p, label=""):
        data = ensure_entropy_run(args, manifest, corruption, eta, p,
                                  log_prefix="  ")
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
                                        DRIVE, eta):
        if p in points and points[p].get("valid"):
            continue
        path = sb.run_output_path_drive(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, p, eta)
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
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}]",
          flush=True)


# ----------------------------------------------------------------------------
# Phase 4 — analysis / verdict.
# ----------------------------------------------------------------------------

def analyze_e7(args, preds):
    rows = []
    for corruption, eta in e7_cells():
        key = cell_key(corruption, eta)
        pred = preds["cells"].get(key, {})
        ps = sb.discover_drive_p_values(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, eta)
        row = sb.analyze_cell_generic(
            args.results_dir, args.arch, corruption, args.severity, eta,
            args.seed, ps,
            loader=lambda p, c=corruption, e=eta: (
                pc.load_json(sb.run_output_path_drive(
                    args.results_dir, args.arch, c, args.severity, args.seed,
                    DRIVE, p, e))
                if sb.run_output_path_drive(args.results_dir, args.arch, c,
                                            args.severity, args.seed, DRIVE,
                                            p, e).exists() else None))
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
            "within_25pct": within, "pred_in_bracket": in_bracket,
            "cell_hit": bool(within or in_bracket),
            "p_star_soft": row["soft"]["p_star"],
            "n_p_run": len(ps),
        })
    return rows


def matched_calmness(args, corruption, eta=1e-3):
    """Largest common STABLE (hard) p between the E1 energy cell and the E7
    entropy cell at eta; calmness metrics for both runs at that p."""
    src_acc = sc.source_acc_sc(args.results_dir, args.arch, corruption,
                               args.severity, args.seed, eta)

    def stable_ps(ps, loader):
        out = {}
        for p in ps:
            data = loader(p)
            if data is None:
                continue
            hard, _soft = sb.classify_both(data, src_acc)
            if not hard["collapsed"]:
                out[p] = data
        return out

    energy = stable_ps(
        sc.discover_p_values_sc(args.results_dir, args.arch, corruption,
                                args.severity, args.seed, eta),
        lambda p: sc.load_run_sc(args.results_dir, args.arch, corruption,
                                 args.severity, args.seed, p=p, eta=eta))
    entropy = stable_ps(
        sb.discover_drive_p_values(args.results_dir, args.arch, corruption,
                                   args.severity, args.seed, DRIVE, eta),
        lambda p: (pc.load_json(sb.run_output_path_drive(
            args.results_dir, args.arch, corruption, args.severity, args.seed,
            DRIVE, p, eta))
            if sb.run_output_path_drive(args.results_dir, args.arch,
                                        corruption, args.severity, args.seed,
                                        DRIVE, p, eta).exists() else None))
    common = sorted(set(energy) & set(entropy))
    if not common:
        return {"corruption": corruption, "eta": eta, "matched_p": None,
                "note": "no common stable p between drives"}
    p = common[-1]
    return {"corruption": corruption, "eta": eta, "matched_p": p,
            "energy": sb.calmness_metrics(energy[p]),
            "entropy": sb.calmness_metrics(entropy[p])}


def e7_verdict(rows):
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
    if strong:
        label = "DRIVE-AGNOSTIC-STRONG"
    elif linear and monotone:
        label = "DRIVE-AGNOSTIC-WEAK"
    elif (pts and not monotone) or unresolved > len(rows) / 2:
        label = "DRIVE-SENSITIVE"
    else:
        label = "UNCLASSIFIABLE"
    return {"verdict": label, "entropy_fit": fit, "n_points": len(pts),
            "monotone_in_x": monotone, "strong_hits": hits,
            "n_evaluable": n_ev, "strong_hits_needed": strong_need,
            "weak_form_linear_r2_ge_09": linear,
            "n_unresolved_cells": unresolved}


def e7_plot(args, rows, verdict, energy_rows_hard, out_png: Path):
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
        ax.plot(xx, [args.s_frozen * x for x in xx], color="0.3",
                linestyle=":", linewidth=1.6,
                label=f"energy reference S_frozen={args.s_frozen}")
        f = verdict["entropy_fit"]
        if f:
            ax.plot(xx, [f["slope"] * x + f["intercept"] for x in xx],
                    color="#2ca02c", linestyle="--", linewidth=1.5,
                    label=f"entropy fit slope={f['slope']:.3g} "
                          f"R²={_f(f['r2'], 3)}")
    ax.scatter([], [], color="0.4", s=45, label="energy p*_hard (E1, circles)")
    ax.scatter([], [], marker="D", facecolors="none", edgecolors="k", s=80,
               label="entropy p*_hard (E7, diamonds)")
    ax.set_xlabel(r"$\eta \cdot \|\bar g\|$  (per-drive $\|\bar g\|$)")
    ax.set_ylabel(r"$p^*$ (hard)")
    ax.set_title(f"E7 drive-swap (hard-only) — {verdict['verdict']}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def e7_markdown(args, preds, rows, verdict, calm) -> str:
    L = [f"# E7 drive-swap verdict: **{verdict['verdict']}**", "",
         "Two-tier hypothesis (pre-registered). Strong form: p_hat = S_frozen "
         "* eta * ||g_bar||_entropy (same R — drive-independent); hit = "
         "measured p*_hard within +/-25% (or within the bisection bracket). "
         f"Strong label needs hits >= ceil(0.75*n_evaluable) = "
         f"{verdict['strong_hits_needed']} AND weak-form linearity. Weak form: "
         f"entropy points linear on their own (R^2 >= {WEAK_R2}), own slope. "
         "PRIMARY criterion = HARD collapse; soft boundaries separate.", "",
         f"Pre-registration: {preds.get('written_at_utc')} — "
         f"{preds.get('prediction_formula')}", ""]
    viol = preds.get("preexisting_grid_violations") or {}
    if viol:
        L += ["**PRE-REGISTRATION VIOLATIONS:**"]
        L += [f"- {k}: p={v}" for k, v in viol.items()]
        L += [""]
    L += ["| cell | ||g_bar||_entropy | p_hat | p*_hard | bracket | within 25% "
          "| in bracket | hit | p*_soft |", "|" + "---|" * 9]
    for r in rows:
        L.append("| " + " | ".join([
            r["cell"], _f(r["gbar_entropy_frozen"], 4), _f(r["p_hat_strong"]),
            _f(r["p_star_hard"]),
            f"[{_f(r['bracket'][0])},{_f(r['bracket'][1])}]",
            str(r["within_25pct"]), str(r["pred_in_bracket"]),
            "YES" if r["cell_hit"] else "no", _f(r["p_star_soft"])]) + " |")
    f = verdict["entropy_fit"]
    L += ["", "## Fits / tiers",
          f"- (a) strong-form hit rate: {verdict['strong_hits']}/"
          f"{verdict['n_evaluable']} (need {verdict['strong_hits_needed']})",
          f"- (b) weak-form entropy fit (n={verdict['n_points']}): "
          + ("n/a" if f is None else
             f"slope={_f(f['slope'], 4)} intercept={_f(f['intercept'], 3)} "
             f"R^2={_f(f['r2'], 4)} "
             f"({'PASS' if verdict['weak_form_linear_r2_ge_09'] else 'FAIL'})"),
          f"- monotone in x: {verdict['monotone_in_x']}; unresolved cells: "
          f"{verdict['n_unresolved_cells']}",
          f"- energy reference: S_frozen={args.s_frozen} (A2 bracket-weighted "
          f"hard-only)"]
    L += ["", "## Calmness at matched stable cells (eta=1e-3)",
          "| corruption | matched p | drive | drift var (stationary) | "
          "grad_l2 step-to-step var | n steps |", "|" + "---|" * 6]
    for c in calm:
        if c.get("matched_p") is None:
            L.append(f"| {c['corruption']} | n/a ({c.get('note')}) |" + " |" * 4)
            continue
        for drive in ("energy", "entropy"):
            m = c.get(drive)
            L.append("| " + " | ".join([
                c["corruption"], _f(c["matched_p"]), drive,
                _f(m["drift_var_stationary"], 4) if m else "n/a",
                _f(m["grad_l2_step_var_stationary"], 4) if m else "n/a",
                str(m["n_stationary_steps"]) if m else "n/a"]) + " |")
    L += ["", f"**Verdict: {verdict['verdict']}.**",
          "", "**STOP — Stage 1b complete (E6 + E7); Stage 2 awaits "
          "instruction.**"]
    return "\n".join(L) + "\n"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    args.corruptions = list(sb.E7_CORRUPTIONS)  # for ensure_run_e1
    pred_path = Path(args.results_dir) / "analysis" / "e7_predictions.json"

    if not args.analyze_only:
        if not args.ckpt_wrn or not Path(args.ckpt_wrn).exists():
            raise SystemExit(f"[fatal] WRN checkpoint not found: "
                             f"{args.ckpt_wrn!r} (or pass --analyze-only)")
        manifest = Manifest(Path(args.results_dir), args,
                            campaign="stage1b_e7",
                            filename="stage1b_e7_manifest.jsonl")
        print(f"[e7] env={manifest.env}  S_frozen={args.s_frozen}", flush=True)
        preds, pred_path = build_predictions(args, manifest)
        for corruption, eta in e7_cells():
            sweep_cell(args, manifest, preds, corruption, eta)

    if not pred_path.exists():
        raise SystemExit(f"[fatal] predictions file missing: {pred_path}")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))

    rows = analyze_e7(args, preds)
    verdict = e7_verdict(rows)
    energy_rows_hard = e1_all_rows(Path(args.results_dir), args.severity,
                                   args.seed, hard_only=True)
    calm = [matched_calmness(args, c) for c in sb.E7_CORRUPTIONS]

    analysis_dir = Path(args.results_dir) / "analysis"
    e7_plot(args, rows, verdict, energy_rows_hard,
            analysis_dir / "e7_driveswap.png")
    md = e7_markdown(args, preds, rows, verdict, calm)
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
