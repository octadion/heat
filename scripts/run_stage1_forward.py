"""
Stage-1 follow-up, PART B — pre-registered FORWARD confirmation (small GPU).

Cells: {fog, shot_noise} x etas {5e-4, 2e-3}, wrn28_10, CIFAR-10-C sev5,
seed 42, single-corruption streams, batch 64 — the exact E1 pipeline.

THE ORDER IS THE POINT (enforced structurally):
  Phase 1  For ALL four cells: source run + p_ref run(s) only (the eta-scaled
           ladder; same reference policy as E1, so ||g_bar|| is measured
           identically). NO grid p is run in this phase.
  Phase 2  Predictions p_hat_hard = S_frozen * eta * ||g_bar|| are computed for
           all four cells and WRITTEN (timestamped, with the frozen-slope file
           + value recorded) to <results-dir>/analysis/stage1b_predictions.json
           BEFORE any p-grid run executes. If the predictions file already
           exists it is NEVER overwritten (it is the pre-registration; resume
           re-uses it). If grid runs beyond the p_ref ladder already exist for
           any cell when the file is first written, that is a pre-registration
           violation: it is recorded inside the predictions file and printed —
           never silently ignored.
  Phase 3  Only then: a run AT p_hat itself, the eta-scaled coarse p-grid,
           upward extension, and bisection — classified with the HARD collapse
           criterion as primary (pstar_common.classify_run(hard_only=True),
           unchanged math). The soft classification of every run is recorded
           alongside but never used for p*.
  Phase 4  Verdict: measured p*_hard vs the pre-registered prediction.

VERDICT (mechanical, from the follow-up spec):
  cell OK  := (measured p*_hard within +/-25% of prediction, OR the prediction
               lies within the final bisection bracket) AND no hard collapse
               in the run AT p_hat.
  CONFIRMED >= 3 of 4 cells OK;  PARTIAL = 2 of 4;  FAILED <= 1 of 4.
Also reports E4-style zero-shot accuracy at p_hat vs the cell's grid oracle.

Outputs: analysis/stage1b_predictions.json (pre-registration),
         analysis/stage1_forward_verdict.md, stage1b_manifest.jsonl.

Prerequisite: analysis/stage1_hardslope_frozen.json written by
scripts/analyze_stage1_followup.py (Part A / A2).

Usage:
  python scripts/run_stage1_forward.py \
      --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c
  # read-only verdict recompute (no new runs):
  python scripts/run_stage1_forward.py --results-dir <dir> --analyze-only
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

from scripts import pstar_common as pc
from scripts import stage1_common as sc
from scripts.run_stage1_e1 import Manifest, ensure_run, select_pref

FORWARD_CORRUPTIONS = ["fog", "shot_noise"]
FORWARD_ETAS = [5e-4, 2e-3]
PRED_TOL = 0.25  # +/-25% of the pre-registered prediction


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 Part B forward confirmation")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=sc.E1_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=list(FORWARD_CORRUPTIONS))
    p.add_argument("--etas", type=float, nargs="+", default=list(FORWARD_ETAS))
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID))
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--frozen-slope-file", type=str, default="",
                   help="Default: <results-dir>/analysis/stage1_hardslope_frozen.json")
    p.add_argument("--analyze-only", action="store_true",
                   help="Recompute the verdict from disk; run nothing.")
    return p.parse_args()


def _f(v, nd=5):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def cell_key(corruption, eta):
    return f"{corruption}_lr{pc.format_p(eta)}"


# ----------------------------------------------------------------------------
# Phase 1+2 — reference measurement, then pre-registered predictions.
# ----------------------------------------------------------------------------

def load_frozen_slope(args):
    path = Path(args.frozen_slope_file) if args.frozen_slope_file else (
        Path(args.results_dir) / "analysis" / "stage1_hardslope_frozen.json")
    if not path.exists():
        raise SystemExit(
            f"[fatal] frozen hard-only slope not found: {path}\n"
            f"Run Part A first: python scripts/analyze_stage1_followup.py "
            f"--results-dir <E1 dir>")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("S_frozen"):
        raise SystemExit(f"[fatal] {path} has no S_frozen value.")
    return payload, path


def preexisting_grid_violations(args, corruption, eta):
    """p-values already on disk for this cell that are NOT p_ref-ladder rungs.
    Any such run existing before the predictions file is first written is a
    pre-registration violation (recorded, never hidden)."""
    ladder = set(pc.scaled_pref_ladder(args.arch, eta))
    on_disk = sc.discover_p_values_sc(args.results_dir, args.arch, corruption,
                                      args.severity, args.seed, eta)
    return sorted(p for p in on_disk if p not in ladder)


def build_predictions(args, manifest, frozen, frozen_path):
    """Phase 1 (source + p_ref for every cell), then phase 2 (write the
    pre-registration file). Returns the predictions dict."""
    pred_path = Path(args.results_dir) / "analysis" / "stage1b_predictions.json"
    if pred_path.exists():
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        print(f"[preregistration] {pred_path} already exists "
              f"(written {preds.get('written_at_utc')}); it is NEVER "
              f"overwritten — resuming against it.", flush=True)
        return preds, pred_path

    S = float(frozen["S_frozen"])
    cells = {}
    violations = {}
    for corruption in args.corruptions:
        for eta in sorted(set(args.etas)):
            key = cell_key(corruption, eta)
            print(f"\n=== phase 1: {corruption} eta={pc.format_p(eta)} "
                  f"(reference measurement only) ===", flush=True)
            viol = preexisting_grid_violations(args, corruption, eta)
            if viol:
                violations[key] = viol
                print(f"  [VIOLATION] non-ladder grid runs already on disk "
                      f"BEFORE pre-registration: {viol} — recorded.", flush=True)
            src = ensure_run(args, manifest, corruption, eta=eta, source=True,
                             log_prefix="  ")
            source_acc = pc.source_mean_acc(src) if src is not None else None
            p_ref, pref_drift, gbar = select_pref(args, manifest, corruption,
                                                  eta, source_acc,
                                                  log_prefix="  ")
            p_hat = (S * eta * gbar) if gbar is not None else None
            if p_hat is not None:
                p_hat = min(max(p_hat, 0.0), 1.0)
            cells[key] = {
                "corruption": corruption, "eta": eta,
                "source_acc": source_acc, "p_ref_used": p_ref,
                "grad_norm_gbar": gbar,
                "eta_times_gbar": (eta * gbar) if gbar is not None else None,
                "p_hat_hard": p_hat,
            }
            print(f"  ||g_bar||={_f(gbar)}  ->  p_hat_hard = "
                  f"{_f(S, 5)} * {pc.format_p(eta)} * {_f(gbar)} = "
                  f"{_f(p_hat)}", flush=True)

    preds = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "S_frozen": S,
        "S_frozen_source_file": str(frozen_path),
        "S_frozen_definition": frozen.get("S_frozen_definition"),
        "prediction_formula": "p_hat_hard = S_frozen * eta * ||g_bar|| "
                              "(through origin; ||g_bar|| from this cell's "
                              "p_ref run, E1 reference policy)",
        "preexisting_grid_violations": violations,
        "cells": cells,
        "env": manifest.env,
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    print(f"\n[preregistration] wrote {pred_path} "
          f"({preds['written_at_utc']}) — grid runs may now begin.", flush=True)
    manifest.log(event="preregistration_written", file=pred_path.name,
                 S_frozen=S, n_cells=len(cells),
                 violations=bool(violations))
    return preds, pred_path


# ----------------------------------------------------------------------------
# Phase 3 — hard-criterion p* search (soft recorded alongside).
# ----------------------------------------------------------------------------

def _store_point_hard(points, p, data, source_acc):
    """Classify with HARD-only as primary; record the soft classification of
    the same run alongside (never used for p*)."""
    if data is None:
        points[p] = {"p": p, "collapsed": False, "valid": False}
        return points[p]
    v_hard = pc.classify_run(data, source_acc, pref_drift=None,
                             use_drift_criterion=False, hard_only=True)
    v_soft = pc.classify_run(data, source_acc, pref_drift=None,
                             use_drift_criterion=False, hard_only=False)
    points[p] = {"p": p, "collapsed": v_hard["collapsed"], "valid": True,
                 "criterion": v_hard["criterion"], "mean_acc": v_hard["mean_acc"],
                 "gbar": v_hard["gbar"],
                 "soft_collapsed": v_soft["collapsed"],
                 "soft_criterion": v_soft["criterion"]}
    return points[p]


def sweep_cell_hard(args, manifest, corruption, eta, p_hat, log_prefix=""):
    """Grid + bisection with hard-only classification; the run AT p_hat is
    executed first so the confirmation point always exists."""
    print(f"\n{log_prefix}=== phase 3: {corruption} eta={pc.format_p(eta)} "
          f"(hard-criterion grid; pre-registered p_hat={_f(p_hat)}) ===",
          flush=True)
    src = ensure_run(args, manifest, corruption, eta=eta, source=True,
                     log_prefix=log_prefix)
    source_acc = pc.source_mean_acc(src) if src is not None else None

    points: dict[float, dict] = {}

    # The AT-p_hat run first (verdict requires its hard classification).
    if p_hat is not None:
        p_at = round(p_hat, 6)
        data = ensure_run(args, manifest, corruption, eta=eta, p=p_at,
                          log_prefix=log_prefix)
        rec = _store_point_hard(points, p_at, data, source_acc)
        print(f"{log_prefix}  AT p_hat={pc.format_p(p_at):>8s} -> "
              f"{'HARD-COLLAPSE' if rec.get('collapsed') else 'stable'} "
              f"({rec.get('criterion')}; soft: {rec.get('soft_criterion')})",
              flush=True)

    # Eta-scaled coarse grid.
    for p in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        if p in points:
            continue
        data = ensure_run(args, manifest, corruption, eta=eta, p=p,
                          log_prefix=log_prefix)
        rec = _store_point_hard(points, p, data, source_acc)
        if rec["valid"]:
            print(f"{log_prefix}  p={pc.format_p(p):>8s} -> "
                  f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable':13s} "
                  f"({rec['criterion']}; soft: {rec['soft_criterion']}) "
                  f"mean_acc={_f(rec['mean_acc'], 4)}", flush=True)

    # Fold in ladder rungs / anything on disk.
    for p in sc.discover_p_values_sc(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, eta):
        if p in points and points[p].get("valid"):
            continue
        data = sc.load_run_sc(args.results_dir, args.arch, corruption,
                              args.severity, args.seed, p=p, eta=eta)
        if data is None:
            continue
        rec = _store_point_hard(points, p, data, source_acc)
        print(f"{log_prefix}  (disk) p={pc.format_p(p):>8s} -> "
              f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable'} "
              f"({rec.get('criterion')})", flush=True)

    # Upward extension if every tested p hard-collapses (existing logic, cap 5).
    if pc.choose_pstar(list(points.values()))["bracket_high"] is None:
        print(f"{log_prefix}  all tested p hard-collapse; extending upward "
              f"(cap 5).", flush=True)
        extra = 0
        for p in pc.scaled_extend_grid(eta):
            if extra >= 5:
                break
            if p in points and points[p].get("valid"):
                if not points[p]["collapsed"]:
                    break
                continue
            data = ensure_run(args, manifest, corruption, eta=eta, p=p,
                              log_prefix=log_prefix)
            extra += 1
            rec = _store_point_hard(points, p, data, source_acc)
            print(f"{log_prefix}  extend p={pc.format_p(p):>8s} -> "
                  f"{'run_error' if not rec['valid'] else ('stable' if not rec['collapsed'] else 'HARD-COLLAPSE')}",
                  flush=True)
            if rec["valid"] and not rec["collapsed"]:
                break

    # Bisection on the hard boundary.
    for i in range(max(0, args.bisect_steps)):
        sel = pc.choose_pstar(list(points.values()))
        lo, hi = sel["bracket_low"], sel["bracket_high"]
        if lo is None or hi is None or (hi - lo) <= 1e-6:
            break
        mid = round((lo + hi) / 2.0, 6)
        if mid in points:
            break
        data = ensure_run(args, manifest, corruption, eta=eta, p=mid,
                          log_prefix=log_prefix)
        if data is None:
            points[mid] = {"p": mid, "collapsed": False, "valid": False}
            break
        rec = _store_point_hard(points, mid, data, source_acc)
        print(f"{log_prefix}  bisect[{i+1}] p={pc.format_p(mid):>8s} -> "
              f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable':13s} "
              f"({rec['criterion']}) (bracket was "
              f"[{pc.format_p(lo)},{pc.format_p(hi)}])", flush=True)

    final = pc.choose_pstar(list(points.values()))
    print(f"{log_prefix}  => p*_hard~={_f(final['p_star'])} "
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}]",
          flush=True)
    return points, final


# ----------------------------------------------------------------------------
# Phase 4 — verdict.
# ----------------------------------------------------------------------------

def evaluate_cell(args, preds, corruption, eta):
    """Read-only comparison of the measured hard boundary vs the
    pre-registered prediction, using the FROZEN ||g_bar||/p_hat from the
    predictions file (never re-measured)."""
    key = cell_key(corruption, eta)
    pred = preds["cells"].get(key, {})
    p_hat = pred.get("p_hat_hard")
    row_hard = sc.analyze_cell_sc(args.results_dir, args.arch, corruption,
                                  args.severity, eta, args.seed, hard_only=True)
    row_soft = sc.analyze_cell_sc(args.results_dir, args.arch, corruption,
                                  args.severity, eta, args.seed, hard_only=False)
    p_star = row_hard["p_star"]
    blo, bhi = row_hard["p_star_bracket_low"], row_hard["p_star_bracket_high"]

    within_tol = (p_hat is not None and p_hat > 0 and p_star is not None
                  and abs(p_star - p_hat) / p_hat <= PRED_TOL)
    in_bracket = (p_hat is not None and blo is not None and bhi is not None
                  and blo <= p_hat <= bhi)

    # The AT-p_hat run's hard classification + accuracy.
    at_info, at_acc, at_collapsed = None, None, None
    if p_hat is not None:
        p_at = pc.format_p(round(p_hat, 6))
        at_info = (row_hard.get("per_p") or {}).get(p_at)
        if at_info and at_info.get("valid"):
            at_acc = at_info.get("mean_acc")
            at_collapsed = bool(at_info.get("collapsed"))

    # Grid oracle (highest mean accuracy over all valid measured p).
    best_p, best_acc = None, None
    for pstr, q in (row_hard.get("per_p") or {}).items():
        if q.get("valid") and q.get("mean_acc") is not None:
            if best_acc is None or q["mean_acc"] > best_acc:
                best_p, best_acc = float(pstr), q["mean_acc"]

    ok = (within_tol or in_bracket) and (at_collapsed is False)
    return {
        "cell": key, "corruption": corruption, "eta": eta,
        "p_hat_prereg": p_hat, "gbar_frozen": pred.get("grad_norm_gbar"),
        "p_star_hard": p_star, "bracket": [blo, bhi],
        "collapse_criterion_hard": row_hard["collapse_criterion"],
        "p_star_soft": row_soft["p_star"],
        "collapse_criterion_soft": row_soft["collapse_criterion"],
        "within_25pct": within_tol, "pred_in_bracket": in_bracket,
        "at_p_hat_ran": at_info is not None and at_info.get("valid", False),
        "at_p_hat_hard_collapse": at_collapsed,
        "acc_at_p_hat": at_acc, "oracle_p": best_p, "oracle_acc": best_acc,
        "acc_ratio": (at_acc / best_acc) if (at_acc is not None and best_acc)
        else None,
        "cell_ok": bool(ok),
    }


def forward_verdict_md(preds, cells, violations) -> str:
    n_ok = sum(1 for c in cells if c["cell_ok"])
    n = len(cells)
    if n_ok >= 3:
        verdict = "CONFIRMED"
    elif n_ok == 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAILED"
    L = [f"# Stage-1 forward confirmation verdict: **{verdict}** ({n_ok}/{n} cells)",
         "",
         "Criterion (verbatim): CONFIRMED: >=3 of 4 cells with measured p*_hard "
         "within +/-25% of prediction (or within bisection bracket of it), and "
         "no hard collapse when running AT p_hat. PARTIAL: 2 of 4. "
         "FAILED: <=1 of 4.",
         "",
         f"Pre-registration: {preds.get('written_at_utc')} — "
         f"S_frozen = {_f(preds.get('S_frozen'), 5)} from "
         f"`{preds.get('S_frozen_source_file')}` "
         f"({preds.get('S_frozen_definition')}). "
         f"Formula: {preds.get('prediction_formula')}.",
         ""]
    if violations:
        L += ["**PRE-REGISTRATION VIOLATIONS (grid runs predating the "
              "predictions file):**"]
        L += [f"- {k}: p={v}" for k, v in violations.items()]
        L += [""]
    L += ["| cell | p_hat (prereg) | p*_hard | bracket | within 25% | in bracket "
          "| collapse@p_hat | acc@p_hat | oracle acc (p) | ratio | OK |",
          "|" + "---|" * 11]
    for c in cells:
        L.append("| " + " | ".join([
            c["cell"], _f(c["p_hat_prereg"]), _f(c["p_star_hard"]),
            f"[{_f(c['bracket'][0])},{_f(c['bracket'][1])}]",
            str(c["within_25pct"]), str(c["pred_in_bracket"]),
            ("n/a" if c["at_p_hat_hard_collapse"] is None
             else ("YES" if c["at_p_hat_hard_collapse"] else "no")),
            _f(c["acc_at_p_hat"], 4),
            f"{_f(c['oracle_acc'], 4)} ({_f(c['oracle_p'])})",
            _f(c["acc_ratio"], 4), "YES" if c["cell_ok"] else "no",
        ]) + " |")
    L += ["", "Soft boundaries (recorded alongside, never used for p*):",
          "| cell | p*_soft | criterion_soft |", "|---|---|---|"]
    for c in cells:
        L.append(f"| {c['cell']} | {_f(c['p_star_soft'])} | "
                 f"{c['collapse_criterion_soft']} |")
    L += ["", f"**Verdict: {verdict}.**",
          "", "**STOP — Part B complete; Stage 1b (E6/E7) awaits instruction "
          "and will use hard-only as the primary law criterion with soft "
          "reported as a separate boundary.**"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.results_dir) / "analysis" / "stage1b_predictions.json"

    if not args.analyze_only:
        if not args.ckpt_wrn or not Path(args.ckpt_wrn).exists():
            raise SystemExit(f"[fatal] WRN checkpoint not found: "
                             f"{args.ckpt_wrn!r} (or pass --analyze-only)")
        frozen, frozen_path = load_frozen_slope(args)
        manifest = Manifest(Path(args.results_dir), args,
                            campaign="stage1b_forward",
                            filename="stage1b_manifest.jsonl")
        print(f"[forward] env={manifest.env}", flush=True)
        print(f"[forward] S_frozen={_f(frozen['S_frozen'], 5)} from "
              f"{frozen_path}", flush=True)

        t0 = time.time()
        preds, pred_path = build_predictions(args, manifest, frozen, frozen_path)
        for corruption in args.corruptions:
            for eta in sorted(set(args.etas)):
                key = cell_key(corruption, eta)
                p_hat = preds["cells"].get(key, {}).get("p_hat_hard")
                sweep_cell_hard(args, manifest, corruption, eta, p_hat)
        print(f"\n[forward] runs done in {time.time() - t0:.1f}s.", flush=True)

    if not pred_path.exists():
        raise SystemExit(f"[fatal] predictions file missing: {pred_path}")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))

    cells = [evaluate_cell(args, preds, corruption, eta)
             for corruption in args.corruptions
             for eta in sorted(set(args.etas))]
    md = forward_verdict_md(preds, cells,
                            preds.get("preexisting_grid_violations") or {})
    out_md = Path(args.results_dir) / "analysis" / "stage1_forward_verdict.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"[saved] {out_md}")
    print(f"[saved] {pred_path}")


if __name__ == "__main__":
    main()
