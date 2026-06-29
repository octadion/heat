"""
Orchestrate the drift-budget-law (p*) sweep.

For each (arch, severity) it:
  1. Runs the source (no-adapt) baseline once -> source accuracy for that
     severity (used by the soft-collapse criterion).
  2. Picks a known-stable reference tether p_ref (escalating up a fallback
     ladder if the default p_ref itself collapses) -> the run used to measure
     ||g_bar|| and the reference drift.
  3. Runs the coarse p-grid, applies the collapse criterion, brackets the
     stable/collapse boundary, and BISECTS it twice to localize p* to ~+-0.0025.

Everything routes through scripts/run_tier2.py --protocol p9 (we never
reimplement the adaptation loop). Every run's JSON is written to RESULTS_DIR
immediately by run_tier2, so the sweep is fully resumable: a run whose JSON
already exists is skipped, so re-running after a Colab disconnect continues.

We do NOT decide the final p* here -- scripts/analyze_pstar_law.py is the single
source of truth for the deliverable and recomputes p* from all runs present.
The sweep only ensures the necessary runs exist and uses the SAME bracketing
logic (pstar_common.choose_pstar) to decide which bisection midpoints to run.

Usage (cheaper arch first is automatic):
  python scripts/run_pstar_sweep.py \
      --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-resnet18 experiments/checkpoints/resnet18_final.pt \
      --ckpt-wrn      experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c \
      --archs resnet18 wrn28_10 --severities 1 3 5 \
      --p-grid 0.0 0.005 0.010 0.020 --bisect-steps 2 --seed 42

  # Pipeline-validation gate (section 6.1 anchor) -- run this before the sweep:
  python scripts/run_pstar_sweep.py --sanity-check-only \
      --results-dir <dir> --ckpt-wrn <wrn.pt> --c10c-root data/cifar10c
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"


# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="p* drift-budget-law sweep")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Drive-backed dir for run JSONs (resumable).")
    p.add_argument("--ckpt-resnet18", type=str, default="",
                   help="Path to resnet18_final.pt (required if resnet18 in --archs).")
    p.add_argument("--ckpt-wrn", type=str, default="",
                   help="Path to wrn28_10_final.pt (required if wrn28_10 in --archs).")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--archs", type=str, nargs="+", default=list(pc.DEFAULT_ARCHS),
                   choices=["resnet18", "wrn28_10"])
    p.add_argument("--severities", type=int, nargs="+", default=None,
                   help="Global severity override applied to ALL archs. If "
                        "omitted, per-arch overrides / per-arch defaults are used.")
    p.add_argument("--severities-resnet18", type=int, nargs="+", default=None,
                   help="Per-arch severity override for resnet18 (wins over "
                        "--severities and the per-arch default).")
    p.add_argument("--severities-wrn28_10", type=int, nargs="+", default=None,
                   help="Per-arch severity override for wrn28_10 (wins over "
                        "--severities and the per-arch default).")
    p.add_argument("--p-grid", type=float, nargs="+", default=list(pc.DEFAULT_P_GRID))
    p.add_argument("--bisect-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heat-lr", type=float, default=pc.ETA)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sanity-check-only", action="store_true",
                   help="Run only the WRN severity-5 pipeline-validation gate "
                        "(section 6.1 anchor) and exit non-zero if it fails.")
    return p.parse_args()


def _ckpt_for(arch: str, args) -> str:
    ckpt = args.ckpt_resnet18 if arch == "resnet18" else args.ckpt_wrn
    if not ckpt:
        raise SystemExit(
            f"[fatal] no checkpoint provided for {arch}. Pass "
            f"{'--ckpt-resnet18' if arch == 'resnet18' else '--ckpt-wrn'} <path>."
        )
    if not Path(ckpt).exists():
        raise SystemExit(f"[fatal] checkpoint for {arch} not found: {ckpt}")
    return ckpt


def _severities_for(arch: str, args) -> list[int]:
    """Resolve severities for an arch: per-arch flag > global --severities >
    per-arch default."""
    per_arch = getattr(args, f"severities_{arch}", None)
    if per_arch:
        return list(per_arch)
    if args.severities:
        return list(args.severities)
    return list(pc.DEFAULT_SEVERITIES_BY_ARCH.get(arch, pc.DEFAULT_SEVERITIES))


def _store_point(points, p, data, source_acc, pref_drift):
    """Classify a loaded run and record it in `points` keyed by p."""
    if data is None:
        points[p] = {"p": p, "collapsed": False, "valid": False}
        return points[p]
    v = pc.classify_run(data, source_acc, pref_drift)
    points[p] = {"p": p, "collapsed": v["collapsed"], "valid": True,
                 "criterion": v["criterion"], "mean_acc": v["mean_acc"],
                 "gbar": v["gbar"]}
    return points[p]


def ensure_run(args, arch, severity, ckpt, *, p=None, source=False,
               log_prefix="") -> dict | None:
    """Ensure the run JSON for (arch, sev, p|source) exists; run it if not.

    Returns the loaded JSON dict, or None if the run failed (recorded as a
    run_error sidecar so the whole sweep does not crash).
    """
    out_path = pc.run_output_path(args.results_dir, arch, severity, args.seed,
                                  p=p, source=source)
    if out_path.exists():
        try:
            return pc.load_json(out_path)
        except Exception as exc:
            print(f"{log_prefix}[warn] existing JSON unreadable, re-running: "
                  f"{out_path.name} ({exc})", flush=True)

    label = "source" if source else f"p={pc.format_p(p)}"
    t0 = time.time()
    cmd = pc.build_run_command(
        run_tier2=RUN_TIER2, arch=arch, checkpoint=ckpt, severity=severity,
        seed=args.seed, results_dir=Path(args.results_dir),
        c10c_root=args.c10c_root, heat_lr=args.heat_lr,
        batch_size=args.batch_size, num_workers=args.num_workers,
        p=p, source=source,
    )
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    elapsed = time.time() - t0

    if not ok or not out_path.exists():
        err_path = pc.run_error_path(args.results_dir, arch, severity, args.seed,
                                     p=p, source=source)
        try:
            err_path.write_text(
                f"run_error for {arch} sev{severity} {label}\n"
                f"elapsed={elapsed:.1f}s\n\n{tail}",
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"{log_prefix}[run_error] {arch} sev{severity} {label} "
              f"({elapsed:.1f}s) -> recorded {err_path.name}, continuing.",
              flush=True)
        return None

    print(f"{log_prefix}[ok] {arch} sev{severity} {label} ({elapsed:.1f}s) "
          f"-> {out_path.name}", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception as exc:
        print(f"{log_prefix}[warn] wrote but could not reload {out_path.name}: {exc}",
              flush=True)
        return None


def select_pref(args, arch, severity, ckpt, source_acc, log_prefix=""):
    """Find a known-stable reference tether p_ref for ||g_bar|| (section 6.1).

    Stability here uses hard collapse + below-source only (NOT the drift-ratio
    criterion, which is defined relative to p_ref itself). Returns
    (p_ref_used, pref_drift, gbar) -- gbar/pref_drift are None if no stable
    reference could be found.
    """
    candidates = pc.PREF_FALLBACKS.get(arch, [pc.PREF_DEFAULT.get(arch, 0.01)])
    last_data = None
    last_p = None
    for cand in candidates:
        data = ensure_run(args, arch, severity, ckpt, p=cand, log_prefix=log_prefix)
        if data is None:
            continue
        last_data, last_p = data, cand
        verdict = pc.classify_run(data, source_acc, pref_drift=None,
                                  use_drift_criterion=False)
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        if not verdict["collapsed"]:
            drift = pc.drift_stationary(data)
            print(f"{log_prefix}  p_ref={pc.format_p(cand)} STABLE  "
                  f"||g_bar||={_fmt(gbar)}  drift={_fmt(drift)}", flush=True)
            return cand, drift, gbar
        print(f"{log_prefix}  p_ref={pc.format_p(cand)} collapsed "
              f"({verdict['criterion']}); escalating.", flush=True)

    print(f"{log_prefix}  [warn] no stable p_ref found among "
          f"{[pc.format_p(c) for c in candidates]}; using last="
          f"{pc.format_p(last_p) if last_p is not None else 'none'} "
          f"(||g_bar|| may be unreliable).", flush=True)
    if last_data is not None:
        gbar_res = pc.grad_norm_gbar(last_data)
        return last_p, pc.drift_stationary(last_data), (gbar_res[0] if gbar_res else None)
    return None, None, None


def sweep_cell(args, arch, severity, ckpt, log_prefix=""):
    """Run the full p* search for one (arch, severity). Returns a summary dict."""
    print(f"\n{log_prefix}=== {arch} severity={severity} ===", flush=True)

    # 1. Source baseline.
    src_data = ensure_run(args, arch, severity, ckpt, source=True, log_prefix=log_prefix)
    source_acc = pc.source_mean_acc(src_data) if src_data is not None else None
    print(f"{log_prefix}  source_acc={_fmt(source_acc)}", flush=True)

    # 2. Reference p_ref for ||g_bar||.
    p_ref, pref_drift, gbar = select_pref(args, arch, severity, ckpt, source_acc,
                                          log_prefix=log_prefix)

    # 3. Coarse grid (p_ref candidates already ran; grid points reuse them).
    points: dict[float, dict] = {}
    for p in sorted(set(args.p_grid)):
        data = ensure_run(args, arch, severity, ckpt, p=p, log_prefix=log_prefix)
        rec = _store_point(points, p, data, source_acc, pref_drift)
        if rec["valid"]:
            print(f"{log_prefix}  p={pc.format_p(p):>7s} -> "
                  f"{'COLLAPSE' if rec['collapsed'] else 'stable':8s} "
                  f"({rec['criterion']}) mean_acc={_fmt(rec['mean_acc'])} "
                  f"||g_bar||={_fmt(rec['gbar'])}", flush=True)

    # 3b. Fold in EVERY existing run on disk for this cell -- notably the
    # p_ref-ladder rungs (0.040 / 0.080) that the coarse grid never visits. At
    # high severity the whole coarse grid collapses, and surfacing those stable
    # rungs is what gives choose_pstar a real [collapse, stable] bracket. These
    # runs already exist on disk, so we LOAD only (never re-run).
    for p in pc.discover_p_values(args.results_dir, arch, severity, args.seed):
        if p in points and points[p].get("valid"):
            continue
        path = pc.run_output_path(args.results_dir, arch, severity, args.seed, p=p)
        data = None
        if path.exists():
            try:
                data = pc.load_json(path)
            except Exception:
                data = None
        if data is None:
            continue
        rec = _store_point(points, p, data, source_acc, pref_drift)
        print(f"{log_prefix}  (disk) p={pc.format_p(p):>7s} -> "
              f"{'COLLAPSE' if rec['collapsed'] else 'stable':8s} "
              f"({rec.get('criterion')})", flush=True)

    # 3c. If even the strongest run on disk collapses (no stable point at all,
    # so choose_pstar can't bracket), extend the grid UPWARD until one p is
    # stable -- capped at ~5 new runs. If nothing is stable, p* is legitimately
    # unresolved for this cell ("law region exhausted") and we say so.
    sel = pc.choose_pstar(list(points.values()))
    if sel["bracket_high"] is None:
        print(f"{log_prefix}  all tested p collapse; extending grid upward "
              f"(cap 5 runs).", flush=True)
        extra = 0
        for p in pc.EXTEND_P_GRID:
            if extra >= 5:
                break
            if p in points and points[p].get("valid"):
                if not points[p]["collapsed"]:
                    break  # already have a stable upper anchor
                continue
            data = ensure_run(args, arch, severity, ckpt, p=p, log_prefix=log_prefix)
            extra += 1
            rec = _store_point(points, p, data, source_acc, pref_drift)
            status = ("run_error" if not rec["valid"]
                      else ("stable" if not rec["collapsed"] else "COLLAPSE"))
            print(f"{log_prefix}  extend p={pc.format_p(p):>7s} -> {status}",
                  flush=True)
            if rec["valid"] and not rec["collapsed"]:
                break
        if pc.choose_pstar(list(points.values()))["bracket_high"] is None:
            tested = max((q for q in points), default=None)
            print(f"{log_prefix}  [warn] law region exhausted: all tested p "
                  f"(through {pc.format_p(tested) if tested is not None else 'n/a'}) "
                  f"collapse; p* left UNRESOLVED for this cell.", flush=True)

    # 4. Bisect the stable/collapse boundary.
    for i in range(max(0, args.bisect_steps)):
        sel = pc.choose_pstar(list(points.values()))
        lo, hi = sel["bracket_low"], sel["bracket_high"]
        if lo is None or hi is None or (hi - lo) <= 1e-6:
            break
        mid = round((lo + hi) / 2.0, 6)
        if mid in points:
            break  # nothing new to learn
        data = ensure_run(args, arch, severity, ckpt, p=mid, log_prefix=log_prefix)
        if data is None:
            points[mid] = {"p": mid, "collapsed": False, "valid": False}
            break
        v = pc.classify_run(data, source_acc, pref_drift)
        points[mid] = {"p": mid, "collapsed": v["collapsed"], "valid": True,
                       "criterion": v["criterion"]}
        print(f"{log_prefix}  bisect[{i+1}] p={pc.format_p(mid):>7s} -> "
              f"{'COLLAPSE' if v['collapsed'] else 'stable':8s} "
              f"({v['criterion']}) (bracket was [{pc.format_p(lo)},{pc.format_p(hi)}])",
              flush=True)

    final = pc.choose_pstar(list(points.values()))
    print(f"{log_prefix}  => p*~={_fmt(final['p_star'])} "
          f"bracket=[{_fmt(final['bracket_low'])},{_fmt(final['bracket_high'])}] "
          f"p_ref={pc.format_p(p_ref) if p_ref is not None else 'none'} "
          f"||g_bar||={_fmt(gbar)} eta*||g_bar||={_fmt((gbar or 0) * args.heat_lr)}",
          flush=True)
    return {
        "arch": arch, "severity": severity, "source_acc": source_acc,
        "p_ref": p_ref, "gbar": gbar, "p_star": final["p_star"],
        "bracket": [final["bracket_low"], final["bracket_high"]],
    }


def run_sanity_check(args) -> int:
    """WRN severity-5 pipeline-validation gate (section 6.1 anchor).

    Known anchor: WRN sev-5: p=0 -> NaN ~step 603; p=0.005 -> NaN ~step 920;
    p>=0.010 -> stable, with stationary ||g_bar|| ~= 13. We require the QUALITATIVE
    pattern (p0 collapses, p005 collapses later than p0, p010 stable with ||g_bar||
    in a sane band). Exact step numbers vary with hardware/seed, so they are
    reported but not asserted to the integer.
    """
    print("\n=== PIPELINE VALIDATION: WRN-28-10 severity 5 ===", flush=True)
    ckpt = _ckpt_for("wrn28_10", args)
    sev = 5
    results = {}
    for p in (0.0, 0.005, 0.010):
        data = ensure_run(args, "wrn28_10", sev, ckpt, p=p, log_prefix="  ")
        if data is None:
            print(f"  [FAIL] run p={pc.format_p(p)} did not complete.", flush=True)
            return 2
        nan_step = pc.first_nonfinite_step(pc.stream_rows(data))
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        summ = pc.heat_summary(data)
        collapsed = nan_step is not None
        results[p] = {"nan_step": nan_step, "gbar": gbar,
                      "mean_acc": summ.get("mean_accuracy"), "collapsed": collapsed}
        print(f"  p={pc.format_p(p):>6s}  nan_step={nan_step}  "
              f"||g_bar||={_fmt(gbar)}  mean_acc={_fmt(summ.get('mean_accuracy'))}  "
              f"{'COLLAPSE' if collapsed else 'stable'}", flush=True)

    r0, r005, r010 = results[0.0], results[0.005], results[0.010]
    checks = {
        "p0_collapses": r0["collapsed"],
        "p005_collapses": r005["collapsed"],
        "p010_stable": not r010["collapsed"],
        "p005_later_than_p0": (
            r0["nan_step"] is not None and r005["nan_step"] is not None
            and r005["nan_step"] > r0["nan_step"]
        ),
        "p010_gbar_in_band": (r010["gbar"] is not None and 8.0 <= r010["gbar"] <= 18.0),
    }
    print("\n  --- anchor checks ---", flush=True)
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}", flush=True)

    passed = all(checks.values())
    if passed:
        print("\n[PIPELINE VALIDATION PASSED] Known WRN sev-5 collapse pattern "
              "reproduced. Safe to run the full sweep.", flush=True)
        return 0
    print("\n[PIPELINE VALIDATION FAILED] The known WRN sev-5 anchor did NOT "
          "reproduce. Do NOT trust the sweep. Check: correct checkpoint, all 5 "
          "severities present, --heat-diagnostic-snapshot enabled, GPU in use.",
          flush=True)
    return 1


def _fmt(v):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.sanity_check_only:
        sys.exit(run_sanity_check(args))

    sev_map = {arch: _severities_for(arch, args) for arch in args.archs}
    print(f"[sweep] results_dir={args.results_dir}", flush=True)
    print(f"[sweep] archs={args.archs} severities(by arch)={sev_map} "
          f"p_grid={[pc.format_p(p) for p in args.p_grid]} "
          f"bisect_steps={args.bisect_steps} seed={args.seed}", flush=True)
    n_cells = sum(len(v) for v in sev_map.values())
    print(f"[sweep] {n_cells} (arch,severity) cells. resnet18 processed first.",
          flush=True)

    # resnet18 (cheaper) first; severities in configured order.
    arch_order = sorted(args.archs, key=lambda a: 0 if a == "resnet18" else 1)
    t_start = time.time()
    summaries = []
    for arch in arch_order:
        ckpt = _ckpt_for(arch, args)
        for sev in sev_map[arch]:
            summaries.append(sweep_cell(args, arch, sev, ckpt, log_prefix=""))

    print(f"\n[sweep] done in {time.time() - t_start:.1f}s. "
          f"Run scripts/analyze_pstar_law.py for the verdict.", flush=True)
    print("[sweep] cell summary:", flush=True)
    for s in summaries:
        print(f"  {s['arch']:10s} sev{s['severity']}  p*~={_fmt(s['p_star'])}  "
              f"||g_bar||={_fmt(s['gbar'])}  p_ref={pc.format_p(s['p_ref']) if s['p_ref'] is not None else 'none'}",
              flush=True)


if __name__ == "__main__":
    main()
