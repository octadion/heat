"""
Stage 1 / E1 — the ||g_bar||-factor test (GPU campaign orchestrator).

Single-corruption p* cells on WRN-28-10, CIFAR-10-C severity 5:
  corruptions = {gaussian_noise, elastic_transform} (CALIBRATION — the only
                cells ever used to fit the slope) +
                {impulse_noise, contrast}          (HELD-OUT — zero-shot test)
  etas        = {5e-4, 1e-3, 2e-3} per corruption
  seed 42, batch 64.

Each (corruption, eta) cell gets: a source (no-adapt) run, a p_ref run from the
eta-scaled ladder, the eta-scaled coarse p-grid, the existing upward-extension
logic, and bisection of the stable/collapse bracket — exactly the continual
sweep's policy (run_pstar_sweep.py), re-pointed at single-corruption streams.

The stream is one corruption's FULL severity-5 split (run_tier2.py --protocol p9
--corruptions <c>, natively supported — no runner change), i.e. 10,000 images
~= 157 steps/run at batch 64, NOT the 15-corruption continual.

Everything routes through scripts/run_tier2.py (we never reimplement the
adaptation loop). Collapse math is pstar_common.classify_run UNCHANGED. Every
run's JSON is written to --results-dir immediately (Drive-backed => resumable:
a run whose corruption+eta-tagged JSON exists is skipped). A per-run manifest
line (arch, dataset, corruption, severity, eta, p, seed, p_ref_used, criterion,
git hash, GPU name, torch version, elapsed) is appended to
<results-dir>/stage1_e1_manifest.jsonl.

The final p*/verdict is NOT decided here — scripts/analyze_stage1.py is the
single source of truth and recomputes everything from the JSONs on disk.

Usage (Colab, Drive-backed):
  python scripts/run_stage1_e1.py \
      --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-wrn experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1_common as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 / E1 single-corruption p* sweep")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Drive-backed dir for run JSONs (resumable).")
    p.add_argument("--ckpt-wrn", type=str, required=True,
                   help="Path to wrn28_10_final.pt (same artifact as the eta-sweep).")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=sc.E1_ARCH, choices=["wrn28_10"],
                   help="E1 is defined on wrn28_10 only.")
    p.add_argument("--severity", type=int, default=sc.E1_SEVERITY)
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=list(sc.E1_CORRUPTIONS),
                   help="Cells to run. Default: calibration first "
                        "(gaussian_noise, elastic_transform), then held-out "
                        "(impulse_noise, contrast).")
    p.add_argument("--etas", type=float, nargs="+", default=list(sc.E1_ETAS))
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID),
                   help="BASE coarse p-grid (at eta=1e-3); scaled by eta/1e-3 "
                        "per cell, exactly like the continual sweep.")
    p.add_argument("--bisect-steps", type=int, default=3,
                   help="Bisections of the stable/collapse bracket (plan: 2-3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def _fmt(v):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


class Manifest:
    """Append-only per-run log (§1.4). One JSONL line per ensure_run outcome."""

    def __init__(self, results_dir: Path, args, campaign: str = "stage1_e1",
                 filename: str = "stage1_e1_manifest.jsonl"):
        self.path = Path(results_dir) / filename
        self.env = sc.environment_fingerprint(REPO_ROOT)
        self.base = {"campaign": campaign, "dataset": "cifar10",
                     "severity": args.severity, "seed": args.seed,
                     "batch_size": args.batch_size, **self.env}

    def log(self, **fields):
        row = {**self.base, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:  # never let logging kill the sweep
            print(f"[manifest warn] {exc}", flush=True)


def ensure_run(args, manifest, corruption, *, eta, p=None, source=False,
               log_prefix="") -> dict | None:
    """Ensure the JSON for (corruption, eta, p|source) exists; run it if not."""
    out_path = sc.run_output_path_sc(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, p=p,
                                     source=source, eta=eta)
    if out_path.exists():
        try:
            data = pc.load_json(out_path)
            manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                         p=(None if source else p), source=source,
                         status="reused")
            return data
        except Exception as exc:
            print(f"{log_prefix}[warn] existing JSON unreadable, re-running: "
                  f"{out_path.name} ({exc})", flush=True)

    label = "source" if source else f"p={pc.format_p(p)}"
    cmd = sc.build_run_command_sc(
        run_tier2=RUN_TIER2, arch=args.arch, checkpoint=args.ckpt_wrn,
        corruption=corruption, severity=args.severity, seed=args.seed,
        results_dir=Path(args.results_dir), c10c_root=args.c10c_root,
        heat_lr=eta, batch_size=args.batch_size, num_workers=args.num_workers,
        p=p, source=source,
    )
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    elapsed = time.time() - t0

    if not ok or not out_path.exists():
        err_path = sc.run_error_path_sc(args.results_dir, args.arch, corruption,
                                        args.severity, args.seed, p=p,
                                        source=source, eta=eta)
        try:
            err_path.write_text(
                f"run_error {args.arch} {corruption} sev{args.severity} "
                f"lr{pc.format_p(eta)} {label}\nelapsed={elapsed:.1f}s\n\n{tail}",
                encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                     p=(None if source else p), source=source,
                     status="run_error", elapsed_s=round(elapsed, 1))
        print(f"{log_prefix}[run_error] {corruption} lr{pc.format_p(eta)} {label} "
              f"({elapsed:.1f}s) -> recorded {err_path.name}, continuing.",
              flush=True)
        return None

    manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                 p=(None if source else p), source=source, status="ok",
                 elapsed_s=round(elapsed, 1))
    print(f"{log_prefix}[ok] {corruption} lr{pc.format_p(eta)} {label} "
          f"({elapsed:.1f}s) -> {out_path.name}", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception as exc:
        print(f"{log_prefix}[warn] wrote but could not reload {out_path.name}: "
              f"{exc}", flush=True)
        return None


def _store_point(points, p, data, source_acc, pref_drift):
    if data is None:
        points[p] = {"p": p, "collapsed": False, "valid": False}
        return points[p]
    v = pc.classify_run(data, source_acc, pref_drift)
    points[p] = {"p": p, "collapsed": v["collapsed"], "valid": True,
                 "criterion": v["criterion"], "mean_acc": v["mean_acc"],
                 "gbar": v["gbar"]}
    return points[p]


def select_pref(args, manifest, corruption, eta, source_acc, log_prefix=""):
    """Known-stable reference tether from the eta-scaled ladder (same policy
    as the continual sweep: hard + below-source stability only)."""
    candidates = pc.scaled_pref_ladder(args.arch, eta)
    last_data, last_p = None, None
    for cand in candidates:
        data = ensure_run(args, manifest, corruption, eta=eta, p=cand,
                          log_prefix=log_prefix)
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

    print(f"{log_prefix}  [warn] no stable p_ref among "
          f"{[pc.format_p(c) for c in candidates]}; using last="
          f"{pc.format_p(last_p) if last_p is not None else 'none'} "
          f"(||g_bar|| may be unreliable).", flush=True)
    if last_data is not None:
        gbar_res = pc.grad_norm_gbar(last_data)
        return last_p, pc.drift_stationary(last_data), (gbar_res[0] if gbar_res else None)
    return None, None, None


def sweep_cell(args, manifest, corruption, eta, log_prefix=""):
    """Full p* search for one (corruption, eta) cell: source, p_ref, scaled
    coarse grid, fold-in of on-disk rungs, upward extension, bisection."""
    role = "CALIBRATION" if corruption in sc.CALIBRATION_CORRUPTIONS else "HELD-OUT"
    print(f"\n{log_prefix}=== {args.arch} {corruption} ({role}) sev{args.severity} "
          f"eta={pc.format_p(eta)} ===", flush=True)

    # 1. Source baseline for THIS corruption (soft-collapse reference).
    src_data = ensure_run(args, manifest, corruption, eta=eta, source=True,
                          log_prefix=log_prefix)
    source_acc = pc.source_mean_acc(src_data) if src_data is not None else None
    print(f"{log_prefix}  source_acc={_fmt(source_acc)}", flush=True)

    # 2. Reference p_ref for ||g_bar|| (eta-scaled ladder).
    p_ref, pref_drift, gbar = select_pref(args, manifest, corruption, eta,
                                          source_acc, log_prefix=log_prefix)

    # 3. Coarse grid, scaled by eta/1e-3.
    points: dict[float, dict] = {}
    for p in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        data = ensure_run(args, manifest, corruption, eta=eta, p=p,
                          log_prefix=log_prefix)
        rec = _store_point(points, p, data, source_acc, pref_drift)
        if rec["valid"]:
            print(f"{log_prefix}  p={pc.format_p(p):>8s} -> "
                  f"{'COLLAPSE' if rec['collapsed'] else 'stable':8s} "
                  f"({rec['criterion']}) mean_acc={_fmt(rec['mean_acc'])} "
                  f"||g_bar||={_fmt(rec['gbar'])}", flush=True)

    # 3b. Fold in every on-disk run for this cell (e.g. p_ref-ladder rungs).
    for p in sc.discover_p_values_sc(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, eta):
        if p in points and points[p].get("valid"):
            continue
        data = sc.load_run_sc(args.results_dir, args.arch, corruption,
                              args.severity, args.seed, p=p, eta=eta)
        if data is None:
            continue
        rec = _store_point(points, p, data, source_acc, pref_drift)
        print(f"{log_prefix}  (disk) p={pc.format_p(p):>8s} -> "
              f"{'COLLAPSE' if rec['collapsed'] else 'stable':8s} "
              f"({rec.get('criterion')})", flush=True)

    # 3c. Upward extension if everything collapses (existing logic, cap 5).
    sel = pc.choose_pstar(list(points.values()))
    if sel["bracket_high"] is None:
        print(f"{log_prefix}  all tested p collapse; extending grid upward "
              f"(cap 5 runs).", flush=True)
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
            rec = _store_point(points, p, data, source_acc, pref_drift)
            status = ("run_error" if not rec["valid"]
                      else ("stable" if not rec["collapsed"] else "COLLAPSE"))
            print(f"{log_prefix}  extend p={pc.format_p(p):>8s} -> {status}",
                  flush=True)
            if rec["valid"] and not rec["collapsed"]:
                break
        if pc.choose_pstar(list(points.values()))["bracket_high"] is None:
            tested = max((q for q in points), default=None)
            print(f"{log_prefix}  [warn] law region exhausted: all tested p "
                  f"(through {pc.format_p(tested) if tested is not None else 'n/a'})"
                  f" collapse; p* left UNRESOLVED for this cell.", flush=True)

    # 4. Bisect the stable/collapse boundary.
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
        v = pc.classify_run(data, source_acc, pref_drift)
        points[mid] = {"p": mid, "collapsed": v["collapsed"], "valid": True,
                       "criterion": v["criterion"]}
        print(f"{log_prefix}  bisect[{i+1}] p={pc.format_p(mid):>8s} -> "
              f"{'COLLAPSE' if v['collapsed'] else 'stable':8s} "
              f"({v['criterion']}) (bracket was "
              f"[{pc.format_p(lo)},{pc.format_p(hi)}])", flush=True)

    final = pc.choose_pstar(list(points.values()))
    print(f"{log_prefix}  => p*~={_fmt(final['p_star'])} "
          f"bracket=[{_fmt(final['bracket_low'])},{_fmt(final['bracket_high'])}] "
          f"p_ref={pc.format_p(p_ref) if p_ref is not None else 'none'} "
          f"||g_bar||={_fmt(gbar)} eta*||g_bar||={_fmt((gbar or 0) * eta)}",
          flush=True)
    return {
        "corruption": corruption, "role": role, "eta": eta,
        "source_acc": source_acc, "p_ref": p_ref, "gbar": gbar,
        "p_star": final["p_star"],
        "bracket": [final["bracket_low"], final["bracket_high"]],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # run_tier2.py prints a U+2192 arrow in its P9 summary AFTER the run but
    # BEFORE saving the JSON; on a cp1252 console (Windows) that would crash
    # the child and lose the run. Force UTF-8 in children (no-op on Colab).
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    if not Path(args.ckpt_wrn).exists():
        raise SystemExit(f"[fatal] WRN checkpoint not found: {args.ckpt_wrn}")

    manifest = Manifest(Path(args.results_dir), args)
    print(f"[e1] results_dir={args.results_dir}", flush=True)
    print(f"[e1] env={manifest.env}", flush=True)
    print(f"[e1] corruptions={args.corruptions} etas="
          f"{[pc.format_p(e) for e in args.etas]} base_p_grid="
          f"{[pc.format_p(p) for p in args.p_grid]} "
          f"bisect_steps={args.bisect_steps} seed={args.seed}", flush=True)
    print(f"[e1] {len(args.corruptions) * len(args.etas)} (corruption, eta) "
          f"cells; calibration cells first.", flush=True)

    t0 = time.time()
    summaries = []
    for corruption in args.corruptions:
        for eta in sorted(set(args.etas)):
            summaries.append(sweep_cell(args, manifest, corruption, eta))

    print(f"\n[e1] done in {time.time() - t0:.1f}s. Run "
          f"scripts/analyze_stage1.py for E2-E5 + the Stage-1 verdict.", flush=True)
    print("[e1] cell summary:", flush=True)
    for s in summaries:
        exg = (s["gbar"] * s["eta"]) if s["gbar"] is not None else None
        print(f"  {s['corruption']:18s} ({s['role']:11s}) "
              f"lr{pc.format_p(s['eta']):>7s}  p*~={_fmt(s['p_star'])}  "
              f"||g_bar||={_fmt(s['gbar'])}  eta*||g_bar||={_fmt(exg)}  "
              f"p_ref={pc.format_p(s['p_ref']) if s['p_ref'] is not None else 'none'}",
              flush=True)


if __name__ == "__main__":
    main()
