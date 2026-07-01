"""
DomainNet-126 eta-sweep (resnet50, source=real, target=clipart, seed=42 ONLY).

Decisive single-target slice. Reuses the pstar_common engine (build_run_command,
scaled grids, choose_pstar, execute_run) and the DomainNet runner
(scripts/run_p9_domainnet.py, which reuses run_tier2.run_p9). The CIFAR sweep
(run_pstar_sweep.py) is left untouched; this is a sibling orchestrator because
the DomainNet slice needs a different boundary (HARD-only) + baselines.

NOTE ("TFF"): code identifier is `heat`; the analysis labels results TFF.

Per eta (DESCENDING so the informative high-eta points land first):
  * heat over the eta-scaled coarse p-grid + upward extension {0.1, 0.2}
    (graceful-fallback check), classified by HARD collapse only
    (nan_inf OR mean_acc <= chance_acc ~= 0.02, since soft:below_source fires at
    all p here and cannot define a boundary), then bisect the boundary.
Plus, ONCE (eta-independent): source (no-adapt) and bn_adapt baselines.

The analysis (analyze_domainnet_clipart.py) is the source of truth for
p*/gbar/mechanism/Q1-Q3; this script only ensures the JSONs exist (resumable).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc

REPO_ROOT = Path(__file__).resolve().parents[1]
DN_RUNNER = REPO_ROOT / "scripts" / "run_p9_domainnet.py"
ARCH = "resnet50"
SEV = 5  # filename placeholder (DomainNet has no severity)
UPWARD_P = [0.1, 0.2]  # graceful-fallback anchors appended to every eta's grid


def parse_args():
    p = argparse.ArgumentParser(description="DomainNet-126 eta-sweep (resnet50)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ckpt-real", type=str, required=True)
    p.add_argument("--target-domain", type=str, default="clipart")
    p.add_argument("--etas", type=float, nargs="+",
                   default=[1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2e-4])
    p.add_argument("--p-grid", type=float, nargs="+", default=list(pc.DEFAULT_P_GRID),
                   help="BASE coarse grid (at eta=1e-3); scaled per eta.")
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--chance-acc", type=float, default=0.02,
                   help="Hard-collapse accuracy floor (126-class chance ~0.008).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    # Numeric regime: inherit (default; == CIFAR/run_tier2, comparable across
    # experiments), on (fast A100, differs), off (full FP32, most reproducible).
    p.add_argument("--tf32-mode", choices=["inherit", "on", "off"], default="inherit")
    return p.parse_args()


def _fmt(v):
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def ensure_run(args, *, method="heat", p=None, eta=pc.ETA, log=""):
    """Ensure the JSON for a domainnet run exists; run it if not. Returns data|None."""
    if method == "heat":
        out_path = pc.run_output_path(args.results_dir, ARCH, SEV, args.seed, p=p, eta=eta)
    else:
        tag = "pstar_source" if method == "source" else "pstar_bnadapt"
        out_path = Path(args.results_dir) / f"p9_{ARCH}_{tag}_seed{args.seed}_sev{SEV}.json"
    if out_path.exists():
        try:
            return pc.load_json(out_path)
        except Exception:
            pass
    cmd = pc.build_run_command(
        run_tier2=None, arch=ARCH, checkpoint=args.ckpt_real, severity=SEV,
        seed=args.seed, results_dir=Path(args.results_dir), c10c_root=args.data_root,
        heat_lr=eta, batch_size=args.batch_size, num_workers=args.num_workers, p=p,
        dataset="domainnet126", data_root=args.data_root,
        target_domain=args.target_domain, method=method, domainnet_runner=DN_RUNNER,
        tf32_mode=args.tf32_mode,
    )
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        errp = out_path.with_suffix(".run_error.txt")
        try:
            errp.write_text(f"run_error {method} p={p} eta={eta}\n{tail}", encoding="utf-8")
        except Exception:
            pass
        print(f"{log}[run_error] {method} p={p} eta={pc.format_p(eta)} ({dt:.0f}s)", flush=True)
        return None
    print(f"{log}[ok] {method} p={p} eta={pc.format_p(eta)} ({dt:.0f}s) -> {out_path.name}",
          flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def sweep_eta(args, eta, chance):
    print(f"\n=== resnet50 clipart eta={pc.format_p(eta)} ===", flush=True)
    grid = sorted(set([q for q in pc.scaled_p_grid(eta, base=args.p_grid) + UPWARD_P
                       if 0.0 <= q <= 1.0]))
    points = {}
    for p in grid:
        data = ensure_run(args, method="heat", p=p, eta=eta, log="  ")
        if data is None:
            points[p] = {"p": p, "collapsed": False, "valid": False}
            continue
        v = pc.hard_collapse(data, chance)
        points[p] = {"p": p, "collapsed": v["collapsed"], "valid": True,
                     "criterion": v["criterion"], "mean_acc": v["mean_acc"]}
        print(f"    p={pc.format_p(p):>7s} acc={_fmt(v['mean_acc'])} "
              f"{'HARD-COLLAPSE' if v['collapsed'] else 'stable'} ({v['criterion']})",
              flush=True)
    # Bisect the hard boundary.
    for i in range(max(0, args.bisect_steps)):
        sel = pc.choose_pstar(list(points.values()))
        lo, hi = sel["bracket_low"], sel["bracket_high"]
        if lo is None or hi is None or (hi - lo) <= 1e-6:
            break
        mid = round((lo + hi) / 2.0, 6)
        if mid in points:
            break
        data = ensure_run(args, method="heat", p=mid, eta=eta, log="  ")
        if data is None:
            points[mid] = {"p": mid, "collapsed": False, "valid": False}
            break
        v = pc.hard_collapse(data, chance)
        points[mid] = {"p": mid, "collapsed": v["collapsed"], "valid": True,
                       "criterion": v["criterion"], "mean_acc": v["mean_acc"]}
        print(f"    bisect[{i+1}] p={pc.format_p(mid):>7s} -> "
              f"{'HARD-COLLAPSE' if v['collapsed'] else 'stable'}", flush=True)
    sel = pc.choose_pstar(list(points.values()))
    print(f"  => p*(hard)~={_fmt(sel['p_star'])} "
          f"bracket=[{_fmt(sel['bracket_low'])},{_fmt(sel['bracket_high'])}]", flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    etas = sorted(set(args.etas), reverse=True)  # DESCENDING
    print(f"[dn-sweep] resnet50 source=real target={args.target_domain} seed={args.seed}")
    print(f"[dn-sweep] etas(desc)={[pc.format_p(e) for e in etas]} chance_acc={args.chance_acc} "
          f"tf32_mode={args.tf32_mode}"
          + (" (== CIFAR/run_tier2 defaults; comparable)" if args.tf32_mode == "inherit"
             else (" (force FP32; most reproducible)" if args.tf32_mode == "off"
                   else " (force TF32; fast A100 but differs from CIFAR)")))

    # Baselines (eta-independent): run once.
    print("\n[dn-sweep] baselines (source, bn_adapt) ...", flush=True)
    src = ensure_run(args, method="source", log="  ")
    if src is not None:
        print(f"  source_acc={_fmt(pc.source_mean_acc(src))}", flush=True)
    ensure_run(args, method="bn_adapt", log="  ")

    t0 = time.time()
    for eta in etas:
        sweep_eta(args, eta, args.chance_acc)
    print(f"\n[dn-sweep] done in {time.time()-t0:.0f}s. Run analyze_domainnet_clipart.py.",
          flush=True)


if __name__ == "__main__":
    main()
