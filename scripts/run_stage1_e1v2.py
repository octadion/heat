"""
STAGE 1 RESTART (v2) — E1 ||g_bar||-factor campaign on the LOCKED protocol.
heat==TFF. Additive; run_tier2/heat.py/frozen math untouched.

Protocol (STEP 1, recorded in every tag + fingerprint):
  wrn28_10, CIFAR-10-C sev5, seed 42, batch 64;
  stream = ONE corruption's severity-5 split CYCLED x15 (~2355 steps), the
  continual horizon at which genuine hard collapse occurs (validated:
  gaussian x15 p=0 eta=1e-3 -> first NaN = 601).
  Cells: {gaussian_noise, elastic_transform} (CALIBRATION — only cells ever
  used to fit R) + {impulse_noise, contrast} (HELD-OUT) x etas {5e-4,1e-3,2e-3}.
  Per cell: source (1 cycle; no adaptation => cycle-invariant) -> eta-scaled
  p_ref ladder -> eta-scaled coarse p-grid -> upward extension -> bisection.
  HARD criterion primary (nan_inf / chance); soft recorded alongside.

Provenance guards (STEP 2):
  (a) RESULTS_DIR must be Drive-backed (or --allow-ephemeral), with a sentinel
      write+read check BEFORE any run;
  (b) every run JSON gets an embedded protocol_fingerprint (incl. checkpoint
      sha256) injected atomically after it lands;
  (c) end of campaign: Drive-side file count vs expected, gaps flagged.
Checkpoint verified FIRST: the continual sanity gate must exit 0 (reuses the
continual runs on Drive) before any campaign run; --skip-gate to bypass only
if already validated in this session.

STEP 0 (recover-or-declare) runs automatically at startup (read-only scan).

Usage (Colab):
  python scripts/run_stage1_e1v2.py \
      --results-dir /content/drive/MyDrive/pstar_results \
      --ckpt-wrn /content/drive/MyDrive/heat/experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pstar_common as pc
from scripts import stage1v2_common as v2
from scripts.run_stage1_e1 import Manifest
from scripts.stage1b_common import store_point, boundary_from_points

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TIER2 = REPO_ROOT / "scripts" / "run_tier2.py"


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 RESTART E1 campaign "
                                            "(cycled x15, hard-primary)")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--ckpt-wrn", type=str, required=True)
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--arch", type=str, default=v2.V2_ARCH, choices=["wrn28_10"])
    p.add_argument("--severity", type=int, default=v2.V2_SEVERITY)
    p.add_argument("--cycles", type=int, default=v2.CYCLES,
                   help="Cycle count (LOCKED protocol default 15; recorded in "
                        "tags + fingerprints, never hidden).")
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=list(v2.V2_CORRUPTIONS))
    p.add_argument("--etas", type=float, nargs="+", default=list(v2.V2_ETAS))
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=list(pc.DEFAULT_P_GRID))
    p.add_argument("--bisect-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--allow-ephemeral", action="store_true",
                   help="Provenance guard override: permit a non-Drive "
                        "RESULTS_DIR (recorded; use only deliberately).")
    p.add_argument("--skip-gate", action="store_true",
                   help="Skip the continual sanity gate (only if it already "
                        "passed in this session).")
    return p.parse_args()


def _f(v):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


EXPECTED_FILES: set[str] = set()


def ensure_run(args, manifest, ckpt_hash, corruption, *, eta, p=None,
               source=False, log_prefix="") -> dict | None:
    out_path = v2.run_output_path_v2(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, p=p,
                                     source=source, eta=eta,
                                     cycles=args.cycles)
    EXPECTED_FILES.add(out_path.name)
    n_cyc = 1 if source else args.cycles
    if out_path.exists():
        try:
            v2.inject_fingerprint(out_path, corruption=corruption,
                                  cycle_count=n_cyc, eta=eta,
                                  p=(None if source else p),
                                  checkpoint_hash=ckpt_hash, seed=args.seed,
                                  batch_size=args.batch_size)
            data = pc.load_json(out_path)
            manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                         p=(None if source else p), source=source,
                         cycles=n_cyc, status="reused")
            return data
        except Exception as exc:
            print(f"{log_prefix}[warn] unreadable existing JSON, re-running: "
                  f"{out_path.name} ({exc})", flush=True)

    label = "source" if source else f"p={pc.format_p(p)}"
    cmd = v2.build_run_command_v2(
        run_tier2=RUN_TIER2, arch=args.arch, checkpoint=args.ckpt_wrn,
        corruption=corruption, severity=args.severity, seed=args.seed,
        results_dir=Path(args.results_dir), c10c_root=args.c10c_root,
        heat_lr=eta, batch_size=args.batch_size,
        num_workers=args.num_workers, p=p, source=source, cycles=args.cycles)
    t0 = time.time()
    ok, tail = pc.execute_run(cmd, log_prefix=log_prefix)
    dt = time.time() - t0
    if not ok or not out_path.exists():
        err = out_path.with_suffix(".run_error.txt")
        try:
            err.write_text(f"run_error {corruption} cyc{n_cyc} "
                           f"lr{pc.format_p(eta)} {label}\n{tail}",
                           encoding="utf-8")
        except Exception:
            pass
        manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                     p=(None if source else p), source=source, cycles=n_cyc,
                     status="run_error", elapsed_s=round(dt, 1))
        print(f"{log_prefix}[run_error] {corruption} lr{pc.format_p(eta)} "
              f"{label} ({dt:.1f}s) -> {err.name}, continuing.", flush=True)
        return None
    fp = v2.inject_fingerprint(out_path, corruption=corruption,
                               cycle_count=n_cyc, eta=eta,
                               p=(None if source else p),
                               checkpoint_hash=ckpt_hash, seed=args.seed,
                               batch_size=args.batch_size)
    manifest.log(file=out_path.name, corruption=corruption, eta=eta,
                 p=(None if source else p), source=source, cycles=n_cyc,
                 status="ok", elapsed_s=round(dt, 1),
                 n_steps=fp.get("n_steps"))
    print(f"{log_prefix}[ok] {corruption} cyc{n_cyc} lr{pc.format_p(eta)} "
          f"{label} ({dt:.1f}s, n_steps={fp.get('n_steps')}) -> "
          f"{out_path.name}", flush=True)
    try:
        return pc.load_json(out_path)
    except Exception:
        return None


def select_pref(args, manifest, ckpt_hash, corruption, eta, source_acc,
                log_prefix=""):
    """Stable reference tether (eta-scaled ladder; continual reference policy:
    hard + below-source stability, no drift-ratio)."""
    last = (None, None, None)
    for cand in pc.scaled_pref_ladder(args.arch, eta):
        data = ensure_run(args, manifest, ckpt_hash, corruption, eta=eta,
                          p=cand, log_prefix=log_prefix)
        if data is None:
            continue
        g = v2.grad_norm_gbar_cycled(data)
        gbar = g[0] if g else None
        drift = v2.drift_stationary_cycled(data)
        last = (cand, drift, gbar)
        verd = pc.classify_run(data, source_acc, pref_drift=None,
                               use_drift_criterion=False)
        if not verd["collapsed"]:
            print(f"{log_prefix}  p_ref={pc.format_p(cand)} STABLE "
                  f"||g_bar||={_f(gbar)} (cycled window)", flush=True)
            return cand, drift, gbar
        print(f"{log_prefix}  p_ref={pc.format_p(cand)} collapsed "
              f"({verd['criterion']}); escalating.", flush=True)
    print(f"{log_prefix}  [warn] no stable p_ref; using last rung.", flush=True)
    return last


def sweep_cell(args, manifest, ckpt_hash, corruption, eta):
    role = ("CALIBRATION" if corruption in v2.CALIBRATION_CORRUPTIONS
            else "HELD-OUT")
    print(f"\n=== {args.arch} {corruption} ({role}) sev{args.severity} "
          f"cyc{args.cycles} eta={pc.format_p(eta)} ===", flush=True)
    src = ensure_run(args, manifest, ckpt_hash, corruption, eta=eta,
                     source=True, log_prefix="  ")
    source_acc = pc.source_mean_acc(src) if src is not None else None
    print(f"  source_acc={_f(source_acc)}", flush=True)

    p_ref, _drift, gbar = select_pref(args, manifest, ckpt_hash, corruption,
                                      eta, source_acc, log_prefix="  ")

    points: dict[float, dict] = {}

    def run_and_store(p, label=""):
        data = ensure_run(args, manifest, ckpt_hash, corruption, eta=eta,
                          p=p, log_prefix="  ")
        rec = store_point(points, p, data, source_acc)  # hard primary + soft recorded
        if rec["valid"]:
            print(f"  {label}p={pc.format_p(p):>9s} -> "
                  f"{'HARD-COLLAPSE' if rec['collapsed'] else 'stable':13s} "
                  f"({rec['criterion']}; soft: {rec['soft_criterion']}) "
                  f"acc={_f(rec['mean_acc'])}", flush=True)
        return rec

    for p in sorted(set(pc.scaled_p_grid(eta, base=args.p_grid))):
        if p in points:
            continue
        run_and_store(p)
    for p in v2.discover_p_values_v2(args.results_dir, args.arch, corruption,
                                     args.severity, args.seed, eta,
                                     cycles=args.cycles):
        if p in points and points[p].get("valid"):
            continue
        data = v2.load_run_v2(args.results_dir, args.arch, corruption,
                              args.severity, args.seed, p=p, eta=eta,
                              cycles=args.cycles)
        if data is not None:
            store_point(points, p, data, source_acc)

    if boundary_from_points(points)["bracket_high"] is None:
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
        sel = boundary_from_points(points)
        lo, hi = sel["bracket_low"], sel["bracket_high"]
        if lo is None or hi is None or (hi - lo) <= 1e-6:
            break
        mid = round((lo + hi) / 2.0, 6)
        if mid in points:
            break
        run_and_store(mid, label=f"bisect[{i+1}] ")

    final = boundary_from_points(points)
    final_soft = boundary_from_points(points, use_soft=True)
    print(f"  => p*_hard~={_f(final['p_star'])} "
          f"bracket=[{_f(final['bracket_low'])},{_f(final['bracket_high'])}] "
          f"monotone={final['monotone']} | p*_soft~={_f(final_soft['p_star'])} "
          f"| ||g_bar||={_f(gbar)} eta*||g_bar||="
          f"{_f((gbar or 0) * eta) if gbar is not None else 'n/a'}", flush=True)
    return {"corruption": corruption, "role": role, "eta": eta,
            "p_star_hard": final["p_star"], "p_star_soft": final_soft["p_star"],
            "gbar": gbar, "p_ref": p_ref}


def run_gate(args) -> None:
    print("[gate] verifying checkpoint against the continual anchor "
          "(reuses the continual runs on Drive)...", flush=True)
    rc = subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts" / "run_pstar_sweep.py"),
        "--sanity-check-only", "--results-dir", str(args.results_dir),
        "--ckpt-wrn", args.ckpt_wrn, "--c10c-root", args.c10c_root,
        "--seed", str(args.seed), "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers)]).returncode
    if rc != 0:
        raise SystemExit("[ABORT] sanity gate FAILED — the checkpoint does "
                         "not reproduce the continual anchor. Fix the "
                         "checkpoint before any campaign run.")
    print("[gate] PASSED — checkpoint reproduces the continual anchor.",
          flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()

    # STEP 2(a): Drive guard + sentinel BEFORE anything.
    v2.require_drive_dir(args.results_dir, allow_ephemeral=args.allow_ephemeral)
    if not Path(args.ckpt_wrn).exists():
        raise SystemExit(f"[fatal] WRN checkpoint not found: {args.ckpt_wrn}")
    ckpt_hash = v2.file_sha256(args.ckpt_wrn)
    print(f"[ckpt] {args.ckpt_wrn} sha256[:16]={ckpt_hash}", flush=True)

    # STEP 0: recover-or-declare (read-only, bounded effort).
    v2.recover_scan(args.results_dir)

    # STEP 3 pre-flight: checkpoint gate.
    if args.skip_gate:
        print("[gate] SKIPPED by flag (must have passed this session).",
              flush=True)
    else:
        run_gate(args)

    manifest = Manifest(Path(args.results_dir), args,
                        campaign="stage1_e1v2_cyc%d" % args.cycles,
                        filename="stage1_e1v2_manifest.jsonl")
    manifest.log(event="campaign_start", cycles=args.cycles,
                 checkpoint_hash=ckpt_hash,
                 allow_ephemeral=args.allow_ephemeral)
    print(f"[e1v2] env={manifest.env}", flush=True)
    print(f"[e1v2] protocol: cycled x{args.cycles} single-corruption streams; "
          f"HARD criterion primary; cells="
          f"{len(args.corruptions) * len(args.etas)}", flush=True)

    t0 = time.time()
    summaries = []
    for corruption in args.corruptions:  # calibration corruptions first
        for eta in sorted(set(args.etas)):
            summaries.append(sweep_cell(args, manifest, ckpt_hash,
                                        corruption, eta))

    # STEP 2(c): Drive-side count check.
    report = v2.drive_count_check(args.results_dir, EXPECTED_FILES)
    manifest.log(event="campaign_end", elapsed_s=round(time.time() - t0, 1),
                 **{k: v for k, v in report.items() if k != "missing"},
                 n_missing=len(report["missing"]))

    print(f"\n[e1v2] done in {(time.time() - t0)/3600:.2f} h. Cell summary:",
          flush=True)
    for s in summaries:
        print(f"  {s['corruption']:18s} ({s['role']:11s}) "
              f"lr{pc.format_p(s['eta']):>7s}  p*_hard~={_f(s['p_star_hard'])}"
              f"  p*_soft~={_f(s['p_star_soft'])}  ||g_bar||={_f(s['gbar'])}",
              flush=True)
    print("\nRun scripts/analyze_stage1_v2.py for E1-E5 + the verdict.",
          flush=True)


if __name__ == "__main__":
    main()
