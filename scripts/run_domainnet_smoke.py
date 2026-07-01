"""
DomainNet-126 integration SMOKE TEST (not a full experiment).

Goal: prove the pipeline works end-to-end on ONE narrow slice
(arch=resnet50, source=real, target=clipart, eta=1e-3) before investing in the
full domain x eta sweep. Reuses the EXISTING adaptation loop + diagnostics:
it imports scripts/run_tier2.run_p9 and feeds it a hand-built args namespace +
the AdaContrast ResNet-50 wrapper, so run_tier2.py / the runner / heat.py are
all UNCHANGED (we never go through run_tier2's argparse, which would reject the
new arch/dataset). ||g_bar|| and the collapse criterion come from
scripts/pstar_common (unchanged math / 50-150 window).

A "block" of diagnostics = one target-domain pass; for the smoke the stream is
just the real-source model adapting over the clipart stream.

PASS iff all four checks are sane (printed itemized, like the WRN sanity gate):
  1. checkpoint loads into the wrapper (strict, or exact mismatched keys);
  2. source (no-adapt) clipart acc in a plausible band (~0.35-0.55);
  3. ||g_bar|| finite and in a sane magnitude (reported);
  4. the collapse criterion triggers across the coarse p-grid (>=1 stable AND
     >=1 collapse => p* is measurable here).

Usage:
  python scripts/run_domainnet_smoke.py \
      --domainnet-root data/domainnet-126 \
      --ckpt experiments/checkpoints/best_real_2020.pth.tar \
      --target clipart --heat-lr 1e-3 \
      --p-grid 0.0 0.005 0.02 0.05 --out-dir experiments/results/domainnet_smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import run_tier2
from scripts import pstar_common as pc
from src.models.resnet50_domainnet import ResNet50DomainNet, load_domainnet_checkpoint
from src.utils import get_device, set_seed


SOURCE_ACC_BAND = (0.35, 0.55)   # plausible AdaContrast real->clipart source acc
GBAR_BAND = (1e-3, 1e4)          # finite + not absurd


def parse_args():
    p = argparse.ArgumentParser(description="DomainNet-126 smoke test")
    p.add_argument("--domainnet-root", type=str, required=True,
                   help="Dir with <domain>_list.txt + image folders.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="AdaContrast source=real checkpoint (best_real_2020.pth.tar).")
    p.add_argument("--target", type=str, default="clipart")
    p.add_argument("--heat-lr", type=float, default=1e-3)
    p.add_argument("--p-grid", type=float, nargs="+", default=[0.0, 0.005, 0.02, 0.05])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="experiments/results/domainnet_smoke")
    return p.parse_args()


def _make_args(a, *, methods, restore_prob, diagnostic):
    """Build the namespace run_tier2.run_p9 / build_method / the loader read.

    Most heat_* / tea_* fields are read via getattr(..., default), so only the
    load-bearing ones are set. dataset != cifar10 makes run_tier2._corruption_root
    return c100c_root, so we point that at the DomainNet root."""
    return SimpleNamespace(
        arch="resnet50", dataset="domainnet126",
        c10c_root=a.domainnet_root, c100c_root=a.domainnet_root,
        cifar10_root=a.domainnet_root, cifar100_root=a.domainnet_root,
        severity=5,                       # ignored by the domainnet loader
        corruptions=[a.target],           # single "block" = the target domain
        batch_size=a.batch_size, num_workers=a.num_workers, seed=a.seed,
        methods=methods,
        heat_lr=a.heat_lr, heat_momentum=0.0, heat_temperatures=[1.0],
        heat_aggregation="sum", heat_eval_mode=False, heat_adapt_params="full",
        heat_bn_running_stats="train", heat_stages=None,
        heat_restore_prob=restore_prob, heat_diagnostic_snapshot=diagnostic,
    )


def _wrap(args_ns, run_p9_out):
    """Wrap run_p9 output into the run_tier2 JSON schema pstar_common reads."""
    return {"args": {"heat_lr": args_ns.heat_lr}, "results": run_p9_out}


def main():
    # run_tier2's progress prints contain non-ASCII (→); force UTF-8 stdout so a
    # cp1252 locale / piped shell can't crash the run. (Colab is already UTF-8.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[smoke] device={device} arch=resnet50 source=real target={a.target} "
          f"eta={a.heat_lr}")

    # ---- Build wrapper + load checkpoint (CHECK 1) ----
    model = ResNet50DomainNet(num_classes=126)
    ckpt_report = load_domainnet_checkpoint(model, a.ckpt, device=device, verbose=True)
    model.to(device).eval()
    # deepcopy sanity: build_method deep-copies base_model; ensure that works
    # with the shared-reference head before we rely on it.
    _ = copy.deepcopy(model)

    # ---- Source (no-adapt) over the target stream (CHECK 2) ----
    set_seed(a.seed)
    src_out = run_tier2.run_p9(_make_args(a, methods=["source"], restore_prob=0.0,
                                          diagnostic=False), model, device)
    source_acc = src_out["summary"]["source"]["mean_accuracy"]
    print(f"\n[smoke] source (no-adapt) {a.target} acc = {source_acc:.4f}")

    # ---- HEAT over the coarse p-grid (CHECK 3 + 4) ----
    grid_rows = []
    for p in sorted(set(a.p_grid)):
        set_seed(a.seed)
        out = run_tier2.run_p9(_make_args(a, methods=["heat"], restore_prob=p,
                                          diagnostic=True), model, device)
        data = _wrap(_make_args(a, methods=["heat"], restore_prob=p, diagnostic=True),
                     out)
        gbar_res = pc.grad_norm_gbar(data)
        gbar = gbar_res[0] if gbar_res else None
        drift = pc.drift_stationary(data)
        v = pc.classify_run(data, source_acc, pref_drift=None)
        mean_acc = out["summary"]["heat"]["mean_accuracy"]
        grid_rows.append({"p": p, "mean_acc": mean_acc, "gbar": gbar,
                          "drift": drift, "collapsed": v["collapsed"],
                          "criterion": v["criterion"]})
        print(f"[smoke] p={pc.format_p(p):>6s}  mean_acc={mean_acc:.4f}  "
              f"||g_bar||={'n/a' if gbar is None else f'{gbar:.4g}'}  "
              f"drift={'n/a' if drift is None else f'{drift:.4g}'}  "
              f"{'COLLAPSE' if v['collapsed'] else 'stable'} ({v['criterion']})")

    # Representative ||g_bar||: from the most-stable (largest stable p) run; else
    # the largest-p run available.
    stable = [r for r in grid_rows if not r["collapsed"] and r["gbar"] is not None]
    ref = (max(stable, key=lambda r: r["p"]) if stable
           else max([r for r in grid_rows if r["gbar"] is not None],
                    key=lambda r: r["p"], default=None))
    ref_gbar = ref["gbar"] if ref else None

    n_stable = sum(1 for r in grid_rows if not r["collapsed"])
    n_collapse = sum(1 for r in grid_rows if r["collapsed"])

    # ---- Four checks ----
    c1 = bool(ckpt_report["ok"])
    c2 = (source_acc is not None and SOURCE_ACC_BAND[0] <= source_acc <= SOURCE_ACC_BAND[1])
    c3 = (ref_gbar is not None and math.isfinite(ref_gbar)
          and GBAR_BAND[0] <= ref_gbar <= GBAR_BAND[1])
    c4 = (n_stable >= 1 and n_collapse >= 1)
    checks = {
        "1_checkpoint_loads": c1,
        "2_source_acc_in_band": c2,
        "3_gbar_finite_sane": c3,
        "4_collapse_boundary_present": c4,
    }

    print("\n================ DOMAINNET-126 SMOKE: CHECKS ================")
    print(f"  [{'PASS' if c1 else 'FAIL'}] 1. checkpoint loads "
          f"(strict={ckpt_report['strict']}, missing={len(ckpt_report['missing'])}, "
          f"unexpected={len(ckpt_report['unexpected'])})")
    print(f"  [{'PASS' if c2 else 'FAIL'}] 2. source {a.target} acc={source_acc:.4f} "
          f"in {SOURCE_ACC_BAND}")
    print(f"  [{'PASS' if c3 else 'FAIL'}] 3. ||g_bar||="
          f"{'n/a' if ref_gbar is None else f'{ref_gbar:.4g}'} finite & in {GBAR_BAND}")
    print(f"  [{'PASS' if c4 else 'FAIL'}] 4. collapse boundary present "
          f"({n_stable} stable, {n_collapse} collapse across the p-grid)")

    passed = all(checks.values())
    summary = {
        "arch": "resnet50", "source": "real", "target": a.target,
        "heat_lr": a.heat_lr, "source_acc": source_acc,
        "ref_gbar": ref_gbar, "checkpoint_report": ckpt_report,
        "grid": grid_rows, "checks": checks, "passed": passed,
    }
    (out_dir / f"domainnet_smoke_{a.target}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    if passed:
        print("\n[DOMAINNET SMOKE PASSED] Integration works end-to-end on "
              f"real->{a.target}. Safe to scale to the domain x eta sweep.")
        sys.exit(0)
    print("\n[DOMAINNET SMOKE FAILED] Fix integration before scaling. Failed: "
          f"{[k for k, ok in checks.items() if not ok]}")
    if not c4:
        print("  (Note: if all p are stable, p*~=0 here — try a low p or report "
              "that clipart doesn't destabilize this source at eta=1e-3; if all "
              "collapse, the source/transform/label mapping is likely wrong.)")
    sys.exit(1)


if __name__ == "__main__":
    main()
