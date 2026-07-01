"""
PART B — Is corruption-type a wide-enough ||g_bar|| lever? (cheap, ~minutes GPU)

The CIFAR eta-sweep moved the x-axis via eta only (Audit 2: ||g_bar|| CV<7%), so the
||g_bar|| FACTOR of the law was never independently tested. Before designing a
p*-vs-||g_bar|| sweep, measure whether varying the CORRUPTION TYPE (fixed severity,
NO adaptation) moves the INTRINSIC source-point ||g_bar|| enough to be a usable lever.

For each arch, at the SOURCE checkpoint theta* (eval mode, BN frozen at source stats so
nothing drifts across batches), we compute ||grad L_TFF(theta*)|| over ~n batches for
each of the 15 CIFAR-10-C corruptions at severity 5 — NO optimizer step is ever taken, so
every batch is evaluated at theta* (uncontaminated by drift). eta is irrelevant (no step).

L_TFF = the SAME hierarchical energy HEAT uses (temperatures=[1.0], aggregation='sum',
all params trainable). We only READ HEAT's energy + gradient; we never step, and we do
NOT modify heat.py / the collapse math / src/.

NAMING: code identifier `heat` == TFF in reporting.

DECISION GATE (explicit, do not proceed past it):
  range factor (max/min ||g_bar||) >= ~3x  -> GATE PASS (corruption is a viable lever)
  range factor < ~2x                        -> GATE FAIL (pick another lever)
  in between                                -> MARGINAL (treat as not-pass; decide)

Usage:
  python scripts/measure_gbar_per_corruption.py \
      --ckpt-resnet18 experiments/checkpoints/resnet18_final.pt \
      --ckpt-wrn      experiments/checkpoints/wrn28_10_final.pt \
      --c10c-root data/cifar10c --severity 5 --n-batches 8 --out-dir experiments/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import run_tier2
from src.data import CORRUPTIONS, get_corruption_loader, num_classes_for
from src.methods import HEAT
from src.utils import set_seed, get_device

GATE_PASS = 3.0
GATE_FAIL = 2.0


def grad_norm_at_source(heat: HEAT, x: torch.Tensor) -> float:
    """||grad L_TFF|| at the current params, WITHOUT stepping (theta unchanged).
    Matches HEAT's grad_l2 definition (L2 over trainable-param grads)."""
    logits, stages = heat.model(x, return_stages=True)
    P = heat.hierarchical_energy(stages)
    heat.optimizer.zero_grad(set_to_none=True)
    P.backward()
    sq = 0.0
    for p in heat.model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().norm().item()
            sq += g * g
    heat.optimizer.zero_grad(set_to_none=True)   # leave clean; never step
    return sq ** 0.5, logits.detach()


def measure_arch(arch, ckpt, device, args):
    num_classes = num_classes_for("cifar10")
    base = run_tier2.load_model(arch, ckpt, device, num_classes)
    # HEAT at theta*: eval mode (BN frozen at source stats -> no buffer drift across
    # batches), all params trainable, same L_TFF as the sweep (temps [1.0], sum).
    heat = HEAT(base, lr=1e-3, temperatures=[1.0], aggregation="sum",
                update_all_params=True, eval_mode=True).to(device)
    rows = []
    for corr in CORRUPTIONS:
        loader = get_corruption_loader("cifar10", args.c10c_root, corr,
                                       severity=args.severity, batch_size=args.batch_size,
                                       num_workers=args.num_workers, shuffle=False, arch=arch)
        set_seed(args.seed)
        gnorms, correct, total = [], 0, 0
        for i, (x, y) in enumerate(loader):
            if i >= args.n_batches:
                break
            x = x.to(device); y = y.to(device)
            gn, logits = grad_norm_at_source(heat, x)
            if math.isfinite(gn):
                gnorms.append(gn)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        gbar = statistics.mean(gnorms) if gnorms else None
        acc = correct / total if total else None
        rows.append({"corruption": corr, "source_acc": acc, "gbar_source": gbar})
    return rows


def _fmt(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def parse_args():
    p = argparse.ArgumentParser(description="Part B: ||g_bar|| per corruption at source (TFF)")
    p.add_argument("--archs", type=str, nargs="+", default=["wrn28_10", "resnet18"],
                   choices=["wrn28_10", "resnet18"])
    p.add_argument("--ckpt-resnet18", type=str, default="")
    p.add_argument("--ckpt-wrn", type=str, default="")
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = parse_args()
    device = get_device()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {"resnet18": a.ckpt_resnet18, "wrn28_10": a.ckpt_wrn}

    per_arch, gate_by_arch = {}, {}
    for arch in a.archs:
        ckpt = ckpts.get(arch, "")
        if not ckpt or not Path(ckpt).exists():
            print(f"[skip] {arch}: checkpoint missing ({ckpt or 'not provided'})")
            continue
        print(f"\n=== {arch}: ||grad L_TFF(theta*)|| per corruption (sev{a.severity}, "
              f"{a.n_batches} batches, NO update) ===")
        rows = measure_arch(arch, ckpt, device, a)
        rows_sorted = sorted(rows, key=lambda r: (r["gbar_source"] is None,
                                                  r["gbar_source"] or 0.0))
        print(f"  {'corruption':20s}  {'source_acc':>10s}  {'||g_bar||_source':>16s}")
        for r in rows_sorted:
            print(f"  {r['corruption']:20s}  {_fmt(r['source_acc']):>10s}  "
                  f"{_fmt(r['gbar_source']):>16s}")
        gbars = [r["gbar_source"] for r in rows if r["gbar_source"] is not None]
        if len(gbars) >= 2:
            gmin, gmax = min(gbars), max(gbars)
            rf = gmax / gmin if gmin > 0 else float("inf")
            cv = statistics.pstdev(gbars) / statistics.mean(gbars)
            per_arch[arch] = {"rows": rows, "min": gmin, "max": gmax,
                              "range_factor": rf, "cv": cv}
            gate_by_arch[arch] = ("PASS" if rf >= GATE_PASS
                                  else "FAIL" if rf < GATE_FAIL else "MARGINAL")
            print(f"  -> ||g_bar|| range=[{_fmt(gmin)}, {_fmt(gmax)}]  "
                  f"RANGE FACTOR={rf:.2f}x  CV={cv*100:.1f}%  gate={gate_by_arch[arch]}")

    # ---- Overall decision gate ----
    factors = {ar: per_arch[ar]["range_factor"] for ar in per_arch}
    print("\n================ DECISION GATE (corruption as ||g_bar|| lever) ================")
    for ar, rf in factors.items():
        print(f"  {ar}: range factor = {rf:.2f}x -> {gate_by_arch[ar]}")
    if factors and max(factors.values()) >= GATE_PASS:
        best = max(factors, key=factors.get)
        verdict = (f"GATE PASS: viable — corruption-type moves ||g_bar|| by "
                   f"{factors[best]:.2f}x ({best}). Recommend a p*-vs-||g_bar|| sweep at "
                   f"fixed eta across corruptions. STOP here — design that sweep next.")
    elif factors and all(rf < GATE_FAIL for rf in factors.values()):
        verdict = (f"GATE FAIL: corruption-type does NOT move ||g_bar|| enough "
                   f"(all range factors < {GATE_FAIL:.0f}x: "
                   f"{ {a: round(f,2) for a,f in factors.items()} }). Pick a different lever "
                   f"(CIFAR-100-C, batch size, or accept ||g_bar|| is tested only "
                   f"cross-architecture). Do NOT run any p* sweep.")
    elif factors:
        verdict = (f"GATE MARGINAL: range factors { {a: round(f,2) for a,f in factors.items()} } "
                   f"sit between {GATE_FAIL:.0f}x and {GATE_PASS:.0f}x — treat as NOT-PASS; do "
                   f"not run the sweep. Decide (denser corruption set, or a different lever).")
    else:
        verdict = ("GATE: no arch measured (checkpoints missing). Provide --ckpt-* and rerun.")
    print("\n" + verdict)

    (out_dir / "gbar_per_corruption.json").write_text(json.dumps(
        {"severity": a.severity, "n_batches": a.n_batches,
         "per_arch": {ar: {k: v for k, v in d.items()} for ar, d in per_arch.items()},
         "gate_by_arch": gate_by_arch, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'gbar_per_corruption.json'}")


if __name__ == "__main__":
    main()
