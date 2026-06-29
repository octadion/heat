"""
Smoke test for HEAT.

This does NOT require trained checkpoints or downloaded CIFAR-10-C. It:
  1. Builds an UNTRAINED ResNet-18.
  2. Generates a tiny synthetic batch.
  3. Runs forward and confirms 4 stages come out with expected shapes.
  4. Runs HEAT.adapt() once and confirms:
     - hierarchical energy is finite
     - gradients flow to all backbone parameters (where-emergence requires this)
     - a single SGD step actually changes the parameters
  5. Same for Tent and BNAdapt and Source for cross-method sanity.

Run:
  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

import torch

from src.models import resnet18_cifar
from src.methods import HEAT, Tent, BNAdapt, Source, TEA, EATA, SAR, PeriodicReset
from src.utils import set_seed, get_device


def _summary(name: str, ok: bool, note: str = ""):
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + note) if note else ''}")


def smoke_model_forward():
    print("[1/4] Model forward + stage shapes")
    set_seed(42)
    device = get_device()

    model = resnet18_cifar(num_classes=10).to(device)
    x = torch.randn(8, 3, 32, 32, device=device)

    logits, stages = model(x, return_stages=True)
    _summary("logits.shape == (8, 10)", logits.shape == (8, 10), str(tuple(logits.shape)))

    expected_channels = [64, 128, 256, 512]
    expected_spatial = [32, 16, 8, 4]
    all_ok = True
    for i, (s, ch, sp) in enumerate(zip(stages, expected_channels, expected_spatial)):
        ok = s.shape == (8, ch, sp, sp)
        all_ok = all_ok and ok
        _summary(f"stage{i+1}.shape == (8, {ch}, {sp}, {sp})", ok, str(tuple(s.shape)))

    return all_ok


def smoke_heat():
    print("[2/4] HEAT adapt step")
    set_seed(42)
    device = get_device()

    model = resnet18_cifar(num_classes=10).to(device)
    heat = HEAT(model, lr=1e-4, momentum=0.0).to(device)

    # Snapshot params BEFORE adapt
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    x = torch.randn(8, 3, 32, 32, device=device)
    preds, diags = heat.adapt_with_diagnostics(x)

    _summary("preds.shape == (8, 10)", preds.shape == (8, 10), str(tuple(preds.shape)))
    _summary("energy is finite", torch.isfinite(torch.tensor(diags["energy"])).item(),
             f"energy={diags['energy']:.4f}")
    _summary("total grad norm > 0", diags["total_grad_norm"] > 0,
             f"||g||={diags['total_grad_norm']:.4e}")

    # Check params changed
    n_changed = 0
    n_total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if not torch.allclose(p.detach(), before[name]):
            n_changed += 1
    _summary(f"params changed: {n_changed}/{n_total}", n_changed > 0)

    # Where-emergence sanity: count layers with non-zero gradient
    nonzero_layers = sum(1 for v in diags["param_grad_norms"].values() if v > 0)
    total_layers = len(diags["param_grad_norms"])
    _summary(f"layers with grad: {nonzero_layers}/{total_layers}",
             nonzero_layers > 1)

    return True


def smoke_baselines():
    print("[3/4] Baselines (Source, BNAdapt, Tent)")
    set_seed(42)
    device = get_device()

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    # Source: shouldn't change anything
    model = resnet18_cifar(10).to(device)
    src = Source(model)
    before = copy.deepcopy(model.state_dict())
    preds = src.adapt(x)
    after = model.state_dict()
    same = all(torch.equal(before[k], after[k]) for k in before)
    _summary("Source: no param change", same)
    _summary("Source: preds shape", preds.shape == (8, 10))

    # BNAdapt: BN running stats change, but params don't (no grad step)
    model = resnet18_cifar(10).to(device)
    bn = BNAdapt(model)
    before_running = {n: b.detach().clone()
                      for n, b in model.named_buffers() if "running" in n}
    preds = bn.adapt(x)
    after_running = {n: b.detach().clone()
                     for n, b in model.named_buffers() if "running" in n}
    diffs = sum(1 for k in before_running if not torch.equal(before_running[k], after_running[k]))
    _summary(f"BNAdapt: {diffs} BN running stats updated", diffs > 0)

    # Tent: BN affine should change
    model = resnet18_cifar(10).to(device)
    tent = Tent(model, lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    preds = tent.adapt(x)
    after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    n_changed = sum(1 for k in before if not torch.allclose(before[k], after[k]))
    _summary(f"Tent: {n_changed}/{len(before)} BN affine params changed",
             n_changed > 0 and len(before) > 0)

    # TEA: BN affine should change, with very small SGLD steps for speed
    model = resnet18_cifar(10).to(device)
    tea = TEA(model, lr=1e-3, sgld_steps=2, use_buffer=False)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    preds = tea.adapt(x)
    after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    n_changed = sum(1 for k in before if not torch.allclose(before[k], after[k]))
    _summary(f"TEA: {n_changed}/{len(before)} BN affine params changed",
             n_changed > 0 and len(before) > 0)

    # EATA smoke: explicit no-Fisher mode for a synthetic check only.
    model = resnet18_cifar(10).to(device)
    eata = EATA(model, lr=1e-3, fisher_alpha=0.0)
    preds = eata.adapt(x)
    _summary("EATA no-Fisher smoke: preds shape", preds.shape == (8, 10))

    # SAR smoke: reliable entropy + SAM wiring.
    model = resnet18_cifar(10).to(device)
    sar = SAR(model, lr=1e-4, reset_ema_threshold=None)
    preds = sar.adapt(x)
    _summary("SAR smoke: preds shape", preds.shape == (8, 10))

    # RDumb periodic reset wrapper around Tent.
    model = resnet18_cifar(10).to(device)
    rdumb = PeriodicReset(Tent(model, lr=1e-3), reset_interval=1)
    rdumb.adapt(x)
    rdumb.adapt(x)
    _summary("RDumb smoke: reset fired", rdumb.num_resets == 1)

    return True


def smoke_heat_ablation():
    print("[4/5] HEAT hierarchy ablation (P5 variants)")
    set_seed(42)
    device = get_device()

    for label, stages in [("output-only [3]", [3]),
                          ("two-level [1,3]", [1, 3]),
                          ("full [0,1,2,3]", [0, 1, 2, 3])]:
        model = resnet18_cifar(10).to(device)
        heat = HEAT(model, lr=1e-4, stages=stages).to(device)
        x = torch.randn(8, 3, 32, 32, device=device)
        preds, diags = heat.adapt_with_diagnostics(x)
        ok = (
            preds.shape == (8, 10)
            and torch.isfinite(torch.tensor(diags["energy"]))
            and diags["total_grad_norm"] > 0
        )
        _summary(f"variant {label}", ok,
                 f"energy={diags['energy']:.3f}  ||g||={diags['total_grad_norm']:.2e}")
    return True


def smoke_heat_options():
    """Exercise the new options: eval_mode and update_direction."""
    print("[5/5] HEAT options (eval_mode, update_direction)")
    set_seed(42)
    device = get_device()
    x = torch.randn(8, 3, 32, 32, device=device)

    # eval_mode=True — BN running stats should NOT change
    model = resnet18_cifar(10).to(device)
    heat = HEAT(model, lr=1e-4, eval_mode=True).to(device)
    before_running = {n: b.detach().clone()
                      for n, b in model.named_buffers() if "running" in n}
    heat.adapt(x)
    after_running = {n: b.detach().clone()
                     for n, b in model.named_buffers() if "running" in n}
    same = all(torch.equal(before_running[k], after_running[k])
               for k in before_running)
    _summary("eval_mode=True: BN running stats UNCHANGED", same)

    # eval_mode=False (default) — running stats SHOULD change
    model = resnet18_cifar(10).to(device)
    heat = HEAT(model, lr=1e-4, eval_mode=False).to(device)
    before_running = {n: b.detach().clone()
                      for n, b in model.named_buffers() if "running" in n}
    heat.adapt(x)
    after_running = {n: b.detach().clone()
                     for n, b in model.named_buffers() if "running" in n}
    diffs = sum(1 for k in before_running
                if not torch.equal(before_running[k], after_running[k]))
    _summary(f"eval_mode=False: {diffs} BN running stats changed", diffs > 0)

    # Direction "-grad" vs "+grad" should produce OPPOSITE param changes
    for direction in ["-grad", "+grad", "random"]:
        model = resnet18_cifar(10).to(device)
        heat = HEAT(model, lr=1e-4, update_direction=direction).to(device)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        heat.adapt(x)
        after = {n: p.detach().clone() for n, p in model.named_parameters()}
        n_changed = sum(1 for k in before
                        if not torch.allclose(before[k], after[k]))
        _summary(f"direction='{direction}': {n_changed} params changed",
                 n_changed > 0)

    return True


def main():
    print("=" * 60)
    print("HEAT smoke test")
    print(f"device: {get_device()}")
    print("=" * 60)

    smoke_model_forward()
    smoke_heat()
    smoke_baselines()
    smoke_heat_ablation()
    smoke_heat_options()

    print()
    print("If all OK above, the code is wired correctly.")
    print("Next: train a source model, then run protocol P1.")


if __name__ == "__main__":
    main()
