"""
Tier-2 protocols (P6, P7, P8, P9, P10) — v1.5 with calibration + memory metrics.

P9 (continual) now logs ECE per-corruption AND forgetting metric (peak − last).
Other protocols (P6, P7, P8, P10) also log ECE for completeness but main use
case is P9 for continual stability framing.

NEW (vit/cifar100 patch):
  --arch vit_s — works with P9; other Tier-2 protocols are not in scope
                 for ViT in this patch but will still run if the model loads.
  --dataset {cifar10, cifar100}.
  Graceful skip for bn_adapt on non-BN architectures.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import (
    CORRUPTIONS,
    get_corruption_loader,
    get_clean_loaders,
    num_classes_for,
)
from src.models import build_arch
from src.methods import HEAT, NoBatchNormError
from src.adapt import evaluate_online
from src.utils import set_seed, get_device
from src.utils.method_factory import build_method, add_method_args


P9_DEFAULT_STREAM = list(CORRUPTIONS)
P10_DIRECTIONS = ["-grad", "+grad", "random"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10", "vit_s"])
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--protocol", type=str, required=True,
                   choices=["p6", "p7", "p8", "p9", "p10"])
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--c100c-root", type=str, default="data/cifar100c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--cifar100-root", type=str, default="data/cifar100")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["heat", "tent", "tea"])
    p.add_argument("--corruptions", type=str, nargs="+", default=None)
    p.add_argument("--out-dir", type=str, default="experiments/results")
    p.add_argument("--variant-tag", type=str, default="")
    add_method_args(p)
    return p.parse_args()


def _corruption_root(args) -> str:
    return args.c10c_root if args.dataset == "cifar10" else args.c100c_root


def _clean_root(args) -> str:
    return args.cifar10_root if args.dataset == "cifar10" else args.cifar100_root


def load_model(arch, checkpoint_path, device, num_classes):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extra = {"pretrained": False} if arch == "vit_s" else {}
    model = build_arch(arch, num_classes=num_classes, **extra).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _result_to_dict(r):
    return {
        "accuracy": r.accuracy,
        "ece": r.ece,
        "mean_confidence": r.mean_confidence,
        "mean_entropy": r.mean_entropy,
    }


def _safe_build(m_name, base_model, device, args):
    """Wrap build_method with NoBatchNormError handling for graceful skip."""
    try:
        return build_method(m_name, base_model, device, args,
                            dataset_root=_clean_root(args))
    except NoBatchNormError as e:
        print(f"  {m_name:10s}  SKIPPED: {e}")
        return None


def run_p6(args, base_model, device):
    corruptions = args.corruptions or ["gaussian_noise", "defocus_blur"]
    severities = [1, 2, 3, 4, 5]
    results = {m: {} for m in args.methods}
    for c in corruptions:
        for sev in severities:
            print(f"\n=== {c} severity={sev} ===")
            loader = get_corruption_loader(
                args.dataset, _corruption_root(args), c, severity=sev,
                batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=False, arch=args.arch,
            )
            for m_name in args.methods:
                set_seed(args.seed)
                method = _safe_build(m_name, base_model, device, args)
                if method is None:
                    results[m_name]["_skipped"] = True
                    continue
                r = evaluate_online(method, loader, device, progress=False)
                results[m_name][f"{c}_sev{sev}"] = _result_to_dict(r)
                print(f"  {m_name:10s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}")
    return results


def run_p7(args, base_model, device):
    _, test_loader = get_clean_loaders(
        args.dataset, _clean_root(args),
        batch_size=args.batch_size, num_workers=args.num_workers,
        arch=args.arch,
    )
    results = {}
    print(f"\n=== clean {args.dataset} ===")
    for m_name in args.methods:
        set_seed(args.seed)
        method = _safe_build(m_name, base_model, device, args)
        if method is None:
            results[m_name] = {"_skipped": True}
            continue
        r = evaluate_online(method, test_loader, device, progress=False)
        results[m_name] = {"clean": _result_to_dict(r)}
        print(f"  {m_name:10s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}  "
              f"conf={r.mean_confidence:.3f}")
    return results


def run_p8(args, base_model, device):
    _, clean_loader = get_clean_loaders(
        args.dataset, _clean_root(args),
        batch_size=args.batch_size, num_workers=args.num_workers,
        arch=args.arch,
    )
    corruption = (args.corruptions or ["gaussian_noise"])[0]
    corr_loader = get_corruption_loader(
        args.dataset, _corruption_root(args), corruption,
        severity=args.severity, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, arch=args.arch,
    )
    results = {m: {} for m in args.methods}
    for m_name in args.methods:
        set_seed(args.seed)
        method = _safe_build(m_name, base_model, device, args)
        if method is None:
            results[m_name] = {"_skipped": True}
            continue
        print(f"\n=== {m_name} ===")
        r1 = evaluate_online(method, clean_loader, device, progress=False)
        print(f"  stage 1 (clean):       acc={r1.accuracy:.4f}  ece={r1.ece:.4f}")
        r2 = evaluate_online(method, corr_loader, device, progress=False)
        print(f"  stage 2 ({corruption}): acc={r2.accuracy:.4f}  ece={r2.ece:.4f}")
        r3 = evaluate_online(method, clean_loader, device, progress=False)
        print(f"  stage 3 (clean):       acc={r3.accuracy:.4f}  ece={r3.ece:.4f}")
        print(f"  recovery: {r3.accuracy - r1.accuracy:+.4f}")
        results[m_name] = {
            "clean_initial": _result_to_dict(r1),
            f"{corruption}_corrupt": _result_to_dict(r2),
            "clean_recovered": _result_to_dict(r3),
            "recovery_delta": r3.accuracy - r1.accuracy,
        }
    return results


def run_p9(args, base_model, device):
    """Continual stream — extends with ECE + drift/grad diagnostics for HEAT."""
    corruptions = args.corruptions or P9_DEFAULT_STREAM
    results = {m: {} for m in args.methods}
    summary = {}

    for m_name in args.methods:
        print(f"\n=== {m_name} (continual stream of {len(corruptions)}) ===")
        set_seed(args.seed)
        method = _safe_build(m_name, base_model, device, args)

        if method is None:
            results[m_name] = {"_skipped": True}
            summary[m_name] = {"skipped": True}
            continue

        if m_name == "heat":
            print(
                f"  [config] lr={args.heat_lr}, "
                f"restore_prob={args.heat_restore_prob}, "
                f"diagnostic_snapshot={getattr(args, 'heat_diagnostic_snapshot', False)}"
            )

        per_corruption = {}
        stream_diagnostics = []
        global_step = 0

        for c in corruptions:
            loader = get_corruption_loader(
                args.dataset,
                _corruption_root(args),
                c,
                severity=args.severity,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                arch=args.arch,
            )

            # IMPORTANT:
            # collect_diagnostics=True makes evaluate_online call
            # method.adapt_with_diagnostics(x), not method.adapt(x).
            collect_diag = (m_name == "heat")

            r = evaluate_online(
                method,
                loader,
                device,
                collect_diagnostics=collect_diag,
                progress=False,
            )

            result_dict = _result_to_dict(r)

            # Save per-step HEAT diagnostics.
            # Each corruption has its own local step, but global_step continues
            # across the whole continual stream.
            if collect_diag:
                diagnostics = []
                for local_step, d in enumerate(r.diagnostics):
                    row = {
                        "step": global_step + local_step,
                        "local_step": local_step,
                        "corruption": c,
                        "restore_prob": d.get("restore_prob"),
                        "drift_l2": d.get("drift_l2"),
                        "grad_l2": d.get("grad_l2"),
                        "total_grad_norm": d.get("total_grad_norm"),
                        "energy": d.get("energy"),
                    }
                    diagnostics.append(row)
                    stream_diagnostics.append(row)

                result_dict["diagnostics"] = diagnostics
                global_step += len(r.diagnostics)

            per_corruption[c] = result_dict

            if collect_diag and r.diagnostics:
                last_diag = r.diagnostics[-1]
                print(
                    f"  {c:20s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}  "
                    f"drift={last_diag.get('drift_l2', 0.0):.3f}  "
                    f"grad={last_diag.get('grad_l2', 0.0):.3f}"
                )
            else:
                print(f"  {c:20s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}")

        results[m_name] = per_corruption

        accs = [per_corruption[c]["accuracy"] for c in corruptions]
        eces = [per_corruption[c]["ece"] for c in corruptions]

        peak_acc = max(accs)
        peak_idx = accs.index(peak_acc)
        last_acc = accs[-1]
        forgetting = peak_acc - last_acc
        mean_acc = sum(accs) / len(accs)
        mean_ece = sum(eces) / len(eces)

        summary[m_name] = {
            "mean_accuracy": mean_acc,
            "last_accuracy": last_acc,
            "peak_accuracy": peak_acc,
            "peak_at_corruption": corruptions[peak_idx],
            "forgetting": forgetting,
            "mean_ece": mean_ece,
        }

        if m_name == "heat":
            summary[m_name]["num_diagnostic_steps"] = len(stream_diagnostics)
            summary[m_name]["stream_diagnostics"] = stream_diagnostics

        print(
            f"  → mean={mean_acc:.4f}  last={last_acc:.4f}  "
            f"peak={peak_acc:.4f}  forgetting={forgetting:.4f}  "
            f"mean_ece={mean_ece:.4f}"
        )

        if m_name == "heat":
            print(f"  → diagnostic_steps={len(stream_diagnostics)}")

    return {"per_corruption": results, "summary": summary}


def run_p10(args, base_model, device):
    corruptions = args.corruptions or ["gaussian_noise", "defocus_blur",
                                       "snow", "contrast"]
    results = {d: {} for d in P10_DIRECTIONS}
    for direction in P10_DIRECTIONS:
        print(f"\n=== direction: {direction} ===")
        for c in corruptions:
            loader = get_corruption_loader(
                args.dataset, _corruption_root(args), c,
                severity=args.severity, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=False, arch=args.arch,
            )
            set_seed(args.seed)
            m = copy.deepcopy(base_model).to(device)
            heat = HEAT(
                m, lr=args.heat_lr, momentum=args.heat_momentum,
                temperatures=args.heat_temperatures,
                aggregation=args.heat_aggregation,
                eval_mode=args.heat_eval_mode,
                update_direction=direction,
                restore_prob=args.heat_restore_prob,
            ).to(device)
            r = evaluate_online(heat, loader, device, progress=False)
            results[direction][c] = _result_to_dict(r)
            print(f"  {c:20s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}")
    print("\n--- P10 summary ---")
    for d in P10_DIRECTIONS:
        accs = [results[d][c]["accuracy"] for c in corruptions]
        mean = sum(accs) / len(accs)
        print(f"  {d:8s}  mean_acc={mean:.4f}")
    return results


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = num_classes_for(args.dataset)
    base_model = load_model(args.arch, args.checkpoint, device, num_classes)
    print(f"[ok] arch={args.arch}, dataset={args.dataset}, ckpt={args.checkpoint}")
    print(f"[ok] HEAT: lr={args.heat_lr}, temps={args.heat_temperatures}, "
          f"agg={args.heat_aggregation}, restore_prob={args.heat_restore_prob}")

    if args.protocol == "p6":
        results = run_p6(args, base_model, device)
    elif args.protocol == "p7":
        results = run_p7(args, base_model, device)
    elif args.protocol == "p8":
        results = run_p8(args, base_model, device)
    elif args.protocol == "p9":
        results = run_p9(args, base_model, device)
    elif args.protocol == "p10":
        results = run_p10(args, base_model, device)

    tag = f"_{args.variant_tag}" if args.variant_tag else ""
    dataset_tag = f"_{args.dataset}" if args.dataset != "cifar10" else ""
    out_path = (out_dir /
                f"{args.protocol}_{args.arch}{dataset_tag}{tag}_seed{args.seed}_sev{args.severity}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
