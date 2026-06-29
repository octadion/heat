"""
Protocol P1 — Standard online TTA benchmark.

v1.5: logs ECE, mean_confidence, mean_entropy, peak_memory_mb, trainable_params
alongside accuracy. Backward-compatible JSON structure (new fields added; old
'results' dict still has accuracy values).

NEW (vit/cifar100 patch):
  --arch vit_s: evaluates all methods except bn_adapt on a ViT-S/16 source
    model. bn_adapt is skipped cleanly (ViT has no BatchNorm).
  --dataset {cifar10, cifar100}: chooses the corruption dataset and
    num_classes. The model checkpoint is still passed via --checkpoint, so
    the user supplies the correctly-trained one.
  --batch-size now defaults to None and is resolved per-arch (CNN: 64,
    ViT: 64 — but for ViT consider passing a smaller value because of the
    224x224 memory cost). TEA on ViT is slow (SGLD-20 x 224x224).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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
from src.methods import (
    Source, BNAdapt, Tent, TEA, TEANoNoise, TEADirectEnergy,
    HEAT, EPOTTA, ReTTA,
)
from src.methods import NoBatchNormError
from src.adapt import evaluate_online
from src.utils import set_seed, get_device
from src.utils.method_factory import build_method as build_method_from_factory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "wrn28_10", "vit_s"])
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--c10c-root", type=str, default="data/cifar10c")
    p.add_argument("--c100c-root", type=str, default="data/cifar100c")
    p.add_argument("--cifar10-root", type=str, default="data/cifar10")
    p.add_argument("--cifar100-root", type=str, default="data/cifar100")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["source", "bn_adapt", "tent", "tea", "heat",
                            "epotta", "retta"])
    p.add_argument("--corruptions", type=str, nargs="+", default=None)

    # Tent
    p.add_argument("--tent-lr", type=float, default=1e-3)
    p.add_argument("--tent-optimizer", type=str, default="adam",
                   choices=["adam", "sgd"])
    # TEA — NOTE: TEA on ViT is slow due to SGLD-20 over 224x224 images.
    p.add_argument("--tea-lr", type=float, default=5e-4)
    p.add_argument("--tea-optimizer", type=str, default="adam",
                   choices=["adam", "sgd"])
    p.add_argument("--tea-sgld-steps", type=int, default=20)
    p.add_argument("--tea-sgld-lr", type=float, default=0.1)
    p.add_argument("--tea-sgld-noise", type=float, default=0.01)
    p.add_argument("--tea-use-buffer", dest="tea_use_buffer",
                   action="store_true", default=True)
    p.add_argument("--tea-no-buffer", dest="tea_use_buffer",
                   action="store_false")
    p.add_argument("--tea-buffer-size", type=int, default=1000)
    p.add_argument("--tea-buffer-reinit-prob", type=float, default=0.05)
    p.add_argument("--tea-entropy-coef", type=float, default=1.0)
    # HEAT
    p.add_argument("--heat-lr", type=float, default=1e-3)
    p.add_argument("--heat-momentum", type=float, default=0.0)
    p.add_argument("--heat-eval-mode", action="store_true")
    p.add_argument("--heat-temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--heat-aggregation", type=str, default="sum",
                   choices=["sum", "self_gated", "self_gated_temperature"])
    p.add_argument("--heat-restore-prob", type=float, default=0.0)
    p.add_argument("--heat-adapt-params", type=str, default="full",
                   choices=["full", "bn_affine_only"])
    p.add_argument("--heat-bn-running-stats", type=str, default="train",
                   choices=["train", "frozen"])
    # Optional explicit stage list. Default (None) means "all stages" — i.e.
    # the prior HEAT behavior, bit-identical to before this patch. The
    # heat_singlestage variant defaults to the last stage when this is
    # omitted; passing it overrides that default for the variant too.
    p.add_argument("--heat-stages", type=int, nargs="+", default=None,
                   help="Explicit stage indices for HEAT / heat_singlestage. "
                        "Default = all stages.")
    # EPOTTA
    p.add_argument("--epotta-lr", type=float, default=1e-3)
    p.add_argument("--epotta-beta", type=float, default=1.0)
    p.add_argument("--epotta-buffer-size", type=int, default=500)
    # ReTTA
    p.add_argument("--retta-lr", type=float, default=1e-3)
    p.add_argument("--retta-lambda-energy", type=float, default=1.0)
    # EATA
    p.add_argument("--eata-lr", type=float, default=5e-3)
    p.add_argument("--eata-optimizer", type=str, default="sgd",
                   choices=["sgd", "adam"])
    p.add_argument("--eata-momentum", type=float, default=0.9)
    p.add_argument("--eata-e-margin", type=float, default=None)
    p.add_argument("--eata-d-margin", type=float, default=0.05)
    p.add_argument("--eata-fisher-alpha", type=float, default=2000.0)
    p.add_argument("--eata-fisher-samples", type=int, default=2000)
    p.add_argument("--eata-fisher-split", type=str, default="train",
                   choices=["train", "test"])
    # SAR
    p.add_argument("--sar-lr", type=float, default=2.5e-4)
    p.add_argument("--sar-momentum", type=float, default=0.9)
    p.add_argument("--sar-rho", type=float, default=0.05)
    p.add_argument("--sar-e-margin", type=float, default=None)
    p.add_argument("--sar-reset-ema-threshold", type=float, default=0.2)
    p.add_argument("--sar-ema-momentum", type=float, default=0.9)
    # RDumb / periodic reset
    p.add_argument("--rdumb-base-method", type=str, default="tent",
                   choices=["tent", "eata", "sar", "heat", "tea"])
    p.add_argument("--rdumb-reset-interval", type=int, default=157)
    p.add_argument("--rdumb-reset-scope", type=str, default="full_model",
                   choices=["full_model", "trainable_params"])
    p.add_argument("--rdumb-reset-optimizer-state",
                   dest="rdumb_reset_optimizer_state",
                   action="store_true", default=True)
    p.add_argument("--rdumb-keep-optimizer-state",
                   dest="rdumb_reset_optimizer_state",
                   action="store_false")
    p.add_argument("--rdumb-reset-bn-running-stats",
                   dest="rdumb_reset_bn_running_stats",
                   action="store_true", default=True)
    p.add_argument("--rdumb-keep-bn-running-stats",
                   dest="rdumb_reset_bn_running_stats",
                   action="store_false")

    p.add_argument("--out-dir", type=str, default="experiments/results")
    p.add_argument("--variant-tag", type=str, default="")
    return p.parse_args()


def _corruption_root(args) -> str:
    return args.c10c_root if args.dataset == "cifar10" else args.c100c_root


def _clean_root(args) -> str:
    return args.cifar10_root if args.dataset == "cifar10" else args.cifar100_root


def load_model(arch, checkpoint_path, device, num_classes):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # For vit_s, pretrained=False — the checkpoint provides weights.
    extra = {"pretrained": False} if arch == "vit_s" else {}
    model = build_arch(arch, num_classes=num_classes, **extra).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def build_method(name, base_model, device, args):
    return build_method_from_factory(
        name,
        base_model,
        device,
        args,
        dataset_root=_clean_root(args),
    )
    model = copy.deepcopy(base_model).to(device)
    if name == "source":
        return Source(model)
    elif name == "bn_adapt":
        # Raises NoBatchNormError for ViT — caller handles graceful skip.
        return BNAdapt(model)
    elif name == "tent":
        return Tent(model, lr=args.tent_lr,
                    optimizer_name=args.tent_optimizer, momentum=0.9)
    elif name == "tea":
        return TEA(model, lr=args.tea_lr, optimizer_name=args.tea_optimizer,
                   sgld_steps=args.tea_sgld_steps, sgld_lr=args.tea_sgld_lr)
    elif name == "tea_nonoise":
        # TEA with SGLD Langevin noise zeroed; everything else identical.
        return TEANoNoise(
            model, lr=args.tea_lr, optimizer_name=args.tea_optimizer,
            sgld_steps=args.tea_sgld_steps, sgld_lr=args.tea_sgld_lr,
        )
    elif name == "tea_directenergy":
        # TEA with objective replaced by direct free-energy descent on the
        # test batch. Parameter set + optimizer kept identical to TEA.
        return TEADirectEnergy(
            model, lr=args.tea_lr, optimizer_name=args.tea_optimizer,
            sgld_steps=args.tea_sgld_steps, sgld_lr=args.tea_sgld_lr,
        )
    elif name == "heat":
        stages = list(args.heat_stages) if args.heat_stages is not None else None
        return HEAT(
            model, lr=args.heat_lr, momentum=args.heat_momentum,
            stages=stages,
            eval_mode=args.heat_eval_mode,
            temperatures=args.heat_temperatures,
            aggregation=args.heat_aggregation,
            restore_prob=args.heat_restore_prob,
        ).to(device)
    elif name == "heat_singlestage":
        # HEAT instantiated with a single stage = the last stage. Does NOT
        # modify heat.py; just passes stages=[last_idx]. If --heat-stages
        # was supplied explicitly, honor that instead.
        if args.heat_stages is not None:
            stages = list(args.heat_stages)
        else:
            stages = [len(model.stage_channels) - 1]
        return HEAT(
            model, lr=args.heat_lr, momentum=args.heat_momentum,
            stages=stages,
            eval_mode=args.heat_eval_mode,
            temperatures=args.heat_temperatures,
            aggregation=args.heat_aggregation,
            restore_prob=args.heat_restore_prob,
        ).to(device)
    elif name == "epotta":
        train_loader, _ = get_clean_loaders(
            args.dataset, _clean_root(args),
            batch_size=args.batch_size, num_workers=args.num_workers,
            arch=args.arch,
        )
        return EPOTTA(model, source_loader=train_loader,
                      lr=args.epotta_lr, beta=args.epotta_beta,
                      buffer_size=args.epotta_buffer_size, device=device)
    elif name == "retta":
        return ReTTA(model, lr=args.retta_lr,
                     lambda_energy=args.retta_lambda_energy)
    raise ValueError(name)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = num_classes_for(args.dataset)
    base_model = load_model(args.arch, args.checkpoint, device, num_classes)
    base_model.eval()
    print(f"[ok] loaded {args.arch} ({args.dataset}, {num_classes} classes) "
          f"checkpoint {args.checkpoint}")

    corruptions = args.corruptions or CORRUPTIONS
    print(f"[ok] {len(corruptions)} corruptions, severity {args.severity}")
    print(f"[ok] HEAT: lr={args.heat_lr}, temps={args.heat_temperatures}, "
          f"agg={args.heat_aggregation}, restore_prob={args.heat_restore_prob}")

    # results[method][corruption] = {acc, ece, mean_conf, mean_ent, peak_mem, params, time}
    results = {m: {} for m in args.methods}
    summary = {m: {"trainable_params": 0, "peak_memory_mb": 0.0} for m in args.methods}
    timings = {m: [] for m in args.methods}
    skipped_methods: list[str] = []

    for corruption in corruptions:
        print(f"\n=== {corruption} ===")
        loader = get_corruption_loader(
            args.dataset, _corruption_root(args), corruption,
            severity=args.severity, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False,
            arch=args.arch,
        )
        for method_name in args.methods:
            if method_name in skipped_methods:
                continue
            set_seed(args.seed)
            try:
                method = build_method(method_name, base_model, device, args)
            except NoBatchNormError as e:
                print(f"  {method_name:10s}  SKIPPED: {e}")
                results[method_name] = {"_skipped": str(e)}
                summary[method_name]["skipped"] = True
                skipped_methods.append(method_name)
                continue
            t0 = time.time()
            r = evaluate_online(method, loader, device, progress=False)
            t1 = time.time()
            results[method_name][corruption] = {
                "accuracy": r.accuracy,
                "ece": r.ece,
                "mean_confidence": r.mean_confidence,
                "mean_entropy": r.mean_entropy,
                "time_seconds": t1 - t0,
            }
            timings[method_name].append(t1 - t0)
            summary[method_name]["trainable_params"] = r.trainable_params
            summary[method_name]["peak_memory_mb"] = max(
                summary[method_name]["peak_memory_mb"], r.peak_memory_mb
            )
            print(f"  {method_name:10s}  acc={r.accuracy:.4f}  ece={r.ece:.4f}  "
                  f"conf={r.mean_confidence:.3f}  time={t1-t0:.1f}s")

    # Summary table
    print("\n" + "=" * 80)
    print(f"P1 — arch={args.arch}, dataset={args.dataset}, "
          f"severity={args.severity}, seed={args.seed}")
    print("=" * 80)
    print(f"  {'method':10s}  {'acc':>8s}  {'err%':>6s}  {'ece':>7s}  "
          f"{'conf':>6s}  {'ent':>6s}  {'mem(MB)':>8s}  {'params':>10s}  {'time/c':>8s}")
    for m in args.methods:
        if m in skipped_methods:
            print(f"  {m:10s}  SKIPPED")
            continue
        accs = [results[m][c]["accuracy"] for c in corruptions]
        eces = [results[m][c]["ece"] for c in corruptions]
        confs = [results[m][c]["mean_confidence"] for c in corruptions]
        ents = [results[m][c]["mean_entropy"] for c in corruptions]
        acc_mean = sum(accs) / len(accs)
        err_mean = (1 - acc_mean) * 100
        ece_mean = sum(eces) / len(eces)
        conf_mean = sum(confs) / len(confs)
        ent_mean = sum(ents) / len(ents)
        mem = summary[m]["peak_memory_mb"]
        params = summary[m]["trainable_params"]
        t_mean = sum(timings[m]) / len(timings[m]) if timings[m] else 0.0
        print(f"  {m:10s}  {acc_mean:>8.4f}  {err_mean:>6.2f}  {ece_mean:>7.4f}  "
              f"{conf_mean:>6.3f}  {ent_mean:>6.3f}  {mem:>8.1f}  {params:>10d}  {t_mean:>7.1f}s")

    tag = f"_{args.variant_tag}" if args.variant_tag else ""
    dataset_tag = f"_{args.dataset}" if args.dataset != "cifar10" else ""
    out_path = (out_dir /
                f"p1_{args.arch}{dataset_tag}{tag}_seed{args.seed}_sev{args.severity}.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "summary": summary,
            "timings_seconds": timings,
            "skipped_methods": skipped_methods,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
