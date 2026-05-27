"""
Method factory v1.4 — adds --heat-restore-prob support.

When restore_prob == 0 (default), HEAT behaves identically to v1.2 monad.

NEW in vit/cifar100 patch:
  - Dataset-aware EPOTTA source loader (cifar10 or cifar100).
  - Architecture-aware source loader transform (CNN 32x32 or ViT 224).
  - Graceful skip for bn_adapt on non-BN models (NoBatchNormError surfaced
    to callers, which decide how to record the skip in their results JSON).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.methods import (
    HEAT, EPOTTA, ReTTA, Tent, TEA, TEANoNoise, TEADirectEnergy,
    Source, BNAdapt,
)
from src.data import get_clean_loaders


def _tea_kwargs(args):
    """Common TEA / TEA-variant kwargs resolved from args."""
    return dict(
        lr=getattr(args, "tea_lr", 5e-4),
        optimizer_name=getattr(args, "tea_optimizer", "adam"),
        sgld_steps=getattr(args, "tea_sgld_steps", 20),
        sgld_lr=getattr(args, "tea_sgld_lr", 0.1),
    )


def _heat_stages_for(args, model):
    """
    Resolve HEAT's `stages` argument.

    - If args.heat_stages is None (CLI default), return None so HEAT picks
      `list(range(len(model.stage_channels)))` — i.e. ALL stages — exactly
      as before this patch. Bit-identical to pre-patch behavior.
    - If args.heat_stages is a list of ints, return that list. Used by the
      `heat_singlestage` variant to restrict to the final stage.
    """
    stages = getattr(args, "heat_stages", None)
    if stages is None:
        return None
    return list(stages)


def build_method(name, base_model, device, args, dataset_root="data/cifar10"):
    model = copy.deepcopy(base_model).to(device)

    # Resolve dataset + arch for loaders that need them (EPOTTA source loader).
    dataset = getattr(args, "dataset", "cifar10")
    arch = getattr(args, "arch", None)

    if name == "source":
        return Source(model)
    if name == "bn_adapt":
        # Raises NoBatchNormError if the model has no BatchNorm — caller
        # (run_p1_benchmark / run_tier2) should catch and skip cleanly.
        return BNAdapt(model)
    if name == "tent":
        return Tent(
            model,
            lr=getattr(args, "tent_lr", 1e-3),
            optimizer_name=getattr(args, "tent_optimizer", "adam"),
            momentum=0.9,
        )
    if name == "tea":
        return TEA(model, **_tea_kwargs(args))
    if name == "tea_nonoise":
        # Same as TEA but with SGLD Langevin noise zeroed. Subclass forces
        # sgld_noise=0.0 regardless of any kwarg passed in.
        return TEANoNoise(model, **_tea_kwargs(args))
    if name == "tea_directenergy":
        # Same setup (norm-affine params, Adam, same lr) but objective is
        # direct free-energy descent on the test batch — no SGLD samples.
        # sgld_* kwargs are accepted (for shape compat) and ignored.
        return TEADirectEnergy(model, **_tea_kwargs(args))
    if name == "heat":
        return HEAT(
            model,
            lr=getattr(args, "heat_lr", 1e-3),
            momentum=getattr(args, "heat_momentum", 0.0),
            stages=_heat_stages_for(args, model),
            eval_mode=getattr(args, "heat_eval_mode", False),
            temperatures=getattr(args, "heat_temperatures", [1.0]),
            aggregation=getattr(args, "heat_aggregation", "sum"),
            restore_prob=getattr(args, "heat_restore_prob", 0.0),
            diagnostic_snapshot=getattr(args, "heat_diagnostic_snapshot", False),
        ).to(device)
    if name == "heat_singlestage":
        # HEAT with a single stage = the final (output-side) stage only.
        # Identical to HEAT in every other way.
        last_stage = len(model.stage_channels) - 1
        explicit_stages = _heat_stages_for(args, model)
        stages = explicit_stages if explicit_stages is not None else [last_stage]

        return HEAT(
            model,
            lr=getattr(args, "heat_lr", 1e-3),
            momentum=getattr(args, "heat_momentum", 0.0),
            stages=stages,
            eval_mode=getattr(args, "heat_eval_mode", False),
            temperatures=getattr(args, "heat_temperatures", [1.0]),
            aggregation=getattr(args, "heat_aggregation", "sum"),
            restore_prob=getattr(args, "heat_restore_prob", 0.0),
            diagnostic_snapshot=getattr(args, "heat_diagnostic_snapshot", False),
        ).to(device)
    if name == "epotta":
        train_loader, _ = get_clean_loaders(
            dataset,
            dataset_root,
            batch_size=getattr(args, "batch_size", 64),
            num_workers=getattr(args, "num_workers", 2),
            arch=arch,
        )
        return EPOTTA(
            model,
            source_loader=train_loader,
            lr=getattr(args, "epotta_lr", 1e-3),
            beta=getattr(args, "epotta_beta", 1.0),
            buffer_size=getattr(args, "epotta_buffer_size", 500),
            device=device,
        )
    if name == "retta":
        return ReTTA(
            model,
            lr=getattr(args, "retta_lr", 1e-3),
            lambda_energy=getattr(args, "retta_lambda_energy", 1.0),
        )
    raise ValueError(f"unknown method '{name}'")


def add_method_args(parser):
    # HEAT
    parser.add_argument("--heat-lr", type=float, default=1e-3)
    parser.add_argument("--heat-momentum", type=float, default=0.0)
    parser.add_argument("--heat-eval-mode", action="store_true")
    parser.add_argument("--heat-temperatures", type=float, nargs="+",
                        default=[1.0])
    parser.add_argument("--heat-aggregation", type=str, default="sum",
                        choices=["sum", "self_gated", "self_gated_temperature"])
    parser.add_argument("--heat-restore-prob", type=float, default=0.0,
                        help="Stochastic restore probability for HEAT-dyad. "
                             "Default 0 = HEAT-monad (no restore).")
    parser.add_argument("--heat-diagnostic-snapshot", action="store_true",
                    help="Store source snapshot for diagnostics only. "
                         "Needed to measure drift_l2 when restore_prob=0.")
    # Explicit HEAT stage selection. Default (None) is "all stages" — exactly
    # the prior behavior. Used by the heat_singlestage ablation variant and
    # available as a manual override for plain `heat` too. Bit-identical to
    # pre-patch when omitted.
    parser.add_argument("--heat-stages", type=int, nargs="+", default=None,
                        help="Explicit list of stage indices for HEAT. "
                             "Default = all stages (HEAT's intrinsic behavior). "
                             "Used by heat_singlestage to restrict to the last "
                             "stage. Pass space-separated 0-based indices.")

    # Tent
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-optimizer", type=str, default="adam",
                        choices=["adam", "sgd"])

    # TEA
    parser.add_argument("--tea-lr", type=float, default=5e-4)
    parser.add_argument("--tea-optimizer", type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--tea-sgld-steps", type=int, default=20)
    parser.add_argument("--tea-sgld-lr", type=float, default=0.1)

    # EPOTTA
    parser.add_argument("--epotta-lr", type=float, default=1e-3)
    parser.add_argument("--epotta-beta", type=float, default=1.0)
    parser.add_argument("--epotta-buffer-size", type=int, default=500)

    # ReTTA
    parser.add_argument("--retta-lr", type=float, default=1e-3)
    parser.add_argument("--retta-lambda-energy", type=float, default=1.0)
