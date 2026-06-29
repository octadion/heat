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
import math

import torch
import torch.nn as nn

from src.methods import (
    HEAT, EPOTTA, ReTTA, Tent, TEA, TEANoNoise, TEADirectEnergy,
    Source, BNAdapt, EATA, SAR, PeriodicReset, select_norm_affine_params,
)
from src.methods.eata import compute_fisher_diagonal
from src.data import get_clean_loaders


def _tea_kwargs(args):
    """Common TEA / TEA-variant kwargs resolved from args."""
    return dict(
        lr=getattr(args, "tea_lr", 5e-4),
        optimizer_name=getattr(args, "tea_optimizer", "adam"),
        sgld_steps=getattr(args, "tea_sgld_steps", 20),
        sgld_lr=getattr(args, "tea_sgld_lr", 0.1),
        sgld_noise=getattr(args, "tea_sgld_noise", 0.01),
        use_buffer=getattr(args, "tea_use_buffer", True),
        buffer_size=getattr(args, "tea_buffer_size", 1000),
        buffer_reinit_prob=getattr(args, "tea_buffer_reinit_prob", 0.05),
        entropy_coef=getattr(args, "tea_entropy_coef", 1.0),
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


def _num_classes_from_model(model: nn.Module) -> int:
    for attr in ("linear", "fc", "classifier", "head"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return 10


def _entropy_margin(default_value, model: nn.Module) -> float | None:
    if default_value is not None:
        return float(default_value)
    return 0.4 * math.log(float(_num_classes_from_model(model)))


def _build_eata_fishers(model, device, args, dataset, dataset_root, arch):
    fisher_alpha = float(getattr(args, "eata_fisher_alpha", 2000.0))
    fisher_samples = int(getattr(args, "eata_fisher_samples", 2000))
    if fisher_alpha <= 0.0:
        return None
    if fisher_samples <= 0:
        raise ValueError(
            "EATA requires Fisher statistics for faithful comparison. "
            "Set --eata-fisher-samples > 0, or set --eata-fisher-alpha 0 "
            "and label the run as an explicit no-Fisher ablation."
        )

    select_norm_affine_params(model)
    train_loader, test_loader = get_clean_loaders(
        dataset,
        dataset_root,
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 2),
        arch=arch,
    )
    split = getattr(args, "eata_fisher_split", "train")
    loader = test_loader if split == "test" else train_loader
    return compute_fisher_diagonal(
        model,
        loader,
        device=device,
        max_samples=fisher_samples,
    )


def _build_method_impl(name, base_model, device, args, dataset_root="data/cifar10"):
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
    if name == "eata":
        fishers = _build_eata_fishers(model, device, args, dataset, dataset_root, arch)
        return EATA(
            model,
            lr=getattr(args, "eata_lr", 5e-3),
            optimizer_name=getattr(args, "eata_optimizer", "sgd"),
            momentum=getattr(args, "eata_momentum", 0.9),
            e_margin=_entropy_margin(getattr(args, "eata_e_margin", None), model),
            d_margin=getattr(args, "eata_d_margin", 0.05),
            fisher_alpha=getattr(args, "eata_fisher_alpha", 2000.0),
            fishers=fishers,
        )
    if name == "sar":
        return SAR(
            model,
            lr=getattr(args, "sar_lr", 2.5e-4),
            momentum=getattr(args, "sar_momentum", 0.9),
            rho=getattr(args, "sar_rho", 0.05),
            e_margin=_entropy_margin(getattr(args, "sar_e_margin", None), model),
            reset_ema_threshold=getattr(args, "sar_reset_ema_threshold", 0.2),
            ema_momentum=getattr(args, "sar_ema_momentum", 0.9),
        )
    if name == "heat":
        adapt_params = getattr(args, "heat_adapt_params", "full")
        return HEAT(
            model,
            lr=getattr(args, "heat_lr", 1e-3),
            momentum=getattr(args, "heat_momentum", 0.0),
            stages=_heat_stages_for(args, model),
            eval_mode=getattr(args, "heat_eval_mode", False),
            temperatures=getattr(args, "heat_temperatures", [1.0]),
            aggregation=getattr(args, "heat_aggregation", "sum"),
            update_all_params=(adapt_params == "full"),
            bn_running_stats=getattr(args, "heat_bn_running_stats", "train"),
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
            update_all_params=(getattr(args, "heat_adapt_params", "full") == "full"),
            bn_running_stats=getattr(args, "heat_bn_running_stats", "train"),
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


def build_method(name, base_model, device, args, dataset_root="data/cifar10"):
    if name == "rdumb":
        base_name = getattr(args, "rdumb_base_method", "tent")
        if base_name == "rdumb":
            raise ValueError("rdumb_base_method cannot be 'rdumb'")
        inner = _build_method_impl(base_name, base_model, device, args, dataset_root)
        return PeriodicReset(
            inner,
            reset_interval=getattr(args, "rdumb_reset_interval", 157),
            reset_scope=getattr(args, "rdumb_reset_scope", "full_model"),
            reset_optimizer_state=getattr(args, "rdumb_reset_optimizer_state", True),
            reset_bn_running_stats=getattr(args, "rdumb_reset_bn_running_stats", True),
            label="rdumb",
        )
    return _build_method_impl(name, base_model, device, args, dataset_root)


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
    parser.add_argument("--heat-adapt-params", type=str, default="full",
                        choices=["full", "bn_affine_only"],
                        help="Parameter subset for HEAT/TFF. Default full "
                             "preserves legacy behavior.")
    parser.add_argument("--heat-bn-running-stats", type=str, default="train",
                        choices=["train", "frozen"],
                        help="BN running-stat behavior for HEAT/TFF. Default "
                             "train preserves legacy behavior.")
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
    parser.add_argument("--tea-sgld-noise", type=float, default=0.01)
    parser.add_argument("--tea-use-buffer", dest="tea_use_buffer",
                        action="store_true", default=True)
    parser.add_argument("--tea-no-buffer", dest="tea_use_buffer",
                        action="store_false")
    parser.add_argument("--tea-buffer-size", type=int, default=1000)
    parser.add_argument("--tea-buffer-reinit-prob", type=float, default=0.05)
    parser.add_argument("--tea-entropy-coef", type=float, default=1.0)

    # EATA
    parser.add_argument("--eata-lr", type=float, default=5e-3)
    parser.add_argument("--eata-optimizer", type=str, default="sgd",
                        choices=["sgd", "adam"])
    parser.add_argument("--eata-momentum", type=float, default=0.9)
    parser.add_argument("--eata-e-margin", type=float, default=None)
    parser.add_argument("--eata-d-margin", type=float, default=0.05)
    parser.add_argument("--eata-fisher-alpha", type=float, default=2000.0)
    parser.add_argument("--eata-fisher-samples", type=int, default=2000)
    parser.add_argument("--eata-fisher-split", type=str, default="train",
                        choices=["train", "test"])

    # SAR
    parser.add_argument("--sar-lr", type=float, default=2.5e-4)
    parser.add_argument("--sar-momentum", type=float, default=0.9)
    parser.add_argument("--sar-rho", type=float, default=0.05)
    parser.add_argument("--sar-e-margin", type=float, default=None)
    parser.add_argument("--sar-reset-ema-threshold", type=float, default=0.2)
    parser.add_argument("--sar-ema-momentum", type=float, default=0.9)

    # RDumb / periodic reset
    parser.add_argument("--rdumb-base-method", type=str, default="tent",
                        choices=["tent", "eata", "sar", "heat", "tea"])
    parser.add_argument("--rdumb-reset-interval", type=int, default=157)
    parser.add_argument("--rdumb-reset-scope", type=str, default="full_model",
                        choices=["full_model", "trainable_params"])
    parser.add_argument("--rdumb-reset-optimizer-state",
                        dest="rdumb_reset_optimizer_state",
                        action="store_true", default=True)
    parser.add_argument("--rdumb-keep-optimizer-state",
                        dest="rdumb_reset_optimizer_state",
                        action="store_false")
    parser.add_argument("--rdumb-reset-bn-running-stats",
                        dest="rdumb_reset_bn_running_stats",
                        action="store_true", default=True)
    parser.add_argument("--rdumb-keep-bn-running-stats",
                        dest="rdumb_reset_bn_running_stats",
                        action="store_false")

    # EPOTTA
    parser.add_argument("--epotta-lr", type=float, default=1e-3)
    parser.add_argument("--epotta-beta", type=float, default=1.0)
    parser.add_argument("--epotta-buffer-size", type=int, default=500)

    # ReTTA
    parser.add_argument("--retta-lr", type=float, default=1e-3)
    parser.add_argument("--retta-lambda-energy", type=float, default=1.0)
