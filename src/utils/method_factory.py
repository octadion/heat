"""
Method factory v1.4 — adds --heat-restore-prob support.

When restore_prob == 0 (default), HEAT behaves identically to v1.2 monad.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.methods import HEAT, EPOTTA, ReTTA, Tent, TEA, Source, BNAdapt
from src.data import get_cifar10_loaders


def build_method(name, base_model, device, args, dataset_root="data/cifar10"):
    model = copy.deepcopy(base_model).to(device)

    if name == "source":
        return Source(model)
    if name == "bn_adapt":
        return BNAdapt(model)
    if name == "tent":
        return Tent(
            model,
            lr=getattr(args, "tent_lr", 1e-3),
            optimizer_name=getattr(args, "tent_optimizer", "adam"),
            momentum=0.9,
        )
    if name == "tea":
        return TEA(
            model,
            lr=getattr(args, "tea_lr", 5e-4),
            optimizer_name=getattr(args, "tea_optimizer", "adam"),
            sgld_steps=getattr(args, "tea_sgld_steps", 20),
            sgld_lr=getattr(args, "tea_sgld_lr", 0.1),
        )
    if name == "heat":
        return HEAT(
            model,
            lr=getattr(args, "heat_lr", 1e-3),
            momentum=getattr(args, "heat_momentum", 0.0),
            eval_mode=getattr(args, "heat_eval_mode", False),
            temperatures=getattr(args, "heat_temperatures", [1.0]),
            aggregation=getattr(args, "heat_aggregation", "sum"),
            restore_prob=getattr(args, "heat_restore_prob", 0.0),
        ).to(device)
    if name == "epotta":
        train_loader, _ = get_cifar10_loaders(
            dataset_root,
            batch_size=getattr(args, "batch_size", 64),
            num_workers=getattr(args, "num_workers", 2),
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
