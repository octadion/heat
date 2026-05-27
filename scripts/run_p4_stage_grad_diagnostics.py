from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.adapt import evaluate_online
from src.data import get_cifar10c_loader
from src.methods import HEAT
from src.models import build_arch
from src.utils import get_device, set_seed


P4_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "snow",
    "contrast",
    "frost",
    "brightness",
]

P4_METHODS = ["heat", "heat_singlestage"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect compact stage-gradient diagnostics for P4."
    )
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--c10c-root", type=str, default="data/cifar10c")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruptions", type=str, nargs="+", default=P4_CORRUPTIONS)
    parser.add_argument("--methods", type=str, nargs="+", default=P4_METHODS,
                        choices=P4_METHODS)
    parser.add_argument("--heat-lr", type=float, default=1e-3)
    parser.add_argument("--heat-momentum", type=float, default=0.0)
    parser.add_argument("--heat-temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--heat-aggregation", type=str, default="sum",
                        choices=["sum", "self_gated", "self_gated_temperature"])
    parser.add_argument("--heat-eval-mode", action="store_true")
    parser.add_argument("--out-path", type=str,
                        default="experiments/results/p4_per_stage_grad_seed42_sev5.json")
    return parser.parse_args()


def load_base_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_arch(args.arch, num_classes=10).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def build_method(method_name: str, base_model, args, device):
    model = copy.deepcopy(base_model).to(device)
    if method_name == "heat":
        stages = None
    elif method_name == "heat_singlestage":
        stages = [len(model.stage_channels) - 1]
    else:
        raise ValueError(method_name)

    return HEAT(
        model,
        lr=args.heat_lr,
        momentum=args.heat_momentum,
        stages=stages,
        temperatures=args.heat_temperatures,
        aggregation=args.heat_aggregation,
        eval_mode=args.heat_eval_mode,
        diagnostic_snapshot=True,
    ).to(device)


def compact_diag(diag: dict, method: str, corruption: str, local_step: int) -> dict:
    row = {
        "method": method,
        "corruption": corruption,
        "local_step": local_step,
        "stage_grad_norms": diag.get("stage_grad_norms", {}),
        "energy": diag.get("energy"),
        "grad_l2": diag.get("grad_l2", diag.get("total_grad_norm")),
    }
    if diag.get("drift_l2") is not None:
        row["drift_l2"] = diag.get("drift_l2")
    return row


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = load_base_model(args, device)
    results = {method: {} for method in args.methods}
    summary = {}

    print(f"[ok] arch={args.arch}, checkpoint={args.checkpoint}")
    print(f"[ok] severity={args.severity}, seed={args.seed}")

    for method_name in args.methods:
        method_accs: list[float] = []
        method_rows = 0
        for corruption in args.corruptions:
            print(f"\n=== {method_name} / {corruption} ===")
            set_seed(args.seed)
            method = build_method(method_name, base_model, args, device)
            loader = get_cifar10c_loader(
                args.c10c_root,
                corruption,
                severity=args.severity,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                arch=args.arch,
            )
            run = evaluate_online(
                method,
                loader,
                device,
                collect_diagnostics=True,
                progress=False,
            )
            diagnostics = [
                compact_diag(diag, method_name, corruption, local_step)
                for local_step, diag in enumerate(run.diagnostics)
            ]
            results[method_name][corruption] = {
                "accuracy": run.accuracy,
                "num_total": run.num_total,
                "diagnostics": diagnostics,
            }
            method_accs.append(run.accuracy)
            method_rows += len(diagnostics)
            print(f"  acc={run.accuracy:.4f}  diagnostic_rows={len(diagnostics)}")

        summary[method_name] = {
            "mean_accuracy": sum(method_accs) / len(method_accs) if method_accs else None,
            "num_diagnostic_rows": method_rows,
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "summary": summary,
        }, f, indent=2)
        f.write("\n")

    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
