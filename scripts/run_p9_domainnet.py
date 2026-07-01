"""
Thin p9 runner for DomainNet-126 (resnet50 / AdaContrast source).

Why a separate entry: run_tier2.py's argparse only knows cifar arch/dataset, and
its load_model expects a CIFAR-style ckpt["model"]. Rather than edit that
load-bearing entry point, this script builds the DomainNet model (via the
resnet50 wrapper + AdaContrast checkpoint loader) and REUSES the unchanged
scripts/run_tier2.run_p9 (adaptation loop + diagnostics) on a single-domain
stream. It writes the SAME JSON schema/name as run_tier2, so pstar_common /
the analysis read it identically.

NOTE ("TFF"): the code identifier stays `heat`; results/plots call it TFF.

Invoked by pstar_common.build_run_command for --dataset domainnet126. One call =
one method (heat | source | bn_adapt) at one (eta, p).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_tier2
from src.models.resnet50_domainnet import ResNet50DomainNet, load_domainnet_checkpoint
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="DomainNet-126 p9 runner (resnet50)")
    p.add_argument("--protocol", type=str, default="p9")   # accepted, always p9
    p.add_argument("--arch", type=str, default="resnet50")
    p.add_argument("--dataset", type=str, default="domainnet126")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--target-domain", type=str, default="clipart")
    p.add_argument("--severity", type=int, default=5)      # filename placeholder
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", type=str, nargs="+", default=["heat"])
    p.add_argument("--heat-lr", type=float, default=1e-3)
    p.add_argument("--heat-restore-prob", type=float, default=0.0)
    p.add_argument("--heat-diagnostic-snapshot", action="store_true")
    p.add_argument("--heat-bn-running-stats", type=str, default="train",
                   choices=["train", "frozen"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--variant-tag", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    # Numeric regime (matters for cross-EXPERIMENT comparability, not just speed):
    #   inherit (default) : leave torch backends at their defaults -- IDENTICAL to
    #       run_tier2.py, so DomainNet is comparable to the existing CIFAR sweep
    #       (matmul FP32, cudnn-conv TF32 on, benchmark off). RECOMMENDED.
    #   on      : force TF32 (matmul+conv) + cudnn.benchmark -> fast on A100 but
    #       DIFFERS from CIFAR/default; use only for a one-off speed run.
    #   off     : force full FP32 + benchmark off -> max cross-GPU reproducibility
    #       (use for ALL experiments if you adopt FP32 as the fixed convention).
    p.add_argument("--tf32-mode", choices=["inherit", "on", "off"], default="inherit")
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = parse_args()
    device = get_device()
    # Numeric regime (batch/steps UNCHANGED, so this only sets the precision path).
    # 'inherit' touches nothing -> identical to run_tier2/CIFAR (comparable). 'on'
    # forces TF32+autotune (fast A100, differs). 'off' forces FP32 (reproducible).
    try:
        import torch
        if torch.cuda.is_available():
            if a.tf32_mode == "on":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            elif a.tf32_mode == "off":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
            # 'inherit': leave torch defaults untouched (== run_tier2 / CIFAR).
        print(f"[perf] tf32_mode={a.tf32_mode}"
              + (" (== CIFAR/run_tier2 defaults)" if a.tf32_mode == "inherit" else ""),
              flush=True)
    except Exception:
        pass
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ResNet50DomainNet(num_classes=126)
    load_domainnet_checkpoint(model, a.checkpoint, device=device, verbose=True)
    model.to(device).eval()

    # run_p9 reads these via attributes / getattr(..., default). dataset !=
    # cifar10 makes run_tier2._corruption_root return c100c_root -> data_root.
    args_ns = SimpleNamespace(
        arch=a.arch, dataset=a.dataset,
        c10c_root=a.data_root, c100c_root=a.data_root,
        cifar10_root=a.data_root, cifar100_root=a.data_root,
        severity=a.severity, corruptions=[a.target_domain],
        batch_size=a.batch_size, num_workers=a.num_workers, seed=a.seed,
        methods=list(a.methods),
        heat_lr=a.heat_lr, heat_momentum=0.0, heat_temperatures=[1.0],
        heat_aggregation="sum", heat_eval_mode=False, heat_adapt_params="full",
        heat_bn_running_stats=a.heat_bn_running_stats, heat_stages=None,
        heat_restore_prob=a.heat_restore_prob,
        heat_diagnostic_snapshot=a.heat_diagnostic_snapshot,
    )

    set_seed(a.seed)
    results = run_tier2.run_p9(args_ns, model, device)

    tag = f"_{a.variant_tag}" if a.variant_tag else ""
    out_path = out_dir / f"p9_{a.arch}{tag}_seed{a.seed}_sev{a.severity}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": {k: v for k, v in vars(a).items()},
                   "results": results}, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
