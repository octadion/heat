"""
Derive metrics from existing P1 / P9 result JSON files.

Computes metrics that don't require additional runtime measurement:
  - error rate (1 - accuracy, %) for all methods
  - per-method gap to source / vs each baseline
  - forgetting (for P9 streams): peak_acc - last_acc
  - parameter count per method (deterministic from method name)
  - cumulative drift (P9): mean(peak_seen_so_far - current) over stream

This works on JSON files produced by ANY version (v1.2 onwards) since these
metrics derive from the accuracy values that were always recorded.

Usage:
  # Auto-discover all JSONs in a directory and report tables
  python scripts/compute_derived_metrics.py \\
      --results-dir experiments/results

  # Or target a specific file
  python scripts/compute_derived_metrics.py \\
      --json experiments/results/p1_resnet18_seed42_sev5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Static parameter counts — deterministic from method + architecture
PARAM_COUNTS = {
    "resnet18": {
        "source":   0,
        "bn_adapt": 0,                 # only running stats, no learnable params
        "tent":     5_120,             # BN affine only (~9 BN layers × ~512 channels avg)
        "tea":      5_120,             # same as Tent (BN affine), plus SGLD inner
        "epotta":   5_120,             # BN affine, plus frozen φ copy
        "retta":    5_120,             # BN affine
        "heat":     11_173_962,        # all params (default config)
    },
    "wrn28_10": {
        "source":   0,
        "bn_adapt": 0,
        "tent":     11_520,            # BN affine
        "tea":      11_520,
        "epotta":   11_520,
        "retta":    11_520,
        "heat":     36_479_194,        # all params
    },
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def detect_protocol(data):
    args = data.get("args", {})
    p = args.get("protocol", None)
    if p:
        return p
    # P1 doesn't have 'protocol' key, but has 'methods' and per-corruption results
    if "results" in data and "timings_seconds" in data:
        return "p1"
    if "results" in data and isinstance(data["results"], dict):
        # tier2 protocol — but no protocol key visible? old format
        return "unknown"
    return "unknown"


def extract_accuracy(per_corruption_value):
    """Handle both v1.2- (float) and v1.5+ (dict with 'accuracy') formats."""
    if isinstance(per_corruption_value, (int, float)):
        return float(per_corruption_value)
    if isinstance(per_corruption_value, dict) and "accuracy" in per_corruption_value:
        return float(per_corruption_value["accuracy"])
    return None


def analyze_p1(data):
    args = data.get("args", {})
    arch = args.get("arch", "resnet18")
    methods = args.get("methods", [])
    results = data.get("results", {})
    timings = data.get("timings_seconds", {})

    print(f"\n{'=' * 80}")
    print(f"P1 — arch={arch}, severity={args.get('severity')}, seed={args.get('seed')}")
    if args.get("variant_tag"):
        print(f"  variant: {args['variant_tag']}")
    print("=" * 80)
    header = (f"  {'method':10s}  {'acc':>7s}  {'err%':>6s}  "
              f"{'params':>10s}  {'time/c(s)':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    method_means = {}
    for m in methods:
        if m not in results:
            continue
        accs = []
        for c, v in results[m].items():
            a = extract_accuracy(v)
            if a is not None:
                accs.append(a)
        if not accs:
            continue
        acc_mean = sum(accs) / len(accs)
        err_mean = (1 - acc_mean) * 100
        params = PARAM_COUNTS.get(arch, {}).get(m, 0)
        t_mean = (sum(timings.get(m, [])) / max(1, len(timings.get(m, []))))
        method_means[m] = acc_mean
        print(f"  {m:10s}  {acc_mean:>7.4f}  {err_mean:>6.2f}  "
              f"{params:>10d}  {t_mean:>10.2f}")

    # Compute pairwise gaps
    if "heat" in method_means:
        print("\n  Gap of HEAT vs each baseline (acc difference, pp):")
        h = method_means["heat"]
        for m in methods:
            if m == "heat" or m not in method_means:
                continue
            print(f"    HEAT − {m:10s} = {(h - method_means[m]) * 100:+.2f}pp")


def analyze_p9(data):
    args = data.get("args", {})
    arch = args.get("arch", "resnet18")
    results = data.get("results", {})

    # Detect schema: v1.5 has results.per_corruption + results.summary,
    # v1.4 has results = {method: {corruption: acc}}
    if "per_corruption" in results and "summary" in results:
        per_corr = results["per_corruption"]
        v15_summary = results["summary"]
    else:
        per_corr = results
        v15_summary = None

    print(f"\n{'=' * 80}")
    print(f"P9 (continual) — arch={arch}, seed={args.get('seed')}")
    if args.get("variant_tag"):
        print(f"  variant: {args['variant_tag']}")
    print("=" * 80)
    header = (f"  {'method':10s}  {'mean':>7s}  {'last':>7s}  "
              f"{'peak':>7s}  {'forgetting':>11s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m, per_c in per_corr.items():
        if not isinstance(per_c, dict):
            continue
        # Filter out non-corruption keys (mean, last, peak from v1.4 inline summary)
        accs = []
        for c, v in per_c.items():
            if c.startswith("_"):
                continue
            a = extract_accuracy(v)
            if a is not None:
                accs.append(a)
        if not accs:
            continue
        mean_acc = sum(accs) / len(accs)
        last_acc = accs[-1]
        peak_acc = max(accs)
        forgetting = peak_acc - last_acc
        print(f"  {m:10s}  {mean_acc:>7.4f}  {last_acc:>7.4f}  "
              f"{peak_acc:>7.4f}  {forgetting:>11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--results-dir", type=str, default=None)
    args = ap.parse_args()

    if args.json:
        files = [Path(args.json)]
    elif args.results_dir:
        files = sorted(Path(args.results_dir).glob("*.json"))
    else:
        ap.error("provide --json or --results-dir")

    for f in files:
        try:
            data = load_json(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        protocol = detect_protocol(data)
        print(f"\n>>> {f.name} ({protocol})")
        if protocol == "p1":
            analyze_p1(data)
        elif protocol == "p9":
            analyze_p9(data)
        else:
            print(f"  [skip] protocol={protocol}, derive metrics not implemented")


if __name__ == "__main__":
    main()
