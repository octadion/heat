"""Summarize future review-response result JSONs into markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_SPECS = [
    ("review_baseline_continual_summary.md", "baseline_continual"),
    ("review_bn_full_ablation_summary.md", "bn_full_ablation"),
    ("review_batch_size_summary.md", "batch_size"),
    ("review_tea_sensitivity_summary.md", "tea_sensitivity"),
    ("review_vit_lr_summary.md", "vit_lr"),
    ("review_cifar100_summary.md", "cifar100"),
]

HEADERS = [
    "method", "architecture", "dataset", "severity", "batch size", "seed",
    "accuracy", "ECE", "forgetting", "final-block accuracy", "runtime",
    "result JSON path",
]


def _safe_load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _runtime_from_method(data: dict[str, Any], method: str) -> float | None:
    timings = data.get("timings_seconds")
    if isinstance(timings, dict) and isinstance(timings.get(method), list):
        vals = [float(v) for v in timings[method] if isinstance(v, (int, float))]
        return _mean(vals)
    return None


def _rows_from_p1(data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    args = data.get("args", {})
    results = data.get("results", {})
    rows = []
    if not isinstance(results, dict):
        return rows
    for method, per_corr in results.items():
        if not isinstance(per_corr, dict) or method.startswith("_"):
            continue
        accs = []
        eces = []
        runtimes = []
        for entry in per_corr.values():
            if isinstance(entry, dict):
                if isinstance(entry.get("accuracy"), (int, float)):
                    accs.append(float(entry["accuracy"]))
                if isinstance(entry.get("ece"), (int, float)):
                    eces.append(float(entry["ece"]))
                if isinstance(entry.get("time_seconds"), (int, float)):
                    runtimes.append(float(entry["time_seconds"]))
            elif isinstance(entry, (int, float)):
                accs.append(float(entry))
        if not accs:
            continue
        rows.append({
            "method": method,
            "architecture": args.get("arch"),
            "dataset": args.get("dataset", "cifar10"),
            "severity": args.get("severity"),
            "batch size": args.get("batch_size"),
            "seed": args.get("seed"),
            "accuracy": _mean(accs),
            "ECE": _mean(eces),
            "forgetting": None,
            "final-block accuracy": None,
            "runtime": _mean(runtimes) or _runtime_from_method(data, method),
            "result JSON path": str(path),
            "_variant": args.get("variant_tag", ""),
            "_protocol": "p1",
        })
    return rows


def _entry_accuracy(entry: Any) -> float | None:
    if isinstance(entry, dict):
        value = entry.get("accuracy", entry.get("acc"))
        return float(value) if isinstance(value, (int, float)) else None
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


def _entry_ece(entry: Any) -> float | None:
    if isinstance(entry, dict) and isinstance(entry.get("ece"), (int, float)):
        return float(entry["ece"])
    return None


def _rows_from_p9(data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    args = data.get("args", {})
    root = data.get("results", {})
    if isinstance(root, dict) and "per_corruption" in root:
        per_method = root.get("per_corruption", {})
        summaries = root.get("summary", {})
    else:
        per_method = root
        summaries = {}
    rows = []
    if not isinstance(per_method, dict):
        return rows
    for method, per_corr in per_method.items():
        if not isinstance(per_corr, dict) or method.startswith("_"):
            continue
        accs = []
        eces = []
        for key, entry in per_corr.items():
            if str(key).startswith("_"):
                continue
            acc = _entry_accuracy(entry)
            ece = _entry_ece(entry)
            if acc is not None:
                accs.append(acc)
            if ece is not None:
                eces.append(ece)
        if not accs:
            continue
        summary = summaries.get(method, {}) if isinstance(summaries, dict) else {}
        rows.append({
            "method": method,
            "architecture": args.get("arch"),
            "dataset": args.get("dataset", "cifar10"),
            "severity": args.get("severity"),
            "batch size": args.get("batch_size"),
            "seed": args.get("seed"),
            "accuracy": summary.get("mean_accuracy", _mean(accs)),
            "ECE": summary.get("mean_ece", _mean(eces)),
            "forgetting": summary.get(
                "forgetting",
                (max(accs) - accs[-1]) if accs else None,
            ),
            "final-block accuracy": summary.get("last_accuracy", accs[-1]),
            "runtime": None,
            "result JSON path": str(path),
            "_variant": args.get("variant_tag", ""),
            "_protocol": args.get("protocol", "p9"),
        })
    return rows


def _rows_from_hp_search(data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    args = data.get("args", {})
    if "variant" not in args or not isinstance(data.get("results"), list):
        return []
    rows = []
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        rows.append({
            "method": args.get("variant"),
            "architecture": args.get("arch"),
            "dataset": args.get("dataset", "cifar10"),
            "severity": args.get("severity"),
            "batch size": args.get("batch_size"),
            "seed": args.get("seed"),
            "accuracy": item.get("p1_mean", item.get("p9_mean")),
            "ECE": None,
            "forgetting": None,
            "final-block accuracy": item.get("p9_last"),
            "runtime": None,
            "result JSON path": str(path),
            "_variant": f"lr={item.get('lr')};p={args.get('restore_prob')}",
            "_protocol": "hp_search",
        })
    return rows


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        data = _safe_load(path)
        if data is None:
            continue
        args = data.get("args", {})
        protocol = args.get("protocol")
        if protocol == "p9":
            rows.extend(_rows_from_p9(data, path))
        elif data.get("timings_seconds") is not None:
            rows.extend(_rows_from_p1(data, path))
        rows.extend(_rows_from_hp_search(data, path))
    return rows


def classify(row: dict[str, Any], group: str) -> bool:
    variant = str(row.get("_variant", "")).lower()
    method = str(row.get("method", "")).lower()
    dataset = str(row.get("dataset", "")).lower()
    arch = str(row.get("architecture", "")).lower()
    protocol = str(row.get("_protocol", "")).lower()

    if group == "baseline_continual":
        return protocol == "p9" and dataset == "cifar10" and method in {
            "rdumb", "eata", "sar", "cotta", "source", "bn_adapt", "tent", "heat"
        }
    if group == "bn_full_ablation":
        return "bn_affine_only" in variant or "bn_stats_frozen" in variant
    if group == "batch_size":
        return "batch_size" in variant or "bs8" in variant or "bs1" in variant
    if group == "tea_sensitivity":
        return method == "tea" and ("sgld" in variant or "tea" in variant)
    if group == "vit_lr":
        return arch == "vit_s"
    if group == "cifar100":
        return dataset == "cifar100"
    return False


def write_summary(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Generated by `scripts/summarize_review_results.py` from future result JSONs.",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    for row in rows:
        values = [_fmt(row.get(header)).replace("|", "\\|") for header in HEADERS]
        lines.append("| " + " | ".join(values) + " |")
    if not rows:
        lines.append("| " + " | ".join(["no matching results yet"] + [""] * (len(HEADERS) - 1)) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="experiments/results")
    ap.add_argument("--out-dir", type=str, default="experiments/analysis")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_dir) if results_dir.exists() else []
    for filename, group in SUMMARY_SPECS:
        selected = [row for row in rows if classify(row, group)]
        title = filename.removesuffix(".md").replace("_", " ").title()
        write_summary(out_dir / filename, selected, title)
        print(f"[saved] {out_dir / filename}")


if __name__ == "__main__":
    main()

