from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis_common import (
    DEFAULT_P9_PATTERNS,
    accuracy_by_corruption,
    compute_r2,
    diagnostics_for_file,
    ensure_analysis_dirs,
    extract_metadata,
    find_key,
    find_json_files,
    group_rows_by_block,
    load_json,
    mean_or_none,
    row_metric,
    safe_float,
    stationary_rows,
    write_json,
    write_text,
)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_dev = [x - x_mean for x in xs]
    y_dev = [y - y_mean for y in ys]
    denom = math.sqrt(sum(v * v for v in x_dev) * sum(v * v for v in y_dev))
    if denom == 0:
        return None
    return float(sum(x * y for x, y in zip(x_dev, y_dev)) / denom)


def spearman_correlation(xs: list[float], ys: list[float]) -> dict[str, Any]:
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {"available": False, "correlation": None, "pvalue": None}
    if len(xs) < 2:
        return {"available": True, "correlation": None, "pvalue": None}
    result = spearmanr(xs, ys)
    corr = None if math.isnan(result.correlation) else float(result.correlation)
    pvalue = None if math.isnan(result.pvalue) else float(result.pvalue)
    return {"available": True, "correlation": corr, "pvalue": pvalue}


def fit_models(xs: list[float], ys: list[float]) -> dict[str, Any]:
    if len(xs) < 2 or len(set(xs)) < 2:
        return {
            "linear": None,
            "quadratic_x2_only": None,
        }
    try:
        import numpy as np
    except Exception:
        return {
            "linear": None,
            "quadratic_x2_only": None,
        }

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    linear_a, linear_b = np.polyfit(x_arr, y_arr, 1)
    linear_pred = (linear_a * x_arr + linear_b).tolist()

    quad_design = np.column_stack([x_arr ** 2, np.ones_like(x_arr)])
    quad_a, quad_b = np.linalg.lstsq(quad_design, y_arr, rcond=None)[0]
    quad_pred = (quad_a * (x_arr ** 2) + quad_b).tolist()

    return {
        "linear": {
            "form": "y = a*x + b",
            "a": float(linear_a),
            "b": float(linear_b),
            "r2": compute_r2(ys, linear_pred),
        },
        "quadratic_x2_only": {
            "form": "y = a*x^2 + b",
            "a": float(quad_a),
            "b": float(quad_b),
            "r2": compute_r2(ys, quad_pred),
        },
    }


def energy_is_invalid(row: dict[str, Any]) -> bool:
    if row.get("energy") is None:
        return False
    try:
        value = float(row["energy"])
    except (TypeError, ValueError):
        return True
    return not math.isfinite(value)


def config_health(rows: list[dict[str, Any]], data: dict[str, Any] | None = None) -> dict[str, Any]:
    invalid_flag = any(bool(row.get("invalid")) for row in rows)
    if data is not None:
        invalid_flag = invalid_flag or bool(find_key(data, {"invalid"}))
    has_nan_or_inf_energy = any(energy_is_invalid(row) for row in rows)
    drift_values = [row_metric(row, "drift_l2") for row in rows]
    drift_values = [v for v in drift_values if v is not None]
    max_drift_l2 = max(drift_values) if drift_values else None
    drift_over_100 = max_drift_l2 is not None and max_drift_l2 > 100.0
    stable = not (invalid_flag or has_nan_or_inf_energy or drift_over_100)
    reasons = []
    if invalid_flag:
        reasons.append("invalid=True")
    if has_nan_or_inf_energy:
        reasons.append("NaN/Inf energy")
    if drift_over_100:
        reasons.append("drift_l2 > 100")
    return {
        "invalid": invalid_flag,
        "has_nan_or_inf_energy": has_nan_or_inf_energy,
        "max_drift_l2": max_drift_l2,
        "drift_over_100": drift_over_100,
        "stable": stable,
        "unstable_reasons": reasons,
    }


def build_points_for_file(
    data: dict[str, Any],
    path: Path,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    method, rows, _layout = diagnostics_for_file(data, preferred_method="heat")
    warnings: list[str] = []
    if method is None or not rows:
        return [], ["no stream or per-corruption diagnostics"], method

    accuracies = accuracy_by_corruption(data, method)
    if not accuracies:
        return [], ["accuracy not present in the same file"], method

    meta = extract_metadata(data, path, rows)
    health = config_health(rows, data)
    groups = group_rows_by_block(rows)
    points: list[dict[str, Any]] = []
    for corruption, group in groups.items():
        if corruption not in accuracies:
            warnings.append(f"{corruption}: missing accuracy")
            continue
        stat_rows = stationary_rows(group)
        drift_mean = mean_or_none([row_metric(row, "drift_l2") for row in stat_rows])
        grad_mean = mean_or_none([row_metric(row, "grad_l2") for row in stat_rows])
        if drift_mean is None:
            warnings.append(f"{corruption}: missing stationary drift_l2")
            continue
        log_drift = math.log10(drift_mean + 1e-12)
        accuracy = accuracies[corruption]
        points.append({
            "file": str(path),
            "arch": meta.get("arch"),
            "seed": meta.get("seed"),
            "restore_prob": meta.get("restore_prob"),
            "eta": meta.get("eta"),
            "method": method,
            "corruption": corruption,
            "stationary_drift_l2": drift_mean,
            "log10_stationary_drift_l2": log_drift,
            "stationary_grad_l2": grad_mean,
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
            "stationary_rows": len(stat_rows),
            "config_invalid": health["invalid"],
            "has_nan_or_inf_energy": health["has_nan_or_inf_energy"],
            "config_max_drift_l2": health["max_drift_l2"],
            "config_drift_over_100": health["drift_over_100"],
            "stable": health["stable"],
            "unstable_reasons": health["unstable_reasons"],
        })
    return points, warnings, method


def summarize_axis(points: list[dict[str, Any]], x_key: str) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        x = safe_float(point.get(x_key))
        y = safe_float(point.get("error"))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return {
        "n": len(xs),
        "pearson": pearson_correlation(xs, ys),
        "spearman": spearman_correlation(xs, ys),
        "fits": fit_models(xs, ys),
    }


def summarize_subset(points: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(points),
        "raw_drift": summarize_axis(points, "stationary_drift_l2"),
        "log10_drift": summarize_axis(points, "log10_stationary_drift_l2"),
    }


def report_groups(points: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [p for p in points if p.get("stable")]
    return {
        "all_points": summarize_subset(points),
        "stable_only": summarize_subset(stable),
        "resnet18_stable_only": summarize_subset([
            p for p in stable if p.get("arch") == "resnet18"
        ]),
        "wrn28_10_stable_only": summarize_subset([
            p for p in stable if p.get("arch") == "wrn28_10"
        ]),
        "dyad_only_p_gt_0_stable_only": summarize_subset([
            p for p in stable
            if p.get("restore_prob") is not None and p["restore_prob"] > 0
        ]),
    }


def filter_points(points: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(points)
    if args.stable_only:
        selected = [p for p in selected if p.get("stable")]
    if args.arch != "all":
        selected = [p for p in selected if p.get("arch") == args.arch]
    if args.exclude_p0:
        selected = [
            p for p in selected
            if p.get("restore_prob") is None or p.get("restore_prob") != 0.0
        ]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-only", action="store_true",
                        help="Exclude configs with invalid=True, NaN/Inf energy, or drift_l2 > 100.")
    parser.add_argument("--arch", choices=["resnet18", "wrn28_10", "all"], default="all")
    parser.add_argument("--exclude-p0", action="store_true",
                        help="Exclude restore_prob == 0 points from the filtered point set.")
    return parser.parse_args()


def plot_scatter(
    points: list[dict[str, Any]],
    fits: dict[str, Any],
    out_path: Path,
    x_key: str,
    x_label: str,
) -> bool:
    if not points:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    xs = [p[x_key] for p in points]
    ys = [p["error"] for p in points]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(xs, ys, s=42)
    for point, x, y in zip(points, xs, ys):
        ax.annotate(point["corruption"], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")

    if fits.get("linear") is not None and min(xs) != max(xs):
        x_grid = np.linspace(min(xs), max(xs), 100)
        a = fits["linear"]["a"]
        b = fits["linear"]["b"]
        ax.plot(x_grid, a * x_grid + b, linewidth=1.2, label="linear")
        ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel("error (1 - accuracy)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def build_summary_md(payload: dict[str, Any]) -> str:
    groups = payload["statistics"]["groups"]
    lines = [
        "# Drift Accuracy Filtered",
        "",
        "This analysis builds per-corruption points from existing P9 diagnostic JSON files.",
        "",
        f"- Files matched: {len(payload['files_considered'])}",
        f"- Total points: {len(payload['points'])}",
        f"- Filtered points: {len(payload['filtered_points'])}",
        f"- Files skipped: {len(payload['skipped_files'])}",
        f"- Active filters: {payload['active_filters']}",
        "",
    ]
    if not payload["points"]:
        lines.extend([
            "No drift-to-error points were available.",
            "",
        ])

    lines.extend([
        "| cohort | n | raw Pearson | raw Spearman | log10 Pearson | log10 Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, stats in groups.items():
        raw = stats["raw_drift"]
        log_stats = stats["log10_drift"]
        lines.append(
            "| {name} | {n} | {rp} | {rs} | {lp} | {ls} |".format(
                name=name,
                n=stats.get("n"),
                rp=raw.get("pearson"),
                rs=raw.get("spearman", {}).get("correlation"),
                lp=log_stats.get("pearson"),
                ls=log_stats.get("spearman", {}).get("correlation"),
            )
        )
    lines.append("")

    if payload["skipped_files"]:
        lines.extend(["## Skipped Files", ""])
        for item in payload["skipped_files"]:
            lines.append(f"- {item['file']}: {item['reason']}")
        lines.append("")

    if payload["warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    analysis_dir, plot_dir = ensure_analysis_dirs()
    files, matches = find_json_files(DEFAULT_P9_PATTERNS)
    warnings: list[str] = []
    skipped_files: list[dict[str, str]] = []
    points: list[dict[str, Any]] = []

    if not files:
        warnings.append("No files matched the P9 diagnostic glob patterns.")

    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": f"json_load_failed: {exc}"})
            continue
        file_points, file_warnings, _method = build_points_for_file(data, path)
        if not file_points:
            skipped_files.append({"file": str(path), "reason": "; ".join(file_warnings)})
        else:
            points.extend(file_points)
            warnings.extend(f"{path}: {w}" for w in file_warnings)

    filtered_points = filter_points(points, args)
    groups = report_groups(points)
    stats = {
        "groups": groups,
        "filtered_raw_drift": summarize_axis(filtered_points, "stationary_drift_l2"),
        "filtered_log10_drift": summarize_axis(filtered_points, "log10_stationary_drift_l2"),
    }

    plots: list[str] = []
    scatter_path = plot_dir / "drift_accuracy_scatter.png"
    filtered_raw = stats["filtered_raw_drift"]
    if plot_scatter(
        filtered_points,
        filtered_raw.get("fits", {}),
        scatter_path,
        "stationary_drift_l2",
        "stationary drift_l2",
    ):
        plots.append(str(scatter_path))
    log_scatter_path = plot_dir / "drift_accuracy_log_scatter.png"
    filtered_log = stats["filtered_log10_drift"]
    if plot_scatter(
        filtered_points,
        filtered_log.get("fits", {}),
        log_scatter_path,
        "log10_stationary_drift_l2",
        "log10(stationary drift_l2 + 1e-12)",
    ):
        plots.append(str(log_scatter_path))

    payload = {
        "patterns": DEFAULT_P9_PATTERNS,
        "matches_by_pattern": matches,
        "files_considered": [str(p) for p in files],
        "skipped_files": skipped_files,
        "warnings": warnings,
        "points": points,
        "filtered_points": filtered_points,
        "active_filters": {
            "stable_only": args.stable_only,
            "arch": args.arch,
            "exclude_p0": args.exclude_p0,
        },
        "statistics": stats,
        "plots": plots,
    }
    write_json(analysis_dir / "drift_accuracy_filtered.json", payload)
    write_text(analysis_dir / "drift_accuracy_filtered_summary.md", build_summary_md(payload))
    # Keep the original filenames current for callers that still expect them.
    write_json(analysis_dir / "drift_accuracy.json", payload)
    write_text(analysis_dir / "drift_accuracy_summary.md", build_summary_md(payload))
    print(f"[saved] {analysis_dir / 'drift_accuracy_filtered.json'}")
    print(f"[saved] {analysis_dir / 'drift_accuracy_filtered_summary.md'}")
    print(f"[plots] {len(plots)}")


if __name__ == "__main__":
    main()
