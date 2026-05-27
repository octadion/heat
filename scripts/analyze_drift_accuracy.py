from __future__ import annotations

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
    find_json_files,
    group_rows_by_block,
    load_json,
    mean_or_none,
    row_metric,
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
            "stationary_grad_l2": grad_mean,
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
            "stationary_rows": len(stat_rows),
        })
    return points, warnings, method


def plot_scatter(points: list[dict[str, Any]], fits: dict[str, Any], out_path: Path) -> bool:
    if not points:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    xs = [p["stationary_drift_l2"] for p in points]
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

    ax.set_xlabel("stationary drift_l2")
    ax.set_ylabel("error (1 - accuracy)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def build_summary_md(payload: dict[str, Any]) -> str:
    stats = payload["statistics"]
    lines = [
        "# Drift Accuracy",
        "",
        "This analysis builds per-corruption points from existing P9 diagnostic JSON files.",
        "",
        f"- Files matched: {len(payload['files_considered'])}",
        f"- Points analyzed: {len(payload['points'])}",
        f"- Files skipped: {len(payload['skipped_files'])}",
        "",
    ]
    if not payload["points"]:
        lines.extend([
            "No drift-to-error points were available.",
            "",
        ])
    else:
        lines.extend([
            f"- Pearson correlation: {stats.get('pearson')}",
            f"- Spearman correlation: {stats.get('spearman')}",
            f"- Linear fit: {stats.get('fits', {}).get('linear')}",
            f"- Quadratic fit: {stats.get('fits', {}).get('quadratic_x2_only')}",
            "",
        ])

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

    xs = [p["stationary_drift_l2"] for p in points]
    ys = [p["error"] for p in points]
    fits = fit_models(xs, ys)
    stats = {
        "pearson": pearson_correlation(xs, ys),
        "spearman": spearman_correlation(xs, ys),
        "fits": fits,
    }

    plots: list[str] = []
    scatter_path = plot_dir / "drift_accuracy_scatter.png"
    if plot_scatter(points, fits, scatter_path):
        plots.append(str(scatter_path))

    payload = {
        "patterns": DEFAULT_P9_PATTERNS,
        "matches_by_pattern": matches,
        "files_considered": [str(p) for p in files],
        "skipped_files": skipped_files,
        "warnings": warnings,
        "points": points,
        "statistics": stats,
        "plots": plots,
    }
    write_json(analysis_dir / "drift_accuracy.json", payload)
    write_text(analysis_dir / "drift_accuracy_summary.md", build_summary_md(payload))
    print(f"[saved] {analysis_dir / 'drift_accuracy.json'}")
    print(f"[saved] {analysis_dir / 'drift_accuracy_summary.md'}")
    print(f"[plots] {len(plots)}")


if __name__ == "__main__":
    main()
