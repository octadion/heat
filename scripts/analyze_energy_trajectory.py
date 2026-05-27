from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis_common import (
    DEFAULT_P9_PATTERNS,
    config_id,
    diagnostics_for_file,
    ensure_analysis_dirs,
    extract_metadata,
    find_key,
    find_json_files,
    load_json,
    safe_float,
    safe_int,
    sorted_rows_for_plot,
    write_json,
    write_text,
)


def energy_value(row: dict[str, Any]) -> tuple[float | None, bool]:
    raw = row.get("energy")
    if raw is None:
        return None, False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, True
    if not math.isfinite(value):
        return None, True
    return value, False


def classify_monotonic(values: list[float]) -> str | None:
    if len(values) < 2:
        return None
    diffs = [b - a for a, b in zip(values[:-1], values[1:])]
    eps = 1e-12
    if all(d <= eps for d in diffs):
        return "monotonic decreasing"
    if all(d >= -eps for d in diffs):
        return "monotonic increasing"
    return "non-monotonic"


def detect_jumps(values: list[float], steps: list[int]) -> dict[str, Any]:
    if len(values) < 2:
        return {"threshold": None, "median_abs_diff": None, "jumps": []}
    diffs = [b - a for a, b in zip(values[:-1], values[1:])]
    abs_diffs = [abs(d) for d in diffs]
    median_abs_diff = statistics.median(abs_diffs)
    threshold = 10.0 * median_abs_diff
    jumps = []
    for idx, (diff, abs_diff) in enumerate(zip(diffs, abs_diffs)):
        if abs_diff > threshold:
            jumps.append({
                "from_step": steps[idx],
                "to_step": steps[idx + 1],
                "diff": diff,
                "abs_diff": abs_diff,
            })
    return {
        "threshold": threshold,
        "median_abs_diff": median_abs_diff,
        "jumps": jumps,
    }


def summarize_energy(
    meta: dict[str, Any],
    method: str,
    rows: list[dict[str, Any]],
    layout: str,
) -> dict[str, Any]:
    sorted_rows = sorted_rows_for_plot(rows)
    values: list[float] = []
    steps: list[int] = []
    invalid_positions: list[dict[str, Any]] = []

    for idx, row in enumerate(sorted_rows):
        value, invalid = energy_value(row)
        if invalid:
            invalid_positions.append({
                "row_index": idx,
                "step": row.get("step"),
                "value": row.get("energy"),
            })
            continue
        if value is None:
            continue
        step = safe_int(row.get("step"))
        values.append(value)
        steps.append(step if step is not None else idx)

    if values:
        mean = sum(values) / len(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
    else:
        mean = std = min_value = max_value = None

    jumps = detect_jumps(values, steps)
    return {
        **meta,
        "method": method,
        "diagnostic_layout": layout,
        "num_diagnostic_rows": len(rows),
        "num_energy_values": len(values),
        "energy_min": min_value,
        "energy_max": max_value,
        "energy_mean": mean,
        "energy_std": std,
        "monotonicity": classify_monotonic(values),
        "has_nan_or_inf": bool(invalid_positions),
        "invalid_energy_positions": invalid_positions,
        "jump_threshold": jumps["threshold"],
        "median_abs_diff": jumps["median_abs_diff"],
        "extreme_jumps": jumps["jumps"],
    }


def summarize_stability_group(configs: list[dict[str, Any]]) -> dict[str, Any]:
    energy_means = [
        cfg["energy_mean"]
        for cfg in configs
        if cfg.get("energy_mean") is not None
    ]
    jump_counts = [len(cfg.get("extreme_jumps", [])) for cfg in configs]
    return {
        "num_configs": len(configs),
        "files": [cfg.get("file") for cfg in configs],
        "mean_energy_mean": (sum(energy_means) / len(energy_means)) if energy_means else None,
        "total_extreme_jumps": sum(jump_counts),
    }


def stability_summary(configs: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [cfg for cfg in configs if not cfg.get("collapse")]
    collapsed = [cfg for cfg in configs if cfg.get("collapse")]
    return {
        "stable_configs": summarize_stability_group(stable),
        "collapsed_configs": summarize_stability_group(collapsed),
    }


def plot_energy(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    method: str,
    out_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    sorted_rows = sorted_rows_for_plot(rows)
    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(sorted_rows):
        value = safe_float(row.get("energy"))
        if value is None:
            continue
        step = safe_int(row.get("step"))
        xs.append(step if step is not None else idx)
        ys.append(value)
    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, ys, linewidth=1.4)
    ax.set_xlabel("global step")
    ax.set_ylabel("energy")
    ax.set_title(config_id(meta, method))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def build_summary_md(payload: dict[str, Any]) -> str:
    configs = payload["configs"]
    lines = [
        "# Energy Trajectory",
        "",
        "This descriptive analysis uses observed energy values from existing P9 diagnostic JSON files.",
        "",
        f"- Files matched: {len(payload['files_considered'])}",
        f"- Configs analyzed: {len(configs)}",
        f"- Files skipped: {len(payload['skipped_files'])}",
        "",
    ]
    if not configs:
        lines.extend([
            "No usable energy diagnostics were found.",
            "",
        ])
    else:
        lines.extend([
            "| file | arch | seed | p | n | min | max | mean | std | monotonicity | jumps | invalid | collapse |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
        ])
        for cfg in configs:
            lines.append(
                "| {file} | {arch} | {seed} | {p} | {n} | {mn} | {mx} | {mean} | {std} | {mono} | {jumps} | {invalid} | {collapse} |".format(
                    file=Path(cfg["file"]).name,
                    arch=cfg.get("arch"),
                    seed=cfg.get("seed"),
                    p=cfg.get("restore_prob"),
                    n=cfg.get("num_energy_values"),
                    mn=cfg.get("energy_min"),
                    mx=cfg.get("energy_max"),
                    mean=cfg.get("energy_mean"),
                    std=cfg.get("energy_std"),
                    mono=cfg.get("monotonicity"),
                    jumps=len(cfg.get("extreme_jumps", [])),
                    invalid=cfg.get("invalid") or cfg.get("has_nan_or_inf"),
                    collapse=cfg.get("collapse"),
                )
            )
        lines.append("")

    stability = payload.get("stability_summary", {})
    lines.extend([
        "## Stable vs Collapsed",
        "",
        "| group | configs | mean_energy_mean | total_extreme_jumps |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ("stable_configs", "collapsed_configs"):
        info = stability.get(name, {})
        lines.append(
            "| {name} | {n} | {mean} | {jumps} |".format(
                name=name,
                n=info.get("num_configs"),
                mean=info.get("mean_energy_mean"),
                jumps=info.get("total_extreme_jumps"),
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
    analysis_dir, plot_dir = ensure_analysis_dirs()
    files, matches = find_json_files(DEFAULT_P9_PATTERNS)
    warnings: list[str] = []
    skipped_files: list[dict[str, str]] = []
    configs: list[dict[str, Any]] = []
    plots: list[str] = []

    if not files:
        warnings.append("No files matched the P9 diagnostic glob patterns.")

    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": f"json_load_failed: {exc}"})
            continue

        method, rows, layout = diagnostics_for_file(data, preferred_method="heat")
        if not rows or method is None:
            skipped_files.append({"file": str(path), "reason": "no stream or per-corruption diagnostics"})
            continue

        has_energy = any(row.get("energy") is not None for row in rows)
        if not has_energy:
            skipped_files.append({"file": str(path), "reason": "diagnostics contain no energy field"})
            continue

        meta = extract_metadata(data, path, rows)
        cfg = summarize_energy(meta, method, rows, layout)
        invalid_flag = bool(find_key(data, {"invalid"})) or any(
            bool(row.get("invalid")) for row in rows
        )
        cfg["invalid"] = invalid_flag
        cfg["collapse"] = bool(
            invalid_flag
            or cfg.get("has_nan_or_inf")
            or len(cfg.get("extreme_jumps", [])) > 0
        )
        cfg["stability_label"] = "collapsed" if cfg["collapse"] else "stable"
        configs.append(cfg)

        plot_path = plot_dir / f"energy_{config_id(meta, method)}.png"
        if plot_energy(rows, meta, method, plot_path):
            plots.append(str(plot_path))

    payload = {
        "patterns": DEFAULT_P9_PATTERNS,
        "matches_by_pattern": matches,
        "files_considered": [str(p) for p in files],
        "skipped_files": skipped_files,
        "warnings": warnings,
        "configs": configs,
        "stability_summary": stability_summary(configs),
        "plots": plots,
    }
    write_json(analysis_dir / "energy_trajectory.json", payload)
    write_text(analysis_dir / "energy_trajectory_summary.md", build_summary_md(payload))
    print(f"[saved] {analysis_dir / 'energy_trajectory.json'}")
    print(f"[saved] {analysis_dir / 'energy_trajectory_summary.md'}")
    print(f"[plots] {len(plots)}")


if __name__ == "__main__":
    main()
