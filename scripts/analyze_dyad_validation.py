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
    find_json_files,
    group_rows_by_block,
    load_json,
    mean_or_none,
    pvariance_or_none,
    row_metric,
    safe_int,
    sorted_rows_for_plot,
    stationary_rows,
    write_json,
    write_text,
)


def infer_steps_per_corruption(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_group: dict[str, int] = {}
    for corruption, rows in groups.items():
        locals_seen = [safe_int(r.get("local_step")) for r in rows]
        locals_seen = [v for v in locals_seen if v is not None]
        if locals_seen:
            per_group[corruption] = max(locals_seen) + 1
        else:
            per_group[corruption] = len(rows)
    values = list(per_group.values())
    if not values:
        return {"N": None, "per_corruption_steps": per_group}
    return {
        "N": int(round(statistics.median(values))),
        "per_corruption_steps": per_group,
    }


def compute_config_summary(
    meta: dict[str, Any],
    method: str,
    rows: list[dict[str, Any]],
    layout: str,
) -> dict[str, Any]:
    groups = group_rows_by_block(rows)
    n_info = infer_steps_per_corruption(groups)

    per_corruption: dict[str, dict[str, Any]] = {}
    drift_means: list[float] = []
    grad_means: list[float] = []
    grad_vars: list[float] = []

    for corruption, group_rows in groups.items():
        stat_rows = stationary_rows(group_rows)
        drift_values = [row_metric(row, "drift_l2") for row in stat_rows]
        grad_values = [row_metric(row, "grad_l2") for row in stat_rows]
        drift_mean = mean_or_none(drift_values)
        grad_mean = mean_or_none(grad_values)
        grad_var = pvariance_or_none(grad_values)

        if drift_mean is not None:
            drift_means.append(drift_mean)
        if grad_mean is not None:
            grad_means.append(grad_mean)
        if grad_var is not None:
            grad_vars.append(grad_var)

        per_corruption[corruption] = {
            "num_rows": len(group_rows),
            "stationary_rows": len(stat_rows),
            "stationary_window": "local_step in [50,150], or all rows if shorter",
            "mean_drift_l2": drift_mean,
            "mean_grad_l2": grad_mean,
            "within_block_grad_l2_variance": grad_var,
        }

    empirical_mu = mean_or_none(drift_means)
    mean_grad = mean_or_none(grad_means)
    within_block_variance = mean_or_none(grad_vars)
    between_block_variance = pvariance_or_none(grad_means)

    p = meta.get("restore_prob")
    eta = meta.get("eta")
    n_steps = n_info["N"]
    alpha = None
    theoretical_mu_proxy = None
    deviation_proxy_pct = None
    if p is not None and p > 0 and n_steps is not None:
        alpha = (1.0 - p) ** n_steps
    if p is not None and p > 0 and eta is not None and mean_grad is not None:
        theoretical_mu_proxy = ((1.0 - p) * eta / p) * mean_grad
        if theoretical_mu_proxy != 0 and empirical_mu is not None:
            deviation_proxy_pct = 100.0 * (
                empirical_mu - theoretical_mu_proxy
            ) / theoretical_mu_proxy

    return {
        **meta,
        "method": method,
        "diagnostic_layout": layout,
        "num_diagnostic_rows": len(rows),
        "num_corruption_blocks": len(groups),
        "N": n_steps,
        "per_corruption_steps": n_info["per_corruption_steps"],
        "alpha": alpha,
        "per_corruption": per_corruption,
        "within_block_grad_l2_variance_mean": within_block_variance,
        "between_block_grad_l2_mean_variance": between_block_variance,
        "empirical_mu": empirical_mu,
        "mean_grad": mean_grad,
        "theoretical_mu_proxy": theoretical_mu_proxy,
        "deviation_proxy_pct": deviation_proxy_pct,
    }


def plot_metric(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    method: str,
    metric: str,
    ylabel: str,
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
        value = row_metric(row, metric)
        if value is None:
            continue
        step = safe_int(row.get("step"))
        xs.append(step if step is not None else idx)
        ys.append(value)
    if not xs:
        return False

    groups = group_rows_by_block(sorted_rows)
    boundaries: list[tuple[str, int]] = []
    for corruption, group in groups.items():
        first = group[0]
        step = safe_int(first.get("step"))
        if step is None:
            step = safe_int(first.get("_row_index"))
        if step is not None:
            boundaries.append((corruption, step))
    boundaries = sorted(boundaries, key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, ys, linewidth=1.4)
    for corruption, step in boundaries[1:]:
        ax.axvline(step, color="0.75", linewidth=0.8, linestyle="--")
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
    ax.set_title(config_id(meta, method))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_empirical_vs_theoretical(configs: list[dict[str, Any]], out_path: Path) -> bool:
    points = [
        c for c in configs
        if c.get("restore_prob") is not None
        and c["restore_prob"] > 0
        and c.get("empirical_mu") is not None
        and c.get("theoretical_mu_proxy") is not None
        and math.isfinite(c["empirical_mu"])
        and math.isfinite(c["theoretical_mu_proxy"])
    ]
    if not points:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = [p["theoretical_mu_proxy"] for p in points]
    ys = [p["empirical_mu"] for p in points]
    lo = min(xs + ys)
    hi = max(xs + ys)
    if lo == hi:
        lo *= 0.9
        hi *= 1.1 if hi != 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(xs, ys, s=46)
    ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0)
    for point, x, y in zip(points, xs, ys):
        label = config_id(point, point.get("method"))
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("theoretical stationary drift proxy")
    ax.set_ylabel("empirical stationary drift")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def close_prob(value: Any, target: float, tol: float = 1e-9) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return abs(number - target) <= tol


def arch_matches(value: Any, target: str) -> bool:
    if value is None:
        return False
    value = str(value).lower()
    if target == "wrn28_10":
        return value == "wrn28_10" or "wrn" in value
    return value == target


def mean_empirical_mu(configs: list[dict[str, Any]], arch: str, p: float) -> float | None:
    values = [
        cfg["empirical_mu"]
        for cfg in configs
        if arch_matches(cfg.get("arch"), arch)
        and close_prob(cfg.get("restore_prob"), p)
        and cfg.get("empirical_mu") is not None
        and math.isfinite(cfg["empirical_mu"])
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def ratio_test(
    configs: list[dict[str, Any]],
    name: str,
    arch: str,
    p_num: float,
    p_den: float,
) -> dict[str, Any]:
    empirical_num = mean_empirical_mu(configs, arch, p_num)
    empirical_den = mean_empirical_mu(configs, arch, p_den)
    empirical_ratio = None
    if empirical_num is not None and empirical_den not in (None, 0.0):
        empirical_ratio = empirical_num / empirical_den
    theory_ratio = ((1.0 - p_num) / p_num) / ((1.0 - p_den) / p_den)
    return {
        "name": name,
        "arch": arch,
        "p_numerator": p_num,
        "p_denominator": p_den,
        "empirical_mu_numerator_mean": empirical_num,
        "empirical_mu_denominator_mean": empirical_den,
        "empirical_ratio": empirical_ratio,
        "theory_ratio": theory_ratio,
        "available": empirical_ratio is not None,
    }


def compute_ratio_tests(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        ratio_test(configs, "resnet18_p005_vs_p025", "resnet18", 0.005, 0.025),
        ratio_test(configs, "wrn28_10_p01_vs_p02", "wrn28_10", 0.01, 0.02),
    ]


def build_summary_md(payload: dict[str, Any]) -> str:
    configs = payload["configs"]
    lines = [
        "# Dyad Validation",
        "",
        "This descriptive analysis was computed from existing P9 diagnostic JSON files only.",
        "",
        "The proxy uses E[||g||], while Theorem 4.5 requires ||E[g]||. Therefore this is an upper-proxy, not the exact theoretical mean drift.",
        "",
        f"- Files matched: {len(payload['files_considered'])}",
        f"- Configs analyzed: {len(configs)}",
        f"- Files skipped: {len(payload['skipped_files'])}",
        "",
    ]
    if not configs:
        lines.extend([
            "No usable diagnostic rows were found. The script still wrote this summary so the absence is explicit.",
            "",
        ])
    else:
        lines.extend([
            "| file | arch | seed | p | eta | N | empirical_mu | theoretical_mu_proxy | deviation_proxy_pct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for cfg in configs:
            lines.append(
                "| {file} | {arch} | {seed} | {p} | {eta} | {N} | {emu} | {tmu} | {dev} |".format(
                    file=Path(cfg["file"]).name,
                    arch=cfg.get("arch"),
                    seed=cfg.get("seed"),
                    p=cfg.get("restore_prob"),
                    eta=cfg.get("eta"),
                    N=cfg.get("N"),
                    emu=cfg.get("empirical_mu"),
                    tmu=cfg.get("theoretical_mu_proxy"),
                    dev=cfg.get("deviation_proxy_pct"),
                )
            )
        lines.append("")

    lines.extend(["## Ratio Tests", ""])
    ratio_tests = payload.get("ratio_tests", [])
    if not ratio_tests:
        lines.extend(["No ratio tests were computed.", ""])
    else:
        lines.extend([
            "| test | empirical_ratio | theory_ratio | available |",
            "| --- | ---: | ---: | --- |",
        ])
        for test in ratio_tests:
            lines.append(
                "| {name} | {emp} | {theory} | {available} |".format(
                    name=test.get("name"),
                    emp=test.get("empirical_ratio"),
                    theory=test.get("theory_ratio"),
                    available=test.get("available"),
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

        meta = extract_metadata(data, path, rows)
        cfg = compute_config_summary(meta, method, rows, layout)
        configs.append(cfg)

        cid = config_id(meta, method)
        drift_plot = plot_dir / f"dyad_drift_{cid}.png"
        grad_plot = plot_dir / f"dyad_grad_{cid}.png"
        if plot_metric(rows, meta, method, "drift_l2", "drift_l2", drift_plot):
            plots.append(str(drift_plot))
        if plot_metric(rows, meta, method, "grad_l2", "grad_l2", grad_plot):
            plots.append(str(grad_plot))

    theory_plot = plot_dir / "dyad_empirical_vs_theoretical.png"
    if plot_empirical_vs_theoretical(configs, theory_plot):
        plots.append(str(theory_plot))

    ratio_tests = compute_ratio_tests(configs)
    payload = {
        "patterns": DEFAULT_P9_PATTERNS,
        "matches_by_pattern": matches,
        "files_considered": [str(p) for p in files],
        "skipped_files": skipped_files,
        "warnings": warnings,
        "configs": configs,
        "ratio_tests": ratio_tests,
        "plots": plots,
    }
    write_json(analysis_dir / "dyad_validation.json", payload)
    write_text(analysis_dir / "dyad_validation_summary.md", build_summary_md(payload))
    print(f"[saved] {analysis_dir / 'dyad_validation.json'}")
    print(f"[saved] {analysis_dir / 'dyad_validation_summary.md'}")
    print(f"[plots] {len(plots)}")


if __name__ == "__main__":
    main()
