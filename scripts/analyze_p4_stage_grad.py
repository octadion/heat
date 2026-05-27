from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis_common import ensure_analysis_dirs, safe_float, safe_int, write_text


DEFAULT_INPUT = Path("experiments/results/p4_per_stage_grad_seed42_sev5.json")


def stage_sort_key(stage: str) -> tuple[int, str]:
    if stage == "stem_or_other":
        return (0, stage)
    if stage.startswith("stage"):
        return (1, stage)
    if stage.startswith("block"):
        return (1, stage)
    if stage == "linear":
        return (2, stage)
    return (3, stage)


def iter_diag_rows(payload: dict[str, Any]):
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return
    for method, per_corr in results.items():
        if not isinstance(per_corr, dict):
            continue
        for corruption, entry in per_corr.items():
            if not isinstance(entry, dict):
                continue
            rows = entry.get("diagnostics", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item.setdefault("method", method)
                item.setdefault("corruption", corruption)
                yield item


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        method = str(row.get("method", "unknown"))
        by_method[method].append(row)
        stage_norms = row.get("stage_grad_norms", {})
        if not isinstance(stage_norms, dict):
            continue
        for stage, value in stage_norms.items():
            number = safe_float(value)
            if number is not None:
                by_method_stage[method][stage].append(number)

    summary: dict[str, Any] = {}
    for method, stage_values in by_method_stage.items():
        means = {
            stage: sum(values) / len(values)
            for stage, values in stage_values.items()
            if values
        }
        ordered = dict(sorted(means.items(), key=lambda item: stage_sort_key(item[0])))
        total = sum(ordered.values())
        top_stage = max(ordered, key=ordered.get) if ordered else None
        summary[method] = {
            "num_rows": len(by_method.get(method, [])),
            "mean_stage_grad_norms": ordered,
            "top_stage": top_stage,
            "top_stage_fraction": (ordered[top_stage] / total) if top_stage and total > 0 else None,
        }
    return summary


def plot_method(rows: list[dict[str, Any]], method: str, out_path: Path) -> bool:
    method_rows = [r for r in rows if r.get("method") == method]
    if not method_rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    by_step_stage: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stages: set[str] = set()
    for idx, row in enumerate(method_rows):
        step = safe_int(row.get("local_step"))
        if step is None:
            step = idx
        stage_norms = row.get("stage_grad_norms", {})
        if not isinstance(stage_norms, dict):
            continue
        for stage, value in stage_norms.items():
            number = safe_float(value)
            if number is None:
                continue
            stages.add(stage)
            by_step_stage[step][stage].append(number)

    if not stages or not by_step_stage:
        return False

    steps = sorted(by_step_stage)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for stage in sorted(stages, key=stage_sort_key):
        ys = []
        xs = []
        for step in steps:
            values = by_step_stage[step].get(stage, [])
            if values:
                xs.append(step)
                ys.append(statistics.mean(values))
        if xs:
            ax.plot(xs, ys, linewidth=1.2, label=stage)
    ax.set_xlabel("local step")
    ax.set_ylabel("mean stage grad norm")
    ax.set_title(method)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def build_md(input_path: Path, payload: dict[str, Any] | None, summary: dict[str, Any], plots: list[str]) -> str:
    lines = [
        "# P4 Reconciliation",
        "",
        f"- Input: {input_path}",
    ]
    if payload is None:
        lines.extend([
            "- Status: input JSON not found, so no stage-gradient reconciliation was computed.",
            "",
        ])
        return "\n".join(lines)

    args = payload.get("args", {})
    lines.extend([
        f"- Arch: {args.get('arch')}",
        f"- Seed: {args.get('seed')}",
        f"- Severity: {args.get('severity')}",
        f"- Plots: {len(plots)}",
        "",
    ])
    if not summary:
        lines.extend([
            "No usable `stage_grad_norms` rows were found in the input JSON.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "| method | rows | top_stage | top_stage_fraction | mean_stage_grad_norms |",
        "| --- | ---: | --- | ---: | --- |",
    ])
    for method, info in summary.items():
        lines.append(
            "| {method} | {rows} | {top} | {frac} | {means} |".format(
                method=method,
                rows=info.get("num_rows"),
                top=info.get("top_stage"),
                frac=info.get("top_stage_fraction"),
                means=info.get("mean_stage_grad_norms"),
            )
        )
    lines.extend([
        "",
        "Interpretation note: this is a descriptive diagnostic only. If `heat_singlestage` keeps the same concentration pattern as `heat`, that is consistent with stage-wise concentration arising from full-network gradient flow and architecture rather than multi-stage aggregation alone.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    analysis_dir, plot_dir = ensure_analysis_dirs()
    output_md = analysis_dir / "p4_reconciliation.md"

    if not input_path.exists():
        write_text(output_md, build_md(input_path, None, {}, []))
        print(f"[warn] missing input: {input_path}")
        print(f"[saved] {output_md}")
        return

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = list(iter_diag_rows(payload))
    summary = summarize_rows(rows)
    plots: list[str] = []
    for method in ("heat", "heat_singlestage"):
        out_path = plot_dir / f"p4_stage_grad_{method}.png"
        if plot_method(rows, method, out_path):
            plots.append(str(out_path))

    write_text(output_md, build_md(input_path, payload, summary, plots))
    print(f"[saved] {output_md}")
    print(f"[plots] {len(plots)}")


if __name__ == "__main__":
    main()
