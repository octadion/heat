from __future__ import annotations

import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_P9_PATTERNS = [
    "experiments/results/p9_resnet18*heat*p0*diag*seed*_sev5.json",
    "experiments/results/p9_resnet18*heat*p005*diag*seed*_sev5.json",
    "experiments/results/p9_resnet18*heat*p025*diag*seed*_sev5.json",
    "experiments/results/p9_wrn28_10*diag*seed*_sev5.json",
    "experiments/results/p9*wrn*diag*seed*_sev5.json",
    "experiments/results/p9_*diag*seed*_sev*.json",
    "experiments/results/p9_*heat*seed*_sev*.json",
]


def ensure_analysis_dirs() -> tuple[Path, Path]:
    analysis_dir = Path("experiments") / "analysis"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir, plot_dir


def find_json_files(patterns: list[str]) -> tuple[list[Path], dict[str, list[str]]]:
    matches_by_pattern: dict[str, list[str]] = {}
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in glob.glob(pattern)]
        matched = sorted(matched, key=lambda p: str(p).lower())
        matches_by_pattern[pattern] = [str(p) for p in matched]
        for path in matched:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    files.sort(key=lambda p: str(p).lower())
    return files, matches_by_pattern


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def finite_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            out.append(number)
    return out


def mean_or_none(values: list[Any]) -> float | None:
    vals = finite_values(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def pvariance_or_none(values: list[Any]) -> float | None:
    vals = finite_values(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(statistics.pvariance(vals))


def find_key(obj: Any, key_names: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm_key = str(key).replace("-", "_").lower()
            if norm_key in key_names:
                return value
        for value in obj.values():
            found = find_key(value, key_names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_key(item, key_names)
            if found is not None:
                return found
    return None


def parse_arch_from_name(stem: str) -> str | None:
    lower = stem.lower()
    if "wrn28_10" in lower or "wrn" in lower:
        return "wrn28_10"
    if "resnet18" in lower:
        return "resnet18"
    if "vit_s" in lower or "vits" in lower:
        return "vit_s"
    return None


def parse_seed_from_name(stem: str) -> int | None:
    match = re.search(r"(?:^|[_-])seed(\d+)(?=[_-]|$)", stem.lower())
    return int(match.group(1)) if match else None


def parse_severity_from_name(stem: str) -> int | None:
    match = re.search(r"(?:^|[_-])sev(\d+)(?=[_-]|$)", stem.lower())
    return int(match.group(1)) if match else None


def parse_restore_prob_from_name(stem: str) -> float | None:
    tokens = re.findall(r"(?:^|[_-])p(\d+)(?=[_-]|$)", stem.lower())
    for token in tokens:
        if token == "9":
            continue
        if token == "0":
            return 0.0
        if token.startswith("0"):
            return int(token) / (10 ** len(token))
        value = float(token)
        if 0.0 <= value <= 1.0:
            return value
    return None


def first_row_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def extract_metadata(
    data: dict[str, Any],
    path: Path,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = rows or []
    stem = path.stem

    arch = find_key(data.get("args", data), {"arch"})
    if arch is None:
        arch = parse_arch_from_name(stem)

    seed = find_key(data.get("args", data), {"seed"})
    if seed is None:
        seed = parse_seed_from_name(stem)

    severity = find_key(data.get("args", data), {"severity"})
    if severity is None:
        severity = parse_severity_from_name(stem)

    restore_prob = find_key(
        data.get("args", data),
        {"heat_restore_prob", "restore_prob", "p_restore"},
    )
    if restore_prob is None:
        restore_prob = first_row_value(rows, "restore_prob")
    if restore_prob is None:
        restore_prob = parse_restore_prob_from_name(stem)

    eta = find_key(
        data.get("args", data),
        {"heat_lr", "eta", "learning_rate", "lr"},
    )

    return {
        "file": str(path),
        "arch": str(arch) if arch is not None else None,
        "seed": safe_int(seed),
        "severity": safe_int(severity),
        "restore_prob": safe_float(restore_prob),
        "eta": safe_float(eta),
    }


def result_roots(data: dict[str, Any]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    if isinstance(data.get("results"), dict):
        roots.append(data["results"])
    roots.append(data)
    return roots


def normalize_diag_rows(
    rows: list[Any],
    corruption: str | None = None,
    method: str | None = None,
    step_offset: int = 0,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if item.get("corruption") is None and corruption is not None:
            item["corruption"] = corruption
        if item.get("method") is None and method is not None:
            item["method"] = method
        if item.get("local_step") is None:
            item["local_step"] = idx
        if item.get("step") is None:
            item["step"] = item.get("global_step", step_offset + idx)
        normalized.append(item)
    return normalized


def extract_stream_diagnostics(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for root in result_roots(data):
        summary = root.get("summary")
        if isinstance(summary, dict):
            for method, info in summary.items():
                if not isinstance(info, dict):
                    continue
                rows = info.get("stream_diagnostics")
                if isinstance(rows, list) and rows and method not in by_method:
                    by_method[str(method)] = normalize_diag_rows(rows, method=str(method))

        rows = root.get("stream_diagnostics")
        if isinstance(rows, list) and rows and "unknown" not in by_method:
            by_method["unknown"] = normalize_diag_rows(rows)
    return by_method


def extract_per_corruption_entries(
    data: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for root in result_roots(data):
        per_corruption = root.get("per_corruption")
        if not isinstance(per_corruption, dict):
            continue

        for key, value in per_corruption.items():
            if not isinstance(value, dict):
                continue

            # Shape: per_corruption[method][corruption] = result dict.
            if any(isinstance(v, dict) for v in value.values()):
                method = str(key)
                for corruption, entry in value.items():
                    if isinstance(entry, dict):
                        by_method.setdefault(method, {})[str(corruption)] = entry

            # Shape: per_corruption[corruption] = result dict.
            if "diagnostics" in value or "accuracy" in value or "acc" in value:
                by_method.setdefault("unknown", {})[str(key)] = value
    return by_method


def extract_per_corruption_diagnostics(
    data: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    entries = extract_per_corruption_entries(data)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    step_offset = 0
    for method, corr_entries in entries.items():
        for corruption, entry in corr_entries.items():
            rows = entry.get("diagnostics") if isinstance(entry, dict) else None
            if not isinstance(rows, list) or not rows:
                continue
            normalized = normalize_diag_rows(
                rows,
                corruption=corruption,
                method=method,
                step_offset=step_offset,
            )
            out.setdefault(method, {})[corruption] = normalized
            step_offset += len(normalized)
    return out


def choose_method(methods: list[str], preferred: str = "heat") -> str | None:
    if preferred in methods:
        return preferred
    for method in methods:
        if preferred in method:
            return method
    return methods[0] if methods else None


def diagnostics_for_file(
    data: dict[str, Any],
    preferred_method: str = "heat",
) -> tuple[str | None, list[dict[str, Any]], str]:
    streams = extract_stream_diagnostics(data)
    method = choose_method(list(streams), preferred_method)
    if method is not None:
        return method, streams[method], "results.summary.<method>.stream_diagnostics"

    per_corr = extract_per_corruption_diagnostics(data)
    method = choose_method(list(per_corr), preferred_method)
    if method is not None:
        rows: list[dict[str, Any]] = []
        for corruption in sorted(per_corr[method]):
            rows.extend(per_corr[method][corruption])
        return method, rows, "results.per_corruption.<method>.<corruption>.diagnostics"

    return None, [], "none"


def group_rows_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    block_idx = 0
    previous_local: int | None = None
    for idx, row in enumerate(rows):
        corruption = row.get("corruption")
        local_step = safe_int(row.get("local_step"))
        if corruption is None:
            if previous_local is not None and local_step is not None and local_step < previous_local:
                block_idx += 1
            corruption = f"block_{block_idx}"
        previous_local = local_step
        item = dict(row)
        item.setdefault("_row_index", idx)
        groups.setdefault(str(corruption), []).append(item)
    return groups


def stationary_rows(rows: list[dict[str, Any]], low: int = 50, high: int = 150) -> list[dict[str, Any]]:
    if not rows:
        return []
    with_local: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        local_step = safe_int(row.get("local_step"))
        if local_step is None:
            local_step = idx
        with_local.append((local_step, row))

    max_local = max(local for local, _ in with_local)
    if max_local < high:
        return [row for _, row in with_local]

    selected = [row for local, row in with_local if low <= local <= high]
    return selected if selected else [row for _, row in with_local]


def row_metric(row: dict[str, Any], key: str) -> float | None:
    if key == "grad_l2":
        value = row.get("grad_l2", row.get("total_grad_norm"))
    else:
        value = row.get(key)
    return safe_float(value)


def sorted_rows_for_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        step = safe_int(item.get("step"))
        idx = safe_int(item.get("_row_index"))
        return (step if step is not None else 10**12, idx if idx is not None else 0)

    return sorted(rows, key=sort_key)


def config_id(meta: dict[str, Any], method: str | None = None) -> str:
    path_stem = Path(str(meta.get("file", "config"))).stem
    parts = []
    if meta.get("arch"):
        parts.append(str(meta["arch"]))
    if method:
        parts.append(str(method))
    if meta.get("restore_prob") is not None:
        parts.append(f"p{meta['restore_prob']}")
    if meta.get("seed") is not None:
        parts.append(f"seed{meta['seed']}")
    if not parts:
        parts = [path_stem]
    raw = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def accuracy_from_entry(entry: dict[str, Any]) -> float | None:
    for key in ("accuracy", "acc", "top1", "mean_accuracy"):
        value = safe_float(entry.get(key))
        if value is not None:
            return value
    return None


def accuracy_by_corruption(
    data: dict[str, Any],
    method: str | None,
) -> dict[str, float]:
    entries = extract_per_corruption_entries(data)
    method_key = method if method in entries else choose_method(list(entries), "heat")
    if method_key is None:
        return {}
    out: dict[str, float] = {}
    for corruption, entry in entries.get(method_key, {}).items():
        acc = accuracy_from_entry(entry)
        if acc is not None:
            out[corruption] = acc
    return out


def compute_r2(y_true: list[float], y_pred: list[float]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    if ss_tot == 0:
        return None
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_true, y_pred))
    return float(1.0 - ss_res / ss_tot)
