"""Inventory existing artifacts for review-response experiment gaps.

This script only inspects files. It does not import experiment runners, load
models, touch datasets, or launch experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path("experiments") / "analysis"
JSON_OUT = ANALYSIS_DIR / "review_gap_inventory.json"
MD_OUT = ANALYSIS_DIR / "review_gap_inventory.md"


ITEMS = [
    {
        "item": "EATA baseline on CIFAR-10-C continual",
        "needles": ["eata"],
        "result_needles": ["eata", "cifar10", "p9"],
        "action": "Run EATA P9 only after Fisher-source stats are declared.",
    },
    {
        "item": "SAR baseline on CIFAR-10-C continual",
        "needles": ["sar"],
        "result_needles": ["sar", "cifar10", "p9"],
        "action": "Run SAR P9 with legacy stream/order and report runtime.",
    },
    {
        "item": "CoTTA baseline on CIFAR-10-C continual",
        "needles": ["cotta"],
        "result_needles": ["cotta", "cifar10", "p9"],
        "action": "Do not run a fake CoTTA; port full official method first.",
    },
    {
        "item": "RDumb / periodic reset baseline",
        "needles": ["rdumb", "periodic reset", "reset_interval"],
        "result_needles": ["rdumb", "p9"],
        "action": "Run RDumb R=157 and R=50 as separate P9 files.",
    },
    {
        "item": "BN-only vs full-parameter TFF",
        "needles": ["bn_affine_only", "bn-only", "adapt-params"],
        "result_needles": ["bn_affine_only", "heat", "p9"],
        "action": "Run TFF BN-affine-only ablation under a labeled variant tag.",
    },
    {
        "item": "TFF with BN running stats frozen",
        "needles": ["bn_running_stats", "bn-running-stats", "stats frozen"],
        "result_needles": ["bn_running_stats", "frozen", "heat", "p9"],
        "action": "Run TFF full-parameter with BN running stats frozen.",
    },
    {
        "item": "Batch-size sensitivity",
        "needles": ["batch size", "batch-size", "bs8", "bs1"],
        "result_needles": ["bs8", "p9"],
        "action": "Run bs8 first; bs1 is optional and should be labeled separate.",
    },
    {
        "item": "TEA SGLD sensitivity / hyperparameter sweep",
        "needles": ["tea-sgld-steps", "sgld"],
        "result_needles": ["tea", "sgld", "steps"],
        "action": "Run 4-corruption single-shift TEA steps 1/5/10 plus default.",
    },
    {
        "item": "ViT LR search results",
        "needles": ["vit_s", "vit", "lr-grid"],
        "result_needles": ["vit_s", "lr"],
        "action": "Run limited ViT LR/p grids only; do not launch full matrix.",
    },
    {
        "item": "CIFAR-100-C or ImageNet-C results",
        "needles": ["cifar100", "cifar-100-c", "imagenet-c"],
        "result_needles": ["cifar100"],
        "action": "CIFAR-100-C commands are safe only if data and checkpoint exist.",
    },
]


def _text_files() -> list[Path]:
    roots = [Path("experiments"), Path("scripts"), Path("src")]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".py", ".md", ".json", ".yaml", ".yml", ".txt"
            }:
                files.append(path)
    if Path("README.md").exists():
        files.append(Path("README.md"))
    return sorted(files, key=lambda p: str(p).lower())


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _flatten_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True).lower()
    except TypeError:
        return str(obj).lower()


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _result_files() -> list[Path]:
    roots = [Path("experiments") / "results", Path("experiments") / "analysis"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob("*.json")))
    return sorted(files, key=lambda p: str(p).lower())


def _first_evidence(needles: list[str], files: list[Path]) -> str:
    lowered_needles = [n.lower() for n in needles]
    for path in files:
        text = _safe_read(path).lower()
        if any(needle in text for needle in lowered_needles):
            return str(path)
    return ""


def _result_evidence(needles: list[str], files: list[Path]) -> str:
    lowered_needles = [n.lower() for n in needles]
    for path in files:
        data = _load_json(path)
        text = _flatten_json(data) if data is not None else _safe_read(path).lower()
        if all(needle in text for needle in lowered_needles):
            return str(path)
    return ""


def build_inventory() -> list[dict[str, str]]:
    files = _text_files()
    result_files = _result_files()
    rows: list[dict[str, str]] = []
    for spec in ITEMS:
        result_path = _result_evidence(spec["result_needles"], result_files)
        support_path = _first_evidence(spec["needles"], files)
        usable = "yes" if result_path else "no"
        exists = "yes" if result_path else "no"
        source = result_path or support_path or "not found"
        action = "Use existing result after metric audit." if result_path else spec["action"]
        rows.append({
            "item": spec["item"],
            "exists?": exists,
            "source file/path": source,
            "usable for paper?": usable,
            "recommended action": action,
        })
    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    headers = [
        "item", "exists?", "source file/path",
        "usable for paper?", "recommended action",
    ]
    lines = [
        "# Review Gap Inventory",
        "",
        "Generated by `scripts/inventory_review_gaps.py`. "
        "This inventory inspects local files only and does not launch experiments.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row[h].replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_inventory()
    write_outputs(rows)
    print(f"[saved] {JSON_OUT}")
    print(f"[saved] {MD_OUT}")


if __name__ == "__main__":
    main()

