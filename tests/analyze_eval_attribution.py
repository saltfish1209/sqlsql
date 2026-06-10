"""
对 evaluate_llm_full_schema_vs_system_plan 的明细 CSV 做分层归因。

目的：定位主系统各模块（实体提取、refiner 等）对准确率的影响。
- 交叉表：两路方案的对错四象限（谁回归、谁增益）
- recall 分桶：每条路径在「召回满 (recall>=1)」与「召回不足 (recall<1)」下的准确率
- 回归清单：base 对、other 错 的题目（附两路 recall，区分 schema 问题 vs 生成问题）

用法：
    python tests/analyze_eval_attribution.py --detail tests/results/full_schema_vs_system_plan_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PATHS = [
    ("full_schema", "全量schema+llm"),
    ("cliff_schema", "断崖截取后schema+llm"),
    ("main_system", "主系统"),
    ("main_no_entity", "主系统(无实体提取)"),
    ("main_no_refiner", "主系统(无refiner)"),
]

_RECALL_KEY = {
    "full_schema": "full_schema_recall",
    "cliff_schema": "cliff_schema_recall",
    "main_system": "main_system_schema_recall",
    "main_no_entity": "main_no_entity_schema_recall",
    "main_no_refiner": "main_no_refiner_schema_recall",
}

# 明细 CSV 可能来自旧版评测（无新列）；缺失列时按 0 / None 处理，不报错。
_OPTIONAL_PATHS = ("main_no_entity", "main_no_refiner")


def _has_path_columns(row: dict, prefix: str) -> bool:
    return f"{prefix}_correct" in row


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_recall(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_correct(row: dict, prefix: str) -> bool:
    if prefix in _OPTIONAL_PATHS and not _has_path_columns(row, prefix):
        return False
    return _to_int(row.get(f"{prefix}_correct")) == 1


def _recall(row: dict, prefix: str) -> float | None:
    if prefix in _OPTIONAL_PATHS and _RECALL_KEY[prefix] not in row:
        return None
    return _to_recall(row.get(_RECALL_KEY[prefix]))


def build_pairwise(rows: list[dict], base_prefix: str, other_prefix: str) -> dict:
    """base vs other 的对错四象限计数。base_only = base 对、other 错（回归）。"""
    tab = {"both_correct": 0, "base_only": 0, "other_only": 0, "both_wrong": 0, "total": 0}
    for row in rows:
        base_ok = _is_correct(row, base_prefix)
        other_ok = _is_correct(row, other_prefix)
        tab["total"] += 1
        if base_ok and other_ok:
            tab["both_correct"] += 1
        elif base_ok and not other_ok:
            tab["base_only"] += 1
        elif other_ok and not base_ok:
            tab["other_only"] += 1
        else:
            tab["both_wrong"] += 1
    return tab


def build_recall_buckets(rows: list[dict], prefix: str) -> dict:
    """按 recall 是否满分分桶，统计每桶准确率。"""
    buckets = {
        "full_recall": {"total": 0, "correct": 0},
        "partial_recall": {"total": 0, "correct": 0},
        "unknown_recall": {"total": 0, "correct": 0},
    }
    for row in rows:
        recall = _recall(row, prefix)
        if recall is None:
            key = "unknown_recall"
        elif recall >= 1.0:
            key = "full_recall"
        else:
            key = "partial_recall"
        buckets[key]["total"] += 1
        if _is_correct(row, prefix):
            buckets[key]["correct"] += 1
    for stats in buckets.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 6) if stats["total"] else 0.0
    return buckets


def collect_regressions(rows: list[dict], *, base_prefix: str, other_prefix: str) -> list[dict]:
    """base 对、other 错 的题目；附两路 recall，便于区分 schema 问题与生成问题。"""
    out: list[dict] = []
    for row in rows:
        if _is_correct(row, base_prefix) and not _is_correct(row, other_prefix):
            out.append(
                {
                    "idx": row.get("idx"),
                    "question": row.get("question", ""),
                    "base_recall": _recall(row, base_prefix),
                    "other_recall": _recall(row, other_prefix),
                    f"{base_prefix}_sql": row.get(f"{base_prefix}_sql", ""),
                    f"{other_prefix}_sql": row.get(f"{other_prefix}_sql", ""),
                }
            )
    return out


def _load_detail(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _available_paths(rows: list[dict]) -> list[str]:
    if not rows:
        return [prefix for prefix, _ in PATHS]
    sample = rows[0]
    return [prefix for prefix, _ in PATHS if prefix not in _OPTIONAL_PATHS or _has_path_columns(sample, prefix)]


def build_report(rows: list[dict]) -> dict:
    paths = _available_paths(rows)
    report: dict = {
        "total": len(rows),
        "paths_in_detail": paths,
        "recall_buckets": {prefix: build_recall_buckets(rows, prefix) for prefix in paths},
        "cliff_vs_main": build_pairwise(rows, "cliff_schema", "main_system"),
        "full_vs_main": build_pairwise(rows, "full_schema", "main_system"),
        "main_regressions_vs_cliff": collect_regressions(
            rows, base_prefix="cliff_schema", other_prefix="main_system"
        ),
    }
    if "main_no_entity" in paths:
        report["main_vs_no_entity"] = build_pairwise(rows, "main_system", "main_no_entity")
        report["entity_extraction_helps"] = collect_regressions(
            rows, base_prefix="main_system", other_prefix="main_no_entity"
        )
        report["entity_extraction_hurts"] = collect_regressions(
            rows, base_prefix="main_no_entity", other_prefix="main_system"
        )
    if "main_no_refiner" in paths:
        report["main_vs_no_refiner"] = build_pairwise(rows, "main_system", "main_no_refiner")
        report["refiner_helps"] = collect_regressions(
            rows, base_prefix="main_system", other_prefix="main_no_refiner"
        )
        report["refiner_hurts"] = collect_regressions(
            rows, base_prefix="main_no_refiner", other_prefix="main_system"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评测明细 CSV 分层归因")
    parser.add_argument(
        "--detail",
        default=str(Path(__file__).resolve().parent / "results" / "full_schema_vs_system_plan_test.csv"),
        help="明细 CSV 路径",
    )
    parser.add_argument("--output", default="", help="归因 JSON 输出路径（可选）")
    args = parser.parse_args()

    rows = _load_detail(Path(args.detail))
    report = build_report(rows)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"归因 JSON: {args.output}")
    print(text)


if __name__ == "__main__":
    main()
