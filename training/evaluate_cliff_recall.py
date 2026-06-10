"""
评估 val/test 上 CrossEncoder + 断崖筛选的召回表现。

默认输出透视表 CSV（纵轴：召回/冗余/保留列数/退化数；横轴：纯cliff、cliff+protect、otus、otus+protect），并打印 JSON 摘要到 stdout：
1. protect_only：protect_ratio（可 auto-select）+ 最大跌幅断崖
2. max_cliff_only：TopK 内纯最大相对跌幅断崖
3. local_adaptive_only：TopK 内裸 Otsu
4. protect_otsu：protect_ratio 保护区 + 尾段 Otsu
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.cross_encoder_passage import build_column_passage_map
from training.calibrate_cliff_coefficients import _estimate_params_for_case
from training.cliff_param_selection import local_adaptive_split, select_dual_thresholds
from training.cross_encoder_eval import load_eval_records


STAGE_NAMES = ("topk", "min_filtered", "final", "entity_added")


def _recall_ratio(gold: set[str], pred: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & pred) / len(gold)


def _merge_columns(base: list[str], additions: list[str]) -> list[str]:
    """Preserve base order and append evidence columns once."""
    merged: list[str] = []
    seen: set[str] = set()
    for col in list(base) + list(additions):
        col = str(col or "").strip()
        if not col or col in seen:
            continue
        seen.add(col)
        merged.append(col)
    return merged


def _column_metrics(gold: set[str], cols: list[str]) -> dict:
    selected = [str(c).strip() for c in cols if str(c).strip()]
    selected_set = set(selected)
    hit_count = len(gold & selected_set)
    selected_count = len(selected_set)
    wrong_count = max(0, selected_count - hit_count)
    return {
        "recall": round(_recall_ratio(gold, selected_set), 6),
        "full_recall": int(bool(gold) and gold.issubset(selected_set)),
        "selected_count": selected_count,
        "gold_hit_count": hit_count,
        "wrong_count": wrong_count,
        "precision": round(hit_count / selected_count, 6) if selected_count else 0.0,
    }


def _empty_stage_sums() -> dict[str, dict[str, float]]:
    return {
        stage: {
            "recall": 0.0,
            "full_recall": 0.0,
            "selected_count": 0.0,
            "gold_hit_count": 0.0,
            "wrong_count": 0.0,
            "precision": 0.0,
        }
        for stage in STAGE_NAMES
    }


def _add_stage_metrics(sums: dict[str, dict[str, float]], stage: str, metrics: dict) -> None:
    for key in sums[stage]:
        sums[stage][key] += float(metrics[key])


def _average_stage_metrics(sums: dict[str, dict[str, float]], total: int) -> dict:
    if total <= 0:
        return {stage: {key: 0.0 for key in sums[stage]} for stage in sums}
    out: dict[str, dict[str, float]] = {}
    for stage, metrics in sums.items():
        out[stage] = {
            key: round(value / total, 6)
            for key, value in metrics.items()
        }
    return out


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _filter_with_cliff_details(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    min_ratio: float,
    protect_ratio: float,
) -> dict:
    """
    返回筛选结果和断崖诊断细节：
    - topk_cols: 纯 TopK
    - min_filtered_cols: TopK 后做 min_ratio 截断
    - final_cols: 在 min 结果上继续做保护区+断崖截断
    """
    topk_cols = ranked_cols[:top_k]
    topk_scores = ranked_scores[:top_k]
    details = {
        "topk_columns": topk_cols,
        "topk_scores": topk_scores,
        "min_filtered_columns": [],
        "min_filtered_scores": [],
        "final_columns": [],
        "final_scores": [],
        "best_score": None,
        "min_score": None,
        "protect_score": None,
        "cutoff": 0,
        "cliff_event": None,
        "params": {
            "top_k": top_k,
            "min_ratio": float(min_ratio),
            "protect_ratio": float(protect_ratio),
        },
    }
    if not topk_cols:
        return details

    best = topk_scores[0]
    details["best_score"] = _round_metric(best)
    if best <= 0:
        details.update(
            {
                "min_filtered_columns": topk_cols,
                "min_filtered_scores": topk_scores,
                "final_columns": topk_cols,
                "final_scores": topk_scores,
                "cutoff": len(topk_cols),
            }
        )
        return details

    min_score = best * float(min_ratio)
    protect_score = best * float(protect_ratio)
    details["min_score"] = _round_metric(min_score)
    details["protect_score"] = _round_metric(protect_score)

    min_filtered_pairs = [
        (c, s) for c, s in zip(topk_cols, topk_scores) if float(s) >= min_score
    ]
    if not min_filtered_pairs:
        min_filtered_pairs = list(zip(topk_cols, topk_scores))

    min_filtered_cols = [c for c, _ in min_filtered_pairs]
    min_filtered_scores = [s for _, s in min_filtered_pairs]
    details["min_filtered_columns"] = min_filtered_cols
    details["min_filtered_scores"] = [_round_metric(s) for s in min_filtered_scores]

    protect_cut_idx = len(min_filtered_scores)
    for i, score in enumerate(min_filtered_scores):
        if float(score) < protect_score:
            protect_cut_idx = i
            break

    cutoff = len(min_filtered_cols)
    # Protected prefix is preserved; cliff search includes the protect boundary pair.
    search_start = max(protect_cut_idx - 1, 0)
    if len(min_filtered_scores) - search_start >= 2:
        cliff_idx = search_start
        max_decay = -1.0
        for i in range(search_start, len(min_filtered_scores) - 1):
            cur = float(min_filtered_scores[i])
            nxt = float(min_filtered_scores[i + 1])
            if cur <= 0:
                continue
            decay = (cur - nxt) / cur
            if decay > max_decay:
                max_decay = decay
                cliff_idx = i
        cur = float(min_filtered_scores[cliff_idx])
        nxt = float(min_filtered_scores[cliff_idx + 1])
        cutoff = cliff_idx + 1
        details["cliff_event"] = {
            "index": cliff_idx,
            "keep_count": cutoff,
            "kept_column": min_filtered_cols[cliff_idx],
            "first_removed_column": min_filtered_cols[cliff_idx + 1],
            "kept_score": _round_metric(cur),
            "first_removed_score": _round_metric(nxt),
            "kept_score_ratio": _round_metric(cur / best),
            "first_removed_score_ratio": _round_metric(nxt / best),
            "protect_score": _round_metric(protect_score),
            "decay": _round_metric(max_decay),
            "both_in_protect_zone": bool(cur >= protect_score and nxt >= protect_score),
            "either_below_protect_zone": bool(cur < protect_score or nxt < protect_score),
        }

    final_cols = min_filtered_cols[:cutoff]
    if not final_cols:
        final_cols = min_filtered_cols
        cutoff = len(min_filtered_cols)
    details["cutoff"] = cutoff
    details["final_columns"] = final_cols
    details["final_scores"] = [
        _round_metric(s) for s in min_filtered_scores[: len(final_cols)]
    ]
    return details


def _filter_with_cliff(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    min_ratio: float,
    protect_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    details = _filter_with_cliff_details(
        ranked_cols,
        ranked_scores,
        top_k=top_k,
        min_ratio=min_ratio,
        protect_ratio=protect_ratio,
    )
    return (
        details["topk_columns"],
        details["min_filtered_columns"],
        details["final_columns"],
    )


def _filter_protect_only(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    protect_ratio: float,
) -> dict:
    """仅 protect_ratio + 断崖；min_ratio 固定为 0（不做分数地板）。"""
    return _filter_with_cliff_details(
        ranked_cols,
        ranked_scores,
        top_k=top_k,
        min_ratio=0.0,
        protect_ratio=protect_ratio,
    )


def _filter_max_cliff_only(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    reference_protect_ratio: float,
) -> dict:
    """
    最大断崖法：在 TopK 内取相邻相对跌幅最大的位置截断，不使用 min_ratio / protect 跳过分支。
    reference_protect_ratio 仅用于判断该断崖是否落在「原参数」保护区内。
    """
    topk_cols = ranked_cols[:top_k]
    topk_scores = [float(s) for s in ranked_scores[:top_k]]
    out: dict = {
        "final_columns": list(topk_cols),
        "cliff_index": None,
        "cliff_in_reference_protect_zone": False,
        "cliff_event": None,
        "reference_protect_ratio": float(reference_protect_ratio),
    }
    if not topk_cols:
        return out

    best = topk_scores[0]
    if best <= 0:
        return out
    protect_score = best * float(reference_protect_ratio)

    if len(topk_scores) < 2:
        return out

    cliff_idx = 0
    max_decay = -1.0
    for i in range(len(topk_scores) - 1):
        cur = topk_scores[i]
        nxt = topk_scores[i + 1]
        if cur <= 0:
            continue
        decay = (cur - nxt) / cur
        if decay > max_decay:
            max_decay = decay
            cliff_idx = i

    cur = topk_scores[cliff_idx]
    nxt = topk_scores[cliff_idx + 1]
    in_protect_zone = cur >= protect_score and nxt >= protect_score
    cutoff = cliff_idx + 1
    out.update(
        {
            "final_columns": topk_cols[:cutoff],
            "cliff_index": cliff_idx,
            "cliff_in_reference_protect_zone": in_protect_zone,
            "cliff_event": {
                "index": cliff_idx,
                "keep_count": cutoff,
                "kept_column": topk_cols[cliff_idx],
                "first_removed_column": topk_cols[cliff_idx + 1],
                "kept_score": _round_metric(cur),
                "first_removed_score": _round_metric(nxt),
                "kept_score_ratio": _round_metric(cur / best),
                "first_removed_score_ratio": _round_metric(nxt / best),
                "reference_protect_score": _round_metric(protect_score),
                "decay": _round_metric(max_decay),
                "both_in_reference_protect_zone": in_protect_zone,
            },
        }
    )
    return out


def _filter_local_adaptive_only(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
) -> dict:
    """
    局部自适应聚类（Otsu / 1D K-Means K=2）：在 TopK 内用最大类间方差找局部断崖，
    保留高分簇 C_keep。不使用 min_ratio / protect_ratio。

    退化情形（候选数 < 2 / 分数均匀 / 无显著断崖）由 local_adaptive_split 兜底为全保留。
    """
    topk_cols = ranked_cols[:top_k]
    topk_scores = [float(s) for s in ranked_scores[:top_k]]
    split = local_adaptive_split(topk_scores)
    keep = int(split.get("keep_count") or len(topk_cols))
    keep = max(0, min(keep, len(topk_cols)))
    final_cols = topk_cols[:keep] if keep > 0 else list(topk_cols)
    return {
        "final_columns": final_cols,
        "keep_count": len(final_cols),
        "drop_count": len(topk_cols) - len(final_cols),
        "degenerate": bool(split.get("degenerate")),
        "between_class_variance": split.get("between_class_variance"),
        "threshold_score": split.get("threshold_score"),
        "split": split,
    }


def _filter_protect_then_local_adaptive(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    protect_ratio: float,
) -> dict:
    """
    保护区 + Otsu：先按 protect_ratio 确定保护前缀，再在剩余 TopK 尾段做
    local_adaptive_split（Otsu）。断崖搜索从 protect 边界对 (protect_cut_idx-1) 开始。
    """
    topk_cols = ranked_cols[:top_k]
    topk_scores = [float(s) for s in ranked_scores[:top_k]]
    out: dict = {
        "final_columns": list(topk_cols),
        "protect_ratio": float(protect_ratio),
        "protect_cut_idx": len(topk_cols),
        "search_start": 0,
        "keep_count": len(topk_cols),
        "degenerate": True,
        "between_class_variance": None,
        "split": None,
    }
    if not topk_cols:
        return out

    best = topk_scores[0]
    if best <= 0:
        return out

    protect_score = best * float(protect_ratio)
    protect_cut_idx = len(topk_scores)
    for i, score in enumerate(topk_scores):
        if score < protect_score:
            protect_cut_idx = i
            break

    search_start = max(protect_cut_idx - 1, 0)
    tail_scores = topk_scores[search_start:]
    split = local_adaptive_split(tail_scores)
    tail_keep = int(split.get("keep_count") or len(tail_scores))
    tail_keep = max(1, min(tail_keep, len(tail_scores)))
    cutoff = min(max(search_start + tail_keep, protect_cut_idx), len(topk_cols))

    out.update(
        {
            "protect_cut_idx": int(protect_cut_idx),
            "protect_score": _round_metric(protect_score),
            "search_start": int(search_start),
            "final_columns": topk_cols[:cutoff],
            "keep_count": cutoff,
            "drop_count": len(topk_cols) - cutoff,
            "degenerate": bool(split.get("degenerate")),
            "between_class_variance": split.get("between_class_variance"),
            "threshold_score": split.get("threshold_score"),
            "split": split,
        }
    )
    return out


def _build_dual_mode_report(
    *,
    protect_ratio: float,
    protect_only_redundant: list[int],
    protect_only_full_recall: int,
    max_cliff_full_recall: int,
    max_cliff_in_protect_zone: int,
    max_cliff_in_protect_zone_fail: int,
    sample_count: int,
    local_adaptive_full_recall: int = 0,
    local_adaptive_redundant: list[int] | None = None,
    local_adaptive_keep_counts: list[int] | None = None,
    local_adaptive_degenerate_count: int = 0,
    protect_otsu_full_recall: int = 0,
    protect_otsu_redundant: list[int] | None = None,
    protect_otsu_keep_counts: list[int] | None = None,
    protect_otsu_degenerate_count: int = 0,
) -> dict:
    n = int(sample_count)
    redundant = list(protect_only_redundant)
    la_redundant = list(local_adaptive_redundant or [])
    la_keep = list(local_adaptive_keep_counts or [])
    po_redundant = list(protect_otsu_redundant or [])
    po_keep = list(protect_otsu_keep_counts or [])
    return {
        "protect_ratio": _round_metric(protect_ratio),
        "sample_count": n,
        "protect_only": {
            "min_ratio": 0.0,
            "avg_redundant_field_count": round(sum(redundant) / n, 6) if n else 0.0,
            "full_recall_count": int(protect_only_full_recall),
        },
        "max_cliff_only": {
            "full_recall_count": int(max_cliff_full_recall),
            "cliff_in_reference_protect_zone_count": int(max_cliff_in_protect_zone),
            "cliff_in_protect_zone_full_recall_fail_count": int(max_cliff_in_protect_zone_fail),
        },
        "local_adaptive_only": {
            "full_recall_count": int(local_adaptive_full_recall),
            "avg_redundant_field_count": round(sum(la_redundant) / n, 6) if n else 0.0,
            "avg_keep_count": round(sum(la_keep) / n, 6) if n else 0.0,
            "degenerate_count": int(local_adaptive_degenerate_count),
        },
        "protect_then_local_adaptive": {
            "full_recall_count": int(protect_otsu_full_recall),
            "avg_redundant_field_count": round(sum(po_redundant) / n, 6) if n else 0.0,
            "avg_keep_count": round(sum(po_keep) / n, 6) if n else 0.0,
            "degenerate_count": int(protect_otsu_degenerate_count),
        },
    }


# 横轴：四种截断策略（列名写入 CSV 表头）
CLIFF_EVAL_STRATEGY_COLUMNS: list[tuple[str, str]] = [
    ("max_cliff_only", "纯cliff"),
    ("protect_only", "cliff+protect"),
    ("local_adaptive_only", "otus"),
    ("protect_otsu", "otus+protect"),
]

CLIFF_EVAL_CSV_FIELDNAMES = ["metric"] + [label for _, label in CLIFF_EVAL_STRATEGY_COLUMNS]

# 纵轴：仅输出下列指标行（metric 列）
CLIFF_EVAL_PIVOT_METRIC_ROWS: list[tuple[str, str]] = [
    ("avg_recall", "avg_recall"),
    ("full_recall_rate", "full_recall_rate"),
    ("avg_redundant_field_count", "avg_redundant_field_count"),
    ("avg_keep_count", "avg_keep_count"),
    ("degenerate_count", "degenerate_count"),
]


def _build_cliff_eval_aggregate_row(
    *,
    split: str,
    top_k: int,
    protect_ratio: float,
    auto_select_params: bool,
    report: dict,
    recall_sums: dict[str, float],
    keep_sums: dict[str, float],
) -> dict:
    """由全集汇总指标构建单行 CSV（平均值 / 比率，不含逐题明细）。"""
    n = int(report["sample_count"])
    if n <= 0:
        raise ValueError("sample_count 必须 > 0")

    def _avg_recall(key: str) -> float:
        return round(float(recall_sums[key]) / n, 6)

    def _avg_keep(key: str) -> float:
        return round(float(keep_sums[key]) / n, 6)

    def _rate(count: int) -> float:
        return round(int(count) / n, 6)

    po = report["protect_only"]
    mc = report["max_cliff_only"]
    la = report["local_adaptive_only"]
    pot = report["protect_then_local_adaptive"]

    return {
        "split": split,
        "top_k": int(top_k),
        "protect_ratio": report.get("protect_ratio", _round_metric(protect_ratio)),
        "auto_select_params": int(bool(auto_select_params)),
        "sample_count": n,
        "protect_only_avg_recall": _avg_recall("protect_only"),
        "protect_only_full_recall_rate": _rate(po["full_recall_count"]),
        "protect_only_avg_redundant_field_count": po["avg_redundant_field_count"],
        "protect_only_avg_keep_count": _avg_keep("protect_only"),
        "max_cliff_only_avg_recall": _avg_recall("max_cliff_only"),
        "max_cliff_only_full_recall_rate": _rate(mc["full_recall_count"]),
        "max_cliff_only_avg_keep_count": _avg_keep("max_cliff_only"),
        "max_cliff_only_cliff_in_protect_zone_rate": _rate(
            mc["cliff_in_reference_protect_zone_count"]
        ),
        "max_cliff_only_cliff_in_protect_zone_fail_rate": _rate(
            mc["cliff_in_protect_zone_full_recall_fail_count"]
        ),
        "local_adaptive_only_avg_recall": _avg_recall("local_adaptive_only"),
        "local_adaptive_only_full_recall_rate": _rate(la["full_recall_count"]),
        "local_adaptive_only_avg_redundant_field_count": la["avg_redundant_field_count"],
        "local_adaptive_only_avg_keep_count": la["avg_keep_count"],
        "local_adaptive_only_degenerate_count": int(la["degenerate_count"]),
        "protect_otsu_avg_recall": _avg_recall("protect_otsu"),
        "protect_otsu_full_recall_rate": _rate(pot["full_recall_count"]),
        "protect_otsu_avg_redundant_field_count": pot["avg_redundant_field_count"],
        "protect_otsu_avg_keep_count": pot["avg_keep_count"],
        "protect_otsu_degenerate_count": int(pot["degenerate_count"]),
        "max_cliff_only_degenerate_count": 0,
        "protect_only_degenerate_count": 0,
    }


def _aggregate_to_pivot_rows(aggregate: dict) -> list[dict]:
    """全集汇总 dict → 透视表行：纵轴 metric，横轴四种策略。"""
    rows: list[dict] = []
    for metric_label, metric_suffix in CLIFF_EVAL_PIVOT_METRIC_ROWS:
        row: dict = {"metric": metric_label}
        for strategy_key, column_label in CLIFF_EVAL_STRATEGY_COLUMNS:
            agg_key = f"{strategy_key}_{metric_suffix}"
            row[column_label] = aggregate.get(agg_key, 0)
        rows.append(row)
    return rows


def _write_cliff_eval_csv(path: Path, aggregate: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _aggregate_to_pivot_rows(aggregate)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLIFF_EVAL_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _empty_failure_summary() -> dict[str, int]:
    return {
        "not_in_topk": 0,
        "removed_by_min_ratio": 0,
        "removed_by_cliff": 0,
    }


def _causal_params_to_keep(
    *,
    reason: str,
    rank: int | None,
    score_ratio: float | None,
    cliff_event: dict,
) -> dict[str, float | int | None]:
    params: dict[str, float | int | None] = {}
    if reason == "not_in_topk":
        params["top_k_gte"] = rank
    elif reason == "removed_by_min_ratio":
        params["candidate_cliff_min_ratio_lte"] = _round_metric(score_ratio)
    elif reason == "removed_by_cliff":
        params["candidate_cliff_protect_ratio_lte"] = _round_metric(score_ratio)
    return params


def _format_failure_for_log(recall_failure: dict) -> str:
    if not recall_failure.get("failed"):
        return ""
    parts: list[str] = []
    for missing in recall_failure.get("missing_gold_columns") or []:
        params = missing.get("causal_params_to_keep") or {}
        param_text = ", ".join(f"{k}={v}" for k, v in params.items())
        parts.append(
            f"{missing.get('column')}[{missing.get('reason')}"
            f"{'; ' + param_text if param_text else ''}]"
        )
    return "; ".join(parts)


def _diagnose_recall_failure(
    gold: set[str],
    ranked_cols: list[str],
    ranked_scores: list[float],
    filter_details: dict,
) -> dict:
    final_set = set(filter_details.get("final_columns") or [])
    missing_gold = sorted(gold - final_set)
    summary = _empty_failure_summary()
    if not missing_gold:
        return {
            "failed": False,
            "missing_gold_columns": [],
            "failure_summary": summary,
        }

    rank_map = {col: i for i, col in enumerate(ranked_cols)}
    score_map = {
        col: float(ranked_scores[i])
        for i, col in enumerate(ranked_cols[: len(ranked_scores)])
    }
    topk_set = set(filter_details.get("topk_columns") or [])
    min_set = set(filter_details.get("min_filtered_columns") or [])
    best = float(filter_details.get("best_score") or 0.0)
    cliff_event = filter_details.get("cliff_event") or {}

    missing_details: list[dict] = []
    for col in missing_gold:
        rank_idx = rank_map.get(col)
        rank = rank_idx + 1 if rank_idx is not None else None
        score = score_map.get(col)
        score_ratio = (score / best) if score is not None and best > 0 else None
        in_topk = col in topk_set
        in_min = col in min_set

        if not in_topk:
            reason = "not_in_topk"
        elif not in_min:
            reason = "removed_by_min_ratio"
        else:
            reason = "removed_by_cliff"
        summary[reason] += 1

        causal_params = _causal_params_to_keep(
            reason=reason,
            rank=rank,
            score_ratio=score_ratio,
            cliff_event=cliff_event,
        )
        missing_details.append(
            {
                "column": col,
                "reason": reason,
                "rank": rank,
                "score": _round_metric(score),
                "score_ratio_to_top1": _round_metric(score_ratio),
                "in_topk": in_topk,
                "in_min_filtered": in_min,
                "cliff_event": cliff_event or None,
                "causal_params_to_keep": causal_params,
            }
        )

    return {
        "failed": True,
        "missing_gold_columns": missing_details,
        "failure_summary": summary,
    }


def _as_text_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _build_entity_schema_context(
    *,
    topk_columns: list[str],
    linker,
    passage_map: dict[str, str],
) -> dict:
    """
    Build the entity-extraction schema from CrossEncoder reranked TopK columns.

    EntityExtractor currently renders `列名` and `列描述`, so `列描述` is enriched
    with the same high-signal pieces used by the main schema items.
    """
    meta_map = {
        str(item.get("column_name") or "").strip(): item
        for item in getattr(linker, "column_metadata", []) or []
        if str(item.get("column_name") or "").strip()
    }
    profile_map = getattr(linker, "profile_detail_map", {}) or {}
    rows: list[dict] = []
    for col in topk_columns:
        col = str(col or "").strip()
        if not col:
            continue
        meta = meta_map.get(col, {})
        profile = profile_map.get(col, {})
        dtype = str(profile.get("字段类型") or meta.get("data_type") or "").strip()
        desc = str(meta.get("column_description") or meta.get("列描述") or "").strip()
        examples = _as_text_list(
            profile.get("枚举值")
            or profile.get("示例值")
            or meta.get("examples")
            or meta.get("示例值")
        )

        description_parts: list[str] = []
        if desc:
            description_parts.append(f"列描述：{desc}")
        if dtype:
            description_parts.append(f"字段类型：{dtype}")
        if profile.get("是否枚举") == "是":
            description_parts.append("是否枚举：是")
        if examples:
            description_parts.append(f"示例值：{'/'.join(examples[:6])}")
        if profile.get("格式") not in (None, ""):
            description_parts.append(f"格式：{profile.get('格式')}")
        if profile.get("范围") not in (None, ""):
            description_parts.append(f"范围：{profile.get('范围')}")
        if profile.get("唯一值数") not in (None, ""):
            description_parts.append(f"唯一值数：{profile.get('唯一值数')}")
        if not description_parts and passage_map.get(col):
            description_parts.append(str(passage_map.get(col)))

        row = {
            "列名": col,
            "列描述": "；".join(description_parts),
        }
        if dtype:
            row["字段类型"] = dtype
        if profile.get("是否枚举") == "是":
            row["是否枚举"] = "是"
        if examples:
            row["示例值"] = examples[:6]
        if profile.get("格式") not in (None, ""):
            row["格式"] = profile.get("格式")
        if profile.get("范围") not in (None, ""):
            row["范围"] = profile.get("范围")
        rows.append(row)
    return {"召回schema": rows}


async def _extract_entity_must_have(
    *,
    linker,
    entity_extractor,
    question: str,
    topk_columns: list[str],
    passage_map: dict[str, str],
):
    from pipeline.utils import TokenTracker

    schema_context = _build_entity_schema_context(
        topk_columns=topk_columns,
        linker=linker,
        passage_map=passage_map,
    )
    tracker = TokenTracker()
    entities = await entity_extractor.extract(
        question,
        schema_context,
        tracker,
        schema_columns=linker.column_names,
    )
    pack = linker.retrieve(question, entities)
    must_have = [str(c).strip() for c in (pack.必须列集合 or []) if str(c).strip()]
    return entities, must_have, pack.证据详情 or {}


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="评估 CrossEncoder + 断崖筛选 recall")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--protect-ratio",
        type=float,
        default=None,
        help="protect_only / 最大断崖保护区对照；未指定且未 auto-select 时用 settings.candidate_cliff_protect_ratio",
    )
    parser.add_argument(
        "--auto-select-params",
        action="store_true",
        help="在验证集上自动选择 protect_ratio（默认 r_min 曲线拐点法）与 min_ratio",
    )
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=50,
        help="auto-select 时用于噪声断崖分析的拉长 TopK，默认 50",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.95,
        help="protect 覆盖率法的目标 tau；elbow 回退时也用作 min_coverage 默认参考",
    )
    parser.add_argument(
        "--protect-min-coverage",
        type=float,
        default=0.80,
        help="elbow 选 protect_ratio 时的最低覆盖率；不足则回退到覆盖率法",
    )
    parser.add_argument(
        "--protect-selection-method",
        choices=("elbow", "coverage"),
        default="elbow",
        help="protect_ratio 选择方式：elbow=曲线拐点（默认），coverage=覆盖率约束",
    )
    parser.add_argument(
        "--min-ratio-margin-eps",
        type=float,
        default=1e-4,
        help="min_ratio 相对 protect_ratio 的安全边距 eps",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出路径，默认 training/cliff_eval_{split}_k{top_k}_p{protect}.csv（策略×指标透视表）；.json 则写 JSON",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k 必须 > 0")
    if args.analysis_top_k <= 0:
        raise ValueError("--analysis-top-k 必须 > 0")
    if not (0.0 < args.target_coverage <= 1.0):
        raise ValueError("--target-coverage 必须在 (0, 1]")
    if not (0.0 < args.protect_min_coverage <= 1.0):
        raise ValueError("--protect-min-coverage 必须在 (0, 1]")
    if args.min_ratio_margin_eps < 0:
        raise ValueError("--min-ratio-margin-eps 必须 >= 0")

    analysis_top_k = max(args.top_k, args.analysis_top_k)
    model_path = str(args.model_path).strip()
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 数据文件不存在: {csv_path}")

    model = CrossEncoder(model_path, trust_remote_code=True)
    records = load_eval_records(args.split)
    passage_map = build_column_passage_map(str(settings.schema_path), csv_path)
    all_cols = list(passage_map.keys())

    pending_cases: list[dict] = []
    r_min_values: list[float] = []
    cliff1_kept_ratios: list[float | None] = []
    cliff2_noise_start_ratios: list[float | None] = []
    total = 0

    for idx, item in enumerate(records, start=1):
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns") or [])
        if not question or not gold:
            continue
        total += 1

        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=args.batch_size)
        sorted_idx = np.argsort(scores)[::-1]
        ranked_cols = [all_cols[j] for j in sorted_idx]
        ranked_scores = [float(scores[j]) for j in sorted_idx]

        est = None
        if args.auto_select_params:
            est = _estimate_params_for_case(
                ranked_cols,
                ranked_scores,
                gold,
                top_k=args.top_k,
                analysis_top_k=analysis_top_k,
            )
            if est.get("covered"):
                r_min_values.append(float(est["min_gold_ratio"]))
                expected = est.get("expected_cliff")
                noise_floor = est.get("noise_floor_cliff")
                cliff1_kept_ratios.append(
                    float(expected["kept_ratio"]) if expected else None
                )
                cliff2_noise_start_ratios.append(
                    float(noise_floor["noise_start_ratio"]) if noise_floor else None
                )
        pending_cases.append(
            {
                "idx": idx,
                "question": question,
                "gold": gold,
                "ranked_cols": ranked_cols,
                "ranked_scores": ranked_scores,
                "estimation": est,
            }
        )

    protect_ratio = (
        float(args.protect_ratio)
        if args.protect_ratio is not None
        else float(settings.candidate_cliff_protect_ratio)
    )
    param_selection: dict | None = None
    if args.auto_select_params:
        if not r_min_values:
            raise RuntimeError("auto-select-params 失败：无 covered 样本可用于标定。")
        param_selection = select_dual_thresholds(
            r_min_values=r_min_values,
            remote_cliff_noise_start_ratios=cliff2_noise_start_ratios,
            target_coverage=args.target_coverage,
            margin_eps=args.min_ratio_margin_eps,
            aggregate_percentile=float(getattr(args, "dual_min_percentile", 50.0)),
            protect_min_coverage=args.protect_min_coverage,
            protect_selection_method=args.protect_selection_method,
            cliff1_kept_ratios=cliff1_kept_ratios,
        )
        protect_ratio = float(param_selection["candidate_cliff_protect_ratio"])

    if not (0.0 <= protect_ratio <= 1.0):
        raise ValueError("protect_ratio 必须在 [0,1]")

    protect_redundant: list[int] = []
    protect_full_recall = 0
    max_cliff_full_recall = 0
    max_cliff_in_protect_zone = 0
    max_cliff_in_protect_zone_fail = 0
    local_adaptive_full_recall = 0
    local_adaptive_redundant: list[int] = []
    local_adaptive_keep_counts: list[int] = []
    local_adaptive_degenerate_count = 0
    protect_otsu_full_recall = 0
    protect_otsu_redundant: list[int] = []
    protect_otsu_keep_counts: list[int] = []
    protect_otsu_degenerate_count = 0
    recall_sums = {
        "protect_only": 0.0,
        "max_cliff_only": 0.0,
        "local_adaptive_only": 0.0,
        "protect_otsu": 0.0,
    }
    keep_sums = {
        "protect_only": 0.0,
        "max_cliff_only": 0.0,
        "local_adaptive_only": 0.0,
        "protect_otsu": 0.0,
    }

    for case in pending_cases:
        gold = case["gold"]
        ranked_cols = case["ranked_cols"]
        ranked_scores = case["ranked_scores"]

        protect_details = _filter_protect_only(
            ranked_cols,
            ranked_scores,
            top_k=args.top_k,
            protect_ratio=protect_ratio,
        )
        protect_metrics = _column_metrics(gold, protect_details["final_columns"])
        protect_redundant.append(int(protect_metrics["wrong_count"]))
        protect_keep = int(
            protect_details.get("cutoff") or len(protect_details["final_columns"])
        )
        recall_sums["protect_only"] += float(protect_metrics["recall"])
        keep_sums["protect_only"] += float(protect_keep)
        if protect_metrics["full_recall"]:
            protect_full_recall += 1

        max_cliff_details = _filter_max_cliff_only(
            ranked_cols,
            ranked_scores,
            top_k=args.top_k,
            reference_protect_ratio=protect_ratio,
        )
        max_cliff_metrics = _column_metrics(gold, max_cliff_details["final_columns"])
        max_cliff_keep = len(max_cliff_details["final_columns"])
        recall_sums["max_cliff_only"] += float(max_cliff_metrics["recall"])
        keep_sums["max_cliff_only"] += float(max_cliff_keep)
        if max_cliff_metrics["full_recall"]:
            max_cliff_full_recall += 1
        if max_cliff_details["cliff_in_reference_protect_zone"]:
            max_cliff_in_protect_zone += 1
            if not max_cliff_metrics["full_recall"]:
                max_cliff_in_protect_zone_fail += 1

        local_adaptive_details = _filter_local_adaptive_only(
            ranked_cols,
            ranked_scores,
            top_k=args.top_k,
        )
        local_adaptive_metrics = _column_metrics(
            gold, local_adaptive_details["final_columns"]
        )
        local_adaptive_redundant.append(int(local_adaptive_metrics["wrong_count"]))
        la_keep = int(local_adaptive_details["keep_count"])
        local_adaptive_keep_counts.append(la_keep)
        recall_sums["local_adaptive_only"] += float(local_adaptive_metrics["recall"])
        keep_sums["local_adaptive_only"] += float(la_keep)
        if local_adaptive_details["degenerate"]:
            local_adaptive_degenerate_count += 1
        if local_adaptive_metrics["full_recall"]:
            local_adaptive_full_recall += 1

        protect_otsu_details = _filter_protect_then_local_adaptive(
            ranked_cols,
            ranked_scores,
            top_k=args.top_k,
            protect_ratio=protect_ratio,
        )
        protect_otsu_metrics = _column_metrics(
            gold, protect_otsu_details["final_columns"]
        )
        protect_otsu_redundant.append(int(protect_otsu_metrics["wrong_count"]))
        po_keep = int(protect_otsu_details["keep_count"])
        protect_otsu_keep_counts.append(po_keep)
        recall_sums["protect_otsu"] += float(protect_otsu_metrics["recall"])
        keep_sums["protect_otsu"] += float(po_keep)
        if protect_otsu_details["degenerate"]:
            protect_otsu_degenerate_count += 1
        if protect_otsu_metrics["full_recall"]:
            protect_otsu_full_recall += 1

    if total == 0:
        raise RuntimeError("无可评估样本（question/gold_columns 为空）。")

    report = _build_dual_mode_report(
        protect_ratio=protect_ratio,
        protect_only_redundant=protect_redundant,
        protect_only_full_recall=protect_full_recall,
        max_cliff_full_recall=max_cliff_full_recall,
        max_cliff_in_protect_zone=max_cliff_in_protect_zone,
        max_cliff_in_protect_zone_fail=max_cliff_in_protect_zone_fail,
        sample_count=total,
        local_adaptive_full_recall=local_adaptive_full_recall,
        local_adaptive_redundant=local_adaptive_redundant,
        local_adaptive_keep_counts=local_adaptive_keep_counts,
        local_adaptive_degenerate_count=local_adaptive_degenerate_count,
        protect_otsu_full_recall=protect_otsu_full_recall,
        protect_otsu_redundant=protect_otsu_redundant,
        protect_otsu_keep_counts=protect_otsu_keep_counts,
        protect_otsu_degenerate_count=protect_otsu_degenerate_count,
    )
    if param_selection is not None:
        report["param_selection"] = param_selection

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).parent
        / f"cliff_eval_{args.split}_k{args.top_k}_p{protect_ratio:.6f}.csv"
    )
    aggregate_csv_row = _build_cliff_eval_aggregate_row(
        split=args.split,
        top_k=args.top_k,
        protect_ratio=protect_ratio,
        auto_select_params=bool(args.auto_select_params),
        report=report,
        recall_sums=recall_sums,
        keep_sums=keep_sums,
    )

    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        _write_cliff_eval_csv(out_path, aggregate_csv_row)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"输出: {out_path}")


if __name__ == "__main__":
    asyncio.run(_main_async())

