"""Top-K 选用规则（与主系统 candidate_top_k 区分）。"""
from __future__ import annotations

from typing import Iterable


def _row_recall(row: dict) -> float:
    """兼容 CSV 列名 recall_at_k 与旧版 full_recall。"""
    if "recall_at_k" in row and row["recall_at_k"] not in (None, ""):
        return float(row["recall_at_k"])
    return float(row.get("full_recall", 0.0))


def pick_top_k_from_curve(
    curve_rows: Iterable[dict],
    *,
    gold_p95: int,
    min_k: int = 6,
    recall_ratio_of_max: float = 0.95,
) -> dict:
    """
    从 Recall@K 曲线选「部署 K」：
    在 K >= max(min_k, gold_p95) 的档位中，取满足
    full_recall >= recall_ratio_of_max * max_recall 的最小 K。
    """
    rows = sorted(
        (r for r in curve_rows if int(r.get("top_k", 0)) > 0),
        key=lambda x: int(x["top_k"]),
    )
    if not rows:
        return {
            "chosen_k": min_k,
            "reason": "empty_curve",
            "max_recall": 0.0,
            "target_recall": 0.0,
            "k_floor": max(min_k, gold_p95),
        }

    recalls = [_row_recall(r) for r in rows]
    max_recall = max(recalls)
    target = max_recall * recall_ratio_of_max
    k_floor = max(min_k, int(gold_p95))

    for r in rows:
        k = int(r["top_k"])
        if k >= k_floor and _row_recall(r) >= target:
            return {
                "chosen_k": k,
                "reason": f"smallest_k_with_recall>={target:.4f}({recall_ratio_of_max:.0%} of max)",
                "max_recall": max_recall,
                "target_recall": target,
                "k_floor": k_floor,
                "recall_at_chosen": _row_recall(r),
            }

    best = max(rows, key=_row_recall)
    return {
        "chosen_k": int(best["top_k"]),
        "reason": "fallback_max_recall",
        "max_recall": max_recall,
        "target_recall": target,
        "k_floor": k_floor,
        "recall_at_chosen": _row_recall(best),
    }
