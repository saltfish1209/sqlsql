"""排序评估指标（纯 Python，无重依赖）。"""
from __future__ import annotations

import math


def compute_ndcg(ranked_cols: list[str], gold: set[str], k: int) -> float:
    """单条 query 的 NDCG@K（二值相关性: gold 列=1, 其余=0）。"""
    n_rel = len(gold)
    if n_rel == 0:
        return 0.0
    dcg = 0.0
    for i, col in enumerate(ranked_cols[:k]):
        if col in gold:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_rel)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranked_cols: list[str], gold: set[str], k: int = 0) -> float:
    """单条 query 的 Reciprocal Rank（第一个 gold 列的倒数排名）。

    k > 0 时只在前 K 个位置中查找；k <= 0 时遍历全部排名。
    """
    search = ranked_cols[:k] if k > 0 else ranked_cols
    for i, col in enumerate(search):
        if col in gold:
            return 1.0 / (i + 1)
    return 0.0
