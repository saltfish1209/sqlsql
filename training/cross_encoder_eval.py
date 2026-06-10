"""
CrossEncoder 离线评估（可返回结构化指标，供 figure/crossencoder 实验脚本写 CSV）。

指标体系:
  - NDCG@K  : 归一化折损累积增益，衡量模型把 gold 列往前排的综合能力（训练北极星指标）
  - MRR     : 平均倒数排名，第一个 gold 列排在多靠前
  - Recall@K: gold 列集合是否全部落在 Top-K 内（推理/部署阶段参考）
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.cross_encoder_passage import build_column_passage_map
from training.dataset_io import read_jsonl

VAL_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_val_data.jsonl")
TEST_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.jsonl")
VAL_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_val_data.json")
TEST_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.json")


def get_eval_file(split: str) -> str:
    split_norm = split.strip().lower()
    if split_norm == "val":
        return VAL_FILE_JSONL if os.path.isfile(VAL_FILE_JSONL) else VAL_FILE_JSON
    if split_norm == "test":
        return TEST_FILE_JSONL if os.path.isfile(TEST_FILE_JSONL) else TEST_FILE_JSON
    raise ValueError(f"不支持的数据集类型: {split}，仅支持 val/test")


def load_eval_records(split: str) -> list[dict]:
    path = get_eval_file(split)
    if path.endswith(".jsonl"):
        return list(read_jsonl(path))
    import json as _json

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _json.load(f)


def gold_column_stats(split: str = "val") -> dict:
    """统计 gold 列数量分布，用于推荐评估 Top-K。"""
    records = load_eval_records(split)
    sizes: list[int] = []
    for item in records:
        gold = item.get("gold_columns") or []
        if not gold:
            continue
        sizes.append(len(set(gold)))
    if not sizes:
        return {
            "split": split,
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "recommended_k_primary": 6,
            "recommended_k_operational": int(getattr(settings, "candidate_top_k", 20)),
        }
    arr = np.array(sizes, dtype=np.int32)
    p95 = int(np.percentile(arr, 95))
    p90 = int(np.percentile(arr, 90))
    k_primary = max(6, p95)
    k_operational = int(getattr(settings, "candidate_top_k", 20))
    return {
        "split": split,
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": p90,
        "p95": p95,
        "recommended_k_primary": k_primary,
        "recommended_k_operational": k_operational,
    }


# ── 排序指标计算（从 ranking_metrics 导入，保持公开 API 不变）──

from training.ranking_metrics import compute_mrr, compute_ndcg  # noqa: F401


# ── 主评估函数 ────────────────────────────────────────────────


def _load_eval_context(split: str) -> tuple[list[str], dict[str, str], list[dict]]:
    """加载列 passage 映射和评估数据（不加载模型）。"""
    eval_file = get_eval_file(split)
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"评估数据不存在: {eval_file}")

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 数据文件不存在: {csv_path}")

    df_raw = pd.read_csv(csv_path, nrows=1)
    raw_cols = [str(c).strip() for c in df_raw.columns]
    passage_map = build_column_passage_map(
        str(settings.schema_path), csv_path, active_columns=raw_cols
    )
    eval_data = load_eval_records(split)
    all_cols = list(passage_map.keys())
    return all_cols, passage_map, eval_data


def _load_model_and_data(
    model_path: str, split: str
) -> tuple[CrossEncoder, list[str], dict[str, str], list[dict]]:
    """加载模型 + 列 passage 映射 + 评估数据（从磁盘加载模型）。"""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    model = CrossEncoder(model_path, trust_remote_code=True)
    all_cols, passage_map, eval_data = _load_eval_context(split)
    return model, all_cols, passage_map, eval_data


def _score_and_rank(
    model: CrossEncoder,
    all_cols: list[str],
    passage_map: dict[str, str],
    eval_data: list[dict],
    top_k: int,
    batch_size: int,
) -> dict:
    """用给定模型实例对 eval_data 做排序评估，返回指标字典。"""
    recall_success = 0
    total = 0
    gold_sizes: list[int] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []

    for item in eval_data:
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns", []))
        if not question or not gold:
            continue
        gold_sizes.append(len(gold))
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=batch_size)
        sorted_idx = np.argsort(scores)[::-1]
        ranked_cols = [all_cols[j] for j in sorted_idx]

        ndcg_scores.append(compute_ndcg(ranked_cols, gold, top_k))
        mrr_scores.append(compute_mrr(ranked_cols, gold, top_k))

        pred_topk_set = set(ranked_cols[:top_k])
        if gold.issubset(pred_topk_set):
            recall_success += 1
        total += 1

    full_recall = recall_success / total if total else 0.0
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    return {
        "top_k": int(top_k),
        "num_candidates": len(all_cols),
        "total": total,
        "success": recall_success,
        "recall_at_k": full_recall,
        "full_recall": full_recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "gold_size_mean": float(np.mean(gold_sizes)) if gold_sizes else 0.0,
        "gold_size_max": int(max(gold_sizes)) if gold_sizes else 0,
    }


def evaluate_cross_encoder(
    model_path: str = "",
    *,
    split: str = "val",
    top_k: int = 6,
    batch_size: int = 32,
    model: CrossEncoder | None = None,
) -> dict:
    """
    对全表列做 CrossEncoder 排序，返回 NDCG@K / MRR / Recall@K。

    传入 model 参数可直接使用内存中的模型实例（跳过磁盘加载），
    用于训练中期 eval 避免不必要的 save→load 开销。
    """
    if model is not None:
        all_cols, passage_map, eval_data = _load_eval_context(split)
        eval_model = model
    else:
        eval_model, all_cols, passage_map, eval_data = _load_model_and_data(model_path, split)

    result = _score_and_rank(eval_model, all_cols, passage_map, eval_data, top_k, batch_size)
    result["model_path"] = model_path
    result["split"] = split
    return result


def evaluate_cross_encoder_multi_k(
    model_path: str = "",
    *,
    split: str = "val",
    top_ks: Iterable[int] = (3, 4, 5, 6, 8, 10, 12, 15, 20),
    batch_size: int = 32,
    model: CrossEncoder | None = None,
) -> list[dict]:
    """同一模型在多个 K 上评估 Recall@K / NDCG@K 曲线。"""
    if model is not None:
        all_cols, passage_map, eval_data = _load_eval_context(split)
        eval_model = model
    else:
        eval_model, all_cols, passage_map, eval_data = _load_model_and_data(model_path, split)

    cases: list[tuple[str, set[str]]] = []
    for item in eval_data:
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns", []))
        if question and gold:
            cases.append((question, gold))

    max_k = max(top_ks)
    per_case_ranked: list[list[str]] = []
    for question, _gold in cases:
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = eval_model.predict(inputs, batch_size=batch_size)
        sorted_idx = np.argsort(scores)[::-1][:max_k]
        per_case_ranked.append([all_cols[j] for j in sorted_idx])

    ks = sorted({int(k) for k in top_ks if k > 0})
    rows: list[dict] = []
    total = len(cases)
    for k in ks:
        recall_success = 0
        ndcg_scores: list[float] = []
        for (_question, gold), ranked in zip(cases, per_case_ranked):
            pred_k = set(ranked[:k])
            if gold.issubset(pred_k):
                recall_success += 1
            ndcg_scores.append(compute_ndcg(ranked, gold, k))
        rows.append({
            "model_path": model_path,
            "split": split,
            "top_k": k,
            "total": total,
            "success": recall_success,
            "recall_at_k": recall_success / total if total else 0.0,
            "full_recall": recall_success / total if total else 0.0,
            "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        })
    return rows
