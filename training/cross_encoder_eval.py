"""
CrossEncoder 离线评估（可返回结构化指标，供 figure/crossencoder 实验脚本写 CSV）。
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


def evaluate_cross_encoder(
    model_path: str,
    *,
    split: str = "val",
    top_k: int = 6,
    batch_size: int = 32,
) -> dict:
    """
    对全表列做 CrossEncoder 排序，计算 Full Recall@K 与 Top-1 Hit Rate。

    Full Recall@K: gold 列集合是否为 Top-K 的子集。
    Top-1 Hit Rate: 排名第一的列是否属于 gold（多 gold 时的弱指标）。
    """
    eval_file = get_eval_file(split)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"评估数据不存在: {eval_file}")

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 数据文件不存在: {csv_path}")

    model = CrossEncoder(model_path, trust_remote_code=True)
    df_raw = pd.read_csv(csv_path, nrows=1)
    raw_cols = [str(c).strip() for c in df_raw.columns]
    passage_map = build_column_passage_map(str(settings.schema_path), csv_path, active_columns=raw_cols)
    all_cols = list(passage_map.keys())
    eval_data = load_eval_records(split)

    success = 0
    total = 0
    gold_sizes: list[int] = []

    for item in eval_data:
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns", []))
        if not question or not gold:
            continue
        gold_sizes.append(len(gold))
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=batch_size)
        sorted_idx = np.argsort(scores)[::-1]
        topk_idx = sorted_idx[:top_k]
        pred_topk = [all_cols[j] for j in topk_idx]
        pred_topk_set = set(pred_topk)
        if gold.issubset(pred_topk_set):
            success += 1
        total += 1

    full_recall = success / total if total else 0.0
    return {
        "model_path": model_path,
        "split": split,
        "top_k": int(top_k),
        "num_candidates": len(all_cols),
        "total": total,
        "success": success,
        "recall_at_k": full_recall,
        "full_recall": full_recall,
        "gold_size_mean": float(np.mean(gold_sizes)) if gold_sizes else 0.0,
        "gold_size_max": int(max(gold_sizes)) if gold_sizes else 0,
    }


def evaluate_cross_encoder_multi_k(
    model_path: str,
    *,
    split: str = "val",
    top_ks: Iterable[int] = (3, 4, 5, 6, 8, 10, 12, 15, 20),
    batch_size: int = 32,
) -> list[dict]:
    """同一模型在多个 K 上评估，用于 Recall@K 曲线 CSV。"""
    eval_file = get_eval_file(split)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"评估数据不存在: {eval_file}")

    csv_path = str(settings.csv_path)
    df_raw = pd.read_csv(csv_path, nrows=1)
    raw_cols = [str(c).strip() for c in df_raw.columns]
    passage_map = build_column_passage_map(str(settings.schema_path), csv_path, active_columns=raw_cols)
    all_cols = list(passage_map.keys())
    eval_data = load_eval_records(split)

    cases: list[tuple[str, set[str]]] = []
    for item in eval_data:
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns", []))
        if question and gold:
            cases.append((question, gold))

    model = CrossEncoder(model_path, trust_remote_code=True)
    max_k = max(top_ks)
    per_case_top: list[list[str]] = []
    for question, _gold in cases:
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=batch_size)
        sorted_idx = np.argsort(scores)[::-1][:max_k]
        per_case_top.append([all_cols[j] for j in sorted_idx])

    ks = sorted({int(k) for k in top_ks if k > 0})
    rows: list[dict] = []
    total = len(cases)
    for k in ks:
        success = 0
        for (_question, gold), pred_full in zip(cases, per_case_top):
            pred_k = set(pred_full[:k])
            if gold.issubset(pred_k):
                success += 1
        rows.append({
            "model_path": model_path,
            "split": split,
            "top_k": k,
            "total": total,
            "success": success,
            "recall_at_k": success / total if total else 0.0,
            "full_recall": success / total if total else 0.0,
        })
    return rows
