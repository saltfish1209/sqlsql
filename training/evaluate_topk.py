"""
CrossEncoder Top-K 全覆盖率评估 —— 验证 Schema Pruner 的部署级召回能力。

此脚本用于推理/部署阶段评估，在具体 K 值下的 Recall 与 NDCG。
训练阶段的指标对比请使用 evaluate_cross_encoder.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.cross_encoder_passage import build_column_passage_map
from training.ranking_metrics import compute_mrr, compute_ndcg
from training.dataset_io import read_jsonl

TEST_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.jsonl")
TEST_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.json")
TOP_K = 15


def _load_eval_records(jsonl_path: str, json_path: str) -> list[dict] | None:
    if os.path.isfile(jsonl_path):
        return list(read_jsonl(jsonl_path))
    if os.path.isfile(json_path):
        import json as _json

        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                return _json.load(f)
            except _json.JSONDecodeError as exc:
                print(f"[ERROR] 评估文件解析失败 ({exc.msg}): {json_path}")
                return None
    return None


def evaluate():
    print("开始评估 Top-K 全覆盖率...")
    model_path = str(settings.cross_encoder_model)
    if not os.path.isdir(model_path):
        print(f"模型未找到: {model_path}，请先运行 train_cross_encoder.py")
        return
    model = CrossEncoder(model_path, trust_remote_code=True)

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV 数据文件不存在: {csv_path}")
        return
    df_raw = pd.read_csv(csv_path, nrows=1)
    raw_cols = [str(c).strip() for c in df_raw.columns]
    passage_map = build_column_passage_map(str(settings.schema_path), csv_path, active_columns=raw_cols)
    all_cols = list(passage_map.keys())
    print(f"候选列数: {len(all_cols)}")

    test_data = _load_eval_records(TEST_FILE_JSONL, TEST_FILE_JSON)
    if test_data is None:
        print(
            f"[ERROR] 测试数据不存在: {TEST_FILE_JSONL} 或 {TEST_FILE_JSON}，"
            "请先运行 prepare_data.py"
        )
        return

    success = total = 0
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    for i, item in enumerate(test_data):
        question = item["question"]
        gold = set(item["gold_columns"])
        if not gold:
            continue
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=32)
        sorted_idx = np.argsort(scores)[::-1]
        ranked_cols = [all_cols[j] for j in sorted_idx]
        top_pred = set(ranked_cols[:TOP_K])
        hit = gold.issubset(top_pred)
        if hit:
            success += 1
        total += 1
        ndcg_scores.append(compute_ndcg(ranked_cols, gold, TOP_K))
        mrr_scores.append(compute_mrr(ranked_cols, gold, TOP_K))
        if i < 3:
            print(f"\n[Case {i}] Q: {question}")
            print(f"  Gold: {gold}")
            print(f"  Top-{TOP_K}: {ranked_cols[:5]} ...")
            print(f"  {'OK' if hit else 'FAIL'}")

    recall = success / total if total else 0
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    print(f"\n{'='*50}")
    print(f"Top-{TOP_K} Full Recall: {recall:.2%} ({success}/{total})")
    print(f"NDCG@{TOP_K}: {ndcg:.4f}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    evaluate()
