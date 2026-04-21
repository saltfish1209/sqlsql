"""
CrossEncoder 验证/测试评估脚本。
───────────────────────────────────────────────────────────────
读取 prepare_data.py 生成的验证集或测试集，
使用已训练好的 CrossEncoder 权重计算 Top-K Full Recall。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

VAL_FILE = os.path.join(os.path.dirname(__file__), "cross_encoder_val_data.json")
TEST_FILE = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.json")
TOP_K = 6


def get_eval_file(split: str) -> str:
    split_norm = split.strip().lower()
    if split_norm == "val":
        return VAL_FILE
    if split_norm == "test":
        return TEST_FILE
    raise ValueError(f"不支持的数据集类型: {split}，仅支持 val/test")


def evaluate(split: str = "val", top_k: int = TOP_K) -> None:
    eval_file = get_eval_file(split)
    model_path = str(settings.cross_encoder_model)
    # model_path = str(settings.reranker_base_model)


    print(f"开始评估 CrossEncoder ({split})...")
    print(f"模型路径: {model_path}")
    print(f"评估文件: {eval_file}")
    print(f"Top-K: {top_k}")

    if not os.path.isdir(model_path):
        print(f"[ERROR] 模型未找到: {model_path}，请先运行 train_cross_encoder.py")
        return
    if not os.path.isfile(eval_file):
        print(f"[ERROR] 评估数据文件不存在: {eval_file}，请先运行 prepare_data.py")
        return

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV 数据文件不存在: {csv_path}")
        return

    model = CrossEncoder(model_path, trust_remote_code=True)


    # 候选列：来自业务表 CSV 表头
    df_raw = pd.read_csv(csv_path, nrows=1)
    all_cols = [str(c).strip() for c in df_raw.columns]
    print(f"候选列数: {len(all_cols)}")

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    success = 0
    total = 0
    top1_hits = 0

    for i, item in enumerate(eval_data):
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns", []))
        if not question or not gold:
            continue

        inputs = [[question, col] for col in all_cols]
        scores = model.predict(inputs, batch_size=32)
        sorted_idx = np.argsort(scores)[::-1]
        topk_idx = sorted_idx[:top_k]
        pred_topk = [all_cols[j] for j in topk_idx]
        pred_topk_set = set(pred_topk)

        is_full_recall = gold.issubset(pred_topk_set)
        if is_full_recall:
            success += 1

        if pred_topk and pred_topk[0] in gold:
            top1_hits += 1

        total += 1

        if i < 3:
            print(f"\n[Case {i}] Q: {question}")
            print(f"  Gold: {sorted(gold)}")
            print(f"  Top-1: {pred_topk[0] if pred_topk else 'N/A'}")
            print(f"  Top-{top_k} (前5): {pred_topk[:5]}")
            print(f"  {'OK' if is_full_recall else 'FAIL'}")

    full_recall = success / total if total else 0.0
    top1_acc = top1_hits / total if total else 0.0

    print(f"\n{'=' * 50}")
    print(f"{split.upper()} Top-{top_k} Full Recall: {full_recall:.2%} ({success}/{total})")
    print(f"{split.upper()} Top-1 Hit Rate: {top1_acc:.2%} ({top1_hits}/{total})")


if __name__ == "__main__":
    # 在这里切换评估集：可选 "val" 或 "test"
    EVAL_SPLIT = "val"
    evaluate(split=EVAL_SPLIT, top_k=TOP_K)
