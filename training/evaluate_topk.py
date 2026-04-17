"""
CrossEncoder Top-K 全覆盖率评估 —— 验证 Schema Pruner 的召回能力。
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

TEST_FILE = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.json")
TOP_K = 15


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
    all_cols = [str(c).strip() for c in df_raw.columns]
    print(f"候选列数: {len(all_cols)}")

    if not os.path.isfile(TEST_FILE):
        print(f"[ERROR] 测试数据文件不存在: {TEST_FILE}，请先运行 prepare_data.py")
        return
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    success = total = 0
    for i, item in enumerate(test_data):
        question = item["question"]
        gold = set(item["gold_columns"])
        if not gold:
            continue
        inputs = [[question, col] for col in all_cols]
        scores = model.predict(inputs, batch_size=32)
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        pred = set(all_cols[j] for j in top_idx)
        hit = gold.issubset(pred)
        if hit:
            success += 1
        total += 1
        if i < 3:
            print(f"\n[Case {i}] Q: {question}")
            print(f"  Gold: {gold}")
            print(f"  Top-{TOP_K}: {list(pred)[:5]} ...")
            print(f"  {'OK' if hit else 'FAIL'}")

    acc = success / total if total else 0
    print(f"\n{'='*50}")
    print(f"Top-{TOP_K} Full Recall: {acc:.2%} ({success}/{total})")


if __name__ == "__main__":
    evaluate()
