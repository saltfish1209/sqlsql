"""
CrossEncoder 训练数据准备 —— 从标注 CSV 生成 train/val/test JSON。
──────────────────────────────────────────────────────────────────
与原项目 cross_encoder_data.py 等价，统一使用 config.settings 管理路径与参数。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import pandas as pd
from difflib import SequenceMatcher

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

# 输出路径
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_train_data.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_val_data.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_test_data.json")

NUM_HARD_NEGATIVES = 3
NUM_EASY_NEGATIVES = 7


def extract_cols_from_template(template_str):
    if pd.isna(template_str):
        return []
    matches = re.findall(r"\{([^}]+)\}", str(template_str))
    cleaned = []
    for m in matches:
        core = m.split("|")[0].strip()
        for sub in core.split("，"):
            f = sub.split(",")[0].strip()
            if f:
                cleaned.append(f)
    return list(set(cleaned))


def get_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def generate_negatives(target_cols, all_cols, num_hard, num_easy):
    potential = [c for c in all_cols if c not in target_cols]
    if not potential:
        return []
    candidates = []
    keywords = ["金额", "价", "数量", "码", "日期", "名称"]
    for neg in potential:
        max_sim = max((get_similarity(neg, t) for t in target_cols), default=0)
        if any(kw in t for t in target_cols for kw in keywords if kw in neg):
            max_sim += 0.2
        candidates.append((neg, max_sim))
    candidates.sort(key=lambda x: x[1], reverse=True)
    hard = [x[0] for x in candidates[:num_hard]]
    remaining = [c for c in potential if c not in hard]
    easy = random.sample(remaining, min(len(remaining), num_easy))
    return hard + easy


def process_data(df_slice, all_cols, is_training=True):
    dataset = []
    for _, row in df_slice.iterrows():
        question = row.get("生成问题", "")
        if pd.isna(question) or str(question).strip() == "":
            continue
        pos_cols = list(set(
            extract_cols_from_template(row.get("问题模版", ""))
            + extract_cols_from_template(row.get("回答模版", ""))
        ))
        valid = [c for c in pos_cols if c in all_cols]
        if not valid:
            continue
        if is_training:
            for col in valid:
                dataset.append({"question": question, "column": col, "label": 1})
            for col in generate_negatives(valid, all_cols, NUM_HARD_NEGATIVES, NUM_EASY_NEGATIVES):
                dataset.append({"question": question, "column": col, "label": 0})
        else:
            dataset.append({"question": question, "gold_columns": valid})
    return dataset


def main():
    print("开始处理数据...")
    raw_schema_file = str(settings.csv_path)
    train_file = str(settings.train_csv)

    for label, fpath in [("CSV数据", raw_schema_file), ("训练数据", train_file)]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] {label}文件不存在: {fpath}")
            return

    df_raw = pd.read_csv(raw_schema_file, nrows=1)
    ALL_COLUMNS = [str(c).strip() for c in df_raw.columns]

    df_full = pd.read_csv(
        train_file,
        usecols=["生成问题", "问题模版", "回答模版", "SQL验证状态"],
        engine="python",
        encoding="utf-8",
    )
    df_full = df_full[df_full["SQL验证状态"] == "MATCH"].copy()
    df_full = df_full.sample(frac=1, random_state=settings.random_state).reset_index(drop=True)

    n = len(df_full)
    train_end = int(n * settings.train_split)
    val_end = int(n * (settings.train_split + settings.val_split))

    df_train = df_full.iloc[:train_end]
    df_val = df_full.iloc[train_end:val_end]
    df_test = df_full.iloc[val_end:]

    print(f"训练集: {len(df_train)} | 验证集: {len(df_val)} | 测试集: {len(df_test)}")

    for data, path, is_train in [
        (df_train, TRAIN_FILE, True),
        (df_val, VAL_FILE, False),
        (df_test, TEST_FILE, False),
    ]:
        result = process_data(data, ALL_COLUMNS, is_training=is_train)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  → {path} ({len(result)} 条)")


if __name__ == "__main__":
    main()
