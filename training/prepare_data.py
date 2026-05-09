"""
CrossEncoder 训练数据准备 —— 从标注 CSV 生成 train/val/test JSON。
──────────────────────────────────────────────────────────────────
难负例采用**双通道**采样（论文新增）：
  - 通道 A: 字符 / token 相似度（SequenceMatcher）
  - 通道 B: 语义相似度（SentenceTransformer 嵌入余弦）
两路各取 top-K，合并去重 → 既覆盖"工厂描述 / 工厂编码"这类**字面相近**易混对，
也覆盖"供应商描述 / 工厂描述 / 中标厂家"这类**语义相近**易混对。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

# 输出路径
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_train_data.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_val_data.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_test_data.json")

NUM_HARD_NEGATIVES_CHAR = 2     # 字符相似度通道
NUM_HARD_NEGATIVES_SEMANTIC = 2  # 语义相似度通道
NUM_EASY_NEGATIVES = 6


# ──────────── 模板解析 ────────────

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


# ──────────── 相似度工具 ────────────

def get_similarity(a: str, b: str) -> float:
    """字符级 SequenceMatcher 比例。"""
    return SequenceMatcher(None, a, b).ratio()


_EMBED_MODEL = None  # 全局缓存，避免重复加载
_COL_EMBED_CACHE: dict[str, np.ndarray] = {}


def _get_embed_model():
    """懒加载 SentenceTransformer，复用 settings.embed_model。"""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[WARN] 未安装 sentence-transformers，语义相似度通道将退化为字符相似度。")
        return None
    embed_path = str(settings.embed_model)
    if not os.path.isdir(embed_path):
        print(f"[WARN] Embed 模型路径不存在: {embed_path}，语义通道关闭。")
        return None
    print(f"[Embed] 加载 SentenceTransformer 用于难负例语义采样: {embed_path}")
    _EMBED_MODEL = SentenceTransformer(embed_path, trust_remote_code=True)
    return _EMBED_MODEL


def _embed_columns(all_cols: list[str]) -> dict[str, np.ndarray] | None:
    """对所有列名做一次嵌入并缓存。失败则返回 None，调用方退化为单通道。"""
    if _COL_EMBED_CACHE:
        return _COL_EMBED_CACHE
    model = _get_embed_model()
    if model is None:
        return None
    try:
        vecs = model.encode(all_cols, convert_to_numpy=True, show_progress_bar=False)
        # L2 归一化便于直接点积当余弦
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        for col, v in zip(all_cols, vecs):
            _COL_EMBED_CACHE[col] = v
        return _COL_EMBED_CACHE
    except Exception as e:
        print(f"[WARN] 列嵌入失败({type(e).__name__}: {e})，语义通道退化。")
        return None


# ──────────── 难负例 ────────────

def generate_negatives(target_cols, all_cols, num_hard_char, num_hard_sem, num_easy):
    """
    双通道难负例采样：
      - char_top: SequenceMatcher 比例 + 关键词匹配加成
      - sem_top : 嵌入余弦相似度
    最终 hard = unique(char_top + sem_top)，easy 从剩余里随机采样。
    """
    potential = [c for c in all_cols if c not in target_cols]
    if not potential:
        return []

    # —— 通道 A：字符相似度 ——
    keywords = ["金额", "价", "数量", "码", "日期", "名称", "描述", "编码"]
    char_scored: list[tuple[str, float]] = []
    for neg in potential:
        max_sim = max((get_similarity(neg, t) for t in target_cols), default=0.0)
        if any(kw in t for t in target_cols for kw in keywords if kw in neg):
            max_sim += 0.2
        char_scored.append((neg, max_sim))
    char_scored.sort(key=lambda x: x[1], reverse=True)
    char_top = [c for c, _ in char_scored[:num_hard_char]]

    # —— 通道 B：语义相似度（嵌入余弦） ——
    sem_top: list[str] = []
    embeds = _embed_columns(all_cols)
    if embeds is not None:
        try:
            target_vecs = np.stack([embeds[t] for t in target_cols if t in embeds])
            if len(target_vecs):
                sem_scored: list[tuple[str, float]] = []
                for neg in potential:
                    if neg not in embeds:
                        continue
                    sim = float(np.max(target_vecs @ embeds[neg]))
                    sem_scored.append((neg, sim))
                sem_scored.sort(key=lambda x: x[1], reverse=True)
                sem_top = [c for c, _ in sem_scored[:num_hard_sem]]
        except Exception as e:
            print(f"[WARN] 语义相似度采样异常: {type(e).__name__}: {e}")

    # 合并去重
    hard: list[str] = []
    seen: set[str] = set()
    for c in char_top + sem_top:
        if c not in seen:
            hard.append(c)
            seen.add(c)
    # 字符通道失效时，至少凑够 num_hard_char 个
    if len(hard) < num_hard_char:
        for c, _ in char_scored:
            if c not in seen:
                hard.append(c)
                seen.add(c)
            if len(hard) >= num_hard_char:
                break

    remaining = [c for c in potential if c not in seen]
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
            negatives = generate_negatives(
                valid, all_cols,
                num_hard_char=NUM_HARD_NEGATIVES_CHAR,
                num_hard_sem=NUM_HARD_NEGATIVES_SEMANTIC,
                num_easy=NUM_EASY_NEGATIVES,
            )
            for col in negatives:
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
