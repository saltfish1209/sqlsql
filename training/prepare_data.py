"""
CrossEncoder 训练数据准备 —— 从标注 CSV 生成 train/val/test JSON。
──────────────────────────────────────────────────────────────────
每条样本的第二段文本为「列名 + m_schema 描述 + profiler/schema 示例值」拼接，
与 pipeline/schema_linker 中 CrossEncoder 推理输入一致（见 pipeline/cross_encoder_passage.py）。
──────────────────────────────────────────────────────────────────
负样本按每个正字段独立采样：
  - 2 个向量相似列（SentenceTransformer 余弦）
  - 3 个字符串相似列（0.7 * 序列相似度 + 0.3 * 字符 Jaccard）
  - 1 个除此以外的随机列
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
from pipeline.cross_encoder_passage import build_column_passage_map
from pipeline.profiler import DatabaseProfiler
from training.dataset_io import load_split_dataframes, normalize_cell_text, write_jsonl
from training.template_split import split_dataframe_by_template

# 输出路径（统一改为 JSONL，避免单文件 JSON 写入被截断导致 UnicodeDecodeError）
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_train_data.jsonl")
TRAIN_EVAL_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_train_eval_data.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_val_data.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_test_data.jsonl")
FILTER_REPORT_FILE = os.path.join(OUTPUT_DIR, "cross_encoder_column_filter_report.json")

NUM_HARD_NEGATIVES_CHAR = 3
NUM_HARD_NEGATIVES_SEMANTIC = 2
NUM_EASY_NEGATIVES = 1


def _normalize_column_name(name: str) -> str:
    """列名归一化：与 dataset_io 完全一致。"""
    return normalize_cell_text(name)


def _parse_m_schema_columns(schema_path: str) -> list[str]:
    if not os.path.isfile(schema_path):
        return []
    with open(schema_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            items = obj.get("columns") or obj.get("字段") or obj.get("schema") or []
        else:
            items = obj
        cols: list[str] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    raw_col = item.get("column_name") or item.get("字段名") or item.get("列名") or ""
                elif isinstance(item, str):
                    raw_col = item
                else:
                    continue
                col = _normalize_column_name(raw_col)
                if col:
                    cols.append(col)
            if cols:
                return list(dict.fromkeys(cols))
    except Exception:
        pass
    cols = []
    pattern = re.compile(r"\(([^:]+):")
    for line in text.splitlines():
        m = pattern.search(line.strip())
        if m:
            col = _normalize_column_name(m.group(1))
            if col:
                cols.append(col)
    return list(dict.fromkeys(cols))


def _parse_ratio(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        val = float(value)
        return val if val <= 1 else val / 100.0
    s = str(value).strip()
    if not s:
        return 0.0
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return 0.0
    try:
        val = float(s)
        return val if val <= 1 else val / 100.0
    except ValueError:
        return 0.0


def resolve_active_columns(
    schema_columns: list[str],
    raw_columns: list[str],
    profile_detail_map: dict[str, dict],
    null_ratio_threshold: float | None = None,
) -> tuple[list[str], dict]:
    if null_ratio_threshold is None:
        null_ratio_threshold = float(getattr(settings, "deprecated_column_null_ratio_threshold", 0.95))
    schema_set = {_normalize_column_name(c) for c in schema_columns if c}
    raw_set = {_normalize_column_name(c) for c in raw_columns if c}
    in_schema_not_in_raw = sorted(schema_set - raw_set)
    high_null_columns: list[str] = []
    for col in sorted(schema_set & raw_set):
        detail = profile_detail_map.get(col, {})
        null_ratio = _parse_ratio(detail.get("空值率"))
        if null_ratio >= null_ratio_threshold:
            high_null_columns.append(col)
    deprecated = set(in_schema_not_in_raw) | set(high_null_columns)
    active = sorted((schema_set & raw_set) - deprecated)
    report = {
        "null_ratio_threshold": null_ratio_threshold,
        "schema_columns_count": len(schema_columns),
        "raw_columns_count": len(raw_columns),
        "active_columns_count": len(active),
        "deprecated_columns_count": len(deprecated),
        "deprecated": {
            "schema_not_in_raw": in_schema_not_in_raw,
            "high_null_ratio": high_null_columns,
        },
        "active_columns": active,
    }
    return active, report


def print_column_filter_report(
    report: dict,
    *,
    header: str = "[CrossEncoder][列过滤策略]",
) -> None:
    """打印列过滤统计及废弃列具体名单。"""
    deprecated = report.get("deprecated") or {}
    schema_missing = list(deprecated.get("schema_not_in_raw") or [])
    high_null = list(deprecated.get("high_null_ratio") or [])
    threshold = float(report.get("null_ratio_threshold", 0.95))
    threshold_pct = f"{threshold * 100:.0f}%" if threshold <= 1 else str(threshold)

    print(header)
    print(
        f"  有效列: {report.get('active_columns_count', 0)} | "
        f"废弃列: {report.get('deprecated_columns_count', 0)}"
    )
    print(
        f"  - m_schema有但原始数据缺失 ({len(schema_missing)}): "
        f"{('、'.join(schema_missing) if schema_missing else '无')}"
    )
    print(
        f"  - profiler空值率>={threshold_pct} ({len(high_null)}): "
        f"{('、'.join(high_null) if high_null else '无')}"
    )
    if int(report.get("deprecated_columns_count", 0)) > 0:
        all_deprecated = sorted(set(schema_missing) | set(high_null))
        print(f"  废弃列合计 ({len(all_deprecated)}): {'、'.join(all_deprecated)}")
    if schema_missing:
        print(
            "  [说明]「m_schema有但原始数据缺失」比对的是 config.settings.csv_path "
            "（业务宽表 CSV 表头），不是 train_dataset_with_sql_and_slots.csv。"
        )
        for col in schema_missing:
            print(f"    · {col}  (归一化键: {_normalize_column_name(col)!r})")


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

def _normalize_lsh_text(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s)).upper()


def _char_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def get_similarity(a: str, b: str) -> float:
    """与 schema_linker LSH 模糊匹配一致的字符串组合相似度。"""
    na, nb = _normalize_lsh_text(a), _normalize_lsh_text(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    jac = _char_jaccard(na, nb)
    return 0.7 * seq + 0.3 * jac


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
    对每个正字段独立采样负字段，保留正负对应关系供 pairwise loss 使用。
    """
    embeds = _embed_columns(all_cols)
    groups: list[dict] = []
    target_set = set(target_cols)
    for pos in target_cols:
        potential = [c for c in all_cols if c not in target_set]
        if not potential:
            continue

        selected: list[str] = []
        sources: list[str] = []

        if embeds is not None and pos in embeds:
            try:
                sem_scored = [
                    (neg, float(np.asarray(embeds[pos]) @ np.asarray(embeds[neg])))
                    for neg in potential
                    if neg in embeds
                ]
                sem_scored.sort(key=lambda x: x[1], reverse=True)
                for neg, _ in sem_scored:
                    if neg not in selected:
                        selected.append(neg)
                        sources.append("semantic")
                    if sources.count("semantic") >= num_hard_sem:
                        break
            except Exception as e:
                print(f"[WARN] 语义相似度采样异常: {type(e).__name__}: {e}")

        char_pool = [c for c in potential if c not in selected]
        char_scored = [(neg, get_similarity(neg, pos)) for neg in char_pool]
        char_scored.sort(key=lambda x: x[1], reverse=True)
        for neg, _score in char_scored:
            selected.append(neg)
            sources.append("string")
            if sources.count("string") >= num_hard_char:
                break

        remaining = [c for c in potential if c not in selected]
        for neg in random.sample(remaining, min(len(remaining), num_easy)):
            selected.append(neg)
            sources.append("random")

        groups.append({
            "positive_column": pos,
            "negative_columns": selected,
            "negative_sources": sources,
        })
    return groups


def process_data(
    df_slice,
    all_cols,
    passage_map: dict[str, str],
    is_training=True,
):
    available_cols = [c for c in all_cols if c in passage_map]
    available_set = set(available_cols)
    dataset = []
    for _, row in df_slice.iterrows():
        question = row.get("生成问题", "")
        if pd.isna(question) or str(question).strip() == "":
            continue
        pos_cols = list(set(
            extract_cols_from_template(row.get("问题模版", ""))
            + extract_cols_from_template(row.get("回答模版", ""))
        ))
        valid = [_normalize_column_name(c) for c in pos_cols if _normalize_column_name(c) in available_set]
        if not valid:
            continue
        if is_training:
            for col in valid:
                dataset.append({
                    "question": question,
                    "column": passage_map[col],
                    "column_name": col,
                    "label": 1,
                })
            negative_groups = generate_negatives(
                valid, available_cols,
                num_hard_char=NUM_HARD_NEGATIVES_CHAR,
                num_hard_sem=NUM_HARD_NEGATIVES_SEMANTIC,
                num_easy=NUM_EASY_NEGATIVES,
            )
            for group in negative_groups:
                pos_col = group["positive_column"]
                for col, source in zip(group["negative_columns"], group["negative_sources"]):
                    dataset.append({
                        "question": question,
                        "column": passage_map[col],
                        "column_name": col,
                        "label": 0,
                        "pos_column": pos_col,
                        "neg_column": col,
                        "positive_column": passage_map[pos_col],
                        "negative_source": source,
                    })
        else:
            dataset.append({"question": question, "gold_columns": valid})
    return dataset


def main():
    print("开始处理数据...")
    raw_schema_file = str(settings.csv_path)
    train_file = str(settings.train_csv)
    schema_file = str(settings.schema_path)

    for label, fpath in [("CSV数据", raw_schema_file), ("训练数据", train_file), ("m_schema", schema_file)]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] {label}文件不存在: {fpath}")
            return

    df_raw = pd.read_csv(raw_schema_file, nrows=1)
    raw_columns = [_normalize_column_name(c) for c in df_raw.columns]
    schema_columns = _parse_m_schema_columns(schema_file)
    profiler = DatabaseProfiler(csv_path=raw_schema_file)
    profile_detail_raw = profiler.get_profile_detail_map(profiler.profile_all())
    profile_detail = {
        _normalize_column_name(col): detail
        for col, detail in profile_detail_raw.items()
        if _normalize_column_name(col)
    }
    ALL_COLUMNS, filter_report = resolve_active_columns(schema_columns, raw_columns, profile_detail)
    passage_map_raw = build_column_passage_map(schema_file, raw_schema_file)
    all_columns_set = set(ALL_COLUMNS)
    passage_map: dict[str, str] = {}
    for col, passage in passage_map_raw.items():
        norm_col = _normalize_column_name(col)
        if norm_col and norm_col in all_columns_set and norm_col not in passage_map:
            passage_map[norm_col] = passage
    with open(FILTER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(filter_report, f, ensure_ascii=False, indent=2)

    print_column_filter_report(filter_report, header="[CrossEncoder][列过滤策略] 已启用废弃列排除")
    print(f"  过滤报告已写入: {FILTER_REPORT_FILE}")

    splits = load_split_dataframes(settings.data_dir)
    if splits is not None:
        df_train, df_val, df_test = splits
        print(
            f"[Split] 已从 {settings.data_dir} 加载统一分割 "
            f"train_split.jsonl / val_split.jsonl / test_split.jsonl"
        )
    else:
        print(
            f"[Split][WARN] 未找到 {settings.data_dir}/train_split.jsonl 等统一分割文件，"
            "回退到按 CSV 重新切分。建议先运行 training/run_full_split_pipeline.py。"
        )
        df_full = pd.read_csv(
            train_file,
            usecols=["生成问题", "问题模版", "回答模版", "SQL验证状态"],
            engine="python",
            encoding="utf-8",
        )
        df_full = df_full[df_full["SQL验证状态"] == "MATCH"].copy()
        df_train, df_val, df_test = split_dataframe_by_template(
            df_full,
            template_col="问题模版",
            train_split=settings.train_split,
            val_split=settings.val_split,
        )

    print(f"训练集: {len(df_train)} | 验证集: {len(df_val)} | 测试集: {len(df_test)}")

    for data, path, is_train in [
        (df_train, TRAIN_FILE, True),
        (df_val, VAL_FILE, False),
        (df_test, TEST_FILE, False),
    ]:
        result = process_data(data, ALL_COLUMNS, passage_map, is_training=is_train)
        write_jsonl(path, result)
        print(f"  → {path} ({len(result)} 条)")

    # 额外输出一份“评估格式”的训练集：
    # 与 val/test 一致，每题一条，字段为 question + gold_columns。
    train_eval = process_data(df_train, ALL_COLUMNS, passage_map, is_training=False)
    write_jsonl(TRAIN_EVAL_FILE, train_eval)
    print(f"  → {TRAIN_EVAL_FILE} ({len(train_eval)} 条, 评估格式)")


if __name__ == "__main__":
    main()
