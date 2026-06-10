"""
自动数据库 Profiling（论文核心新增模块）
────────────────────────────────────────
受 BIRD 榜首论文 "Automatic Metadata Extraction" 启发，
对数据库进行自动化分析，生成以下元数据：

1. 列级统计信息 (Column Profile)
   - 数据类型推断、NULL 率、唯一值数量、高频值 Top-K
   - 值格式模式检测（日期、编码、混合型等）

2. 列间关系 (Cross-Column Insights)
   - 疑似外键 / 引用关系
   - 值域重叠检测

3. 增强 Schema 描述
   - 基于统计特征自动补充列描述中缺失的隐式业务语义

这些元数据将被注入 Generator Prompt，提升 SQL 生成的准确性。
"""
from __future__ import annotations

import os
import random
import re
from collections import Counter
from typing import Any

import pandas as pd

from config.settings import settings
from pipeline.column_types import NUMERIC_COLS_SET
from pipeline.utils import debug_print


def _format_numeric_bound(value: Any) -> str:
    if value is None:
        return ""
    num = float(value)
    if num == int(num):
        return str(int(num))
    return str(value)


def _format_numeric_range(min_val: Any, max_val: Any) -> str:
    return f"[{_format_numeric_bound(min_val)},{_format_numeric_bound(max_val)}]"


class ColumnProfile:
    """单列的统计 Profile。"""
    __slots__ = (
        "name", "dtype_inferred", "total", "null_count", "null_ratio",
        "distinct_count", "top_values", "sample_values", "format_pattern",
        "is_categorical", "min_val", "max_val",
    )

    def __init__(self, name: str):
        self.name = name
        self.dtype_inferred = "TEXT"
        self.total = 0
        self.null_count = 0
        self.null_ratio = 0.0
        self.distinct_count = 0
        self.top_values: list[tuple[str, int]] = []
        # 供 schema 注入的示例值列表：优先高频值，不足 2 个时补随机值，保证至少 2 个
        self.sample_values: list[str] = []
        self.format_pattern = ""
        self.is_categorical = False
        self.min_val: Any = None
        self.max_val: Any = None

    def to_summary(self) -> str:
        """生成自然语言摘要（含列名），供独立 profile 预览脚本使用。"""
        parts = [f"「{self.name}」"]
        parts.append(f"类型={self.dtype_inferred}")
        if self.null_ratio > 0:
            parts.append(f"空值率={self.null_ratio:.0%}")
        parts.append(f"唯一值={self.distinct_count}")
        if self.is_categorical and self.top_values:
            vals = [v for v, _ in self.top_values[:5]]
            parts.append(f"常见值=[{', '.join(vals)}]")
        if self.format_pattern:
            parts.append(f"格式={self.format_pattern}")
        if self.min_val is not None and self.max_val is not None:
            parts.append(f"范围=[{self.min_val}, {self.max_val}]")
        return " | ".join(parts)

    def to_inline_summary(self) -> str:
        """
        生成内联摘要（不含列名），直接拼接到 schema 字段描述末尾。
        始终输出类型 + 是否枚举 + 示例值（至少 2 个，已在 _profile_column 中补齐）。
        - 低基数枚举列（distinct ≤ enum_full_threshold）会**全部列出取值**，
          这样 Generator 才能识别到 "招标模式 ∈ {总部直接组织实施, 公司集中招标, …}" 这类
          细粒度过滤值（error.txt 案例 17 的根因）。
        - 数值列额外附加范围。空值率偏高时额外提示。
        """
        parts: list[str] = []
        parts.append(f"字段类型={self.dtype_inferred}")
        if self.null_ratio > 0:
            parts.append(f"空值率={self.null_ratio:.0%}")
        full_threshold = settings.profile_enum_full_threshold
        if self.distinct_count <= 20 and self.top_values:
            all_vals = [v for v, _ in self.top_values[: self.distinct_count]]
            if 1 < self.distinct_count <= full_threshold:
                parts.append(f"枚举值={'/'.join(all_vals)}")
            elif self.sample_values:
                parts.append(f"示例={'/'.join(self.sample_values[:settings.profile_example_k])}")
        elif self.sample_values:
            parts.append(f"示例={'/'.join(self.sample_values[:settings.profile_example_k])}")
        if self.dtype_inferred == "NUMERIC" and self.min_val is not None and self.max_val is not None:
            parts.append(f"范围=[{self.min_val},{self.max_val}]")
        if self.format_pattern:
            parts.append(f"格式={self.format_pattern}")
        return "[" + ", ".join(parts) + "]"


# ─── 格式检测正则 ───
_DATE_SLASH = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")
_DATE_DASH = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_PURE_DIGITS = re.compile(r"^\d+$")
_CODE_PATTERN = re.compile(r"^[A-Za-z]\d+$")


def _pick_k_from_list(values: list[str], k: int, rng: random.Random) -> list[str]:
    pool = [str(v).strip() for v in values if str(v).strip()]
    if not pool or k <= 0:
        return []
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def _pick_by_frequency_tiers(top_values: list[tuple[str, int]], k: int, rng: random.Random) -> list[str]:
    if not top_values or k <= 0:
        return []
    by_freq: dict[int, list[str]] = {}
    for val, cnt in top_values:
        text = str(val).strip()
        if not text:
            continue
        by_freq.setdefault(int(cnt), []).append(text)
    result: list[str] = []
    for freq in sorted(by_freq.keys(), reverse=True):
        tier = by_freq[freq]
        needed = k - len(result)
        if needed <= 0:
            break
        if len(tier) <= needed:
            result.extend(tier)
        else:
            result.extend(rng.sample(tier, needed))
    return result[:k]


def pick_k_instance_values(
    *,
    is_enum: bool,
    enum_values: list[str],
    top_values: list[tuple[str, int]],
    fallback_pool: list[str],
    k: int | None = None,
    rng: random.Random | None = None,
    max_len: int = 25,
) -> list[str]:
    """为候选列选取 k 个实例值：枚举列取枚举值，非枚举列按频次优先、同频随机。"""
    rng = rng or random.Random()
    k = max(1, int(k if k is not None else settings.profile_example_k))

    if is_enum and enum_values:
        pool = _pick_k_from_list(enum_values, k, rng)
    else:
        pool = _pick_by_frequency_tiers(top_values, k, rng)

    if len(pool) < k and fallback_pool:
        existing = set(pool)
        extras = [
            str(v).strip()
            for v in fallback_pool
            if str(v).strip() and str(v).strip() not in existing
        ]
        needed = k - len(pool)
        if extras:
            pool.extend(rng.sample(extras, min(needed, len(extras))))

    return [v[:max_len] for v in pool[:k]]


def apply_instance_fields(
    item: dict,
    profile_detail: dict | None = None,
    *,
    k: int | None = None,
    rng: random.Random | None = None,
) -> None:
    """枚举列只保留枚举值，非枚举列只保留示例值（二者互斥）。"""
    prof = profile_detail or {}
    enum_values = prof.get("枚举值") if prof.get("枚举值") not in (None, "", []) else item.get("枚举值")
    if enum_values not in (None, "", []):
        item["枚举值"] = enum_values
        item.pop("示例值", None)
        return
    item.pop("枚举值", None)
    examples = resolve_column_examples(item, prof, k=k, rng=rng)
    if examples:
        item["示例值"] = examples
    else:
        item.pop("示例值", None)


def resolve_column_examples(
    item: dict,
    profile_detail: dict | None = None,
    *,
    k: int | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """基于 profiler 明细或已合并列项，解析出 k 个实例值。"""
    k = max(1, int(k if k is not None else settings.profile_example_k))
    prof = profile_detail or {}
    is_enum = str(prof.get("是否枚举") or item.get("是否枚举") or "").strip() == "是"
    enum_values = prof.get("枚举值") if prof.get("枚举值") not in (None, "", []) else item.get("枚举值")
    if is_enum and enum_values not in (None, "", []):
        enum_list = enum_values if isinstance(enum_values, list) else [enum_values]
        return pick_k_instance_values(
            is_enum=True,
            enum_values=[str(v) for v in enum_list],
            top_values=[],
            fallback_pool=[],
            k=k,
            rng=rng,
        )

    raw_examples = prof.get("示例值") if prof.get("示例值") not in (None, "", []) else item.get("示例值")
    if raw_examples in (None, "", []):
        return []
    if isinstance(raw_examples, list):
        pool = [str(v).strip() for v in raw_examples if str(v).strip()]
    elif isinstance(raw_examples, str):
        pool = [x.strip() for x in raw_examples.split("/") if x.strip()] if "/" in raw_examples else ([raw_examples.strip()] if raw_examples.strip() else [])
    else:
        pool = [str(raw_examples).strip()]
    if len(pool) >= k:
        return pool[:k]
    return pool


def _detect_format(values: list[str]) -> str:
    """对采样值检测主要格式模式。"""
    if not values:
        return ""
    sample = values[:50]
    counters: dict[str, int] = Counter()
    for v in sample:
        v = v.strip()
        if _DATE_SLASH.match(v):
            counters["YYYY/MM/DD"] += 1
        elif _DATE_DASH.match(v):
            counters["YYYY-MM-DD"] += 1
        elif _PURE_DIGITS.match(v):
            counters["纯数字"] += 1
        elif _CODE_PATTERN.match(v):
            counters["编码(字母+数字)"] += 1
    if not counters:
        return ""
    top_fmt, cnt = counters.most_common(1)[0]
    if cnt / len(sample) > 0.5:
        return top_fmt
    return ""


class DatabaseProfiler:
    """对内存 SQLite 或 DataFrame 做自动 Profiling。"""

    def __init__(self, df: pd.DataFrame | None = None, csv_path: str | None = None):
        if df is not None:
            self.df = df
        elif csv_path:
            csv_path = str(csv_path)
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
            self.df = pd.read_csv(csv_path, dtype=str, nrows=settings.profile_sample_rows * 100)
        else:
            raise ValueError("需要提供 df 或 csv_path")

    def profile_all(self) -> list[ColumnProfile]:
        profiles = []
        for col in self.df.columns:
            profiles.append(self._profile_column(col))
        debug_print(f"[Profiler] 完成 {len(profiles)} 列的自动 Profiling")
        return profiles

    def get_profile_detail_map(self, profiles: list[ColumnProfile] | None = None) -> dict[str, dict]:
        if profiles is None:
            profiles = self.profile_all()
        detail: dict[str, dict] = {}
        for p in profiles:
            enum_values = [v for v, _ in p.top_values] if p.distinct_count <= settings.profile_enum_full_threshold else []
            row: dict[str, Any] = {
                "字段类型": p.dtype_inferred,
            }
            if p.distinct_count <= 20:
                row["唯一值数"] = p.distinct_count
            if p.format_pattern:
                row["格式"] = p.format_pattern
            if enum_values:
                row["枚举值"] = enum_values
                row["是否枚举"] = "是"
            elif p.sample_values:
                row["示例值"] = p.sample_values[: settings.profile_example_k]
            if p.null_ratio > 0:
                row["空值率"] = f"{p.null_ratio:.2%}" if p.total else "0.00%"
            if p.name in NUMERIC_COLS_SET and p.min_val is not None and p.max_val is not None:
                row["范围"] = _format_numeric_range(p.min_val, p.max_val)
            detail[p.name] = row
        return detail

    def _profile_column(self, col: str) -> ColumnProfile:
        p = ColumnProfile(col)
        series = self.df[col]
        p.total = len(series)
        p.null_count = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
        p.null_ratio = p.null_count / p.total if p.total else 0.0

        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        p.distinct_count = int(non_null.nunique())

        # 类型推断：仅白名单数值列标为 NUMERIC，业务编码列保持 TEXT
        if col in NUMERIC_COLS_SET:
            p.dtype_inferred = "NUMERIC"
            nums = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(nums):
                p.min_val = round(float(nums.min()), 2)
                p.max_val = round(float(nums.max()), 2)
        else:
            p.dtype_inferred = "TEXT"

        # 是否为枚举 / 分类列
        p.is_categorical = p.distinct_count <= settings.profile_distinct_threshold

        # Top 频率值：低基数枚举列保留全部，其它列保留 top-10
        full_threshold = settings.profile_enum_full_threshold
        head_n = max(10, full_threshold) if (
            p.is_categorical and p.distinct_count <= full_threshold
        ) else 10
        value_counts = non_null.value_counts().head(head_n)
        p.top_values = [(str(v), int(c)) for v, c in value_counts.items()]

        # ── 示例值（供 schema 注入）──────────────────────────────────────
        # 枚举列取枚举值；非枚举列按频次优先，同频随机；默认 k=settings.profile_example_k。
        is_enum = p.is_categorical and p.distinct_count <= full_threshold
        enum_values = [v for v, _ in p.top_values] if is_enum else []
        if is_enum and enum_values:
            p.sample_values = []
        else:
            p.sample_values = pick_k_instance_values(
                is_enum=False,
                enum_values=[],
                top_values=p.top_values,
                fallback_pool=non_null.unique().tolist(),
                k=settings.profile_example_k,
            )

        # 格式检测
        p.format_pattern = _detect_format(non_null.head(50).tolist())

        return p

    @staticmethod
    def _is_numeric(val: str) -> bool:
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def generate_profile_text(self, profiles: list[ColumnProfile] | None = None) -> str:
        """生成可注入 prompt 的 Profile 文本摘要。"""
        if profiles is None:
            profiles = self.profile_all()
        lines = ["[数据库 Profile 统计]"]
        for p in profiles:
            lines.append(f"  {p.to_summary()}")
        return "\n".join(lines)

    def get_profile_map(self, profiles: list[ColumnProfile] | None = None) -> dict[str, str]:
        """
        返回 {列名: 内联摘要字符串} 字典。
        供 build_m_schema_prompt 按选中列逐行注入，不做全量拼接。
        """
        if profiles is None:
            profiles = self.profile_all()
        return {p.name: p.to_inline_summary() for p in profiles}

    def get_profile_name_map(self, profiles: list[ColumnProfile] | None = None) -> dict[str, ColumnProfile]:
        """返回 {列名: ColumnProfile}，便于读取类型/枚举标记等结构化信息。"""
        if profiles is None:
            profiles = self.profile_all()
        return {p.name: p for p in profiles}

    def get_categorical_values(self, profiles: list[ColumnProfile] | None = None) -> dict[str, list[str]]:
        """返回所有分类列的合法值列表，供 Refiner 做 Literal 校验。"""
        if profiles is None:
            profiles = self.profile_all()
        result = {}
        for p in profiles:
            if p.is_categorical and p.top_values:
                result[p.name] = [v for v, _ in p.top_values]
        return result
