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
from pipeline.utils import debug_print


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
        if self.null_ratio > 0.3:
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
        始终输出类型 + 示例值（至少 2 个，已在 _profile_column 中补齐）。
        数值列额外附加范围。空值率偏高时额外提示。
        """
        parts: list[str] = []
        parts.append(f"类型={self.dtype_inferred}")
        if self.null_ratio > 0.3:
            parts.append(f"空值率={self.null_ratio:.0%}")
        if self.sample_values:
            parts.append(f"示例={'/'.join(self.sample_values)}")
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

    def _profile_column(self, col: str) -> ColumnProfile:
        p = ColumnProfile(col)
        series = self.df[col]
        p.total = len(series)
        p.null_count = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
        p.null_ratio = p.null_count / p.total if p.total else 0.0

        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        p.distinct_count = int(non_null.nunique())

        # 类型推断
        numeric_count = non_null.apply(self._is_numeric).sum()
        if numeric_count / max(len(non_null), 1) > 0.8:
            p.dtype_inferred = "NUMERIC"
            nums = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(nums):
                p.min_val = round(float(nums.min()), 2)
                p.max_val = round(float(nums.max()), 2)
        else:
            p.dtype_inferred = "TEXT"

        # 是否为枚举 / 分类列
        p.is_categorical = p.distinct_count <= settings.profile_distinct_threshold

        # Top 频率值（最多取 10 个，用于 to_summary 等统计展示）
        value_counts = non_null.value_counts().head(10)
        p.top_values = [(str(v), int(c)) for v, c in value_counts.items()]

        # ── 示例值（供 schema 注入）──────────────────────────────────────
        # 规则：优先使用高频值；若高频值不足 2 个，从剩余唯一值中随机补齐至 2 个。
        # 最终保留不超过 6 个值，截断过长的单个值（> 25 字符），避免 prompt 过宽。
        _MAX_SAMPLE_LEN = 25
        _TARGET_MIN = 2
        _TARGET_MAX = 6

        top_strs = [v for v, _ in p.top_values[:_TARGET_MAX]]
        if len(top_strs) < _TARGET_MIN:
            # 从不在高频列表中的唯一值里随机采样补充
            top_set = set(top_strs)
            extra_pool = [
                v for v in non_null.unique().tolist()
                if v not in top_set
            ]
            needed = _TARGET_MIN - len(top_strs)
            top_strs += random.sample(extra_pool, min(needed, len(extra_pool)))
        # 截断过长值，避免占用过多 token
        p.sample_values = [v[:_MAX_SAMPLE_LEN] for v in top_strs[:_TARGET_MAX]]

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

    def get_categorical_values(self, profiles: list[ColumnProfile] | None = None) -> dict[str, list[str]]:
        """返回所有分类列的合法值列表，供 Refiner 做 Literal 校验。"""
        if profiles is None:
            profiles = self.profile_all()
        result = {}
        for p in profiles:
            if p.is_categorical and p.top_values:
                result[p.name] = [v for v, _ in p.top_values]
        return result
