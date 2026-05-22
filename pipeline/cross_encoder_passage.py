from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pipeline.profiler import DatabaseProfiler


def _read_schema_text(schema_path_or_text: str) -> str:
    schema_path_or_text = str(schema_path_or_text)
    if os.path.isfile(schema_path_or_text):
        return Path(schema_path_or_text).read_text(encoding="utf-8")
    return schema_path_or_text


def parse_schema_metadata(schema_path_or_text: str) -> list[dict]:
    text = _read_schema_text(schema_path_or_text).strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        items = data.get("columns") or data.get("字段") or data.get("schema") or [] if isinstance(data, dict) else data
        if isinstance(items, list):
            rows: list[dict] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                col_name = str(item.get("column_name") or item.get("字段名") or item.get("列名") or "").strip()
                if not col_name:
                    continue
                examples = item.get("examples") or item.get("示例值") or item.get("枚举值") or []
                if not isinstance(examples, list):
                    examples = [examples]
                rows.append({
                    "column_name": col_name,
                    "data_type": str(item.get("data_type") or item.get("字段类型") or item.get("column_type") or "TEXT").strip(),
                    "column_description": str(item.get("column_description") or item.get("字段描述") or item.get("描述") or "").strip(),
                    "examples": [str(v).strip() for v in examples if str(v).strip()],
                    "raw_text": json.dumps(item, ensure_ascii=False),
                })
            return rows
    except Exception:
        pass

    rows = []
    pattern = re.compile(r'\s*\(([^:]+):\s*([^,]+),\s*(.*?),\s*Examples:\s*\[(.*?)\]\)')
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue
        m = pattern.search(line)
        if not m:
            continue
        rows.append({
            "column_name": m.group(1).strip(),
            "data_type": m.group(2).strip(),
            "column_description": m.group(3).strip(),
            "examples": [e.strip() for e in m.group(4).split(",") if e.strip()],
            "raw_text": line,
        })
    return rows


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def build_column_passage(meta: dict, profile_detail: dict | None = None) -> str:
    col = str(meta.get("column_name") or meta.get("列名") or "").strip()
    desc = str(meta.get("column_description") or meta.get("列描述") or "").strip()
    profile_detail = profile_detail or {}

    examples: list[str] = []
    examples.extend(_as_list(meta.get("examples") or meta.get("示例值")))
    examples.extend(_as_list(profile_detail.get("枚举值")))
    examples.extend(_as_list(profile_detail.get("示例值")))
    examples = list(dict.fromkeys(examples))

    parts = [f"列名称: {col}"]
    if desc:
        parts.append(f"列描述: {desc}")
    if examples:
        parts.append(f"实例值: {'/'.join(examples[:6])}")
    return " | ".join(parts)


def build_column_passage_map(
    schema_path_or_text: str,
    csv_path: str | None = None,
    *,
    active_columns: list[str] | None = None,
    profile_detail_map: dict[str, dict] | None = None,
) -> dict[str, str]:
    metadata = parse_schema_metadata(schema_path_or_text)
    active_set = set(active_columns) if active_columns is not None else None

    if profile_detail_map is None and csv_path:
        profile_detail_map = DatabaseProfiler(csv_path=csv_path).get_profile_detail_map()
    profile_detail_map = profile_detail_map or {}

    passage_map: dict[str, str] = {}
    for meta in metadata:
        col = str(meta.get("column_name") or "").strip()
        if not col:
            continue
        if active_set is not None and col not in active_set:
            continue
        passage_map[col] = build_column_passage(meta, profile_detail_map.get(col, {}))
    return passage_map
