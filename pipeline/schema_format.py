from __future__ import annotations

from typing import Any

from config.settings import settings
from pipeline.profiler import apply_instance_fields


def _pick(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return ""


def _normalize_examples(value: Any, k: int | None = None) -> list[str]:
    limit = max(1, int(k if k is not None else settings.profile_example_k))
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()][:limit]
    if isinstance(value, str):
        if "/" in value:
            return [x.strip() for x in value.split("/") if x.strip()][:limit]
        text = value.strip()
        return [text] if text else []
    return [str(value)][:limit]


def enrich_schema_column(
    item: dict,
    column_metadata: list[dict],
    profile_detail_map: dict[str, dict] | None = None,
    *,
    example_k: int | None = None,
) -> dict:
    """合并 linker 列项与 profiler，输出本项目统一的中文字段结构。"""
    col = str(item.get("列名") or item.get("column_name") or "").strip()
    meta = next((m for m in column_metadata if m.get("column_name") == col), {})
    prof = (profile_detail_map or {}).get(col, {})

    enriched = dict(item)
    enriched["列名"] = col
    enriched["列描述"] = _pick(enriched.get("列描述"), meta.get("column_description"))
    enriched["字段类型"] = _pick(prof.get("字段类型"), enriched.get("字段类型"), meta.get("data_type"), "TEXT")
    enriched["是否枚举"] = _pick(prof.get("是否枚举"), enriched.get("是否枚举"))
    enriched["空值率"] = _pick(prof.get("空值率"), enriched.get("空值率"))
    enriched["唯一值数"] = _pick(prof.get("唯一值数"), enriched.get("唯一值数"))
    enriched["格式"] = _pick(prof.get("格式"), enriched.get("格式"))
    enriched["范围"] = _pick(prof.get("范围"), enriched.get("范围"))
    apply_instance_fields(enriched, prof, k=example_k)
    return enriched


def enrich_schema_columns(
    items: list[dict],
    column_metadata: list[dict],
    profile_detail_map: dict[str, dict] | None = None,
    *,
    example_k: int | None = None,
) -> list[dict]:
    return [
        enrich_schema_column(item, column_metadata, profile_detail_map, example_k=example_k)
        for item in (items or [])
        if str(item.get("列名") or item.get("column_name") or "").strip()
    ]


def build_light_schema_markdown(columns: list[dict], table_name: str) -> str:
    """论文 Figure 2 的 markdown 表格式，字段仍使用本项目中文键。"""
    lines = [
        f"## Table: {table_name}",
        "### Column information",
        "| 列名 | 字段类型 | 列描述 | 示例值 | 是否枚举 | 空值率 | 唯一值数 | 格式 | 范围 | 相关性分数 |",
        "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for col in columns:
        raw = col.get("枚举值") if col.get("枚举值") not in (None, "", []) else col.get("示例值")
        examples = _normalize_examples(raw)
        score = col.get("相关性分数")
        score_text = "" if score in (None, "") else str(score)
        lines.append(
            "| {列名} | {字段类型} | {列描述} | {示例值} | {是否枚举} | {空值率} | {唯一值数} | {格式} | {范围} | {相关性分数} |".format(
                列名=col.get("列名", ""),
                字段类型=col.get("字段类型", ""),
                列描述=col.get("列描述", ""),
                示例值=examples,
                是否枚举=col.get("是否枚举", ""),
                空值率=col.get("空值率", ""),
                唯一值数=col.get("唯一值数", ""),
                格式=col.get("格式", ""),
                范围=col.get("范围", ""),
                相关性分数=score_text,
            )
        )
    lines.extend(["### Primary keys", "[]", "### Foreign keys", "[]"])
    return "\n".join(lines)


def build_evidence_markdown(evidence: dict | None) -> str:
    """把「证据详情」渲染为 markdown 列表：实体 → 命中列（匹配方式 / 对应值）。"""
    evidence = evidence or {}
    lines: list[str] = []
    for match_type, entity_map in evidence.items():
        for entity, hits in (entity_map or {}).items():
            for hit in hits or []:
                col = str(hit.get("所在匹配列") or "").strip()
                if not col:
                    continue
                value = str(hit.get("对应匹配值") or "").strip()
                way = str(hit.get("匹配方式") or match_type or "").strip()
                detail = f"{way}" + (f", 值={value}" if value else "")
                lines.append(f"- 实体「{entity}」→ 列「{col}」（{detail}）")
    return "\n".join(lines)


def build_plan_markdown(
    schema_markdown: str,
    must_have: list[str] | None,
    evidence: dict | None = None,
    *,
    include_evidence: bool = False,
) -> str:
    """把富 schema 表 + 参考证据列合并为单一 markdown。"""
    must = [str(c).strip() for c in (must_have or []) if str(c).strip()]
    must = list(dict.fromkeys(must))

    parts = [schema_markdown.rstrip()]

    parts.append("\n### 参考证据列（不强制使用）")
    if must:
        parts.extend(f"- {col}" for col in must)
    else:
        parts.append("-（无）")

    if include_evidence:
        parts.append("\n### 实体对齐证据（不强制使用）")
        evidence_md = build_evidence_markdown(evidence)
        parts.append(evidence_md if evidence_md else "-（无）")

    return "\n".join(parts)
