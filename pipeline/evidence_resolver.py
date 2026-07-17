from __future__ import annotations

import re
from typing import Any

from pipeline.utils import to_halfwidth


_IDENTIFIER = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff]*)'
_TEXT_CONDITION_RE = re.compile(
    rf"(?P<column>(?:{_IDENTIFIER}\s*\.\s*)?{_IDENTIFIER})"
    r"\s*(?P<operator>=|LIKE)\s*(?:"
    r"'(?P<single_literal>(?:''|[^'])*)'|"
    r'"(?P<double_literal>(?:""|[^"])*)"'
    r")",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    return "".join(to_halfwidth(str(value or "")).split()).lower()


def _unquote_identifier(identifier: str) -> str:
    parts = re.findall(_IDENTIFIER, identifier)
    value = (parts[-1] if parts else identifier).strip()
    if len(value) >= 2 and (
        (value[0] == value[-1] and value[0] in {'"', '`'})
        or (value[0] == '[' and value[-1] == ']')
    ):
        return value[1:-1]
    return value


def extract_text_conditions(sql: str) -> list[dict]:
    """提取候选 SQL 中所有文本等值或 LIKE 条件。"""
    conditions: list[dict] = []
    for match in _TEXT_CONDITION_RE.finditer(str(sql or "")):
        literal = match.group("single_literal")
        if literal is None:
            literal = (match.group("double_literal") or "").replace('""', '"')
        else:
            literal = literal.replace("''", "'")
        conditions.append(
            {
                "column": _unquote_identifier(match.group("column")),
                "operator": match.group("operator").upper(),
                "literal": literal.strip(),
            }
        )
    return conditions


class EvidenceResolver:
    """统一管理问题级实体证据与候选 SQL 条件级证据。"""

    _ROUTES = ("精确匹配", "模糊匹配", "向量匹配")

    def __init__(self, linker, db):
        self.linker = linker
        self.db = db
        self._value_cache: dict[str, dict[str, list[dict]]] = {}
        self._probe_cache: dict[tuple[str, str], dict] = {}

    def prepare(
        self,
        question: str,
        entities: list[str] | list[dict] | None,
        evidence: dict | None,
    ) -> dict:
        """生成前建立问题级证据包，并把已有三路结果放入复用缓存。"""
        entity_texts = self._entity_texts(entities)
        routes = evidence or {route: {} for route in self._ROUTES}
        pregen_values: list[str] = []

        for route in self._ROUTES:
            entity_texts.extend(str(entity) for entity in (routes.get(route) or {}).keys())
        entity_texts = list(dict.fromkeys(entity_texts))

        for entity in entity_texts:
            entity_routes = {
                route: list((routes.get(route) or {}).get(entity) or [])
                for route in self._ROUTES
            }
            self._value_cache[_normalize_text(entity)] = entity_routes
            for hits in entity_routes.values():
                pregen_values.extend(
                    str(hit.get("对应匹配值") or "").strip()
                    for hit in hits
                    if str(hit.get("对应匹配值") or "").strip()
                )
            for route, hits in entity_routes.items():
                for hit in hits:
                    value_norm = _normalize_text(hit.get("对应匹配值"))
                    if not value_norm:
                        continue
                    cached = self._value_cache.setdefault(
                        value_norm,
                        {name: [] for name in self._ROUTES},
                    )
                    if hit not in cached[route]:
                        cached[route].append(hit)

        return {
            "question": str(question or ""),
            "entities": entity_texts,
            "routes": routes,
            "pregen_values": list(dict.fromkeys(pregen_values)),
        }

    def audit_candidate(self, candidate: dict, bundle: dict | None = None) -> dict:
        """生成后审计一条 SQL；只对生成前未查过的字面量增量检索。"""
        bundle = bundle or self.prepare("", [], {})
        condition_evidence = [
            self._resolve_condition(condition, bundle)
            for condition in extract_text_conditions(candidate.get("sql") or "")
        ]
        candidate["condition_evidence"] = {
            "conditions": condition_evidence,
            "lookup_mode": "reuse_then_incremental",
        }
        return candidate

    def audit_candidates(self, candidates: list[dict], bundle: dict | None = None) -> list[dict]:
        for candidate in candidates:
            self.audit_candidate(candidate, bundle)
        return candidates

    def _resolve_condition(self, condition: dict, bundle: dict) -> dict:
        literal = str(condition.get("literal") or "")
        lookup_literal = literal.strip("%_") if condition.get("operator") == "LIKE" else literal
        literal_norm = _normalize_text(lookup_literal)
        question_norm = _normalize_text(bundle.get("question"))
        pregen_norms = {_normalize_text(value) for value in bundle.get("pregen_values") or []}

        if literal_norm and literal_norm in question_norm:
            provenance = "question_exact"
        elif literal_norm and literal_norm in pregen_norms:
            provenance = "pregen_candidate"
        else:
            provenance = "generated_only"

        routes = self._value_cache.get(literal_norm)
        if routes is None:
            lookup = getattr(self.linker, "lookup_value_evidence", None)
            routes = lookup(lookup_literal) if lookup and lookup_literal else {}
            routes = {route: list(routes.get(route) or []) for route in self._ROUTES}
            self._value_cache[literal_norm] = routes

        column = str(condition.get("column") or "")
        probe_key = (_normalize_text(column), literal_norm)
        probe = self._probe_cache.get(probe_key)
        if probe is None:
            probe_fn = getattr(self.db, "probe_literal_in_column", None)
            probe = probe_fn(column, lookup_literal) if probe_fn and column and lookup_literal else {}
            probe = dict(probe or {})
            self._probe_cache[probe_key] = probe

        column_norm = _normalize_text(column)
        column_hits = {
            route: [
                hit
                for hit in hits
                if _normalize_text(hit.get("所在匹配列")) == column_norm
            ]
            for route, hits in routes.items()
        }
        return {
            **condition,
            "provenance": provenance,
            "routes": routes,
            "column_hits": column_hits,
            "exact_exists": bool(probe.get("exact_exists")),
            "like_exists": bool(probe.get("like_exists")),
            "candidate_values": list(probe.get("candidate_values") or []),
        }

    @staticmethod
    def _entity_texts(entities: list[str] | list[dict] | None) -> list[str]:
        texts: list[str] = []
        for item in entities or []:
            if isinstance(item, dict):
                text = item.get("文本") or item.get("text") or item.get("实体") or item.get("entity") or ""
            else:
                text = item
            text = str(text or "").strip()
            if text:
                texts.append(text)
        return list(dict.fromkeys(texts))
