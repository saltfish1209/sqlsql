from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from pipeline.profiler import DatabaseProfiler
from pipeline.utils import debug_print

try:
    from datasketch import MinHash, MinHashLSH
except Exception:  # pragma: no cover
    MinHash = None
    MinHashLSH = None


def _normalize_text(s: Any) -> str:
    return "".join(str(s or "").split()).lower()


@dataclass
class CandidateSchemaPack:
    candidates: list[dict]
    evidence: dict
    selected_columns: list[str]


class EvidenceLinker:
    """Schema linker that uses m_schema.json descriptions plus profiler metadata."""

    def __init__(self, schema_path_or_text: str, csv_path: str):
        self.csv_path = str(csv_path)
        self.schema_path = str(schema_path_or_text)
        self.schema_json = self._load_schema_json(self.schema_path)
        self.column_metadata = self._build_column_metadata(self.schema_json)
        self.column_names = [c["column_name"] for c in self.column_metadata]
        self.df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, na_values=[""])
        self.profiler = DatabaseProfiler(csv_path=self.csv_path)
        self._profiles = self.profiler.profile_all()
        self.profile_map = self.profiler.get_profile_map(self._profiles)
        self.profile_detail_map = self.profiler.get_profile_detail_map(self._profiles)
        self._value_index = self._build_value_index()
        self._semantic_value_index = self._build_semantic_value_index()
        self._index_dir = Path(settings.cache_dir) / "schema_linker"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self.exact_index: dict[str, set[str]] = {}
        self.lsh_index: dict[str, Any] = {}
        self.semantic_value_records: list[dict] = []
        self.semantic_value_embeddings: list[list[float]] = []
        self.semantic_value_model = None
        self._ensure_value_indexes()
        debug_print(f"[EvidenceLinker] loaded {len(self.column_metadata)} columns")

    @staticmethod
    def _load_schema_json(path_or_text: str) -> list[dict]:
        path = Path(path_or_text)
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
        else:
            raw = str(path_or_text)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"m_schema.json 不是合法 JSON: {path_or_text}") from exc
        if isinstance(data, dict):
            items = data.get("columns") or data.get("字段") or []
        else:
            items = data
        if not isinstance(items, list):
            raise ValueError("m_schema.json 格式不正确，必须是数组或包含 columns/字段 的对象。")
        return [item for item in items if isinstance(item, dict)]

    def _build_column_metadata(self, schema_json: list[dict]) -> list[dict]:
        metadata = []
        for item in schema_json:
            col_name = str(item.get("column_name") or item.get("字段名") or "").strip()
            if not col_name:
                continue
            metadata.append({
                "column_name": col_name,
                "column_description": str(item.get("column_description") or "").strip(),
            })
        return metadata

    def _build_value_index(self) -> dict[str, list[str]]:
        value_index: dict[str, list[str]] = {}
        top_k = max(int(getattr(settings, "candidate_value_top_k", 3)), 1)
        for col in self.df.columns:
            series = self.df[col].dropna().astype(str).str.strip()
            series = series[series != ""]
            value_index[str(col)] = [str(v) for v in series.value_counts().head(top_k).index.tolist()]
        return value_index

    def _build_semantic_value_index(self) -> dict[str, set[str]]:
        semantic_index: dict[str, set[str]] = {}
        for col, values in self._value_index.items():
            semantic_index[col] = {_normalize_text(v) for v in values if v}
        return semantic_index

    def _index_path(self, name: str) -> Path:
        return self._index_dir / name

    def _ensure_value_indexes(self) -> None:
        exact_path = self._index_path("exact_index.pkl")
        lsh_path = self._index_path("lsh_index.pkl")
        semantic_path = self._index_path("semantic_value_index.pkl")
        if self._load_indexes(exact_path, lsh_path, semantic_path):
            debug_print("[EvidenceLinker] 本地索引加载成功")
            return
        self._build_indexes()
        self._save_indexes(exact_path, lsh_path, semantic_path)

    def _build_indexes(self) -> None:
        if MinHash is None or MinHashLSH is None:
            raise RuntimeError("缺少 datasketch 依赖，无法构建 LSH 索引")
        self.exact_index = {}
        self.lsh_index = {}
        for col in self.df.columns:
            values = self.df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            lsh = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.lsh_num_perm)
            has_lsh = False
            for val in values.unique():
                self.exact_index.setdefault(val, set()).add(col)
                norm = _normalize_text(val)
                if len(norm) < 2:
                    continue
                mh = MinHash(num_perm=settings.lsh_num_perm)
                for ch in norm:
                    mh.update(ch.encode("utf8"))
                lsh.insert(val, mh)
                has_lsh = True
            if has_lsh:
                self.lsh_index[col] = lsh
        self.semantic_value_records = []
        self.semantic_value_embeddings = []
        if settings.enable_semantic_value_retrieval:
            for col, values in self._value_index.items():
                for val in values:
                    self.semantic_value_records.append({"value": val, "column": col})
        debug_print(f"[EvidenceLinker] 索引构建完成，exact={len(self.exact_index)}")

    def _save_indexes(self, exact_path: Path, lsh_path: Path, semantic_path: Path) -> None:
        import pickle
        with open(exact_path, "wb") as f:
            pickle.dump(self.exact_index, f)
        with open(lsh_path, "wb") as f:
            pickle.dump(self.lsh_index, f)
        with open(semantic_path, "wb") as f:
            pickle.dump({"records": self.semantic_value_records}, f)

    def _load_indexes(self, exact_path: Path, lsh_path: Path, semantic_path: Path) -> bool:
        import pickle
        if not exact_path.exists() or not lsh_path.exists() or not semantic_path.exists():
            return False
        try:
            with open(exact_path, "rb") as f:
                self.exact_index = pickle.load(f)
            with open(lsh_path, "rb") as f:
                self.lsh_index = pickle.load(f)
            with open(semantic_path, "rb") as f:
                payload = pickle.load(f)
            self.semantic_value_records = payload.get("records", [])
            self.semantic_value_embeddings = []
            return True
        except Exception:
            return False

    def retrieve(
        self,
        question: str,
        extracted_entities: list[dict] | list[str] | None = None,
    ) -> CandidateSchemaPack:
        q = _normalize_text(question)
        entities = self._extract_entity_texts(extracted_entities)
        exact_hits: list[dict] = []
        fuzzy_hits: list[dict] = []
        semantic_hits: list[dict] = []
        candidates: list[dict] = []

        for meta in self.column_metadata:
            col = meta["column_name"]
            desc = meta.get("column_description", "")
            profile_inline = self.profile_map.get(col, "")
            profile_detail = self.profile_detail_map.get(col, {})
            col_examples = list(self._value_index.get(col, [])[: settings.candidate_value_top_k])
            score = 0.0
            evidence_types: list[str] = []

            desc_norm = _normalize_text(desc)
            col_norm = _normalize_text(col)
            if desc_norm and any(tok and tok in q for tok in desc_norm.split()):
                score += settings.candidate_semantic_bonus
                evidence_types.append("字段描述命中")
                semantic_hits.append({"文本": desc, "字段名": col, "来源": "字段描述"})

            for ex in col_examples:
                ex_norm = _normalize_text(ex)
                if ex_norm and ex_norm in q:
                    if _is_numeric_like(ex_norm) and _is_numeric_like(q) and len(q) > 4:
                        continue
                    score += settings.candidate_exact_bonus
                    evidence_types.append("字段样例命中")
                    exact_hits.append({"文本": ex, "字段名": col, "来源": "字段样例"})

            if profile_inline:
                profile_norm = _normalize_text(profile_inline)
                if profile_norm and any(tok and tok in profile_norm for tok in q.split()):
                    score += settings.candidate_semantic_bonus
                    evidence_types.append("字段统计命中")

            if meta.get("is_primary_key"):
                score += 0.1
                evidence_types.append("主键标记")

            if col_norm and len(q) > 4 and q.isdigit() and _is_numeric_like(col_norm):
                score -= 0.3
                evidence_types.append("数值串降权")

            semantic_value_hits = self._semantic_value_index.get(col, set())
            if semantic_value_hits and any(tok in semantic_value_hits for tok in self._question_tokens(question)):
                score += settings.candidate_exact_bonus
                evidence_types.append("语义值命中")

            for ent in entities:
                ent_norm = _normalize_text(ent)
                if not ent_norm:
                    continue
                if ent_norm in col_norm:
                    score += settings.candidate_exact_bonus
                    evidence_types.append("实体字段命中")
                    exact_hits.append({"文本": ent, "字段名": col, "来源": "实体字段"})
                elif desc_norm and ent_norm in desc_norm:
                    score += settings.candidate_semantic_bonus
                    evidence_types.append("实体描述命中")
                    semantic_hits.append({"文本": ent, "字段名": col, "来源": "实体描述"})
                elif semantic_value_hits and ent_norm in semantic_value_hits:
                    score += settings.candidate_exact_bonus
                    evidence_types.append("实体语义值命中")
                    exact_hits.append({"文本": ent, "字段名": col, "来源": "实体语义值"})
                else:
                    for ex in col_examples:
                        ex_norm = _normalize_text(ex)
                        if ex_norm and (ent_norm in ex_norm or ex_norm in ent_norm):
                            if _is_numeric_like(ent_norm) and _is_numeric_like(ex_norm) and len(ent_norm) > 4:
                                continue
                            score += settings.candidate_fuzzy_bonus
                            evidence_types.append("实体样例命中")
                            fuzzy_hits.append({"文本": ent, "字段名": col, "字段样例": ex})
                            break

            if score >= settings.candidate_min_score:
                candidates.append({
                    "字段名": col,
                    "字段统计": profile_inline,
                    "字段类型": profile_detail.get("字段类型", ""),
                    "是否枚举": profile_detail.get("是否枚举", "否"),
                    "主键标记": "Primary Key" if col == "物资唯一码" else "",
                    "枚举值": profile_detail.get("枚举值", []),
                    "分数": round(float(score), 4),
                    "证据类型": sorted(set(evidence_types)) or ["字段描述命中"],
                })

        candidates.sort(key=lambda x: x["分数"], reverse=True)
        candidates = candidates[: settings.candidate_max_columns]
        selected_columns = [c["字段名"] for c in candidates[: settings.evidence_schema_top_k]]
        evidence = {
            "精确匹配": exact_hits,
            "模糊匹配": fuzzy_hits,
            "语义匹配": semantic_hits,
        }
        return CandidateSchemaPack(candidates=candidates, evidence=evidence, selected_columns=selected_columns)

    @staticmethod
    def _extract_entity_texts(extracted_entities: list[dict] | list[str] | None) -> list[str]:
        entities: list[str] = []
        if not extracted_entities:
            return entities
        for item in extracted_entities:
            if isinstance(item, dict):
                text = item.get("文本") or item.get("text") or item.get("实体") or item.get("entity") or item.get("值") or item.get("value") or ""
            else:
                text = str(item)
            if text:
                entities.append(str(text))
        return entities

    @staticmethod
    def _question_tokens(question: str) -> list[str]:
        q = _normalize_text(question)
        return [tok for tok in re.split(r"[\s,，。！？?;；:：/\\|]+", q) if tok]


def _is_numeric_like(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return False
    return all(ch.isdigit() or ch == "." for ch in text)
