from __future__ import annotations

import difflib
import json
import os
import pickle
import re
import time
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

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover
    CrossEncoder = None
    SentenceTransformer = None


@dataclass
class CandidateSchemaPack:
    全量排序: list[dict]
    Top20候选: list[dict]
    必须列集合: list[str]
    证据详情: dict
    精简schema: list[dict]
    给LLM的中文结构: dict


def _normalize_lsh_text(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s)).upper()


def _normalize_text(s: Any) -> str:
    return "".join(str(s or "").split()).lower()


def _char_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def _safe_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _parse_m_schema(schema_text: str) -> list[dict]:
    schema_text = (schema_text or "").strip()
    if not schema_text:
        return []

    try:
        data = json.loads(schema_text)
        if isinstance(data, dict):
            items = data.get("columns") or data.get("字段") or data.get("schema") or []
        else:
            items = data
        if isinstance(items, list):
            metadata = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                col_name = str(item.get("column_name") or item.get("字段名") or item.get("列名") or "").strip()
                if not col_name:
                    continue
                metadata.append({
                    "column_name": col_name,
                    "data_type": str(item.get("data_type") or item.get("字段类型") or item.get("column_type") or "TEXT").strip(),
                    "column_description": str(item.get("column_description") or item.get("字段描述") or item.get("描述") or "").strip(),
                    "examples": [str(v).strip() for v in (item.get("examples") or item.get("示例值") or item.get("枚举值") or []) if str(v).strip()],
                    "raw_text": json.dumps(item, ensure_ascii=False),
                })
            debug_print(f"[Schema] 解析 JSON schema，共识别出 {len(metadata)} 个列定义。")
            return metadata
    except Exception:
        pass

    metadata = []
    pattern = re.compile(r'\s*\(([^:]+):\s*([^,]+),\s*(.*?),\s*Examples:\s*\[(.*?)\]\)')
    for line in schema_text.split("\n"):
        line = line.strip()
        if not line.startswith("("):
            continue
        m = pattern.search(line)
        if m:
            metadata.append({
                "column_name": m.group(1).strip(),
                "data_type": m.group(2).strip(),
                "column_description": m.group(3).strip(),
                "examples": [e.strip() for e in m.group(4).split(",") if e.strip()],
                "raw_text": line,
            })
    debug_print(f"[Schema] 解析文本 schema，共识别出 {len(metadata)} 个列定义。")
    return metadata


class SchemaLinker:
    def __init__(self, schema_path_or_text: str, csv_path: str, cross_encoder_path: str | None = None):
        self._cross_encoder_path = cross_encoder_path or settings.cross_encoder_model
        self.csv_path = str(csv_path)
        schema_path_or_text = str(schema_path_or_text)
        self.schema_text = Path(schema_path_or_text).read_text(encoding="utf-8") if os.path.isfile(schema_path_or_text) else schema_path_or_text
        self.column_metadata = _parse_m_schema(self.schema_text)
        self.column_names = [c["column_name"] for c in self.column_metadata]
        self.df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, na_values=[""])
        self.profiler = DatabaseProfiler(csv_path=self.csv_path)
        self._profiles = self.profiler.profile_all()
        self.profile_map = self.profiler.get_profile_map(self._profiles)
        self.profile_detail_map = self.profiler.get_profile_detail_map(self._profiles)
        self._value_index = self._build_value_index()
        self._semantic_value_index = self._build_semantic_value_index()
        self.cache_dir = Path(settings.cache_dir) / "schema_linker"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exact_index: dict[str, set[str]] = {}
        self.lsh_index: dict[str, Any] = {}
        self.semantic_value_records: list[dict] = []
        self.semantic_value_embeddings: list[list[float]] = []
        self.semantic_value_model = None
        self.rank_model = None
        self.cache_status = {
            "exact_index": False,
            "lsh_index": False,
            "semantic_value_index": False,
        }
        self._ensure_indexes()
        self._ensure_rank_model()

    def _ensure_rank_model(self) -> None:
        if CrossEncoder is None:
            raise RuntimeError("缺少 sentence-transformers 依赖，无法加载 CrossEncoder")
        debug_print(f"[Schema][A-route] 加载 CrossEncoder：{self._cross_encoder_path}")
        self.rank_model = CrossEncoder(self._cross_encoder_path, trust_remote_code=True)
        debug_print("[Schema][A-route] CrossEncoder 加载完成。")

    def _build_value_index(self) -> dict[str, list[str]]:
        value_index: dict[str, list[str]] = {}
        top_k = max(int(getattr(settings, "candidate_value_top_k", 3)), 1)
        for col in self.df.columns:
            series = self.df[col].dropna().astype(str).str.strip()
            series = series[series != ""]
            value_index[str(col)] = [str(v) for v in series.value_counts().head(top_k).index.tolist()]
        return value_index

    def _build_semantic_value_index(self) -> dict[str, set[str]]:
        return {col: {_normalize_text(v) for v in values if v} for col, values in self._value_index.items()}

    def _index_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _ensure_indexes(self) -> None:
        exact_path = self._index_path("exact_index.pkl")
        lsh_path = self._index_path("lsh_index.pkl")
        semantic_path = self._index_path("semantic_value_index.pkl")
        debug_print(f"[Schema][Cache] 检查缓存目录：{self.cache_dir}")
        debug_print(f"[Schema][Cache] exact={exact_path.exists()}, lsh={lsh_path.exists()}, semantic={semantic_path.exists()}")
        if self._load_indexes(exact_path, lsh_path, semantic_path):
            self.cache_status.update({"exact_index": True, "lsh_index": True, "semantic_value_index": True})
            debug_print("[Schema][Cache] 已命中全部索引缓存，直接加载。")
            return
        debug_print("[Schema][Cache] 未命中完整缓存，开始重建索引。")
        self._build_indexes()
        self._save_indexes(exact_path, lsh_path, semantic_path)
        debug_print("[Schema][Cache] 索引重建并已保存到缓存。")

    def _build_indexes(self) -> None:
        if MinHash is None or MinHashLSH is None:
            raise RuntimeError("缺少 datasketch 依赖，无法构建 LSH 索引")
        self.exact_index = {}
        self.lsh_index = {}
        debug_print("[Schema][Cache] 开始构建 exact/lsh 索引。")
        for col in self.df.columns:
            values = self.df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            lsh = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.lsh_num_perm)
            has_lsh = False
            for val in values.unique():
                self.exact_index.setdefault(val, set()).add(col)
                norm = _normalize_lsh_text(val)
                if len(norm) < 2:
                    continue
                mh = MinHash(num_perm=settings.lsh_num_perm)
                for ch in norm:
                    mh.update(ch.encode("utf8"))
                lsh.insert(val, mh)
                has_lsh = True
            if has_lsh:
                self.lsh_index[col] = lsh
        debug_print(f"[Schema][Cache] exact_index 唯一值数量：{len(self.exact_index)}，lsh列数：{len(self.lsh_index)}")
        if settings.enable_semantic_value_retrieval:
            debug_print("[Schema][Cache] 开始构建语义值候选缓存。")
            for col, values in self._value_index.items():
                for val in values:
                    self.semantic_value_records.append({"value": val, "column": col})
            self.cache_status["semantic_value_index"] = True
            debug_print(f"[Schema][Cache] 语义值记录数：{len(self.semantic_value_records)}")

    def _save_indexes(self, exact_path: Path, lsh_path: Path, semantic_path: Path) -> None:
        with open(exact_path, "wb") as f:
            pickle.dump(self.exact_index, f)
        with open(lsh_path, "wb") as f:
            pickle.dump(self.lsh_index, f)
        with open(semantic_path, "wb") as f:
            pickle.dump({"records": self.semantic_value_records}, f)

    def _load_indexes(self, exact_path: Path, lsh_path: Path, semantic_path: Path) -> bool:
        if not exact_path.exists() or not lsh_path.exists() or not semantic_path.exists():
            debug_print("[Schema][Cache] 缓存文件不完整。")
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
            self.cache_status.update({"exact_index": True, "lsh_index": True})
            self.cache_status["semantic_value_index"] = bool(self.semantic_value_records)
            debug_print(f"[Schema][Cache] exact_index={len(self.exact_index)}, lsh_index={len(self.lsh_index)}, semantic_records={len(self.semantic_value_records)}")
            return True
        except Exception as exc:
            debug_print(f"[Schema][Cache] 缓存加载失败：{type(exc).__name__}: {exc}")
            return False

    @staticmethod
    def _lsh_secondary_verify(norm_kw: str, norm_val: str) -> bool:
        if not norm_kw or not norm_val:
            return False
        if norm_kw == norm_val:
            return True
        seq = _safe_ratio(norm_kw, norm_val)
        jac = _char_jaccard(norm_kw, norm_val)
        combined = 0.5 * seq + 0.5 * jac
        return combined >= settings.lsh_query_combined_threshold

    @staticmethod
    def _as_float_vector(vec) -> list[float]:
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        va = SchemaLinker._as_float_vector(a)
        vb = SchemaLinker._as_float_vector(b)
        if not va or not vb or len(va) != len(vb):
            return 0.0
        dot = sum(x * y for x, y in zip(va, vb))
        norm_a = sum(x * x for x in va) ** 0.5
        norm_b = sum(y * y for y in vb) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_semantic_value_model(self):
        if self.semantic_value_model is None:
            if SentenceTransformer is None:
                raise RuntimeError("缺少 sentence-transformers 依赖，无法进行语义值检索")
            debug_print(f"[Schema][D-route] 加载语义值模型：{settings.embed_model}")
            self.semantic_value_model = SentenceTransformer(settings.embed_model, trust_remote_code=True)
            debug_print("[Schema][D-route] 语义值模型加载完成。")
        return self.semantic_value_model

    def _search_semantic_values(self, keyword: str, query_embedding=None, top_k: int | None = None, threshold: float | None = None) -> list[dict]:
        if not self.semantic_value_records:
            return []
        k = top_k or settings.semantic_value_top_k
        min_score = settings.semantic_value_threshold if threshold is None else threshold
        if query_embedding is None:
            model = self._get_semantic_value_model()
            query_embedding = model.encode(keyword, convert_to_numpy=False, normalize_embeddings=True)
        if not self.semantic_value_embeddings:
            self.semantic_value_embeddings = [
                self._as_float_vector(self._get_semantic_value_model().encode(r["value"], convert_to_numpy=False, normalize_embeddings=True))
                for r in self.semantic_value_records
            ]
        scored: list[tuple[float, dict]] = []
        for record, emb in zip(self.semantic_value_records, self.semantic_value_embeddings):
            score = self._cosine_similarity(query_embedding, emb)
            if score >= min_score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"对应匹配值": r["value"], "所在匹配列": r["column"], "匹配方式": "向量匹配", "相关性分数": round(float(s), 4)} for s, r in scored[:k]]

    def retrieve(self, question: str, extracted_entities: list[str] | list[dict] | None = None) -> CandidateSchemaPack:
        q = _normalize_text(question)
        entities = self._extract_entity_texts(extracted_entities)
        exact_hits: list[dict] = []
        fuzzy_hits: list[dict] = []
        semantic_hits: list[dict] = []

        ranked: list[dict] = []
        pairs = [[question, c.get("column_description", "")] for c in self.column_metadata]
        debug_print(f"[Schema][A-route] CrossEncoder 输入对数: {len(pairs)}")
        scores = self.rank_model.predict(pairs)
        for idx, (meta, score) in enumerate(sorted(zip(self.column_metadata, scores), key=lambda x: x[1], reverse=True), start=1):
            col = meta["column_name"]
            item = {
                "列名": col,
                "相关性分数": round(float(score), 4),
                "列描述": meta.get("column_description", ""),
                "字段类型": self.profile_detail_map.get(col, {}).get("字段类型", meta.get("data_type", "")),
                "是否枚举": "是" if self.profile_detail_map.get(col, {}).get("枚举值") else "否",
                "空值率": self._extract_profile_piece(col, "空值率"),
                "唯一值数": self._extract_profile_piece(col, "唯一值数"),
                "示例值": self._extract_profile_piece(col, "示例值"),
                "格式": self._extract_profile_piece(col, "格式"),
                "范围": self._extract_profile_piece(col, "范围"),
            }
            ranked.append(item)

        top20 = ranked[:20]
        must_have = []
        evidence = {
            "精确匹配": {},
            "模糊匹配": {},
            "向量匹配": {},
        }
        for ent in entities:
            ent_norm = _normalize_text(ent)
            if not ent_norm:
                continue
            matched_cols: list[str] = []
            matched_values: list[dict] = []
            for meta in self.column_metadata:
                col = meta["column_name"]
                examples = list(self._value_index.get(col, [])[: settings.candidate_value_top_k])
                if ent_norm in _normalize_text(col):
                    matched_cols.append(col)
                    matched_values.append({"实体文本": ent, "对应匹配值": col, "所在匹配列": col, "匹配方式": "字段语义命中"})
                if ent in self.exact_index:
                    cols = list(self.exact_index[ent])
                    for c in cols:
                        matched_cols.append(c)
                        matched_values.append({"实体文本": ent, "对应匹配值": ent, "所在匹配列": c, "匹配方式": "精确匹配"})
                for ex in examples:
                    ex_norm = _normalize_text(ex)
                    if not ex_norm:
                        continue
                    # 收缩模糊范围：只允许长度接近且字符重叠足够时进入模糊匹配
                    seq = _safe_ratio(ent_norm, ex_norm)
                    jac = _char_jaccard(ent_norm, ex_norm)
                    combined = 0.5 * seq + 0.5 * jac
                    if combined >= settings.lsh_query_combined_threshold:
                        matched_cols.append(col)
                        matched_values.append({"实体文本": ent, "对应匹配值": ex, "所在匹配列": col, "匹配方式": "模糊匹配", "相关性分数": round(float(combined), 4)})
                        break
            if settings.enable_semantic_value_retrieval:
                sem_hits = self._search_semantic_values(ent)
                for hit in sem_hits:
                    matched_cols.append(hit["所在匹配列"])
                    matched_values.append({"实体文本": ent, **hit})
            deduped = self._dedupe_alignments(matched_values)
            if deduped:
                evidence["精确匹配"][ent] = [x for x in deduped if x["匹配方式"] == "精确匹配"]
                evidence["模糊匹配"][ent] = [x for x in deduped if x["匹配方式"] == "模糊匹配"]
                evidence["向量匹配"][ent] = [x for x in deduped if x["匹配方式"] == "向量匹配"]
                must_have.extend([x["所在匹配列"] for x in deduped])
        must_have = list(dict.fromkeys(must_have))
        compact_schema = self._build_compact_schema(top20, must_have, evidence, entities)
        llm_struct = {
            "召回schema": top20,
            "证据实体": entities,
            "实体对齐结果": self._format_entity_alignment_for_llm(evidence),
            "精简schema": compact_schema,
        }
        return CandidateSchemaPack(
            全量排序=ranked,
            Top20候选=top20,
            必须列集合=must_have,
            证据详情=evidence,
            精简schema=compact_schema,
            给LLM的中文结构=llm_struct,
        )

    def _extract_profile_piece(self, col: str, key: str):
        detail = self.profile_detail_map.get(col, {})
        return detail.get(key, "")

    @staticmethod
    def _extract_entity_texts(extracted_entities: list[str] | list[dict] | None) -> list[str]:
        out: list[str] = []
        if not extracted_entities:
            return out
        for item in extracted_entities:
            if isinstance(item, dict):
                text = item.get("文本") or item.get("text") or item.get("实体") or item.get("entity") or ""
            else:
                text = str(item)
            text = str(text).strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _dedupe_alignments(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for item in items:
            key = (item.get("所在匹配列"), item.get("对应匹配值"))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out[:3]

    def _build_compact_schema(self, top20: list[dict], must_have: list[str], evidence: dict, entities: list[str]) -> list[dict]:
        selected: list[dict] = []
        selected_cols = set()
        evidence_cols = set(must_have)
        for item in top20:
            if item["相关性分数"] <= 0:
                continue
            if item["列名"] in evidence_cols or len(selected) < 8:
                selected.append({
                    "列名": item["列名"],
                    "相关性分数": item["相关性分数"],
                    "列描述": item["列描述"],
                    "字段类型": item["字段类型"],
                    "是否枚举": item["是否枚举"],
                    "空值率": item["空值率"],
                    "唯一值数": item["唯一值数"],
                    "示例值": item["示例值"],
                    "格式": item["格式"],
                    "范围": item["范围"],
                })
                selected_cols.add(item["列名"])
        for col in evidence_cols:
            if col in selected_cols:
                continue
            meta = next((m for m in self.column_metadata if m["column_name"] == col), None)
            if not meta:
                continue
            score = next((x["相关性分数"] for x in top20 if x["列名"] == col), 0.0)
            if score <= 0:
                continue
            selected.append({
                "列名": col,
                "相关性分数": score,
                "列描述": meta.get("column_description", ""),
                "字段类型": self.profile_detail_map.get(col, {}).get("字段类型", meta.get("data_type", "")),
                "是否枚举": "是" if self.profile_detail_map.get(col, {}).get("枚举值") else "否",
                "空值率": self._extract_profile_piece(col, "空值率"),
                "唯一值数": self._extract_profile_piece(col, "唯一值数"),
                "示例值": self._extract_profile_piece(col, "示例值"),
                "格式": self._extract_profile_piece(col, "格式"),
                "范围": self._extract_profile_piece(col, "范围"),
            })
        return selected[:12]

    def _format_entity_alignment_for_llm(self, evidence: dict) -> list[dict]:
        out = []
        for ent, items in evidence.get("精确匹配", {}).items():
            merged = items + evidence.get("模糊匹配", {}).get(ent, []) + evidence.get("向量匹配", {}).get(ent, [])
            merged = self._dedupe_alignments(merged)
            out.append({"实体文本": ent, "候选对齐": merged[:3]})
        return out

    def build_llm_prompt_payload(self, pack: CandidateSchemaPack) -> dict:
        return {
            "召回schema": pack.Top20候选,
            "证据实体": pack.给LLM的中文结构.get("证据实体", []),
            "实体对齐结果": pack.给LLM的中文结构.get("实体对齐结果", []),
            "精简schema": pack.精简schema,
        }
