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

import numpy as np

import pandas as pd

from config.settings import settings
from pipeline.cross_encoder_passage import build_column_passage
from pipeline.profiler import DatabaseProfiler
from pipeline.utils import debug_print

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

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


def _find_exact_column_name_mentions(question: str, column_names: list[str]) -> list[str]:
    """Find column names that appear as exact normalized substrings in question."""
    q_norm = _normalize_text(question)
    if not q_norm:
        return []
    hits: list[str] = []
    for col in column_names:
        col_norm = _normalize_text(col)
        if not col_norm:
            continue
        if col_norm in q_norm:
            hits.append(col)
    return list(dict.fromkeys(hits))


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


def _parse_ratio(value: Any) -> float:
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
        parsed_metadata = _parse_m_schema(self.schema_text)
        self.df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, na_values=[""])
        csv_columns = {str(col).strip() for col in self.df.columns if str(col).strip()}
        raw_schema_count = len(parsed_metadata)
        self.profiler = DatabaseProfiler(csv_path=self.csv_path)
        self._profiles = self.profiler.profile_all()
        self.profile_map = self.profiler.get_profile_map(self._profiles)
        self.profile_detail_map = self.profiler.get_profile_detail_map(self._profiles)
        schema_not_in_raw: list[str] = []
        high_null_ratio: list[str] = []
        threshold = float(getattr(settings, "deprecated_column_null_ratio_threshold", 0.95))
        for meta in parsed_metadata:
            col = str(meta.get("column_name", "")).strip()
            if not col:
                continue
            if col not in csv_columns:
                schema_not_in_raw.append(col)
                continue
            detail = self.profile_detail_map.get(col, {})
            if _parse_ratio(detail.get("空值率")) >= threshold:
                high_null_ratio.append(col)
        deprecated_columns = set(schema_not_in_raw) | set(high_null_ratio)
        self.column_metadata = [
            m for m in parsed_metadata
            if str(m.get("column_name", "")).strip() in csv_columns
            and str(m.get("column_name", "")).strip() not in deprecated_columns
        ]
        self.column_names = [c["column_name"] for c in self.column_metadata]
        self.column_passage_map = {
            meta["column_name"]: build_column_passage(meta, self.profile_detail_map.get(meta["column_name"], {}))
            for meta in self.column_metadata
        }
        self.deprecated_columns_report = {
            "null_ratio_threshold": threshold,
            "schema_columns_count": raw_schema_count,
            "raw_columns_count": len(csv_columns),
            "active_columns_count": len(self.column_metadata),
            "deprecated_columns_count": len(deprecated_columns),
            "deprecated": {
                "schema_not_in_raw": sorted(set(schema_not_in_raw)),
                "high_null_ratio": sorted(set(high_null_ratio)),
            },
        }
        if deprecated_columns:
            debug_print(
                "[Schema][Init] 废弃列过滤已启用: "
                f"有效列={len(self.column_metadata)}, 废弃列={len(deprecated_columns)} "
                f"(schema缺失={len(set(schema_not_in_raw))}, 高空值={len(set(high_null_ratio))})"
            )
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
        self.faiss_index = None
        self.faiss_index_meta: list[dict] = []
        self.cache_status = {
            "exact_index": False,
            "lsh_index": False,
            "semantic_value_index": False,
        }
        self._ensure_indexes()
        self._ensure_rank_model()

    def _rank_candidates(self, items: list[dict], top_k: int | None = None) -> list[dict]:
        if not items:
            return []
        sorted_items = sorted(items, key=lambda x: (-(float(x.get("相关性分数") or 0.0)), x.get("列名", "")))
        best_score = float(sorted_items[0].get("相关性分数") or 0.0)
        if best_score <= 0:
            return sorted_items[: (top_k or settings.candidate_top_k)]

        min_score = best_score * float(getattr(settings, "candidate_cliff_min_ratio", 0.15))
        protect_score = best_score * float(getattr(settings, "candidate_cliff_protect_ratio", 0.4))
        decay_threshold = float(getattr(settings, "candidate_cliff_decay_threshold", 0.5))
        target_k = top_k or settings.candidate_top_k

        filtered = [item for item in sorted_items if float(item.get("相关性分数") or 0.0) >= min_score]
        if not filtered:
            return sorted_items[:target_k]

        cutoff = len(filtered)
        for idx in range(len(filtered) - 1):
            cur = float(filtered[idx].get("相关性分数") or 0.0)
            nxt = float(filtered[idx + 1].get("相关性分数") or 0.0)
            if cur <= 0:
                continue
            decay_rate = (cur - nxt) / cur
            if cur >= protect_score or nxt >= protect_score:
                continue
            if decay_rate >= decay_threshold:
                cutoff = idx + 1
                break

        cliff_selected = filtered[:cutoff]
        if len(cliff_selected) >= target_k:
            return cliff_selected[:target_k]
        if len(filtered) >= target_k:
            return filtered[:target_k]
        return cliff_selected

    def _ensure_rank_model(self) -> None:
        if CrossEncoder is None:
            raise RuntimeError("缺少 sentence-transformers 依赖，无法加载 CrossEncoder")
        debug_print(f"[Schema][A-route] 加载 CrossEncoder：{self._cross_encoder_path}")
        self.rank_model = CrossEncoder(self._cross_encoder_path, trust_remote_code=True)
        debug_print("[Schema][A-route] CrossEncoder 加载完成。")

    @staticmethod
    def _is_vector_eligible_column(meta: dict, profile_detail: dict | None = None) -> bool:
        dtype = str((profile_detail or {}).get("字段类型") or meta.get("data_type") or "").upper()
        if dtype != "TEXT":
            return False
        col_name = str(meta.get("column_name") or "")
        if not col_name:
            return False
        if re.fullmatch(r"\d+", col_name):
            return False
        if re.fullmatch(r"[A-Za-z]+\d+", col_name):
            return False
        return True

    def _is_text_query(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text))

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

    def _build_faiss_index(self) -> None:
        if faiss is None:
            raise RuntimeError("缺少 faiss 依赖，无法构建向量索引")
        model = self._get_semantic_value_model()
        vectors: list[np.ndarray] = []
        meta: list[dict] = []
        for col, values in self._value_index.items():
            profile_detail = self.profile_detail_map.get(col, {})
            meta_def = next((m for m in self.column_metadata if m.get("column_name") == col), {})
            if not self._is_vector_eligible_column(meta_def, profile_detail):
                continue
            for val in values:
                text = str(val or "").strip()
                if not text or not self._is_text_query(text):
                    continue
                emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                emb = np.asarray(emb, dtype="float32")
                if emb.ndim != 1:
                    emb = emb.reshape(-1)
                vectors.append(emb)
                meta.append({"column": col, "value": text})
        if not vectors:
            self.faiss_index = None
            self.faiss_index_meta = []
            return
        dim = int(vectors[0].shape[0])
        index = faiss.IndexFlatIP(dim)
        matrix = np.vstack([v.reshape(1, -1) for v in vectors]).astype("float32")
        faiss.normalize_L2(matrix)
        index.add(matrix)
        self.faiss_index = index
        self.faiss_index_meta = meta

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
        self._build_faiss_index()
        self._save_indexes(exact_path, lsh_path, semantic_path)
        self._save_faiss_index()
        debug_print("[Schema][Cache] 索引重建并已保存到缓存。")

    def _build_indexes(self) -> None:
        self.exact_index = {}
        self.lsh_index = {}
        debug_print("[Schema][Cache] 开始构建 exact/lsh 索引。")
        for col in self.df.columns:
            values = self.df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            has_lsh = False
            lsh = None
            if MinHash is not None and MinHashLSH is not None:
                lsh = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.lsh_num_perm)
            for val in values.unique():
                self.exact_index.setdefault(val, set()).add(col)
                if lsh is None:
                    continue
                norm = _normalize_lsh_text(val)
                if len(norm) < 2:
                    continue
                mh = MinHash(num_perm=settings.lsh_num_perm)
                for ch in norm:
                    mh.update(ch.encode("utf8"))
                lsh.insert(val, mh)
                has_lsh = True
            if has_lsh and lsh is not None:
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

    def _faiss_index_path(self) -> Path:
        return self.cache_dir / "faiss_index.bin"

    def _faiss_meta_path(self) -> Path:
        return self.cache_dir / "faiss_meta.pkl"

    def _save_faiss_index(self) -> None:
        if faiss is None or self.faiss_index is None:
            return
        faiss.write_index(self.faiss_index, str(self._faiss_index_path()))
        with open(self._faiss_meta_path(), "wb") as f:
            pickle.dump(self.faiss_index_meta, f)

    def _load_faiss_index(self) -> None:
        if faiss is None:
            return
        idx_path = self._faiss_index_path()
        meta_path = self._faiss_meta_path()
        if not idx_path.exists() or not meta_path.exists():
            return
        self.faiss_index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            self.faiss_index_meta = pickle.load(f)

    def _load_indexes(self, exact_path: Path, lsh_path: Path, semantic_path: Path) -> bool:
        if not exact_path.exists() or not semantic_path.exists():
            debug_print("[Schema][Cache] 缓存文件不完整。")
            return False
        try:
            with open(exact_path, "rb") as f:
                self.exact_index = pickle.load(f)
            if lsh_path.exists():
                with open(lsh_path, "rb") as f:
                    self.lsh_index = pickle.load(f)
            else:
                self.lsh_index = {}
            with open(semantic_path, "rb") as f:
                payload = pickle.load(f)
            self.semantic_value_records = payload.get("records", [])
            self.semantic_value_embeddings = []
            self.cache_status.update({"exact_index": True, "lsh_index": bool(self.lsh_index)})
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
            self._load_faiss_index()
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

    def _search_lsh_fuzzy_candidates(self, keyword: str) -> list[dict]:
        if not self.lsh_index:
            return []
        norm_kw = _normalize_lsh_text(keyword)
        if len(norm_kw) < 2:
            return []
        scored: list[tuple[float, str, str]] = []
        for col, lsh in self.lsh_index.items():
            try:
                matches = list(lsh.query(self._make_lsh_minhash(norm_kw)))
            except Exception:
                matches = []
            for val in matches:
                val_norm = _normalize_lsh_text(val)
                seq = _safe_ratio(norm_kw, val_norm)
                jac = _char_jaccard(norm_kw, val_norm)
                score = 0.7 * seq + 0.3 * jac
                scored.append((score, col, val))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for score, col, val in scored:
            if score <= 0:
                continue
            out.append({"实体文本": keyword, "对应匹配值": val, "所在匹配列": col, "匹配方式": "模糊匹配", "相关性分数": round(float(score), 4)})
        return self._dedupe_alignments(out)[: settings.candidate_value_top_k]

    def _make_lsh_minhash(self, text: str):
        mh = MinHash(num_perm=settings.lsh_num_perm)
        for ch in text:
            mh.update(ch.encode("utf8"))
        return mh

    def _search_faiss_semantic(self, keyword: str, top_k: int | None = None) -> list[dict]:
        if faiss is None or self.faiss_index is None or not self.faiss_index_meta:
            return []
        model = self._get_semantic_value_model()
        emb = model.encode(keyword, convert_to_numpy=True, normalize_embeddings=True)
        vec = np.asarray(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(top_k or settings.semantic_value_top_k, len(self.faiss_index_meta))
        scores, idxs = self.faiss_index.search(vec, k)
        out: list[dict] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.faiss_index_meta):
                continue
            if score <= 0:
                continue
            meta = self.faiss_index_meta[idx]
            out.append({"对应匹配值": meta["value"], "所在匹配列": meta["column"], "匹配方式": "向量匹配", "相关性分数": round(float(score), 4)})
        return self._dedupe_alignments(out)

    def retrieve(self, question: str, extracted_entities: list[str] | list[dict] | None = None) -> CandidateSchemaPack:
        q = _normalize_text(question)
        entities = self._extract_entity_texts(extracted_entities)

        ranked: list[dict] = []
        exact_mentioned_columns = _find_exact_column_name_mentions(question, self.column_names)
        if exact_mentioned_columns:
            debug_print(f"[Schema][NameMatch] 问题中命中字段名: {exact_mentioned_columns}")

        pairs = [
            [question, self.column_passage_map.get(c["column_name"], c.get("column_description", ""))]
            for c in self.column_metadata
        ]
        debug_print(f"[Schema][A-route] CrossEncoder 输入对数: {len(pairs)}")
        scores = self.rank_model.predict(pairs)
        for idx, (meta, score) in enumerate(sorted(zip(self.column_metadata, scores), key=lambda x: x[1], reverse=True), start=1):
            col = meta["column_name"]
            prof = self.profile_detail_map.get(col, {})
            item = {
                "列名": col,
                "相关性分数": round(float(score), 4),
                "列描述": meta.get("column_description", ""),
                "字段类型": prof.get("字段类型", meta.get("data_type", "")),
            }
            if prof.get("是否枚举") == "是":
                item["是否枚举"] = "是"
            null_rate = prof.get("空值率")
            if null_rate not in (None, "", "0.00%"):
                item["空值率"] = null_rate
            unique_count = prof.get("唯一值数")
            if unique_count not in (None, "") and int(unique_count) <= 20:
                item["唯一值数"] = unique_count
            samples = prof.get("示例值")
            if samples not in (None, "", []):
                item["示例值"] = samples[:3] if isinstance(samples, list) else samples
            if prof.get("格式") not in (None, ""):
                item["格式"] = prof.get("格式")
            if prof.get("范围") not in (None, ""):
                item["范围"] = prof.get("范围")
            ranked.append(item)

        target_k = settings.candidate_top_k
        top20 = ranked[:min(target_k, len(ranked))]
        top20_cols = [item.get("列名") for item in top20 if item.get("列名")]
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
            exact_hits: list[dict] = []
            if ent in self.exact_index:
                for c in self.exact_index[ent]:
                    exact_hits.append({"实体文本": ent, "对应匹配值": ent, "所在匹配列": c, "匹配方式": "精确匹配", "相关性分数": 1.0})
            if exact_hits:
                deduped = self._dedupe_alignments(exact_hits)
                evidence["精确匹配"][ent] = deduped
                must_have.extend([x["所在匹配列"] for x in deduped])
                continue

            fuzzy_candidates: list[dict] = []
            for meta in self.column_metadata:
                col = meta["column_name"]
                for ex in list(self._value_index.get(col, [])[: settings.candidate_value_top_k]):
                    ex_norm = _normalize_text(ex)
                    if not ex_norm:
                        continue
                    if ex in self.exact_index:
                        continue
                    seq = _safe_ratio(ent_norm, ex_norm)
                    jac = _char_jaccard(ent_norm, ex_norm)
                    combined = 0.7 * seq + 0.3 * jac
                    if combined >= settings.lsh_query_combined_threshold:
                        fuzzy_candidates.append({"实体文本": ent, "对应匹配值": ex, "所在匹配列": col, "匹配方式": "模糊匹配", "相关性分数": round(float(combined), 4)})
                        break
            lsh_hits = self._search_lsh_fuzzy_candidates(ent)
            fuzzy_candidates.extend(lsh_hits)
            if fuzzy_candidates:
                fuzzy_candidates = self._dedupe_alignments(fuzzy_candidates)
                evidence["模糊匹配"][ent] = fuzzy_candidates
                must_have.extend([x["所在匹配列"] for x in fuzzy_candidates])

            if settings.enable_semantic_value_retrieval and self.faiss_index is not None and self._is_text_query(ent):
                sem_hits = self._search_faiss_semantic(ent)
                for hit in sem_hits:
                    evidence["向量匹配"].setdefault(ent, []).append(hit)
                    must_have.append(hit["所在匹配列"])

        must_have = list(dict.fromkeys(must_have))

        top20_set = set(top20_cols)
        missing_name_hits = [c for c in exact_mentioned_columns if c not in top20_set]
        if missing_name_hits:
            debug_print(f"[Schema][NameMatch] 命中字段未进入重排候选，加入必须列: {missing_name_hits}")
            must_have.extend(missing_name_hits)

        must_have = list(dict.fromkeys(must_have))

        cliff_filtered = self._rank_candidates(top20, target_k)
        compact_schema = self._merge_must_have(cliff_filtered, must_have, ranked)

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
        seen_cols = set()
        seen_pairs = set()
        out = []
        for item in items:
            col = item.get("所在匹配列")
            key = (col, item.get("对应匹配值"))
            if col in seen_cols or key in seen_pairs:
                continue
            seen_cols.add(col)
            seen_pairs.add(key)
            out.append(item)
        return out[:3]

    def _merge_must_have(self, top20: list[dict], must_have: list[str], ranked: list[dict]) -> list[dict]:
        """把必须列补入候选，保持 top20 作为主体，避免重复冗余。
        对于不在 ranked 中的列，用 profile_detail_map 构建完整数据结构。"""
        if not must_have:
            return top20
        top20_cols = {item.get("列名") for item in top20 if item.get("列名")}
        ranked_map = {item.get("列名"): item for item in ranked if item.get("列名")}
        meta_map = {m.get("column_name"): m for m in self.column_metadata if m.get("column_name")}
        merged = list(top20)
        for col in must_have:
            if not col or col in top20_cols:
                continue
            item = ranked_map.get(col)
            if not item:
                prof = self.profile_detail_map.get(col, {})
                meta = meta_map.get(col, {})
                item = {
                    "列名": col,
                    "相关性分数": 0.0,
                    "列描述": meta.get("column_description", ""),
                    "字段类型": prof.get("字段类型", meta.get("data_type", "TEXT")),
                }
                if prof.get("是否枚举") == "是":
                    item["是否枚举"] = "是"
                null_rate = prof.get("空值率")
                if null_rate not in (None, "", "0.00%"):
                    item["空值率"] = null_rate
                unique_count = prof.get("唯一值数")
                if unique_count not in (None, "") and int(unique_count) <= 20:
                    item["唯一值数"] = unique_count
                samples = prof.get("示例值")
                if samples not in (None, "", []):
                    item["示例值"] = samples[:3] if isinstance(samples, list) else samples
                if prof.get("格式") not in (None, ""):
                    item["格式"] = prof.get("格式")
                if prof.get("范围") not in (None, ""):
                    item["范围"] = prof.get("范围")
            merged.append(item)
        return merged

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
