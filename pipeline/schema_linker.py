"""
Schema Linking 模块 —— 三路混合召回 + CrossEncoder 精排。
────────────────────────────────────────────────────────────
A路 (CrossEncoder)   : 「问题 × 列描述」语义相关性精排
B路 (Exact Match)    : 实体值在数据库列值中的精确匹配
C路 (LSH Fuzzy)      : 基于 MinHash LSH 的模糊匹配，仅在 B路 未命中时启用

项目特色：三路互补 + 扩展窗口梯队回退。
"""
from __future__ import annotations

import difflib
import os
import pickle
import re
import time

import pandas as pd
from datasketch import MinHash, MinHashLSH
from sentence_transformers import CrossEncoder

from config.settings import settings
from pipeline.utils import debug_print


def _normalize_lsh_text(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s)).upper()


def _char_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def _parse_m_schema(schema_text: str) -> list[dict]:
    """解析 M-Schema 文本 → 结构化列元数据列表。"""
    metadata = []
    pattern = re.compile(
        r'\s*\(([^:]+):\s*([^,]+),\s*(.*?),\s*Examples:\s*\[(.*?)\]\)'
    )
    for line in schema_text.split("\n"):
        line = line.strip()
        if not line.startswith("("):
            continue
        m = pattern.search(line)
        if m:
            col_name = m.group(1).strip()
            metadata.append({
                "column_name": col_name,
                "data_type": m.group(2).strip(),
                "column_description": m.group(3).strip(),
                "examples": [e.strip() for e in m.group(4).split(",") if e.strip()],
                "raw_text": line,
            })
    debug_print(f"[Schema] 解析 M-Schema，共识别出 {len(metadata)} 个列定义。")
    return metadata


class SchemaLinker:
    """三路混合检索 + CrossEncoder 精排 Schema Linker。"""

    def __init__(
        self,
        schema_path_or_text: str,
        csv_path: str,
        cross_encoder_path: str | None = None,
    ):
        self._cross_encoder_path = cross_encoder_path or settings.cross_encoder_model

        self.csv_path = str(csv_path)
        schema_path_or_text = str(schema_path_or_text)

        self.bert_ranker = CrossEncoder(
            self._cross_encoder_path, trust_remote_code=True,
        )

        if os.path.isfile(schema_path_or_text):
            with open(schema_path_or_text, "r", encoding="utf-8") as f:
                self.schema_text = f.read()
        else:
            self.schema_text = schema_path_or_text

        self.column_metadata = _parse_m_schema(self.schema_text)
        self.column_names = [c["column_name"] for c in self.column_metadata]
        self.exact_index: dict[str, set[str]] = {}
        self.lsh_index: dict[str, MinHashLSH] = {}

        self.cache_dir = str(settings.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        if not self._load_cache():
            debug_print("[Schema] 开始构建值索引...")
            self._build_value_indices()
            self._save_cache()
        else:
            debug_print("[Schema] 索引加载成功")

    # ──────────── 索引构建 ────────────

    def _build_value_indices(self):
        df = pd.read_csv(self.csv_path, dtype=str)
        target_cols = [
            c["column_name"] for c in self.column_metadata
            if c["column_name"] in df.columns
        ]
        for col in target_cols:
            lsh = MinHashLSH(
                threshold=settings.lsh_threshold,
                num_perm=settings.lsh_num_perm,
            )
            has_lsh = False
            for val in df[col].dropna().unique():
                val_str = str(val).strip()
                if not val_str:
                    continue
                if val_str not in self.exact_index:
                    self.exact_index[val_str] = set()
                self.exact_index[val_str].add(col)

                if len(val_str) >= 2:
                    norm = _normalize_lsh_text(val_str)
                    if norm:
                        mh = MinHash(num_perm=settings.lsh_num_perm)
                        for ch in norm:
                            mh.update(ch.encode("utf8"))
                        lsh.insert(val_str, mh)
                        has_lsh = True
            if has_lsh:
                self.lsh_index[col] = lsh
        debug_print(
            f"[Schema] 值索引构建完成。精准索引含 {len(self.exact_index)} 个唯一值。"
        )

    # ──────────── 缓存 ────────────

    def _save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        data = {
            "version": 2,
            "names": self.column_names,
            "exact": self.exact_index,
            "lsh": self.lsh_index,
            "timestamp": os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else time.time(),
            "cross_encoder_path": self._cross_encoder_path,
        }
        path = os.path.join(self.cache_dir, "index.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        debug_print(f"[Schema] 索引已缓存至: {path}")

    def _load_cache(self) -> bool:
        path = os.path.join(self.cache_dir, "index.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != 2:
                debug_print("[Schema] 缓存版本不兼容，重建索引。")
                return False
            if data["timestamp"] != os.path.getmtime(self.csv_path):
                debug_print("[Schema] CSV已更新，缓存失效。")
                return False
            if data.get("cross_encoder_path") != self._cross_encoder_path:
                debug_print("[Schema] 模型路径已变更，缓存失效。")
                return False
            self.column_names = data["names"]
            self.exact_index = data["exact"]
            self.lsh_index = data["lsh"]
            return True
        except Exception:
            return False

    # ──────────── C路二级校验 ────────────

    @staticmethod
    def _lsh_secondary_verify(norm_kw: str, norm_val: str) -> bool:
        if not norm_kw or not norm_val:
            return False
        if norm_kw == norm_val:
            return True
        seq = difflib.SequenceMatcher(None, norm_kw, norm_val).ratio()
        jac = _char_jaccard(norm_kw, norm_val)
        sa, sb = set(norm_kw), set(norm_val)
        q_cov = len(sa & sb) / len(sa) if sa else 0.0
        if seq >= settings.c_secondary_seq_ratio:
            return True
        if jac >= settings.c_secondary_jaccard and seq >= settings.c_secondary_seq_with_jac:
            return True
        if len(norm_kw) >= 2 and q_cov >= settings.c_query_cover:
            return True
        return False

    # ──────────── 核心混合检索 ────────────

    def hybrid_retrieve(
        self,
        question: str,
        extracted_keywords: list[str],
        top_k_embed: int | None = None,
    ) -> tuple[list[tuple[str, float]], set[str], dict]:
        """
        三路混合检索。返回 (排序列表, 必须包含列集合, 证据详情)。
        """
        top_k = top_k_embed or settings.top_k_embed
        evidence: dict = {"exact_matches": {}, "fuzzy_matches": {}, "debug_retrieve": {}}
        debug_info: dict = {"B_exact": [], "C_lsh": [], "A_topk_for_c": []}

        # B路：精准匹配
        for kw in extracted_keywords:
            kw_clean = kw.strip()
            if kw_clean in self.exact_index:
                matched_cols = self.exact_index[kw_clean]
                evidence["exact_matches"][kw_clean] = list(matched_cols)
                debug_info["B_exact"].append((kw_clean, list(matched_cols)))

        # A路：CrossEncoder 精排
        pairs = [[question, c["column_description"]] for c in self.column_metadata]
        scores = self.bert_ranker.predict(pairs)
        ranked = sorted(
            zip([c["column_name"] for c in self.column_metadata], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        a_topk = [name for name, _ in ranked[:max(1, top_k)]]
        a_topk_set = set(a_topk)
        debug_info["A_topk_for_c"] = a_topk

        # C路：LSH 模糊匹配（仅 B路 未命中的关键词）
        kw_for_c = []
        for kw in extracted_keywords:
            kw_clean = kw.strip()
            if kw_clean in self.exact_index:
                continue
            norm = _normalize_lsh_text(kw_clean)
            if len(norm) >= 2:
                kw_for_c.append((kw, norm))

        for kw, norm_kw in kw_for_c:
            kw_mh = MinHash(num_perm=settings.lsh_num_perm)
            for ch in norm_kw:
                kw_mh.update(ch.encode("utf8"))
            fuzzy_hits: dict = {}
            for col in a_topk_set:
                if col not in self.lsh_index:
                    continue
                res = self.lsh_index[col].query(kw_mh)
                if not res:
                    continue
                verified = [
                    v for v in res
                    if self._lsh_secondary_verify(norm_kw, _normalize_lsh_text(v))
                ]
                if verified:
                    fuzzy_hits[col] = verified[:2]
                    debug_info["C_lsh"].append((kw, col, verified[:3]))
            if fuzzy_hits:
                evidence["fuzzy_matches"][kw] = fuzzy_hits

        evidence["debug_retrieve"] = debug_info

        must_have = set()
        for cols in evidence["exact_matches"].values():
            must_have.update(cols)
        for col_dict in evidence["fuzzy_matches"].values():
            must_have.update(col_dict.keys())

        return ranked, must_have, evidence

    # ──────────── 辅助方法 ────────────

    def build_entity_schema(self, col_names: list[str]) -> str:
        """构建不含 Examples 的精简 Schema（用于实体提取 Prompt）。"""
        lines = ["以下是数据库中与问题相关的字段："]
        for col in col_names:
            meta = next(
                (m for m in self.column_metadata if m["column_name"] == col), None
            )
            if meta:
                lines.append(f"- {col}：{meta.get('column_description', '')}")
        return "\n".join(lines)

    def format_for_selector(self, candidates: list[str]) -> str:
        lines = ["[Schema Candidates]"]
        for col_name in candidates:
            meta = next(
                (c for c in self.column_metadata if c["column_name"] == col_name), None
            )
            if meta:
                exs = ", ".join(meta.get("examples", [])[:2])
                lines.append(f"- {col_name}: {meta['column_description']} (Ex: {exs})")
        return "\n".join(lines)
