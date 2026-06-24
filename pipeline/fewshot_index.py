from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config.settings import settings
from training.template_split import non_test_template_dataframe


@dataclass
class FewShotRecord:
    q_masked: str
    sql_masked: str


class FewShotFaissStore:
    """离线构建 + 在线召回：
    1) 用槽位信息JSON对`生成问题`与`SQL语句`做value_n掩码
    2) 基于掩码后问题创建 FAISS 索引
    3) 维护问题->SQL 一一映射，在线召回时拼接 few-shot prompt
    """

    def __init__(
        self,
        embed_model_name_or_path: str,
        index_dir: str | Path,
        csv_path: str | Path,
        question_col: str = "生成问题",
        sql_col: str = "SQL语句",
        slots_col: str = "槽位信息JSON",
        normalize_embeddings: bool = True,
    ) -> None:
        self.embed_model_name_or_path = embed_model_name_or_path
        self.index_dir = Path(index_dir)
        self.csv_path = Path(csv_path)
        self.question_col = question_col
        self.sql_col = sql_col
        self.slots_col = slots_col
        self.normalize_embeddings = normalize_embeddings

        self.index_path = self.index_dir / "fewshot_questions.faiss"
        self.meta_path = self.index_dir / "fewshot_q2sql.pkl"

        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.records: list[FewShotRecord] = []

    # -------------------- 掩码逻辑 --------------------
    @staticmethod
    def _parse_slots(slot_json: str | dict[str, Any] | None) -> dict[str, str]:
        if slot_json is None:
            return {}
        if isinstance(slot_json, dict):
            if "raw" in slot_json and isinstance(slot_json["raw"], dict):
                return {str(k): str(v) for k, v in slot_json["raw"].items() if v is not None}
            return {str(k): str(v) for k, v in slot_json.items() if v is not None}

        text = str(slot_json).strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if "raw" in obj and isinstance(obj["raw"], dict):
                    return {str(k): str(v) for k, v in obj["raw"].items() if v is not None}
                return {str(k): str(v) for k, v in obj.items() if v is not None}
        except Exception:
            return {}
        return {}

    @staticmethod
    def _build_raw_value_to_token(slot_mapping: dict[str, str]) -> dict[str, str]:
        raw_value_to_token: dict[str, str] = {}
        idx = 1
        for _, v in slot_mapping.items():
            raw = str(v).strip()
            if not raw:
                continue
            raw_value_to_token[raw] = f"value_{idx}"
            idx += 1
        return raw_value_to_token

    @staticmethod
    def _mask_text(text: str, raw_value_to_token: dict[str, str]) -> str:
        if not text:
            return ""
        masked = text
        for raw in sorted(raw_value_to_token.keys(), key=len, reverse=True):
            masked = masked.replace(raw, raw_value_to_token[raw])
        return masked

    @staticmethod
    def _mask_sql(sql: str, raw_value_to_token: dict[str, str]) -> str:
        if not sql:
            return ""
        masked = sql
        for raw in sorted(raw_value_to_token.keys(), key=len, reverse=True):
            token = raw_value_to_token[raw]
            masked = re.sub(r"'" + re.escape(raw) + r"'", f"'{token}'", masked)
            masked = re.sub(re.escape(raw), token, masked)
        return masked

    # -------------------- 编码 & 索引 --------------------
    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.embed_model_name_or_path)
        return self.model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def build_offline_index(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.csv_path)
        if self.question_col not in df.columns or self.sql_col not in df.columns or self.slots_col not in df.columns:
            raise ValueError(
                f"CSV缺少必要列，要求包含: {self.question_col}, {self.sql_col}, {self.slots_col}"
            )

        # 防越界：ICL 索引仅使用非测试模板（train+val），避免 test 模板泄漏
        if "问题模版" in df.columns:
            df = non_test_template_dataframe(
                df,
                template_col="问题模版",
                train_split=settings.train_split,
                val_split=settings.val_split,
            )

        records: list[FewShotRecord] = []
        for _, row in df.iterrows():
            q = str(row.get(self.question_col, "") or "").strip()
            sql = str(row.get(self.sql_col, "") or "").strip()
            if not q or not sql:
                continue

            slots = self._parse_slots(row.get(self.slots_col))
            raw_to_token = self._build_raw_value_to_token(slots)
            q_masked = self._mask_text(q, raw_to_token)
            sql_masked = self._mask_sql(sql, raw_to_token)
            records.append(FewShotRecord(q_masked=q_masked, sql_masked=sql_masked))

        if not records:
            raise RuntimeError("未生成任何可用样本，请检查CSV内容")

        questions = [r.q_masked for r in records]
        vectors = self._encode(questions)

        dim = vectors.shape[1]
        if self.normalize_embeddings:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(records, f)

        self.index = index
        self.records = records

    def load_online_resources(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("索引或映射文件不存在，请先执行 build_offline_index()")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.records = pickle.load(f)
        self._ensure_model()

    def retrieve(self, query: str, top_k: int = 3) -> list[FewShotRecord]:
        if self.index is None or not self.records:
            self.load_online_resources()

        query = (query or "").strip()
        if not query:
            return []

        qv = self._encode([query])
        top_k = max(1, min(top_k, len(self.records)))
        _, idx = self.index.search(qv, top_k)

        results: list[FewShotRecord] = []
        for i in idx[0].tolist():
            if 0 <= i < len(self.records):
                results.append(self.records[i])
        return results

    def build_fewshot_prompt(self, query: str, top_k: int = 3) -> str:
        hits = self.retrieve(query=query, top_k=top_k)
        if not hits:
            return ""

        blocks: list[str] = []
        for i, h in enumerate(hits, 1):
            blocks.append(
                f"[示例{i}]\n问题: {h.q_masked}\nSQL: {h.sql_masked}"
            )
        return "\n\n".join(blocks)


if __name__ == "__main__":
    from config.settings import settings

    store = FewShotFaissStore(
        embed_model_name_or_path=settings.embed_model,
        index_dir=settings.cache_dir / "fewshot_index",
        csv_path=settings.qa_template_csv,
    )
    store.build_offline_index()
    print(f"索引已构建: {store.index_path}")
    print(f"映射已构建: {store.meta_path}")
