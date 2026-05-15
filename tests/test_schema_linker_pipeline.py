from __future__ import annotations

import sys
import types


fake_sentence_transformers = types.ModuleType("sentence_transformers")


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        scores = []
        for question, desc in pairs:
            q = (question or "").lower()
            d = (desc or "").lower()
            if "物料编码" in d or "物料编码" in q:
                scores.append(0.99)
            elif "中标厂家" in d or "中标厂家" in q:
                scores.append(0.95)
            else:
                scores.append(0.1)
        return scores


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, value, **kwargs):
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]


fake_sentence_transformers.CrossEncoder = _FakeCrossEncoder
fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

from pipeline.schema_linker import SchemaLinker


class FakeProfiles:
    pass


def test_schema_linker_returns_chinese_payload_and_deduped_alignments(monkeypatch, tmp_path):
    csv_path = tmp_path / "data.csv"
    schema_path = tmp_path / "schema.json"
    csv_path.write_text("物料编码,中标厂家,中标日期\n500116755,甲公司,2024-01-01\n500116756,乙公司,2024-01-02\n", encoding="utf-8")
    schema_path.write_text(
        "[{\"column_name\": \"物料编码\", \"column_description\": \"物料的唯一编码\"},"
        "{\"column_name\": \"中标厂家\", \"column_description\": \"中标供应商名称\"},"
        "{\"column_name\": \"中标日期\", \"column_description\": \"中标时间\"}]",
        encoding="utf-8",
    )

    linker = SchemaLinker(str(schema_path), str(csv_path))
    pack = linker.retrieve("物料编码500116755的中标厂家有哪些？", ["500116755", "中标厂家"])

    assert len(pack.全量排序) == 3
    assert pack.全量排序[0]["相关性分数"] >= pack.全量排序[1]["相关性分数"]
    assert len(pack.Top20候选) == 3
    assert "相关性分数" in pack.Top20候选[0]
    assert "证据来源" not in pack.Top20候选[0]
    assert len(pack.精简schema) >= 2

    payload = linker.build_llm_prompt_payload(pack)
    assert "召回schema" in payload
    assert "证据实体" in payload
    assert "实体对齐结果" in payload
    assert "精简schema" in payload

    assert all(isinstance(x, str) for x in pack.给LLM的中文结构["证据实体"])


def test_schema_linker_entity_alignment_is_deduped_to_top3(tmp_path):
    csv_path = tmp_path / "data.csv"
    schema_path = tmp_path / "schema.json"
    csv_path.write_text("物料编码,中标厂家,物资唯一码\n500116755,甲公司,500116755\n", encoding="utf-8")
    schema_path.write_text(
        "[{\"column_name\": \"物料编码\", \"column_description\": \"物料的唯一编码\"},"
        "{\"column_name\": \"中标厂家\", \"column_description\": \"中标供应商名称\"},"
        "{\"column_name\": \"物资唯一码\", \"column_description\": \"物资唯一标识\"}]",
        encoding="utf-8",
    )

    linker = SchemaLinker(str(schema_path), str(csv_path))
    pack = linker.retrieve("物料编码500116755", ["500116755"])
    alignments = pack.给LLM的中文结构["实体对齐结果"]
    assert alignments
    assert len(alignments[0]["候选对齐"]) <= 3
    cols = [item["所在匹配列"] for item in alignments[0]["候选对齐"]]
    assert len(cols) == len(set(cols))
