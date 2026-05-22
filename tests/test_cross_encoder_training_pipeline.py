from __future__ import annotations

import json
import random
import sys
import types

import pandas as pd
import torch


fake_sentence_transformers = types.ModuleType("sentence_transformers")


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.9 if "物料编码" in pair[1] else 0.5 for pair in pairs]


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, value, **kwargs):
        mapping = {
            "正列": [1.0, 0.0],
            "向量近1": [0.95, 0.05],
            "向量近2": [0.9, 0.1],
            "字符近1": [0.1, 0.9],
            "字符近2": [0.0, 1.0],
            "字符近3": [0.2, 0.8],
            "随机列": [-1.0, 0.0],
        }
        if isinstance(value, list):
            return [mapping.get(v, [0.0, 0.0]) for v in value]
        return mapping.get(value, [0.0, 0.0])


fake_sentence_transformers.CrossEncoder = _FakeCrossEncoder
fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

from pipeline.cross_encoder_passage import build_column_passage_map
from training.dataset_io import load_split_dataframes, read_jsonl, write_jsonl
from training.prepare_data import generate_negatives, process_data
from training.train_cross_encoder import _ensure_padding_token, compute_mixed_loss


def test_column_passage_includes_name_description_and_examples(tmp_path):
    schema_path = tmp_path / "schema.json"
    csv_path = tmp_path / "data.csv"
    schema_path.write_text(
        json.dumps([
            {
                "column_name": "物料编码",
                "column_description": "物料唯一编码",
                "examples": ["500116755"],
            }
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    csv_path.write_text("物料编码\n500116755\n500116756\n", encoding="utf-8")

    passage_map = build_column_passage_map(str(schema_path), str(csv_path), active_columns=["物料编码"])

    passage = passage_map["物料编码"]
    assert "列名称: 物料编码" in passage
    assert "列描述: 物料唯一编码" in passage
    assert "实例值:" in passage
    assert "500116755" in passage


def test_generate_negatives_samples_per_positive_with_sources(monkeypatch):
    import training.prepare_data as prepare_data

    prepare_data._COL_EMBED_CACHE.clear()
    monkeypatch.setattr(prepare_data, "_get_embed_model", lambda: _FakeSentenceTransformer())
    random.seed(3)

    groups = generate_negatives(
        ["正列"],
        ["正列", "向量近1", "向量近2", "字符近1", "字符近2", "字符近3", "随机列"],
        num_hard_char=3,
        num_hard_sem=2,
        num_easy=1,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group["positive_column"] == "正列"
    assert group["negative_sources"].count("semantic") == 2
    assert group["negative_sources"].count("string") == 3
    assert group["negative_sources"].count("random") == 1
    assert len(group["negative_columns"]) == len(set(group["negative_columns"]))
    assert "正列" not in group["negative_columns"]


def test_process_data_keeps_question_columns_when_answer_template_empty(monkeypatch):
    import training.prepare_data as prepare_data

    monkeypatch.setattr(
        prepare_data,
        "generate_negatives",
        lambda *args, **kwargs: [{"positive_column": "问题字段", "negative_columns": ["负字段"], "negative_sources": ["random"]}],
    )
    df = pd.DataFrame([
        {
            "生成问题": "查询问题字段",
            "问题模版": "{问题字段}",
            "回答模版": "{}",
        }
    ])
    passage_map = {"问题字段": "列名称: 问题字段", "负字段": "列名称: 负字段"}

    rows = process_data(df, ["问题字段", "负字段"], passage_map, is_training=True)

    assert any(row["label"] == 1 and row["column_name"] == "问题字段" for row in rows)
    assert any(row["label"] == 0 and row["pos_column"] == "问题字段" for row in rows)


def test_compute_mixed_loss_applies_lambda_to_pair_loss():
    total, point, pair = compute_mixed_loss(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        pos_pair_scores=torch.tensor([0.2]),
        neg_pair_scores=torch.tensor([0.1]),
        margin=0.15,
        pair_lambda=0.7,
    )

    expected_pair = torch.tensor(0.05)
    expected_point = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(pair, expected_pair)
    assert torch.allclose(point, expected_point)
    assert torch.allclose(total, expected_point + 0.7 * expected_pair)


def test_ensure_padding_token_fallbacks_to_eos():
    class _Tokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "</s>"
        eos_token_id = 2

    class _Cfg:
        pad_token_id = None

    class _Backbone:
        config = _Cfg()

    class _Wrapper:
        tokenizer = _Tokenizer()
        model = _Backbone()

    wrapper = _Wrapper()
    _ensure_padding_token(wrapper)  # type: ignore[arg-type]

    assert wrapper.tokenizer.pad_token == "</s>"
    assert wrapper.tokenizer.pad_token_id == 2
    assert wrapper.model.config.pad_token_id == 2


def test_read_jsonl_skips_corrupted_trailing_bytes(tmp_path):
    path = tmp_path / "broken.jsonl"
    good = [{"question": "q1", "label": 1}, {"question": "q2", "label": 0}]
    raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in good) + "\n"
    raw += '{"question": "q3", "label":'
    path.write_bytes(raw.encode("utf-8"))

    items = list(read_jsonl(str(path)))

    assert items == good


def test_write_then_read_jsonl_roundtrip(tmp_path):
    path = tmp_path / "ok.jsonl"
    records = [{"question": f"q{i}", "label": i % 2} for i in range(5)]
    write_jsonl(str(path), records)
    assert list(read_jsonl(str(path))) == records


def test_load_split_dataframes_from_jsonl(tmp_path):
    data_dir = tmp_path
    sample = [
        {"生成问题": "查询甲", "问题模版": "{字段A}", "回答模版": "{字段A}"},
        {"生成问题": "查询乙", "问题模版": "{字段B}", "回答模版": "{}"},
    ]
    write_jsonl(str(data_dir / "train_split.jsonl"), sample)
    write_jsonl(str(data_dir / "val_split.jsonl"), [sample[0]])
    write_jsonl(str(data_dir / "test_split.jsonl"), [sample[1]])

    splits = load_split_dataframes(str(data_dir))

    assert splits is not None
    df_train, df_val, df_test = splits
    assert list(df_train["生成问题"]) == ["查询甲", "查询乙"]
    assert list(df_val["生成问题"]) == ["查询甲"]
    assert list(df_test["生成问题"]) == ["查询乙"]


def test_ensure_padding_token_populates_model_config_when_tokenizer_has_pad():
    class _Tokenizer:
        pad_token_id = 7
        pad_token = "<pad>"
        eos_token = "</s>"
        eos_token_id = 2

    class _Cfg:
        pad_token_id = None

    class _Backbone:
        config = _Cfg()

    class _Wrapper:
        tokenizer = _Tokenizer()
        model = _Backbone()

    wrapper = _Wrapper()
    _ensure_padding_token(wrapper)  # type: ignore[arg-type]

    assert wrapper.tokenizer.pad_token_id == 7
    assert wrapper.model.config.pad_token_id == 7
