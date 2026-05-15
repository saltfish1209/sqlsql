import sys
import types


fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", fake_sentence_transformers)

from pipeline.generator import SQLGenerator
from training.template_split import split_template_keys


class FakeSeries:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeTopResults:
    indices = [1, 0]


class FakeScores:
    def topk(self, k):
        return FakeTopResults()


class FakeUtil:
    @staticmethod
    def cos_sim(q_emb, template_embs):
        return [FakeScores()]


class FakeRow:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class FakeILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return FakeRow(self._rows[idx])


class FakeTemplateDF:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = FakeILoc(rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return FakeSeries([row[key] for row in self._rows])


class FakeEmbedModel:
    def encode(self, values, **kwargs):
        return values


def test_split_template_keys_keeps_boundaries_between_templates():
    keys = ["T1", "T2", "T3", "T4", "T5"]

    train_keys, val_keys, test_keys = split_template_keys(
        keys,
        train_split=0.6,
        val_split=0.0,
    )

    assert train_keys == ["T1", "T2", "T3"]
    assert val_keys == []
    assert test_keys == ["T4", "T5"]
    assert set(train_keys).isdisjoint(test_keys)


def test_icl_examples_use_question_and_answer_templates(monkeypatch):
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.util = FakeUtil
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    generator = object.__new__(SQLGenerator)
    generator.embed_model = FakeEmbedModel()
    generator.template_embs = ["emb0", "emb1"]
    generator.qa_template_df = FakeTemplateDF([
        {
            "问题模版": "训练问题模板A",
            "回答模版": "{供应商名称}",
            "SQL模版": "SELECT old_a",
        },
        {
            "问题模版": "训练问题模板B",
            "回答模版": "{采购订单号}",
            "SQL模版": "SELECT old_b",
        },
    ])

    examples = generator._get_top_k_examples("查询订单", k=2)

    assert "类似问题1：训练问题模板B" in examples
    assert "目标回答字段1：{采购订单号}" in examples
    assert "类似问题2：训练问题模板A" in examples
    assert "目标回答字段2：{供应商名称}" in examples
    assert "问题模版" not in examples
    assert "回答模版" not in examples
