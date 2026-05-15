from __future__ import annotations

import sys
import types


fake_sentence_transformers = types.ModuleType("sentence_transformers")


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.9 for _ in pairs]


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

from pipeline.system import TextToSQLSystem


class _FakeLinkerPack:
    def __init__(self):
        self.全量排序 = [
            {"列名": "物料编码", "相关性分数": 0.99, "列描述": "物料编码"},
            {"列名": "中标厂家", "相关性分数": 0.98, "列描述": "中标厂家"},
        ]
        self.Top20候选 = self.全量排序
        self.必须列集合 = ["物料编码", "中标厂家"]
        self.证据详情 = {}
        self.精简schema = [
            {"列名": "物料编码", "相关性分数": 0.99, "列描述": "物料编码", "字段类型": "TEXT", "是否枚举": "否", "空值率": "", "唯一值数": "", "示例值": "", "格式": "", "范围": ""},
            {"列名": "中标厂家", "相关性分数": 0.98, "列描述": "中标厂家", "字段类型": "TEXT", "是否枚举": "否", "空值率": "", "唯一值数": "", "示例值": "", "格式": "", "范围": ""},
        ]
        self.给LLM的中文结构 = {
            "召回schema": self.Top20候选,
            "证据实体": ["500116755"],
            "实体对齐结果": [{"实体文本": "500116755", "候选对齐": [{"对应匹配值": "500116755", "所在匹配列": "物料编码", "匹配方式": "精确匹配"}]}],
            "精简schema": self.精简schema,
        }


class _FakeLinker:
    column_names = ["物料编码", "中标厂家"]
    column_metadata = [
        {"column_name": "物料编码", "column_description": "物料编码", "data_type": "TEXT"},
        {"column_name": "中标厂家", "column_description": "中标厂家", "data_type": "TEXT"},
    ]
    profile_map = {"物料编码": "[字段类型=TEXT, 示例=500116755]", "中标厂家": "[字段类型=TEXT, 示例=甲公司]"}

    def retrieve(self, question, entities):
        return _FakeLinkerPack()

    def build_llm_prompt_payload(self, pack):
        return pack.给LLM的中文结构


class _FakeExtractor:
    async def extract(self, question, candidate_schema_pack, tracker=None, max_retries=2, schema_columns=None):
        return ["500116755"]


class _FakeGenerator:
    def build_m_schema_prompt(self, *args, **kwargs):
        return "[Schema Prompt]"

    async def generate_from_plan_async(self, question, schema_prompt, plan_json, tracker):
        return {"sql": "SELECT \"物料编码\" FROM procurement_table", "status": "success", "confidence": 0.9, "result": [("500116755",)]}


class _FakeRefiner:
    async def refine_async(self, question, schema_prompt, candidates, valid_columns, tracker):
        return candidates


class _FakeSelector:
    def select_best(self, question, schema_prompt, candidates):
        return candidates[0], "confidence=0.90", "success"


class _FakeDB:
    def execute_sql(self, sql):
        return [("500116755",)], None


class _FakeSystem(TextToSQLSystem):
    def __init__(self):
        self.linker = _FakeLinker()
        self.entity_extractor = _FakeExtractor()
        self.generator = _FakeGenerator()
        self.refiner = _FakeRefiner()
        self.selector = _FakeSelector()
        self.db_engine = _FakeDB()


def test_build_plan_json_uses_new_chinese_payload_shape():
    system = _FakeSystem()
    pack = system.linker.retrieve("q", [])
    plan = system._build_plan_json("物料编码500116755的中标厂家有哪些？", pack, ["500116755"])

    assert plan["候选字段"][0]["列名"] == "物料编码"
    assert plan["证据实体"] == ["500116755"]
    assert plan["精简schema"][0]["相关性分数"] == 0.99
    assert plan["选中字段"] == ["物料编码", "中标厂家"]


def test_debug_payload_builders_expose_llm_ready_sections():
    system = _FakeSystem()
    pack = system.linker.retrieve("q", [])
    payload = system.linker.build_llm_prompt_payload(pack)
    assert set(payload) == {"召回schema", "证据实体", "实体对齐结果", "精简schema"}
    assert payload["证据实体"] == ["500116755"]
