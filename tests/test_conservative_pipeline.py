from __future__ import annotations

import asyncio
import sys
import types
import importlib.util
from pathlib import Path


def test_intent_planner_parses_and_normalizes_chinese_response():
    from pipeline.intent_planner import IntentPlanner

    raw = """
    ```json
    {
      "查询目标": "供应商描述",
      "筛选条件": [
        {"条件文本": "4501602436", "含义": "采购订单号", "建议匹配方式": "="}
      ],
      "计算条件": {"聚合": "none", "排序": "none", "top_k": null},
      "风险提示": ["不要替换用户原文订单号"]
    }
    ```
    """

    plan = IntentPlanner.parse_response(raw)

    assert plan["query_target"] == "供应商描述"
    assert plan["filters"][0]["text"] == "4501602436"
    assert plan["filters"][0]["meaning"] == "采购订单号"
    assert plan["calculation"]["aggregation"] == "none"
    assert plan["risk_notes"] == ["不要替换用户原文订单号"]


def test_generator_builds_two_variants_for_each_route_without_llm():
    fake_fewshot = types.ModuleType("pipeline.fewshot_index")

    class FakeFewShotFaissStore:
        pass

    fake_fewshot.FewShotFaissStore = FakeFewShotFaissStore
    sys.modules.setdefault("pipeline.fewshot_index", fake_fewshot)

    from pipeline.generator import SQLGenerator

    generator = SQLGenerator.__new__(SQLGenerator)
    generator._build_fewshot_context = lambda question: ""

    specs = generator.build_generation_prompt_specs(
        question="订单4501602436的供应商是哪家?",
        schema_prompt="## Table: procurement_table",
        intent_plan={"query_target": "供应商描述", "calculation": {"aggregation": "none"}},
    )

    route_counts = {}
    for spec in specs:
        route_counts[spec["type"]] = route_counts.get(spec["type"], 0) + 1

    assert len(specs) == 6
    assert route_counts == {"direct": 2, "icl": 2, "intent_plan": 2}
    assert "plan" not in route_counts
    assert all(spec["variant_id"] in (1, 2) for spec in specs)
    assert all("不得为了覆盖候选字段" in spec["prompt"] for spec in specs)
    assert all(
        "[弱意图解析]" not in spec["prompt"]
        for spec in specs
        if spec["type"] != "intent_plan"
    )
    assert all(
        "[弱意图解析]" in spec["prompt"]
        for spec in specs
        if spec["type"] == "intent_plan"
    )


def test_generator_import_does_not_require_faiss():
    sys.modules.pop("pipeline.generator", None)
    sys.modules.pop("pipeline.fewshot_index", None)

    import pipeline.generator as generator

    assert generator.SQLGenerator.extract_sql('{"sql": "SELECT 1"}') == "SELECT 1"


def test_generator_extracts_sql_from_json_output():
    from pipeline.generator import SQLGenerator

    assert SQLGenerator.extract_sql('{"sql": "SELECT 1"}') == "SELECT 1"
    assert SQLGenerator.extract_sql('```json\n{"sql": "SELECT 2"}\n```') == "SELECT 2"


def test_selector_prefers_result_consensus_over_raw_confidence():
    from pipeline.selector import SQLSelector

    selector = SQLSelector()
    low_confidence_consensus = {
        "type": "direct",
        "sql": 'SELECT DISTINCT "供应商描述" FROM procurement_table WHERE "采购订单号" = "4501602436"',
        "status": "success",
        "result": [("供应商A",)],
        "checker_issues": [],
        "confidence": 0.3,
    }
    matching_candidate = {
        "type": "intent_plan",
        "sql": 'SELECT "供应商描述" FROM procurement_table WHERE "采购订单号" = "4501602436"',
        "status": "success",
        "result": [("供应商A",)],
        "checker_issues": [],
        "confidence": 0.2,
    }
    high_confidence_singleton = {
        "type": "icl",
        "sql": 'SELECT "采购订单号" FROM procurement_table WHERE "采购订单号" = "4501602436"',
        "status": "success",
        "result": [("4501602436",)],
        "checker_issues": [],
        "confidence": 0.95,
    }

    selected, reason, status = selector.select_best(
        "订单4501602436的供应商是哪家?",
        "schema",
        [high_confidence_singleton, low_confidence_consensus, matching_candidate],
    )

    assert status == "success"
    assert selected is low_confidence_consensus
    assert "score=" in reason


def test_schema_markdown_labels_evidence_columns_as_non_mandatory():
    from pipeline.schema_format import build_plan_markdown

    markdown = build_plan_markdown(
        "## Table: procurement_table",
        ["供应商描述"],
        evidence={"精确匹配": {"4501602436": [{"所在匹配列": "采购订单号", "对应匹配值": "4501602436"}]}},
        include_evidence=True,
    )

    assert "参考证据列" in markdown
    assert "不强制" in markdown
    assert "必须" not in markdown


def test_ablation_config_places_no_judge_after_no_entity():
    module_path = Path(__file__).with_name("evaluate_llm_full_schema_vs_system_plan.py")
    spec = importlib.util.spec_from_file_location("eval_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    prefixes = [prefix for _label, prefix, _recall_key in module.ABLATION_CONFIGS]
    no_entity_idx = prefixes.index("main_no_entity")
    no_judge_idx = prefixes.index("main_no_judge")
    no_judge_label = module.ABLATION_CONFIGS[no_judge_idx][0]

    assert no_judge_idx == no_entity_idx + 1
    assert "refiner+一致性投票" in no_judge_label


def test_question_splitter_prompt_uses_markdown_and_preserves_information():
    from pipeline.question_splitter import QuestionSplitter

    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = '{"是否多问题": false, "子问题": []}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]
                usage = None

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    splitter = QuestionSplitter(FakeClient(), "fake-model")
    asyncio.run(splitter._llm_split("问题A？问题B？"))

    user_prompt = calls[0]["messages"][1]["content"]
    assert "## 输入" in user_prompt
    assert "## 输出格式" in user_prompt
    assert "JSON" in user_prompt
    assert "Markdown" in user_prompt
    assert "不得扭曲原文语义" in user_prompt
    assert "不得省略或忽略原文提到的信息" in user_prompt
