from __future__ import annotations

import sys
import types

import pytest


def test_intent_planner_parses_minimal_json_response():
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

    assert plan["查询目标"] == "供应商描述"
    assert plan["筛选条件"][0]["条件文本"] == "4501602436"
    assert plan["计算条件"]["聚合"] == "none"
    assert plan["风险提示"] == ["不要替换用户原文订单号"]


def test_generator_builds_two_variants_for_each_route_without_llm():
    fake_fewshot = types.ModuleType("pipeline.fewshot_index")

    class FakeFewShotFaissStore:
        pass

    fake_fewshot.FewShotFaissStore = FakeFewShotFaissStore
    sys.modules.setdefault("pipeline.fewshot_index", fake_fewshot)

    from pipeline.generator import SQLGenerator

    generator = SQLGenerator.__new__(SQLGenerator)
    generator._build_fewshot_context = lambda question, query_signature=None: ""

    specs = generator.build_generation_prompt_specs(
        question="订单4501602436的供应商是哪家?",
        schema_prompt="## Table: procurement_table",
        intent_plan={"查询目标": "供应商描述", "计算条件": {"聚合": "none"}},
    )

    route_counts = {}
    for spec in specs:
        route_counts[spec["type"]] = route_counts.get(spec["type"], 0) + 1

    assert len(specs) == 8
    assert route_counts == {"direct": 2, "icl": 2, "plan": 2, "intent_plan": 2}
    assert all(spec["variant_id"] in (1, 2) for spec in specs)
    assert all("不得为了覆盖候选字段" in spec["prompt"] for spec in specs)


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
        "type": "plan",
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
