from __future__ import annotations

import asyncio
import json
import sys
import types
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def test_batch_judge_response_keeps_reasons_only_for_wrong_sqls():
    from pipeline.sql_consistency_judge import parse_batch_judge_response

    raw = """
    {
      "items": [
        {"index": 0, "correct": true},
        {"index": 1, "correct": false, "reason": "semantic mismatch"}
      ]
    }
    """

    items = parse_batch_judge_response(raw)

    assert items == [
        {"index": 0, "correct": True, "status": "pass", "reason": "", "risk": 0},
        {"index": 1, "correct": False, "status": "fail", "reason": "semantic mismatch", "risk": 2},
    ]


def test_batch_judge_response_supports_three_state_status():
    from pipeline.sql_consistency_judge import parse_batch_judge_response

    raw = """
    {
      "items": [
        {"index": 0, "status": "pass"},
        {"index": 1, "status": "suspicious", "reason": "condition may be too broad"},
        {"index": 2, "status": "fail", "reason": "wrong target column"}
      ]
    }
    """

    items = parse_batch_judge_response(raw)

    assert items == [
        {"index": 0, "correct": True, "status": "pass", "reason": "", "risk": 0},
        {
            "index": 1,
            "correct": False,
            "status": "suspicious",
            "reason": "condition may be too broad",
            "risk": 1,
        },
        {"index": 2, "correct": False, "status": "fail", "reason": "wrong target column", "risk": 2},
    ]


def test_batch_judge_prompt_includes_count_sum_rule():
    from pipeline.sql_consistency_judge import judge_sql_batch_consistency

    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = '{"items": [{"index": 0, "correct": true}]}'

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

    asyncio.run(
        judge_sql_batch_consistency(
            FakeClient(),
            "fake-model",
            question="how many suppliers?",
            schema_prompt="## Table: procurement_table",
            candidates=[{"sql": "SELECT COUNT(DISTINCT supplier) FROM procurement_table", "result": [(1,)]}],
            intent_plan={},
        )
    )

    prompt = calls[0]["messages"][0]["content"]
    assert "COUNT" in prompt
    assert "SUM" in prompt


def test_refiner_appends_repair_only_for_fail_status(monkeypatch):
    fake_db_engine = types.ModuleType("pipeline.db_engine")
    fake_db_engine.DBEngine = object
    fake_fewshot = types.ModuleType("pipeline.fewshot_index")
    fake_fewshot.FewShotFaissStore = object
    monkeypatch.setitem(sys.modules, "pipeline.db_engine", fake_db_engine)
    monkeypatch.setitem(sys.modules, "pipeline.fewshot_index", fake_fewshot)
    sys.modules.pop("pipeline.generator", None)
    sys.modules.pop("pipeline.refiner", None)

    import pipeline.sql_consistency_judge as judge_module
    from pipeline.refiner import SQLRefiner

    class FakeTracker:
        def track(self, _resp):
            pass

    class FakeDB:
        def execute_sql(self, sql):
            if sql == "SELECT fixed":
                return [("fixed",)], None
            return [("row",)], None

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)

            class Message:
                content = '{"sql": "SELECT fixed"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]
                usage = None

            return Response()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    judge_calls = []

    async def fake_batch_judge(*_args, **_kwargs):
        judge_calls.append(_kwargs)
        if len(judge_calls) > 1:
            return [
                {"index": 0, "correct": True, "reason": ""},
                {"index": 1, "correct": True, "reason": ""},
                {"index": 2, "correct": True, "reason": ""},
                {"index": 3, "correct": True, "reason": ""},
            ]
        return [
            {"index": 0, "status": "suspicious", "reason": "may be too broad"},
            {"index": 1, "status": "fail", "reason": "missing status filter"},
            {"index": 2, "status": "pass", "reason": ""},
        ]

    monkeypatch.setattr(judge_module, "judge_sql_batch_consistency", fake_batch_judge)
    client = FakeClient()
    refiner = SQLRefiner(client, "fake-model", FakeDB())

    refined = asyncio.run(
        refiner.refine_async(
            "schema",
            [
                {"type": "direct", "variant_id": 1, "sql": "SELECT suspicious"},
                {"type": "direct", "variant_id": 2, "sql": "SELECT wrong"},
                {"type": "intent_plan", "variant_id": 1, "sql": "SELECT ok"},
            ],
            [],
            FakeTracker(),
            repair_schema_prompt="topk schema",
            question="which suppliers are already stocked?",
            judge_schema_prompt="cliff schema",
        )
    )

    assert len(refined) == 4
    assert refined[0]["sql"] == "SELECT suspicious"
    assert refined[0]["status"] == "success"
    assert refined[0]["judge_status"] == "suspicious"
    assert refined[0]["judge_risk"] == 1
    assert refined[1]["sql"] == "SELECT wrong"
    assert refined[1]["status"] == "success"
    assert refined[1]["judge_status"] == "fail"
    assert refined[1]["judge_suggestion"] == "missing status filter"
    assert refined[2]["sql"] == "SELECT ok"
    assert refined[2]["status"] == "success"
    assert refined[3]["sql"] == "SELECT fixed"
    assert refined[3]["status"] == "success"
    assert refined[3]["is_refined"] is True
    assert refined[3]["refined_from"] == 1
    assert refined[3]["repair_changed"] is True
    assert refined[3]["judge_status"] == "fail"
    assert refined[3]["judge_suggestion"] == "missing status filter"
    assert len(client.chat.completions.calls) == 1
    assert len(judge_calls) == 2


def test_consensus_tiebreak_prefers_low_risk_precise_result():
    from pipeline.consensus_vote import select_by_consensus

    broad = {
        "type": "intent_plan",
        "variant_id": 1,
        "status": "success",
        "judge_status": "suspicious",
        "judge_risk": 1,
        "result": [("wrong",), ("right",)],
    }
    precise = {
        "type": "intent_plan",
        "variant_id": 2,
        "status": "success",
        "judge_status": "pass",
        "judge_risk": 0,
        "result": [("right",)],
    }

    selected, reason, status = select_by_consensus([broad, precise])

    assert status == "success"
    assert selected is precise
    assert "risk=0" in reason


def test_value_link_probe_adds_risk_without_rewriting_sql():
    from pipeline.consensus_vote import annotate_candidate_risk

    sql = (
        'SELECT "供应商描述" FROM procurement_table WHERE '
        '("物料描述" LIKE \'%项目长名称%\' OR "物料描述" LIKE \'%具体物料描述%\') '
        'AND "市级属地" = \'不存在属地\''
    )
    candidate = {"sql": sql, "status": "success", "result": [("A",), ("B",)]}

    annotate_candidate_risk(candidate, lambda _column, _literal: False)

    assert candidate["sql"] == sql
    assert candidate["value_link_risk"] > 0
    assert "broad_or_result" in candidate["value_link_probe"]["reasons"]
    assert "same_column_multi_literal:物料描述" in candidate["value_link_probe"]["reasons"]
    assert any(
        reason.startswith("literal_not_found:市级属地=")
        for reason in candidate["value_link_probe"]["reasons"]
    )


def test_eval_mechanism_metrics_track_candidate_hits_and_repairs():
    module_path = Path(__file__).with_name("evaluate_llm_full_schema_vs_system_plan.py")
    spec = importlib.util.spec_from_file_location("eval_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    source = {
        "type": "direct",
        "variant_id": 1,
        "sql": "SELECT wrong",
        "status": "success",
        "result": [("wrong",)],
        "judge_status": "fail",
    }
    repaired = {
        "type": "direct",
        "variant_id": 1,
        "sql": "SELECT right",
        "status": "success",
        "result": [("right",)],
        "judge_status": "fail",
        "is_refined": True,
        "refined_from": 0,
        "source_sql": "SELECT wrong",
    }
    trace = module._pipeline_trace_fields([source], [source, repaired], repaired)
    output = {
        "final_sql": repaired["sql"],
        "execution_result": repaired["result"],
        "reason": "success",
        **trace,
    }

    detail = module._build_experiment_detail_record(
        idx=1,
        label="main",
        prefix="main_system",
        question="question",
        raw_gt="right",
        output=output,
    )

    assert detail["pass_at_k"] == 1
    assert detail["candidate_pool_has_correct_sql"] is True
    assert detail["selected_from_refiner"] is True
    assert detail["selected_changed_by_refiner"] == 1
    assert detail["judge_trigger_count"] == 1
    assert detail["repair_success_count"] == 1
    assert detail["repair_harm_count"] == 0
    assert detail["refined_candidates"][0]["candidate_correct"] is False
    assert detail["refined_candidates"][1]["candidate_correct"] is True


def test_generator_normalizes_spaces_inside_sql_literals():
    from pipeline.generator import SQLGenerator

    sql = '{"sql": "SELECT * FROM procurement_table WHERE \\"计划批次名称\\" = \'协议库存可视化选购 20230421\'"}'

    assert SQLGenerator.extract_sql(sql) == (
        'SELECT * FROM procurement_table WHERE "计划批次名称" = '
        "'协议库存可视化选购20230421'"
    )


def test_evaluator_matches_legacy_comma_gt_to_structured_rows():
    from training.evaluate import _compare_results, normalize_execution_result, parse_ground_truth

    gt = parse_ground_truth("祥兴电气有限公司,杭州普安科技有限公司")
    pred = normalize_execution_result([("祥兴电气有限公司",), ("杭州普安科技有限公司",)])

    ok, _score, match_type = _compare_results(gt, pred)

    assert ok
    assert match_type == "legacy_comma_list"


def test_evaluator_matches_legacy_comma_gt_to_row_separator_prediction():
    from training.evaluate import _compare_results, normalize_execution_result, parse_ground_truth

    gt = parse_ground_truth("祥兴电气有限公司,杭州普安科技有限公司")
    pred = normalize_execution_result([("祥兴电气有限公司&杭州普安科技有限公司",)])

    ok, _score, match_type = _compare_results(gt, pred)

    assert ok
    assert match_type == "legacy_comma_list"


def test_evaluator_keeps_comma_material_description_as_single_value():
    from training.evaluate import _compare_results, normalize_execution_result, parse_ground_truth

    gt = parse_ground_truth("10kV变压器,800kVA,普通,硅钢片,干式")
    pred = normalize_execution_result([("10kV变压器",)])

    ok, _score, match_type = _compare_results(gt, pred)

    assert not ok
    assert match_type == "mismatch"


def test_eval_runner_compare_output_handles_legacy_comma_rows():
    module_path = Path(__file__).with_name("evaluate_llm_full_schema_vs_system_plan.py")
    spec = importlib.util.spec_from_file_location("eval_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    ok, match_type = module._compare_output(
        "祥兴电气有限公司,杭州普安科技有限公司",
        {"execution_result": [("祥兴电气有限公司",), ("杭州普安科技有限公司",)]},
    )

    assert ok
    assert match_type == "legacy_comma_list"


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
    labels = {prefix: label for label, prefix, _recall_key in module.ABLATION_CONFIGS}
    recall_keys = {prefix: recall_key for _label, prefix, recall_key in module.ABLATION_CONFIGS}
    main_idx = prefixes.index("main_system")
    topk_idx = prefixes.index("main_topk_schema")
    no_entity_idx = prefixes.index("main_no_entity")
    no_judge_idx = prefixes.index("main_no_judge")
    no_judge_label = module.ABLATION_CONFIGS[no_judge_idx][0]

    assert topk_idx == main_idx + 1
    assert labels["main_system"] == "主系统"
    assert labels["main_topk_schema"] == "主系统+topk截断schema"
    assert recall_keys["main_system"] == "main_system_schema_recall"
    assert recall_keys["main_topk_schema"] == "main_topk_schema_recall"
    assert no_judge_idx == no_entity_idx + 1
    assert "refiner+一致性投票" in no_judge_label


def test_extra_correct_loader_keeps_existing_baseline_record():
    module_path = Path(__file__).with_name("evaluate_llm_full_schema_vs_system_plan.py")
    spec = importlib.util.spec_from_file_location("eval_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    path = Path(__file__).with_name(".tmp_ablation_extra_correct_test.json")
    try:
        path.write_text(
            json.dumps(
                {
                    "baseline": {
                        "label": "direct",
                        "prefix": "main_direct_only",
                        "correct_count": 1,
                        "correct_idx": [1],
                    },
                    "experiments": [
                        {
                            "label": "main",
                            "prefix": "main_system",
                            "correct_count": 2,
                            "correct_idx": [1, 2],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        records = module._load_extra_correct_records(path)
        report = module._build_extra_correct_report(
            group_label="Ablation",
            configs=[
                ("main", "main_system", "main_system_schema_recall"),
                ("direct", "main_direct_only", "main_direct_only_recall"),
            ],
            records=records,
            total_questions=2,
        )

        assert records["main_direct_only"]["correct_idx"] == [1]
        assert report["baseline"]["prefix"] == "main_system"
    finally:
        if path.exists():
            path.unlink()


def test_ablation_detail_path_defaults_to_results_dir():
    module_path = Path(__file__).with_name("evaluate_llm_full_schema_vs_system_plan.py")
    spec = importlib.util.spec_from_file_location("eval_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    assert module._default_ablation_detail_path(Path("tests/results")) == Path(
        "tests/results/ablation_outputs.jsonl"
    )


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
