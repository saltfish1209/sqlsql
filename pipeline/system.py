from __future__ import annotations

import asyncio
import json
import re
import time

from config.settings import settings
from generation.multi_result_utils import MULTI_RESULT_SEP
from pipeline.db_engine import DBEngine
from pipeline.schema_linker import SchemaLinker
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.intent_planner import IntentPlanner
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler, apply_instance_fields
from pipeline.question_splitter import QuestionSplitter
from pipeline.refiner import SQLRefiner
from pipeline.schema_format import build_light_schema_markdown, build_plan_markdown, enrich_schema_columns
from pipeline.consensus_vote import select_by_consensus
from pipeline.utils import TokenTracker, debug_print, to_halfwidth


class TextToSQLSystem:
    def __init__(self):
        debug_print(">>> [System Init] 初始化 retrieval-first Text-to-SQL...")
        csv_path = str(settings.csv_path)
        schema_path = str(settings.schema_json_path)
        self.client = create_async_client()
        self.llm_model = get_model_name()
        self.db_engine = DBEngine(csv_path, settings.table_name)
        self.linker = SchemaLinker(schema_path, csv_path)
        report = getattr(self.linker, "deprecated_columns_report", {}) or {}
        if report:
            deprecated = report.get("deprecated", {})
            debug_print(
                "[System Init] 废弃列过滤: "
                f"有效列={report.get('active_columns_count', 0)}, "
                f"废弃列={report.get('deprecated_columns_count', 0)}, "
                f"schema缺失={len(deprecated.get('schema_not_in_raw', []))}, "
                f"高空值={len(deprecated.get('high_null_ratio', []))}"
            )
        self.entity_extractor = EntityExtractor(self.client, self.llm_model)
        self.intent_planner = IntentPlanner(self.client, self.llm_model)
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.splitter = QuestionSplitter(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.profiler = DatabaseProfiler(csv_path=csv_path)
        debug_print(">>> [System Init] 完成.\n")

    async def run_pipeline_async(self, question: str) -> dict:
        start = time.time()
        question = to_halfwidth(question)
        tracker = TokenTracker()
        sub_questions = await self._maybe_split_question(question, tracker)
        if len(sub_questions) > 1:
            outputs = [await self._run_single_pipeline(q) for q in sub_questions]
            return self._merge_sub_outputs(question, sub_questions, outputs, start)
        return await self._run_single_pipeline(question)

    async def _run_single_pipeline(self, question: str) -> dict:
        start = time.time()
        tracker = TokenTracker()
        stages = await self._prepare_single_pipeline(question, tracker)
        question = stages["question"]
        candidate_pack = stages["candidate_pack"]
        entities = stages["entities"]
        plan_schema = stages["plan_schema"]
        final_schema = stages["final_schema"]
        schema_prompt = stages["schema_prompt"]
        repair_schema_prompt = stages["repair_schema_prompt"]
        intent_plan = stages.get("intent_plan") or {}

        candidates = await self.generator.generate_candidates_async(
            question,
            schema_prompt,
            tracker,
            intent_plan=intent_plan,
        )
        if not candidates:
            return {
                "final_sql": None,
                "execution_result": None,
                "reason": "generation_failed",
                "cost_time": time.time() - start,
                "token_usage": tracker.get_report(),
                "entities": entities,
                "candidate_schema_pack": plan_schema,
                "sql_generation_spec": schema_prompt,
                "intent_plan": intent_plan,
                "is_multi_question": False,
            }

        confidence = float(candidate_pack.Top20候选[0]["相关性分数"]) if candidate_pack.Top20候选 else 0.0
        for cand in candidates:
            cand["confidence"] = confidence
        cliff_schema_prompt = stages.get("cliff_schema_prompt") or schema_prompt
        refined = await self.refiner.refine_async(
            schema_prompt,
            candidates,
            candidate_pack.Top20候选,
            tracker,
            repair_schema_prompt=repair_schema_prompt,
            question=question,
            judge_schema_prompt=cliff_schema_prompt,
            intent_plan=intent_plan,
        )
        selected, reason, status = select_by_consensus(refined)
        if selected is None:
            return {
                "final_sql": None,
                "execution_result": None,
                "reason": reason,
                "cost_time": time.time() - start,
                "token_usage": tracker.get_report(),
                "entities": entities,
                "candidate_schema_pack": plan_schema,
                "sql_generation_spec": schema_prompt,
                "intent_plan": intent_plan,
                "candidate_sqls": [
                    {"type": c.get("type"), "variant_id": c.get("variant_id"), "sql": c.get("sql")}
                    for c in refined
                ],
                "is_multi_question": False,
            }

        result = selected.get("result")
        unique_rows: list[tuple] = []
        if result:
            seen_rows = set()
            for row in result:
                if row is None:
                    continue
                tup = tuple(row)
                if tup in seen_rows:
                    continue
                seen_rows.add(tup)
                unique_rows.append(tup)

        return {
            "final_sql": selected.get("sql"),
            "execution_result": unique_rows if result else result,
            "unique_rows_count": len(unique_rows),
            "reason": status,
            "first_inference_time": 0.0,
            "repair_times": [],
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "证据实体": entities,
            "候选字段包": plan_schema,
            "sql_generation_spec": schema_prompt,
            "intent_plan": intent_plan,
            "candidate_sqls": [
                {"type": c.get("type"), "variant_id": c.get("variant_id"), "sql": c.get("sql"), "status": c.get("status")}
                for c in refined
            ],
            "repair_schema": final_schema,
            "is_multi_question": False,
        }

    async def run_pipeline(self, question: str) -> dict:
        return await self.run_pipeline_async(question)

    async def _prepare_single_pipeline(self, question: str, tracker: TokenTracker | None = None) -> dict:
        tracker = tracker or TokenTracker()
        question = to_halfwidth(question)
        norm_question = self._normalize_question(question)

        initial_pack = self.linker.retrieve(norm_question, [])
        entities: list[str] = []
        if settings.enable_entity_extraction:
            default_candidates = initial_pack.Top20候选 or []
            entities = await self.entity_extractor.extract(
                norm_question,
                {"召回schema": default_candidates},
                tracker,
                schema_columns=self.linker.column_names,
            )
        candidate_pack = self.linker.retrieve(norm_question, entities)
        plan_schema = self._assemble_plan_schema(candidate_pack)
        final_schema = self._assemble_final_schema(candidate_pack)
        cliff_schema = list(candidate_pack.精简schema or [])
        cliff_schema_prompt = self._build_schema_markdown(cliff_schema)
        schema_prompt = self._build_plan_markdown(candidate_pack, plan_schema)
        repair_schema_prompt = self._build_plan_markdown(candidate_pack, final_schema)
        intent_plan = await self.intent_planner.plan_async(
            question,
            schema_prompt,
            entities=entities,
            tracker=tracker,
        )
        return {
            "question": question,
            "norm_question": norm_question,
            "initial_pack": initial_pack,
            "candidate_pack": candidate_pack,
            "entities": entities,
            "intent_plan": intent_plan,
            "plan_schema": plan_schema,
            "final_schema": final_schema,
            "cliff_schema": cliff_schema,
            "cliff_schema_prompt": cliff_schema_prompt,
            "schema_prompt": schema_prompt,
            "repair_schema_prompt": repair_schema_prompt,
        }

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question).strip()

    def _build_schema_markdown(self, schema_columns: list[dict]) -> str:
        enriched = enrich_schema_columns(
            schema_columns,
            self.linker.column_metadata,
            self.linker.profile_detail_map,
        )
        return build_light_schema_markdown(enriched, settings.table_name)

    def _build_plan_markdown(self, candidate_pack, schema_columns: list[dict]) -> str:
        """合并为单一 markdown：富 schema 表 + Must-have（无证据列）。"""
        return build_plan_markdown(
            self._build_schema_markdown(schema_columns),
            candidate_pack.必须列集合,
            candidate_pack.证据详情,
            include_evidence=True,
        )

    async def _maybe_split_question(self, question: str, tracker: TokenTracker | None = None) -> list[str]:
        if not getattr(settings, "enable_question_split", True):
            return [question]
        return await self.splitter.split(question, tracker=tracker)

    def _assemble_plan_schema(self, candidate_pack) -> list[dict]:
        """生成用：精简 schema + 问题中完整出现的列名（must_have）。"""
        compact = list(candidate_pack.精简schema or [])
        must_have = [str(c).strip() for c in (candidate_pack.必须列集合 or []) if str(c).strip()]
        ranked_lookup = {
            str(item.get("列名") or "").strip(): dict(item)
            for item in (candidate_pack.Top20候选 or []) + (candidate_pack.全量排序 or [])
            if str(item.get("列名") or "").strip()
        }

        plan_map: dict[str, dict] = {}
        for item in compact:
            col = str(item.get("列名") or "").strip()
            if col:
                plan_map[col] = dict(item)

        for col in must_have:
            if col not in plan_map:
                plan_map[col] = ranked_lookup.get(col) or self._column_schema_item(col)
        return list(plan_map.values())

    def _column_schema_item(self, col: str) -> dict:
        meta = next((m for m in self.linker.column_metadata if m.get("column_name") == col), {})
        prof = self.linker.profile_detail_map.get(col, {})
        item = {
            "列名": col,
            "列描述": meta.get("column_description", ""),
            "字段类型": prof.get("字段类型", meta.get("data_type", "")),
            "是否枚举": prof.get("是否枚举", "否"),
            "空值率": prof.get("空值率", ""),
            "唯一值数": prof.get("唯一值数", ""),
            "格式": prof.get("格式", ""),
            "范围": prof.get("范围", ""),
        }
        apply_instance_fields(item, prof)
        return item

    def _assemble_final_schema(self, candidate_pack) -> list[dict]:
        """执行失败修复专用：保留完整 Top20 及 must_have 补列。"""
        top20 = list(candidate_pack.Top20候选 or [])
        compact = list(candidate_pack.精简schema or [])
        must_have = [str(c).strip() for c in (candidate_pack.必须列集合 or []) if str(c).strip()]

        final_map: dict[str, dict] = {}
        for item in top20 + compact:
            col = str(item.get("列名") or "").strip()
            if col:
                final_map[col] = dict(item)

        ranked_lookup = {
            str(item.get("列名") or "").strip(): dict(item)
            for item in top20
            if str(item.get("列名") or "").strip()
        }
        for col in must_have:
            if col not in final_map:
                final_map[col] = ranked_lookup.get(col, {"列名": col})
        return list(final_map.values())

    @staticmethod
    def _merge_sub_outputs(original_question: str, sub_questions: list[str], sub_outputs: list[dict], start_ts: float) -> dict:
        sub_sqls = [str(o.get("final_sql") or "") for o in sub_outputs]
        sub_results = [list(o.get("execution_result") or []) for o in sub_outputs]
        return {
            "final_sql": MULTI_RESULT_SEP.join(sub_sqls),
            "execution_result": sub_results,
            "unique_rows_count": sum(len(r) for r in sub_results),
            "reason": "multi_success",
            "first_inference_time": max((float(o.get("first_inference_time", 0.0) or 0.0) for o in sub_outputs), default=0.0),
            "repair_times": [],
            "cost_time": time.time() - start_ts,
            "token_usage": {},
            "证据实体": [e for o in sub_outputs for e in (o.get("证据实体") or o.get("entities") or [])],
            "候选字段包": [c for o in sub_outputs for c in (o.get("候选字段包") or o.get("candidate_schema_pack") or [])],
            "sql_generation_spec": {"子问题": sub_questions, "原始问题": original_question},
            "is_multi_question": True,
            "子问题": sub_questions,
            "子问题输出": sub_outputs,
        }


def _pretty(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def _format_entity_match_detail(pack, entities: list) -> dict:
    alignment: list = []
    if hasattr(pack, "给LLM的中文结构") and isinstance(pack.给LLM的中文结构, dict):
        alignment = pack.给LLM的中文结构.get("实体对齐结果") or []
    return {
        "证据实体": entities,
        "实体对齐结果": alignment,
        "证据详情": getattr(pack, "证据详情", None) or {},
    }


def print_debug_pipeline_trace(
    *,
    initial_top20: list,
    candidate_pack,
    entities: list,
    compact_schema: list,
    plan_markdown: str,
) -> None:
    """调试模式精简输出：top20、实体匹配、精简 schema、合并 markdown。"""
    print("\n[初始top20]")
    print(_pretty(initial_top20))

    print("\n[实体提取匹配详情]")
    print(_pretty(_format_entity_match_detail(candidate_pack, entities)))

    print("\n[精简schema]")
    print(_pretty(compact_schema))

    print("\n[plan_markdown]")
    print(plan_markdown)


def print_debug_refiner_trace(refined_candidates: list[dict]) -> None:
    """refiner 因 checker 触发修复时，输出 check 与 refiner 日志。"""
    for cand in refined_candidates or []:
        refiner_debug = cand.get("refiner_debug")
        if not refiner_debug:
            continue
        print("\n[judge]")
        print(_pretty(refiner_debug.get("judge")))
        print("\n[refiner]")
        print(_pretty(refiner_debug.get("refiner")))
        return


def print_debug_generation_trace(
    all_candidates: list[dict],
    refined: list[dict],
    selected: dict | None,
    reason: str,
    output: dict,
) -> None:
    """输出每条路径的 SQL、refiner 日志、最终结果。"""
    print("\n[各路径生成SQL]")
    for cand in all_candidates:
        print(f"  [{cand.get('type', '?')}] {cand.get('sql', '')}")

    print_debug_refiner_trace(refined)

    print("\n[最终结果]")
    print(_pretty({
        "final_sql": output.get("final_sql"),
        "execution_result": output.get("execution_result"),
        "reason": output.get("reason"),
        "cost_time": output.get("cost_time"),
    }))


if __name__ == "__main__":
    import pandas as pd

    system = TextToSQLSystem()

    try:
        df = pd.read_csv(str(settings.train_csv))
        if {"生成问题", "生成结果"}.issubset(df.columns):
            row = df.iloc[0]
            sample_question = str(row["生成问题"]).strip()
            sample_answer = str(row["生成结果"]).strip()
        elif "生成问题" in df.columns:
            row = df.iloc[0]
            sample_question = str(row["生成问题"]).strip()
            sample_answer = ""
        else:
            sample_question = "物料编码500116755的中标厂家有哪些？"
            sample_answer = ""
    except Exception:
        sample_question = "物料编码500116755的中标厂家有哪些？"
        sample_answer = ""

    print("\n" + "=" * 80)
    print("[DEBUG] question:", sample_question)
    print("=" * 80)

    async def _debug_run():
        tracker = TokenTracker()
        stages = await system._prepare_single_pipeline(sample_question, tracker)
        pack0 = stages["initial_pack"]
        pack = stages["candidate_pack"]
        schema_prompt = stages["schema_prompt"]
        repair_schema_prompt = stages["repair_schema_prompt"]

        print_debug_pipeline_trace(
            initial_top20=pack0.Top20候选 or [],
            candidate_pack=pack,
            entities=stages["entities"],
            compact_schema=pack.精简schema or [],
            plan_markdown=schema_prompt,
        )

        all_candidates = await system.generator.generate_candidates_async(
            stages["question"], schema_prompt, tracker
        )
        refined: list[dict] = []
        if all_candidates:
            refined = await system.refiner.refine_async(
                schema_prompt,
                all_candidates,
                pack.Top20候选,
                tracker,
                repair_schema_prompt=repair_schema_prompt,
                question=stages["question"],
                judge_schema_prompt=stages.get("cliff_schema_prompt") or schema_prompt,
                intent_plan=stages.get("intent_plan"),
            )

        selected, reason, status = select_by_consensus(refined) if refined else (None, "no_candidates", "failed")

        output = await system.run_pipeline_async(sample_question)

        print_debug_generation_trace(
            all_candidates=all_candidates,
            refined=refined,
            selected=selected,
            reason=reason,
            output=output,
        )

    asyncio.run(_debug_run())
