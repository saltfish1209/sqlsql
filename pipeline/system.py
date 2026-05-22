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
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.question_splitter import QuestionSplitter
from pipeline.refiner import SQLRefiner
from pipeline.selector import SQLSelector
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
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.splitter = QuestionSplitter(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.selector = SQLSelector()
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
        norm_question = self._normalize_question(question)

        candidate_pack = self.linker.retrieve(norm_question, [])
        default_candidates = candidate_pack.Top20候选 or []
        entities = await self.entity_extractor.extract(
            norm_question,
            {"召回schema": default_candidates},
            tracker,
            schema_columns=self.linker.column_names,
        )
        candidate_pack = self.linker.retrieve(norm_question, entities)
        final_schema = self._assemble_final_schema(candidate_pack)

        plan_json = self._build_plan_json(question, candidate_pack, entities, final_schema)
        schema_prompt = self.generator.build_m_schema_prompt(
            final_schema,
            self.linker.column_metadata,
            table_name=settings.table_name,
            profile_detail_map=self.linker.profile_detail_map,
        )

        cand = await self.generator.generate_from_plan_async(
            question, schema_prompt, plan_json, tracker
        )
        if cand is None:
            return {
                "final_sql": None,
                "execution_result": None,
                "reason": "generation_failed",
                "cost_time": time.time() - start,
                "token_usage": tracker.get_report(),
                "entities": entities,
                "candidate_schema_pack": final_schema,
                "plan_json": plan_json,
                "is_multi_question": False,
            }

        cand["confidence"] = float(candidate_pack.Top20候选[0]["相关性分数"]) if candidate_pack.Top20候选 else 0.0
        refined = await self.refiner.refine_async(
            schema_prompt,
            [cand],
            candidate_pack.Top20候选,
            tracker,
            plan_json=plan_json,
        )
        selected, reason, status = self.selector.select_best(question, schema_prompt, refined)
        if selected is None:
            return {
                "final_sql": None,
                "execution_result": None,
                "reason": reason,
                "cost_time": time.time() - start,
                "token_usage": tracker.get_report(),
                "entities": entities,
                "candidate_schema_pack": final_schema,
                "plan_json": plan_json,
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
            "候选字段包": final_schema,
            "sql_generation_spec": plan_json,
            "is_multi_question": False,
        }

    async def run_pipeline(self, question: str) -> dict:
        return await self.run_pipeline_async(question)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question).strip()

    def _build_plan_json(self, question: str, candidate_pack, entities: list[str], final_schema: list[dict]) -> dict:
        return {
            "用户问题": question,
            "证据实体": entities,
            "初始top20": candidate_pack.Top20候选,
            "must_have": candidate_pack.必须列集合,
            "证据详情": candidate_pack.证据详情,
            "精简schema": candidate_pack.精简schema,
            "最终schema": final_schema,
        }

    async def _maybe_split_question(self, question: str, tracker: TokenTracker | None = None) -> list[str]:
        if not getattr(settings, "enable_question_split", True):
            return [question]
        return await self.splitter.split(question, tracker=tracker)

    def _assemble_final_schema(self, candidate_pack) -> list[dict]:
        top20 = list(candidate_pack.Top20候选 or [])
        compact = list(candidate_pack.精简schema or [])
        must_have = [str(c).strip() for c in (candidate_pack.必须列集合 or []) if str(c).strip()]
        evidence = candidate_pack.证据详情 or {}

        final_map: dict[str, dict] = {}
        for item in top20 + compact:
            col = str(item.get("列名") or "").strip()
            if col:
                final_map[col] = dict(item)

        for col in must_have:
            if col not in final_map:
                final_map[col] = {"列名": col}

        mentioned_cols: list[str] = []
        for match_type, entity_map in evidence.items():
            for _, hits in (entity_map or {}).items():
                for hit in hits or []:
                    col = str(hit.get("所在匹配列") or "").strip()
                    if col:
                        mentioned_cols.append(col)
        for col in dict.fromkeys(mentioned_cols):
            if col not in final_map and col in self.linker.column_names:
                meta = next((m for m in self.linker.column_metadata if m.get("column_name") == col), {})
                prof = self.linker.profile_detail_map.get(col, {})
                final_map[col] = {
                    "列名": col,
                    "列描述": meta.get("column_description", ""),
                    "字段类型": prof.get("字段类型", meta.get("data_type", "")),
                    "是否枚举": prof.get("是否枚举", "否"),
                    "空值率": prof.get("空值率", ""),
                    "唯一值数": prof.get("唯一值数", ""),
                    "示例值": prof.get("示例值", []),
                    "格式": prof.get("格式", ""),
                    "范围": prof.get("范围", ""),
                }
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
    print("[DEBUG] sample_question:")
    print(sample_question)
    print("[DEBUG] sample_answer:")
    print(sample_answer)
    print("=" * 80)

    async def _debug_run():
        start = time.time()
        tracker = TokenTracker()
        q = system._normalize_question(sample_question)

        print("\n[Layer 0] 归一化问题")
        print(q)

        pack0 = system.linker.retrieve(q, [])
        print("\n[Layer 1] 初次召回schema")
        print(_pretty(pack0.Top20候选))

        entities = await system.entity_extractor.extract(
            q,
            {"召回schema": pack0.Top20候选},
            tracker,
            schema_columns=system.linker.column_names,
        )
        print("\n[Layer 2] 证据实体")
        print(_pretty(entities))

        pack = system.linker.retrieve(q, entities)
        print("\n[Layer 3] 实体对齐结果")
        print(_pretty(pack.给LLM的中文结构.get("实体对齐结果", [])))
        print("\n[Layer 3b] 对齐证据详情(精确/模糊/向量)")
        print(_pretty(pack.证据详情))
        print("\n[Layer 3c] 最终精简schema")
        print(_pretty(pack.精简schema))

        plan_json = system._build_plan_json(sample_question, pack, entities)
        print("\n[Layer 4] plan_json")
        print(_pretty(plan_json))

        schema_prompt = system.generator.build_m_schema_prompt(
            pack.精简schema,
            system.linker.column_metadata,
            table_name=settings.table_name,
            profile_detail_map=system.linker.profile_detail_map,
        )
        print("\n[Layer 5] schema_prompt")
        print(schema_prompt)

        cand = await system.generator.generate_from_plan_async(sample_question, schema_prompt, plan_json, tracker)
        print("\n[Layer 6] 生成候选SQL")
        print(_pretty(cand))

        refined = await system.refiner.refine_async(
            schema_prompt,
            [cand] if cand else [],
            pack.精简schema,
            tracker,
            plan_json=plan_json,
        ) if cand else []
        print("\n[Layer 7] Refiner输出")
        print(_pretty(refined))

        selected, reason, status = system.selector.select_best(sample_question, schema_prompt, refined)
        print("\n[Layer 8] Selector")
        print(_pretty({"selected": selected, "reason": reason, "status": status}))

        output = await system.run_pipeline_async(sample_question)
        print("\n[Pipeline Output]")
        print(_pretty(output))

        is_correct = False
        answer_hint = sample_answer.strip()
        if answer_hint:
            final_text = _pretty(output.get("execution_result"))
            is_correct = answer_hint in final_text

        print("\n[Verdict]")
        print(_pretty({
            "是否正确": is_correct,
            "标准结果": sample_answer,
            "预测SQL": output.get("final_sql"),
            "执行结果": output.get("execution_result"),
            "reason": output.get("reason"),
        }))
        print("\n[Total elapsed]")
        print(f"{time.time() - start:.2f}s")

    asyncio.run(_debug_run())
