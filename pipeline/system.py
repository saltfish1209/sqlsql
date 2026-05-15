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
        self.entity_extractor = EntityExtractor(self.client, self.llm_model)
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.selector = SQLSelector()
        self.profiler = DatabaseProfiler(csv_path=csv_path)
        debug_print(">>> [System Init] 完成.\n")

    async def run_pipeline_async(self, question: str) -> dict:
        start = time.time()
        question = to_halfwidth(question)
        sub_questions = await self._maybe_split_question(question)
        if len(sub_questions) > 1:
            outputs = [await self._run_single_pipeline(q) for q in sub_questions]
            return self._merge_sub_outputs(question, sub_questions, outputs, start)
        return await self._run_single_pipeline(question)

    async def _run_single_pipeline(self, question: str) -> dict:
        start = time.time()
        tracker = TokenTracker()
        norm_question = self._normalize_question(question)

        candidate_pack = self.linker.retrieve(norm_question, [])
        entities = await self.entity_extractor.extract(
            norm_question,
            {"召回schema": candidate_pack.Top20候选},
            tracker,
            schema_columns=self.linker.column_names,
        )
        candidate_pack = self.linker.retrieve(norm_question, entities)

        plan_json = self._build_plan_json(question, candidate_pack, entities)
        schema_prompt = self.generator.build_m_schema_prompt(
            candidate_pack.精简schema,
            self.linker.column_metadata,
            table_name=settings.table_name,
            profile_map=self.linker.profile_map,
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
                "candidate_schema_pack": candidate_pack.Top20候选,
                "plan_json": plan_json,
                "is_multi_question": False,
            }

        cand["confidence"] = float(candidate_pack.Top20候选[0]["相关性分数"]) if candidate_pack.Top20候选 else 0.0
        refined = await self.refiner.refine_async(
            question,
            schema_prompt,
            [cand],
            candidate_pack.精简schema,
            tracker,
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
                "candidate_schema_pack": candidate_pack.Top20候选,
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
            "候选字段包": candidate_pack.Top20候选,
            "结构化计划": plan_json,
            "is_multi_question": False,
        }

    async def run_pipeline(self, question: str) -> dict:
        return await self.run_pipeline_async(question)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question).strip()

    def _build_plan_json(self, question: str, candidate_pack, entities: list[str]) -> dict:
        evidence = []
        alternatives = candidate_pack.Top20候选[1:4] if len(candidate_pack.Top20候选) > 1 else []
        fallback_field = candidate_pack.Top20候选[0]["列名"] if candidate_pack.Top20候选 else None
        for ent in entities:
            evidence.append({"文本": ent, "匹配字段": fallback_field, "角色猜测": "未知"})

        confidence = 0.0
        if candidate_pack.Top20候选:
            confidence = min(1.0, candidate_pack.Top20候选[0]["相关性分数"])

        return {
            "用户问题": question,
            "候选字段": candidate_pack.Top20候选,
            "选中字段": [c["列名"] for c in candidate_pack.Top20候选[: settings.evidence_schema_top_k]],
            "证据实体": entities,
            "证据": evidence,
            "备选字段": alternatives,
            "不确定匹配": [],
            "置信度": confidence,
            "精简schema": candidate_pack.精简schema,
        }

    async def _maybe_split_question(self, question: str) -> list[str]:
        if not getattr(settings, "enable_question_split", True):
            return [question]
        qmark_count = question.count("？") + question.count("?")
        if qmark_count < 2:
            return [question]
        return [question]

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
            "结构化计划": {"子问题": sub_questions, "原始问题": original_question},
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
        norm_question = system._normalize_question(sample_question)

        print("\n[Layer 0] norm_question")
        print(norm_question)

        candidate_pack_0 = system.linker.retrieve(norm_question, [])
        print("\n[Layer 1] candidate_pack（初次召回）")
        print(_pretty(system.linker.build_llm_prompt_payload(candidate_pack_0)))
        print("\n[Layer 1 full rank] 全量排序前5")
        print(_pretty(candidate_pack_0.全量排序[:5]))
        print("\n[Index Status]")
        print(_pretty({
            "exact_index_size": len(system.linker.exact_index),
            "lsh_index_size": len(system.linker.lsh_index),
            "semantic_value_records": len(system.linker.semantic_value_records),
            "index_cache_dir": str(system.linker.cache_dir),
        }))

        entities = await system.entity_extractor.extract(
            norm_question,
            {"召回schema": candidate_pack_0.Top20候选},
            tracker,
            schema_columns=system.linker.column_names,
        )
        print("\n[Layer 2] 提取实体")
        print(_pretty(entities))

        candidate_pack = system.linker.retrieve(norm_question, entities)
        print("\n[Layer 1b] candidate_pack（实体反哺后）")
        print(_pretty(system.linker.build_llm_prompt_payload(candidate_pack)))

        plan_json = system._build_plan_json(sample_question, candidate_pack, entities)
        print("\n[Layer 3] plan_json")
        print(_pretty(plan_json))

        schema_prompt = system.generator.build_m_schema_prompt(
            candidate_pack.精简schema,
            system.linker.column_metadata,
            table_name=settings.table_name,
            profile_map=system.linker.profile_map,
        )
        print("\n[Layer 4 prep] schema_prompt")
        print(schema_prompt)

        cand = await system.generator.generate_from_plan_async(
            sample_question, schema_prompt, plan_json, tracker
        )
        print("\n[Layer 4] generator raw output")
        print(_pretty(cand))

        if cand is None:
            print("\n[Layer 5] generation failed, stop here.")
            return

        cand["confidence"] = plan_json.get("置信度", 0.0)
        print("\n[Layer 4b] candidate before refiner")
        print(_pretty(cand))

        refined = await system.refiner.refine_async(
            sample_question,
            schema_prompt,
            [cand],
            candidate_pack.精简schema,
            tracker,
        )
        print("\n[Layer 5] refiner outputs")
        print(_pretty(refined))

        selected, reason, status = system.selector.select_best(
            sample_question, schema_prompt, refined
        )
        print("\n[Layer 6] selector result")
        print(_pretty({"selected": selected, "reason": reason, "status": status}))

        final_sql = None
        execution_result = None
        execution_error = None
        if selected is not None and selected.get("sql"):
            final_sql = selected["sql"]
            execution_result, execution_error = system.db_engine.execute_sql(final_sql)
            print("\n[Execution] sql")
            print(final_sql)
            print("\n[Execution] result")
            print(_pretty(execution_result))
            print("\n[Execution] error")
            print(execution_error)

        is_correct = False
        answer_hint = sample_answer.strip()
        if answer_hint and final_sql is not None:
            final_text = _pretty(execution_result)
            is_correct = answer_hint in final_text

        print("\n[Verdict]")
        print(_pretty({
            "是否正确": is_correct,
            "标准结果": sample_answer,
            "预测SQL": final_sql,
            "执行结果": execution_result,
            "执行错误": execution_error,
            "reason": reason,
            "status": status,
        }))

        print("\n[Token Usage]")
        print(_pretty(tracker.get_report()))
        print("\n[Total elapsed]")
        print(f"{time.time() - start:.2f}s")

    asyncio.run(_debug_run())
