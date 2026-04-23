"""
主编排器 —— 串联 Schema Linking → Entity Extraction → SQL Generation
→ Refinement → Selection 的完整 Text-to-SQL 推理流水线。

项目创新:
  1. A路预排序 → 精简 Schema 实体提取（避免 Examples 值污染）
  2. 三路混合召回 (CrossEncoder + ExactMatch + LSH)
  3. 扩展窗口梯队回退（保留高相关列）
  4. 自动 Profiling 注入 Prompt（论文新增）
  5. Literal-Column 校验（论文新增）
  6. Schema 字段随机化增强多样性（论文新增）
"""
from __future__ import annotations

import asyncio
import time

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.refiner import SQLRefiner
from pipeline.schema_linker import SchemaLinker
from pipeline.selector import SQLSelector
from pipeline.utils import to_halfwidth, debug_print, TokenTracker


class TextToSQLSystem:
    """
    完整 Text-to-SQL 系统。

    初始化时加载数据库、构建索引、运行 Profiler。
    调用 run_pipeline_async(question) 返回推理结果字典。
    """

    def __init__(self):
        debug_print(">>> [System Init] 正在初始化 Text-to-SQL 各组件...")

        csv_path = str(settings.csv_path)
        schema_path = str(settings.schema_path)

        self.client = create_async_client()
        self.llm_model = get_model_name()

        self.db_engine = DBEngine(csv_path, settings.table_name)
        self.linker = SchemaLinker(schema_path, csv_path)
        self.entity_extractor = EntityExtractor(self.client, self.llm_model)
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.selector = SQLSelector()

        # 自动 Profiling（论文新增）
        self.profiler = DatabaseProfiler(csv_path=csv_path)
        self._profiles = self.profiler.profile_all()
        self._profile_text = self.profiler.generate_profile_text(self._profiles)

        debug_print(">>> [System Init] 初始化完成。\n")

    # ──────────── 主入口 ────────────

    async def run_pipeline_async(self, question: str) -> dict:
        start = time.time()
        tracker = TokenTracker()
        question = to_halfwidth(question)
        K = settings.top_k_embed
        first_inference_time = 0.0
        selected_repair_times: list[float] = []

        # Step 1: A路预排序（无实体），构建精简 Schema 用于实体提取
        debug_print("[Pipeline] A路预排序，构建精简 Schema...")
        pre_ranked, _, _ = self.linker.hybrid_retrieve(question, [], top_k_embed=K)
        pre_top_cols = [x[0] for x in pre_ranked[:K]]
        entity_schema = self.linker.build_entity_schema(pre_top_cols)

        # Step 2: LLM + 规则联合实体提取
        entity_start = time.time()
        debug_print("[Pipeline] Step2 实体提取开始...")
        entities = await self.entity_extractor.extract(
            question, entity_schema, tracker,
            schema_columns=self.linker.column_names,
        )
        debug_print(
            f"[Pipeline] Step2 实体提取完成，耗时 {time.time() - entity_start:.2f}s"
        )

        # Step 3: 带实体的完整混合召回
        ranked, must_have, evidence = self.linker.hybrid_retrieve(
            question, entities, top_k_embed=K
        )
        full_list = [x[0] for x in ranked]

        # 梯队 1: Top-K + 必须命中列
        tier1 = list(set(full_list[:K]) | must_have)
        debug_print(f"[Pipeline] 梯队1(Top {K}): {tier1}")
        best = await self._try_flow(question, tier1, entities, evidence, tracker)
        if best:
            first_inference_time = best.get("first_inference_time", 0.0)
            selected_repair_times = best.get("repair_times", []) or []

        # 梯队 2: 扩展至 Top-2K（保留梯队1 + 新增列）
        if not best or best.get("status") != "success":
            tier2 = list(set(full_list[:K * 2]) | must_have)
            debug_print(f"[Pipeline] 梯队2 扩展至 Top {K*2}")
            best = await self._try_flow(question, tier2, entities, evidence, tracker)
            if best:
                selected_repair_times = best.get("repair_times", []) or []

        # 梯队 3: 全新列段 Top-2K+1 ~ 3K
        if not best or best.get("status") != "success":
            tier3 = list(set(full_list[K * 2:K * 3]) | must_have)
            debug_print(f"[Pipeline] 梯队3 兜底列")
            best = await self._try_flow(question, tier3, entities, evidence, tracker)
            if best:
                selected_repair_times = best.get("repair_times", []) or []

        total_time = time.time() - start

        if best is None:
            return {
                "final_sql": None,
                "execution_result": None,
                "unique_rows_count": 0,
                "reason": "all_paths_failed",
                "first_inference_time": first_inference_time,
                "repair_times": selected_repair_times,
                "cost_time": total_time,
                "token_usage": tracker.get_report(),
            }

        final_res = best.get("result")
        unique_rows = []
        unique_count = 0
        if final_res:
            unique_set = set(tuple(row) for row in final_res)
            unique_count = len(unique_set)
            unique_rows = list(unique_set)

        return {
            "final_sql": best.get("sql", "SELECT 1"),
            "execution_result": unique_rows if final_res else final_res,
            "unique_rows_count": unique_count,
            "reason": best.get("status", "success"),
            "first_inference_time": first_inference_time,
            "repair_times": selected_repair_times,
            "cost_time": total_time,
            "token_usage": tracker.get_report(),
        }

    async def run_pipeline(self, question: str) -> dict:
        """别名，供 evaluate 脚本调用。"""
        return await self.run_pipeline_async(question)

    # ──────────── 单梯队尝试 ────────────

    async def _try_flow(
        self,
        question: str,
        cols: list[str],
        entities: list[str],
        evidence: dict,
        tracker: TokenTracker,
        randomize_schema: bool = False,
    ) -> dict | None:
        m_schema = self.generator.build_m_schema_prompt(
            cols, self.linker.column_metadata,
            randomize=randomize_schema,
            profile_text=self._profile_text,
        )

        gen_start = time.time()
        raw_cands = await self.generator.generate_candidates_async(
            question, m_schema, entities, evidence, tracker
        )
        first_inference_time = time.time() - gen_start
        debug_print(f"[Pipeline] Generator 原始候选 ({len(raw_cands)}个)")

        refined = await self.refiner.refine_async(
            question, m_schema, raw_cands, cols, tracker
        )
        debug_print(f"[Pipeline] Refiner 修正后 ({len(refined)}个)")

        selected, reason, status = self.selector.select_best(
            question, m_schema, refined
        )
        if selected is None:
            debug_print(f"[Pipeline] 选择结果: {reason}")
            return None
        if status == "tie":
            debug_print("[Pipeline] 触发平票，按路径优先级决策")
            resolved = SQLRefiner.resolve_tie(selected)
            resolved["first_inference_time"] = first_inference_time
            resolved["repair_times"] = resolved.get("repair_times", []) or []
            return resolved

        debug_print(f"[Pipeline] 选择结果: {reason}")
        selected["first_inference_time"] = first_inference_time
        selected["repair_times"] = selected.get("repair_times", []) or []
        return selected


# ──────────── CLI 入口 ────────────

if __name__ == "__main__":
    system = TextToSQLSystem()

    test_questions = [
        "\"协议库存可视化选购20230407\"批次的采购实施模式是怎样的？",
    ]

    loop = asyncio.get_event_loop()
    for q in test_questions:
        print("\n" + "=" * 60)
        result = loop.run_until_complete(system.run_pipeline_async(q))
        print(f"FINAL OUTPUT (Time: {result['cost_time']:.2f}s)")
        print(f"SQL: {result['final_sql']}")
        tokens = result.get("token_usage", {})
        print(
            f"Token Usage: Input={tokens.get('input_tokens', 0)} | "
            f"Output={tokens.get('output_tokens', 0)} | "
            f"Total={tokens.get('total_tokens', 0)}"
        )
        print(f"Unique Rows: {result['unique_rows_count']}")
        res = result["execution_result"]
        print(f"Result (First 10): {res[:10] if res else 'Empty'}")
        print("=" * 60)
