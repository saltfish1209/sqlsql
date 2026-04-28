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
from pipeline.refiner import SQLRefiner, majority_agreed
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
        # profile_map：{列名: 内联摘要}，只对当前 selected_columns 中的列注入，
        # 拼接在各字段描述尾部，不做全量底部拼接。
        self.profiler = DatabaseProfiler(csv_path=csv_path)
        self._profiles = self.profiler.profile_all()
        self._profile_map = self.profiler.get_profile_map(self._profiles)

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
            # 过滤全 NULL / 全空串行，保持与 evaluate 归一化逻辑一致
            unique_set = {
                row for row in unique_set
                if any(
                    v is not None and str(v).strip() != ""
                    for v in row
                )
            }
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
        """
        单梯队三路生成 + 快路早停。

        编排逻辑：
          1) 同时启动 thinking / icl / direct 三路（thinking 路较慢）。
          2) **先等** ICL + Direct 两条快路；refine 后若结果一致 (majority_agreed)，
             立即取消 thinking 路返回 —— 用于减少单条问题的尾延迟。
          3) 否则等 thinking 路返回并加入 refine + 投票。
        """
        m_schema = self.generator.build_m_schema_prompt(
            cols, self.linker.column_metadata,
            randomize=randomize_schema,
            profile_map=self._profile_map,
        )

        gen_start = time.time()
        task_map = self.generator.start_candidate_tasks(
            question, m_schema, entities, evidence, tracker
        )
        fast_tasks = task_map.get("icl", []) + task_map.get("direct", [])
        slow_tasks = task_map.get("thinking", [])

        # —— 阶段一：等两条快路 ——
        fast_raw = (
            await asyncio.gather(*fast_tasks, return_exceptions=True)
            if fast_tasks else []
        )
        fast_cands = [r for r in fast_raw if isinstance(r, dict)]
        first_inference_time = time.time() - gen_start
        debug_print(f"[Pipeline] 快路完成 ({len(fast_cands)}个候选)")

        fast_refined = await self.refiner.refine_async(
            question, m_schema, fast_cands, cols, tracker
        )

        # 早停判定：两条快路投票一致即可
        fast_success = [c for c in fast_refined if c.get("status") == "success"]
        if majority_agreed(fast_success) and slow_tasks:
            debug_print(
                f"[Pipeline] 快路 {len(fast_success)} 路结果一致 → 取消 thinking 路"
            )
            for t in slow_tasks:
                t.cancel()
            await asyncio.gather(*slow_tasks, return_exceptions=True)
            all_refined = fast_refined
        elif slow_tasks:
            debug_print("[Pipeline] 快路未达成一致 → 等待 thinking 路兜底")
            slow_raw = await asyncio.gather(*slow_tasks, return_exceptions=True)
            slow_cands = [r for r in slow_raw if isinstance(r, dict)]
            slow_refined = await self.refiner.refine_async(
                question, m_schema, slow_cands, cols, tracker
            )
            all_refined = fast_refined + slow_refined
            # 把"等到所有路全部生成完"的耗时也视作首次推理时间
            first_inference_time = time.time() - gen_start
        else:
            all_refined = fast_refined

        debug_print(f"[Pipeline] Refiner 修正后 ({len(all_refined)}个)")

        selected, reason, status = self.selector.select_best(
            question, m_schema, all_refined
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
        "物料编码500116755的中标厂家有哪些？",
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
