"""
Baseline 系统 —— 单次直连 LLM 的最朴素 Text-to-SQL 流程。
─────────────────────────────────────────────────────────────────
- 模型：通过环境变量 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 配置
        （与主流程完全一致，复用 ``pipeline.llm_client``）
- Schema 模式：
    * mode="full"   ：直接把 ``data/m_schema.txt`` 全量 Schema 喂给 LLM
    * mode="pruned" ：复用主流程中得到的精简 Schema
                       （SchemaLinker 三路混合检索 + EntityExtractor + Profiler 注入）
- 流程：单次 LLM 调用 → ``SQLGenerator.extract_sql`` 提取 SQL
        → ``DBEngine.execute_sql`` 执行 → 返回结果
- 不含三路并发 / Refiner 修复 / Selector 投票 / Literal 校验，
  作为对比实验的"干净对照组"。

返回结构与 ``TextToSQLSystem.run_pipeline_async`` 对齐，
可直接被 ``training.evaluate.run_evaluation`` 评估。
"""
from __future__ import annotations

import asyncio
import time

from openai import APIConnectionError

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.schema_linker import SchemaLinker
from pipeline.utils import TokenTracker, debug_print, to_halfwidth


class BaselineSystem:
    """
    最朴素的 Text-to-SQL 基线。

    Args:
        mode: ``"full"``  → 使用 ``data/m_schema.txt`` 全量 Schema；
              ``"pruned"`` → 使用主流程导出的精简 Schema（SchemaLinker + Profiler）。
    """

    VALID_MODES = ("full", "pruned")

    def __init__(self, mode: str = "full"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode 必须是 {self.VALID_MODES} 之一，收到: {mode}")
        self.mode = mode

        debug_print(f">>> [Baseline][{mode}] 正在初始化...")

        csv_path = str(settings.csv_path)
        schema_path = str(settings.schema_path)

        self.client = create_async_client()
        self.llm_model = get_model_name()
        self.db_engine = DBEngine(csv_path, settings.table_name)

        if mode == "full":
            with open(schema_path, "r", encoding="utf-8") as f:
                self._full_schema_text = f.read()
            self.linker = None
            self.entity_extractor = None
            self._profile_map: dict[str, str] = {}
        else:
            self.linker = SchemaLinker(schema_path, csv_path)
            self.entity_extractor = EntityExtractor(self.client, self.llm_model)
            profiler = DatabaseProfiler(csv_path=csv_path)
            self._profile_map = profiler.get_profile_map(profiler.profile_all())
            self._full_schema_text = ""

        debug_print(f">>> [Baseline][{mode}] 初始化完成。\n")

    # ──────────── Schema 构建 ────────────

    async def _build_schema_prompt(
        self, question: str, tracker: TokenTracker
    ) -> str:
        """根据当前模式构建喂给 LLM 的 Schema 文本。"""
        if self.mode == "full":
            return self._full_schema_text

        # mode == "pruned"：复用主流程中得到的精简 Schema 逻辑
        # 等价于 pipeline.system._try_flow 的 tier1 (Top-K + must_have) 输入
        K = settings.top_k_embed
        pre_ranked, _, _ = self.linker.hybrid_retrieve(question, [], top_k_embed=K)
        pre_top_cols = [x[0] for x in pre_ranked[:K]]
        entity_schema = self.linker.build_entity_schema(pre_top_cols)

        entities = await self.entity_extractor.extract(
            question, entity_schema, tracker,
            schema_columns=self.linker.column_names,
        )
        ranked, must_have, _ = self.linker.hybrid_retrieve(
            question, entities, top_k_embed=K
        )
        full_list = [x[0] for x in ranked]
        cols = list(set(full_list[:K]) | must_have)

        return SQLGenerator.build_m_schema_prompt(
            cols,
            self.linker.column_metadata,
            table_name=settings.table_name,
            randomize=False,
            profile_map=self._profile_map,
        )

    # ──────────── 主入口 ────────────

    async def run_pipeline_async(self, question: str) -> dict:
        start = time.time()
        tracker = TokenTracker()
        question = to_halfwidth(question)

        schema_prompt = await self._build_schema_prompt(question, tracker)

        prompt = (
            f"你是一名SQL专家。请根据Schema为下列问题生成一条 SQLite SQL 查询。\n"
            f"采用 sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[用户问题]\n{question}\n"
            f"请直接输出SQL，用```sql ... ```包裹，不需要思考过程、解释或其他内容。\n"
        )

        gen_start = time.time()
        sql = ""
        reason = "success"
        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.direct_temperature,
                max_tokens=settings.max_gen_tokens,
                timeout=settings.llm_request_timeout_sec,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            debug_print(f"[Baseline][raw] {content!r}")
            sql = SQLGenerator.extract_sql(content)
        except APIConnectionError as e:
            base_url = str(getattr(self.client, "base_url", "") or "?")
            print(
                f"[Baseline][FATAL] 无法连接 LLM 服务 "
                f"(base_url={base_url}, model={self.llm_model}): {e}"
            )
            reason = f"llm_connection_error: {e}"
        except Exception as e:
            debug_print(f"[Baseline] LLM 调用失败: {type(e).__name__}: {e}")
            reason = f"llm_error: {type(e).__name__}: {e}"

        first_inference_time = time.time() - gen_start

        result, error = (None, "EMPTY_SQL") if not sql else self.db_engine.execute_sql(sql)
        if error is not None:
            reason = error

        unique_rows: list = []
        unique_count = 0
        if result:
            unique_set = set(tuple(row) for row in result)
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
            "final_sql": sql or None,
            "execution_result": unique_rows if result else result,
            "unique_rows_count": unique_count,
            "reason": reason,
            "first_inference_time": first_inference_time,
            "repair_times": [],
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
        }

    async def run_pipeline(self, question: str) -> dict:
        """别名，使其与 ``TextToSQLSystem.run_pipeline`` 接口对齐，供 evaluate 复用。"""
        return await self.run_pipeline_async(question)


# ──────────── CLI 入口 ────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Text-to-SQL 单条体验")
    parser.add_argument(
        "--mode", choices=BaselineSystem.VALID_MODES, default="full",
        help="full=全量Schema, pruned=主流程精简Schema",
    )
    parser.add_argument(
        "--question", type=str,
        default='"协议库存可视化选购20230407"批次的采购实施模式是怎样的？',
    )
    args = parser.parse_args()

    system = BaselineSystem(mode=args.mode)
    loop = asyncio.get_event_loop()
    print("\n" + "=" * 60)
    res = loop.run_until_complete(system.run_pipeline_async(args.question))
    print(f"[Baseline][{args.mode}] Time: {res['cost_time']:.2f}s")
    print(f"SQL: {res['final_sql']}")
    tokens = res.get("token_usage", {})
    print(
        f"Token Usage: Input={tokens.get('input_tokens', 0)} | "
        f"Output={tokens.get('output_tokens', 0)} | "
        f"Total={tokens.get('total_tokens', 0)}"
    )
    print(f"Unique Rows: {res['unique_rows_count']}")
    rows = res["execution_result"]
    print(f"Result (First 10): {rows[:10] if rows else 'Empty'}")
    print(f"Reason: {res['reason']}")
    print("=" * 60)
