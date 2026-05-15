"""
Baseline 系统 —— 单次直连 LLM 的最朴素 Text-to-SQL 流程。
─────────────────────────────────────────────────────────────────
- 模型：通过环境变量 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 配置
        （与主流程完全一致，复用 ``pipeline.llm_client``）
- Schema 模式：
    * mode="full"   ：直接把 ``data/m_schema.txt`` 全量 Schema 喂给 LLM
    * mode="pruned" ：只使用 ``EvidenceLinker`` 召回的精简候选 Schema
                       （不注入证据实体，由消融实验决定是否启用）
- 流程：单次 LLM 调用 → ``SQLGenerator.extract_sql`` 提取 SQL
        → ``DBEngine.execute_sql`` 执行 → 返回结果
- 不含三路并发 / Refiner 修复 / Selector 投票 / Literal 校验，
  作为对比实验的"干净对照组"。

返回结构与 ``TextToSQLSystem.run_pipeline_async`` 对齐，
可直接被 ``training.evaluate.run_evaluation`` 评估。
"""
from __future__ import annotations

import asyncio
import re
import time

from openai import APIConnectionError

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.evidence_linker import EvidenceLinker
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.utils import TokenTracker, debug_print, to_halfwidth

_SQL_SINGLE_QUOTE_LITERAL_RE = re.compile(r"'([^']*)'")
_CJK_RE = r"\u4e00-\u9fff"
_ASCII_WORD_RE = r"A-Za-z0-9"


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

        self.linker = EvidenceLinker(schema_path, csv_path)
        profiler = DatabaseProfiler(csv_path=csv_path)
        self._profile_map = profiler.get_profile_map(profiler.profile_all())
        self._all_columns = list(self.linker.column_names)
        if mode == "full":
            self.entity_extractor = None
        else:
            self.entity_extractor = EntityExtractor(self.client, self.llm_model)

        debug_print(f">>> [Baseline][{mode}] 初始化完成。\n")

    # ──────────── Schema 构建 ────────────

    async def _build_schema_prompt(
        self, question: str, tracker: TokenTracker
    ) -> tuple[str, list[str]]:
        """根据当前模式构建喂给 LLM 的 Schema 文本，并返回提取实体。"""
        if self.mode == "full":
            # full 模式：沿用与 pruned 一致的 M-Schema 格式，
            # 仅将列集合替换为"全字段"以保持 baseline 的全量输入设定。
            return (
                SQLGenerator.build_m_schema_prompt(
                    self._all_columns,
                    self.linker.column_metadata,
                    table_name=settings.table_name,
                    randomize=False,
                    profile_map=self._profile_map,
                ),
                [],
            )

        # mode == "pruned"：只做候选 schema 精简，不注入证据实体
        candidate_pack = self.linker.retrieve(question, [])
        cols = candidate_pack.selected_columns or [
            c["field"] for c in candidate_pack.candidates[: settings.evidence_schema_top_k]
        ]

        return (
            SQLGenerator.build_m_schema_prompt(
                cols,
                self.linker.column_metadata,
                table_name=settings.table_name,
                randomize=False,
                profile_map=self._profile_map,
            ),
            [],
        )

    # ──────────── 主入口 ────────────

    @staticmethod
    def _normalize_sql_literals(sql: str) -> str:
        """轻量归一化生成 SQL：中文标点半角化 + 中英混排空格修正。"""
        if not sql:
            return sql
        # 先做整句半角化：中文标点/全角符号 -> 英文半角符号
        sql = to_halfwidth(sql)
        # 修正标识符里的中英混排空格：计划批次 ID -> 计划批次ID
        sql = re.sub(rf"(?<=[{_CJK_RE}])\s+(?=[{_ASCII_WORD_RE}])", "", sql)
        sql = re.sub(rf"(?<=[{_ASCII_WORD_RE}])\s+(?=[{_CJK_RE}])", "", sql)

        def _norm_match(m: re.Match[str]) -> str:
            lit = to_halfwidth(m.group(1))
            # ECP 招标合同 -> ECP招标合同, II 型 -> II型
            lit = re.sub(rf"(?<=[{_ASCII_WORD_RE}])\s+(?=[{_CJK_RE}])", "", lit)
            lit = re.sub(rf"(?<=[{_CJK_RE}])\s+(?=[{_ASCII_WORD_RE}])", "", lit)
            return f"'{lit}'"

        return _SQL_SINGLE_QUOTE_LITERAL_RE.sub(_norm_match, sql)

    async def run_pipeline_async(self, question: str, *, enable_thinking: bool = False) -> dict:
        start = time.time()
        tracker = TokenTracker()
        question = to_halfwidth(question)

        schema_prompt, entities = await self._build_schema_prompt(question, tracker)

        prompt = (
            f"你是一名SQL专家。请根据Schema为下列问题生成一条 SQLite SQL 查询。\n"
            f"采用 sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"以问题信息为生成SQL主要条件，Schema提供辅助。\n"
            f"不要添加除问题所给信息外多余的约束。\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[实体候选]\n{entities}\n"
            f"[用户问题]\n{question}\n"
            f"[硬性约束]\n"
            f"1. WHERE/LIKE 中字符串字面量必须与问题或实体候选逐字符一致，不要改写空格和标点。\n"
            f"2. SQL 关键字（SELECT/FROM/WHERE/AND/OR/JOIN/ON/DISTINCT/GROUP BY/ORDER BY/LIMIT 等）"
            f"与列名、表名之间**必须有空格**分隔，禁止写成 `SELECT列名FROM` 这种无空格形式。\n"
            f"3. 列名使用双引号包裹，如 `\"列名\"`。\n"
            f"请直接输出SQL，用```sql ... ```包裹，不需要解释或其他内容。\n"
        )

        gen_start = time.time()
        sql = ""
        reason = "success"
        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.direct_temperature,
                timeout=settings.llm_request_timeout_sec,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            debug_print(f"[Baseline][raw] {content!r}")
            sql = self._normalize_sql_literals(SQLGenerator.extract_sql(content))
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
            "entities": entities,
        }

    async def run_pipeline(self, question: str, *, enable_thinking: bool | None = None) -> dict:
        """别名，使其与 ``TextToSQLSystem.run_pipeline`` 接口对齐，供 evaluate 复用。"""
        if enable_thinking is None:
            enable_thinking = settings.baseline_enable_thinking
        return await self.run_pipeline_async(question, enable_thinking=enable_thinking)


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
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="启用 LLM 思考模式（enable_thinking）",
    )
    args = parser.parse_args()

    system = BaselineSystem(mode=args.mode)
    loop = asyncio.get_event_loop()
    print("\n" + "=" * 60)
    res = loop.run_until_complete(
        system.run_pipeline_async(args.question, enable_thinking=args.enable_thinking)
    )
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
