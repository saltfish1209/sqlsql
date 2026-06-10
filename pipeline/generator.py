from __future__ import annotations

import asyncio
import re

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.fewshot_index import FewShotFaissStore
from pipeline.schema_format import build_light_schema_markdown, enrich_schema_columns
from pipeline.utils import TokenTracker, debug_print


class SQLGenerator:
    def __init__(self, client: AsyncOpenAI, model: str, embed_model_path: str | None = None):
        self.client = client
        self.model = model
        self.fewshot_store = FewShotFaissStore(
            embed_model_name_or_path=embed_model_path or settings.embed_model,
            index_dir=settings.fewshot_index_dir,
            csv_path=settings.qa_template_csv,
        )
        self._ensure_fewshot_index_ready()

    def _ensure_fewshot_index_ready(self) -> None:
        """启动自检：缺少 few-shot 索引时按配置自动构建。"""
        index_exists = self.fewshot_store.index_path.exists()
        meta_exists = self.fewshot_store.meta_path.exists()
        if index_exists and meta_exists:
            return

        if not settings.fewshot_autobuild_on_start:
            debug_print(
                "[Generator][fewshot] 索引缺失且已关闭自动构建，"
                f"请先执行离线构建: {self.fewshot_store.index_dir}"
            )
            return

        try:
            debug_print("[Generator][fewshot] 检测到索引缺失，开始自动构建...")
            self.fewshot_store.build_offline_index()
            debug_print("[Generator][fewshot] 自动构建完成")
        except Exception as e:
            debug_print(f"[Generator][fewshot] 自动构建失败，将降级为空召回: {type(e).__name__}: {e}")

    @staticmethod
    def extract_sql(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"(SELECT\s+.*)", text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else text.strip()

    @staticmethod
    def highlight_question_for_prompt(question: str) -> str:
        """在 prompt 中加粗问题全文，提醒模型以问题原文为字面量来源。"""
        text = str(question or "").strip()
        return f"**{text}**" if text else ""

    @staticmethod
    def _sql_generation_rules() -> str:
        return (
            "要求：\n"
            "1. 不要重新做 schema linking。\n"
            "2. WHERE/HAVING 中的编码、单号、名称等过滤值必须来自【用户问题】原文，"
            "**严禁使用 Schema 示例值、枚举值或范围中的数字替代问题字面量**。\n"
            "3. 业务编码/单号列（如物料编码、采购订单号）请用带引号的文本字面量，例如 `\"物料编码\" = '500138627'`。\n"
            "4. 结果行如存在多条重复值，请优先考虑使用 DISTINCT 去重。\n"
            "5. 只输出 SQL，用 ```sql 包裹。\n"
        )

    @staticmethod
    def build_m_schema_prompt(
        selected_columns: list[dict],
        all_metadata: list[dict],
        table_name: str = "procurement_table",
        randomize: bool = False,
        profile_detail_map: dict[str, dict] | None = None,
    ) -> str:
        cols = list(selected_columns)
        if randomize:
            import random
            random.shuffle(cols)
        enriched = enrich_schema_columns(cols, all_metadata or [], profile_detail_map)
        return build_light_schema_markdown(enriched, table_name)

    async def _call_llm_sql(
        self,
        prompt: str,
        tracker: TokenTracker,
        temperature: float,
        path_type: str,
    ) -> dict | None:
        try:
            extra_body: dict = {"chat_template_kwargs": {"enable_thinking": False}}
            if settings.generator_prefix_code_fence:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "```sql\n"},
                ]
                extra_body["continue_final_message"] = True
                extra_body["add_generation_prompt"] = False
            else:
                messages = [{"role": "user", "content": prompt}]
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=settings.llm_request_timeout_sec,
                max_tokens=settings.max_gen_tokens,
                extra_body=extra_body or None,
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            sql = self.extract_sql(content)
            if not sql:
                return None
            return {"type": path_type, "sql": sql, "raw_content": content}
        except APIConnectionError as e:
            debug_print(f"[Generator][FATAL][{path_type}] LLM 连接失败: {e}")
            return None
        except Exception as e:
            debug_print(f"[Generator][{path_type}] 生成失败: {type(e).__name__}: {e}")
            return None

    def _build_fewshot_context(self, question: str) -> str:
        try:
            return self.fewshot_store.build_fewshot_prompt(
                query=question,
                top_k=settings.icl_few_shot_k,
            )
        except Exception as e:
            debug_print(f"[Generator][fewshot] 召回失败，降级为空: {type(e).__name__}: {e}")
            return ""

    async def generate_from_plan_async(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
    ) -> dict | None:
        fewshot_context = self._build_fewshot_context(question)
        fewshot_block = f"[Few-shot示例]\n{fewshot_context}\n\n" if fewshot_context else ""
        highlighted_question = self.highlight_question_for_prompt(question)

        base_prompt = (
            "你是一名SQL专家。请只基于给定的 Schema 生成一条 SQLite SQL。\n\n"
            + fewshot_block
            + f"[Schema]\n{schema_prompt}\n"
            + f"[用户问题]\n{highlighted_question}\n"
            + self._sql_generation_rules()
        )

        prompt_direct = base_prompt + "\n[路径提示] 直接根据 Schema 与加粗问题原文生成最简洁 SQL。"
        prompt_icl = base_prompt + "\n[路径提示] 参考字段语义，过滤条件必须逐字来自加粗问题原文。"
        prompt_plan = base_prompt + (
            "\n[路径提示] 在生成最终 SQL 前，必须先给出结构化 Plan，再给出唯一 SQL。\n"
            "在生成最终的 SQL 之前，请严格按照以下 4 个步骤输出你的规划分析（Plan）：\n\n"
            "<Plan>\n"
            "1. 【核心意图】：用户到底想查什么？（例如：求和、计数、最值、条件罗列、比例推算？）\n"
            "2. 【列与实体映射】：\n"
            "   - SELECT 目标列：___\n"
            "   - WHERE 条件列：___ (过滤值必须来自加粗问题原文，禁止使用 Schema 示例值)\n"
            "   - GROUP BY/ORDER BY 列 (如果需要)：___\n"
            "3. 【逻辑陷阱排查】：\n"
            "   - 是否需要过滤空值 (IS NOT NULL)？\n"
            "   - 是否需要去重 (DISTINCT)？\n"
            "   - 排序时是升序 (ASC) 还是降序 (DESC)？\n"
            "4. 【草稿组装】：简述 SQL 各个子句的连接逻辑。\n"
            "</Plan>\n\n"
            "[最终生成]\n"
            "请基于上述 Plan，输出唯一可执行的 SQLite SQL。"
        )

        tasks = [
            asyncio.create_task(self._call_llm_sql(prompt_direct, tracker, settings.direct_temperature, "direct")),
            asyncio.create_task(self._call_llm_sql(prompt_icl, tracker, settings.icl_temperature, "icl")),
            asyncio.create_task(self._call_llm_sql(prompt_plan, tracker, settings.direct_temperature, "plan")),
        ]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r and r.get("sql")]
        if not valid:
            return None
        return valid[0]

    async def start_candidate_tasks(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
    ) -> list[asyncio.Task]:
        fewshot_context = self._build_fewshot_context(question)
        fewshot_block = f"[Few-shot示例]\n{fewshot_context}\n\n" if fewshot_context else ""

        highlighted_question = self.highlight_question_for_prompt(question)
        base_prompt = (
            "你是一名SQL专家。请只基于给定的 Schema 生成一条 SQLite SQL。\n\n"
            + fewshot_block
            + f"[Schema]\n{schema_prompt}\n"
            + f"[用户问题]\n{highlighted_question}\n"
            + self._sql_generation_rules()
        )
        return [
            asyncio.create_task(self._call_llm_sql(base_prompt + "\n[路径提示] direct", tracker, settings.direct_temperature, "direct")),
            asyncio.create_task(self._call_llm_sql(base_prompt + "\n[路径提示] icl", tracker, settings.icl_temperature, "icl")),
            asyncio.create_task(self._call_llm_sql(base_prompt + "\n[路径提示] plan", tracker, settings.direct_temperature, "plan")),
        ]

    async def generate_candidates_async(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
    ) -> list[dict]:
        tasks = await self.start_candidate_tasks(question, schema_prompt, tracker)
        results = await asyncio.gather(*tasks)
        return [r for r in results if r and r.get("sql")]
