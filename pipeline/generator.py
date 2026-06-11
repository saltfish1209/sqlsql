from __future__ import annotations

import asyncio
import re

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.fewshot_index import FewShotFaissStore
from pipeline.intent_planner import IntentPlanner
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
        index_exists = self.fewshot_store.index_path.exists()
        meta_exists = self.fewshot_store.meta_path.exists()
        if index_exists and meta_exists:
            return
        if not settings.fewshot_autobuild_on_start:
            debug_print(f"[Generator][fewshot] index missing: {self.fewshot_store.index_dir}")
            return
        try:
            debug_print("[Generator][fewshot] building index...")
            self.fewshot_store.build_offline_index()
        except Exception as exc:
            debug_print(f"[Generator][fewshot] build skipped: {type(exc).__name__}: {exc}")

    @staticmethod
    def extract_sql(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"(SELECT\s+.*)", text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def highlight_question_for_prompt(question: str) -> str:
        text = str(question or "").strip()
        return f"**{text}**" if text else ""

    @staticmethod
    def _sql_generation_rules() -> str:
        return (
            "要求：\n"
            "1. 不要重新做 schema linking。\n"
            "2. WHERE/HAVING 中的编码、单号、名称等过滤值必须来自用户问题原文，"
            "严禁使用 Schema 示例值、枚举值或范围值替代问题字面量。\n"
            "3. 业务编码/单号列请优先使用带引号的文本字面量，例如 `\"物料编码\" = '500138627'`。\n"
            "4. 结果存在重复值时，优先考虑 DISTINCT。\n"
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
        *,
        variant_id: int = 1,
        prompt_variant: str = "",
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
            return {
                "type": path_type,
                "variant_id": variant_id,
                "prompt_variant": prompt_variant or path_type,
                "sql": sql,
                "raw_content": content,
            }
        except APIConnectionError as exc:
            debug_print(f"[Generator][FATAL][{path_type}] LLM connection failed: {exc}")
            return None
        except Exception as exc:
            debug_print(f"[Generator][{path_type}] failed: {type(exc).__name__}: {exc}")
            return None

    def _build_fewshot_context(self, question: str) -> str:
        try:
            return self.fewshot_store.build_fewshot_prompt(
                query=question,
                top_k=settings.icl_few_shot_k,
            )
        except Exception as exc:
            debug_print(f"[Generator][fewshot] recall skipped: {type(exc).__name__}: {exc}")
            return ""

    def build_generation_prompt_specs(
        self,
        question: str,
        schema_prompt: str,
        intent_plan: dict | None = None,
    ) -> list[dict]:
        query_signature = IntentPlanner.build_query_signature(intent_plan)
        fewshot_context = self._build_fewshot_context(question)
        if fewshot_context and query_signature:
            fewshot_context = f"[Query Signature]\n{query_signature}\n\n{fewshot_context}"
        fewshot_block = f"[Few-shot示例]\n{fewshot_context}\n\n" if fewshot_context else ""
        intent_block = IntentPlanner.to_prompt_block(intent_plan)
        intent_section = f"[弱意图解析]\n{intent_block}\n\n" if intent_block else ""
        highlighted_question = self.highlight_question_for_prompt(question)
        diversity_rule = (
            "多候选要求：每条 SQL 必须对应用户问题中的明确查询意图；可以在 SELECT、DISTINCT、"
            "聚合或宽松匹配方式上做合理差异，但不得为了覆盖候选字段而遍历生成无问题依据的 SELECT/WHERE。"
        )
        base_prompt = (
            "你是一名 SQL 专家。请只基于给定 Schema 生成 SQLite SQL。\n\n"
            + fewshot_block
            + intent_section
            + f"[Schema]\n{schema_prompt}\n"
            + f"[用户问题]\n{highlighted_question}\n"
            + self._sql_generation_rules()
            + diversity_rule
            + "\n只输出一条 SQL，并用 ```sql 包裹。\n"
        )
        route_prompts = {
            "direct": [
                "直接根据 Schema 和问题原文生成最简 SQL。",
                "在保持问题条件不变的前提下，生成一个更稳健的等价 SQL，优先考虑 DISTINCT 或必要的非空过滤。",
            ],
            "icl": [
                "参考 Few-shot 和字段语义生成 SQL，过滤值必须逐字来自用户问题原文。",
                "参考相似查询结构生成 SQL，不得借用示例值或枚举值作为过滤条件。",
            ],
            "plan": [
                "先在心中规划 SELECT、WHERE、聚合和排序，再输出唯一 SQL。",
                "优先检查用户到底要返回描述值、编码值还是统计值，再输出唯一 SQL。",
            ],
            "intent_plan": [
                "优先参考弱意图解析，但它不是硬约束；如有冲突，以用户问题原文为准。",
                "结合弱意图解析中的风险提示生成 SQL，避免替换用户原文条件。",
            ],
        }
        per_route = max(1, int(getattr(settings, "generator_candidates_per_route", 2)))
        specs: list[dict] = []
        for route, hints in route_prompts.items():
            for idx, hint in enumerate(hints[:per_route], start=1):
                specs.append(
                    {
                        "type": route,
                        "variant_id": idx,
                        "prompt_variant": hint,
                        "prompt": base_prompt + f"\n[路径提示:{route}-{idx}] {hint}\n",
                    }
                )
        return specs

    async def generate_from_plan_async(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
        intent_plan: dict | None = None,
    ) -> dict | None:
        candidates = await self.generate_candidates_async(
            question,
            schema_prompt,
            tracker,
            intent_plan=intent_plan,
        )
        return candidates[0] if candidates else None

    async def start_candidate_tasks(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
        intent_plan: dict | None = None,
    ) -> list[asyncio.Task]:
        specs = self.build_generation_prompt_specs(question, schema_prompt, intent_plan=intent_plan)
        tasks: list[asyncio.Task] = []
        for spec in specs:
            temperature = settings.icl_temperature if spec["type"] == "icl" else settings.direct_temperature
            tasks.append(
                asyncio.create_task(
                    self._call_llm_sql(
                        spec["prompt"],
                        tracker,
                        temperature,
                        spec["type"],
                        variant_id=spec["variant_id"],
                        prompt_variant=spec["prompt_variant"],
                    )
                )
            )
        return tasks

    async def generate_candidates_async(
        self,
        question: str,
        schema_prompt: str,
        tracker: TokenTracker,
        intent_plan: dict | None = None,
    ) -> list[dict]:
        tasks = await self.start_candidate_tasks(
            question,
            schema_prompt,
            tracker,
            intent_plan=intent_plan,
        )
        results = await asyncio.gather(*tasks)
        return [r for r in results if r and r.get("sql")]
