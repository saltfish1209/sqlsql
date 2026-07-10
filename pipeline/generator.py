from __future__ import annotations

import asyncio
import re

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.intent_planner import IntentPlanner
from pipeline.prompt_rules import aggregation_rule_text
from pipeline.utils import TokenTracker, debug_print


_SQL_JSON_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
}


class SQLGenerator:
    def __init__(self, client: AsyncOpenAI, model: str, embed_model_path: str | None = None):
        from pipeline.fewshot_index import FewShotFaissStore

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
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            try:
                import json

                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict) and obj.get("sql"):
                    return SQLGenerator.normalize_sql_literal_spaces(str(obj.get("sql") or "").strip())
            except Exception:
                pass
        match = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return SQLGenerator.normalize_sql_literal_spaces(match.group(1).strip())
        match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            return SQLGenerator.normalize_sql_literal_spaces(match.group(1).strip())
        match = re.search(r"(SELECT\s+.*)", text, flags=re.DOTALL | re.IGNORECASE)
        sql = match.group(1).strip() if match else text.strip()
        return SQLGenerator.normalize_sql_literal_spaces(sql)

    @staticmethod
    def normalize_sql_literal_spaces(sql: str) -> str:
        """删除 SQL 单引号业务值内部空白；不改 SQL 语法空格和双引号列名。"""
        if not sql:
            return ""

        def repl(match: re.Match) -> str:
            literal = match.group(0)
            inner = re.sub(r"\s+", "", literal[1:-1])
            return f"'{inner}'"

        return re.sub(r"'(?:''|[^'])*'", repl, sql)

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
            "5. 只输出 JSON，不要输出 Markdown 代码块或解释文字。\n"
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
        from pipeline.schema_format import build_light_schema_markdown, enrich_schema_columns

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
            extra_body: dict = {
                "guided_json": _SQL_JSON_SCHEMA,
                "chat_template_kwargs": {"enable_thinking": False},
            }
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
        fewshot_context = self._build_fewshot_context(question)
        fewshot_block = f"[Few-shot示例]\n{fewshot_context}\n\n" if fewshot_context else ""
        intent_block = IntentPlanner.to_prompt_block(intent_plan)
        intent_section = f"[弱意图解析]\n{intent_block}\n\n" if intent_block else ""
        highlighted_question = self.highlight_question_for_prompt(question)
        diversity_rule = (
            "多候选要求：每条 SQL 必须对应用户问题中的明确查询意图；可以在 SELECT、DISTINCT、"
            "聚合或宽松匹配方式上做合理差异，但不得为了覆盖候选字段而遍历生成无问题依据的 SELECT/WHERE。"
        )
        aggregation_rule = aggregation_rule_text()
        prompt_prefix = "# Role\n你是一名 SQL 专家。请只基于给定 Schema 生成 SQLite SQL。\n\n" + fewshot_block
        prompt_suffix = (
            f"## Schema\n{schema_prompt}\n\n"
            f"## 用户问题\n{highlighted_question}\n\n"
            "## 生成规则\n"
            f"{self._sql_generation_rules()}"
            f"{diversity_rule}\n"
            f"{aggregation_rule}"
            '\n\n## 输出格式\n只输出 JSON：{"sql": "SELECT ..."}。\n'
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
            "intent_plan": [
                "优先参考弱意图解析，但它不是硬约束；如有冲突，以用户问题原文为准。",
                "结合弱意图解析中的风险提示生成 SQL，避免替换用户原文条件。",
            ],
        }
        per_route = max(1, int(getattr(settings, "generator_candidates_per_route", 2)))
        specs: list[dict] = []
        for route, hints in route_prompts.items():
            for idx, hint in enumerate(hints[:per_route], start=1):
                prompt = prompt_prefix
                if route == "intent_plan":
                    prompt += intent_section
                prompt += prompt_suffix
                specs.append(
                    {
                        "type": route,
                        "variant_id": idx,
                        "prompt_variant": hint,
                        "prompt": prompt + f"\n[路径提示:{route}-{idx}] {hint}\n",
                    }
                )
        return specs

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
