from __future__ import annotations

import json
import re

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.utils import TokenTracker, debug_print


class SQLGenerator:
    def __init__(self, client: AsyncOpenAI, model: str, embed_model_path: str | None = None):
        self.client = client
        self.model = model

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
    def build_m_schema_prompt(
        selected_columns: list[dict],
        all_metadata: list[dict],
        table_name: str = "procurement_table",
        randomize: bool = False,
        profile_map: dict[str, str] | None = None,
    ) -> str:
        cols = list(selected_columns)
        if randomize:
            import random
            random.shuffle(cols)
        lines = [f"[数据库表] {table_name}", "[候选字段]"]
        for meta in cols:
            col_name = meta.get("列名") or meta.get("column_name") or ""
            if not col_name:
                continue
            lines.append(f"- 列名：{col_name}")
            if meta.get("列描述"):
                lines.append(f"  列描述：{meta.get('列描述')}")
            if meta.get("字段类型"):
                lines.append(f"  字段类型：{meta.get('字段类型')}")
            if meta.get("是否枚举") is not None:
                lines.append(f"  是否枚举：{meta.get('是否枚举')}")
            if meta.get("空值率") not in (None, ""):
                lines.append(f"  空值率：{meta.get('空值率')}")
            if meta.get("唯一值数") not in (None, ""):
                lines.append(f"  唯一值数：{meta.get('唯一值数')}")
            if meta.get("示例值") not in (None, ""):
                lines.append(f"  示例值：{meta.get('示例值')}")
            if meta.get("格式") not in (None, ""):
                lines.append(f"  格式：{meta.get('格式')}")
            if meta.get("范围") not in (None, ""):
                lines.append(f"  范围：{meta.get('范围')}")
            inline = (profile_map or {}).get(col_name, "")
            if inline:
                lines.append(f"  内联统计：{inline}")
        return "\n".join(lines)

    async def generate_from_plan_async(
        self,
        question: str,
        schema_prompt: str,
        plan_json: dict,
        tracker: TokenTracker,
    ) -> dict | None:
        prompt = (
            "你是一名SQL专家。请只基于给定的结构化 JSON 和 Schema 生成一条 SQLite SQL。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[JSON]\n{json.dumps(plan_json, ensure_ascii=False)}\n"
            f"[用户问题]\n{question}\n"
            "要求：\n"
            "1. 不要重新做 schema linking。\n"
            "2. 不要重新抽实体。\n"
            "3. 结果行如存在多条重复值，请优先考虑使用 DISTINCT 去重，但不要强制使用。\n"
            "4. 只输出 SQL，用 ```sql 包裹。\n"
        )
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
                temperature=settings.direct_temperature,
                timeout=settings.llm_request_timeout_sec,
                max_tokens=settings.max_gen_tokens,
                extra_body=extra_body or None,
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            sql = self.extract_sql(content)
            return {"type": "json_sql", "sql": sql, "raw_content": content, "plan_json": plan_json}
        except APIConnectionError as e:
            debug_print(f"[Generator][FATAL] LLM 连接失败: {e}")
            return None
        except Exception as e:
            debug_print(f"[Generator] generate_from_plan_async error: {type(e).__name__}: {e}")
            return None

    async def start_candidate_tasks(self, *args, **kwargs):
        raise NotImplementedError("旧多路径启动接口已废弃，请使用 generate_from_plan_async")

    async def generate_candidates_async(self, *args, **kwargs):
        raise NotImplementedError("旧多路径接口已废弃")
