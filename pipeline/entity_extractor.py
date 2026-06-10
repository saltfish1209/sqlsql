from __future__ import annotations

import json
import re

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.utils import TokenTracker, debug_print


_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "提取实体": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["提取实体"],
}


class EntityExtractor:
    """Extract evidence entity texts only."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def extract(
        self,
        question: str,
        candidate_schema_pack: dict,
        tracker: TokenTracker | None = None,
        max_retries: int = 2,
        schema_columns: list[str] | None = None,
    ) -> list[str]:
        schema_cols = schema_columns or []
        schema_lines = []
        candidates = candidate_schema_pack.get("召回schema") or candidate_schema_pack.get("Top20候选") or []
        for c in candidates[: settings.evidence_schema_top_k]:
            field_name = c.get("列名") or ""
            field_desc = c.get("列描述") or ""
            schema_lines.append(f"- 列名：{field_name}\n  列描述：{field_desc}")
        schema_text = "\n".join(schema_lines)
        llm_entities = await self._llm_extract(question, schema_text, tracker, max_retries)
        cleaned = self._post_filter(llm_entities, question, schema_cols)
        debug_print(f"[Entity] raw={llm_entities}")
        debug_print(f"[Entity] cleaned={cleaned}")
        return cleaned

    async def _llm_extract(
        self,
        question: str,
        schema_text: str,
        tracker: TokenTracker | None,
        max_retries: int,
    ) -> list[str]:
        system_msg = "你是一个询价问题实体抽取器。只输出 JSON，不要输出任何其他内容。"
        user_msg = f"""请从用户问题中抽取用于价格查询的实体。

【候选字段上下文】
{schema_text}

【抽取要求】
1. 只抽与查询有关的原始文本片段。
2. 宁可少抽，也不要切错。
3. 不要输出字段名解释、不要输出编号、不要输出置信度、不要输出任何额外文字。
4. 【核心警告】当遇到由逗号、顿号连写的长规格、长型号或物料长描述（如："型号A,参数B,材质C......"）时，必须将它们作为一个整体连续提取！绝对禁止按标点符号将其切碎！
   - 错误提取示例：["型号A", "参数B", "材质C", "..."]
   - 正确提取示例：["型号A,参数B,材质C..."]
5.不要提取其中的口语字段、**只提取关键查找文本内容**
6. 只输出合法 JSON，且 JSON 字段名必须使用中文，格式严格为：
{{"提取实体":["实体1","实体2"]}}
7. 如果没有可提取实体，输出：
{{"提取实体":[]}}

【示例】
输入：
问题KSH20230106环网柜,AC10kV,630A,电压互感器柜,SF6,户内含税单价是多少？
输出：
{{"提取实体":["KSH20230106环网柜,AC10kV,630A,电压互感器柜,SF6,户内","含税单价"]}}

【当前问题】
{question}
"""
        for attempt in range(max_retries):
            try:
                extra_body: dict = {}
                if settings.evidence_use_guided_json:
                    extra_body["guided_json"] = _ENTITY_SCHEMA
                if not settings.enable_thinking_for_entity:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    temperature=0.0,
                    max_tokens=settings.evidence_json_max_tokens,
                    timeout=settings.llm_request_timeout_sec,
                    extra_body=extra_body or None,
                )
                if tracker:
                    tracker.track(resp)
                raw = resp.choices[0].message.content or ""
                data = self._parse_json(raw)
                if data:
                    return data
            except APIConnectionError:
                return []
            except Exception as e:
                debug_print(f"[Entity] attempt={attempt + 1} error={type(e).__name__}: {e}")
        return []

    @staticmethod
    def _parse_json(text: str) -> list[str]:
        if not text:
            return []
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return []
        ents = obj.get("提取实体") or obj.get("evidence_entities", []) if isinstance(obj, dict) else []
        out: list[str] = []
        for item in ents:
            if not isinstance(item, str):
                item = str(item)
            item = item.strip()
            if item:
                out.append(item)
        return out

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", "", s)

    def _post_filter(self, entities: list[str], question: str, schema_cols: list[str]) -> list[str]:
        q_norm = self._normalize(question)
        schema_norm = {self._normalize(c) for c in schema_cols}
        seen: set[str] = set()
        cleaned: list[str] = []
        for ent in entities:
            text = self._normalize(str(ent).strip().strip("，,。.；;:：\"'`"))
            if not text or text in seen:
                continue
            if text in schema_norm:
                continue
            if text not in q_norm:
                continue
            seen.add(text)
            cleaned.append(text)
            if len(cleaned) >= settings.evidence_entity_max_items:
                break
        return cleaned
