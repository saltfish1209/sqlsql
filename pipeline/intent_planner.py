from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.utils import TokenTracker, debug_print


_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "query_target": {"type": "string"},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "meaning": {"type": "string"},
                    "match": {"type": "string"},
                },
                "required": ["text", "meaning", "match"],
            },
        },
        "calculation": {
            "type": "object",
            "properties": {
                "aggregation": {"type": "string"},
                "order": {"type": "string"},
                "top_k": {},
            },
            "required": ["aggregation", "order", "top_k"],
        },
        "risk_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query_target", "filters", "calculation", "risk_notes"],
}


class IntentPlanner:
    """Builds a weak, non-binding intent sketch for SQL generation."""

    DEFAULT_PLAN = {
        "query_target": "",
        "filters": [],
        "calculation": {"aggregation": "none", "order": "none", "top_k": None},
        "risk_notes": [],
    }

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    @classmethod
    def normalize_plan(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            return dict(cls.DEFAULT_PLAN)
        calc = value.get("calculation") or value.get("计算条件")
        if not isinstance(calc, dict):
            calc = {}
        filters = value.get("filters") or value.get("筛选条件")
        if not isinstance(filters, list):
            filters = []
        risks = value.get("risk_notes") or value.get("风险提示")
        if not isinstance(risks, list):
            risks = []
        normalized_filters = []
        for item in filters:
            if not isinstance(item, dict):
                continue
            normalized_filters.append(
                {
                    "text": str(item.get("text") or item.get("条件文本") or "").strip(),
                    "meaning": str(item.get("meaning") or item.get("含义") or "").strip(),
                    "match": str(item.get("match") or item.get("建议匹配方式") or "").strip(),
                }
            )
        return {
            "query_target": str(value.get("query_target") or value.get("查询目标") or "").strip(),
            "filters": normalized_filters,
            "calculation": {
                "aggregation": str(calc.get("aggregation") or calc.get("聚合") or "none").strip() or "none",
                "order": str(calc.get("order") or calc.get("排序") or "none").strip() or "none",
                "top_k": calc.get("top_k"),
            },
            "risk_notes": [str(x).strip() for x in risks if str(x).strip()],
        }

    @classmethod
    def parse_response(cls, text: str) -> dict:
        if not text:
            return dict(cls.DEFAULT_PLAN)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return dict(cls.DEFAULT_PLAN)
        try:
            data = json.loads(match.group(0))
        except Exception:
            return dict(cls.DEFAULT_PLAN)
        return cls.normalize_plan(data)

    @staticmethod
    def to_prompt_block(intent_plan: dict | None) -> str:
        if not intent_plan:
            return ""
        return json.dumps(IntentPlanner.normalize_plan(intent_plan), ensure_ascii=False, indent=2)

    @staticmethod
    def build_query_signature(intent_plan: dict | None) -> str:
        plan = IntentPlanner.normalize_plan(intent_plan)
        calc = plan.get("calculation") or {}
        filter_types = [
            str(item.get("meaning") or item.get("text") or "").strip()
            for item in plan.get("filters", [])
            if isinstance(item, dict)
        ]
        parts = [
            f"target={plan.get('query_target') or 'unknown'}",
            f"agg={calc.get('aggregation') or 'none'}",
            f"order={calc.get('order') or 'none'}",
            "filters=" + ",".join([x for x in filter_types if x]),
        ]
        return " | ".join(parts)

    async def plan_async(
        self,
        question: str,
        schema_prompt: str,
        *,
        entities: list[str] | None = None,
        tracker: TokenTracker | None = None,
    ) -> dict:
        entity_text = json.dumps(entities or [], ensure_ascii=False)
        prompt = (
            "# Role\n你是一个 Text-to-SQL 弱意图解析器。请只输出 JSON，不要生成 SQL。\n\n"
            "## 约束\n"
            "- 该意图只作为参考，不是硬约束；如果不确定可以留空或写 none。\n"
            "- 筛选条件必须来自用户问题原文，不得使用 Schema 示例值替代。\n\n"
            f"## Schema\n{schema_prompt}\n\n"
            f"## 实体参考\n{entity_text}\n\n"
            f"## 用户问题\n{question}\n\n"
            "## 输出格式\n"
            '{"query_target":"", "filters":[{"text":"","meaning":"","match":"="}], '
            '"calculation":{"aggregation":"none","order":"none","top_k":null}, "risk_notes":[]}'
        )
        extra_body: dict = {}
        if getattr(settings, "intent_plan_use_guided_json", True):
            extra_body["guided_json"] = _INTENT_SCHEMA
        if not getattr(settings, "enable_thinking_for_entity", True):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=settings.llm_request_timeout_sec,
                max_tokens=settings.intent_plan_max_tokens,
                extra_body=extra_body or None,
            )
            if tracker:
                tracker.track(resp)
            return self.parse_response(resp.choices[0].message.content or "")
        except Exception as exc:
            debug_print(f"[IntentPlanner] skipped: {type(exc).__name__}: {exc}")
            return dict(self.DEFAULT_PLAN)
