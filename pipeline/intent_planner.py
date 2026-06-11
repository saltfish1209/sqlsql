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
        "查询目标": {"type": "string"},
        "筛选条件": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "条件文本": {"type": "string"},
                    "含义": {"type": "string"},
                    "建议匹配方式": {"type": "string"},
                },
                "required": ["条件文本", "含义", "建议匹配方式"],
            },
        },
        "计算条件": {
            "type": "object",
            "properties": {
                "聚合": {"type": "string"},
                "排序": {"type": "string"},
                "top_k": {},
            },
            "required": ["聚合", "排序", "top_k"],
        },
        "风险提示": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["查询目标", "筛选条件", "计算条件", "风险提示"],
}


class IntentPlanner:
    """Builds a weak, non-binding intent sketch for SQL generation."""

    DEFAULT_PLAN = {
        "查询目标": "",
        "筛选条件": [],
        "计算条件": {"聚合": "none", "排序": "none", "top_k": None},
        "风险提示": [],
    }

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    @classmethod
    def normalize_plan(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            return dict(cls.DEFAULT_PLAN)
        calc = value.get("计算条件") if isinstance(value.get("计算条件"), dict) else {}
        filters = value.get("筛选条件") if isinstance(value.get("筛选条件"), list) else []
        risks = value.get("风险提示") if isinstance(value.get("风险提示"), list) else []
        return {
            "查询目标": str(value.get("查询目标") or "").strip(),
            "筛选条件": [x for x in filters if isinstance(x, dict)],
            "计算条件": {
                "聚合": str(calc.get("聚合") or "none").strip() or "none",
                "排序": str(calc.get("排序") or "none").strip() or "none",
                "top_k": calc.get("top_k"),
            },
            "风险提示": [str(x).strip() for x in risks if str(x).strip()],
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
        calc = plan.get("计算条件") or {}
        filter_types = [
            str(item.get("含义") or item.get("条件文本") or "").strip()
            for item in plan.get("筛选条件", [])
            if isinstance(item, dict)
        ]
        parts = [
            f"target={plan.get('查询目标') or 'unknown'}",
            f"agg={calc.get('聚合') or 'none'}",
            f"order={calc.get('排序') or 'none'}",
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
            "你是一个 Text-to-SQL 弱意图解析器。请只输出 JSON，不要生成 SQL。\n"
            "该意图只作为参考，不是硬约束；如果不确定可以留空或写 none。\n"
            "筛选条件必须来自用户问题原文，不得使用 Schema 示例值替代。\n\n"
            f"[Schema]\n{schema_prompt}\n\n"
            f"[实体参考]\n{entity_text}\n\n"
            f"[用户问题]\n{question}\n\n"
            "输出格式：\n"
            '{"查询目标":"", "筛选条件":[{"条件文本":"","含义":"","建议匹配方式":"="}], '
            '"计算条件":{"聚合":"none","排序":"none","top_k":null}, "风险提示":[]}'
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
