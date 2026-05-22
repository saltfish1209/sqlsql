from __future__ import annotations

import json

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.utils import TokenTracker, debug_print

_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "是否多问题": {"type": "boolean"},
        "子问题": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["是否多问题", "子问题"],
}


class QuestionSplitter:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def split(self, question: str, tracker: TokenTracker | None = None) -> list[str]:
        if not getattr(settings, "enable_question_split", True):
            return [question]
        try:
            result = await self._llm_split(question, tracker)
            if not result:
                return [question]
            if not result.get("是否多问题"):
                return [question]
            subs = [str(x).strip() for x in (result.get("子问题") or []) if str(x).strip()]
            if len(subs) <= 1:
                return [question]
            return subs
        except Exception as e:
            debug_print(f"[Splitter] fallback single question: {type(e).__name__}: {e}")
            return [question]

    async def _llm_split(self, question: str, tracker: TokenTracker | None = None) -> dict:
        system_msg = "你是问题拆解助手。只输出 JSON，不输出任何额外文字。"
        user_msg = f"""请判断用户问题是否由多个独立子问题组成，并在需要时拆分。

要求：
1. 若不是多问题，返回：{{"是否多问题": false, "子问题": []}}
2. 若是多问题，返回：{{"是否多问题": true, "子问题": ["...", "..."]}}
3. 子问题必须可独立生成SQL，保持原意，不补充不存在的信息。每个子问题包含原问题中的所有信息

用户问题：
{question}
"""
        extra_body = {
            "guided_json": _SPLIT_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=256,
            timeout=settings.llm_request_timeout_sec,
            extra_body=extra_body,
        )
        if tracker:
            tracker.track(resp)
        raw = resp.choices[0].message.content or ""
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = (text or "").strip()
        try:
            obj = json.loads(text)
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}
        return obj
