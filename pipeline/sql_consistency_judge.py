from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.utils import TokenTracker, debug_print

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "一致": {"type": "boolean"},
        "原因": {"type": "string"},
    },
    "required": ["一致", "原因"],
}

_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "best_index": {"type": "integer"},
        "原因": {"type": "string"},
    },
    "required": ["best_index", "原因"],
}


def _format_sql_result(result: list | None, *, limit: int = 5) -> str:
    if not result:
        return "（空结果）"
    lines: list[str] = []
    for row in result[:limit]:
        if isinstance(row, (list, tuple)):
            lines.append(str(list(row)))
        else:
            lines.append(str(row))
    if len(result) > limit:
        lines.append(f"... 共 {len(result)} 行，仅展示前 {limit} 行")
    return "\n".join(lines)


def parse_judge_response(text: str) -> tuple[bool, str]:
    if not text:
        return True, ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return True, ""
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return True, ""
    if not isinstance(obj, dict):
        return True, ""
    consistent = obj.get("一致")
    if consistent is None:
        consistent = obj.get("consistent")
    reason = str(obj.get("原因") or obj.get("reason") or "").strip()
    if consistent is None:
        return True, reason
    return bool(consistent), reason


def parse_choice_response(text: str) -> tuple[int | None, str]:
    if not text:
        return None, ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, ""
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, ""
    if not isinstance(obj, dict):
        return None, ""
    try:
        idx = int(obj.get("best_index"))
    except (TypeError, ValueError):
        return None, str(obj.get("原因") or obj.get("reason") or "").strip()
    return idx, str(obj.get("原因") or obj.get("reason") or "").strip()


async def judge_sql_consistency(
    client: AsyncOpenAI,
    model: str,
    *,
    question: str,
    schema_prompt: str,
    sql: str,
    result: list | None,
    intent_plan: dict | None = None,
    tracker: TokenTracker | None = None,
) -> tuple[bool, str]:
    """判断可执行 SQL 的过滤/查询语义是否与问题原文一致。"""
    prompt = (
        "你是 SQL 语义审查员。判断给定 SQL 是否忠实使用了【用户问题】中的字面条件，"
        "而不是偷换为 Schema 示例值或其它无关值。\n\n"
        f"[Schema（断崖剪枝列）]\n{schema_prompt}\n\n"
        f"[用户问题]\n{question}\n\n"
        f"[弱意图解析]\n{json.dumps(intent_plan or {}, ensure_ascii=False)}\n\n"
        f"[SQL]\n{sql}\n\n"
        f"[SQL执行结果]\n{_format_sql_result(result)}\n\n"
        "审查要点：\n"
        "1. WHERE/HAVING 中的编码、单号、名称等字面量必须来自用户问题原文，不得使用 Schema 示例值替代。\n"
        "2. SELECT 的目标列应回答用户问题（例如问「是什么/哪家」应查描述类列而非编码列）。\n"
        "3. 若 SQL 能执行但条件与问题不一致，判定为不一致。\n"
        "只输出 JSON：{\"一致\": true/false, \"原因\": \"...\"}\n"
    )
    extra_body: dict = {}
    if getattr(settings, "evidence_use_guided_json", False):
        extra_body["guided_json"] = _JUDGE_SCHEMA
    if not getattr(settings, "enable_thinking_for_entity", True):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            timeout=settings.llm_request_timeout_sec,
            extra_body=extra_body or None,
        )
        if tracker:
            tracker.track(resp)
        content = resp.choices[0].message.content or ""
        consistent, reason = parse_judge_response(content)
        debug_print(f"[ConsistencyJudge] consistent={consistent} reason={reason}")
        return consistent, reason
    except Exception as exc:
        debug_print(f"[ConsistencyJudge] skipped: {type(exc).__name__}: {exc}")
        return True, ""


async def choose_sql_candidate(
    client: AsyncOpenAI,
    model: str,
    *,
    question: str,
    intent_plan: dict | None,
    candidates: list[dict],
    tracker: TokenTracker | None = None,
) -> tuple[int | None, str]:
    """LLM tie-breaker: choose one candidate, never rewrite SQL."""
    lines: list[str] = []
    for idx, cand in enumerate(candidates):
        issues = cand.get("checker_issues") or []
        issue_text = json.dumps(
            [
                {"code": getattr(x, "code", None) or x.get("code", ""), "message": getattr(x, "message", None) or x.get("message", "")}
                if isinstance(x, dict)
                else {"code": getattr(x, "code", ""), "message": getattr(x, "message", "")}
                for x in issues
            ],
            ensure_ascii=False,
        )
        lines.append(
            f"候选 {idx}\n"
            f"type={cand.get('type', '')}, variant={cand.get('variant_id', '')}\n"
            f"SQL:\n{cand.get('sql', '')}\n"
            f"执行结果预览:\n{_format_sql_result(cand.get('result'), limit=3)}\n"
            f"checker issues: {issue_text}\n"
        )
    prompt = (
        "你是 SQL 候选仲裁器。只能从候选中选择最能回答用户问题的一条 SQL，禁止改写或新增 SQL。\n"
        "选择标准：优先忠实使用用户原问题条件、SELECT 目标能回答问题、结果非空且不过度约束。\n\n"
        f"[用户问题]\n{question}\n\n"
        f"[弱意图解析]\n{json.dumps(intent_plan or {}, ensure_ascii=False)}\n\n"
        f"[候选]\n{chr(10).join(lines)}\n"
        "只输出 JSON：{\"best_index\": 0, \"原因\": \"...\"}\n"
    )
    extra_body: dict = {}
    if getattr(settings, "evidence_use_guided_json", False):
        extra_body["guided_json"] = _CHOICE_SCHEMA
    if not getattr(settings, "enable_thinking_for_entity", True):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            timeout=settings.llm_request_timeout_sec,
            extra_body=extra_body or None,
        )
        if tracker:
            tracker.track(resp)
        idx, reason = parse_choice_response(resp.choices[0].message.content or "")
        if idx is None or idx < 0 or idx >= len(candidates):
            return None, reason
        return idx, reason
    except Exception as exc:
        debug_print(f"[ConsistencyJudge][choice] skipped: {type(exc).__name__}: {exc}")
        return None, ""
