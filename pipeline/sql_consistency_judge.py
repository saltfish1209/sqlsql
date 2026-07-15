from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.prompt_rules import aggregation_rule_text
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

_BATCH_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "status": {"type": "string", "enum": ["pass", "suspicious", "fail"]},
                    "correct": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "status"],
            },
        },
    },
    "required": ["items"],
}

_JUDGE_STATUS_RISK = {
    "pass": 0,
    "suspicious": 1,
    "fail": 2,
}


def _normalize_batch_judge_status(item: dict) -> tuple[str, bool, int]:
    raw_status = str(
        item.get("status")
        or item.get("verdict")
        or item.get("decision")
        or ""
    ).strip().lower()
    status_aliases = {
        "ok": "pass",
        "true": "pass",
        "correct": "pass",
        "pass": "pass",
        "warning": "suspicious",
        "warn": "suspicious",
        "uncertain": "suspicious",
        "maybe": "suspicious",
        "suspicious": "suspicious",
        "false": "fail",
        "wrong": "fail",
        "incorrect": "fail",
        "failed": "fail",
        "fail": "fail",
    }
    status = status_aliases.get(raw_status)
    if status is None:
        correct = item.get("correct")
        if correct is None:
            correct = item.get("\u4e00\u81f4")
        status = "pass" if correct is None or bool(correct) else "fail"
    return status, status == "pass", _JUDGE_STATUS_RISK[status]


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


def _format_failed_sqls(
    failed_sqls: list[dict] | None,
    *,
    sql: str,
    execution_error: str | None,
) -> str:
    items = failed_sqls or [{"order": 1, "sql": sql, "error": execution_error}]
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        order = item.get("order") or idx
        type_text = item.get("type") or ""
        variant_id = item.get("variant_id")
        path_text = type_text
        if variant_id not in (None, ""):
            path_text = f"{path_text} variant={variant_id}".strip()
        if path_text:
            lines.append(f"失败顺序 #{order}（{path_text}）")
        else:
            lines.append(f"失败顺序 #{order}")
        lines.append("SQL:")
        lines.append(str(item.get("sql") or ""))
        lines.append("执行错误:")
        lines.append(str(item.get("error") or ""))
        lines.append("")
    return "\n".join(lines).strip()


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


def parse_batch_judge_response(text: str) -> list[dict]:
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
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []

    parsed: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        status, correct, risk = _normalize_batch_judge_status(item)
        reason = "" if correct else str(item.get("reason") or item.get("原因") or "").strip()
        parsed.append(
            {
                "index": index,
                "correct": correct,
                "status": status,
                "reason": reason,
                "risk": risk,
            }
        )
    return parsed


def _format_batch_candidates(
    candidates: list[dict],
    indices: list[int] | None = None,
) -> str:
    lines: list[str] = []
    candidate_indices = indices or list(range(len(candidates)))
    for idx, cand in zip(candidate_indices, candidates):
        lines.append(f"候选 {idx}")
        path_text = f"type={cand.get('type', '')}, variant={cand.get('variant_id', '')}".strip()
        if path_text:
            lines.append(path_text)
        lines.append("SQL:")
        lines.append(str(cand.get("sql") or ""))
        error = cand.get("execution_error") or cand.get("error_msg")
        if error:
            lines.append("执行错误:")
            lines.append(str(error))
        else:
            lines.append("执行结果预览:")
            lines.append(_format_sql_result(cand.get("result"), limit=3))
        probe = cand.get("value_link_probe") or {}
        if probe.get("reasons"):
            lines.append("事实风险信号:")
            lines.append(json.dumps(probe, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines).strip()


async def judge_sql_batch_consistency(
    client: AsyncOpenAI,
    model: str,
    *,
    question: str,
    schema_prompt: str,
    candidates: list[dict],
    intent_plan: dict | None = None,
    tracker: TokenTracker | None = None,
) -> list[dict]:
    """批量判断同题 SQL 是否语义正确；只给错误 SQL 返回原因。"""
    if not candidates:
        return []
    extra_body: dict = {}
    if getattr(settings, "evidence_use_guided_json", False):
        extra_body["guided_json"] = _BATCH_JUDGE_SCHEMA
    if not getattr(settings, "enable_thinking_for_entity", True):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}


    async def _judge_subset(indexed_candidates: list[tuple[int, dict]]) -> list[dict]:
        indices = [idx for idx, _cand in indexed_candidates]
        subset = [cand for _idx, cand in indexed_candidates]
        output_example = {
            "items": [
                {"index": idx, "status": "pass", "correct": True}
                for idx in indices
            ]
        }
        prompt = (
            "你是 SQL 语义审查员。请逐条判断候选 SQL 是否忠实回答用户问题。\n"
            "审查时使用与 SQL 生成相同的 Schema、用户问题和聚合规则，但你的任务不是生成 SQL，而是判断正误。\n\n"
            f"[Schema]\n{schema_prompt}\n\n"
            f"[用户问题]\n{question}\n\n"
            f"[弱意图解析]\n{json.dumps(intent_plan or {}, ensure_ascii=False)}\n\n"
            f"[聚合规则]\n{aggregation_rule_text()}\n\n"
            f"[候选SQL]\n{_format_batch_candidates(subset, indices)}\n\n"
            "判断要点：\n"
            "1. WHERE/HAVING 条件、过滤值、项目名、订单号、物料编码等必须来自用户问题原文。\n"
            "2. SELECT 目标必须能回答用户问的对象；问供应商就查供应商，问物料类别就查类别。\n"
            "3. 聚合函数必须遵循聚合规则，个数问题用 COUNT，明确求和问题才用 SUM。\n"
            "4. JOIN、GROUP BY、DISTINCT、非空过滤不能改变用户问题语义。\n"
            "5. SQL 有执行错误时直接判 fail，并在原因中简要说明。\n"
            "6. 初始候选使用精确等值 `=`。若 exact_exists=false 且 candidate_values 非空，"
            "优先建议使用规范值并继续保持 `=`；只有 like_exists=true 且目标列是名称或描述类文本字段时，"
            "才能建议 `LIKE '%原值%'`。订单号、编码、编号等精确标识不得建议 LIKE。"
            "仅凭空结果不能建议放宽，证据不足时判 suspicious。\n"
            "7. 弱意图解析只是参考；若它与用户问题、Schema 或 SQL 实际字段冲突，以后三者为准。\n"
            "输出三态 status：pass=可直接采用；suspicious=能执行但可能过宽/过窄，仅降权；fail=高置信错误，可进入修复。\n"
            f"必须为索引 {indices} 中的每个候选各返回且只返回一项，不得遗漏索引。\n"
            f"只输出同样结构的 JSON，例如：{json.dumps(output_example, ensure_ascii=False)}。\n"
            "pass 不要填写 reason；suspicious/fail 必须填写 reason。"
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                timeout=settings.llm_request_timeout_sec,
                extra_body=extra_body or None,
            )
            if tracker:
                tracker.track(resp)
            expected = set(indices)
            parsed = parse_batch_judge_response(resp.choices[0].message.content or "")
            unique: dict[int, dict] = {}
            for item in parsed:
                idx = item.get("index")
                if idx in expected and idx not in unique:
                    unique[idx] = item
            return list(unique.values())
        except Exception as exc:
            debug_print(f"[ConsistencyJudge][batch] skipped: {type(exc).__name__}: {exc}")
            return []

    indexed = list(enumerate(candidates))
    item_by_index: dict[int, dict] = {}
    batch_size = 4
    for start in range(0, len(indexed), batch_size):
        for item in await _judge_subset(indexed[start : start + batch_size]):
            item_by_index[item["index"]] = item

    missing = [(idx, cand) for idx, cand in indexed if idx not in item_by_index]
    for idx, cand in missing:
        retried = await _judge_subset([(idx, cand)])
        if retried:
            item_by_index[idx] = retried[0]
        else:
            item_by_index[idx] = {
                "index": idx,
                "correct": False,
                "status": "suspicious",
                "reason": "judge_output_missing",
                "risk": 1,
            }

    items = [item_by_index[idx] for idx in range(len(candidates))]
    debug_print(f"[ConsistencyJudge][batch] items={items}")
    return items


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
    execution_error: str | None = None,
    failed_sqls: list[dict] | None = None,
) -> tuple[bool, str]:
    """判断可执行 SQL 的过滤/查询语义是否与问题原文一致。"""
    if execution_error:
        prompt = (
            "你是 SQL 错误诊断员。给定 SQL 无法执行，请判断最可能的出错原因，"
            "并给出面向 SQL 修复器的简短修复建议。\n\n"
            f"[Schema]\n{schema_prompt}\n\n"
            f"[用户问题]\n{question}\n\n"
            f"[弱意图解析]\n{json.dumps(intent_plan or {}, ensure_ascii=False)}\n\n"
            f"[运行失败SQL列表]\n{_format_failed_sqls(failed_sqls, sql=sql, execution_error=execution_error)}\n\n"
            "诊断要点：\n"
            "1. 优先判断列名、表名、聚合结构、SQL 语法是否错误。\n"
            "2. 修复建议必须保持用户问题中的编号、单号、名称等过滤值不被替换。\n"
            "3. 对每条失败 SQL 结合其错误信息给出可操作的修改建议。\n"
            "4. 不要要求删除用户问题中的必要条件。\n"
            "只输出 JSON：{\"一致\": false, \"原因\": \"...\"}\n"
        )
    else:
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
