from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.consensus_vote import annotate_candidate_risk
from pipeline.db_engine import DBEngine
from pipeline.generator import SQLGenerator
from pipeline.utils import TokenTracker


_TEXT_CONDITION_RE = re.compile(
    r'(?:(?:"(?P<quoted_col>[^"]+)")|(?P<plain_col>[A-Za-z_][\w]*))'
    r"\s*(?P<operator>=|LIKE)\s*'(?P<literal>(?:''|[^'])*)'",
    re.IGNORECASE,
)
_EXACT_IDENTIFIER_MARKERS = ("编码", "编号", "单号", "订单号")


def _allows_new_like(candidate: dict, proposed_sql: str, judge_suggestion: str) -> bool:
    source_sql = str(candidate.get("sql") or "")
    source_likes = {
        (match.group("quoted_col") or match.group("plain_col") or "", match.group("literal"))
        for match in _TEXT_CONDITION_RE.finditer(source_sql)
        if match.group("operator").upper() == "LIKE"
    }
    proposed_likes = [
        (
            match.group("quoted_col") or match.group("plain_col") or "",
            match.group("literal").replace("''", "'").strip("%_"),
        )
        for match in _TEXT_CONDITION_RE.finditer(proposed_sql)
        if match.group("operator").upper() == "LIKE"
        and (match.group("quoted_col") or match.group("plain_col") or "", match.group("literal"))
        not in source_likes
    ]
    if not proposed_likes:
        return True
    if "LIKE" not in str(judge_suggestion or "").upper():
        return False

    checks = (candidate.get("value_link_probe") or {}).get("literal_checks", [])
    for column, literal in proposed_likes:
        if any(marker in column for marker in _EXACT_IDENTIFIER_MARKERS):
            return False
        if not any(
            check.get("column") == column
            and str(check.get("literal") or "").strip("%_") == literal
            and bool(check.get("like_exists"))
            for check in checks
        ):
            return False
    return True


class SQLRefiner:
    def __init__(self, client: AsyncOpenAI, model: str, db: DBEngine):
        self.client = client
        self.model = model
        self.db = db

    async def refine_async(
        self,
        schema_prompt: str,
        candidates: list[dict],
        top20_candidates: list[dict] | list[str],
        tracker: TokenTracker,
        max_retries: int | None = None,
        *,
        repair_schema_prompt: str | None = None,
        question: str | None = None,
        judge_schema_prompt: str | None = None,
        intent_plan: dict | None = None,
    ) -> list[dict]:
        refined = []
        failure_order = 0
        literal_exists = getattr(self.db, "check_literal_in_column", None)
        literal_probe = getattr(self.db, "probe_literal_in_column", None)
        for cand in candidates:
            repaired = dict(cand)
            result, error = self.db.execute_sql(repaired["sql"])
            repaired["execution_error"] = error
            repaired["refiner_debug"] = {
                "judge_rounds": [],
                "refiner_rounds": [],
            }
            if error:
                failure_order += 1
                repaired["error_msg"] = error
                repaired.setdefault("result", None)
                repaired["status"] = "needs_repair"
                repaired["failure_order"] = failure_order
                repaired["error_history"] = [
                    {
                        "order": failure_order,
                        "round": 0,
                        "sql": repaired.get("sql"),
                        "error": error,
                    }
                ]
            else:
                repaired["status"] = "success"
                repaired["result"] = result if result is not None else []
            annotate_candidate_risk(repaired, literal_exists, literal_probe)
            refined.append(repaired)

        max_rounds = max(1, int(max_retries or settings.max_repair_retries))
        for round_no in range(1, max_rounds + 1):
            judge_schema = judge_schema_prompt or schema_prompt
            judge_items: list[dict] = []
            judge_debug = {
                "ran": False,
                "round": round_no,
                "schema": "generation",
                "items": [],
                "candidate_sqls": [
                    {
                        "index": idx,
                        "type": item.get("type"),
                        "variant_id": item.get("variant_id"),
                        "sql": item.get("sql"),
                        "error": item.get("execution_error") or "",
                        "value_link_probe": item.get("value_link_probe") or {},
                    }
                    for idx, item in enumerate(refined)
                ],
            }
            if settings.enable_sql_consistency_judge and question:
                from pipeline.sql_consistency_judge import judge_sql_batch_consistency

                judge_items = await judge_sql_batch_consistency(
                    self.client,
                    self.model,
                    question=question,
                    schema_prompt=judge_schema,
                    candidates=refined,
                    intent_plan=intent_plan,
                    tracker=tracker,
                )
                judge_debug = {
                    **judge_debug,
                    "ran": True,
                    "items": judge_items,
                }

            targets: list[tuple[int, dict]] = []
            target_reasons: dict[int, str] = {}
            seen_target_ids: set[int] = set()
            for item in judge_items:
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(refined):
                    continue
                target = refined[idx]
                status = str(item.get("status") or "").strip().lower()
                if status not in {"pass", "suspicious", "fail"}:
                    status = "fail" if item.get("correct") is False else "pass"
                risk = int(item.get("risk") if item.get("risk") is not None else {"pass": 0, "suspicious": 1, "fail": 2}[status])
                reason = str(item.get("reason") or "").strip()
                if target.get("judge_status") is None:
                    target["judge_status"] = status
                    target["judge_risk"] = risk
                    target["judge_reason"] = reason
                elif target.get("is_refined"):
                    target["post_repair_judge_status"] = status
                    target["post_repair_judge_risk"] = risk
                    target["post_repair_judge_reason"] = reason
                if reason:
                    target.setdefault("judge_suggestion", reason)
                    target_reasons[id(target)] = reason
                if (
                    status == "fail"
                    and not target.get("_repair_attempted")
                    and id(target) not in seen_target_ids
                ):
                    targets.append((idx, target))
                    seen_target_ids.add(id(target))

            if judge_debug.get("ran"):
                for item in refined:
                    debug = item.setdefault("refiner_debug", {})
                    debug.setdefault("judge_rounds", []).append(judge_debug)

            for idx, item in enumerate(refined):
                if (
                    item.get("status") != "success"
                    and not item.get("_repair_attempted")
                    and id(item) not in seen_target_ids
                ):
                    item.setdefault("judge_status", "fail")
                    item.setdefault("judge_risk", 2)
                    if item.get("execution_error") and not item.get("judge_reason"):
                        item["judge_reason"] = str(item.get("execution_error") or "")
                    targets.append((idx, item))
                    seen_target_ids.add(id(item))

            if not targets:
                break

            failed_sqls = [
                {
                    "order": item.get("failure_order"),
                    "type": item.get("type"),
                    "variant_id": item.get("variant_id"),
                    "sql": item.get("sql"),
                    "error": item.get("execution_error") or item.get("error_msg"),
                }
                for _idx, item in targets
            ]
            judge_debug["failed_sqls"] = failed_sqls

            repair_schema = (
                judge_schema_prompt or schema_prompt
                if round_no == 1
                else repair_schema_prompt or schema_prompt
            )
            repair_schema_name = "generation" if round_no == 1 else "repair"
            for target_index, target in targets:
                target["_repair_attempted"] = True
                sql_before = target.get("sql")
                error_before = target.get("execution_error") or target.get("error_msg") or ""
                judge_reason = target_reasons.get(id(target), target.get("judge_suggestion") or target.get("judge_reason") or "")
                repair_candidate = dict(target)
                repair_candidate.pop("_repair_attempted", None)
                repair_candidate["is_refined"] = True
                repair_candidate["refined_from"] = target_index
                repair_candidate["refiner_round"] = round_no
                repair_candidate["source_sql"] = sql_before
                target_debug = target.get("refiner_debug") or {}
                repair_candidate["refiner_debug"] = {
                    "judge_rounds": list(target_debug.get("judge_rounds") or []),
                    "refiner_rounds": list(target_debug.get("refiner_rounds") or []),
                }
                if judge_reason:
                    repair_candidate["error_msg"] = (
                        f"{error_before}\nJudge suggestion: {judge_reason}"
                        if error_before
                        else judge_reason
                    )
                    repair_candidate["judge_suggestion"] = judge_reason
                repaired = await self._attempt_llm_repair(
                    schema_prompt=repair_schema,
                    candidate=repair_candidate,
                    valid_columns=top20_candidates,
                    tracker=tracker,
                    max_retries=max_rounds,
                    judge_suggestion=judge_reason,
                    repair_round=round_no,
                    schema_name=repair_schema_name,
                )
                repaired["is_refined"] = True
                repaired["refined_from"] = target_index
                repaired["refiner_round"] = round_no
                repaired["source_sql"] = sql_before
                repaired["repair_changed"] = (
                    str(repaired.get("sql") or "").strip() != str(sql_before or "").strip()
                )
                if not repaired["repair_changed"]:
                    repaired["status"] = "repair_failed"
                    repaired["error_msg"] = "REPAIR_NO_CHANGE"
                elif repaired.get("status") == "success":
                    repaired["post_repair_judge_status"] = "suspicious"
                    repaired["post_repair_judge_risk"] = 1
                    repaired["post_repair_judge_reason"] = "repair_pending_rejudge"
                annotate_candidate_risk(repaired, literal_exists, literal_probe)
                if repaired.get("status") != "success":
                    failure_order += 1
                    repaired["failure_order"] = failure_order
                    repaired.setdefault("error_history", []).append(
                        {
                            "order": failure_order,
                            "round": round_no,
                            "sql": repaired.get("sql"),
                            "error": repaired.get("execution_error") or repaired.get("error_msg"),
                        }
                    )
                refiner_round = {
                    "round": round_no,
                    "schema": repair_schema_name,
                    "sql_before": sql_before,
                    "sql_after": repaired.get("sql"),
                    "status": repaired.get("status"),
                    "error_msg": repaired.get("error_msg"),
                    "post_execution_error": repaired.get("execution_error"),
                    "judge_suggestion": judge_reason,
                }
                debug = repaired.setdefault("refiner_debug", {})
                debug.setdefault("refiner_rounds", []).append(refiner_round)
                debug["judge"] = judge_debug
                debug["refiner"] = refiner_round
                refined.append(repaired)
        return refined

    async def _attempt_llm_repair(
        self,
        schema_prompt: str,
        candidate: dict,
        valid_columns: list[dict] | list[str],
        tracker: TokenTracker,
        max_retries: int,
        *,
        judge_suggestion: str = "",
        repair_round: int = 1,
        schema_name: str = "",
    ) -> dict:
        sql = candidate.get("sql") or ""
        prompt = (
            "你是SQL修复专家。SQL 无法执行或语义审查认为错误，请基于 Schema 与错误诊断重新生成正确 SQL。\n"
            f"[修复轮次]\n{repair_round}/{max_retries}\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[错误SQL]\n{sql}\n"
            f"[错误原因]\n{candidate.get('error_msg', '')}\n"
            f"[Judge修改建议]\n{judge_suggestion or '无'}\n"
            f"[数据库字面量证据]\n{json.dumps(candidate.get('value_link_probe') or {}, ensure_ascii=False)}\n"
            "要求：只输出可执行SQL，用```sql包裹。"
        )
        prompt += (
            "\n[最小修改原则]\n"
            "1. 只围绕错误原因修复，不要额外添加用户问题没有要求的过滤条件。\n"
            "2. WHERE/HAVING 中的编号、单号、名称等过滤值必须来自用户问题原文或错误原因中明确指出的原文条件。\n"
            "3. 不得使用 Schema 示例值或枚举值替换用户原文；只有数据库字面量证据中的 candidate_values "
            "可作为规范实体值，并继续使用精确等值 `=`。\n"
            "4. 空结果本身不能触发模糊匹配；默认保留精确等值 `=`。\n"
            "5. 只有 Judge 明确建议放宽，且对应字面量证据为 like_exists=true 时，"
            "才把名称或描述类文本条件改为 `LIKE '%原值%'`；"
            "必须保留用户原始字面量，订单号、编码、编号等精确标识始终使用 `=`。\n"
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.refiner_temperature,
                timeout=settings.llm_request_timeout_sec,
                max_tokens=settings.refiner_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": settings.enable_thinking_for_refiner}},
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            sql2 = SQLGenerator.extract_sql(content)
            if sql2 and not _allows_new_like(candidate, sql2, judge_suggestion):
                sql2 = sql
            candidate["sql"] = sql2 or sql
            result, error = self.db.execute_sql(candidate["sql"])
            candidate["execution_error"] = error
            candidate["result"] = result if error is None else None
            if error is None:
                candidate["status"] = "success"
                candidate.pop("error_msg", None)
            else:
                candidate["status"] = "needs_repair"
                candidate["error_msg"] = error
            return candidate
        except Exception:
            candidate["status"] = "repair_failed"
            return candidate
