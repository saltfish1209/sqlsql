from __future__ import annotations

import json

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.checkers import SQLCheckerChain, CheckIssue
from pipeline.db_engine import DBEngine
from pipeline.generator import SQLGenerator
from pipeline.utils import TokenTracker, debug_print


class SQLRefiner:
    def __init__(self, client: AsyncOpenAI, model: str, db: DBEngine):
        self.client = client
        self.model = model
        self.db = db
        self.checker = SQLCheckerChain(db)

    async def refine_async(
        self,
        schema_prompt: str,
        candidates: list[dict],
        top20_candidates: list[dict] | list[str],
        tracker: TokenTracker,
        max_retries: int | None = None,
        *,
        repair_schema_prompt: str | None = None,
    ) -> list[dict]:
        refined = []
        for cand in candidates:
            repaired = dict(cand)
            result, error = self.db.execute_sql(repaired["sql"])
            issues = self.checker.check(
                repaired["sql"],
                result=result if error is None else None,
                execution_error=error,
            )
            if issues:
                repaired["checker_issues"] = issues
                repaired["error_msg"] = self._format_checker_feedback(issues)
                repaired.setdefault("result", None)
                repaired["status"] = "needs_repair"
                sql_before = repaired.get("sql")
                repaired = await self._attempt_llm_repair(
                    schema_prompt=repair_schema_prompt or schema_prompt,
                    candidate=repaired,
                    valid_columns=top20_candidates,
                    tracker=tracker,
                    max_retries=max_retries or settings.max_repair_retries,
                    repair_reason="execution",
                )
                repaired["refiner_debug"] = {
                    "check": self._issues_to_dict(issues),
                    "refiner": {
                        "sql_before": sql_before,
                        "sql_after": repaired.get("sql"),
                        "status": repaired.get("status"),
                        "error_msg": repaired.get("error_msg"),
                        "post_check": self._issues_to_dict(repaired.get("checker_issues") or []),
                    },
                }
                refined.append(repaired)
            else:
                repaired["status"] = "success"
                repaired["result"] = result if result is not None else []
                refined.append(repaired)
        return refined

    @staticmethod
    def _format_checker_feedback(issues: list[CheckIssue]) -> str:
        return "\n".join(
            f"{issue.code}: {issue.message} 修复建议: {issue.directive}"
            for issue in issues
        )

    @staticmethod
    def _issues_to_dict(issues: list[CheckIssue]) -> list[dict]:
        return [
            {"code": issue.code, "message": issue.message, "directive": issue.directive}
            for issue in (issues or [])
        ]

    async def _attempt_llm_repair(
        self,
        schema_prompt: str,
        candidate: dict,
        valid_columns: list[dict] | list[str],
        tracker: TokenTracker,
        max_retries: int,
        *,
        repair_reason: str = "execution",
    ) -> dict:
        sql = candidate.get("sql") or ""
        if repair_reason == "semantic":
            prompt = (
                "你是SQL修复专家。当前 SQL 能执行，但语义审查认为其与用户问题不一致。\n"
                "请基于断崖剪枝 Schema、用户问题、原 SQL 与执行结果，重新生成正确 SQL。\n"
                f"[Schema]\n{schema_prompt}\n"
                f"[错误SQL]\n{sql}\n"
                f"[错误原因]\n{candidate.get('error_msg', '')}\n"
                "要求：过滤值必须来自用户问题原文，禁止使用 Schema 示例值；只输出可执行 SQL，用```sql包裹。"
            )
        else:
            prompt = (
                "你是SQL修复专家。SQL 无法执行或 checker 报错，请基于 TopK 扩展 Schema 重新生成正确 SQL。\n"
                f"[Schema]\n{schema_prompt}\n"
                f"[错误SQL]\n{sql}\n"
                f"[错误原因]\n{candidate.get('error_msg', '')}\n"
                "要求：只输出可执行SQL，用```sql包裹。"
            )
        prompt += (
            "\n[最小修改原则]\n"
            "1. 只围绕错误原因修复，不要额外添加用户问题没有要求的过滤条件。\n"
            "2. WHERE/HAVING 中的编号、单号、名称等过滤值必须来自用户问题原文或错误原因中明确指出的原文条件。\n"
            "3. 不得使用 Schema 示例值、枚举值或其它候选值替换用户原文条件。\n"
            "4. 如果错误是空结果，优先放宽过严条件，不要替换用户给出的编号或单号。\n"
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
            candidate["sql"] = sql2 or sql
            result, error = self.db.execute_sql(candidate["sql"])
            candidate["result"] = result if error is None else None
            candidate["checker_issues"] = self.checker.check(candidate["sql"], result=result if error is None else None, execution_error=error)
            if not candidate["checker_issues"]:
                candidate["status"] = "success"
                candidate.pop("error_msg", None)
            else:
                candidate["status"] = "needs_repair"
            return candidate
        except Exception:
            return candidate

    async def repair_semantic_async(
        self,
        cliff_schema_prompt: str,
        candidate: dict,
        *,
        question: str,
        tracker: TokenTracker,
        judge_reason: str = "",
    ) -> dict:
        """SQL 可执行但语义不一致时，用断崖 schema + 问题 + SQL + 结果重生成。"""
        repaired = dict(candidate)
        result_preview = repaired.get("result")
        if result_preview is None:
            result_preview = []
        preview_lines = []
        for row in (result_preview or [])[:5]:
            preview_lines.append(str(list(row) if isinstance(row, (list, tuple)) else row))
        result_text = "\n".join(preview_lines) if preview_lines else "（空结果）"
        repaired["error_msg"] = (
            f"语义审查不一致: {judge_reason}\n"
            f"[用户问题]\n{question}\n"
            f"[SQL执行结果]\n{result_text}"
        )
        return await self._attempt_llm_repair(
            schema_prompt=cliff_schema_prompt,
            candidate=repaired,
            valid_columns=[],
            tracker=tracker,
            max_retries=settings.max_repair_retries,
            repair_reason="semantic",
        )

    @staticmethod
    def _build_relaxed_recall_schema(recall_schema: list[dict]) -> list[dict]:
        if not recall_schema:
            return []
        best = max(float(x.get("相关性分数") or 0.0) for x in recall_schema)
        threshold = best * float(getattr(settings, "candidate_cliff_min_ratio", 0.15))
        return [x for x in recall_schema if float(x.get("相关性分数") or 0.0) >= threshold]

    @staticmethod
    def _format_schema_prompt(schema_prompt: str, top20_candidates: list[dict] | list[str]) -> str:
        if top20_candidates and isinstance(top20_candidates[0], dict):
            lines = []
            for col in top20_candidates:
                lines.append(
                    f'- 列名：{col.get("列名", "")} | 相关性分数：{col.get("相关性分数", "")} | 列描述：{col.get("列描述", "")} | 字段类型：{col.get("字段类型", "")} | 是否枚举：{col.get("是否枚举", "")}'
                )
                if col.get("空值率") not in (None, ""):
                    lines.append(f'  空值率：{col.get("空值率")}')
                if col.get("唯一值数") not in (None, ""):
                    lines.append(f'  唯一值数：{col.get("唯一值数")}')
                if col.get("示例值") not in (None, ""):
                    lines.append(f'  示例值：{col.get("示例值")}')
                if col.get("格式") not in (None, ""):
                    lines.append(f'  格式：{col.get("格式")}')
                if col.get("范围") not in (None, ""):
                    lines.append(f'  范围：{col.get("范围")}')
            return "\n".join(lines)
        if isinstance(valid_columns, list):
            return json.dumps(valid_columns, ensure_ascii=False, indent=2)
        return str(schema_prompt)

    @staticmethod
    def resolve_tie(tie_candidates: list[dict]) -> dict:
        priority_order = ["json_sql", "icl", "direct", "plan"]
        for priority in priority_order:
            for cand in tie_candidates:
                if priority in cand.get("type", "").lower():
                    return cand
        return tie_candidates[0]
