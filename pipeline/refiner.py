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
        question: str,
        schema_prompt: str,
        candidates: list[dict],
        valid_columns: list[str],
        tracker: TokenTracker,
        max_retries: int | None = None,
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
                repaired = await self._attempt_llm_repair(
                    question=question,
                    schema_prompt=schema_prompt,
                    candidate=repaired,
                    valid_columns=valid_columns,
                    tracker=tracker,
                    max_retries=max_retries or settings.max_repair_retries,
                )
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

    async def _attempt_llm_repair(
        self,
        question: str,
        schema_prompt: str,
        candidate: dict,
        valid_columns: list[dict] | list[str],
        tracker: TokenTracker,
        max_retries: int,
    ) -> dict:
        sql = candidate.get("sql") or ""
        prompt_schema = self._format_schema_prompt(schema_prompt, valid_columns)
        prompt = (
            "你是SQL修复专家。请根据错误SQL、错误原因和精简schema重新生成正确SQL。\n"
            f"[问题]\n{question}\n"
            f"[错误SQL]\n{sql}\n"
            f"[错误原因]\n{candidate.get('error_msg', '')}\n"
            f"[精简schema]\n{prompt_schema}\n"
            "要求：只输出可执行SQL，用```sql包裹。"
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

    @staticmethod
    def _format_schema_prompt(schema_prompt: str, valid_columns: list[dict] | list[str]) -> str:
        if valid_columns and isinstance(valid_columns[0], dict):
            lines = []
            for col in valid_columns:
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
