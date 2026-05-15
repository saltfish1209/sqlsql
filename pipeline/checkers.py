"""
Deterministic SQL checker chain inspired by DeepEye-SQL.

Each checker produces an explicit directive that the refiner can hand to the
LLM. The checker chain does not try to prove semantic correctness; it catches
high-signal, locally verifiable issues before release.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class CheckIssue:
    code: str
    message: str
    directive: str


_WHERE_LITERAL_RE = re.compile(r'"?([^"=\s]+)"?\s*=\s*\'([^\']+)\'')
_SELECT_STAR_RE = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
_ORDER_BY_RE = re.compile(
    r"\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_COL_RE = re.compile(r'"([^"]+)"')


class SQLCheckerChain:
    """Sequential Syntax -> Logic -> Quality checker chain."""

    def __init__(self, db):
        self.db = db

    def check(
        self,
        sql: str,
        *,
        result: list[tuple] | list[list] | None,
        execution_error: str | None,
    ) -> list[CheckIssue]:
        issues: list[CheckIssue] = []
        if execution_error:
            issues.append(
                CheckIssue(
                    "EXECUTION_ERROR",
                    execution_error,
                    "SQL 无法执行。优先修正语法、表名、列名或聚合结构错误。",
                )
            )
            return issues

        issues.extend(self._literal_issues(sql))
        if self._is_empty_result(result):
            issues.append(
                CheckIssue(
                    "EMPTY_RESULT",
                    "SQL 执行成功但结果为空。",
                    "重新检查 WHERE 条件和值链接；不要为了通过执行而删除用户问题中的必要条件。",
                )
            )
        if _SELECT_STAR_RE.search(sql or ""):
            issues.append(
                CheckIssue(
                    "SELECT_STAR",
                    "SQL 使用了 SELECT *。",
                    "把 SELECT * 改成用户问题真正需要的具体列。",
                )
            )
        missing_null_guards = self._order_by_missing_null_guards(sql)
        for col in missing_null_guards:
            issues.append(
                CheckIssue(
                    "NULL_GUARD_MISSING",
                    f"ORDER BY 列 '{col}' 缺少 NULL 过滤。",
                    f"为排序列 \"{col}\" 增加 `\"{col}\" IS NOT NULL` 过滤，避免 NULL 干扰排序。",
                )
            )
        return issues

    def _literal_issues(self, sql: str) -> list[CheckIssue]:
        issues: list[CheckIssue] = []
        for col, literal in _WHERE_LITERAL_RE.findall(sql or ""):
            if not self.db.check_literal_in_column(col.strip(), literal.strip()):
                issues.append(
                    CheckIssue(
                        "LITERAL_MISMATCH",
                        f"'{literal}' 不存在于列 '{col}'",
                        "修正字面量对应列；若用户给的是简称，可改为 LIKE '%值%' 或换到证据提示的真实列。",
                    )
                )
        return issues

    @staticmethod
    def _is_empty_result(result: Any) -> bool:
        if result is None:
            return False
        if not result:
            return True
        for row in result:
            if not isinstance(row, (list, tuple)):
                if row is not None and str(row).strip() != "":
                    return False
                continue
            if any(v is not None and str(v).strip() != "" for v in row):
                return False
        return True

    @staticmethod
    def _order_by_missing_null_guards(sql: str) -> list[str]:
        match = _ORDER_BY_RE.search(sql or "")
        if not match:
            return []
        order_expr = match.group(1)
        columns = _QUOTED_COL_RE.findall(order_expr)
        missing: list[str] = []
        for col in columns:
            null_guard = re.search(
                rf'"{re.escape(col)}"\s+IS\s+NOT\s+NULL',
                sql,
                flags=re.IGNORECASE,
            )
            not_empty_guard = re.search(
                rf'"{re.escape(col)}"\s*!=\s*\'\'',
                sql,
                flags=re.IGNORECASE,
            )
            if not null_guard and not_empty_guard:
                missing.append(col)
            elif not null_guard and col not in missing:
                missing.append(col)
        return missing
