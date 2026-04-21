"""
SQL 精炼器 —— 执行反馈驱动的多轮自修正 + Literal-Column 校验。
───────────────────────────────────────────────────────────────────
1. 预检：直接执行候选 SQL，成功即跳过修复
2. 修复循环：每轮把错误反馈注入 Prompt，让 LLM 修正
3. Literal-Column 校验（论文新增）：
   检查 WHERE 中的字面量是否真实存在于对应列，
   若不存在则提示 LLM 修正为 LIKE 或替换值

项目特色：平票按路径优先级决策 (thinking > ICL > direct)，不调用 LLM。
"""
from __future__ import annotations

import asyncio
import re
import time

from openai import AsyncOpenAI

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.generator import SQLGenerator
from pipeline.utils import debug_print, TokenTracker


def _is_meaningful(result) -> bool:
    if not result or len(result) == 0:
        return False
    return any(val is not None for val in result[0])


def _extract_where_literals(sql: str) -> list[tuple[str, str]]:
    """从 SQL 中提取 WHERE 子句中的 (列名, 字面量) 对。"""
    pairs = []
    # "col" = 'val'  或  col = 'val'
    for m in re.finditer(
        r'"?([^"=\s]+)"?\s*=\s*\'([^\']+)\'', sql
    ):
        pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


class SQLRefiner:
    def __init__(self, client: AsyncOpenAI, model: str, db: DBEngine):
        self.client = client
        self.model = model
        self.db = db

    async def refine_async(
        self,
        question: str,
        schema_prompt: str,
        candidates: list[dict],
        valid_columns: list[str],
        tracker: TokenTracker,
        max_retries: int | None = None,
    ) -> list[dict]:
        retries = max_retries or settings.max_repair_retries
        refined = []
        need_repair = []

        for cand in candidates:
            result, error = self.db.execute_sql(cand["sql"])
            if error is None and result:
                # Literal-Column 校验
                literal_issues = self._check_literals(cand["sql"])
                if literal_issues:
                    cand["error_msg"] = f"LITERAL_MISMATCH: {literal_issues}"
                    need_repair.append(cand)
                else:
                    cand["status"] = "success"
                    cand["result"] = result
                    refined.append(cand)
            else:
                cand["error_msg"] = error or "Execution returned empty result."
                need_repair.append(cand)

        if not need_repair:
            return refined

        debug_print(f"[Refiner] 启动修正，需修复: {len(need_repair)} 个")
        tasks = [
            self._repair_worker(question, schema_prompt, c, valid_columns, tracker, retries)
            for c in need_repair
        ]
        repaired = await asyncio.gather(*tasks)
        refined.extend(repaired)
        return refined

    async def _repair_worker(
        self,
        question: str,
        schema_prompt: str,
        cand: dict,
        valid_cols: list[str],
        tracker: TokenTracker,
        max_retries: int,
    ) -> dict:
        current_sql = cand["sql"]
        current_error = cand["error_msg"]
        valid_cols_str = ", ".join(valid_cols)
        repair_times: list[float] = []

        for i in range(max_retries):
            t_repair_start = time.time()
            system_msg = "你是 SQLite 修复专家。只输出修复后的 SQL，不要输出任何解释或其他内容。"
            user_msg = (
                f"【Schema】\n{schema_prompt}\n"
                f"【合法列名】{valid_cols_str}\n"
                f"【问题】{question}\n"
                f"【错误SQL】\n{current_sql}\n"
                f"【执行反馈】{current_error}\n"
                f"【修复规则】\n"
                f"1. 修正列名错误 (no such column)，只能使用合法列名列表中的列。\n"
                f"2. 结果为空时将 '=' 改为 'LIKE'。\n"
                f"3. 如果字面量不存在于该列，使用 LIKE '%关键词%' 替代精确匹配。\n"
                f"4. 补全不完整的 SELECT-FROM-WHERE 结构。\n"
                f"SQL:"
            )
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=settings.refiner_temperature,
                )
                tracker.track(resp)
                content = resp.choices[0].message.content
                fixed_sql = SQLGenerator.extract_sql(content)

                result, error = self.db.execute_sql(fixed_sql)
                repair_times.append(time.time() - t_repair_start)
                if error is None and _is_meaningful(result):
                    debug_print(f"[Refiner] {cand['type']} 第{i+1}次修正成功")
                    return {
                        "type": f"{cand['type']}_Refined_{i + 1}",
                        "sql": fixed_sql,
                        "status": "success",
                        "result": result,
                        "repair_times": repair_times,
                    }
                current_sql = fixed_sql
                current_error = error or "Fixed SQL still returns empty."
            except Exception as e:
                repair_times.append(time.time() - t_repair_start)
                debug_print(f"[Refiner] 修复异常: {e}")

        return {
            "type": cand["type"],
            "sql": current_sql,
            "status": "failed",
            "error_msg": current_error,
            "result": None,
            "repair_times": repair_times,
        }

    def _check_literals(self, sql: str) -> list[str]:
        """Literal-Column 校验：检查 WHERE 中的字面量是否存在于对应列。"""
        issues = []
        for col, literal in _extract_where_literals(sql):
            if not self.db.check_literal_in_column(col, literal):
                issues.append(f"'{literal}' 不存在于列 '{col}'")
        return issues

    @staticmethod
    def resolve_tie(tie_candidates: list[dict]) -> dict:
        """平票决策：thinking > ICL > direct，无需 LLM 调用。"""
        priority_order = ["thinking", "icl", "direct"]
        for priority in priority_order:
            for cand in tie_candidates:
                if priority in cand.get("type", "").lower():
                    return cand
        return tie_candidates[0]
