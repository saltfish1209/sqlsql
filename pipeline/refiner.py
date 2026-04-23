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
from collections import Counter

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.generator import SQLGenerator
from pipeline.utils import debug_print, TokenTracker


def _result_key(result) -> str:
    """把执行结果折叠成可哈希的字符串键，用于多路投票一致性比较。"""
    if result is None:
        return "<NONE>"
    try:
        # 归一化：顺序不同但集合相同也视作一致
        return str(sorted([tuple(r) for r in result]))
    except Exception:
        return str(result)


def _majority_agreed(successful: list[dict]) -> bool:
    """至少两路成功且存在一对结果完全一致，则投票已锁定。"""
    if len(successful) < 2:
        return False
    counts = Counter(_result_key(c.get("result")) for c in successful)
    return max(counts.values()) >= 2


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
            if error is not None:
                cand["error_msg"] = error
                need_repair.append(cand)
                continue
            # 执行成功：即便结果为空或 NULL 也视为合法业务结果，不再算运行错误。
            # 仅当 WHERE 字面量确实与列不匹配时，才进入修复（改 LIKE）。
            literal_issues = self._check_literals(cand["sql"]) if result else []
            if literal_issues:
                cand["error_msg"] = f"LITERAL_MISMATCH: {literal_issues}"
                need_repair.append(cand)
            else:
                cand["status"] = "success"
                cand["result"] = result if result is not None else []
                refined.append(cand)

        if not need_repair:
            return refined

        # 早停：按投票一致性，若已有两路成功且结果一致，
        # 无论剩余路是否出错都不再修复——第三路正确与否不影响多数票。
        if _majority_agreed(refined):
            debug_print(
                f"[Refiner] 已有 {len(refined)} 路结果一致，跳过 {len(need_repair)} 个修复任务"
            )
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
                f"直接输出修复后的 SQL，用```sql ... ```包裹，不要思考过程。"
            )
            try:
                extra_body: dict = {}
                if not settings.enable_thinking_for_refiner:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                create_kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=settings.refiner_temperature,
                    max_tokens=settings.refiner_max_tokens,
                    extra_body=extra_body or None,
                )
                if settings.refiner_enforce_timeout:
                    create_kwargs["timeout"] = settings.llm_request_timeout_sec
                resp = await self.client.chat.completions.create(**create_kwargs)
                tracker.track(resp)
                content = resp.choices[0].message.content
                debug_print(f"[Refiner][raw][{cand['type']}] {content!r}")
                fixed_sql = SQLGenerator.extract_sql(content)

                result, error = self.db.execute_sql(fixed_sql)
                repair_times.append(time.time() - t_repair_start)
                # 修复后只要无执行错误即接受（空结果 / NULL 均视为合法业务结果）
                if error is None:
                    debug_print(f"[Refiner] {cand['type']} 第{i+1}次修正成功")
                    return {
                        "type": f"{cand['type']}_Refined_{i + 1}",
                        "sql": fixed_sql,
                        "status": "success",
                        "result": result if result is not None else [],
                        "repair_times": repair_times,
                    }
                current_sql = fixed_sql
                current_error = error
            except APIConnectionError as e:
                repair_times.append(time.time() - t_repair_start)
                base_url = str(getattr(self.client, "base_url", "") or "?")
                print(
                    f"[Refiner][FATAL] 无法连接 LLM 服务 "
                    f"(base_url={base_url}, model={self.model}): {e}"
                )
                break
            except Exception as e:
                repair_times.append(time.time() - t_repair_start)
                debug_print(f"[Refiner] 修复异常: {type(e).__name__}: {e}")

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
