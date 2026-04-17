"""
SQL 选择器 —— 自洽性投票 (Self-Consistency) 选出最优 SQL。
───────────────────────────────────────────────────────────────
1. 过滤出执行成功且有实际返回数据的候选
2. 按执行结果分组，多数票胜出
3. 平票时返回 tie 标记，由 Refiner.resolve_tie 按路径优先级决策
"""
from __future__ import annotations


class SQLSelector:
    def select_best(
        self,
        question: str,
        schema_prompt: str,
        candidates: list[dict],
    ) -> tuple:
        """
        返回 (best_cand_or_tie_list, reason_str, status)。
        status 取值: "success" | "tie" | "failed"
        """
        valid = [
            c for c in candidates
            if c.get("status") == "success"
            and any(
                any(v is not None for v in row)
                for row in c.get("result", [])
            )
        ]

        if not valid:
            reason = "所有路径均执行失败或结果为空"
            return (candidates[0] if candidates else None), reason, "failed"

        groups: dict[str, list[dict]] = {}
        for cand in valid:
            key = str(cand["result"])
            groups.setdefault(key, []).append(cand)

        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

        # 平票检测
        if (
            len(sorted_groups) > 1
            and len(sorted_groups[0][1]) == len(sorted_groups[1][1])
        ):
            top2 = sorted_groups[:2]
            tie_cands = [top2[0][1][0], top2[1][1][0]]
            reason = f"出现平票：两组结果各获得 {len(top2[0][1])} 票"
            return tie_cands, reason, "tie"

        winning = sorted_groups[0][1]
        votes = len(winning)

        best = winning[0]
        for cand in winning:
            if "thinking" in cand.get("type", "").lower():
                best = cand
                break

        reason = f"一致性投票通过：该结果获得 {votes} 票。"
        return best, reason, "success"
