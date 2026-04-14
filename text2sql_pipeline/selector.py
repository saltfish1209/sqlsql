class SQLSelector:
    def __init__(self):
        # 轻量化模式下，我们甚至不需要调用 LLM 即可完成筛选
        pass

    def select_best(self, question, schema_prompt, candidates):
        """
        核心逻辑：一致性检验 (Self-Consistency) [cite: 87]
        1. 按照 SQL 执行结果对候选进行分组 
        2. 票数最多的结果即为最终选择
        """
        # 1. 过滤：只保留执行成功且有返回数据的候选 SQL 
        valid_candidates = [
            c for c in candidates
            if c.get('status') == 'success' and any(any(v is not None for v in row) for row in c.get('result', []))
        ]
        # 兜底逻辑：如果没有一个 SQL 执行成功
        if not valid_candidates:
            reason = "所有路径均执行失败或结果为空"
            # 即使失败也要返回 (结果, 理由)，防止 main2.py 报错
            return (candidates[0] if candidates else None), reason, "failed"

        # 2. 统计逻辑：将执行结果转为字符串作为 Key 进行归类 
        result_groups = {}
        for cand in valid_candidates:
            # 使用结果内容的字符串形式作为标识
            res_key = str(cand['result']) 
            if res_key not in result_groups:
                result_groups[res_key] = []
            result_groups[res_key].append(cand)

        sorted_groups = sorted(result_groups.items(), key=lambda x: len(x[1]), reverse=True)

        # 真正处理平票
        if len(sorted_groups) > 1 and len(sorted_groups[0][1]) == len(sorted_groups[1][1]):
            top_groups = sorted_groups[:2]
            tie_candidates = [top_groups[0][1][0], top_groups[1][1][0]]
            reason = f"出现平票：两组结果各获得 {len(top_groups[0][1])} 票"
            return tie_candidates, reason, "tie"

        winning_group = sorted_groups[0][1]
        vote_count = len(winning_group)

        best_cand = winning_group[0]
        for cand in winning_group:
            if "Thinking" in cand.get('type', ''):
                best_cand = cand
                break

        reason = f"一致性投票通过：该结果获得 {vote_count} 票。"
        return best_cand, reason, "success"