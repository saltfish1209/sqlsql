import asyncio
from generator import SQLGenerator
from openai import AsyncOpenAI
from utils import TokenTracker


def is_meaningful_result(result):
    """检查执行结果是否包含实际内容而非全是 None"""
    if not result or len(result) == 0:
        return False
    # 检查第一行结果中是否有任何一个字段不是 None
    return any(val is not None for val in result[0])

class SQLRefiner:
    def __init__(self, base_url, api_key, db_engine, model="qwen3-14b"):
        # 独立维护一个异步客户端
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.db = db_engine
        self.model = model


    async def refine_async(self, question, schema_prompt, candidates,all_valid_columns,max_retries=2):
        """
        对候选 SQL 进行执行检查和修正。
        """
        refined_candidates = []
        retry_tasks = []

        # 1. 先进行一轮“预检”：看看有没有天生就成功的 SQL
        for cand in candidates:
            result, error = self.db.execute_sql(cand['sql'])
            if error is None and result:
                cand['status'] = 'success'
                cand['result'] = result
                refined_candidates.append(cand)
            else:
                cand['error_msg'] = error if error else "Execution returned empty result."
                retry_tasks.append(cand)

        if not retry_tasks:
            return refined_candidates

        print(f">>> 启动异步修正，需修复数量: {len(retry_tasks)}")

        tasks = [
            self._repair_worker(question, schema_prompt, cand, all_valid_columns, max_retries)
            for cand in retry_tasks
        ]

        repaired_results = await asyncio.gather(*tasks)
        refined_candidates.extend(repaired_results)

        return refined_candidates

    async def _repair_worker(self, question, schema_prompt, cand, valid_cols, max_retries):
        """单个候选 SQL 的多轮修复逻辑"""
        current_sql = cand['sql']
        current_error = cand['error_msg']

        for i in range(max_retries):
            valid_cols_str = ",".join(valid_cols)
            prompt = f"""
            你是 SQLite 修复专家。请根据反馈修正 SQL。

            【数据库 Schema】
            {schema_prompt}

            【合法列名列表】
            {valid_cols_str}

            【用户问题】
            {question}

            【错误 SQL】
            {current_sql}

            【执行反馈】
            {current_error}

            【修复规则】
            1. 修正列名拼写或引用错误 (no such column)。
            2. 如果结果为空，尝试将 '=' 改为 'LIKE'。
            3. 如果 SQL 语法不完整，请补全为标准 SELECT-FROM-WHERE 结构。
            3. 只返回 SQL 代码。**
            """

            content = ""  # 【修复】预定义变量，防止 NameError

            try:
                # 异步调用 LLM
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01
                )
                content = resp.choices[0].message.content

                # 调用 Generator 的静态方法清洗 SQL
                # 确保你已经在文件头部 import 了 SQLGenerator
                fixed_sql = SQLGenerator._extract_sql(content)

                # 立即验证
                result, error = self.db.execute_sql(fixed_sql)
                if error is None and is_meaningful_result(result):
                    print(f"    >>> [Refiner] {cand['type']} 在第 {i + 1} 次修正成功。")
                    return {
                        "type": f"{cand['type']}_Refined_{i + 1}",
                        "sql": fixed_sql,
                        "status": "success",
                        "result": result
                    }
                else:
                    # 更新状态继续下一轮
                    current_sql = fixed_sql
                    current_error = error if error else "Fixed SQL still returns empty."
            except Exception as e:
                print(f"    >>> [Refiner] 修复调用异常: {e}")
                # 发生异常也不要退出循环，或者可以选择 break
                # break

        # 耗尽重试次数仍失败，或者发生异常跳出循环后
        # 【关键修复】确保返回一个字典结构，而不是 None
        return {
            "type": cand['type'],
            "sql": current_sql,
            "status": "failed",
            "error_msg": current_error,  # 统一字段名
            "result": None
        }

    async def resolve_tie_async(self, question, schema_prompt, tie_candidates, token_tracker):
        """处理平票：LLM裁判"""
        cand_a, cand_b = tie_candidates[0], tie_candidates[1]

        prompt = f"""你是一名高级数据分析师。针对同一个问题，现有两个不同的SQL查询结果，请判断哪个是正确的。
        [Schema] {schema_prompt}
        [用户问题] {question}

        [方案 A] SQL: {cand_a['sql']}
        结果样例: {str(cand_a['result'])[:200]}...

        [方案 B] SQL: {cand_b['sql']}
        结果样例: {str(cand_b['result'])[:200]}...

        请分析差异，输出最终正确的SQL。只输出SQL代码。"""

        try:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            token_tracker.track(resp)  # 统计

            final_sql = SQLGenerator._extract_sql(resp.choices[0].message.content)
            result, error = self.db.execute_sql(final_sql)
            status = "success" if error is None else "failed"
            return {"type": "Tie_Resolver", "sql": final_sql, "result": result, "status": status, "error": error}
        except Exception:
            return cand_a  # 兜底