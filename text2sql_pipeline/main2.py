import json
import os
import re
import time
import asyncio
from openai import AsyncOpenAI
from similarity2 import Similarity
from generator import SQLGenerator
from refiner import SQLRefiner
from selector import SQLSelector
from db_env import DBEngine
from utils import to_halfwidth , TokenTracker


DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"


# 配置信息
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
VLLM_BASE_URL = "http://localhost:8000/v1"  # 确保这里是 vLLM 地址
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "qwen3-14B"  # 必须与 vLLM 启动参数 --served-model-name 一致

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
CSV_PATH = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")
SCHEMA_PATH = os.path.join(DATA_DIR, "m_schema.txt")



def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)




class XiYanSQLSystem:
    def __init__(self):
        """
        初始化系统组件：对应论文中的各个 Agent
        """
        print(">>> [System Init] 正在初始化 XiYan-SQL 各个组件...")
        self.client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                self.schema_text = f.read()

        # 1. 数据库环境 (用于 Refiner 执行反馈)
        self.db_engine = DBEngine(CSV_PATH)

        # 2. 模式链接模块 (Schema Linking)
        self.sim_engine = Similarity(SCHEMA_PATH, CSV_PATH)

        # 3. 生成器 (Generator - Multi-path)
        self.generator = SQLGenerator(VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME)

        # 4. 精炼器 (Refiner - Self-correction)
        self.refiner = SQLRefiner(VLLM_BASE_URL, VLLM_API_KEY, self.db_engine, MODEL_NAME)

        # 5. 选择器 (Selector)
        self.selector = SQLSelector()
        print(">>> [System Init] 初始化完成。\n")

    async def extract_entities_mock(self, question,schema_text=None,tracker = None):
        # 1. 定义 Prompt (请确保这一段没有被删掉)
        target_schema = schema_text if schema_text else self.schema_text
        prompt = f"""
        你是一个电力物资采购数据库专家。请严格从用户问题中提取**完整的、不可分割的业务实体值**，用于后续 SQL 查询。

        【数据库 Schema 摘要】
        {target_schema}

        [规则]
        1. 提取用于 SQL WHERE 条件的具体值（如批次号、物料码、项目名）。
        2. 保持原文拼写，不要拆分复合词。
        3. 必须输出标准的 JSON 字符串数组格式。

        [Few-Shot 示例]
        User: 国家电网2022年第七十二批采购(输变电项目)有哪些？
        Assistant: ["国家电网2022年第七十二批采购(输变电项目)"]

        User: 500061873物料的中标单位是谁？
        Assistant: ["500061873"]

        User: 查询协议库存可视化选购20230407的详情
        Assistant: ["协议库存可视化选购20230407"]

        [当前任务]
        User: {question}
        Assistant:
        """

        try:
            resp = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01
            )
            tracker.track(resp)
            content = resp.choices[0].message.content.strip()

            # --- 清洗逻辑 ---
            # 1. 去除 <think> 标签
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            # 2. 提取列表字符串
            try:
                # 尝试标准 JSON 解析
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"警告: JSON 解析失败，尝试正则修复: {content}")

                # 【优化点 3】更强的正则提取
                # 解释：匹配引号内的内容，非贪婪模式，忽略转义符
                # 专门处理像 "value"" 这种末尾多引号的情况
                matches = re.findall(r'"([^"]+?)"', content)

                # 如果是单引号的情况
                if not matches:
                    matches = re.findall(r"'([^']+?)'", content)

                # 再次清洗：去掉可能残留的边缘符号
                clean_entities = [m.strip() for m in matches if m.strip()]
                return clean_entities

        except Exception as e:
            print(f"实体提取严重错误: {e}")
            return []



    async def run_pipeline_async(self, question):
        start_time = time.time()  # 1. 记录开始时间
        tracker = TokenTracker()

        question = to_halfwidth(question)


        debug_print(f">>> [Schema Pruning] 正在使用 BERT 检索 Top-20 相关列...")

        # 1. 调用 similarity2.py 中已有的方法，利用 CrossEncoder 计算语义相似度
        # 这会返回一个列名列表，例如 ['project_name', 'material_code', ...]
        bert_top_cols = self.sim_engine.retrieve_schema_by_question_only(question, top_k=20)

        # 2. 将列名列表转换为 M-Schema 文本格式
        # 这一步非常重要，因为 LLM 需要看到类型和描述才能判断实体
        lite_schema_text = self.generator.build_m_schema_prompt(
            bert_top_cols,
            self.sim_engine.column_metadata,
            table_name="procurement_table"
        )

        # 打印一下看看效果 (调试用)
        # debug_print(f"精简 Schema 内容:\n{lite_schema_text}")


        # Step 1 : 实体提取(Async)
        entities = await self.extract_entities_mock(question,schema_text=lite_schema_text,tracker = tracker)
        print(f">>> 提取实体: {entities}")


        # Step 1.2 混合召回 (BERT + 精准匹配)
        ranked_tuples, must_have_set, evidence_details = self.sim_engine.hybrid_retrieve(question, entities)
        full_ranked_list = [x[0] for x in ranked_tuples]

        # 尝试第一梯队 (Top 1-20)
        top_20_cols = list(set(full_ranked_list[:20]) | must_have_set)
        print(f">>> 尝试第一梯队: {top_20_cols}")

        # Step 1.3: 执行流程 (Async)
        best_cand = await self._try_execute_flow_async(question, top_20_cols, entities,tracker)

        # 如果第一梯队失败，尝试第二梯队 (Top 20-40)
        if not best_cand or best_cand.get('status') != 'success':
            print(">>> [Fallback] 第一梯队失败，尝试 20-40 列...")
            top_20_40_cols = list(set(full_ranked_list[20:40]) | must_have_set)
            best_cand = await self._try_execute_flow_async(question, top_20_40_cols, entities,tracker)

        # 2. 计算总耗时
        total_time = time.time() - start_time

        final_res = best_cand.get('result')

        # 初始化默认值
        unique_rows_list = []
        unique_count = 0
        if final_res:
            # 使用 set 去重
            unique_rows_set = set(tuple(row) for row in final_res)
            unique_count = len(unique_rows_set)
            # 【重要】将去重后的 set 转回 list，以便输出
            unique_rows_list = list(unique_rows_set)

        # 3. 封装返回字典
        return {
            "final_sql": best_cand.get('sql', 'SELECT 1'),
            "execution_result": unique_rows_list if final_res else final_res,
            "unique_rows_count": unique_count,
            "reason": best_cand.get('status', 'success'),
            "cost_time": total_time,
            "token_usage": tracker.get_report()
        }

    # main2.py 修改 _try_execute_flow 方法
    async def _try_execute_flow_async(self, question, cols, entities,tracker):
        # 1. 重新获取包含证据的检索结果
        _, _, evidence_dict = self.sim_engine.hybrid_retrieve(question, entities)

        # 2. 构建 Schema (注意传入表名)
        m_schema = self.generator.build_m_schema_prompt(cols, self.sim_engine.column_metadata)

        # 3. 异步生成 (Async Generator)
        print(">>> [Async] 正在并发生成 SQL...")
        raw_cands = await self.generator.generate_candidates_async(question, m_schema, entities, evidence_dict,tracker)


        print(f"\n--- [DEBUG] Generator 生成的原始候选 SQL ({len(raw_cands)}个) ---")
        for i, cand in enumerate(raw_cands):
            print(f"[{i}] Type: {cand.get('type')} | SQL: {cand.get('sql')}")
        print("----------------------------------------------------------\n")


        # 4. 异步修正 (Async Refiner)
        print(f">>> [Async] 生成结束，获得 {len(raw_cands)} 个候选。开始修正...")
        refined_cands = await self.refiner.refine_async(question, m_schema, raw_cands, self.sim_engine.column_names,tracker)
        print(f"\n--- [DEBUG] Refiner 修正后的候选 SQL ---")
        for i, cand in enumerate(refined_cands):
            print(f"[{i}] Status: {cand.get('status')} | SQL: {cand.get('sql')}")
        print("----------------------------------------------------------\n")

        # 5. 选择 (Selector)

        selected_res, reason, status = self.selector.select_best(question, m_schema, refined_cands)
        if status == "tie":
            print(f">>> [System] 触发平票仲裁: {reason}")
            # selected_res 在 tie 状态下是一个包含两个候选的 list
            final_cand = await self.refiner.resolve_tie_async(question, m_schema, selected_res, tracker)
            return final_cand

        print(f">>> 选择结果: {reason}")

        return selected_res

if __name__ == '__main__':
    # 1. 实例化系统 (加载一次资源)
    xiyan_system = XiYanSQLSystem()

    # 2. 定义测试问题
    test_questions = [
        "协议库存可视化选购20230407”批次的采购实施模式是怎样的？"
    ]

    # 3. 循环运行 (模拟批量测试)
    loop = asyncio.get_event_loop()
    for q in test_questions:
        print("\n" + "=" * 60)
        result = loop.run_until_complete(xiyan_system.run_pipeline_async(q))
        print(f"FINAL OUTPUT (Time: {result['cost_time']:.2f}s)")
        print(f"SQL: {result['final_sql']}")
        tokens = result.get('token_usage', {})
        print(f"Token Usage: Input={tokens.get('input_tokens', 0)} | "
              f"Output={tokens.get('output_tokens', 0)} | "
              f"Total={tokens.get('total_tokens', 0)}")
        print(f"Unique Rows Count: {result['unique_rows_count']}")
        print(f"Result (First 10 rows): {result['execution_result'][:10] if result['execution_result'] else 'Empty'}")
        print("=" * 60)
