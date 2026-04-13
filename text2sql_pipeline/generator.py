import re
import asyncio
import os
import pandas as pd
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
from utils import TokenTracker


class SQLGenerator:
    def __init__(self, base_url,api_key, model="qwen3-14b"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
        self.qa_template_df = pd.read_csv(os.path.join(data_dir, "train_dataset_with_sql_template.csv"))
        self.embed_model = SentenceTransformer(
            os.getenv("EMBED_MODEL_PATH", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )

        self.template_embs = self.embed_model.encode(
            self.qa_template_df['问题模版'].tolist(),
            convert_to_tensor=True
        )

    def _format_evidence(self, evidence_dict):
        """
        将 B 路和 C 路匹配到的“证据”转成文字，消除 AI 的歧义 [cite: 225]
        """
        parts = []
        # B 路：精准匹配
        if evidence_dict.get("exact_matches"):
            for val, cols in evidence_dict["exact_matches"].items():
                parts.append(f"值 '{val}' 在列 {cols} 中被精准发现。")
        # C 路：模糊匹配 (LSH)
        if evidence_dict.get("fuzzy_matches"):
            for kw, hits in evidence_dict["fuzzy_matches"].items():
                for col, vals in hits.items():
                    parts.append(f"关键词 '{kw}' 与列 '{col}' 中的这些值相似: {vals}。")
        return "\n".join(parts) if parts else "无直接数据库值参考。"

    def _get_top_3_from_log(self, question):
        """向量匹配最相似的3个模板"""
        q_emb = self.embed_model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.template_embs)[0]
        top_results = scores.topk(k=3)

        example_str = ""
        for idx in top_results.indices:
            row = self.qa_template_df.iloc[idx.item()]
            example_str += f"问题: {row['问题模版']}\nSQL: {row['SQL模版']}\n\n"
        return example_str

    @staticmethod
    def _extract_sql(text):
        """
        静态工具方法：清洗并提取 SQL
        """
        if not text:
            return ""

        # 1. 去除 <think> 标签 (正则复用)
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. 提取 Markdown ```sql
        match = re.search(r"```sql\s*(.*?)\s*```", text_clean, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 3. 提取通用 Markdown ```
        match_general = re.search(r"```\s*(.*?)\s*```", text_clean, re.DOTALL)
        if match_general:
             candidate = match_general.group(1).strip()
             if candidate.upper().startswith("SELECT"):
                 return candidate

        # 4. 兜底 SELECT
        sql_match = re.search(r"(SELECT\s+.*)", text_clean, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        return text_clean

    def build_m_schema_prompt(self, selected_columns, all_metadata, table_name = "procurement_table"):
        """
        根据 Schema Linking 选出的列，构建精简版的 M-Schema 文本。
        """
        schema_text = f"[DB_ID] procurement_db\n[Schema]\n# Table: {table_name}\n[\n"
        for col_name in selected_columns:
            # 查找元数据中的描述和类型
            meta = next((item for item in all_metadata if item["column_name"] == col_name), None)
            if meta:
                desc = meta.get('column_description', '')
                dtype = meta.get('data_type', 'TEXT')
                # M-Schema 格式
                schema_text += f"  ({col_name}:{dtype}, {desc})\n"
        schema_text += "]\n"
        return schema_text

    async def generate_candidates_async(self, question, schema_prompt, entities, evidence_dict,token_tracker,num_per_path=2):
        """
        主函数：利用 n 参数并行生成两条路径的候选
        """
        evidence_str = self._format_evidence(evidence_dict)
        candidates = []


        # --- 路径 A: Thinking 模式 (利用流式+ n 参数) ---
        thinking_prompt = f"""你是一名SQL专家。请结合Schema和检测到的证据，通过深度思考生成SQL。
        采用sqlite，不需要加上数据库名，直接使用对应表名即可。
        
        [Schema]
        {schema_prompt}
        [数据库证据]
        {evidence_str}
        [用户问题]
        {question}
        请务必先输出思考过程,请输出高质量SQL，用```sql ... ```包裹,不需要解释和输出其他内容。
        /think"""


        # --- 路径 B: ICL 模式 (利用流式+ n 参数) ---
        examples = self._get_top_3_from_log(question)
        icl_prompt = f"""你是一名SQL专家。请参考以下相似案例生成SQL。
        采用sqlite，不需要加上数据库名，直接使用对应表名即可。
        
        [相似案例]
        {examples}
        [Schema]
        {schema_prompt}
        [数据库证据]
        {evidence_str}
        [用户问题]
        {question}
        请输出SQL，用```sql ... ```包裹，不需要解释和输出其他内容。
        /no_think  """

        # --- 路径 C: 直接生成 模式---
        direct_prompt = f"""你是一名SQL专家。请参考以下内容生成SQL。
        采用sqlite，不需要加上数据库名，直接使用对应表名即可。
        
        [Schema]
        {schema_prompt}
        [数据库证据]
        {evidence_str}
        [用户问题]
        {question}
        请输出SQL，用```sql ... ```包裹，不需要解释和输出其他内容。
        /no_think
        """


        # 创建任务列表
        tasks = []

        # 提交 Path A 任务 (启用 temperature 增加多样性)
        for _ in range(num_per_path):
            tasks.append(self._call_llm_async(thinking_prompt, "thinking_path", token_tracker, temperature=0.5))

        # 提交 Path B 任务 (低温精确匹配)
        for _ in range(num_per_path):
            tasks.append(self._call_llm_async(icl_prompt , "ICL_Path",  token_tracker, temperature=0.1))
        # 提交 Path C 任务 直接生成
        for _ in range(num_per_path):
            tasks.append(self._call_llm_async(direct_prompt, "Direct_Path", token_tracker, temperature=0.0))

        # 并发执行 (vLLM 会自动 batch 处理)
        results = await asyncio.gather(*tasks)

        # 过滤无效结果
        return [r for r in results if r is not None]

    async def _call_llm_async(self, prompt,path_type, token_tracker,temperature=0.0):
        try:
            # 异步调用，不使用 extra_body
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,  # 给足空间让它思考
                stream=False  # 高并发下建议关闭 stream，降低客户端处理开销
            )
            token_tracker.track(response)
            content = response.choices[0].message.content
            sql = self._extract_sql(content)

            if sql:
                return {"type": path_type, "sql": sql, "raw_content": content}  # 保留原始内容用于调试
            return None

        except Exception as e:
            print(f">>> [Async] 生成失败: {e}")
            return None
