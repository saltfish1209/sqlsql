import os
import pickle
import re
import json
import time
import torch
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sentence_transformers import CrossEncoder


def _parse_m_schema(schema_text: str) -> list[dict]:
    """
    【新增】解析 M-Schema 文本，提取列名、描述和示例。
    目标格式: (列名: 类型, 描述, Examples: [e1, e2])
    """
    metadata = []
    # 使用正则匹配括号内的行
    # 假设格式为: (列名: 类型, 描述内容, Examples: [ex1, ex2])
    # 正则解释：
    # \s*\(          匹配开头的 (
    # ([^:]+)        捕获列名 (到冒号前)
    # :\s*([^,]+)    捕获类型 (到第一个逗号前)
    # ,\s*(.*?)      捕获描述 (非贪婪匹配到 Examples 前)
    # ,\s*Examples:\s*\[(.*?)\]  捕获示例列表
    pattern = re.compile(r'\s*\(([^:]+):\s*([^,]+),\s*(.*?),\s*Examples:\s*\[(.*?)\]\)')

    lines = schema_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line.startswith('('):
            continue

        match = pattern.search(line)
        if match:
            col_name = match.group(1).strip()
            data_type = match.group(2).strip()
            desc = match.group(3).strip()
            examples_str = match.group(4).strip()

            # 处理示例列表
            examples = [e.strip() for e in examples_str.split(',')] if examples_str else []

            metadata.append({
                "column_name": col_name,
                "data_type": data_type,
                "column_description": desc,
                "examples": examples,
                # 保存原始文本行用于 embedding，避免重新拼接
                "raw_text": line
            })

    print(f"成功解析 M-Schema，共识别出 {len(metadata)} 个列定义。")
    return metadata




class Similarity:
    def __init__(
        self,
        schema_path_or_text: str,
        csv_path: str,
        embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        :param schema_path_or_text: 可以是 .txt 文件的路径，也可以是 M-Schema 文本内容字符串
        :param csv_path: 原始数据路径
        """
        self.model_name = embed_model_name
        self.csv_path = csv_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rerank_model = os.path.join(current_dir, "my_schema_pruner_model")
        self.bert_ranker = CrossEncoder(rerank_model)

        # --- 修改点 1: 解析 M-Schema ---
        # 如果传入的是路径，读取文件；如果是文本，直接使用
        if os.path.exists(schema_path_or_text) and schema_path_or_text.endswith('.txt'):
            with open(schema_path_or_text, 'r', encoding='utf-8') as f:
                self.schema_text = f.read()
        else:
            self.schema_text = schema_path_or_text

        # 核心：将文本解析回结构化元数据，以便后续处理 CSV
        self.column_metadata = _parse_m_schema(self.schema_text)
        # -----------------------------

        self.column_names = [col['column_name'] for col in self.column_metadata]
        self.col_embeddings = None
        self.exact_index = {}
        self.lsh_index = {}

        # 设置缓存路径（同级目录）
        self.cache_dir = os.path.join(current_dir, "similarity_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"缓存目录: {self.cache_dir}")
        self.model = SentenceTransformer(embed_model_name)

        if not self._load_cache():
            print("开始构建索引...")
            self._build_schema_embeddings()
            self._build_value_indices()
            self._save_cache()
        else:
            print("索引加载成功")

    def _build_schema_embeddings(self):
        """
        修改：直接使用解析出的原始 M-Schema 行文本进行 Embedding，保持一致性
        """
        texts = [col['raw_text'] for col in self.column_metadata]
        # 如果 raw_text 解析有问题，可以用老的拼接逻辑作为 fallback

        self.col_embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        print(f"Schema Embedding 构建完成。")


    def _build_value_indices(self):
        """
        同时构建 精准索引 (B路) 和 LSH 索引 (C路)
        优化：一次遍历完成两种索引构建
        """
        df = pd.read_csv(self.csv_path, dtype=str)
        target_cols = [c['column_name'] for c in self.column_metadata if c['column_name'] in df.columns]

        for col in target_cols:
            # 判断列类型，决定索引策略
            # 简单启发式：如果是纯文本描述类，用 LSH；如果是ID/编码类，用 Exact Match；或者混合使用
            # 这里为了保险，对所有列都做 Exact，对长文本做 LSH

            # 初始化 LSH (针对当前列)
            lsh = MinHashLSH(threshold=0.75, num_perm=128)
            has_lsh_data = False

            unique_values = df[col].dropna().unique()

            for val in unique_values:
                val_str = str(val).strip()
                if not val_str: continue

                # --- B路：精准索引构建 (保留 -/\ 等符号) ---
                # 能够匹配 "2301-S-001" 或 "AB/123"
                if val_str not in self.exact_index:
                    self.exact_index[val_str] = set()
                self.exact_index[val_str].add(col)

                # --- C路：LSH 索引构建 (针对长文本/模糊匹配) ---
                # 只对长度 > 2 的值做 LSH，太短的没意义
                if len(val_str) >= 2:
                    # LSH 标准化：去除非字母数字，转大写 (为了抗 OCR 错误或拼写错误)
                    # 注意：Exact Match 不走这里
                    norm_val = re.sub(r'[^\w]', '', val_str).upper()
                    if norm_val:
                        mh = MinHash(num_perm=128)
                        for char in norm_val:
                            mh.update(char.encode('utf8'))
                        # Key 格式: "原始值" (查询回来后再校验)
                        lsh.insert(val_str, mh)
                        has_lsh_data = True

            if has_lsh_data:
                self.lsh_index[col] = lsh

        print(f"值索引构建完成。精准索引包含 {len(self.exact_index)} 个唯一值。")

    def _save_cache(self):
        """保存所有索引到磁盘"""
        # --- 修改点：再次确保目录存在 ---
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        # -----------------------------

        data = {
            "names": self.column_names,
            "embeddings": self.col_embeddings,
            "exact": self.exact_index,
            "lsh": self.lsh_index,
            # 获取 csv 文件的修改时间，确保 csv_path 是绝对路径或相对于执行路径正确
            "timestamp": os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else time.time()
        }

        save_path = os.path.join(self.cache_dir, "index.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"索引已缓存至: {save_path}")

    def _load_cache(self):
        """加载缓存"""
        cache_path = os.path.join(self.cache_dir, "index.pkl")
        if not os.path.exists(cache_path): return False

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            # 校验 CSV 文件是否修改过
            if data['timestamp'] != os.path.getmtime(self.csv_path):
                print("CSV文件已更新，缓存失效。")
                return False

            self.column_names = data['names']
            self.col_embeddings = data['embeddings']
            self.exact_index = data['exact']
            self.lsh_index = data['lsh']
            print("从缓存加载索引成功。")
            return True
        except Exception:
            return False

    def hybrid_retrieve(self, question: str, extracted_keywords: list[str], top_k_embed: int = 15):
        """
        核心方法：混合检索获取候选列
        :param question: 原始问题 (用于语义找 SELECT 列)
        :param extracted_keywords: LLM提取的实体值 (用于找 WHERE 列)
        :return: (candidates_list, debug_info)
        """
        candidates = set()
        debug_info = {"A_embed": [], "B_exact": [], "C_lsh": []}
        must_have_cols = set()
        evidence_map = {}


        # --- B路：精准匹配 (针对编码、ID、带符号的值) ---
        # 比如："2301-S-001", "±800kV"
        for kw in extracted_keywords:
            kw_clean = kw.strip()
            # 直接查倒排索引
            if kw_clean in self.exact_index:
                matched_cols = self.exact_index[kw_clean]
                candidates.update(matched_cols)
                debug_info["B_exact"].append((kw_clean, list(matched_cols)))
                must_have_cols.update(matched_cols)



        # --- C路：LSH 模糊匹配 (针对文本、拼写错误) ---
        # 比如："南瑞集团" (数据库里是"南瑞集团有限公司")
        for kw in extracted_keywords:
            # 只有当精准匹配没命中的时候，或者我们想扩大搜索范围时才跑 LSH
            # 简单的 LSH 查询逻辑
            norm_kw = re.sub(r'[^\w]', '', kw).upper()
            if not norm_kw: continue

            kw_mh = MinHash(num_perm=128)
            for char in norm_kw:
                kw_mh.update(char.encode('utf8'))

            # 遍历有 LSH 索引的列 (通常是文本列)
            for col, lsh in self.lsh_index.items():
                res = lsh.query(kw_mh)
                if res:
                    # LSH 返回的是近似值的列表，我们认为只要命中，该列就是候选列
                    # 这里可以加一步 Jaccard 校验，确保相似度真的很高
                    candidates.add(col)
                    must_have_cols.add(col)
                    debug_info["C_lsh"].append((kw, col, res[:3]))  # 记录匹配到了哪些值




        # --- A路：Embedding (针对结果列 / 语义列) ---
            # A路：BERT 排名
        all_col_metas = self.column_metadata
        pairs = [[question, col['column_description']] for col in all_col_metas]
        scores = self.bert_ranker.predict(pairs)

        # 变成 [(列名, 分数), ...] 的有序列表
        ranked_results = sorted(zip([c['column_name'] for c in all_col_metas], scores),
                                key=lambda x: x[1], reverse=True)

            # 返回：排序结果(List) 和 必选结果(Set)
        evidence_details = {
            "exact_matches": {},  # kw -> [col1, col2]
            "fuzzy_matches": {}  # kw -> {col: [matched_values]}
        }

        # --- B路：填充 exact_matches ---
        for kw in extracted_keywords:
            kw_clean = kw.strip()
            if kw_clean in self.exact_index:
                evidence_details["exact_matches"][kw_clean] = list(self.exact_index[kw_clean])

        # --- C路：填充 fuzzy_matches ---
        for kw in extracted_keywords:
            norm_kw = re.sub(r'[^\w]', '', kw).upper()
            if not norm_kw: continue
            kw_mh = MinHash(num_perm=128)
            for char in norm_kw:
                kw_mh.update(char.encode('utf8'))

            fuzzy_hits = {}
            for col, lsh in self.lsh_index.items():
                res = lsh.query(kw_mh)
                if res:
                    fuzzy_hits[col] = res[:2]  # 只取前2个匹配值
            if fuzzy_hits:
                evidence_details["fuzzy_matches"][kw] = fuzzy_hits

        # 合并所有候选列
        all_candidate_cols = set()
        for cols in evidence_details["exact_matches"].values():
            all_candidate_cols.update(cols)
        for col_dict in evidence_details["fuzzy_matches"].values():
            all_candidate_cols.update(col_dict.keys())

        # # A路结果（用于 SELECT 列）
        # all_col_metas = self.column_metadata
        # pairs = [[question, col['column_description']] for col in all_col_metas]
        # scores = self.bert_ranker.predict(pairs)
        # ranked_results = sorted(
        #     zip([c['column_name'] for c in all_col_metas], scores),
        #     key=lambda x: x[1], reverse=True
        # )

        return ranked_results, all_candidate_cols, evidence_details

    def format_for_selector(self, candidates: list):
        """将候选列列表格式化为 M-Schema，供 LLM 裁决"""
        lines = ["[Schema Candidates]"]
        for col_name in candidates:
            # 找回元数据
            meta = next((c for c in self.column_metadata if c['column_name'] == col_name), None)
            if meta:
                exs = ", ".join(meta.get('examples', [])[:2])
                lines.append(f"- {col_name}: {meta['column_description']} (Ex: {exs})")
        return "\n".join(lines)

    def retrieve_schema_by_question_only(self, question: str, top_k: int = 20):
        """
        【新增】仅根据问题语义，召回最相关的 Top-K 列。
        用于在实体提取阶段缩减 Schema，节省 Token。
        """
        all_col_metas = self.column_metadata
        # 构造 (问题, 列描述) 对
        pairs = [[question, col['column_description']] for col in all_col_metas]

        # BERT 打分
        scores = self.bert_ranker.predict(pairs)

        # 排序
        ranked_cols = sorted(
            zip([c['column_name'] for c in all_col_metas], scores),
            key=lambda x: x[1], reverse=True
        )

        # 取 Top-K 列名
        selected_col_names = [x[0] for x in ranked_cols[:top_k]]
        return selected_col_names
