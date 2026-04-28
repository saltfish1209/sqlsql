"""
SQL 生成器 —— 三路并发多样性生成。
──────────────────────────────────────
路径 A (Thinking)  : 启用推理链，高温度增加多样性
路径 B (ICL)       : 基于相似历史 QA 的 In-Context Learning
路径 C (Direct)    : 无推理链、无示例的直接生成

论文新增改进:
  - Schema 字段随机化 (Randomized Schema) 增加候选多样性
  - Profile 统计信息注入 Prompt
"""
from __future__ import annotations

import asyncio
import os
import random
import re

import pandas as pd
from openai import AsyncOpenAI, APIConnectionError
from sentence_transformers import SentenceTransformer, util

from config.settings import settings
from pipeline.utils import debug_print, TokenTracker


class SQLGenerator:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        embed_model_path: str | None = None,
    ):
        self.client = client
        self.model = model

        qa_path = str(settings.qa_template_csv)
        if os.path.isfile(qa_path):
            self.qa_template_df = pd.read_csv(qa_path)
            _embed = embed_model_path or settings.embed_model
            self.embed_model = SentenceTransformer(
                _embed, trust_remote_code=True,
            )
            self.template_embs = self.embed_model.encode(
                self.qa_template_df["问题模版"].tolist(), convert_to_tensor=True
            )
        else:
            debug_print(f"[Generator] QA 模板文件未找到: {qa_path}，ICL 路径将跳过。")
            self.qa_template_df = None
            self.embed_model = None
            self.template_embs = None

    # ──────────── 证据格式化 ────────────

    @staticmethod
    def format_evidence(evidence_dict: dict) -> str:
        parts = []
        if evidence_dict.get("exact_matches"):
            for val, cols in evidence_dict["exact_matches"].items():
                parts.append(f"值 '{val}' 在列 {cols} 中被精准发现。")
        if evidence_dict.get("fuzzy_matches"):
            for kw, hits in evidence_dict["fuzzy_matches"].items():
                for col, vals in hits.items():
                    parts.append(f"关键词 '{kw}' 与列 '{col}' 中的值相似: {vals}。")
        return "\n".join(parts) if parts else "无直接数据库值参考。"

    # ──────────── ICL 示例检索 ────────────

    def _get_top_k_examples(self, question: str, k: int = 3) -> str:
        if self.embed_model is None or self.qa_template_df is None:
            return ""
        q_emb = self.embed_model.encode(
            question, convert_to_tensor=True,
            prompt=settings.embed_query_prompt,
        )
        scores = util.cos_sim(q_emb, self.template_embs)[0]
        top_results = scores.topk(k=k)
        parts = []
        for idx in top_results.indices:
            row = self.qa_template_df.iloc[idx.item()]
            parts.append(f"问题: {row['问题模版']}\nSQL: {row['SQL模版']}")
        return "\n\n".join(parts)

    # ──────────── SQL 提取 ────────────

    @staticmethod
    def _first_select(sql: str) -> str:
        sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
        for part in sql.split(";"):
            part = part.strip()
            if part and part.upper().lstrip().startswith("SELECT"):
                return part
        return sql.rstrip(";").strip()

    @staticmethod
    def extract_sql(text: str) -> str:
        if not text:
            return ""
        text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        raw_sql = ""
        match = re.search(r"```sql\s*(.*?)\s*```", text_clean, re.DOTALL | re.IGNORECASE)
        if match:
            raw_sql = match.group(1).strip()
        else:
            match_general = re.search(r"```\s*(.*?)\s*```", text_clean, re.DOTALL)
            if match_general:
                candidate = match_general.group(1).strip()
                if candidate.upper().startswith("SELECT"):
                    raw_sql = candidate
            else:
                sql_match = re.search(r"(SELECT\s+.*)", text_clean, re.DOTALL | re.IGNORECASE)
                raw_sql = sql_match.group(1).strip() if sql_match else text_clean
        return SQLGenerator._first_select(raw_sql)

    # ──────────── M-Schema 构建 ────────────

    @staticmethod
    def build_m_schema_prompt(
        selected_columns: list[str],
        all_metadata: list[dict],
        table_name: str = "procurement_table",
        randomize: bool = False,
        profile_map: dict[str, str] | None = None,
    ) -> str:
        """
        构建精简 M-Schema Prompt。
        randomize=True 时随机打乱字段顺序（论文 Schema Randomization 策略）。

        列行格式：(列名, 描述 [内联 profile])
          - 不再从原始 schema 读取 data_type 和 Examples，由 profile_map 统一提供
          - profile_map 内联摘要包含：类型 / 示例值（常见值或随机样本） / 数值范围 / 格式
          - 仅对 selected_columns 中的列注入，不做全量底部拼接
        """
        cols = list(selected_columns)
        if randomize:
            random.shuffle(cols)
        lines = [f"[DB_ID] procurement_db\n[Schema]\n# Table: {table_name}\n["]
        for col_name in cols:
            meta = next(
                (m for m in all_metadata if m["column_name"] == col_name), None
            )
            if meta:
                desc = meta.get("column_description", "")
                inline = (profile_map or {}).get(col_name, "")
                if inline:
                    lines.append(f"  ({col_name}, {desc} {inline})")
                else:
                    lines.append(f"  ({col_name}, {desc})")
        lines.append("]")
        return "\n".join(lines)

    # ──────────── 异步生成 ────────────

    def _build_path_specs(
        self,
        question: str,
        schema_prompt: str,
        evidence_dict: dict,
    ) -> dict[str, tuple[str, str, float]]:
        """
        构建三路 SQL 生成的 (path_type, prompt, temperature) 配置。

        以 path key (``"thinking" / "icl" / "direct"``) 为索引，
        供 ``start_candidate_tasks`` 按需挑选启动哪几路。
        """
        evidence_str = self.format_evidence(evidence_dict)

        thinking_prompt = (
            f"你是一名SQL专家。请结合Schema和检测到的证据，通过深度思考生成SQL。\n"
            f"采用sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[数据库证据]\n{evidence_str}\n"
            f"[用户问题]\n{question}\n"
            f"请务必先输出思考过程，思考结束后**必须**输出高质量 SQL，"
            f"用```sql ... ```包裹，不需要其他多余文本。\n"
        )

        examples = self._get_top_k_examples(question)
        icl_prompt = (
            f"你是一名SQL专家。请参考以下相似案例生成SQL。\n"
            f"采用sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[相似案例]\n{examples}\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[数据库证据]\n{evidence_str}\n"
            f"[用户问题]\n{question}\n"
            f"请直接输出SQL，用```sql ... ```包裹，不需要思考过程、解释或其他内容。\n"
        )

        direct_prompt = (
            f"你是一名SQL专家。请参考以下内容生成SQL。\n"
            f"采用sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[数据库证据]\n{evidence_str}\n"
            f"[用户问题]\n{question}\n"
            f"请直接输出SQL，用```sql ... ```包裹，不需要思考过程、解释或其他内容。\n"
        )

        return {
            "thinking": ("thinking_path", thinking_prompt, settings.thinking_temperature),
            "icl": ("ICL_Path", icl_prompt, settings.icl_temperature),
            "direct": ("Direct_Path", direct_prompt, settings.direct_temperature),
        }

    def start_candidate_tasks(
        self,
        question: str,
        schema_prompt: str,
        entities: list[str],
        evidence_dict: dict,
        tracker: TokenTracker,
        num_per_path: int | None = None,
        paths: tuple[str, ...] = ("thinking", "icl", "direct"),
    ) -> dict[str, list[asyncio.Task]]:
        """
        立即启动指定路径的 SQL 生成任务（不等待），返回 ``{path_key: [Task,...]}``。

        调用方负责自己 ``await`` / ``cancel`` 这些任务，便于实现"快路一致即取消
        thinking 路"等早停策略。
        """
        n = num_per_path or settings.num_sql_per_path
        specs = self._build_path_specs(question, schema_prompt, evidence_dict)

        task_map: dict[str, list[asyncio.Task]] = {}
        for key in paths:
            if key not in specs:
                continue
            path_type, prompt, temperature = specs[key]
            task_map[key] = [
                asyncio.create_task(
                    self._call_llm(prompt, path_type, tracker, temperature)
                )
                for _ in range(n)
            ]
        return task_map

    async def generate_candidates_async(
        self,
        question: str,
        schema_prompt: str,
        entities: list[str],
        evidence_dict: dict,
        tracker: TokenTracker,
        num_per_path: int | None = None,
    ) -> list[dict]:
        """
        默认入口：并发跑三路，等所有路返回，过滤 None。

        若需要"快路先返回、按需取消 thinking"的行为，请改用
        ``start_candidate_tasks`` + ``asyncio.gather`` / ``cancel`` 自行编排。
        """
        task_map = self.start_candidate_tasks(
            question, schema_prompt, entities, evidence_dict, tracker,
            num_per_path=num_per_path,
        )
        all_tasks = [t for ts in task_map.values() for t in ts]
        results = await asyncio.gather(*all_tasks)
        return [r for r in results if r is not None]

    async def _call_llm(
        self, prompt: str, path_type: str,
        tracker: TokenTracker, temperature: float,
    ) -> dict | None:
        try:
            messages: list[dict] = [{"role": "user", "content": prompt}]
            extra_body: dict = {}
            prefix = ""

            # 只有 thinking_path 启用 CoT，其余路径通过 chat_template_kwargs 关闭思考
            is_thinking_path = (path_type == "thinking_path")
            if not is_thinking_path:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            if settings.generator_prefix_code_fence:
                messages.append({"role": "assistant", "content": "```sql\n"})
                extra_body["continue_final_message"] = True
                extra_body["add_generation_prompt"] = False
                prefix = "```sql\n"

            # 构建请求：
            #   thinking_path 不设 max_tokens 上限（让 CoT 充分展开）
            #   非思考路径使用 settings.max_gen_tokens
            create_kwargs: dict = dict(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=settings.llm_request_timeout_sec,
                stream=False,
                extra_body=extra_body or None,
            )
            if is_thinking_path:
                if settings.thinking_max_tokens > 0:
                    create_kwargs["max_tokens"] = settings.thinking_max_tokens
                # thinking_max_tokens == 0 时完全不传 max_tokens，交由 vLLM 上限决定
            else:
                create_kwargs["max_tokens"] = settings.max_gen_tokens

            resp = await self.client.chat.completions.create(**create_kwargs)
            tracker.track(resp)
            content = (resp.choices[0].message.content or "")
            full_content = prefix + content
            debug_print(f"[Generator][raw][{path_type}] {full_content!r}")
            sql = self.extract_sql(full_content)
            if sql:
                return {"type": path_type, "sql": sql, "raw_content": full_content}
            return None
        except APIConnectionError as e:
            base_url = str(getattr(self.client, "base_url", "") or "?")
            print(
                f"[Generator][FATAL] 无法连接 LLM 服务 "
                f"(base_url={base_url}, model={self.model}, path={path_type}): {e}"
            )
            return None
        except Exception as e:
            debug_print(f"[Generator] 生成失败({path_type}): {type(e).__name__}: {e}")
            return None
