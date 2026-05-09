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
import json
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
    def _strip_thinking(text: str) -> str:
        """Remove thinking/reasoning blocks in various formats."""
        # Format 1: <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Format 2: <thinking>...</thinking>
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        # Fallback: if a closing tag remains (non-standard opening), discard everything before it
        for tag in ("</think>", "</thinking>"):
            if tag in text:
                text = text.split(tag)[-1]
        # Format 3: "Thinking Process:" / "思考过程:" — discard up to the last code fence pair
        text = re.sub(
            r"^(?:Thinking Process|思考过程)\s*[:：].*?(?=```)",
            "",
            text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return text.strip()

    @staticmethod
    def extract_sql(text: str) -> str:
        if not text:
            return ""
        text_clean = SQLGenerator._strip_thinking(text)
        raw_sql = ""

        # Core rule: extract the LAST code block (between second-last ``` and last ```)
        fence_positions = [m.start() for m in re.finditer(r"```", text_clean)]
        if len(fence_positions) >= 2:
            start = fence_positions[-2] + 3
            end = fence_positions[-1]
            candidate = text_clean[start:end].strip()
            candidate = re.sub(r"^\w+\s*\n", "", candidate, count=1)
            raw_sql = candidate
        else:
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

    # ──────────── 公共 SQL 生成约束 ────────────

    _COMMON_SQL_RULES = (
        "[硬性约束]\n"
        "1. 所有字面量值**必须用单引号**包裹；编号字段（采购订单号/采购申请号/物料编码/"
        "供应商编码/工厂编码/项目定义 等）即使值是纯数字也写成 `\"列\" = '12345'`，绝不写成 = 12345。\n"
        "2. 当 SELECT 时尽量在 WHERE 中追加 `AND \"目标列\" != '' AND \"目标列\" IS NOT NULL` 把空串和 NULL 过滤掉。\n"
        "3. 实体若是**简称 尽量WHERE时使用 `LIKE '%X%'` 而不是 `=`。"
        "4. 用户问题里出现的**每一个具体实体**都必须在 WHERE 子句中得到体现，不要遗漏任何过滤维度。\n"
        "5. 列名固定使用双引号 `\"列名\"`；不要使用列编码、不要发明 schema 中不存在的列。\n"
        "6. 不要添加除问题所给信息或筛选条件外多余的约束。\n"
        "7. SQL 关键字（SELECT/FROM/WHERE/AND/OR/JOIN/ON/DISTINCT/GROUP BY/ORDER BY/LIMIT 等）"
        "与列名、表名之间**必须有空格**分隔，禁止写成 `SELECT列名FROM` 这种无空格形式。\n"
    )

    def _build_path_specs(
        self,
        question: str,
        schema_prompt: str,
        evidence_dict: dict,
        entities: list[str] | None = None,
    ) -> dict[str, tuple[str, str, float]]:
        """
        构建三路 SQL 生成的 (path_type, prompt, temperature) 配置。

        以 path key (``"thinking" / "icl" / "direct" / "plan"``) 为索引，
        供 ``start_candidate_tasks`` 按需挑选启动哪几路。
        """
        evidence_str = self.format_evidence(evidence_dict)
        entities = entities or []
        entity_block = (
            "[必须覆盖的实体]\n" + "\n".join(f"- {e}" for e in entities) + "\n"
        ) if entities else ""

        thinking_prompt = (
            f"你是一名SQL专家。请结合Schema和检测到的证据，通过深度思考生成SQL。\n"
            f"采用sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[数据库证据]\n{evidence_str}\n"
            f"{entity_block}"
            f"[用户问题]\n{question}\n"
            f"{self._COMMON_SQL_RULES}"
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
            f"{entity_block}"
            f"[用户问题]\n{question}\n"
            f"{self._COMMON_SQL_RULES}"
            f"请直接输出SQL，用```sql ... ```包裹，不需要思考过程、解释或其他内容。\n"
        )

        direct_prompt = (
            f"你是一名SQL专家。请参考以下内容生成SQL。\n"
            f"采用sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
            f"[Schema]\n{schema_prompt}\n"
            f"[数据库证据]\n{evidence_str}\n"
            f"{entity_block}"
            f"[用户问题]\n{question}\n"
            f"{self._COMMON_SQL_RULES}"
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

        ``"plan"`` 路径走 agentic 多步抽取（先列过滤维度→逐条对齐列→拼装 SQL），
        其它路径仍是单次 LLM 调用。
        """
        n = num_per_path or settings.num_sql_per_path
        specs = self._build_path_specs(
            question, schema_prompt, evidence_dict, entities=entities
        )

        task_map: dict[str, list[asyncio.Task]] = {}
        for key in paths:
            if key == "plan":
                if not settings.enable_plan_path:
                    continue
                task_map[key] = [
                    asyncio.create_task(
                        self._call_plan_path(
                            question, schema_prompt, entities, evidence_dict, tracker,
                        )
                    )
                ]
                continue
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

    # ──────────── Agentic Plan-Path（多步过滤抽取） ────────────

    async def _call_plan_path(
        self,
        question: str,
        schema_prompt: str,
        entities: list[str],
        evidence_dict: dict,
        tracker: TokenTracker,
    ) -> dict | None:
        """
        三步式 agent 化 SQL 生成（缓解"漏过滤条件"问题）：

          Step 1  列出问题中的所有过滤维度（filter intents），仅输出 JSON
          Step 2  逐条把维度对齐到具体的 (列名, 操作符, 字面量)
          Step 3  拼装最终 SQL，并强制覆盖所有维度 + SELECT 子句

        每一步都让 LLM 完成单一职责，**减少一次性输出 SQL 时漏条件的概率**
        （error.txt 案例 [1] 漏物料编码、[2] 漏物料小类描述）。

        失败时返回 None，由 selector 自然忽略。
        """
        try:
            evidence_str = self.format_evidence(evidence_dict)
            entity_block = (
                "[已识别实体]\n" + "\n".join(f"- {e}" for e in entities) + "\n"
            ) if entities else ""

            # —— Step 1: 列出过滤维度 ——
            step1_prompt = (
                f"你是一名SQL分析师。**只罗列**用户问题中可能的"
                f"过滤维度（filter intents），不要写 SQL。\n\n"
                f"[Schema]\n{schema_prompt}\n"
                f"{entity_block}"
                f"[用户问题]\n{question}\n\n"
                f"[输出规则]\n"
                f"- 输出一个 JSON 数组，元素是字符串，描述每一个过滤维度，"
                f"形如 `\"<语义角色>: <字面量值>\"`。\n"
                f"- 例如：[\"供应商: 珠海许继\", \"物料类目: 配电箱\", \"年份: 2023年\"]。\n"
                f"- 只列**WHERE 条件**，不要把 SELECT 目标列写进来。\n"
                f"- **每一个出现在用户问题中的具体值都必须列出**，宁多勿少。\n"
                f"- 只输出 JSON 数组，不要任何额外文字。"
            )
            intents_json = await self._llm_text(
                step1_prompt, "plan_path/step1", tracker,
                settings.plan_path_temperature,
            )
            intents = self._parse_json_array(intents_json)
            debug_print(f"[Generator][plan_path] step1 维度: {intents}")

            # —— Step 2: 逐条对齐到 (列, 操作符, 字面量) ——
            mapped_clauses: list[str] = []
            if intents:
                step2_prompt = (
                    f"你是一名SQL列定位专家。把每一条过滤维度对齐到 Schema 中"
                    f"**最合适的列**，并产出 SQL WHERE 子句片段。\n\n"
                    f"[Schema]\n{schema_prompt}\n"
                    f"[数据库证据]\n{evidence_str}\n"
                    f"{entity_block}"
                    f"[过滤维度]\n"
                    + "\n".join(f"- {it}" for it in intents) + "\n\n"
                    f"[输出规则]\n"
                    f"- 对每一条维度产出一行 `{'{'}\"intent\": <原维度>, "
                    f"\"clause\": \"\\\"列名\\\" = '值'\" 或 "
                    f"\"\\\"列名\\\" LIKE '%值%'\"{'}'}` JSON 对象。\n"
                    f"- 实体若是简称（无'有限/集团/公司'等后缀）使用 LIKE，"
                    f"否则使用 =。\n"
                    f"- 编号 / 编码字面量即使是纯数字也要加单引号。\n"
                    f"- 全部对象放到一个 JSON 数组，按顺序输出，不要其他解释。"
                )
                clauses_json = await self._llm_text(
                    step2_prompt, "plan_path/step2", tracker,
                    settings.plan_path_temperature,
                )
                parsed = self._parse_json_array(clauses_json)
                for item in parsed:
                    if isinstance(item, dict) and item.get("clause"):
                        mapped_clauses.append(str(item["clause"]).strip())
                    elif isinstance(item, str):
                        mapped_clauses.append(item.strip())
                debug_print(f"[Generator][plan_path] step2 clause: {mapped_clauses}")

            # —— Step 3: 拼装最终 SQL ——
            forced_block = ""
            if mapped_clauses:
                forced_block = (
                    "[已确定的 WHERE 子句片段（必须全部包含，可调整 AND/OR 关系）]\n"
                    + "\n".join(f"- {c}" for c in mapped_clauses) + "\n"
                )
            step3_prompt = (
                f"你是一名SQL专家。结合下面已经对齐好的 WHERE 子句片段，"
                f"为用户问题生成**完整且仅一条**的 SQLite SELECT。\n\n"
                f"[Schema]\n{schema_prompt}\n"
                f"[数据库证据]\n{evidence_str}\n"
                f"{entity_block}"
                f"{forced_block}"
                f"[用户问题]\n{question}\n"
                f"{self._COMMON_SQL_RULES}"
                f"请直接输出 SQL，用```sql ... ```包裹，不要解释、不要思考过程。"
            )
            return await self._call_llm(
                step3_prompt, "plan_path", tracker,
                settings.plan_path_temperature,
            )
        except Exception as e:
            debug_print(f"[Generator] plan_path 失败: {type(e).__name__}: {e}")
            return None

    async def _llm_text(
        self,
        prompt: str,
        tag: str,
        tracker: TokenTracker,
        temperature: float,
    ) -> str:
        """普通 LLM 文本调用（用于 plan 子步骤），不做 SQL 提取。"""
        extra_body: dict = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=settings.max_gen_tokens,
                timeout=settings.llm_request_timeout_sec,
                stream=False,
                extra_body=extra_body,
            )
            tracker.track(resp)
            content = resp.choices[0].message.content or ""
            debug_print(f"[Generator][{tag}] {content!r}")
            return content
        except Exception as e:
            debug_print(f"[Generator][{tag}] 调用异常: {type(e).__name__}: {e}")
            return ""

    @staticmethod
    def _parse_json_array(text: str) -> list:
        """
        从 LLM 输出中提取第一段合法的 JSON 数组（兼容 ```json 代码块、思考标签等）。
        失败时返回空列表，调用方自行降级。
        """
        if not text:
            return []
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # 提取首个 [...] 段
        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        snippet = cleaned[start: i + 1]
                        try:
                            obj = json.loads(snippet)
                            return obj if isinstance(obj, list) else []
                        except Exception:
                            return []
        return []

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
