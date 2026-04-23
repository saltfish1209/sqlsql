"""
实体提取模块 —— 纯 LLM 提取 SQL WHERE 条件实体（规则兜底当前禁用）。
──────────────────────────────────────────────────────────────
核心改进:
  1. extra_body.guided_json            —— 强制 LLM 输出合法 JSON 数组 (vLLM)
  2. extra_body.chat_template_kwargs   —— 关闭 Qwen3 思考模式，减少 CoT 干扰
  3. _clean_response 多策略解析         —— 括号配对 + 懒惰匹配 + 最外层兜底
  4. 后置过滤                          —— 必须出现在原问题中（空白归一化对比），
                                          且非 Schema 列名、非 prompt 回声
"""
from __future__ import annotations

import json
import re
import time

from openai import AsyncOpenAI, APIConnectionError

from config.settings import settings
from pipeline.utils import debug_print, TokenTracker


def _describe_endpoint(client: AsyncOpenAI) -> str:
    """从 openai AsyncOpenAI 实例上尽力取出 base_url，用于错误日志。"""
    try:
        return str(getattr(client, "base_url", "") or "?")
    except Exception:
        return "?"


# JSON Schema：字符串数组
_ENTITY_JSON_SCHEMA: dict = {
    "type": "array",
    "items": {"type": "string"},
}

# prompt 中可能被 LLM 回声的噪声关键词，命中即剔除
_PROMPT_NOISE_KEYWORDS = (
    "JSON", "json", "数组", "输出", "示例", "规则", "当前任务",
    "Few-Shot", "Schema", "schema",
)

# 单个实体最大长度（超长基本是 prompt 回声或 reasoning 文本）
_MAX_ENTITY_LEN = 80


class EntityExtractor:
    """异步实体提取器。"""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    # ──────────── 公开入口 ────────────

    async def extract(
        self,
        question: str,
        schema_text: str,
        tracker: TokenTracker | None = None,
        max_retries: int = 2,
        schema_columns: list[str] | None = None,
    ) -> list[str]:
        """LLM 实体提取 + 严格后置过滤。"""
        llm_entities = await self._llm_extract(question, schema_text, tracker, max_retries)

        schema_col_set = set(schema_columns or [])
        filtered_llm = self._post_filter(llm_entities, question, schema_col_set)

        # ── 规则兜底已禁用 ──────────────────────────────────────────
        # 如需重新启用，取消下面三段的注释即可。
        #   1) 规则提取
        #   2) 规则结果过滤
        #   3) 与 LLM 结果合并
        #
        # rule_entities = self._rule_extract(question)
        # filtered_rule = self._post_filter(rule_entities, question, schema_col_set)
        # merged = list(dict.fromkeys(
        #     filtered_llm + [e for e in filtered_rule if e not in filtered_llm]
        # ))
        # ───────────────────────────────────────────────────────────

        merged = list(dict.fromkeys(filtered_llm))
        debug_print(f"[Entity] LLM(raw)={llm_entities}")
        debug_print(f"[Entity] LLM(filtered)={filtered_llm}  合并={merged}")
        return merged

    # ──────────── LLM 调用 ────────────

    async def _llm_extract(
        self, question: str, schema_text: str,
        tracker: TokenTracker | None, max_retries: int,
    ) -> list[str]:
        system_msg = "你是一个电力物资采购数据库专家。只输出一个 JSON 字符串数组，不要输出任何其他内容。"
        user_msg = f"""请严格从用户问题中提取**完整的、不可分割的业务实体值**，用于后续 SQL 查询。

【数据库 Schema 摘要】
{schema_text}

[规则]
1. 只能从用户问题中逐字截取，不允许改写、扩写或补全。
2. 仅提取用于 SQL WHERE 条件的具体值（如批次号、物料码、项目名）。
3. 禁止输出列名、字段名、Schema 描述、示例内容或任何解释性文字。
4. 必须输出标准的 JSON 字符串数组格式，例如 ["值1", "值2"]。
5. 如果没有可提取的实体，输出空数组 []。

[Few-Shot 示例]
问题: 国家电网2022年第七十二批采购(输变电项目)有哪些？
输出: ["国家电网2022年第七十二批采购(输变电项目)"]

问题: 500061873物料的中标单位是谁？
输出: ["500061873"]

问题: 协议库存可视化选购20230407的详情
输出: ["协议库存可视化选购20230407"]

[当前任务]
问题: {question}
输出:"""

        for attempt in range(max_retries):
            try:
                messages, extra_body, prefix = self._build_request(system_msg, user_msg)
                call_start = time.time()
                debug_print(
                    f"[Entity] LLM 调用开始 (attempt={attempt + 1}/{max_retries}, "
                    f"guided_json={settings.entity_use_guided_json}, "
                    f"prefix_bracket={settings.entity_prefix_bracket})"
                )
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=settings.entity_max_tokens,
                    timeout=settings.llm_request_timeout_sec,
                    extra_body=extra_body or None,
                )
                debug_print(
                    f"[Entity] LLM 调用完成，耗时 {time.time() - call_start:.2f}s"
                )
                if tracker:
                    tracker.track(resp)

                raw_content = resp.choices[0].message.content or ""
                # ── 智能拼接 prefix ──
                # 理想情况：continue_final_message 生效时，raw_content 是续写（以 "..."] 等开头）
                # 实际情况：某些 vLLM 版本 / chat template 不认 continue_final_message，
                #   模型会把预填的 "[" 当成独立 assistant 消息，又自己吐出完整 [...]。
                # 若检测到 raw 已经以 '[' 开头，视作模型吐了完整数组，不再前置 prefix，
                # 避免出现 [[...]] 的双层嵌套。
                stripped = raw_content.lstrip()
                if prefix and stripped.startswith("["):
                    full_content = raw_content
                else:
                    full_content = prefix + raw_content

                debug_print(
                    f"[Entity][raw] {full_content!r}"
                    if settings.debug_mode else ""
                )

                entities = self._clean_response(full_content)
                if entities is not None:
                    return entities
                debug_print(
                    f"  [Entity] 第{attempt+1}次响应解析失败，重试... "
                    f"(raw_head: {full_content[:120]!r})"
                )
            except APIConnectionError as e:
                # 网络层直接不通：重试也无济于事，立即中断
                endpoint = _describe_endpoint(self.client)
                print(
                    f"[Entity][FATAL] 无法连接到 LLM 服务 (base_url={endpoint}, "
                    f"model={self.model}): {e}\n"
                    f"  ▶ 请检查:\n"
                    f"    1. vLLM/Ollama 服务是否已启动并监听该端口\n"
                    f"    2. 环境变量 LLM_BASE_URL / LLM_MODEL 是否正确\n"
                    f"    3. 是否需要设置 NO_PROXY=127.0.0.1,localhost\n"
                    f"    4. 服务启动日志中是否有 OOM / CUDA 错误"
                )
                return []
            except Exception as e:
                debug_print(f"  [Entity] 第{attempt+1}次调用异常: {type(e).__name__}: {e}")

        debug_print("  [Entity] 所有重试均失败，返回空列表")
        return []

    # ──────────── 请求装配 ────────────

    @staticmethod
    def _build_request(system_msg: str, user_msg: str) -> tuple[list[dict], dict, str]:
        """
        根据配置决定 messages / extra_body / 需要预填的字符串。
        返回 (messages, extra_body, prefix_str)
        """
        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        extra_body: dict = {}
        prefix = ""

        if settings.entity_use_guided_json:
            # vLLM 原生参数；Ollama 等后端会静默忽略
            extra_body["guided_json"] = _ENTITY_JSON_SCHEMA

        # 实体提取不需要 CoT，通过 chat_template_kwargs 彻底关闭思考模式
        if not settings.enable_thinking_for_entity:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        if settings.entity_prefix_bracket:
            # 替模型开头，直接以 '[' 结尾的 assistant 消息强迫其续写数组内容
            messages.append({"role": "assistant", "content": "["})
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
            prefix = "["

        return messages, extra_body, prefix

    # ──────────── 响应清洗 ────────────

    @staticmethod
    def _clean_response(content: str) -> list[str] | None:
        """
        多策略解析：
          策略 A —— 括号配对找第一个完整闭合的 `[...]` 子串（抗 `]]`, `[...][[`）
          策略 B —— 懒惰匹配 `\[.*?\]`
          策略 C —— 最外层 `[第一个 [` → `最后一个 ]`
          每种策略都再经 "单引号→双引号、去尾逗号" 宽松修复后重试。
        """
        if not content:
            return None
        text = content.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        snippets: list[str] = []

        # —— 策略 A：括号深度配对（关键兜底 `]]` 这种脏尾巴） ——
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        snippets.append(text[start: i + 1])
                        start = -1

        # —— 策略 B：懒惰匹配的最小 `[...]` ——
        for m in re.finditer(r"\[[^\[\]]*\]", text, re.DOTALL):
            if m.group(0) not in snippets:
                snippets.append(m.group(0))

        # —— 策略 C：最外层（宽容模式） ——
        lb = text.find("[")
        rb = text.rfind("]")
        if lb != -1 and rb != -1 and lb < rb:
            outer = text[lb: rb + 1]
            if outer not in snippets:
                snippets.append(outer)

        for snippet in snippets:
            variants = [snippet]
            # 宽松修复：单引号 → 双引号、去尾逗号
            fixed = re.sub(r"(?<!\\)'", '"', snippet)
            fixed = re.sub(r",\s*]", "]", fixed)
            if fixed != snippet:
                variants.append(fixed)
            for var in variants:
                try:
                    result = json.loads(var)
                except json.JSONDecodeError:
                    continue
                flat = EntityExtractor._flatten_string_list(result)
                if flat is not None:
                    return flat
        return None

    @staticmethod
    def _flatten_string_list(obj) -> list[str] | None:
        """
        兼容三种情况：
          ["a", "b"]          → ["a", "b"]
          [["a", "b"]]        → ["a", "b"]   (双层嵌套)
          [["a"], ["b"]]      → ["a", "b"]   (多个子数组)
        非列表或无字符串 → None。
        """
        if not isinstance(obj, list):
            return None
        # 若单元素且其内部又是 list，向下剥一层
        if len(obj) == 1 and isinstance(obj[0], list):
            obj = obj[0]
        flat: list[str] = []
        for item in obj:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, (str, int, float)):
                        s = str(sub).strip()
                        if s:
                            flat.append(s)
            elif isinstance(item, (str, int, float)):
                s = str(item).strip()
                if s:
                    flat.append(s)
        return flat if flat else None

    # ──────────── 规则提取（当前未启用，保留以便日后回退） ────────────

    @staticmethod
    def _rule_extract(question: str) -> list[str]:
        """
        [DISABLED] 高置信度编码兜底提取。
        当前流程中未被 `extract()` 调用；保留函数体供需要时一键复用。
        只提取：**数字编码** 或 **字母+数字组合编码**。
        不抓取中文、单纯字母单词、短数字（如年份 "2022"、量词 "3 个"）。
        """
        found: list[str] = []
        # 连续的 ASCII 字母/数字 token，前后不得相邻字母数字（词边界）
        for m in re.finditer(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])", question):
            token = m.group()
            has_digit = any(ch.isdigit() for ch in token)
            has_alpha = any(ch.isalpha() for ch in token)
            if not has_digit:
                continue
            if not has_alpha and len(token) < 6:
                continue
            if has_alpha and len(token) < 3:
                continue
            if token not in found:
                found.append(token)
        return found

    # ──────────── 后置过滤 ────────────

    @staticmethod
    def _normalize_ws(s: str) -> str:
        """去掉所有空白字符，消除 Qwen tokenizer 在中文+数字边界插入的伪空格。"""
        return re.sub(r"\s+", "", s)

    @staticmethod
    def _post_filter(
        entities: list[str],
        question: str,
        schema_cols: set[str],
    ) -> list[str]:
        """
        实体必须是原问题中出现过的子串；同时剔除 Schema 列名和 prompt 回声。

        对比时做空白归一化：
          LLM 返回 "协议库存可视化选购 20230407" 也算匹配 "协议库存可视化选购20230407"。
          返回值使用归一化后的形式（无空格），保证与数据库中实际存储的值对齐。
        """
        q_norm = EntityExtractor._normalize_ws(question)
        schema_norm = {EntityExtractor._normalize_ws(c) for c in schema_cols}

        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in entities:
            if not isinstance(raw, str):
                continue
            e = raw.strip().strip("，,。.；;:：\"'`")
            if not e or len(e) > _MAX_ENTITY_LEN:
                continue
            # 归一化：去空白后再判断（核心修复点）
            e_norm = EntityExtractor._normalize_ws(e)
            if not e_norm:
                continue
            if e_norm in seen:
                continue
            if any(kw in e_norm for kw in _PROMPT_NOISE_KEYWORDS):
                continue
            if e_norm in schema_norm:
                continue
            if e_norm not in q_norm:
                continue
            cleaned.append(e_norm)
            seen.add(e_norm)
        return cleaned
