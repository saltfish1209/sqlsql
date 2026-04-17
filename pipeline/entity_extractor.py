"""
实体提取模块 —— LLM + 规则双引擎提取 SQL WHERE 条件实体。
──────────────────────────────────────────────────────────────
1. LLM 提取：使用精简 Schema（无 Examples，防止示例值污染）引导 LLM。
2. 规则提取：正则匹配日期编码、物料码等高置信度值作为兜底。
3. 合并去重后返回。
"""
from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from pipeline.utils import debug_print, TokenTracker


class EntityExtractor:
    """异步实体提取器。"""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    # ──────────── LLM 提取 ────────────

    async def extract(
        self,
        question: str,
        schema_text: str,
        tracker: TokenTracker | None = None,
        max_retries: int = 2,
    ) -> list[str]:
        """LLM + 规则联合提取实体。"""
        llm_entities = await self._llm_extract(question, schema_text, tracker, max_retries)
        rule_entities = self._rule_extract(question)
        merged = list(dict.fromkeys(
            llm_entities + [e for e in rule_entities if e not in llm_entities]
        ))
        debug_print(f"[Entity] LLM={llm_entities}  规则={rule_entities}  合并={merged}")
        return merged

    async def _llm_extract(
        self, question: str, schema_text: str,
        tracker: TokenTracker | None, max_retries: int,
    ) -> list[str]:
        system_msg = "你是一个电力物资采购数据库专家。只输出JSON数组，不要输出任何其他内容。"
        user_msg = f"""请严格从用户问题中提取**完整的、不可分割的业务实体值**，用于后续 SQL 查询。

【数据库 Schema 摘要】
{schema_text}

[规则]
1. 提取用于 SQL WHERE 条件的具体值（如批次号、物料码、项目名）。
2. 保持原文拼写，不要拆分复合词。
3. 必须输出标准的 JSON 字符串数组格式，例如 ["值1", "值2"]。
4. 如果没有可提取的实体，输出空数组 []。

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
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.01,
                )
                if tracker:
                    tracker.track(resp)
                raw = resp.choices[0].message.content
                entities = self._clean_response(raw)
                if entities is not None:
                    return entities
                debug_print(f"  [Entity] 第{attempt+1}次响应无效，重试... (raw: {raw[:80]})")
            except Exception as e:
                debug_print(f"  [Entity] 第{attempt+1}次调用异常: {e}")
        debug_print("  [Entity] 所有重试均失败，返回空列表")
        return []

    # ──────────── 响应清洗 ────────────

    @staticmethod
    def _clean_response(content: str) -> list[str] | None:
        """清洗 LLM 输出，成功返回实体列表，否则 None 触发重试。"""
        content = content.strip()
        if not content:
            return None
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^(assistant|user)\s*:\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        if "[" not in content and '"' not in content and "'" not in content:
            return None
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [str(x).strip() for x in result if str(x).strip()]
        except json.JSONDecodeError:
            pass
        matches = re.findall(r'"([^"]+?)"', content)
        if not matches:
            matches = re.findall(r"'([^']+?)'", content)
        clean = [m.strip() for m in matches if m.strip()]
        return clean if clean else None

    # ──────────── 规则提取 ────────────

    @staticmethod
    def _rule_extract(question: str) -> list[str]:
        """正则匹配日期型批次名、纯数字编码等高置信度实体。"""
        found: list[str] = []
        # 中文前缀 + 6-8 位数字
        for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}\d{6,8}", question):
            found.append(m.group())
        # 纯 6-8 位数字
        for m in re.finditer(r"(?<!\d)\d{6,8}(?!\d)", question):
            val = m.group()
            if not any(val in e for e in found):
                found.append(val)
        # 9+ 位数字（物料码等）
        for m in re.finditer(r"(?<!\d)\d{9,}(?!\d)", question):
            val = m.group()
            if not any(val in e for e in found):
                found.append(val)
        return found
