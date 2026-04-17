"""
LLM 异步客户端工厂 —— 封装 OpenAI 兼容 API 的创建与复用逻辑。
"""
from __future__ import annotations

from openai import AsyncOpenAI

from config import get_llm_base_url, get_llm_api_key, get_llm_model


def create_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=get_llm_base_url(),
        api_key=get_llm_api_key(),
    )


def get_model_name() -> str:
    return get_llm_model()
