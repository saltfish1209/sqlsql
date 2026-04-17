"""
公共工具集 —— 字符归一化、调试打印、Token 统计。
"""
from __future__ import annotations


def debug_print(*args, **kwargs):
    """仅在 DEBUG_MODE=True 时输出，用于中间步骤日志。"""
    from config.settings import settings
    if settings.debug_mode:
        print(*args, **kwargs)


_PUNCT_MAP = {
    '。': '.', '，': ',', '；': ';', '：': ':',
    '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
    '【': '[', '】': ']', '、': ',',
}


def to_halfwidth(text) -> str:
    """全角 → 半角 + 中文标点 → ASCII 标点，消除常见 OCR / 输入法差异。"""
    if not isinstance(text, str):
        return text if text is not None else ""
    chars = []
    for ch in text:
        cp = ord(ch)
        if cp == 0x3000:
            cp = 0x0020
        elif 0xFF01 <= cp <= 0xFF5E:
            cp -= 0xFEE0
        chars.append(chr(cp))
    s = "".join(chars)
    for old, new in _PUNCT_MAP.items():
        s = s.replace(old, new)
    return " ".join(s.split()).strip()


class TokenTracker:
    """累计统计 OpenAI 兼容 API 的 token 用量。"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def track(self, response):
        if response and hasattr(response, "usage") and response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

    def get_report(self) -> dict:
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
