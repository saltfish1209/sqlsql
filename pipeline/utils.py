"""
公共工具集 —— 字符归一化、调试打印、Token 统计。
"""
from __future__ import annotations

import re

# 匹配所有应剔除的"脏"字符：
#   \x00-\x08, \x0b-\x0c, \x0e-\x1f  ASCII 控制字符（保留 \t \n \r）
#   \x7f                               DEL
#   \ufffd                             Unicode 替换字符（常见于编码错误）
#   \ufeff                             BOM / 零宽非断空格
#   \u200b-\u200f \u202a-\u202e       零宽 / 方向控制字符
#   \ue000-\uf8ff                      Unicode 私有使用区 (BMP)
#   \U000f0000-\U000fffff              Unicode 私有使用区 (Plane 15)
#   \U00100000-\U0010ffff              Unicode 私有使用区 (Plane 16)
_DIRTY_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\ufffd\ufeff"
    r"\u200b-\u200f\u202a-\u202e"
    r"\ue000-\uf8ff"
    r"\U000f0000-\U000fffff"
    r"\U00100000-\U0010ffff]"
)


def strip_invisible(text) -> str:
    """
    去除字符串中的不可见脏字符（控制字符、私有区字符、零宽字符、BOM 等）。
    非字符串值原样返回。
    """
    if not isinstance(text, str):
        return text if text is not None else ""
    return _DIRTY_CHARS_RE.sub("", text).strip()


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
