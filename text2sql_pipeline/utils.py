import unicodedata


def to_halfwidth(text):
    """
    【核心工具】将全角字符转换为半角字符，同时统一中文标点。
    解决：
    1. （） -> ()
    2. ＡＢＣ -> ABC
    3. １２３ -> 123
    4. “ ” -> " "
    """
    if not isinstance(text, str):
        return text if text is not None else ""

    normalized = []
    for char in text:
        code = ord(char)

        # 1. 处理全角空格 (0x3000 -> 0x0020)
        if code == 0x3000:
            code = 0x0020
        # 2. 处理其他全角字符 (0xFF01-0xFF5E -> 0x0021-0x007E)
        # 偏移量是 0xFEE0
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0

        normalized.append(chr(code))

    s = "".join(normalized)

    # 3. 额外的标点映射（处理那些不在标准全角范围内的特殊中文标点）
    # 可以根据你的数据集情况继续添加
    replacements = {
        '。': '.',
        '，': ',',
        '；': ';',
        '：': ':',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '【': '[',
        '】': ']',
        '、': ','  # 顿号通常也视作逗号
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # 4. 去除多余空格（可选：将连续空格合并为一个）
    s = " ".join(s.split())

    return s.strip()

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def track(self, response):
        """解析 OpenAI 格式的 response 并统计 token"""
        if response and hasattr(response, 'usage') and response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

    def get_report(self):
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }

if __name__ == "__main__":
    # 测试一下
    text = "协议库存可视化选购２０２３（含电缆），“国家电网”"
    print(f"原文本: {text}")
    print(f"归一化: {to_halfwidth(text)}")
    # 输出: 协议库存可视化选购2023(含电缆),"国家电网"