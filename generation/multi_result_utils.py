"""
多结果回答模板工具
─────────────────────────────────────────────
当 `回答模版` 包含**顶层逗号**（中文 '，' 或英文 ','），
表示这条问题需要分别返回多个独立结果。例如：

    count{},count1{中标签报号}
    {供应商描述}，{中标签报号}

会被拆为多个子模板，分别生成各自的 SQL 和结果，
最后通过 `MULTI_RESULT_SEP` 拼回一个字段。

这套工具被三处使用：
  1. `generation/construct.py` —— 数据集生成时拆分 + 拼接
  2. `training/evaluate.py`    —— 读取 GT 时识别多结果
  3. `pipeline/system.py`      —— 推理时让 LLM 逐题作答
"""
from __future__ import annotations

# 多结果分隔符。挑了一个 Unicode 双竖线，业务数据里几乎不会出现。
MULTI_RESULT_SEP = "‖"

# 单个 SQL / Pandas 查询返回多行时的行间分隔符。
# 不使用逗号，避免和物料描述等字段内部的英文逗号冲突。
ROW_RESULT_SEP = "&"


def split_answer_template_top_level(a_str: str) -> list[str]:
    """按顶层 ',' / '，' 切分回答模板，{...} 内部的逗号不切。

    例：
        "count{},count1{中标签报号}"      -> ["count{}", "count1{中标签报号}"]
        "{供应商描述}, {中标签报号}"      -> ["{供应商描述}", "{中标签报号}"]
        "{物料描述,规格}"                 -> ["{物料描述,规格}"]   # 大括号内逗号保留
    """
    if not isinstance(a_str, str):
        return []
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in a_str:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in (",", "，"):
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
        else:
            buf.append(ch)
    seg = "".join(buf).strip()
    if seg:
        parts.append(seg)
    return parts


def split_multi_result_string(s: str) -> list[str]:
    """把 `生成结果 / SQL语句` 字段按 MULTI_RESULT_SEP 拆分。"""
    if not isinstance(s, str):
        return [s] if s is not None else []
    if MULTI_RESULT_SEP in s:
        return [p.strip() for p in s.split(MULTI_RESULT_SEP)]
    return [s]


def join_multi_result(parts: list[str]) -> str:
    """把多个子结果用 MULTI_RESULT_SEP 拼起来。"""
    return MULTI_RESULT_SEP.join(parts)


def is_multi_result(s: str) -> bool:
    """判断字符串是否是多结果格式。"""
    return isinstance(s, str) and MULTI_RESULT_SEP in s
