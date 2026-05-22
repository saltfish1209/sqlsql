"""
统一的训练数据 I/O 工具。
─────────────────────────────────────────────
- read_jsonl / write_jsonl：按行读写 JSONL，遇到末尾损坏的行（断电、磁盘写穿、
  Unicode 截断）会**跳过**而不是整文件失败，避免训练前的硬中断。
- load_split_dataframes：把 run_full_split_pipeline.py 输出的
  data/{train,val,test}_split.jsonl 加载成 DataFrame 三元组。
所有训练 / 评估脚本统一通过本模块读取分割与训练数据。
"""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from pipeline.utils import strip_invisible, to_halfwidth


def normalize_cell_text(value) -> str:
    """统一文本归一化：NFKC + 全角半角/标点统一 + 去除不可见脏字符。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    # NFKC 先统一兼容字符，再走现有半角与脏字符清洗。
    value = unicodedata.normalize("NFKC", value)
    return to_halfwidth(strip_invisible(value))


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 的列名与全部单元格做中英文全半角归一化。"""
    out = df.copy()
    out.columns = [normalize_cell_text(c) for c in out.columns]
    for col in out.columns:
        out[col] = out[col].map(normalize_cell_text)
    return out


def normalize_and_deduplicate_dataframe(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """先归一化后去重，返回去重后的 DataFrame 和删除条数。"""
    normalized = normalize_dataframe(df)
    dedup_subset = None
    if subset:
        dedup_subset = [normalize_cell_text(c) for c in subset if normalize_cell_text(c) in normalized.columns]
        if not dedup_subset:
            dedup_subset = None
    deduped = normalized.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
    removed = len(normalized) - len(deduped)
    return deduped, int(removed)


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    """逐行读取 JSONL；自动跳过空行和**最后一段损坏字节**。

    设计：
      - 每行单独 json.loads，解析失败时打印一条 WARN 并 continue。
      - 不抛 UnicodeDecodeError —— 用 errors="ignore" 把异常字节滤掉，
        从而即使尾部字节被截断也能尽量恢复前面所有合法记录。
    """
    p = str(path)
    if not os.path.isfile(p):
        return
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] read_jsonl: 跳过第 {lineno} 行解析失败 ({exc.msg})")
                continue


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict]) -> int:
    """逐行写出 JSONL；返回写入条数。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def split_jsonl_paths(data_dir: str | os.PathLike[str]) -> dict[str, Path]:
    base = Path(data_dir)
    return {
        "train": base / "train_split.jsonl",
        "val": base / "val_split.jsonl",
        "test": base / "test_split.jsonl",
    }


def load_split_dataframes(
    data_dir: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """读取 run_full_split_pipeline.py 输出的三份 JSONL 切分。
    任一文件缺失时返回 None，调用方据此回退到旧的 CSV 切分流程。
    """
    paths = split_jsonl_paths(data_dir)
    if not all(p.exists() for p in paths.values()):
        return None

    def _load(p: Path) -> pd.DataFrame:
        rows = list(read_jsonl(str(p)))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return _load(paths["train"]), _load(paths["val"]), _load(paths["test"])
