"""
一键重跑数据分割流程（不改动现有模块实现）。

功能：
1) 读取 settings.train_csv
2) 对列名与全部单元格做中英文全半角归一化（全角→半角、中文标点→ASCII）
3) 归一化后按问题去重（优先“生成问题”，回退“原始填充问题”）
4) 过滤 SQL验证状态=MATCH（若列存在）
5) 按问题模版切分 train/val/test（默认 0.8/0.1/0.1）
6) 导出三份 split JSONL（每行一条 CSV 行对应的 JSON 对象，列名作键）
7) 检查每个问题模版扩展条数是否为 5

说明：
- 仅调用/复用现有逻辑，不改原有业务代码。
- 输出目录默认 data/
- 下游读取示例：pd.read_json(path, lines=True)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.settings import settings
from training.dataset_io import normalize_and_deduplicate_dataframe
from training.template_split import split_dataframe_by_template, check_template_expansion_counts


def normalize_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """先归一化再按问题去重，返回去重后的 DataFrame 与去重条数。"""
    question_subset: list[str] | None = None
    if "生成问题" in df.columns:
        question_subset = ["生成问题"]
    elif "原始填充问题" in df.columns:
        question_subset = ["原始填充问题"]
    return normalize_and_deduplicate_dataframe(df, subset=question_subset)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键执行 train/val/test 分割并质检")
    parser.add_argument("--csv", type=str, default="", help="输入CSV路径，默认 settings.train_csv")
    parser.add_argument("--template-col", type=str, default="问题模版")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--expected-per-template", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="", help="输出目录，默认 data")
    args = parser.parse_args()

    src = Path(args.csv) if args.csv else Path(settings.train_csv)
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在: {src}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(settings.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(src, dtype=str, keep_default_na=False)
    rows_before_normalize = len(df_raw)
    df, deduplicated_rows = normalize_and_deduplicate(df_raw)
    if "SQL验证状态" in df.columns:
        df = df[df["SQL验证状态"].astype(str).str.strip().str.upper() == "MATCH"].copy()

    df_train, df_val, df_test = split_dataframe_by_template(
        df,
        template_col=args.template_col,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    train_path = out_dir / "train_split.jsonl"
    val_path = out_dir / "val_split.jsonl"
    test_path = out_dir / "test_split.jsonl"

    for path, part in (
        (train_path, df_train),
        (val_path, df_val),
        (test_path, df_test),
    ):
        part.to_json(path, orient="records", lines=True, force_ascii=False)

    full_report = check_template_expansion_counts(
        df,
        template_col=args.template_col,
        expected_count=args.expected_per_template,
    )
    train_report = check_template_expansion_counts(
        df_train,
        template_col=args.template_col,
        expected_count=args.expected_per_template,
    ) if len(df_train) else {}
    val_report = check_template_expansion_counts(
        df_val,
        template_col=args.template_col,
        expected_count=args.expected_per_template,
    ) if len(df_val) else {}
    test_report = check_template_expansion_counts(
        df_test,
        template_col=args.template_col,
        expected_count=args.expected_per_template,
    ) if len(df_test) else {}

    summary = {
        "source_csv": str(src),
        "text_normalization": "halfwidth+punctuation+invisible_chars",
        "rows_before_normalize_deduplicate": rows_before_normalize,
        "rows_deduplicated_after_normalize": deduplicated_rows,
        "rows_after_match_filter": len(df),
        "splits": {
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
        },
        "ratios": {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": round(1.0 - args.train_split - args.val_split, 6),
        },
        "template_expansion_check": {
            "full": full_report,
            "train": train_report,
            "val": val_report,
            "test": test_report,
        },
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Split pipeline done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
