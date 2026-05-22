"""Utilities for splitting generated datasets on question-template boundaries."""
from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd


def _normalize_template_key(value: object) -> str:
    """归一化模板键，避免超长/异常行带来的隐式字符差异。"""
    s = "" if value is None else str(value)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = " ".join(s.split())
    return s.strip()


def ordered_unique_templates(values: Iterable[str]) -> list[str]:
    """Return unique template keys in first-seen order."""
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_template_key(value)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def split_template_keys(
    template_keys: list[str],
    train_split: float,
    val_split: float,
) -> tuple[list[str], list[str], list[str]]:
    """Split template keys, never rows, so boundaries fall between templates."""
    n = len(template_keys)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    train_end = max(0, min(train_end, n))
    val_end = max(train_end, min(val_end, n))
    return (
        template_keys[:train_end],
        template_keys[train_end:val_end],
        template_keys[val_end:],
    )


def split_dataframe_by_template(
    df,
    template_col: str = "问题模版",
    train_split: float = 0.8,
    val_split: float = 0.0,
) -> tuple:
    """Split a pandas DataFrame by ordered template groups."""
    keys = ordered_unique_templates(df[template_col].tolist())
    train_keys, val_keys, test_keys = split_template_keys(
        keys,
        train_split=train_split,
        val_split=val_split,
    )
    train_set = set(train_keys)
    val_set = set(val_keys)
    test_set = set(test_keys)

    template_series = df[template_col].map(_normalize_template_key)
    df_train = df[template_series.isin(train_set)].copy()
    df_val = df[template_series.isin(val_set)].copy()
    df_test = df[template_series.isin(test_set)].copy()
    return df_train, df_val, df_test


def non_test_template_dataframe(
    df,
    template_col: str = "问题模版",
    train_split: float = 0.8,
    val_split: float = 0.0,
):
    """Return all rows whose template is outside the held-out test split."""
    keys = ordered_unique_templates(df[template_col].tolist())
    train_keys, val_keys, _ = split_template_keys(
        keys,
        train_split=train_split,
        val_split=val_split,
    )
    allowed = set(train_keys) | set(val_keys)
    template_series = df[template_col].astype(str)
    return df[template_series.isin(allowed)].copy()


def check_template_expansion_counts(
    df,
    template_col: str = "问题模版",
    expected_count: int = 5,
) -> dict:
    """检查每个问题模版的扩展问答数量是否达到 expected_count。"""
    if template_col not in df.columns:
        raise KeyError(f"缺少模板列: {template_col}")

    vc = df[template_col].astype(str).value_counts(dropna=False)
    total_templates = int(vc.shape[0])
    bad = vc[vc != expected_count]

    return {
        "total_templates": total_templates,
        "expected_count": int(expected_count),
        "ok_templates": int(total_templates - bad.shape[0]),
        "bad_templates": int(bad.shape[0]),
        "bad_detail": {str(k): int(v) for k, v in bad.to_dict().items()},
    }


def _main():
    parser = argparse.ArgumentParser(description="按问题模板切分并检查扩展样本数")
    parser.add_argument("--csv", type=str, required=True, help="输入CSV路径")
    parser.add_argument("--template-col", type=str, default="问题模版")
    parser.add_argument("--expected-per-template", type=int, default=5)
    args = parser.parse_args()

    train_split = 0.8
    val_split = 0.1

    df = pd.read_csv(args.csv)
    df_train, df_val, df_test = split_dataframe_by_template(
        df,
        template_col=args.template_col,
        train_split=train_split,
        val_split=val_split,
    )

    print("=== Split Summary ===")
    print(f"split ratio -> train/val/test = {train_split:.1f}/{val_split:.1f}/{1-train_split-val_split:.1f}")
    print(f"train rows: {len(df_train)}")
    print(f"val rows  : {len(df_val)}")
    print(f"test rows : {len(df_test)}")

    print("\n=== Expansion Check (whole dataset) ===")
    report = check_template_expansion_counts(
        df,
        template_col=args.template_col,
        expected_count=args.expected_per_template,
    )
    print({k: v for k, v in report.items() if k != "bad_detail"})

    if report["bad_templates"] > 0:
        print("\n[WARN] 以下模板扩展问答数量不等于期望值:")
        for tpl, cnt in report["bad_detail"].items():
            print(f"  - 模板: {tpl} | 扩展条数: {cnt} (期望: {args.expected_per_template})")
    else:
        print(f"\n[OK] 所有模板扩展问答数量均为 {args.expected_per_template}。")


if __name__ == "__main__":
    _main()
