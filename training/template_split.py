"""Utilities for splitting generated datasets on question-template boundaries."""
from __future__ import annotations

from typing import Iterable


def ordered_unique_templates(values: Iterable[str]) -> list[str]:
    """Return unique template keys in first-seen order."""
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
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
    template_series = df[template_col].astype(str)
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
