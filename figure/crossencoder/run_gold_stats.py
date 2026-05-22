"""
统计 val 集 gold 列数量分布，并给出 Top-K 选用建议。

用法:
  python figure/crossencoder/run_gold_stats.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import GOLD_STATS_CSV, ensure_dirs  # noqa: F401
from training.cross_encoder_eval import gold_column_stats


def main() -> None:
    ensure_dirs()
    stats = gold_column_stats("val")
    row = {
        "split": stats["split"],
        "count": stats["count"],
        "gold_min": stats["min"],
        "gold_max": stats["max"],
        "gold_mean": round(stats["mean"], 4),
        "gold_p50": stats["p50"],
        "gold_p90": stats["p90"],
        "gold_p95": stats["p95"],
        "recommended_k_primary": stats["recommended_k_primary"],
        "recommended_k_operational": stats["recommended_k_operational"],
        "note": (
            "k_primary: 用于 epoch/lambda/margin 扫参时固定单点对比; "
            "k_operational: 对齐主系统 candidate_top_k; "
            "topk_curve.csv 再扫多档 K 画曲线"
        ),
    }
    with GOLD_STATS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print("Gold 列数量统计 (val)")
    print(f"  样本数: {stats['count']}")
    print(f"  min/max/mean: {stats['min']} / {stats['max']} / {stats['mean']:.2f}")
    print(f"  p50/p90/p95: {stats['p50']} / {stats['p90']} / {stats['p95']}")
    print(f"  推荐 k_primary (扫参固定 K): {stats['recommended_k_primary']}")
    print(f"  推荐 k_operational (对齐线上 Top20): {stats['recommended_k_operational']}")
    print(f"已写入: {GOLD_STATS_CSV}")


if __name__ == "__main__":
    main()
