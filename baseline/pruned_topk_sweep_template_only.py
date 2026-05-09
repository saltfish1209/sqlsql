"""
精简 Schema baseline：在 ``data/train_dataset_template_only.csv`` 全量 MATCH 上，
遍历 ``top_k_embed`` 从 10 到 50（步距 2），将每轮指标写入 CSV 便于画图。

用法（需已配置 LLM 环境变量）::

    python baseline/pruned_topk_sweep_template_only.py
    DEBUG_MODE=False python baseline/pruned_topk_sweep_template_only.py --output baseline/topk_pruned_metrics.csv

说明：准确率与样本集合与 ``training.evaluate`` / ``baseline.evaluate --use-full-data`` 一致；
Token 统计为每题 pipeline 汇总（pruned 含实体抽取 + SQL 生成等所有经 ``TokenTracker`` 的调用）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))

from baseline.baseline import BaselineSystem
from config.settings import settings
from training.evaluate import (
    _compare_results,
    normalize_execution_result,
    parse_ground_truth,
)


def _load_full_match_df() -> pd.DataFrame | None:
    csv_path = str(settings.train_csv)
    if not Path(csv_path).is_file():
        print(f"[ERROR] 数据文件不存在: {csv_path}")
        return None
    df_full = pd.read_csv(csv_path)
    df_full = df_full[df_full["SQL验证状态"] == "MATCH"].copy()
    df_full = df_full.sample(frac=1, random_state=settings.random_state).reset_index(
        drop=True
    )
    return df_full


async def evaluate_one_pass(system: BaselineSystem, df: pd.DataFrame, *, verbose: bool) -> dict:
    correct = 0
    total = len(df)
    sum_in = 0
    sum_out = 0
    sum_tot = 0
    sum_first_infer = 0.0
    sum_cost = 0.0
    has_gt_sql = "SQL语句" in df.columns

    for idx, row in df.iterrows():
        question = str(row["生成问题"]).strip()
        raw_gt = row["生成结果"]

        try:
            output = await system.run_pipeline(question)
            tu = output.get("token_usage") or {}
            tin = int(tu.get("input_tokens") or 0)
            tout = int(tu.get("output_tokens") or 0)
            ttot = int(tu.get("total_tokens") or (tin + tout))
            sum_in += tin
            sum_out += tout
            sum_tot += ttot
            sum_first_infer += float(output.get("first_inference_time", 0.0) or 0.0)
            sum_cost += float(output.get("cost_time", 0.0) or 0.0)

            gt_parsed = parse_ground_truth(raw_gt)
            pred_parsed = normalize_execution_result(output.get("execution_result"))
            if _compare_results(gt_parsed, pred_parsed):
                correct += 1
            elif verbose:
                pred_sql = output.get("final_sql") or ""
                gt_sql = str(row["SQL语句"]).strip() if has_gt_sql else ""
                print(f"  [FAIL #{idx}] Q: {question[:80]}...")
                print(f"    pred_sql: {pred_sql[:200]}...")
                print(f"    gt_sql:   {gt_sql[:200]}...")
        except Exception as e:
            if verbose:
                print(f"  [EXC #{idx}] {type(e).__name__}: {e}")

    n = total if total else 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "avg_total_tokens": sum_tot / n,
        "avg_input_tokens": sum_in / n,
        "avg_output_tokens": sum_out / n,
        "avg_first_inference_time_seconds": sum_first_infer / n,
        "avg_cost_time_seconds": sum_cost / n,
    }


async def main_async(args: argparse.Namespace) -> None:
    default_csv = (
        Path(__file__).resolve().parent.parent / "data" / "train_dataset_template_only.csv"
    )
    eval_csv = Path(args.eval_csv) if args.eval_csv else default_csv
    settings.train_csv = eval_csv
    settings.train_split = 0.0
    settings.val_split = 0.0

    df = _load_full_match_df()
    if df is None or df.empty:
        print("无可用 MATCH 数据，退出。")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "k",
        "accuracy",
        "correct",
        "total",
        "avg_total_tokens",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_first_inference_time_seconds",
        "avg_cost_time_seconds",
        "round_elapsed_seconds",
    ]

    rows: list[dict] = []
    for k in range(args.k_min, args.k_max + 1, args.k_step):
        settings.top_k_embed = k
        t0 = time.time()
        system = BaselineSystem(mode="pruned")
        metrics = await evaluate_one_pass(system, df, verbose=args.verbose)
        elapsed = time.time() - t0
        row = {
            "k": k,
            **metrics,
            "round_elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"[topk-sweep] k={k} acc={metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total']}) "
            f"avg_tokens={metrics['avg_total_tokens']:.1f} "
            f"round_time={elapsed:.1f}s"
        )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[topk-sweep] 已写入: {out_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="pruned baseline top-k 扫参 → CSV")
    p.add_argument(
        "--eval-csv",
        type=str,
        default="",
        help="评测 CSV；默认 data/train_dataset_template_only.csv",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "pruned_topk_sweep_template_only_metrics.csv"
        ),
        help="输出 CSV 路径",
    )
    p.add_argument("--k-min", type=int, default=52)
    p.add_argument("--k-max", type=int, default=80)
    p.add_argument("--k-step", type=int, default=2)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="打印错题与异常（否则仅每轮一行汇总）",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
