"""
Baseline 评估脚本 —— 复用 ``training.evaluate.run_evaluation``。
─────────────────────────────────────────────────────────────────
与 ``training/evaluate.py`` 使用完全一致的测试集（默认 data/test_split.jsonl），
保证与主流程的对比公平。

用法:
    python baseline/evaluate.py --mode full --concurrency 4
    python baseline/evaluate.py --mode pruned --concurrency 15 --enable-thinking
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))

from baseline.baseline import BaselineSystem
from config.settings import settings
from training.evaluate import run_evaluation


async def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline 端到端评估")
    parser.add_argument(
        "--mode", choices=BaselineSystem.VALID_MODES, default="full",
        help="full=全量Schema, pruned=主流程精简Schema",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="错误日志输出路径，缺省写到 baseline/baseline_<mode>_error_analysis.json",
    )
    parser.add_argument(
        "--full-output", type=str, default="",
        help="全量日志输出路径（含正确+错误样本与汇总），缺省写到 baseline/baseline_<mode>_run_report.json",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        default="",
        help="评测比对CSV路径；缺省使用 settings.test_split_jsonl",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="使用评测 CSV 的全量 MATCH 数据（不再按 80/10/10 仅取最后 10%）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="并行请求数（默认 1 串行；建议不超过 vLLM 的 max-num-seqs，如 4~6）",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", default=None,
        help="强制开启思考模式（覆盖 BASELINE_ENABLE_THINKING 环境变量）",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="强制关闭思考模式（覆盖 BASELINE_ENABLE_THINKING 环境变量）",
    )
    args = parser.parse_args()

    # 命令行覆盖 settings 中的思考开关
    if args.enable_thinking:
        settings.baseline_enable_thinking = True
    elif args.no_thinking:
        settings.baseline_enable_thinking = False

    if args.eval_csv:
        settings.train_csv = Path(args.eval_csv)
        settings.eval_use_test_split_jsonl = False
    if args.use_full_data:
        settings.train_split = 0.0
        settings.val_split = 0.0
        settings.eval_use_test_split_jsonl = False

    system = BaselineSystem(mode=args.mode)
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), f"baseline_{args.mode}_error_analysis.json"
    )
    full_out_path = args.full_output or os.path.join(
        os.path.dirname(__file__), f"baseline_{args.mode}_run_report.json"
    )
    await run_evaluation(
        system,
        output_path=out_path,
        label=f"baseline_{args.mode}",
        full_output_path=full_out_path,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    asyncio.run(main())
