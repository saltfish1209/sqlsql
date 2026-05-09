"""
Baseline 评估脚本 —— 复用 ``training.evaluate.run_evaluation``。
─────────────────────────────────────────────────────────────────
与 ``training/evaluate.py`` 使用完全一致的测试集切分（数据集后 10%），
保证与主流程的对比公平。

用法:
    DEBUG_MODE=True python baseline/evaluate.py --mode full --use-full-data
    DEBUG_MODE=True  python baseline/evaluate.py --mode pruned --use-full-data
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
        default=str(Path(__file__).resolve().parent.parent / "data" / "train_dataset_template_only.csv"),
        help="评测比对CSV路径，缺省使用 data/train_dataset_template_only.csv",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="使用评测 CSV 的全量 MATCH 数据（不再按 80/10/10 仅取最后 10%）",
    )
    args = parser.parse_args()

    # 最小改动：复用 training.evaluate.run_evaluation，
    # 仅在入口处把其读取的数据文件切到用户指定 CSV。
    settings.train_csv = Path(args.eval_csv)
    if args.use_full_data:
        # run_evaluation 会按 train_split + val_split 计算测试集起点；
        # 设为 0 即可使用全量 MATCH 数据作为评测集。
        settings.train_split = 0.0
        settings.val_split = 0.0

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
    )


if __name__ == "__main__":
    asyncio.run(main())
