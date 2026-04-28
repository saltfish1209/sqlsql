"""
Baseline 评估脚本 —— 复用 ``training.evaluate.run_evaluation``。
─────────────────────────────────────────────────────────────────
与 ``training/evaluate.py`` 使用完全一致的测试集切分（数据集后 10%），
保证与主流程的对比公平。

用法:
    DEBUG_MODE=False python baseline/evaluate.py --mode full
    DEBUG_MODE=True  python baseline/evaluate.py --mode pruned
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))

from baseline.baseline import BaselineSystem
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
    args = parser.parse_args()

    system = BaselineSystem(mode=args.mode)
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), f"baseline_{args.mode}_error_analysis.json"
    )
    await run_evaluation(system, output_path=out_path, label=f"baseline_{args.mode}")


if __name__ == "__main__":
    asyncio.run(main())
