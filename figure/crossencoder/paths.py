"""CrossEncoder 实验结果与 checkpoint 路径（CSV 输出目录）。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

# ── val Top-K（每 epoch 结束 val 评估、lambda/margin/epoch 扫参对比时固定）──
# 修改此处即可全局生效；也可用环境变量 NL2SQL_CE_VAL_TOP_K 覆盖。
VAL_TOP_K = 6

# ── 早停 patience（连续多少个 epoch val recall 未创新高则停止）──
# 修改此处即可全局生效；也可用环境变量 NL2SQL_CE_EARLY_STOP_PATIENCE 覆盖。
EARLY_STOP_PATIENCE = 2

# 一个扫参 / 一类实验对应一个 CSV
TRAIN_LOSS_CSV = RESULTS / "train_loss.csv"
EPOCH_VAL_CSV = RESULTS / "epoch_val.csv"
GOLD_STATS_CSV = RESULTS / "gold_column_stats.csv"
TOPK_CURVE_CSV = RESULTS / "topk_curve.csv"
LAMBDA_SWEEP_CSV = RESULTS / "lambda_sweep.csv"
MARGIN_SWEEP_CSV = RESULTS / "margin_sweep.csv"
BASELINE_COMPARE_CSV = RESULTS / "baseline_vs_finetuned.csv"
TOPK_CHOICE_CSV = RESULTS / "topk_choice.csv"
EPOCH_SWEEP_CSV = RESULTS / "epoch_sweep.csv"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)


def get_val_top_k(cli_override: int = 0) -> int:
    """
    返回 epoch val / 扫参汇总使用的固定 Top-K。
    优先级: 命令行 --top-k > 环境变量 NL2SQL_CE_VAL_TOP_K > VAL_TOP_K。
    """
    if cli_override > 0:
        return int(cli_override)
    raw = os.getenv("NL2SQL_CE_VAL_TOP_K", "").strip()
    if raw:
        return int(raw)
    return VAL_TOP_K


def get_early_stop_patience(cli_override: int = 0) -> int:
    """
    返回训练早停 patience。
    优先级: 命令行 --early-stop-patience > 环境变量 NL2SQL_CE_EARLY_STOP_PATIENCE > EARLY_STOP_PATIENCE。
    """
    if cli_override > 0:
        return int(cli_override)
    raw = os.getenv("NL2SQL_CE_EARLY_STOP_PATIENCE", "").strip()
    if raw:
        return int(raw)
    return EARLY_STOP_PATIENCE


def read_recommended_k_primary() -> int | None:
    import csv

    if not GOLD_STATS_CSV.is_file():
        return None
    with GOLD_STATS_CSV.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        return None
    return int(float(row["recommended_k_primary"]))
