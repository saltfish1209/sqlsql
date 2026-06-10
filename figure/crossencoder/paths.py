"""CrossEncoder 实验结果与 checkpoint 路径（CSV 输出目录）。

三阶段训练流程:
  Phase 1 — 使用 NDCG@K 作为北极星指标（topk 从训练扫参中剥离）
  Phase 2 — 3×3 grid sweep (margin × lambda)，固定 epoch / LR
  Phase 3 — 最佳 (margin, lambda) + Early Stopping → 冻结模型
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

# ── NDCG 评估 K（训练北极星指标使用，不作为扫参变量）──
NDCG_EVAL_K = 6

# ── 早停 patience（连续多少次 val NDCG 未创新高则停止）──
# val 样本量较小（~120 条）时建议 ≥3，容忍噪声波动
EARLY_STOP_PATIENCE = 3

# ── grid sweep 默认网格 ──
DEFAULT_MARGINS = [0.1, 0.2, 0.3,0.5]
DEFAULT_LAMBDAS = [0.0, 0.3,0.5, 0.7]

# ── CSV 输出路径 ──
TRAIN_LOSS_CSV = RESULTS / "train_loss.csv"
EPOCH_VAL_CSV = RESULTS / "epoch_val.csv"
GOLD_STATS_CSV = RESULTS / "gold_column_stats.csv"
TOPK_CURVE_CSV = RESULTS / "topk_curve.csv"
BASELINE_COMPARE_CSV = RESULTS / "baseline_vs_finetuned.csv"
TOPK_CHOICE_CSV = RESULTS / "topk_choice.csv"
GRID_SWEEP_CSV = RESULTS / "grid_sweep.csv"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)


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


def read_best_grid_params() -> dict | None:
    """从 grid_sweep.csv 读取 NDCG 最高的 (margin, lambda) 组合。"""
    import csv as _csv

    if not GRID_SWEEP_CSV.is_file():
        return None
    best: dict | None = None
    best_ndcg = -1.0
    with GRID_SWEEP_CSV.open(encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            ndcg = float(row.get("best_ndcg", 0))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best = row
    return best
