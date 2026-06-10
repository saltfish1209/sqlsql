"""
Phase 3: 带 Early Stopping 的最终训练。

使用 Phase 2 grid sweep 选出的最佳 (margin, lambda)，设置较大 max_epochs，
在 NDCG@6 连续 N 个 epoch 不提升时自动停止并回滚到最佳权重。

支持自动从 grid_sweep.csv 读取最优参数，也可手动指定。

用法:
  # 自动读取 grid sweep 最优参数
  python figure/crossencoder/run_train_track.py

  # 手动指定参数
  python figure/crossencoder/run_train_track.py --pair-margin 0.15 --pair-lambda 0.7

  # 调整 max epochs 和 patience
  python figure/crossencoder/run_train_track.py --epochs 10 --early-stop-patience 3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import (
    CHECKPOINTS,
    EARLY_STOP_PATIENCE,
    EPOCH_VAL_CSV,
    NDCG_EVAL_K,
    TRAIN_LOSS_CSV,
    ensure_dirs,
    get_early_stop_patience,
    read_best_grid_params,
)
from training.cross_encoder_train import (
    CrossEncoderTrainConfig,
    EVAL_EVERY_STEPS_AUTO,
    train_cross_encoder,
)


def main() -> None:
    grid_best = read_best_grid_params()
    default_margin = float(grid_best["pair_margin"]) if grid_best else 0.15
    default_lambda = float(grid_best["pair_lambda"]) if grid_best else 0.7

    parser = argparse.ArgumentParser(
        description="Phase 3: Early-stopping training with best (margin, lambda)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="最大训练 epoch 数")
    parser.add_argument(
        "--pair-lambda", type=float, default=default_lambda,
        help=f"pair_lambda（默认从 grid_sweep.csv 读取: {default_lambda}）",
    )
    parser.add_argument(
        "--pair-margin", type=float, default=default_margin,
        help=f"pair_margin（默认从 grid_sweep.csv 读取: {default_margin}）",
    )
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            f"val NDCG 连续多少个 epoch 未创新高则早停；"
            f"0 表示 paths.EARLY_STOP_PATIENCE（当前 {EARLY_STOP_PATIENCE}）"
        ),
    )
    parser.add_argument(
        "--eval-every-steps", type=int, default=EVAL_EVERY_STEPS_AUTO,
        help="-1=每 epoch 半程自动 val；0=仅 epoch 末尾；>0=每 N 步",
    )
    parser.add_argument("--no-mid-eval", action="store_true", help="关闭 epoch 中期 val")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关闭早停，跑满 --epochs",
    )
    args = parser.parse_args()
    eval_every_steps = 0 if args.no_mid_eval else args.eval_every_steps

    ensure_dirs()
    for p in (TRAIN_LOSS_CSV, EPOCH_VAL_CSV):
        if p.is_file():
            p.unlink()

    early_patience = get_early_stop_patience(args.early_stop_patience)

    save_path = args.save_path.strip() or os.environ.get(
        "NL2SQL_CE_SAVE_PATH", "",
    )
    if not save_path:
        from config.settings import settings

        save_path = str(settings.cross_encoder_model)

    if grid_best:
        print(f"[Phase 3] 自动加载 grid sweep 最优参数:")
        print(f"  margin={default_margin}  lambda={default_lambda}  "
              f"(NDCG@{NDCG_EVAL_K}={grid_best.get('best_ndcg', '?')})")
    else:
        print(f"[Phase 3] 未找到 grid_sweep.csv，使用默认/手动参数")

    cfg = CrossEncoderTrainConfig(
        epochs=args.epochs,
        pair_lambda=args.pair_lambda,
        pair_margin=args.pair_margin,
        save_path=save_path,
        checkpoint_dir=str(CHECKPOINTS / "train_track"),
        loss_log_path=str(TRAIN_LOSS_CSV),
        epoch_val_csv=str(EPOCH_VAL_CSV),
        eval_top_k=NDCG_EVAL_K,
        eval_split="val",
        early_stop_patience=early_patience,
        early_stop_enabled=not args.no_early_stop,
        eval_every_steps=eval_every_steps,
    )
    print(f"margin={args.pair_margin}  lambda={args.pair_lambda}  "
          f"max_epochs={args.epochs}  early_stop_patience={early_patience}")
    print(f"北极星指标: NDCG@{NDCG_EVAL_K}（连续 {early_patience} epoch 不提升则停止）")
    train_cross_encoder(cfg)
    print(f"train_loss -> {TRAIN_LOSS_CSV}")
    print(f"epoch_val  -> {EPOCH_VAL_CSV}")


if __name__ == "__main__":
    main()

