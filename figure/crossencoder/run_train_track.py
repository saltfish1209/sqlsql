"""
训练 CrossEncoder 并记录:
  - results/train_loss.csv   (逐步 loss)
  - results/epoch_val.csv    (每 epoch 一次 val，固定 top_k)

用法:
  python figure/crossencoder/run_train_track.py
  python figure/crossencoder/run_train_track.py --top-k 12   # 覆盖 paths.VAL_TOP_K
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
    TRAIN_LOSS_CSV,
    VAL_TOP_K,
    ensure_dirs,
    get_early_stop_patience,
    get_val_top_k,
)
from training.cross_encoder_train import CrossEncoderTrainConfig, train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CE with loss + per-epoch val CSV")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pair-lambda", type=float, default=0.7)
    parser.add_argument("--pair-margin", type=float, default=0.15)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=f"epoch val 用的 K；0 表示使用 paths.VAL_TOP_K（当前 {VAL_TOP_K}）",
    )
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            f"val recall 连续多少个 epoch 未创新高则早停；"
            f"0 表示 paths.EARLY_STOP_PATIENCE（当前 {EARLY_STOP_PATIENCE}）"
        ),
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关闭早停，跑满 --epochs",
    )
    args = parser.parse_args()

    ensure_dirs()
    for p in (TRAIN_LOSS_CSV, EPOCH_VAL_CSV):
        if p.is_file():
            p.unlink()

    top_k = get_val_top_k(args.top_k)
    early_patience = get_early_stop_patience(args.early_stop_patience)

    save_path = args.save_path.strip() or os.environ.get(
        "NL2SQL_CE_SAVE_PATH", "",
    )
    if not save_path:
        from config.settings import settings

        save_path = str(settings.cross_encoder_model)

    cfg = CrossEncoderTrainConfig(
        epochs=args.epochs,
        pair_lambda=args.pair_lambda,
        pair_margin=args.pair_margin,
        save_path=save_path,
        checkpoint_dir=str(CHECKPOINTS / "train_track"),
        loss_log_path=str(TRAIN_LOSS_CSV),
        epoch_val_csv=str(EPOCH_VAL_CSV),
        eval_top_k=top_k,
        eval_split="val",
        early_stop_patience=early_patience,
        early_stop_enabled=not args.no_early_stop,
    )
    print(f"epoch val 使用固定 top_k={top_k}（扫 epoch/lambda/margin 时勿变 K）")
    print(f"早停 patience={early_patience}（paths.EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}）")
    train_cross_encoder(cfg)
    print(f"train_loss -> {TRAIN_LOSS_CSV}")
    print(f"epoch_val  -> {EPOCH_VAL_CSV}")


if __name__ == "__main__":
    main()
