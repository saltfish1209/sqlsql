"""
仅扫 epoch 数（固定 lambda/margin/top_k）→ results/epoch_sweep.csv

用法:
  python figure/crossencoder/run_gold_stats.py
  python figure/crossencoder/run_sweep_epochs.py --epochs 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import CHECKPOINTS, EPOCH_SWEEP_CSV, ensure_dirs, get_val_top_k
from training.cross_encoder_eval import evaluate_cross_encoder
from training.cross_encoder_train import CrossEncoderTrainConfig, train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep training epochs")
    parser.add_argument("--epochs-list", type=str, default="1,2,3,4,5")
    parser.add_argument("--pair-lambda", type=float, default=0.7)
    parser.add_argument("--pair-margin", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    epoch_list = [int(x) for x in args.epochs_list.split(",") if x.strip()]
    top_k = get_val_top_k(args.top_k)
    fields = [
        "epochs",
        "pair_lambda",
        "pair_margin",
        "top_k",
        "best_epoch",
        "recall_at_k",
        "total",
        "success",
        "model_path",
    ]
    rows: list[dict] = []

    for n_ep in epoch_list:
        run_dir = CHECKPOINTS / f"epochs_{n_ep}"
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(run_dir / "final")
        epoch_val_csv = str(run_dir / "epoch_val.csv")

        cfg = CrossEncoderTrainConfig(
            epochs=n_ep,
            pair_lambda=args.pair_lambda,
            pair_margin=args.pair_margin,
            save_path=save_path,
            checkpoint_dir=str(run_dir),
            epoch_val_csv=epoch_val_csv,
            eval_top_k=top_k,
            eval_split="val",
        )
        print(f"\n=== epochs={n_ep} top_k={top_k} ===")
        train_cross_encoder(cfg)

        best_epoch = ""
        best_recall = -1.0
        best_ckpt = save_path
        if os.path.isfile(epoch_val_csv):
            with open(epoch_val_csv, encoding="utf-8", newline="") as f:
                for er in csv.DictReader(f):
                    fr = float(er.get("recall_at_k", er.get("full_recall", 0)))
                    if fr >= best_recall:
                        best_recall = fr
                        best_epoch = er["epoch"]
                        best_ckpt = er["model_path"]

        metrics = evaluate_cross_encoder(best_ckpt, split="val", top_k=top_k)
        rows.append({
            "epochs": n_ep,
            "pair_lambda": args.pair_lambda,
            "pair_margin": args.pair_margin,
            "top_k": top_k,
            "best_epoch": best_epoch,
            "recall_at_k": round(metrics["recall_at_k"], 6),
            "total": metrics["total"],
            "success": metrics["success"],
            "model_path": best_ckpt,
        })
        print(f"epochs={n_ep} best_epoch={best_epoch} recall={metrics['recall_at_k']:.2%}")

    with EPOCH_SWEEP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写入: {EPOCH_SWEEP_CSV}")


if __name__ == "__main__":
    main()
