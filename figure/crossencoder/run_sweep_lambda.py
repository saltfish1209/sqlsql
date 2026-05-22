"""
PAIR_LAMBDA 扫参：每个 lambda 训练一轮，val 评估写入 results/lambda_sweep.csv。

固定 val Top-K（默认 paths.VAL_TOP_K=6），只比较 lambda。

用法:
  python figure/crossencoder/run_gold_stats.py
  python figure/crossencoder/run_sweep_lambda.py
  python figure/crossencoder/run_sweep_lambda.py --lambdas 0,0.3,0.7,1.0
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import CHECKPOINTS, LAMBDA_SWEEP_CSV, ensure_dirs, get_val_top_k
from training.cross_encoder_eval import evaluate_cross_encoder
from training.cross_encoder_train import CrossEncoderTrainConfig, train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PAIR_LAMBDA")
    parser.add_argument("--lambdas", type=str, default="0,0.3,0.7,1.0,1.5")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    top_k = get_val_top_k(args.top_k)
    fields = [
        "pair_lambda",
        "pair_margin",
        "epochs",
        "top_k",
        "split",
        "recall_at_k",
        "total",
        "success",
        "model_path",
        "best_epoch",
    ]
    rows: list[dict] = []

    for lam in lambdas:
        run_dir = CHECKPOINTS / f"lambda_{lam}"
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(run_dir / "final")
        epoch_val_csv = str(run_dir / "epoch_val.csv")

        cfg = CrossEncoderTrainConfig(
            pair_lambda=lam,
            pair_margin=args.margin,
            epochs=args.epochs,
            save_path=save_path,
            checkpoint_dir=str(run_dir),
            epoch_val_csv=epoch_val_csv,
            eval_top_k=top_k,
            eval_split="val",
        )
        print(f"\n=== lambda={lam} top_k={top_k} ===")
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
        else:
            best_ckpt = save_path

        metrics = evaluate_cross_encoder(best_ckpt, split="val", top_k=top_k)
        rows.append({
            "pair_lambda": lam,
            "pair_margin": args.margin,
            "epochs": args.epochs,
            "top_k": top_k,
            "split": "val",
            "recall_at_k": round(metrics["recall_at_k"], 6),
            "total": metrics["total"],
            "success": metrics["success"],
            "model_path": best_ckpt,
            "best_epoch": best_epoch,
        })
        print(f"lambda={lam} best_epoch={best_epoch} recall@{top_k}={metrics['recall_at_k']:.2%}")

    with LAMBDA_SWEEP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写入: {LAMBDA_SWEEP_CSV}")


if __name__ == "__main__":
    main()
