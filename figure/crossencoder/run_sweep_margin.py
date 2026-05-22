"""
PAIR_MARGIN 扫参 → results/margin_sweep.csv（固定 pair_lambda 与 top_k）。

用法:
  python figure/crossencoder/run_sweep_margin.py --margins 0.05,0.1,0.15,0.2,0.3
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import CHECKPOINTS, MARGIN_SWEEP_CSV, ensure_dirs, get_val_top_k
from training.cross_encoder_eval import evaluate_cross_encoder
from training.cross_encoder_train import CrossEncoderTrainConfig, train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PAIR_MARGIN")
    parser.add_argument("--margins", type=str, default="0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--pair-lambda", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    margins = [float(x) for x in args.margins.split(",") if x.strip()]
    top_k = get_val_top_k(args.top_k)
    fields = [
        "pair_margin",
        "pair_lambda",
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

    for margin in margins:
        run_dir = CHECKPOINTS / f"margin_{margin}"
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(run_dir / "final")
        epoch_val_csv = str(run_dir / "epoch_val.csv")

        cfg = CrossEncoderTrainConfig(
            pair_lambda=args.pair_lambda,
            pair_margin=margin,
            epochs=args.epochs,
            save_path=save_path,
            checkpoint_dir=str(run_dir),
            epoch_val_csv=epoch_val_csv,
            eval_top_k=top_k,
            eval_split="val",
        )
        print(f"\n=== margin={margin} top_k={top_k} ===")
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
            "pair_margin": margin,
            "pair_lambda": args.pair_lambda,
            "epochs": args.epochs,
            "top_k": top_k,
            "split": "val",
            "recall_at_k": round(metrics["recall_at_k"], 6),
            "total": metrics["total"],
            "success": metrics["success"],
            "model_path": best_ckpt,
            "best_epoch": best_epoch,
        })

    with MARGIN_SWEEP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写入: {MARGIN_SWEEP_CSV}")


if __name__ == "__main__":
    main()
