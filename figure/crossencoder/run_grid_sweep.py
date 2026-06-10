"""
Phase 2: margin × lambda 正交网格搜索（带内部早停）。

默认 3×3 网格（9 组实验），每组训练至多 max_epochs 个 epoch，
NDCG@6 连续 patience 个 epoch 未提升即停止并回滚到最优权重。

结果汇总至 grid_sweep.csv:
  pair_lambda, pair_margin, best_epoch, best_ndcg, best_mrr, best_recall_at_k, ...

用法:
  python figure/crossencoder/run_grid_sweep.py
  python figure/crossencoder/run_grid_sweep.py --margins 0.05,0.15,0.3 --lambdas 0.3,0.7,1.0
  python figure/crossencoder/run_grid_sweep.py --max-epochs 10 --patience 2
  python figure/crossencoder/run_grid_sweep.py --csv-mode append
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from figure.crossencoder.paths import (
    CHECKPOINTS,
    DEFAULT_LAMBDAS,
    DEFAULT_MARGINS,
    EARLY_STOP_PATIENCE,
    GRID_SWEEP_CSV,
    NDCG_EVAL_K,
    ensure_dirs,
    get_early_stop_patience,
)
from training.cross_encoder_train import (
    CrossEncoderTrainConfig,
    EVAL_EVERY_STEPS_AUTO,
    train_cross_encoder,
)

_KEEP_RUN_ARTIFACTS = frozenset({"final", "epoch_val.csv"})


def _keep_only_best_weights(run_dir: Path) -> None:
    """每组实验结束后仅保留 final/ 最优权重与 epoch_val.csv。"""
    if not run_dir.is_dir():
        return
    for child in run_dir.iterdir():
        if child.name in _KEEP_RUN_ARTIFACTS:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: margin × lambda grid sweep with early stopping")
    parser.add_argument(
        "--margins", type=str,
        default=",".join(str(m) for m in DEFAULT_MARGINS),
        help=f"逗号分隔的 margin 值（默认 {DEFAULT_MARGINS}）",
    )
    parser.add_argument(
        "--lambdas", type=str,
        default=",".join(str(l) for l in DEFAULT_LAMBDAS),
        help=f"逗号分隔的 lambda 值（默认 {DEFAULT_LAMBDAS}）",
    )
    parser.add_argument("--max-epochs", type=int, default=10, help="每组最大训练 epoch 数")
    parser.add_argument(
        "--patience", type=int, default=0,
        help=f"NDCG 连续多少次 val 未提升则停止（默认 {EARLY_STOP_PATIENCE}）",
    )
    parser.add_argument(
        "--eval-every-steps", type=int, default=EVAL_EVERY_STEPS_AUTO,
        help="-1=每 epoch 半程自动 val；0=仅 epoch 末尾；>0=每 N 步",
    )
    parser.add_argument("--no-mid-eval", action="store_true", help="关闭 epoch 中期 val")
    parser.add_argument("--lr", type=float, default=2e-5, help="固定学习率")
    parser.add_argument(
        "--csv-mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="grid_sweep.csv 写入模式：overwrite=覆盖，append=追加",
    )
    args = parser.parse_args()
    eval_every_steps = 0 if args.no_mid_eval else args.eval_every_steps

    ensure_dirs()
    margins = [float(x) for x in args.margins.split(",") if x.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    patience = get_early_stop_patience(args.patience)

    fields = [
        "pair_lambda",
        "pair_margin",
        "best_epoch",
        "best_ndcg",
        "recall_at_k",
        "mrr",
        "max_epochs",
        "stopped_early",
        "eval_k",
        "total",
        "lr",
        "model_path",
    ]
    rows: list[dict] = []
    total_runs = len(margins) * len(lambdas)
    run_idx = 0

    csv_mode = "a" if args.csv_mode == "append" else "w"
    write_header = not (args.csv_mode == "append" and GRID_SWEEP_CSV.is_file() and GRID_SWEEP_CSV.stat().st_size > 0)
    with GRID_SWEEP_CSV.open(csv_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        f.flush()
        for margin in margins:
            for lam in lambdas:
                run_idx += 1
                run_dir = CHECKPOINTS / f"grid_m{margin}_l{lam}"
                if run_dir.is_dir():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(run_dir / "final")
                epoch_val_csv = str(run_dir / "epoch_val.csv")

                cfg = CrossEncoderTrainConfig(
                    pair_lambda=lam,
                    pair_margin=margin,
                    epochs=args.max_epochs,
                    lr=args.lr,
                    save_path=save_path,
                    checkpoint_dir=str(run_dir),
                    epoch_val_csv=epoch_val_csv,
                    eval_top_k=NDCG_EVAL_K,
                    eval_split="val",
                    early_stop_enabled=True,
                    early_stop_patience=patience,
                    eval_every_steps=eval_every_steps,
                )
                print(f"\n{'='*60}")
                print(
                    f"[Grid {run_idx}/{total_runs}] margin={margin}  lambda={lam}  "
                    f"max_epochs={args.max_epochs}  patience={patience}"
                )
                print(f"{'='*60}")
                train_cross_encoder(cfg)
                _keep_only_best_weights(run_dir)

                best_epoch = ""
                best_ndcg = -1.0
                best_mrr = 0.0
                best_recall = 0.0
                best_total = 0
                best_model_path = save_path  # 仅保留 run_dir/final/
                stopped_early = False
                if os.path.isfile(epoch_val_csv):
                    with open(epoch_val_csv, encoding="utf-8", newline="") as ef:
                        for er in csv.DictReader(ef):
                            if int(er.get("for_early_stop", 1)) == 0:
                                continue
                            ndcg_val = float(er.get("ndcg", 0))
                            if ndcg_val > best_ndcg:
                                best_ndcg = ndcg_val
                                best_epoch = er["epoch"]
                                best_mrr = float(er.get("mrr", 0))
                                best_recall = float(er.get("recall_at_k", 0))
                                best_total = int(er.get("total", 0))
                                best_model_path = er.get("model_path", save_path)
                            if int(er.get("stopped_early", 0)):
                                stopped_early = True

                row = {
                    "pair_lambda": lam,
                    "pair_margin": margin,
                    "best_epoch": best_epoch,
                    "best_ndcg": round(best_ndcg, 6),
                    "recall_at_k": round(best_recall, 6),
                    "mrr": round(best_mrr, 6),
                    "max_epochs": args.max_epochs,
                    "stopped_early": int(stopped_early),
                    "eval_k": NDCG_EVAL_K,
                    "total": best_total,
                    "lr": args.lr,
                    "model_path": best_model_path,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()

                status = "early_stopped" if stopped_early else f"ran_{args.max_epochs}_epochs"
                print(
                    f"[Grid {run_idx}/{total_runs}] margin={margin} lambda={lam} "
                    f"best_epoch={best_epoch} NDCG@{NDCG_EVAL_K}={best_ndcg:.4f} "
                    f"MRR={best_mrr:.4f} Recall@{NDCG_EVAL_K}={best_recall:.2%} "
                    f"({status})"
                )

    best_row = max(rows, key=lambda r: r["best_ndcg"])
    print(f"\n{'='*60}")
    print(f"Grid Sweep 完成，共 {total_runs} 组实验")
    print(f"最佳组合: margin={best_row['pair_margin']}  lambda={best_row['pair_lambda']}")
    print(
        f"  best_epoch={best_row['best_epoch']}  "
        f"NDCG@{NDCG_EVAL_K}={best_row['best_ndcg']:.4f}  "
        f"MRR={best_row['mrr']:.4f}  "
        f"Recall@{NDCG_EVAL_K}={best_row['recall_at_k']:.2%}  "
        f"stopped_early={best_row['stopped_early']}"
    )
    print(f"已写入: {GRID_SWEEP_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
