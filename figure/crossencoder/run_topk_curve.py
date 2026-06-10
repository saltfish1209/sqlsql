"""
在选定模型上对多档 K 评估，输出 results/topk_curve.csv（画 Recall@K 曲线）。

用法:
  python figure/crossencoder/run_topk_curve.py
  python figure/crossencoder/run_topk_curve.py --model-path models/my_schema_pruner_model
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.settings import settings
from figure.crossencoder.paths import TOPK_CURVE_CSV, ensure_dirs
from training.cross_encoder_eval import evaluate_cross_encoder_multi_k

DEFAULT_KS = (3, 4, 5, 6, 8, 10, 12, 15, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@K curve CSV for one model")
    parser.add_argument("--model-path", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"))
    parser.add_argument("--ks", type=str, default="", help="逗号分隔，如 3,6,10,20")
    args = parser.parse_args()

    ensure_dirs()
    ks = DEFAULT_KS
    if args.ks.strip():
        ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    rows = evaluate_cross_encoder_multi_k(args.model_path, split=args.split, top_ks=ks)
    fields = ["model_path", "split", "top_k", "total", "success", "recall_at_k", "ndcg"]
    with TOPK_CURVE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})

    print(f"模型: {args.model_path}")
    for r in rows:
        print(f"  K={r['top_k']:2d}  Recall@K={r['recall_at_k']:.2%}  NDCG@K={r.get('ndcg', 0):.4f}")
    print(f"已写入: {TOPK_CURVE_CSV}")


if __name__ == "__main__":
    main()
