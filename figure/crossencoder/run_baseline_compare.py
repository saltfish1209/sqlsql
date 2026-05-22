"""
基座 reranker vs 微调模型，在固定 top_k 与 topk 曲线上对比 → baseline_vs_finetuned.csv

用法:
  python figure/crossencoder/run_baseline_compare.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.settings import settings
from figure.crossencoder.paths import BASELINE_COMPARE_CSV, TOPK_CURVE_CSV, ensure_dirs, get_val_top_k
from training.cross_encoder_eval import evaluate_cross_encoder, evaluate_cross_encoder_multi_k

DEFAULT_KS = (3, 4, 5, 6, 8, 10, 12, 15, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs finetuned CSV")
    parser.add_argument("--base-model", type=str, default=str(settings.reranker_base_model))
    parser.add_argument("--finetuned", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    top_k = get_val_top_k(args.top_k)

    rows: list[dict] = []
    for label, path in (("base", args.base_model), ("finetuned", args.finetuned)):
        m = evaluate_cross_encoder(path, split=args.split, top_k=top_k)
        rows.append({
            "model_kind": label,
            "model_path": path,
            "split": args.split,
            "top_k": top_k,
            "recall_at_k": round(m["recall_at_k"], 6),
            "total": m["total"],
            "success": m["success"],
        })

    fields = ["model_kind", "model_path", "split", "top_k", "recall_at_k", "total", "success"]
    with BASELINE_COMPARE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    curve_rows: list[dict] = []
    for label, path in (("base", args.base_model), ("finetuned", args.finetuned)):
        for r in evaluate_cross_encoder_multi_k(path, split=args.split, top_ks=DEFAULT_KS):
            curve_rows.append({"model_kind": label, **r})

    curve_fields = ["model_kind", "model_path", "split", "top_k", "total", "success", "recall_at_k"]
    with TOPK_CURVE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        for r in curve_rows:
            writer.writerow({k: r.get(k) for k in curve_fields})

    print(f"固定 K={top_k}")
    for r in rows:
        print(f"  {r['model_kind']:9s} Recall@K={r['recall_at_k']:.2%}")
    print(f"已写入: {BASELINE_COMPARE_CSV}")
    print(f"Recall@K 曲线: {TOPK_CURVE_CSV}")


if __name__ == "__main__":
    main()
