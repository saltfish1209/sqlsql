"""
根据 topk_curve.csv（若无则现场评估）写出 topk_choice.csv，给出部署用 K。

用法:
  python figure/crossencoder/run_topk_curve.py --model-path models/...
  python figure/crossencoder/run_pick_topk.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.settings import settings
from figure.crossencoder.k_selection import pick_top_k_from_curve
from figure.crossencoder.paths import GOLD_STATS_CSV, TOPK_CHOICE_CSV, TOPK_CURVE_CSV, ensure_dirs
from training.cross_encoder_eval import evaluate_cross_encoder_multi_k, gold_column_stats


def _load_curve_csv(path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick operating Top-K from curve")
    parser.add_argument("--model-path", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--recall-ratio", type=float, default=0.95)
    args = parser.parse_args()

    ensure_dirs()
    stats = gold_column_stats(args.split)
    rows = _load_curve_csv(TOPK_CURVE_CSV)
    rows = [r for r in rows if r.get("model_path", args.model_path) == args.model_path]
    if not rows:
        rows = evaluate_cross_encoder_multi_k(args.model_path, split=args.split)
        fields = ["model_path", "split", "top_k", "total", "success", "recall_at_k"]
        with TOPK_CURVE_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})

    pick = pick_top_k_from_curve(
        rows,
        gold_p95=stats["p95"],
        recall_ratio_of_max=args.recall_ratio,
    )
    out = {
        "model_path": args.model_path,
        "split": args.split,
        "gold_p95": stats["p95"],
        "gold_max": stats["max"],
        "recommended_k_primary": stats["recommended_k_primary"],
        "recommended_k_operational": stats["recommended_k_operational"],
        "chosen_k_deploy": pick["chosen_k"],
        "chosen_reason": pick["reason"],
        "max_recall_on_curve": round(pick["max_recall"], 6),
        "recall_at_chosen": round(pick.get("recall_at_chosen", 0.0), 6),
        "note": (
            "k_primary: epoch/lambda/margin 扫参时固定; "
            "chosen_k_deploy: 曲线 elbow; "
            "k_operational: 对齐 settings.candidate_top_k"
        ),
    }
    with TOPK_CHOICE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow(out)

    print(f"gold p95={stats['p95']}  k_primary={stats['recommended_k_primary']}  "
          f"k_operational={stats['recommended_k_operational']}")
    print(f"部署推荐 K={pick['chosen_k']}  ({pick['reason']})")
    print(f"已写入: {TOPK_CHOICE_CSV}")


if __name__ == "__main__":
    main()
