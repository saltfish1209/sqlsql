"""
CrossEncoder 验证/测试评估脚本。
───────────────────────────────────────────────────────────────
读取 prepare_data.py 生成的验证集或测试集，
使用已训练好的 CrossEncoder 权重计算 NDCG@K / MRR / Recall@K。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

EVAL_K = 6


def evaluate(split: str = "val", eval_k: int = EVAL_K) -> None:
    from training.cross_encoder_eval import evaluate_cross_encoder as _eval_metrics, get_eval_file

    model_path = str(settings.cross_encoder_model)
    eval_file = get_eval_file(split)

    print(f"开始评估 CrossEncoder ({split})...")
    print(f"模型路径: {model_path}")
    print(f"评估文件: {eval_file}")
    print(f"评估 K: {eval_k}")

    if not os.path.isdir(model_path):
        print(f"[ERROR] 模型未找到: {model_path}，请先运行 train_cross_encoder.py")
        return
    if not os.path.isfile(eval_file):
        print(f"[ERROR] 评估数据文件不存在: {eval_file}，请先运行 prepare_data.py")
        return

    try:
        metrics = _eval_metrics(model_path, split=split, top_k=eval_k)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"\n{'=' * 50}")
    print(f"{split.upper()} 评估结果 (K={eval_k}):")
    print(f"  NDCG@{eval_k}:    {metrics['ndcg']:.4f}")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print(
        f"  Recall@{eval_k}:  {metrics['recall_at_k']:.2%} "
        f"({metrics['success']}/{metrics['total']})"
    )


if __name__ == "__main__":
    EVAL_SPLIT = "val"
    evaluate(split=EVAL_SPLIT, eval_k=EVAL_K)
