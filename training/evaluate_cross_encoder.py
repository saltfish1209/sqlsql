"""
CrossEncoder 验证/测试评估脚本。
───────────────────────────────────────────────────────────────
读取 prepare_data.py 生成的验证集或测试集，
使用已训练好的 CrossEncoder 权重计算 Top-K Full Recall。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from training.dataset_io import read_jsonl

VAL_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_val_data.jsonl")
TEST_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.jsonl")
VAL_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_val_data.json")
TEST_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_test_data.json")
TOP_K = 6  # 与 figure/crossencoder/paths.VAL_TOP_K 一致


def get_eval_file(split: str) -> str:
    """优先返回 JSONL 路径；若不存在则回退到旧 JSON 路径，方便迁移期共存。"""
    split_norm = split.strip().lower()
    if split_norm == "val":
        return VAL_FILE_JSONL if os.path.isfile(VAL_FILE_JSONL) else VAL_FILE_JSON
    if split_norm == "test":
        return TEST_FILE_JSONL if os.path.isfile(TEST_FILE_JSONL) else TEST_FILE_JSON
    raise ValueError(f"不支持的数据集类型: {split}，仅支持 val/test")


def _load_eval_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return list(read_jsonl(path))
    import json as _json

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _json.load(f)


def evaluate(split: str = "val", top_k: int = TOP_K) -> None:
    from training.cross_encoder_eval import evaluate_cross_encoder as _eval_metrics

    model_path = str(settings.cross_encoder_model)
    eval_file = get_eval_file(split)

    print(f"开始评估 CrossEncoder ({split})...")
    print(f"模型路径: {model_path}")
    print(f"评估文件: {eval_file}")
    print(f"Top-K: {top_k}")

    if not os.path.isdir(model_path):
        print(f"[ERROR] 模型未找到: {model_path}，请先运行 train_cross_encoder.py")
        return
    if not os.path.isfile(eval_file):
        print(f"[ERROR] 评估数据文件不存在: {eval_file}，请先运行 prepare_data.py")
        return

    try:
        metrics = _eval_metrics(model_path, split=split, top_k=top_k)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"\n{'=' * 50}")
    print(
        f"{split.upper()} Top-{top_k} Recall@K: "
        f"{metrics['recall_at_k']:.2%} ({metrics['success']}/{metrics['total']})"
    )


if __name__ == "__main__":
    # 在这里切换评估集：可选 "val" 或 "test"
    EVAL_SPLIT = "val"
    evaluate(split=EVAL_SPLIT, top_k=TOP_K)
