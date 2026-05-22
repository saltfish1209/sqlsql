"""
CrossEncoder 微调 —— 基于 jina-reranker-v3 训练 Schema Pruner。
────────────────────────────────────────────────────────────────
基座模型: jina-reranker-v3 (通用 Reranker)
微调目标: 在电力物资采购 Schema Linking 数据上做领域适配
输出模型: my_schema_pruner_model (供 schema_linker.py 加载)
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from training.dataset_io import read_jsonl

# 训练数据统一使用 JSONL，避免单文件 JSON 写入中途被截断而引发 UnicodeDecodeError。
# 旧的 cross_encoder_train_data.json 作为兼容回退路径。
TRAIN_FILE_JSONL = os.path.join(os.path.dirname(__file__), "cross_encoder_train_data.jsonl")
TRAIN_FILE_JSON = os.path.join(os.path.dirname(__file__), "cross_encoder_train_data.json")
FILTER_REPORT_FILE = os.path.join(os.path.dirname(__file__), "cross_encoder_column_filter_report.json")
BASE_MODEL = str(settings.reranker_base_model)
SAVE_PATH = str(settings.cross_encoder_model)
PAIR_MARGIN = 0.15
PAIR_LAMBDA = 0.7
DEFAULT_DATALOADER_WORKERS = 25


def _get_use_amp_from_env() -> bool:
    """
    AMP 在部分环境下会触发 bfloat16 + GradScaler 的兼容性问题。
    默认关闭，必要时可手动开启:
      NL2SQL_CE_USE_AMP=1 python training/train_cross_encoder.py
    """
    return os.getenv("NL2SQL_CE_USE_AMP", "0").strip().lower() in {"1", "true", "yes", "on"}


def _get_dataloader_workers_from_env() -> int:
    """
    DataLoader 并行 worker 数，默认 25，可通过环境变量覆盖：
      NL2SQL_CE_NUM_WORKERS=8 python training/train_cross_encoder.py
    """
    raw = os.getenv("NL2SQL_CE_NUM_WORKERS", str(DEFAULT_DATALOADER_WORKERS)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DATALOADER_WORKERS


def compute_mixed_loss(
    point_scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_pair_scores: torch.Tensor | None = None,
    neg_pair_scores: torch.Tensor | None = None,
    margin: float = PAIR_MARGIN,
    pair_lambda: float = PAIR_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.float().view(-1)
    point_scores = point_scores.float().view(-1)
    point_loss = F.binary_cross_entropy_with_logits(point_scores, labels)

    if pos_pair_scores is None or neg_pair_scores is None or pos_pair_scores.numel() == 0:
        pair_loss = point_scores.new_tensor(0.0)
    else:
        pair_loss = F.relu(float(margin) - (pos_pair_scores.view(-1) - neg_pair_scores.view(-1))).mean()
    total = point_loss + float(pair_lambda) * pair_loss
    return total, point_loss.detach(), pair_loss.detach()


def _collate_records(batch: list[dict]) -> list[dict]:
    return batch


def _forward_scores(model: CrossEncoder, pairs: list[list[str]], device: torch.device) -> torch.Tensor:
    features = model.tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=getattr(model, "max_length", 512),
        return_tensors="pt",
    )
    features = {k: v.to(device) for k, v in features.items()}
    try:
        outputs = model.model(**features, return_dict=True)
    except ValueError as exc:
        # 兼容部分 Qwen3/Reranker 权重在 batch>1 场景下依赖 pad_token_id 的检查。
        if "no padding token is defined" not in str(exc).lower():
            raise
        _ensure_padding_token(model)
        outputs = model.model(**features, return_dict=True)
    return outputs.logits.view(-1)


def _ensure_padding_token(model: CrossEncoder) -> None:
    """
    某些 reranker tokenizer 没有 pad_token，batch_size>1 且 padding=True 会报错。
    统一回退到 eos_token 作为 padding。
    """
    tokenizer = model.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise RuntimeError("Tokenizer 未定义 pad_token 且无 eos_token，无法进行 batch padding。")
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
        pad_token_id = eos_token_id

    if getattr(model.model.config, "pad_token_id", None) is None:
        model.model.config.pad_token_id = int(pad_token_id)

    generation_cfg = getattr(model.model, "generation_config", None)
    if generation_cfg is not None and getattr(generation_cfg, "pad_token_id", None) is None:
        generation_cfg.pad_token_id = int(pad_token_id)


def _resolve_train_file() -> str | None:
    if os.path.isfile(TRAIN_FILE_JSONL):
        return TRAIN_FILE_JSONL
    if os.path.isfile(TRAIN_FILE_JSON):
        print(f"[WARN] 未找到 {TRAIN_FILE_JSONL}，回退使用旧的 JSON 文件: {TRAIN_FILE_JSON}")
        return TRAIN_FILE_JSON
    return None


def _load_train_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return list(read_jsonl(path))
    # 旧 JSON 文件作为兼容回退路径；按整文件解析。
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"训练数据文件解析失败: {path}（{exc.msg}）。"
            "建议重新跑 prepare_data.py 生成 JSONL 版本。"
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(f"训练数据文件格式异常，期望 list，实际 {type(data)}: {path}")
    return data


def train():
    from training.cross_encoder_train import config_from_env, train_cross_encoder

    train_cross_encoder(config_from_env())


if __name__ == "__main__":
    train()
