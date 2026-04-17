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

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

TRAIN_FILE = os.path.join(os.path.dirname(__file__), "cross_encoder_train_data.json")
BASE_MODEL = str(settings.reranker_base_model)
SAVE_PATH = str(settings.cross_encoder_model)


def train():
    if not os.path.isfile(TRAIN_FILE):
        print(f"[ERROR] 训练数据不存在: {TRAIN_FILE}，请先运行 prepare_data.py")
        return
    if not os.path.isdir(BASE_MODEL):
        print(f"[ERROR] 基础模型不存在: {BASE_MODEL}")
        return

    print(f"加载基座 Reranker: {BASE_MODEL}")
    model = CrossEncoder(
        BASE_MODEL, num_labels=1, max_length=256, trust_remote_code=True,
    )

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = [
        InputExample(texts=[it["question"], it["column"]], label=float(it["label"]))
        for it in raw
    ]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    epochs = 3
    warmup = int(len(loader) * epochs * 0.1)

    print(f"训练 {epochs} epochs，总步数 {len(loader) * epochs}")
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=warmup,
        output_path=SAVE_PATH,
        use_amp=True,
        show_progress_bar=True,
    )
    model.save(SAVE_PATH)
    print(f"模型已保存: {SAVE_PATH}")


if __name__ == "__main__":
    train()
