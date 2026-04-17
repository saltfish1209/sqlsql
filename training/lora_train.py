"""
LoRA 微调 —— 在 Qwen 基座上训练 Text-to-SQL 适配器。
"""
from __future__ import annotations

import os
import sys

import torch
from datasets import load_dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTTrainer

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings

BASE = settings.project_root
TRAIN_PATH = str(BASE / "data" / "nl2sql_train_for_lora.jsonl")
VAL_PATH = str(BASE / "data" / "nl2sql_val_for_lora.jsonl")
OUTPUT_DIR = str(BASE / "saves" / "qwen3.5-9B-nl2sql-lora")


def _get_model_path() -> str:
    p = os.getenv("LLM_LOCAL_MODEL_PATH", "").strip()
    if not p:
        raise RuntimeError("请设置环境变量 LLM_LOCAL_MODEL_PATH 为本地基座模型目录")
    return p


def train():
    for label, fpath in [("训练数据", TRAIN_PATH), ("验证数据", VAL_PATH)]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] {label}不存在: {fpath}，请先运行 csv_to_json_for_lora.py")
            return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = _get_model_path()

    print(f"加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基础模型 (bf16): {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

    def process(example):
        msgs = example["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        f_ids = tokenizer.encode(full, add_special_tokens=False)
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}

    train_ds = dataset["train"].map(process, remove_columns=dataset["train"].column_names)
    eval_ds = dataset["validation"].map(process, remove_columns=dataset["validation"].column_names)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=4,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        args=args, processing_class=tokenizer, data_collator=collator,
    )
    print("开始 LoRA 训练...")
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成! 权重保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
