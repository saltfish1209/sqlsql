"""使用统一 split 数据对 Gemma 4 12B 进行纯 LoRA 微调。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForMultimodalLM, AutoProcessor
from trl import SFTConfig, SFTTrainer

from config.settings import settings
from pipeline.generator import SQLGenerator
from pipeline.prompt_rules import aggregation_rule_text
from training.dataset_io import load_split_dataframes


# 默认路径：可通过环境变量覆盖，输出目录名始终跟随基座模型目录名。
MODEL_PATH = os.getenv("LORA_MODEL_PATH", "/data2/yj/models/gemma-4-12B-it")
OUTPUT_ROOT = Path(os.getenv("LORA_OUTPUT_ROOT", "/data2/yj/models/loramodels"))
OUTPUT_DIR = OUTPUT_ROOT / Path(MODEL_PATH.rstrip("/\\")).name

MAX_LENGTH = 8192
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4


def _load_full_schema_prompt() -> str:
    schema_path = Path(settings.schema_json_path)
    with schema_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return SQLGenerator.build_m_schema_prompt(
        selected_columns=metadata,
        all_metadata=metadata,
        table_name=settings.table_name,
    )


def _build_generation_prompt(question: str, schema_prompt: str) -> str:
    highlighted_question = SQLGenerator.highlight_question_for_prompt(question)
    diversity_rule = (
        "多候选要求：每条 SQL 必须对应用户问题中的明确查询意图；可以在 SELECT、DISTINCT、"
        "聚合表达方式上做合理差异，但不得为了覆盖候选字段而遍历生成无问题依据的 SELECT/WHERE。"
    )
    direct_hint = "直接根据 Schema 和问题原文生成最简 SQL。"
    return (
        "# Role\n你是一名 SQL 专家。请只基于给定 Schema 生成 SQLite SQL。\n\n"
        f"## Schema\n{schema_prompt}\n\n"
        f"## 用户问题\n{highlighted_question}\n\n"
        "## 生成规则\n"
        f"{SQLGenerator._sql_generation_rules()}"
        f"{diversity_rule}\n"
        f"{aggregation_rule_text()}"
        '\n\n## 输出格式\n只输出 JSON：{"sql": "SELECT ..."}。\n'
        f"\n[路径提示:direct-1] {direct_hint}\n"
    )


def _build_dataset(dataframe, schema_prompt: str, split_name: str) -> Dataset:
    required_columns = {"生成问题", "SQL语句"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{split_name} 缺少字段: {sorted(missing)}")

    records = []
    for row in dataframe.to_dict(orient="records"):
        question = str(row["生成问题"]).strip()
        sql = str(row["SQL语句"]).strip()
        if not question or not sql:
            raise ValueError(f"{split_name} 存在空的生成问题或 SQL语句")
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _build_generation_prompt(question, schema_prompt),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"sql": sql}, ensure_ascii=False),
                    },
                ]
            }
        )

    if not records:
        raise ValueError(f"{split_name} 数据为空")
    return Dataset.from_list(records)


class GemmaAssistantCollator:
    """使用 Gemma 官方聊天模板，仅对 assistant 输出计算损失。"""

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        for example in examples:
            messages = example["messages"]
            full_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        batch = self.processor(
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        for index in range(labels.size(0)):
            prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
            full_length = int(batch["attention_mask"][index].sum().item())
            if prompt_length >= full_length:
                raise ValueError(f"样本超过 max_length={self.max_length}，assistant 输出被截断")
            if not torch.equal(
                batch["input_ids"][index, :prompt_length],
                prompt_batch["input_ids"][index, :prompt_length],
            ):
                raise ValueError("Gemma 聊天模板前缀不一致，无法正确构造 assistant loss mask")
            labels[index, :prompt_length] = -100

        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def train() -> None:
    splits = load_split_dataframes(settings.data_dir)
    if splits is None:
        raise FileNotFoundError(
            "缺少 data/train_split.jsonl、val_split.jsonl 或 test_split.jsonl；"
            "请先运行 python -m training.run_full_split_pipeline"
        )
    train_df, val_df, _ = splits
    if train_df.empty or val_df.empty:
        raise ValueError("train_split.jsonl 或 val_split.jsonl 为空")

    schema_prompt = _load_full_schema_prompt()
    train_dataset = _build_dataset(train_df, schema_prompt, "train")
    eval_dataset = _build_dataset(val_df, schema_prompt, "val")

    if not torch.cuda.is_available():
        raise RuntimeError("Gemma 4 12B LoRA 训练需要 CUDA GPU")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=MAX_LENGTH,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=processor,
        data_collator=GemmaAssistantCollator(processor, MAX_LENGTH),
    )

    print(f"基座模型: {MODEL_PATH}")
    print(f"训练数据: {len(train_dataset)} | 验证数据: {len(eval_dataset)}")
    print(f"LoRA 输出: {OUTPUT_DIR}")
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    train()
