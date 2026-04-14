import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from pathlib import Path

# ----------------- 配置区 -----------------

base_path = Path(__file__).parent.parent
MODEL_PATH = str(base_path / "models" / "qwen3.5-9B")
TRAIN_DATA_PATH = str(base_path / "data" / "nl2sql_train_for_lora.jsonl")
VAL_DATA_PATH = str(base_path / "data" / "nl2sql_val_for_lora.jsonl")
OUTPUT_DIR = str(base_path / "saves" / "qwen3.5-9B-nl2sql-lora")



def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载 Tokenizer ===
    print(f"正在加载 Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Qwen 系列推荐配置
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # === 2. 加载基础模型 ===
    print(f"正在加载基础模型 (bfloat16): {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 充分利用高算力显卡的 bf16 加速
        device_map="auto",
        trust_remote_code=True
    )
    # 开启梯度检查点必需的设置，防止显存溢出
    model.resize_token_embeddings(len(tokenizer))

    model.enable_input_require_grads()

    # === 3. 配置 LoRA 适配器 ===
    print("配置 LoRA 适配器...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save = ["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 4. 数据集加载与硬核处理 (终极防报错方案) ===
    print("加载与处理数据集...")
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA_PATH, "validation": VAL_DATA_PATH})

    def process_data(example):
        messages = example["messages"]

        # 提取提示部分 (System + User)，并加上助手引导符 (add_generation_prompt=True)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        # 提取完整对话部分 (System + User + Assistant)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        # 转换为 Token ID 序列 (不自动加特殊 token，因为模板字符串里已经包含了)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # 【核心逻辑】构造 Labels：让提问部分的 Token 不参与 Loss 计算 (-100)
        # 只有 Assistant 回答部分的 Token 保留原本的 ID
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        }

    # 执行映射，并严格移除原始不需要的字符串列
    train_dataset = dataset["train"].map(process_data, remove_columns=dataset["train"].column_names, desc="处理训练集")
    eval_dataset = dataset["validation"].map(process_data, remove_columns=dataset["validation"].column_names,
                                             desc="处理验证集")

    # === 5. 配置 Data Collator ===
    # 使用标准的 Seq2Seq Collator，仅负责将 batch 内不同长度的序列 Padding 补齐，逻辑极其稳定
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Tensor 维度是 8 的倍数时，计算效率最高
        label_pad_token_id=-100  # 补齐的 Token 同样不计算 Loss
    )

    # === 6. 训练参数配置 ===
    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=4,
        save_strategy="epoch",
        eval_strategy="epoch",  # 每个 epoch 评估一次
        bf16=True,  # 开启混合精度
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        # 弃用所有 TRL 高阶参数，确保最高兼容性
    )

    # === 7. 初始化并启动训练 ===
    print("初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,  # 适配新版 API 命名
        data_collator=data_collator
    )

    print("开始执行 LoRA 训练...")
    trainer.train()

    # === 8. 保存最终模型 ===
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 训练完成！LoRA 权重与配置已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()