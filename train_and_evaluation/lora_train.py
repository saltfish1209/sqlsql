import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ----------------- 配置区 -----------------
MODEL_PATH = "../models/qwen3.5-9B"  # 本地模型路径
DATA_PATH = "../data/nl2sql_train.jsonl"  # 你的数据集
OUTPUT_DIR = "../saves/qwen3.5-9b-nl2sql-lora"  # 权重保存路径


def train():
    # 提前创建保存目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在加载 Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"正在加载基础模型 (bfloat16): {MODEL_PATH}")
    # 在 A100 上使用 bfloat16 获得最大吞吐量
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 修复 gradient_checkpointing 可能导致的报错
    model.enable_input_require_grads()

    # === LoRA 配置 ===
    print("配置 LoRA 适配器...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 数据集加载与处理 ===
    print("加载与处理数据集...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # TRL 的数据格式拆分
    def format_prompts(example):
        messages = example["messages"]

        # 构建 Prompt：提取除最后一条外的所有内容（System + User），自动添加 Assistant 引导符
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        # 构建 Completion：只取 Assistant 的回答，并在末尾加上 eos_token 闭环
        assistant_content = messages[-1]["content"]
        completion_text = assistant_content + tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "completion": completion_text
        }

    # 映射数据集，并强制移除原始的所有列（防止 Trainer 底层张量拼接报错）
    dataset = dataset.map(
        format_prompts,
        remove_columns=dataset.column_names,
        desc="Formatting prompts and completions"
    )

    # === SFTConfig 配置 ===
    print("配置 SFT 训练参数...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=4,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",

        # --- 核心微调配置 ---
        max_seq_length=1536,
        completion_only_loss=True,  # 开启只计算回答的 Loss
        packing=False  # 关闭 packing
    )

    # === 启动训练 ===
    print("初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        # 新版不再需要传 data_collator 和 dataset_text_field 参数
    )

    print("开始执行 LoRA 训练...")
    trainer.train()

    # === 保存最终结果 ===
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成！LoRA 权重已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()