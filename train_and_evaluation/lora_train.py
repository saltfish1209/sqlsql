import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 配置区域 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
QA_CSV_FILE = os.path.join(DATA_DIR, "train_dataset_with_sql_and_slots.csv")  # 你现有生成的 CSV
OUTPUT_DIR = os.path.join(BASE_DIR, "lora_output")
BASE_MODEL = "qwen/qwen-14b"  # 可根据你实际 LLM 调整
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512
TARGET_MODULES = ["q_proj", "v_proj"]  # LoRA 调整的模块

# ==================================================

# 1. 读取问答对 CSV
print("读取问答对 CSV...")
df = pd.read_csv(QA_CSV_FILE, dtype=str)

# 仅保留有效数据
valid_df = df[df['是否有效'] == True]
# 使用生成问题和标准答案作为 prompt / completion
qa_list = []
for idx, row in valid_df.iterrows():
    prompt = str(row['生成问题']).strip()
    completion = str(row['生成结果']).strip()
    if prompt and completion:
        qa_list.append({'prompt': prompt, 'completion': completion})

print(f"有效样本数量: {len(qa_list)}")

# 2. 转换为 HuggingFace Dataset
dataset = Dataset.from_list([{'input_text': d['prompt'], 'target_text': d['completion']} for d in qa_list])

# 3. Tokenizer + 编码
print("🔑 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True
    )
    labels = tokenizer(
        examples['target_text'],
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# 4. 加载基础模型
print(f"加载基础模型: {BASE_MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# 5. 配置 LoRA
print("⚡ 配置 LoRA 微调...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=50,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 8. 开始训练
print("开始 LoRA 微调训练...")
trainer.train()

# 9. 保存 LoRA 权重
print(f"保存 LoRA 权重到 {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
print("✅ 微调完成")
