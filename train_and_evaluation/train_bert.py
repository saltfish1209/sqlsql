from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json
import torch
from pathlib import Path

TRAIN_FILE = "cross_encoder_train_data.json"
MODEL_SAVE_PATH = "./my_schema_pruner_model"
BASE_DIR = Path(__file__).parent.parent

LOCAL_MODEL_PATH = str(BASE_DIR / "models" / "bert-base-chinese")


def train():
    # 1. 加载模型
    print(f"正在从本地加载模型: {LOCAL_MODEL_PATH}")
    model = CrossEncoder(LOCAL_MODEL_PATH, num_labels=1, max_length=128)

    # 2. 读取切分好的训练数据
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    examples = []
    for item in raw_data:
        examples.append(InputExample(
            texts=[item['question'], item['column']],
            label=float(item['label'])
        ))

    # 3. 训练配置
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

    #  3 个 Epoch
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    print(f"开始训练，总步数: {len(train_dataloader) * num_epochs}")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=MODEL_SAVE_PATH,
        use_amp=True,  # 2060 必备
        show_progress_bar=True
    )
    print("训练完成！")
    model.save(MODEL_SAVE_PATH)


if __name__ == '__main__':
    train()