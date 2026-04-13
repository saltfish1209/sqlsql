from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json
import torch

TRAIN_FILE = "cross_encoder_train_data.json"
MODEL_SAVE_PATH = "./my_schema_pruner_model"


def train():
    # 1. 加载模型
    model = CrossEncoder('bert-base-chinese', num_labels=1, max_length=128)

    # 2. 读取我们刚才切分好的训练数据
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

    # 推荐 3 个 Epoch
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