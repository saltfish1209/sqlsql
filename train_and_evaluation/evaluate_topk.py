from sentence_transformers import CrossEncoder
import pandas as pd
import json
import numpy as np
import os

# 配置
TEST_FILE = "cross_encoder_test_data.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
INPUT_RAW_SCHEMA_FILE = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")
MODEL_PATH = "./my_schema_pruner_model"  # 训练好的模型路径
TOP_K = 15  # 设定的阈值


def evaluate():
    print("🚀 开始评估 Top-K 全覆盖率...")

    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型未找到，请先运行训练脚本")
        return
    model = CrossEncoder(MODEL_PATH)

    # 2. 获取所有列名 (Candidate Pool)
    df_raw = pd.read_csv(INPUT_RAW_SCHEMA_FILE, nrows=1)
    all_columns = [str(c).strip() for c in df_raw.columns]
    print(f"候选项总数: {len(all_columns)} 列")

    # 3. 加载测试数据
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    success_count = 0
    total_count = 0

    for i, item in enumerate(test_data):
        question = item['question']
        gold_cols = set(item['gold_columns'])  # 真实答案集合

        if not gold_cols: continue

        # --- 核心预测逻辑 ---
        # 构造 70+ 个 (问题, 列名) 对
        predict_inputs = [[question, col] for col in all_columns]

        # 批量预测分数
        scores = model.predict(predict_inputs,batch_size=32)

        # 获取分数最高的 Top-K 个索引
        # argsort是从小到大，[::-1]反转，[:K]取前K个
        top_k_indices = np.argsort(scores)[::-1][:TOP_K]

        # 映射回列名
        pred_top_k_cols = set([all_columns[idx] for idx in top_k_indices])

        # --- 验证是否全覆盖 ---
        # 判断：标准答案是否是预测结果的子集？
        if gold_cols.issubset(pred_top_k_cols):
            success_count += 1

        total_count += 1

        # 打印部分日志看看效果
        if i < 3:
            print(f"\n[Case {i}]")
            print(f"问题: {question}")
            print(f"标准答案: {gold_cols}")
            print(f"Top-{TOP_K} 预测: {list(pred_top_k_cols)[:5]} ...")  # 只打前5个看一眼
            print(f"结果: {'✅ 成功' if gold_cols.issubset(pred_top_k_cols) else '❌ 失败'}")

    # 输出最终指标
    accuracy = success_count / total_count
    print("=" * 50)
    print(f"评估完成 (K={TOP_K})")
    print(f"测试集样本数: {total_count}")
    print(f"完全覆盖次数: {success_count}")
    print(f"Top-{TOP_K} Full Recall Rate: {accuracy:.2%}")
    print("=" * 50)



if __name__ == '__main__':
    evaluate()
