import pandas as pd
import re
import random
import json
import os
from difflib import SequenceMatcher

# ================= 配置区域 =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
INPUT_TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset_with_sql_and_slots.csv")
INPUT_RAW_SCHEMA_FILE = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")
# 包含所有列名的原始宽表
REQUIRED_COLUMNS = ['生成问题', '问题模版', '回答模版']

# 输出文件路径
TRAIN_OUTPUT_FILE = "cross_encoder_train_data.json"
TEST_OUTPUT_FILE = "cross_encoder_test_data.json"

# 采样配置
NUM_HARD_NEGATIVES = 3  # 每个问题生成的困难负样本数量
NUM_EASY_NEGATIVES = 7  # 每个问题生成的简单负样本数量
SPLIT_INDEX = 1006

# 总计每个正样本会对应 10 个负样本 (1:10 比例)

# ===========================================

def clean_col_name(col_str):
    """清洗列名，去除可能存在的空格"""
    return str(col_str).strip()


def extract_cols_from_template(template_str):
    """
    从模版字符串中提取 {} 内的内容
    例如: "{计划批次名称}”批次的采购模式" -> ['计划批次名称']
    支持: sum{金额}, listdown{金额} 等复杂情况的提取
    """
    if pd.isna(template_str):
        return []

    # 正则提取 {} 内部的内容
    matches = re.findall(r'\{([^}]+)\}', str(template_str))

    cleaned_cols = []
    for m in matches:
        # 处理可能的复杂写法，如 "金额 | 数量 > 0" 或 "sum{金额}" 嵌套情况
        # 这里假设 {} 里面核心就是列名，如果有管道符|，通常左边是列名
        core_col = m.split('|')[0].strip()
        # 处理逗号分隔的情况 {A, B}
        sub_cols = core_col.split('，')  # 中文逗号
        for sc in sub_cols:
            final_c = sc.split(',')[0].strip()  # 英文逗号
            if final_c:
                cleaned_cols.append(final_c)

    return list(set(cleaned_cols))  # 去重


def get_similarity(a, b):
    """计算两个字符串的相似度 (0~1)"""
    return SequenceMatcher(None, a, b).ratio()


def generate_negatives(target_cols, all_cols, num_hard, num_easy):
    """
    生成负样本
    target_cols: 正确列列表
    all_cols: 所有可用列列表
    """
    potential_negatives = [c for c in all_cols if c not in target_cols]

    if not potential_negatives:
        return []

    # --- 1. 挖掘困难负样本 (基于相似度) ---
    # 对每个正确列，去所有负样本里找名字最像的
    candidates = []
    for neg in potential_negatives:
        max_sim = 0
        for target in target_cols:
            sim = get_similarity(target, neg)
            if sim > max_sim:
                max_sim = sim

        # 增加一个逻辑：如果包含相同的关键字（如“金额”、“价”），加分
        keywords = ['金额', '价', '数量', '码', '日期', '名称']
        for kw in keywords:
            if any(kw in t for t in target_cols) and kw in neg:
                max_sim += 0.2  # 人工加权，视为更困难的混淆项

        candidates.append((neg, max_sim))

    # 按相似度降序排列
    candidates.sort(key=lambda x: x[1], reverse=True)
    hard_negatives = [x[0] for x in candidates[:num_hard * 2][:num_hard]]

    remaining = [c for c in potential_negatives if c not in hard_negatives]
    easy_negatives = random.sample(remaining, min(len(remaining),num_easy))

    return hard_negatives + easy_negatives


def process_data(df_slice, all_cols, is_training = True):
    dataset = []
    for idx , row in df_slice.iterrows():
        question = row.get('生成问题', '')
        q_template = row.get('问题模版', '')
        a_template = row.get('回答模版', '')

        if pd.isna(question) or str(question).strip() == '':
            continue

        pos_cols = list(set(extract_cols_from_template(q_template) + extract_cols_from_template(q_template)))
        valid_targets = [c for c in pos_cols if c in all_cols]

        if not valid_targets:
            continue


        if is_training:
            # 1.正样本
            for col in valid_targets:
                dataset.append({'question': question, 'column': col, 'label': 1})

            # 2.负样本
            neg_cols = generate_negatives(valid_targets, all_cols, NUM_HARD_NEGATIVES, NUM_EASY_NEGATIVES)
            for col in neg_cols:
                dataset.append({"question": question, "column": col, "label": 0})

        # 如果是测试集，我们只需要知道“这一题的标准答案是什么”，后续由评估脚本去跑所有列
        else:
            dataset.append({
                "question": question,
                "gold_columns": valid_targets  # 记录标准答案列表
            })

    return dataset


def main():
    print("🚀 开始处理数据...")

    # 1. 读取 Schema
    if not os.path.exists(INPUT_RAW_SCHEMA_FILE):
        print(f"❌ 找不到原始数据文件")
        return
    df_raw = pd.read_csv(INPUT_RAW_SCHEMA_FILE, nrows=1)
    ALL_COLUMNS = [clean_col_name(c) for c in df_raw.columns]

    # 2. 读取全量数据
    if not os.path.exists(INPUT_TRAIN_FILE):
        print(f"❌ 找不到训练数据文件")
        return
    df_full = pd.read_csv(INPUT_TRAIN_FILE, usecols=REQUIRED_COLUMNS, engine='python', encoding="utf-8")

    # 3. 切分数据 (前1006行 vs 剩余)
    # iloc[0:1006] 包含索引 0 到 1005，正好是前 1006 行
    df_train_slice = df_full.iloc[:SPLIT_INDEX]
    df_test_slice = df_full.iloc[SPLIT_INDEX:]

    print(f"原始数据: {len(df_full)} 行")
    print(f"训练集 (前{SPLIT_INDEX}行): {len(df_train_slice)} 行")
    print(f"测试集 (剩余): {len(df_test_slice)} 行")

    # 4. 生成训练 JSON (带负样本)
    train_data = process_data(df_train_slice, ALL_COLUMNS, is_training=True)
    with open(TRAIN_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 5. 生成测试 JSON (只存正样本List，用于验证)
    test_data = process_data(df_test_slice, ALL_COLUMNS, is_training=False)
    with open(TEST_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(
        f"✅ 生成完毕！\n训练数据: {TRAIN_OUTPUT_FILE} ({len(train_data)} 对)\n测试数据: {TEST_OUTPUT_FILE} ({len(test_data)} 条)")


if __name__ == '__main__':
    main()
