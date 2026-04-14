import pandas as pd
import json
import re
import os
import random

INPUT_CSV = "../data/train_dataset_with_sql_and_slots.csv"
INPUT_SCHEMA_FILE = "../data/m_schema.json"  # 全量 schema

# 切分后的输出路径
TRAIN_OUTPUT_JSONL = "../data/nl2sql_train_for_lora.jsonl"
VAL_OUTPUT_JSONL = "../data/nl2sql_val_for_lora.jsonl"
TEST_OUTPUT_JSONL = "../data/nl2sql_test_for_lora.jsonl"

# 负样本数量配置
NUM_HARD_NEG_QUESTION = 7
NUM_EASY_NEG_QUESTION = 3
NUM_HARD_NEG_ANSWER = 3
NUM_EASY_NEG_ANSWER = 2

# ---------------- 负样本生成函数 ----------------
def get_similarity(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def generate_negatives(target_cols, all_cols, num_hard, num_easy, keywords=None):
    if keywords is None:
        keywords = ['id', 'time', 'date', 'name', 'type', 'status', 'code', 'amount', 'num', 'count']

    potential_negatives = [c for c in all_cols if c not in target_cols]
    if not potential_negatives:
        return []

    candidates = []
    for neg in potential_negatives:
        max_sim = max([get_similarity(neg, t) for t in target_cols] + [0])
        for t in target_cols:
            if any(kw in t.lower() for kw in keywords) and any(kw in neg.lower() for kw in keywords):
                max_sim += 0.2
                break
        candidates.append((neg, max_sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    hard_negatives = [x[0] for x in candidates[:num_hard]]
    remaining = [c for c in potential_negatives if c not in hard_negatives]
    easy_negatives = random.sample(remaining, min(len(remaining), num_easy))
    return hard_negatives + easy_negatives

# ---------------- 提取列函数 ----------------
def extract_cols_from_template(template_str):
    if pd.isna(template_str): return []
    matches = re.findall(r'\{([^}]+)\}', str(template_str))
    cols = []
    for m in matches:
        if ',' in m or '，' in m:
            cols.extend([x.strip() for x in re.split('[,，]', m)])
        else:
            cols.append(m.strip())
    return list(set(cols))

# ---------------- 构建精简 Schema ----------------
def build_m_schema(selected_columns, full_schema_meta):
    schema_lines = []
    for col in selected_columns:
        meta = next((item for item in full_schema_meta if item['column_name'] == col), None)
        if meta:
            examples_str = ', '.join(meta.get('examples', [])[:3])
            schema_lines.append(
                f"({meta['column_name']}: {meta['data_type']}, {meta['column_description']}, Examples: [{examples_str}])")
    return schema_lines

# ---------------- 数据处理函数 ----------------
def process_dataframe(df_slice, full_schema_meta, all_cols):
    output_data = []
    for idx, row in df_slice.iterrows():
        question = row.get('生成问题', '')
        q_template = row.get('问题模版', '')
        a_template = row.get('回答模版', '')
        sql_query = str(row.get('SQL语句')).strip()

        try:
            slot_info = json.loads(row.get('槽位信息JSON', '{}'))
        except:
            slot_info = {}

        condition_cols = list(slot_info.keys())
        answer_cols = extract_cols_from_template(a_template)
        if re.fullmatch(r'count\{\s*\}', a_template.strip() if a_template else ''):
            answer_cols = []

        selected_columns = list(dict.fromkeys(condition_cols + answer_cols))

        schema_lines = build_m_schema(selected_columns, full_schema_meta)
        q_neg_schema = build_m_schema(
            generate_negatives(condition_cols, all_cols, NUM_HARD_NEG_QUESTION, NUM_EASY_NEG_QUESTION),
            full_schema_meta)
        a_neg_schema = build_m_schema(
            generate_negatives(answer_cols, all_cols, NUM_HARD_NEG_ANSWER, NUM_EASY_NEG_ANSWER), full_schema_meta)

        all_schema_lines = list(set(schema_lines + q_neg_schema + a_neg_schema))
        random.shuffle(all_schema_lines)
        schema_text = "\n".join(all_schema_lines)

        system_prompt = (
            "你是一名SQL专家。请参考以下内容生成SQL。\n"
            "采用sqlite，不需要加上数据库名。\n\n"
            "【表结构信息】\n"
            f"{schema_text}\n"
            "请输出SQL，用```sql ... ```包裹,不需要解释和输出其他内容。"
        )

        user_prompt = f"【用户问题】\n{question}"

        lora_item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": sql_query}
            ]
        }
        output_data.append(lora_item)
    return output_data

def save_jsonl(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ---------------- 主程序 ----------------
def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)

    with open(INPUT_SCHEMA_FILE, 'r', encoding='utf-8') as f:
        full_schema_meta = json.load(f)

    all_cols = [col['column_name'] for col in full_schema_meta]

    # 1. 过滤数据
    df['SQL验证状态'] = df['SQL验证状态'].astype(str).str.strip().str.upper()
    df_filtered = df[(df['SQL验证状态'] == 'MATCH') & (df['SQL语句'].notna()) & (df['SQL语句'] != 'nan')].copy()
    match_len = len(df_filtered)
    print(f"过滤出 {match_len} 条有效 MATCH 数据。")

    # 2. 随机打乱数据 (保持与 cross_encoder 一致的 random_state=42)
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    # 3. 动态切分数据 (80% Train, 10% Val, 10% Test)
    train_end = int(match_len * 0.8)
    val_end = int(match_len * 0.9)

    df_train = df_filtered.iloc[:train_end]
    df_val = df_filtered.iloc[train_end:val_end]
    df_test = df_filtered.iloc[val_end:]

    print(f"开始处理训练集 ({len(df_train)} 行)...")
    train_data = process_dataframe(df_train, full_schema_meta, all_cols)
    save_jsonl(train_data, TRAIN_OUTPUT_JSONL)

    print(f"开始处理验证集 ({len(df_val)} 行)...")
    val_data = process_dataframe(df_val, full_schema_meta, all_cols)
    save_jsonl(val_data, VAL_OUTPUT_JSONL)

    print(f"开始处理测试集 ({len(df_test)} 行)...")
    test_data = process_dataframe(df_test, full_schema_meta, all_cols)
    save_jsonl(test_data, TEST_OUTPUT_JSONL)

    print(f"✅ 生成完成！")
    print(f"训练数据: {TRAIN_OUTPUT_JSONL} ({len(train_data)} 条)")
    print(f"验证数据: {VAL_OUTPUT_JSONL} ({len(val_data)} 条)")
    print(f"测试数据: {TEST_OUTPUT_JSONL} ({len(test_data)} 条)")

if __name__ == '__main__':
    main()