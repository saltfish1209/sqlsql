import pandas as pd
import json
import re
import os
import random

INPUT_CSV = "../data/train_dataset_with_sql_and_slots.csv"
OUTPUT_JSONL = "../data/nl2sql_train_for_lora.jsonl"
INPUT_SCHEMA_FILE = "../data/m_schema.json"  # 全量 schema

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
            # 模仿精简格式：(字段名: 类型, 描述, 样例)
            examples_str = ', '.join(meta.get('examples', [])[:3])
            schema_lines.append(
                f"({meta['column_name']}: {meta['data_type']}, {meta['column_description']}, Examples: [{examples_str}])")
    return schema_lines


# ---------------- 主程序 ----------------
def main():
    # 强制将所有列作为字符串读取，避免类型问题
    df = pd.read_csv(INPUT_CSV, dtype=str)

    with open(INPUT_SCHEMA_FILE, 'r', encoding='utf-8') as f:
        full_schema_meta = json.load(f)

    all_cols = [col['column_name'] for col in full_schema_meta]
    output_data = []

    for idx, row in df.iterrows():
        question = row.get('生成问题', '')
        q_template = row.get('问题模版', '')
        a_template = row.get('回答模版', '')


        sql_query = str(row.get('SQL语句')).strip()
        sql_status = str(row.get('SQL验证状态')).strip()

        # 数据清洗：如果没有生成 SQL，或者 SQL 验证状态为 False/0/失败，则跳过不用于微调
        if not sql_query or sql_query == 'nan':
            continue
        if sql_status.upper() != "MATCH":
            continue

        try:
            slot_info = json.loads(row.get('槽位信息JSON', '{}'))
        except:
            slot_info = {}

        condition_cols = list(slot_info.keys())
        answer_cols = extract_cols_from_template(a_template)
        if re.fullmatch(r'count\{\s*\}', a_template.strip() if a_template else ''):
            answer_cols = []

        selected_columns = list(dict.fromkeys(condition_cols + answer_cols))

        # 构建正负样本 Schema
        schema_lines = build_m_schema(selected_columns, full_schema_meta)
        q_neg_schema = build_m_schema(
            generate_negatives(condition_cols, all_cols, NUM_HARD_NEG_QUESTION, NUM_EASY_NEG_QUESTION),
            full_schema_meta)
        a_neg_schema = build_m_schema(
            generate_negatives(answer_cols, all_cols, NUM_HARD_NEG_ANSWER, NUM_EASY_NEG_ANSWER), full_schema_meta)

        # 【核心逻辑】合并正负样本 Schema 并打乱顺序
        all_schema_lines = list(set(schema_lines + q_neg_schema + a_neg_schema))
        random.shuffle(all_schema_lines)
        schema_text = "\n".join(all_schema_lines)

        # 构建给大模型的 System Prompt
        system_prompt = (
            "你是一名SQL专家。请参考以下内容生成SQL。\n"
            "采用sqlite，不需要加上数据库名。\n\n"
            "【表结构信息】\n"
            f"{schema_text}"
            "请输出SQL，用```sql ... ```包裹,不需要解释和输出其他内容。"
        )

        # 构建 User Prompt
        user_prompt = f"【用户问题】\n{question}"

        # 组装为标准的 LoRA messages 格式
        lora_item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": sql_query}  # LLM 需要学习生成的输出
            ]
        }

        output_data.append(lora_item)

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 生成完成，共处理并保留 {len(output_data)} 条高质量数据，保存至 {OUTPUT_JSONL}")


if __name__ == '__main__':
    main()