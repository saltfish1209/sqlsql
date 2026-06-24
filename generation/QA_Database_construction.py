from collections import Counter

import pandas as pd
import json
import time
import os
import sys
import re
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from config import get_llm_api_key, get_llm_base_url, get_llm_model
from construct import get_multiple_filled_qa_pairs
from multi_result_utils import MULTI_RESULT_SEP, split_answer_template_top_level
from training.dataset_io import normalize_and_deduplicate_dataframe, normalize_cell_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))


# 选项: 'resume' (跳过已完成的), 'overwrite' (全部重跑), 'retry_errors' (仅重跑之前失败的-需结合日志逻辑，此处简化为跳过已存在)
MODE = 'resume'
OUTPUT_FILE = os.path.join(DATA_DIR, "train_dataset_with_sql_and_slots.csv")
CSV_COLUMNS = [
    "问题模版", "回答模版", "原始填充问题", "生成问题", "生成结果",
    "标准答案", "SQL语句", "SQL验证状态", "槽位信息JSON", "是否有效" # 确保"是否有效"在最后或固定位置
]
TARGET_SAMPLES = 5

COL_Q_TEMPLATE = '提问模版'
COL_A_TEMPLATE = '回答模版'

INPUT_TEMPLATE_FILE = os.path.join(DATA_DIR, "副本问题收集模板.CSV")
INPUT_DATA_FILE = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")

NUMERIC_COLS = [
    "采购申请数量",
    "概算单价",
    "概算总价",
    "中标单价",
    "中标总价",
    "订单单价(含税)",
    "订单总价(含税)",
    "合同数量",
    "合同单价(含税)",
    "合同总价(含税)",
    "采购订单数量",
    "订单单价(不含税)",
    "订单总价(不含税)",
    "已付预付款金额(含税)",
    "已付到货款金额(含税)",
]


LLM_MODEL = get_llm_model()
client = OpenAI(
    api_key=get_llm_api_key(),
    base_url=get_llm_base_url(),
    timeout=600,
    max_retries=3
)
def clean_llm_text(text):
    text = re.sub(r'^\d+\.\s*','',text)
    text = text.replace('**','').replace('__','')
    return text.strip()

def clean_value(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ('', 'nan', 'null', 'none', 'n/a'):
        return ""
    # 新增：如果字符串以 .0 结尾且前面是纯数字，去掉 .0
    # 解决 "2023.0" 在 Pandas 是字符串但在 SQL 生成时被切掉导致的不匹配
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


ANSWER_TEMPLATE_HINTS = {
    "field": "查询目标为“{target}”字段，最终答案为：{answer}",
    "multi_field": "查询目标为“{target}”这些字段的组合结果，最终答案为：{answer}",
    "count": "查询目标为满足问题条件的数据条数，最终答案为：{answer}",
    "sum": "查询目标为“{target}”字段的求和结果，最终答案为：{answer}",
    "avg": "查询目标为“{target}”字段的平均值，最终答案为：{answer}",
    "count1": "查询目标为“{target}”字段的去重数量，最终答案为：{answer}",
    "rank": "查询目标为按“{rank_target}”{direction}排序后返回“{return_target}”字段，范围为{top_n}，最终答案为：{answer}",
    "raw": "查询目标由回答模板“{template}”定义，最终答案为：{answer}",
}


def _format_slot_mapping(slot_mapping: dict) -> str:
    if not slot_mapping:
        return "无"
    return "；".join([f"{k}={v}" for k, v in slot_mapping.items()])


def _split_fields(field_text: str) -> list:
    return [x.strip() for x in re.split(r"[,，]", field_text) if x.strip()]


def build_answer_template_hint(answer_template: str, answer: str) -> str:
    answer_template = str(answer_template or "").strip()
    answer = str(answer or "").strip()

    sub_templates = split_answer_template_top_level(answer_template)
    if len(sub_templates) > 1:
        sub_answers = str(answer).split(MULTI_RESULT_SEP)
        hints = []
        for idx, sub_template in enumerate(sub_templates):
            sub_answer = sub_answers[idx].strip() if idx < len(sub_answers) else answer
            hints.append(build_answer_template_hint(sub_template, sub_answer))
        return "；".join(hints)

    if re.fullmatch(r"count\{\s*\}", answer_template, re.IGNORECASE):
        return ANSWER_TEMPLATE_HINTS["count"].format(answer=answer)

    rank_match = re.fullmatch(
        r"(listdown|listup)\{\s*([^}]+?)\s*\}\{\s*([^}]+?)\s*\}\*(\d+|\*)",
        answer_template,
        re.IGNORECASE,
    )
    if rank_match:
        direction = "降序" if rank_match.group(1).lower() == "listdown" else "升序"
        agg_col = rank_match.group(2).strip()
        return_cols = "、".join(_split_fields(rank_match.group(3)))
        top_n = rank_match.group(4)
        rank_target = "记录数量" if agg_col.lower() == "count" else f"{agg_col}求和结果"
        top_n_text = "全部结果" if top_n == "*" else f"前 {top_n} 个结果"
        return ANSWER_TEMPLATE_HINTS["rank"].format(
            rank_target=rank_target,
            direction=direction,
            return_target=return_cols,
            top_n=top_n_text,
            answer=answer,
        )

    agg_match = re.fullmatch(r"(sum|avg|count1)\{\s*([^}]+?)\s*\}", answer_template, re.IGNORECASE)
    if agg_match:
        agg_type = agg_match.group(1).lower()
        target = agg_match.group(2).strip()
        return ANSWER_TEMPLATE_HINTS[agg_type].format(target=target, answer=answer)

    fields = [x.strip() for x in re.findall(r"\{([^}]+)}", answer_template) if x.strip()]
    if fields:
        if len(fields) == 1:
            fields = _split_fields(fields[0]) or fields
        key = "field" if len(fields) == 1 else "multi_field"
        return ANSWER_TEMPLATE_HINTS[key].format(target="、".join(fields), answer=answer)

    return ANSWER_TEMPLATE_HINTS["raw"].format(template=answer_template, answer=answer)


def build_llm_case_source(index: int, pair: dict, answer_template: str) -> str:
    slot_mapping = pair.get("slot_mapping", {}) or {}
    return (
        f"Case {index}:\n"
        f"   [填充后的问题]: {pair['filled_question']}\n"
        f"   [标准答案]: {pair['answer']}\n"
        f"   [槽位信息]: {_format_slot_mapping(slot_mapping)}\n"
        f"   [查询目标提示]: {build_answer_template_hint(answer_template, pair['answer'])}"
    )


def generate_batch_similar_questions(qa_pairs: list, q_template:str,a_template:str,client_llms=client,) -> list:
    num = len(qa_pairs)
    examples_list = []
    for i, pair in enumerate(qa_pairs):
        examples_list.append(build_llm_case_source(i + 1, pair, a_template))

    examples = "\n\n".join(examples_list)

    system_prompt = f"""你是一个 Text-to-SQL 数据集增强助手。

任务：根据每个 Case 的“填充后的问题、标准答案、槽位信息、查询目标提示”，把问题改写成更自然的用户问法。

硬性要求：
1. 只能改变问法，不能改变查询目标；查询目标以[查询目标提示]为准。
2. [槽位信息]中的值必须在生成问题中原样保留，不能替换、缩写、扩写、模糊化或调整数字/符号/大小写。
3. 答案必须直接使用[标准答案]，不要重新计算或改写答案。
4. 只输出 {num} 行，不要序号、解释、表头；每行格式：生成的问题|||答案:对应的回答。
5. 问题不要添加时间条件，数据集中没有时间列。

[问题模版]: {q_template}
[答案模版]: {a_template}
现在处理以下 {num} 组数据：
"""

    user_prompt = examples

    try:
        completion = client_llms.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # 稍微调高一点，增加多样性，因为有模版限制住了逻辑
            max_tokens=1500,
            extra_body={"enable_thinking": False}
        )

        raw_text = completion.choices[0].message.content.strip()
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        cleaned_lines = []
        for line in lines:
            # 2. 增加代码级清洗（关键步骤！）

            # A. 去除 markdown 加粗
            line = line.replace('**', '').replace('__', '')

            # B. 过滤掉显而易见的表头行或废话
            if "生成的问题" in line or "对应的回答" in line:
                continue
            if "Here is" in line or "如下" in line:
                continue

            # C. 必须包含分隔符才算有效行
            if "|||答案:" not in line:
                continue

            # D. 去除开头的 "1. " 这种序号
            line = re.sub(r'^\d+\.\s*', '', line)

            cleaned_lines.append(line)

        # 兜底补全
        if len(cleaned_lines) < num:
            for i in range(len(cleaned_lines), num):
                fallback_str = f"{qa_pairs[i]['filled_question']}|||答案:{qa_pairs[i]['answer']}"
                cleaned_lines.append(fallback_str)

        return cleaned_lines[:num]

    except Exception as e:
        print(f"   ⚠️ LLM 生成失败: {e}")
        return None



def main():
    print(f"程序启动，模式: {MODE}")
    if not os.path.exists(INPUT_TEMPLATE_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_TEMPLATE_FILE}")
        return

    df_template = pd.read_csv(INPUT_TEMPLATE_FILE)
    if COL_Q_TEMPLATE not in df_template.columns or COL_A_TEMPLATE not in df_template.columns:
        print(f"❌ 错误：输入文件缺少 '{COL_Q_TEMPLATE}' 或 '{COL_A_TEMPLATE}' 列")
        return

    print(f"原始模板数量: {len(df_template)}")

    df_template = df_template.dropna(subset=[COL_Q_TEMPLATE, COL_A_TEMPLATE])
    df_template = df_template[
        (df_template[COL_Q_TEMPLATE].astype(str).str.strip() != '') &
        (df_template[COL_A_TEMPLATE].astype(str).str.strip() != '')
        ]
    print(f"过滤空模板后数量: {len(df_template)}")
    df_template, removed_templates = normalize_and_deduplicate_dataframe(
        df_template,
        subset=[COL_Q_TEMPLATE, COL_A_TEMPLATE],
    )
    print(f"模板去重后数量: {len(df_template)} (删除 {removed_templates} 条重复模板)")

    df_raw = pd.read_csv(INPUT_DATA_FILE, dtype=str)
    df_clean = normalize_and_deduplicate_dataframe(df_raw.map(clean_value))[0]

    print(f"🔄 正在转换数字列类型...")
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            # errors='coerce' 会把无法转数字的文本（如空串、'未知'）变成 NaN
            # 这一点非常重要，因为 SQL 的 SUM 会自动忽略 NULL，Pandas 的 sum 也会忽略 NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        else:
            print(f"   ⚠️ 警告：配置的数字列 '{col}' 在CSV中不存在，已跳过")

    total_templates = len(df_template)
    print(f"待处理模板总数: {total_templates}")

    # 初始化内存数据库用于 SQL 生成 (construct.py 需要)
    import sqlite3
    conn = sqlite3.connect(':memory:')
    sql_dtypes = {col: 'TEXT' for col in df_clean.columns}
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            sql_dtypes[col] = 'REAL'
            print(f"   ⚙️ 已将列 '{col}' 的数据库类型设为 REAL")

        # 3. 写入数据库
    df_clean.to_sql(
        'procurement_table',
        conn,
        index=False,
        dtype=sql_dtypes  # <--- 使用自定义的类型映射
    )

    existing_counts = Counter()
    file_exists = os.path.exists(OUTPUT_FILE)

    if MODE == 'resume' and file_exists:
        try:
            existing_df = normalize_and_deduplicate_dataframe(pd.read_csv(OUTPUT_FILE, dtype=str))[0]
            # 获取已经存在的“问题模版”列表
            if '问题模版' in existing_df.columns:
                existing_counts = Counter(existing_df['问题模版'].map(normalize_cell_text))
            print(f"📂 已读取现有进度，将补齐未满 {TARGET_SAMPLES} 条的任务。")

            # 如果是追加模式，先加载旧数据到内存（如果不嫌大）或者直接以 append 模式写入
            # 这里为了简单，我们采用“追加写入文件”的方式，all_generated 只存新数据
        except Exception as e:
            print(f"读取现有文件失败，将重新开始: {e}")

        # 如果是 overwrite 模式，并且文件存在，最好备份一下或清空，这里逻辑根据 all_generated 最后一次性写入决定
        # 为了防止跑了一半崩了数据全丢，建议改为“每处理一个模板追加写入一次”

        # 准备 CSV Writer 头
    if not file_exists or MODE == 'overwrite':
        # 初始化一个空文件或覆盖
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        empty_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        processed_templates = set() # 清空记录，确保不跳过任何任务

    stats = {
        'success': 0,
        'skipped_full': 0,
        'skipped_no_data': 0,
        'skipped_sql_fail': 0,
        'errors': 0,
        'saved_rows': 0
    }

    # --- 2. 主循环 ---
    for idx, template_row in df_template.iterrows():
        q_temp = normalize_cell_text(template_row.get(COL_Q_TEMPLATE, ''))
        a_template = normalize_cell_text(template_row.get(COL_A_TEMPLATE, ''))

        if not q_temp or pd.isna(q_temp):
            continue
        # 1. 获取当前已有的数量
        current_count = existing_counts.get(q_temp, 0)

        # 2. 判断是否已满

        if current_count >= TARGET_SAMPLES:
            print(f"⏩ 模板 {idx} 已满 ({current_count}/{TARGET_SAMPLES})，跳过")
            stats['skipped_full'] += 1
            continue

        # 3. 计算还需要跑多少条
        needed = TARGET_SAMPLES - current_count
        print(f"\n处理模板 {idx}: {q_temp}")
        print(f"   📊 进度: {current_count}/{TARGET_SAMPLES} (需补生成 {needed} 条)")




        batch_data = []  # 暂存当前模板的数据

        try:
            # 调用 construct.py 的生成函数
            qa_list = get_multiple_filled_qa_pairs(
                template_row=pd.DataFrame([template_row]),
                df_raw=df_clean,
                sql_conn=conn,
                num_samples=needed,  # <--- 使用需要的数量，而不是固定5
                max_retries_per_sample=15,
                q_col=COL_Q_TEMPLATE,  # <--- 传入 "提问模版"
                a_col=COL_A_TEMPLATE  # <--- 传入 "回答模版"
            )

            if not qa_list:
                print("   ❌ 未生成有效基础数据")
                stats['skipped_no_data'] += 1
                continue


            valid_qa_list = []
            failed_pairs = []
            for pair in qa_list:
                # construct.py 返回的字典里现在有了 'is_valid'
                if pair.get('validation') == 'MATCH':
                    valid_qa_list.append(pair)
                else:
                    failed_pairs.append(pair)
                    print(f"   ⚠️ 丢弃一条 SQL 校验失败的数据: {pair.get('validation')}")

            if failed_pairs and len(valid_qa_list) == 0:
                print(f"\n🔍 模板全部校验失败，打印 {len(failed_pairs)} 条失败详情用于诊断：")
                for i, fp in enumerate(failed_pairs[:3], 1):
                    print(f"\n--- 失败样本 {i} ---")
                    print(f"  问题: {fp.get('filled_question')}")
                    print(f"  Python值: {fp.get('py_value_repr', 'N/A')}")
                    print(f"  SQL值: {fp.get('sql_value_repr', 'N/A')}")


            if not valid_qa_list:
                print("   ❌ 当前模板生成的所有数据 SQL 校验均失败，跳过 LLM 生成。")
                stats['skipped_sql_fail'] += 1
                continue

            print(f"   ✅ 基础生成 {len(qa_list)} 条，SQL校验通过 {len(valid_qa_list)} 条 -> 准备 LLM 改写")

            # 调用 LLM 进行改写 (LLM 不需要看 SQL，只需要看问题和答案)
            rewritten_data = generate_batch_similar_questions(valid_qa_list, q_template=q_temp, a_template=a_template)
            if rewritten_data is None:
                print("   🚫 LLM 调用异常（如连接超时），放弃本次结果。")
                print("      -> 本批次未写入文件，下次 Resume 时将自动重试。")
                stats['errors'] += 1
                continue

            for i, pair in enumerate(valid_qa_list):
                llm_output = rewritten_data[i]

                # --- 解析 LLM 输出 ---
                if "|||答案:" in llm_output:
                    parts = llm_output.split("|||答案:", 1)
                    gen_q = clean_llm_text(parts[0])
                    gen_a = parts[1].strip()
                else:
                    gen_q = clean_llm_text(pair['filled_question'])
                    gen_a = pair['answer']

                # --- 构建数据行 ---
                row_data = {
                    "问题模版": q_temp,
                    "回答模版": a_template,
                    "原始填充问题": pair['filled_question'],
                    "生成问题": gen_q,
                    "生成结果": gen_a,
                    "标准答案": pair['answer'],

                    # SQL 相关字段
                    "SQL语句": pair.get('sql', ''),
                    "SQL验证状态": pair.get('validation', ''),

                    # 从 pair 中获取 construct.py 返回的校验布尔值
                    # 如果获取不到，默认为 False
                    "是否有效": pair.get('is_valid', False),
                    "槽位信息JSON": json.dumps(pair.get('slot_mapping', {}), ensure_ascii=False)
                }

                # --- 动态添加 列名 和 对应值 ---
                # pair['slot_mapping'] 是我们刚才在 construct.py 里新增的字典
                slot_info = pair.get('slot_mapping', {})

                # 方案A：存为 JSON 字符串（推荐，方便后续解析，不会造成列数爆炸）
                row_data["槽位信息JSON"] = json.dumps(slot_info, ensure_ascii=False)

                # # 方案B：打散成列 (如果你需要直观地看 CSV)
                # # 格式：Key1, Value1, Key2, Value2...
                # for k_idx, (col_name, col_val) in enumerate(slot_info.items()):
                #     row_data[f"槽位{k_idx + 1}_列名"] = col_name
                #     row_data[f"槽位{k_idx + 1}_值"] = col_val

                batch_data.append(row_data)

            if batch_data:
                df_batch = pd.DataFrame(batch_data)
                # 统一列顺序（可选，防止追加时列错乱）
                # df_batch = df_batch.reindex(columns=...)
                for col in CSV_COLUMNS:
                    if col not in df_batch.columns:
                        df_batch[col] = ""  # 或者默认值

                df_batch = df_batch[CSV_COLUMNS]  # <--- 关键！强制按 CSV_COLUMNS 排序

                df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
                stats['saved_rows'] += len(batch_data)
                stats['success'] += 1
                print(f"   ✅ 已保存 {len(batch_data)} 条数据")


        except Exception as e:
            print(f"   ❌ 错误: {e}")
            stats['errors'] += 1
            import traceback
            traceback.print_exc()

    conn.close()

    print("\n" + "=" * 50)
    print("🎉 所有任务处理完成！最终统计：")
    print(f"   - 成功处理模板: {stats['success']}")
    print(f"   - 跳过(已存在): {stats['skipped_full']}")
    print(f"   - 跳过(无数据): {stats['skipped_no_data']}")
    print(f"   - 跳过(SQL失败): {stats['skipped_sql_fail']}")
    print(f"   - 异常/错误: {stats['errors']}")
    print(f"   - 总计保存行数: {stats['saved_rows']}")
    print("=" * 50)


if __name__ == '__main__':
    main()
