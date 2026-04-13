from collections import Counter

import pandas as pd
import json
import time
import os
import re  # 引入正则用于处理列排序
from openai import OpenAI
from construct import get_multiple_filled_qa_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))


# 选项: 'resume' (跳过已完成的), 'overwrite' (全部重跑), 'retry_errors' (仅重跑之前失败的-需结合日志逻辑，此处简化为跳过已存在)
MODE = 'overwrite'
OUTPUT_FILE = os.path.join(DATA_DIR, "train_dataset_with_sql_and_slots.csv")
CSV_COLUMNS = [
    "问题模版", "回答模版", "原始填充问题", "生成问题", "生成结果",
    "标准答案", "SQL语句", "SQL验证状态", "槽位信息JSON", "是否有效" # 确保"是否有效"在最后或固定位置
]
TARGET_SAMPLES = 5

INPUT_TEMPLATE_FILE = './副本问题收集模板.CSV'
INPUT_DATA_FILE = './一次二次物料长描述2.csv'

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


client = OpenAI(
    api_key="sk-cbbd58b6e9004c30a02c551bdb2f0e9e",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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


def generate_batch_similar_questions(qa_pairs: list, q_template:str,a_template:str,client_llms=client,) -> list:
    num = len(qa_pairs)
    examples_list = []
    for i, pair in enumerate(qa_pairs):
        # 获取该条数据对应的真实槽位值（即数据库里的精确值）
        slots = pair.get('slot_mapping', {})

        # 格式化槽位信息，方便 LLM 阅读
        # 例如：Entities: { "物资描述": "10kV真空断路器", "供应商": "江苏xx公司" }
        slots_str = ", ".join([f"{k}:'{v}'" for k, v in slots.items()])

        example_str = (
            f"Case {i + 1}:\n"
            f"   [关键实体数据]: {{{slots_str}}}\n"
            f"   [原始机械问题]: {pair['filled_question']}\n"
            f"   [对应标准答案]: {pair['answer']}"
        )
        examples_list.append(example_str)

    examples = "\n\n".join(examples_list)

    system_prompt = f"""你是一个Text-to-SQL数据集增强专家。
    
你的任务是将给定的“原始问题”改写成**真实用户**在查询数据库时可能使用的自然语言问题。用户通常会模糊表达、使用简称或口语化提问。

【输入信息说明】
- **[数据库实体]**：这是问题中涉及的数据库精确值（Ground Truth）。
- **[原始问题]**：这是由模版生成的机械化问题，语法可能生硬。
- **[标准答案]**：这是查询的最终结果。

【改写核心要求】
1. **意图一致**：新问题查询不脱离答案模版。对于查询具体内容，答案模版表示答案所在列；对于计算问题，答案模版表示计算方式，一切模糊歧义以[答案模版]为标准
2. **实体模糊化（高优先级）**：
   - 必须参考 [数据库实体] 中的值。
   - **模拟用户输入**：用户很少输入全称。请将实体值改为**简称、别名、去后缀、去除部分字词**的形式（例如："江苏省电力有限公司" -> "江苏电力"；"10kV真空断路器" -> "断路器" 或 "10kV开关"）。
   - **保留关键特征**：模糊化不能导致歧义（例如不能把 "A型" 改成 "B型"）。
   -**语义模糊**对所提到的关键词，可以进行歧义改写，但不能脱离[答案模版]、因为答案模版表示答案所在列和答案计算方式
3. **句式多样性**：
   - 彻底打破原始问题的语法结构。
   -自由编排内容，答案和查询逻辑不要脱离[标准答案]
   - 使用口语（"这个物料的编码是多少？"）。
4. **格式严格**：只输出 {num} 行，不要包含任何序号或多余解释。
   每行格式：生成的问题|||答案:对应的回答

-**[问题模版]**:{q_template}
-**[答案模版]**:{a_template}
现在开始处理以下 {num} 组数据：
                    """

    user_prompt = examples

    try:
        completion = client_llms.chat.completions.create(
            model="qwen3-max-2026-01-23",  # 或你使用的其他模型
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

    df_raw = pd.read_csv(INPUT_DATA_FILE, dtype=str)
    df_clean = df_raw.map(clean_value)
    df_clean.columns = [c.strip() for c in df_clean.columns]

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
            existing_df = pd.read_csv(OUTPUT_FILE)
            # 获取已经存在的“问题模版”列表
            if '问题模版' in existing_df.columns:
                existing_counts = Counter(existing_df['问题模版'])
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
        q_temp = template_row.get(COL_Q_TEMPLATE, '')
        a_template = template_row.get(COL_A_TEMPLATE, '')

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
            rewritten_data = generate_batch_similar_questions(qa_list,q_template= q_temp,a_template= a_template)
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

