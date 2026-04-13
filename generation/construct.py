import csv
import os
import pandas as pd
import re
from collections import Counter
from data_computation import EnhanceDataQueryBuilder
import random
import numpy as np
import math
from sql_generator import SQLQueryBuilder

# 设置固定的随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def init_db(df_raw):
    conn = sqlite3.connect(':memory:')
    df_raw.to_sql('procurement_table',conn,index=False)
    return conn



def extract(text: pd.DataFrame,q_col = '提问模版', a_col = '回答模版'):
    if q_col not in text.columns:
        raise KeyError(f"缺少问题列 '{q_col}'，可用列: {list(text.columns)}")
    if a_col not in text.columns:
        raise KeyError(f"缺少回答列 '{a_col}'，可用列: {list(text.columns)}")

    q_str = text[q_col].values[0].strip()
    a_str = text[a_col].values[0].strip()
    # 提取提问中的字段
    question_name = [s.strip() for s in re.findall(r'\{([^}]+)}', q_str)]
    field_counts = Counter(question_name)

    # 解析回答模板：
    # 1.count
    if re.fullmatch(r'count\{\s*\}', a_str, re.IGNORECASE):
        return question_name, [], field_counts, 'count'

    # 2.排名聚合：listdown{agg_col}{return_cols}*N  或  listup{...}*N
    rank_match = re.fullmatch(
        r'(listdown|listup)\{\s*([^}]+?)\s*\}\{\s*([^}]+?)\s*\}\*(\d+|\*)',
        a_str,
        re.IGNORECASE
    )
    if rank_match:
        direction  = rank_match.group(1).lower()
        original_agg_col = rank_match.group(2).strip()
        if original_agg_col.lower() == 'count':
            agg_type = 'count'
        else:
            agg_type = 'sum'  # 默认对数值列求和
        return_cols_str = rank_match.group(3).strip()
        top_n_str = rank_match.group(4)

        return_cols = [c.strip() for c in re.split(r'[,，]',return_cols_str) if c.strip()]
        top_n = None if top_n_str == '*' else int(top_n_str)

        answer_param = {
            'type': direction,
            'agg_col': original_agg_col,
            'agg_type': agg_type,
            'return_cols': return_cols,
            'top_n': top_n
        }
        return question_name, answer_param, field_counts, direction

    # 3. 基础聚合：sum{A}, avg{A}, count1{A}
    agg_match = re.match(r'^(sum|avg|count1)\{\s*([^}]+?)\s*\}$', a_str, re.IGNORECASE)
    if agg_match:
        agg_op = agg_match.group(1).lower()
        answer_field = agg_match.group(2).strip()

        return question_name, [answer_field], field_counts, agg_op

    # 4. 默认：非聚合字段（支持 {A}{B} 或 {A, B}）
    non_agg_fields = [s.strip() for s in re.findall(r'\{([^}]+)}', a_str)]
    if non_agg_fields:
        final_fields = []
        for item in non_agg_fields:
            if len(non_agg_fields) == 1 and "，" in item:
                final_fields.extend([x.strip() for x in item.split('，')])
            else:
                final_fields.append(item)

        return question_name, final_fields, field_counts, None

    # 5. 无法解析
    raise ValueError(f"无法解析回答模板: '{a_str}'")




def extract_and_compute(row: pd.Series, template_row: pd.DataFrame, df_raw: pd.DataFrame,sql_conn, max_results: int = 50, q_col='提问模版', a_col='回答模版'):

    sql_builder = SQLQueryBuilder(table_name='procurement_table')

    q_fields_raw, a_fields, field_counts, aggregation = extract(template_row ,q_col=q_col, a_col=a_col)


    matched_df = pd.DataFrame()
    base_conditions = {}
    multi_conditions = {}
    if not q_fields_raw:
        matched_df = df_raw.copy()


    else:
        repeated_fields = {f: cnt for f, cnt in field_counts.items() if cnt > 1}
        unique_fields = [f for f, cnt in field_counts.items() if cnt == 1]

        # Step 1: 构建 base_conditions
        for field in unique_fields:
            if field not in row.index or pd.isna(row[field]) or str(row[field]).strip() == '':
                return {"error": f"缺少必要字段: {field}"}
            base_conditions[field] = str(row[field]).strip()

        # Step 2: 初筛
        mask = pd.Series([True] * len(df_raw), index=df_raw.index)
        for field, value in base_conditions.items():
            if field not in df_raw.columns:
                return {"error": f"字段 {field} 不在原始数据中"}
            col_vals = df_raw[field].astype(str).fillna('NaN').str.strip()
            mask &= (col_vals == value)

        candidate_df = df_raw[mask]
        if candidate_df.empty:
            return {"error": "无匹配数据"}

        # Step 3: 处理重复字段（如有）
        for field, n in repeated_fields.items():
            if field not in candidate_df.columns:
                continue
            unique_vals = candidate_df[field].dropna().astype(str).str.strip().unique()
            if len(unique_vals) == 0:
                continue
            selected_vals = unique_vals[:min(n, len(unique_vals))]
            multi_conditions[field] = selected_vals.tolist()

        # Step 4: 构建最终 mask
        final_mask = pd.Series([True] * len(df_raw), index=df_raw.index)
        for field, value in base_conditions.items():
            col_vals = df_raw[field].astype(str).fillna('NaN').str.strip()
            final_mask &= (col_vals == value)
        for field, values in multi_conditions.items():
            col_vals = df_raw[field].astype(str).fillna('NaN').str.strip()
            final_mask &= col_vals.isin([str(v).strip() for v in values])

        matched_df = df_raw[final_mask].copy()
        if matched_df.empty:
            return {"error": "最终筛选无结果"}

    # Step 5: 验证答案字段存在且非空（仅当无聚合时需要逐行验证）
    if not aggregation and a_fields:
        valid_rows = []
        for _, r in matched_df.iterrows():
            if all(f in r.index and pd.notna(r[f]) and str(r[f]).strip() != '' for f in a_fields):
                valid_rows.append(r)
        matched_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()


    py_final_data = {}
    agg_config = {}

    # -----------------------------
    # Step 6: 聚合计算
    # -----------------------------
    if aggregation:
        # list排序
        if aggregation in ('listdown', 'listup'):
            # a_fields 实际是 params dict（由 extract 返回）
            params = a_fields  # 注意：此时 a_fields 是 dict，不是 list！

            builder = EnhanceDataQueryBuilder(matched_df)
            try:
                ranked_results = getattr(builder, aggregation)(matched_df, params)
            except Exception as e:
                return {"error": f"排名计算失败 ({aggregation}): {e}"}

            py_final_data = {
                "base_conditions": base_conditions,
                "multi_conditions": multi_conditions,
                "results": ranked_results,  # List[Dict]
                "aggregation": None,  # 走非聚合拼接路径
                "is_one_to_one": len(ranked_results) == 1
            }

            # 准备sql参数
            agg_config = {
                'type': aggregation,
                'agg_col': params.get('agg_col'),
                'agg_type': params.get('agg_type','sum'),
                'return_cols': params.get('return_cols'),
                'top_n': params.get('top_n')
            }

        # sum,avg,count,count1
        else:
            if aggregation == 'count':
                total = len(matched_df)
                agg_config = {'type': 'count', 'col': None}
            else:
                if not a_fields:
                    return {"error": f"聚合类型 '{aggregation}' 需要指定字段，但回答模板中未提供"}
                agg_field = a_fields[0]
                if isinstance(agg_field, list):
                    agg_field = str(agg_field[0])

                builder = EnhanceDataQueryBuilder(matched_df)
                try:
                    if aggregation == 'sum':
                        total = builder.sum(matched_df, agg_field)
                    elif aggregation == 'avg':
                        total = builder.average(matched_df, agg_field)
                    elif aggregation == 'count1':
                        total = builder.count1(matched_df, agg_field)
                    else:
                        return {"error": f"不支持的聚合类型: {aggregation}"}
                except Exception as e:
                    return {"error": f"聚合计算失败: {e}"}

                # 配置sql参数
                agg_config = {'type': aggregation,'col':agg_field}

            py_final_data =  {
                "base_conditions": base_conditions,
                "multi_conditions": multi_conditions,
                "results": [{"value": total}],
                "aggregation": aggregation,
                "is_one_to_one": True
            }


    # -----------------------------
    # Step 7: 非聚合情况
    # -----------------------------
    else:
        if not matched_df.empty and a_fields:
            simplified = matched_df[a_fields].drop_duplicates()
            results = [{f: str(r[f]).strip() for f in a_fields} for _, r in simplified.iterrows()]
        else:
            results = []

        py_final_data = {
            "base_conditions": base_conditions,
            "multi_conditions": multi_conditions,
            "results": results,
            "aggregation": None,
            "is_one_to_one": len(results) == 1
        }
        # 配置sql参数
        agg_config = {'type': 'select', 'return_cols': a_fields}



    # -----------------------------
    # Step 8: 生成 SQL 并严格验证
    # -----------------------------
    generated_sql = sql_builder.generate(base_conditions, multi_conditions, agg_config)

    validation_status = "UNKNOWN"
    py_set = set()  # <--- 新增：用于存放 Pandas 算出的所有不重复结果
    sql_set = set() # 新增一个布尔标志
    py_val_repr = ""
    sql_val_repr = ""
    is_valid_bool = False

    try:
        cursor = sql_conn.cursor()
        cursor.execute(generated_sql)
        sql_rows = cursor.fetchall()

        # === A. 提取 Pandas 结果集 ===
        if aggregation and aggregation not in ['listdown', 'listup']:
            # 单值聚合 (count, sum)，结果只有一个数值
            val = py_final_data['results'][0]['value']
            py_set.add(str(val))
            py_val_repr = str(val)
        else:
            # 列表结果 (select, listdown)
            # 遍历所有结果行，如果是多列组合，拼接成字符串 "值1|值2"
            for row in py_final_data['results']:
                vals = [str(v).strip() for v in row.values()]
                # 再次过滤空值（双重保险）
                if any(v == '' for v in vals): continue
                py_set.add("|".join(vals))
            py_val_repr = f"Set(len={len(py_set)}) {list(py_set)[:3]}..."

        # --- 2. 获取 SQL 的核心值 ---
        if aggregation and aggregation not in ['listdown', 'listup']:
            # 单值聚合
            if sql_rows and sql_rows[0][0] is not None:
                sql_set.add(str(sql_rows[0][0]))
                sql_val_repr = str(sql_rows[0][0])
            else:
                sql_val_repr = "None"
        else:
            # 列表结果
            for row in sql_rows:
                # row 是 tuple ('val1', 'val2')
                # 过滤掉 SQL 返回的空值行 (重要！配合 sql_generator 的修改)
                if not row or all((v is None or str(v).strip() == '') for v in row):
                    continue

                clean_row = [str(v).strip() for v in row]
                sql_set.add("|".join(clean_row))
            sql_val_repr = f"Set(len={len(sql_set)}) {list(sql_set)[:3]}..."




        # 3.1 集合比对
        # 1. 数值比对 (针对 sum/avg/count)
        if aggregation and aggregation not in ['listdown', 'listup']:
            # 沿用之前的数值近似比对逻辑
            p_v = list(py_set)[0] if py_set else 0
            s_v = list(sql_set)[0] if sql_set else 0
            is_match = False
            try:
                if math.isclose(float(p_v), float(s_v), rel_tol=1e-5): is_match = True
            except:
                if str(p_v) == str(s_v): is_match = True

        # 2. 集合内容比对 (针对 select/listdown)
        else:
            # 允许 Pandas 算出的比 SQL 少 (因为 Pandas 有 head 限制)，但不能多出奇怪的东西
            # 或者严格一点：必须完全相等
            # 这里采用：Pandas 的所有非空结果必须包含在 SQL 结果中
            if len(py_set) == 0:
                is_match = False  # Pandas 没算出东西，肯定不对（前面已经熔断了，这里是兜底）
            else:
                missing_in_sql = py_set - sql_set
                extra_in_sql = sql_set - py_set

                # 只有当两边完全一致时才算 MATCH (严格模式，推荐)
                if not missing_in_sql and not extra_in_sql:
                    is_match = True
                else:
                    is_match = False
                    validation_status = f"Set Mismatch: Py-SQL={missing_in_sql}, SQL-Py={extra_in_sql}"

        if is_match:
            validation_status = "MATCH"
            is_valid_bool = True
        else:
            if validation_status == "UNKNOWN":
                validation_status = f"MISMATCH (Py:{py_val_repr} | SQL:{sql_val_repr})"
            is_valid_bool = False
            # 打印调试信息
            has_py_result = len(py_set) > 0
            has_sql_result = len(sql_set) > 0

            if has_py_result or has_sql_result:
                print("\n" + "!" * 20 + " 🔴 DEBUG: 发现严重不一致 " + "!" * 20)
                print(f"❌ 错误类型: {validation_status}")
                print(f"📋 原始问题: {template_row[q_col].values[0]}")

                print(f"\n🔍 1. 筛选条件:")
                print(f"   Base: {base_conditions}")
                print(f"   Multi: {multi_conditions}")

                print(f"\n💻 2. SQL 语句:")
                print(f"   {generated_sql}")

                print(f"\n📉 3. 结果集合对比:")
                print(f"   Pandas (Len={len(py_set)}): {list(py_set)[:5]}")
                print(f"   SQL    (Len={len(sql_set)}): {list(sql_set)[:5]}")
                print("!" * 60 + "\n")

    except Exception as e:
        validation_status = f"SQL_EXEC_ERROR: {str(e)}"
        is_valid_bool = False

    py_final_data['generated_sql'] = generated_sql
    py_final_data['sql_validation'] = validation_status
    py_final_data['py_value_repr'] = py_val_repr
    py_final_data['sql_value_repr'] = sql_val_repr
    py_final_data['is_valid'] = is_valid_bool

    return py_final_data


def get_multiple_filled_qa_pairs(template_row: pd.DataFrame, df_raw: pd.DataFrame, sql_conn, num_samples: int = 10,
                                 max_retries_per_sample: int = 20, q_col = '提问模版', a_col = '回答模版'):
    """
    为一个模板生成 num_samples 个互不相同的问答对。
    每个问答对来自不同的原始数据行。
    返回列表：[{"filled_question": ..., "answer": ..., "keywords": [...]}, ...]
    """
    global last_error
    q_fields, _, field_counts, _ = extract(template_row,q_col=q_col,a_col=a_col)
    unique_fields = [f for f, cnt in field_counts.items() if cnt == 1]

    # 预过滤：确保字段存在且非空
    mask = pd.Series([True] * len(df_raw))
    for f in unique_fields:
        if f not in df_raw.columns:
            raise ValueError(f"缺失字段: {f}")
        s = df_raw[f].astype(str).str.strip()
        mask &= df_raw[f].notna() & (s != '') & (s != 'nan')
    candidates = df_raw[mask]

    if len(candidates) < num_samples:
        print(f"⚠️ 候选数据不足（仅有 {len(candidates)} 行），将使用全部")
        num_samples = len(candidates)

    if candidates.empty:
        raise RuntimeError("无有效候选数据")

    used_indices = set()
    results = []
    attempts = 0
    max_total_attempts = num_samples * max_retries_per_sample


    last_real_error = None
    while len(results) < num_samples and attempts < max_total_attempts:
        sampled_row = candidates.sample(n=1).iloc[0]
        row_idx = sampled_row.name

        # 防止重复使用同一行
        if row_idx in used_indices:
            attempts += 1
            continue

        used_indices.add(row_idx)

        # 构建 filled_question
        q_str = template_row[q_col].values[0].strip()
        filled_question = q_str
        slot_mapping = {}
        actual_values = []
        for field in q_fields:
            if f"{{{field}}}" in filled_question:
                raw_val = sampled_row[field]
                if pd.isna(raw_val):
                    val = ""
                else:
                    val = str(raw_val).strip()
                    val = remove_trailing_dot_zero(val)
                filled_question = filled_question.replace(f"{{{field}}}", val)
                actual_values.append(val)
                slot_mapping[field] = val

        # 计算答案
        result = extract_and_compute(row=sampled_row, template_row=template_row, df_raw=df_raw,sql_conn=sql_conn, q_col=q_col, a_col=a_col)

        if 'error' not in result and result.get('results'):
            if result['aggregation']:
                answer = str(result['results'][0]['value'])
            else:
                all_rows = []
                for row_dict in result['results']:
                    # 将这一行的所有列的值拼接（例如：'物资A 100个'）
                    row_str = "|".join([str(v) for v in row_dict.values()])
                    all_rows.append(row_str)

                # 将多行数据用中文逗号或分号连接
                answer = "，".join(all_rows)

            results.append({
                "filled_question": filled_question,
                "answer": answer,
                "keywords": actual_values,
                'slot_mapping': slot_mapping,
                "sql": result.get('generated_sql'),
                "validation": result.get('sql_validation'),
                "py_value_repr": result.get('py_value_repr', 'N/A'),
                "sql_value_repr": result.get('sql_value_repr', 'N/A'),

                # 【关键修复 2】：必须显式传递 is_valid，否则主程序读不到，默认为 False
                "is_valid": result.get('is_valid', False)
            })
        else:
            last_real_error = result.get('error','未知错误')
            attempts += 1
            continue

        attempts += 1

    if not results:
        raise RuntimeError(f'无法生成有效问答对（尝试{attempts}次），最后错误{last_real_error}')

    return results[:num_samples]

def remove_trailing_dot_zero(text):
    """
    移除文本中作为独立整数出现的 '.0'，例如：
    - '编号123.0' → '编号123'
    - '456.0批次' → '456批次'
    - 但保留 '567.08'、'0.057V'、'版本2.0'
    """
    if not isinstance(text, str):
        return text
    # \b 表示单词边界，\d+ 匹配整数，\.0 后不能跟数字
    return re.sub(r'\b(\d+)\.0(?!\d)', r'\1', text)



if __name__ == '__main__':
    TARGET_INDEX = 321

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
    INPUT_DATA_FILE = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")
    INPUT_TEMPLATE_FILE = os.path.join(DATA_DIR, "副本问题收集模板.CSV")
    NUM_SAMPLES = 5  # 尝试生成的样本数量
    # ==============================================

    import pandas as pd
    import sqlite3
    import os

    print(f"🚀 启动单行深度调试模式 [Target Index: {TARGET_INDEX}]...")

    # 1. 读取文件
    if not os.path.exists(INPUT_DATA_FILE) or not os.path.exists(INPUT_TEMPLATE_FILE):
        print("❌ 错误：找不到输入文件，请检查路径。")
        exit()

    # ⚠️ 关键：dtype=str 确保 ID 不会变成科学计数法
    df_raw = pd.read_csv(INPUT_DATA_FILE, dtype=str)
    df_templates = pd.read_csv(INPUT_TEMPLATE_FILE)


    # 2. 数据清洗 (模拟 qa_construct 的清洗逻辑)
    def debug_clean_value(x):
        if pd.isna(x): return ""
        s = str(x).strip()
        if s.lower() in ('', 'nan', 'null'): return ""
        # 去除千分位逗号
        if "," in s:
            temp = s.replace(",", "")
            if temp.replace(".", "", 1).isdigit(): s = temp
        # 去除 .0
        if s.endswith(".0") and s[:-2].isdigit(): s = s[:-2]
        return s


    print("🧹 执行数据清洗...")
    df_clean = df_raw.map(debug_clean_value)
    # 去除列名空格
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # 3. 初始化数据库 (模拟 qa_construct 的类型设置)
    print("💾 初始化内存数据库...")
    conn = sqlite3.connect(':memory:')

    # 定义数值列 (必须与 qa_construct 保持一致，确保 SQL > < 比较正确)
    numeric_cols_for_db = [
        "采购申请数量", "概算单价", "概算总价", "中标单价", "中标总价",
        "订单单价(含税)", "订单总价(含税)", "合同数量", "合同单价(含税)",
        "采购订单数量", "订单单价(不含税)", "订单总价(不含税)",
        "已付预付款金额(含税)", "已付到货款金额(含税)"
    ]

    sql_dtypes = {col: 'TEXT' for col in df_clean.columns}
    for col in numeric_cols_for_db:
        if col in df_clean.columns:
            sql_dtypes[col] = 'REAL'  # 强制设为浮点数

    df_clean.to_sql('procurement_table', conn, index=False, dtype=sql_dtypes)

    # 4. 锁定目标模板
    if TARGET_INDEX < 0 or TARGET_INDEX >= len(df_templates):
        print(f"❌ 索引错误：当前模板文件共有 {len(df_templates)} 行，索引范围 0 ~ {len(df_templates) - 1}")
        exit()

    target_template_row = df_templates.iloc[[TARGET_INDEX]]
    q_str = target_template_row['提问模版'].values[0]
    a_str = target_template_row['回答模版'].values[0]

    print("\n" + "=" * 60)
    print(f"🎯 正在调试模板 [Index {TARGET_INDEX}]:")
    print(f"❓ 提问: {q_str}")
    print(f"💡 回答: {a_str}")
    print("=" * 60 + "\n")

    # 5. 执行生成与验证
    try:
        # 强制开启 extract_and_compute 内部的打印
        # (这取决于你的 extract_and_compute 是否依赖环境变量，如果没有依赖，它会自动打印错误)

        results = get_multiple_filled_qa_pairs(
            template_row=target_template_row,
            df_raw=df_clean,
            sql_conn=conn,
            num_samples=NUM_SAMPLES,
            max_retries_per_sample=10,  # 给足够重试次数
            q_col='提问模版',
            a_col='回答模版'
        )

        print(f"\n✅ 生成成功! 共获得 {len(results)} 条有效数据:\n")

        for i, item in enumerate(results, 1):
            status_icon = "🟢" if item['is_valid'] else "🔴"
            print(f"--- [样本 {i}] {status_icon} ---")
            print(f"Q:   {item['filled_question']}")
            print(f"A:   {item['answer']}")
            print(f"SQL: {item['sql']}")
            print(f"状态: {item['validation']}")
            print(f"Py值: {item.get('py_value_repr')}")
            print(f"SQL值: {item.get('sql_value_repr')}")

            # 如果之前在 extract_and_compute 里写了 debug print，这里不需要重复打印详细比对
            print("-" * 40)

    except RuntimeError as e:
        print(f"\n❌ 生成失败 (RuntimeError): {e}")
        print("提示：可能是没有匹配的数据，或者 SQL 语法错误。")
    except Exception as e:
        print(f"\n❌ 未知错误:")
        import traceback

        traceback.print_exc()
    finally:
        conn.close()
        print("\n🏁 调试结束")



