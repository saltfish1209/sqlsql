"""
纯模板填充数据生成脚本（不使用 LLM 改写）
────────────────────────────────────────
目标：
1) 复用 construct.py 的填充 + SQL 校验能力，批量生成问答；
2) 不调用任何 LLM 改写，直接使用 filled_question / answer；
3) 保留 resume / overwrite 工作流，支持每模板目标样本数补齐。
"""
from __future__ import annotations

from collections import Counter
import json
import os
import sqlite3

import pandas as pd

from construct import get_multiple_filled_qa_pairs


# 运行模式：resume / overwrite
MODE = "overwrite"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

INPUT_TEMPLATE_FILE = os.path.join(DATA_DIR, "副本问题收集模板.CSV")
INPUT_DATA_FILE = os.path.join(DATA_DIR, "一次二次物料长描述2.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "train_dataset_template_only.csv")

COL_Q_TEMPLATE = "提问模版"
COL_A_TEMPLATE = "回答模版"

CSV_COLUMNS = [
    "问题模版",
    "回答模版",
    "原始填充问题",
    "生成问题",
    "生成结果",
    "标准答案",
    "SQL语句",
    "SQL验证状态",
    "槽位信息JSON",
    "是否有效",
]

# 每个模板目标样本数
TARGET_SAMPLES = 1
# 每个模板随机填充重试上限（无结果时重新筛选）
MAX_REFILL_ATTEMPTS = 10

# 数值列（写入 SQLite 时设为 REAL，避免聚合误差）
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


def clean_value(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "null", "none", "n/a"):
        return ""
    # 去掉纯整数尾部 .0
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def init_output_file_if_needed(file_exists: bool):
    if (not file_exists) or MODE == "overwrite":
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(
            OUTPUT_FILE, index=False, encoding="utf-8-sig"
        )


def main():
    print(f"程序启动（纯填充模式，无 LLM 改写），模式: {MODE}")

    if not os.path.exists(INPUT_TEMPLATE_FILE):
        print(f"❌ 错误：找不到模板文件 {INPUT_TEMPLATE_FILE}")
        return
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"❌ 错误：找不到数据文件 {INPUT_DATA_FILE}")
        return

    df_template = pd.read_csv(INPUT_TEMPLATE_FILE)
    if COL_Q_TEMPLATE not in df_template.columns or COL_A_TEMPLATE not in df_template.columns:
        print(f"❌ 错误：模板文件缺少列 '{COL_Q_TEMPLATE}' 或 '{COL_A_TEMPLATE}'")
        return

    print(f"原始模板数量: {len(df_template)}")
    df_template = df_template.dropna(subset=[COL_Q_TEMPLATE, COL_A_TEMPLATE])
    df_template = df_template[
        (df_template[COL_Q_TEMPLATE].astype(str).str.strip() != "")
        & (df_template[COL_A_TEMPLATE].astype(str).str.strip() != "")
    ]
    print(f"过滤空模板后数量: {len(df_template)}")

    # 原始数据清洗
    df_raw = pd.read_csv(INPUT_DATA_FILE, dtype=str)
    df_clean = df_raw.map(clean_value)
    df_clean.columns = [str(c).strip() for c in df_clean.columns]

    print("🔄 转换数值列类型...")
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 初始化内存数据库
    conn = sqlite3.connect(":memory:")
    sql_dtypes = {col: "TEXT" for col in df_clean.columns}
    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            sql_dtypes[col] = "REAL"
    df_clean.to_sql("procurement_table", conn, index=False, dtype=sql_dtypes)

    # 读取已有进度（resume）
    existing_counts = Counter()
    file_exists = os.path.exists(OUTPUT_FILE)
    if MODE == "resume" and file_exists:
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if "问题模版" in existing_df.columns:
                existing_counts = Counter(existing_df["问题模版"])
            print(f"📂 已读取现有进度，将补齐未满 {TARGET_SAMPLES} 条的模板。")
        except Exception as e:
            print(f"⚠️ 读取现有输出失败，将从头开始：{e}")

    init_output_file_if_needed(file_exists)

    stats = {
        "success": 0,
        "skipped_full": 0,
        "skipped_no_data": 0,
        "skipped_sql_fail": 0,
        "errors": 0,
        "saved_rows": 0,
    }

    for idx, template_row in df_template.iterrows():
        q_temp = template_row.get(COL_Q_TEMPLATE, "")
        a_temp = template_row.get(COL_A_TEMPLATE, "")
        if not q_temp or pd.isna(q_temp):
            continue

        current_count = existing_counts.get(q_temp, 0)
        if current_count >= TARGET_SAMPLES:
            print(f"⏩ 模板 {idx} 已满 ({current_count}/{TARGET_SAMPLES})，跳过")
            stats["skipped_full"] += 1
            continue

        needed = TARGET_SAMPLES - current_count
        print(f"\n处理模板 {idx}: {q_temp}")
        print(f"   📊 进度: {current_count}/{TARGET_SAMPLES} (需补 {needed} 条)")

        try:
            qa_list = []
            valid_qa_list = []
            failed_pairs = []
            for attempt in range(1, MAX_REFILL_ATTEMPTS + 1):
                qa_list = get_multiple_filled_qa_pairs(
                    template_row=pd.DataFrame([template_row]),
                    df_raw=df_clean,
                    sql_conn=conn,
                    num_samples=needed,
                    max_retries_per_sample=15,
                    q_col=COL_Q_TEMPLATE,
                    a_col=COL_A_TEMPLATE,
                )

                if not qa_list:
                    print(
                        f"   ⚠️ 第 {attempt}/{MAX_REFILL_ATTEMPTS} 次随机填充无结果，重新筛选..."
                    )
                    continue

                valid_qa_list = []
                failed_pairs = []
                for pair in qa_list:
                    if pair.get("validation") == "MATCH":
                        valid_qa_list.append(pair)
                    else:
                        failed_pairs.append(pair)
                        print(f"   ⚠️ 丢弃一条 SQL 校验失败样本: {pair.get('validation')}")

                if valid_qa_list:
                    break
                print(
                    f"   ⚠️ 第 {attempt}/{MAX_REFILL_ATTEMPTS} 次随机填充全部 SQL 校验失败，重新筛选..."
                )

            if not qa_list:
                print(f"   ❌ 连续 {MAX_REFILL_ATTEMPTS} 次随机填充无结果，跳过")
                stats["skipped_no_data"] += 1
                continue

            if failed_pairs and not valid_qa_list:
                print(f"   ❌ 连续 {MAX_REFILL_ATTEMPTS} 次随机填充均 SQL 校验失败，跳过")
                stats["skipped_sql_fail"] += 1
                continue

            batch_data = []
            for pair in valid_qa_list:
                row_data = {
                    "问题模版": q_temp,
                    "回答模版": a_temp,
                    "原始填充问题": pair.get("filled_question", ""),
                    # 纯填充模式：生成问题直接用填充问题，不做 LLM 改写
                    "生成问题": pair.get("filled_question", ""),
                    "生成结果": pair.get("answer", ""),
                    "标准答案": pair.get("answer", ""),
                    "SQL语句": pair.get("sql", ""),
                    "SQL验证状态": pair.get("validation", ""),
                    "是否有效": pair.get("is_valid", False),
                    "槽位信息JSON": json.dumps(pair.get("slot_mapping", {}), ensure_ascii=False),
                }
                batch_data.append(row_data)

            if batch_data:
                df_batch = pd.DataFrame(batch_data)
                for col in CSV_COLUMNS:
                    if col not in df_batch.columns:
                        df_batch[col] = ""
                df_batch = df_batch[CSV_COLUMNS]
                df_batch.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8-sig",
                )
                stats["saved_rows"] += len(batch_data)
                stats["success"] += 1
                print(f"   ✅ 已保存 {len(batch_data)} 条")
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            stats["errors"] += 1

    conn.close()

    print("\n" + "=" * 50)
    print("🎉 处理完成（纯填充模式）")
    print(f"   - 成功处理模板: {stats['success']}")
    print(f"   - 跳过(已满): {stats['skipped_full']}")
    print(f"   - 跳过(无数据): {stats['skipped_no_data']}")
    print(f"   - 跳过(SQL失败): {stats['skipped_sql_fail']}")
    print(f"   - 异常/错误: {stats['errors']}")
    print(f"   - 总计保存行数: {stats['saved_rows']}")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
