import pandas as pd
import asyncio
import time
import json
import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "text2sql_pipeline"))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from text2sql_pipeline.main2 import XiYanSQLSystem


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_value(val):
    """
    值标准化：
    1. 去除首尾空格
    2. 转为字符串
    3. 如果是数字字符串，统一转为 float 比较 (解决 100 == 100.0 问题)
    """
    s = str(val).strip()
    if is_float(s):
        return round(float(s), 2)
    return s


def parse_ground_truth(gt_str):
    """
    解析 CSV 里的真实答案。
    如果包含逗号（如 "A,B"），则视为集合 {'A', 'B'}。
    """
    gt_str = str(gt_str).strip()
    # 处理 CSV 中的 NaN 或空值
    if gt_str.lower() in ['nan', 'none', '', 'null']:
        return set()

    # 按逗号或分号拆分多值答案
    parts = re.split(r'[,，;；]', gt_str)
    return {normalize_value(p) for p in parts if p.strip()}


def normalize_execution_result(result):
    """
    解析 SQL 执行结果。
    默认策略：只取第一列（因为 Text-to-SQL 只有一列是答案，其他列通常是辅助信息）。
    """
    if not result:
        return set()

    normalized_set = set()

    for row in result:
        # 确保 row 是元组或列表
        if not isinstance(row, (list, tuple)):
            continue

        clean_items = []
        for x in row:
            if x is None:
                clean_items.append("")
            else:
                v = normalize_value(x)
                clean_items.append(str(v).strip())

        if all(item == "" for item in clean_items):
            continue

        row_repr = "|".join(clean_items)
        normalized_set.add(row_repr)

    return normalized_set


async def main():
    # 1. 加载数据
    csv_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "train_dataset_with_sql_and_slots.csv"))
    df = pd.read_csv(csv_path)
    df = df.head(100) # 调试时只跑前100条

    print(f"加载 {len(df)} 条测试样本")

    system = XiYanSQLSystem()

    correct_count = 0
    total = len(df)
    logs = []

    for idx, row in df.iterrows():
        question = str(row["生成问题"]).strip()
        raw_gt = row["生成结果"]

        print(f"\n[{idx + 1}/{total}] 问题: {question}")

        # 运行系统
        start_time = time.time()
        try:
            output = await system.run_pipeline(question)
            exec_time = time.time() - start_time

            # --- 核心比对逻辑 ---
            gt_set = parse_ground_truth(raw_gt)
            pred_set = normalize_execution_result(output.get("execution_result"))

            # 集合相等比对 (无视顺序)
            is_correct = (gt_set == pred_set)

            # 如果集合没对上，尝试检测是否是包含关系（可选，防止 GT 不全）
            # if not is_correct and gt_set and gt_set.issubset(pred_set):
            #     is_correct = True # 这种叫 "Recall 100%" 策略，看你需不需要

            status = "✅ 正确" if is_correct else "❌ 错误"
            if is_correct: correct_count += 1

            # 打印调试信息（让你看清楚到底比了什么）
            print(f"SQL: {output.get('final_sql')}")
            print(f"🔴 真实集合 (GT): {gt_set}")
            print(f"🔵 预测集合 (Pred): {pred_set}")
            print(f"结果: {status} | 耗时: {exec_time:.2f}s")

            logs.append({
                "id": idx,
                "question": question,
                "sql": output.get('final_sql'),
                "gt_raw": str(raw_gt),
                "gt_parsed": list(gt_set),
                "pred_parsed": list(pred_set),
                "is_correct": is_correct
            })

        except Exception as e:
            print(f"❌ 系统异常: {e}")
            logs.append({
                "id": idx,
                "question": question,
                "error": str(e),
                "is_correct": False,
            })

    # 汇总
    acc = correct_count / total
    print("\n" + "=" * 50)
    print(f"最终准确率: {acc:.2%} ({correct_count}/{total})")

    # 保存错误日志方便分析
    error_logs = [l for l in logs if not l['is_correct']]
    with open("error_analysis.json", "w", encoding='utf-8') as f:
        json.dump(error_logs, f, ensure_ascii=False, indent=2)
    print("错误日志已保存至 error_analysis.json")


if __name__ == "__main__":
    main()
