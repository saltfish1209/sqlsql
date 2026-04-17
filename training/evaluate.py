"""
端到端评估脚本 —— 使用独立测试集评估 Text-to-SQL 准确率。
───────────────────────────────────────────────────────────────
数据切分与 prepare_data.py 完全一致 (80/10/10, random_state=42)，
只使用最后 10% 作为测试集，避免数据泄露。

DEBUG_MODE=True  : 输出 SQL、GT、Pred 详细信息
DEBUG_MODE=False : 仅输出 [编号] 状态 耗时
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.system import TextToSQLSystem
from pipeline.utils import debug_print

DEBUG = settings.debug_mode


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_value(val):
    s = str(val).strip()
    if is_float(s):
        return round(float(s), 2)
    return s


def parse_ground_truth(gt_str):
    gt_str = str(gt_str).strip()
    if gt_str.lower() in ("nan", "none", "", "null"):
        return set()
    parts = re.split(r"[,，;；]", gt_str)
    return {normalize_value(p) for p in parts if p.strip()}


def normalize_execution_result(result):
    if not result:
        return set()
    out = set()
    for row in result:
        if not isinstance(row, (list, tuple)):
            continue
        items = []
        for x in row:
            items.append("" if x is None else str(normalize_value(x)).strip())
        if all(i == "" for i in items):
            continue
        out.add("|".join(items))
    return out


async def main():
    csv_path = str(settings.train_csv)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 训练数据文件不存在: {csv_path}")
        return
    df_full = pd.read_csv(csv_path)
    df_full = df_full[df_full["SQL验证状态"] == "MATCH"].copy()
    df_full = df_full.sample(frac=1, random_state=settings.random_state).reset_index(drop=True)

    total_len = len(df_full)
    val_end = int(total_len * (settings.train_split + settings.val_split))
    df = df_full.iloc[val_end:].reset_index(drop=True)
    print(f"总 MATCH 数据 {total_len} 条 → 测试集(后 {settings.test_split:.0%}): {len(df)} 条")

    system = TextToSQLSystem()
    correct = 0
    total = len(df)
    logs = []

    for idx, row in df.iterrows():
        question = str(row["生成问题"]).strip()
        raw_gt = row["生成结果"]
        debug_print(f"\n[{idx+1}/{total}] 问题: {question}")

        t0 = time.time()
        try:
            output = await system.run_pipeline(question)
            dt = time.time() - t0
            gt_set = parse_ground_truth(raw_gt)
            pred_set = normalize_execution_result(output.get("execution_result"))
            ok = gt_set == pred_set
            icon = "OK" if ok else "FAIL"
            if ok:
                correct += 1
            if DEBUG:
                print(f"  SQL: {output.get('final_sql')}")
                print(f"  GT:  {gt_set}")
                print(f"  Pred:{pred_set}")
                print(f"  {icon} | {dt:.2f}s")
            else:
                print(f"[{idx+1}/{total}] {icon} {dt:.2f}s")
            logs.append({
                "id": idx, "question": question,
                "sql": output.get("final_sql"),
                "gt_raw": str(raw_gt),
                "gt_parsed": list(gt_set),
                "pred_parsed": list(pred_set),
                "is_correct": ok,
            })
        except Exception as e:
            dt = time.time() - t0
            msg = f"FAIL(异常) {dt:.2f}s"
            print(f"[{idx+1}/{total}] {msg}" if not DEBUG else f"  FAIL 异常: {e}")
            logs.append({"id": idx, "question": question, "error": str(e), "is_correct": False})

    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"最终准确率: {acc:.2%} ({correct}/{total})")

    err_logs = [l for l in logs if not l["is_correct"]]
    out_path = os.path.join(os.path.dirname(__file__), "error_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(err_logs, f, ensure_ascii=False, indent=2)
    print(f"错误日志: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
