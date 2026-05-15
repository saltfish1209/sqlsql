"""
Text-to-SQL evaluation for the retrieval-first pipeline.

This script evaluates the end-to-end system on the MATCH subset of the training
CSV. It prints per-sample logs and writes summary / error reports.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from generation.multi_result_utils import (
    MULTI_RESULT_SEP,
    split_multi_result_string,
)
from training.template_split import split_dataframe_by_template


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_value(val):
    """统一返回字符串：避免 GT/Pred 一边是 float 一边是 str 的类型假阴性。"""
    s = str(val).strip()
    if is_float(s):
        f = round(float(s), 2)
        # 与 SQL 端 ROUND(x, 2) 行为对齐：整数显示为 '1000' 而非 '1000.0'
        return str(int(f)) if f.is_integer() else str(f)
    return s


# 元组 repr 提取：[(v1,), (v2, v3)] -> [['v1'], ['v2', 'v3']]
_TUPLE_RE = re.compile(r"\(([^()]*)\)")


def _split_row_cells(row_str: str) -> list[str]:
    """
    一行内多列的拆分：与 construct.py 的 '|'.join 对齐，仅按 '|' 切。
    单个单元格内的英文 ',' 必须保留（如 '环网柜,AC10kV,630A,SF6,户内'）。
    """
    return [c.strip() for c in row_str.split("|")]


def _parse_single_gt(gt_str: str) -> set:
    """单条 GT 字符串 → set[str]。"""
    gt_str = str(gt_str).strip()
    if gt_str.lower() in ("nan", "none", "", "null"):
        return set()

    # —— 兼容 [(v,), (v2,)] 形态 ——
    if gt_str.startswith("[") and gt_str.endswith("]"):
        tuples = _TUPLE_RE.findall(gt_str[1:-1])
        if tuples:
            collected: list[str] = []
            for t in tuples:
                head = t.rstrip()
                if head.endswith(","):
                    head = head[:-1]
                cells = [c.strip().strip("'\"") for c in re.split(r",(?![^()]*\))", head)]
                cells = [c for c in cells if c]
                collected.append("|".join(cells))
            return {normalize_value(v) for v in collected if v}

    # —— 标准形态：行间中文逗号 / 分号 ——
    rows = re.split(r"[，；;]", gt_str)
    out = set()
    for row in rows:
        row = row.strip()
        if not row:
            continue
        cells = _split_row_cells(row)
        out.add("|".join(normalize_value(c) for c in cells))
    return out


def parse_ground_truth(gt_str):
    """
    GT 由 generation/construct.py 写入：
      - 行间用中文 '，' 分隔
      - 行内多列用 '|' 拼
      - **多结果**模板（如 ``count{},count1{中标签报号}``）由
        ``MULTI_RESULT_SEP`` 分隔多个子结果，每个子结果再按上述结构拼。

    返回：
      - 单结果 → ``set[str]``（与历史代码兼容）
      - 多结果 → ``list[set[str]]``，每个子集合对应一个子问题
    """
    gt_str = str(gt_str)
    if MULTI_RESULT_SEP in gt_str:
        sub_strs = split_multi_result_string(gt_str)
        return [_parse_single_gt(s) for s in sub_strs]
    return _parse_single_gt(gt_str)


def _normalize_single_result(result) -> set:
    """单一执行结果（list[tuple]）→ set[str]，并去掉重复行。"""
    if not result:
        return set()
    out: set[str] = set()
    for row in result:
        if not isinstance(row, (list, tuple)):
            continue
        items = ["" if x is None else normalize_value(x) for x in row]
        if all(i == "" for i in items):
            continue
        out.add("|".join(items))
    return out


def _is_multi_result_payload(result) -> bool:
    """判断 pipeline 返回的 execution_result 是否是多子问题 (list[list[tuple]])。"""
    if not isinstance(result, list) or not result:
        return False
    # 必须每一项都是 list / tuple，并且其中又包含 list / tuple（即第二层是行）
    for sub in result:
        if not isinstance(sub, list):
            return False
        for r in sub:
            if not isinstance(r, (list, tuple)):
                return False
    return True


def normalize_execution_result(result):
    """
    模型执行结果 → 与 GT 同形态：
      - 单结果 → ``set[str]``
      - 多结果 → ``list[set[str]]``（pipeline 在多问题模式下会返回 ``list[list[tuple]]``）
    """
    if _is_multi_result_payload(result):
        return [_normalize_single_result(sub) for sub in result]
    return _normalize_single_result(result)


def _compare_results(gt, pred) -> bool:
    """
    GT vs 预测的统一比较：
      - 都是 list[set] → 元素个数与每个子集合都必须相等
      - 都是 set      → 集合相等
      - 形态不一致     → False（避免单结果模型瞎对上多结果 GT）
    """
    gt_is_list = isinstance(gt, list)
    pred_is_list = isinstance(pred, list)
    if gt_is_list != pred_is_list:
        return False
    if gt_is_list:
        if len(gt) != len(pred):
            return False
        return all(g == p for g, p in zip(gt, pred))
    return gt == pred


def _flatten_result_values(parsed) -> set[str]:
    """Flatten parsed set/list[set] payload into a comparable string set."""
    if isinstance(parsed, list):
        out: set[str] = set()
        for item in parsed:
            if isinstance(item, set):
                out.update(item)
        return out
    if isinstance(parsed, set):
        return set(parsed)
    return set()


def classify_error_type(gt, pred, reason: str = "") -> tuple[str, str]:
    """Classify an incorrect prediction for easier error analysis."""
    reason = str(reason or "")
    reason_lower = reason.lower()
    sql_error_markers = (
        "syntax",
        "no such column",
        "no such table",
        "ambiguous",
        "llm_error",
        "llm_connection_error",
        "empty_sql",
        "all_paths_failed",
    )
    if any(m in reason_lower for m in sql_error_markers):
        return "SQL_EXEC_ERROR", reason or "SQL execution or generation failed"

    gt_is_list = isinstance(gt, list)
    pred_is_list = isinstance(pred, list)
    if gt_is_list != pred_is_list:
        return "SHAPE_MISMATCH", "ground truth and prediction have different result shapes"

    gt_flat = _flatten_result_values(gt)
    pred_flat = _flatten_result_values(pred)

    if not gt_flat and pred_flat:
        return "EMPTY_GT", "ground truth is empty but prediction is non-empty"
    if gt_flat and not pred_flat:
        return "EMPTY_PRED", "prediction is empty"
    if not gt_flat and not pred_flat:
        return "UNKNOWN", "both normalized results are empty but comparison failed"

    overlap = gt_flat & pred_flat
    if overlap:
        return "PARTIAL_MATCH", f"prediction overlaps ground truth but is incomplete/different ({len(overlap)} shared)"
    return "VALUE_MISMATCH", "prediction and ground truth are non-empty but disjoint"


def _stringify_for_log(parsed):
    """日志展示：把 set / list[set] 转成可读结构。"""
    if isinstance(parsed, list):
        return [sorted(s) for s in parsed]
    return sorted(parsed)


def _load_test_df() -> pd.DataFrame | None:
    """Load the evaluation slice from the training CSV."""
    csv_path = str(settings.train_csv)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] training data file not found: {csv_path}")
        return None
    df_full = pd.read_csv(csv_path)
    if "SQL验证状态" in df_full.columns:
        df_full = df_full[df_full["SQL验证状态"] == "MATCH"].copy()
    if "问题模版" in df_full.columns and not settings.train_split == 0.0:
        _, _, df = split_dataframe_by_template(
            df_full,
            template_col="问题模版",
            train_split=settings.train_split,
            val_split=settings.val_split,
        )
    else:
        df = df_full.copy()
    df = df.reset_index(drop=True)
    print(f"[Eval] loaded {len(df)} rows from {csv_path}")
    return df


async def run_evaluation(
    system,
    output_path: str | None = None,
    label: str = "System",
    full_output_path: str | None = None,
    concurrency: int = 1,
) -> dict:
    """
    通用评估入口 —— 任何带 ``async run_pipeline(question)->dict`` 接口的系统都可调用。

    Args:
        system: 已初始化好的系统实例 (例如 TextToSQLSystem 或 BaselineSystem)。
        output_path: 错误日志 JSON 输出路径；为 None 时写到本文件同目录的 error_analysis.json。
        label: 仅用于日志前缀。
        full_output_path: 全量运行日志输出路径；为 None 时写到本文件同目录的 run_report.json。
        concurrency: 并行请求数（默认 1 即串行；建议不超过 vLLM 的 max-num-seqs）。

    Returns:
        包含 accuracy / correct / total / 平均耗时 等指标的字典。
    """
    eval_start = time.time()
    df = _load_test_df()
    if df is None:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    total = len(df)
    has_gt_sql = "SQL语句" in df.columns

    # 准备所有任务数据
    tasks_data = []
    for idx, row in df.iterrows():
        tasks_data.append({
            "idx": idx,
            "question": str(row["生成问题"]).strip(),
            "raw_gt": row["生成结果"],
            "gt_sql": str(row["SQL语句"]).strip() if has_gt_sql else "",
        })

    # 结果收集（按 idx 排序）
    results_map: dict[int, dict] = {}
    semaphore = asyncio.Semaphore(concurrency)
    completed = [0]  # mutable counter for progress

    async def _run_one(task: dict) -> None:
        idx = task["idx"]
        question = task["question"]
        raw_gt = task["raw_gt"]
        gt_sql = task["gt_sql"]

        async with semaphore:
            t0 = time.time()
            try:
                output = await system.run_pipeline(question)
                dt = time.time() - t0
                first_infer = float(output.get("first_inference_time", 0.0) or 0.0)
                total_cost = float(output.get("cost_time", dt) or dt)
                repair_times = output.get("repair_times", []) or []

                gt_parsed = parse_ground_truth(raw_gt)
                pred_parsed = normalize_execution_result(output.get("execution_result"))
                entities = output.get("entities") or []
                pred_sql = output.get("final_sql") or ""
                reason = str(output.get("reason") or "")
                ok = _compare_results(gt_parsed, pred_parsed)
                icon = "OK" if ok else "FAIL"
                error_type = ""
                error_detail = ""
                if not ok:
                    error_type, error_detail = classify_error_type(
                        gt_parsed, pred_parsed, reason=reason
                    )

                completed[0] += 1
                print(f"\n[{label}][{completed[0]}/{total}] {icon} | "
                      f"total={total_cost:.2f}s first_infer={first_infer:.2f}s "
                      f"repairs={repair_times}")
                if not ok:
                    print(f"  错误类型   : {error_type} ({error_detail})")
                print(f"  问题      : {question}")
                print(f"  实体/关键词: {entities}")
                print(f"  正确结果   : {_stringify_for_log(gt_parsed)}")
                print(f"  模型结果   : {_stringify_for_log(pred_parsed)}")
                print(f"  正确 SQL  : {gt_sql}")
                print(f"  模型 SQL  : {pred_sql}")

                results_map[idx] = {
                    "log": {
                        "id": idx,
                        "question": question,
                        "entities": entities,
                        "sql": pred_sql,
                        "gt_sql": gt_sql,
                        "gt_raw": str(raw_gt),
                        "gt_parsed": _stringify_for_log(gt_parsed),
                        "pred_parsed": _stringify_for_log(pred_parsed),
                        "is_multi": isinstance(gt_parsed, list),
                        "is_correct": ok,
                        "error_type": error_type,
                        "error_detail": error_detail,
                        "reason": reason,
                    },
                    "first_infer": first_infer,
                    "total_cost": total_cost,
                    "repair_times": repair_times,
                    "is_correct": ok,
                }
            except Exception as e:
                dt = time.time() - t0
                completed[0] += 1
                print(f"\n[{label}][{completed[0]}/{total}] FAIL(异常) {dt:.2f}s")
                print(f"  问题      : {question}")
                print(f"  正确 SQL  : {gt_sql}")
                print(f"  异常       : {type(e).__name__}: {e}")
                results_map[idx] = {
                    "log": {
                        "id": idx, "question": question,
                        "gt_sql": gt_sql,
                        "error": str(e), "is_correct": False,
                        "error_type": "EXCEPTION",
                        "error_detail": f"{type(e).__name__}: {e}",
                    },
                    "first_infer": 0.0,
                    "total_cost": dt,
                    "repair_times": [],
                    "is_correct": False,
                }

    # 并行执行
    if concurrency > 1:
        print(f"[{label}] 并行模式: concurrency={concurrency}")
    await asyncio.gather(*[_run_one(t) for t in tasks_data])

    # 汇总（按原始顺序）
    correct = 0
    logs = []
    sum_first_infer = 0.0
    sum_total = 0.0
    sum_repair_each = 0.0
    repair_rounds = 0

    for idx in sorted(results_map.keys()):
        r = results_map[idx]
        logs.append(r["log"])
        if r["is_correct"]:
            correct += 1
        sum_first_infer += r["first_infer"]
        sum_total += r["total_cost"]
        sum_repair_each += sum(r["repair_times"])
        repair_rounds += len(r["repair_times"])

    acc = correct / total if total else 0.0
    avg_first_infer = sum_first_infer / total if total else 0.0
    avg_total = sum_total / total if total else 0.0
    avg_repair_each = sum_repair_each / repair_rounds if repair_rounds else 0.0
    eval_elapsed = time.time() - eval_start

    print(f"\n{'=' * 50}")
    print(f"[{label}] 最终准确率: {acc:.2%} ({correct}/{total})")
    print(f"[{label}] 平均首次推理耗时: {avg_first_infer:.2f}s")
    print(f"[{label}] 平均每次修正耗时: {avg_repair_each:.2f}s (共 {repair_rounds} 次修正)")
    print(f"[{label}] 平均总耗时: {avg_total:.2f}s")
    err_logs = [l for l in logs if not l["is_correct"]]
    error_type_counts = dict(Counter(l.get("error_type") or "UNKNOWN" for l in err_logs))
    print(f"[{label}] 评测总耗时: {eval_elapsed:.2f}s")
    print(f"[{label}] 错误类型统计: {error_type_counts}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "error_analysis.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(err_logs, f, ensure_ascii=False, indent=2)
    print(f"[{label}] 错误日志: {output_path}")

    summary = {
        "label": label,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_first_inference_time": avg_first_infer,
        "avg_repair_time_each": avg_repair_each,
        "repair_rounds": repair_rounds,
        "avg_total_time": avg_total,
        "evaluation_elapsed_time": eval_elapsed,
        "error_count": len(err_logs),
        "success_count": total - len(err_logs),
        "error_type_counts": error_type_counts,
    }
    if full_output_path is None:
        full_output_path = os.path.join(os.path.dirname(__file__), "run_report.json")
    os.makedirs(os.path.dirname(full_output_path) or ".", exist_ok=True)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "all_logs": logs,
                "error_logs": err_logs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[{label}] 全量日志: {full_output_path}")

    return {
        **summary,
        "error_log_path": output_path,
        "full_log_path": full_output_path,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="主流程 TextTo-SQL 端到端评估")
    parser.add_argument(
        "--eval-csv",
        type=str,
        default="",
        help="评测 CSV 路径；缺省不改 settings.train_csv",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="使用 CSV 中全部 MATCH 行（train_split=val_split=0）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并行请求数（默认 1；建议不超过 vLLM max-num-seqs）",
    )
    args = parser.parse_args()
    if args.eval_csv:
        settings.train_csv = Path(args.eval_csv)
    if args.use_full_data:
        settings.train_split = 0.0
        settings.val_split = 0.0

    from pipeline.system import TextToSQLSystem

    system = TextToSQLSystem()
    await run_evaluation(
        system,
        label="full_pipeline",
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    asyncio.run(main())
