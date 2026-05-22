"""
评估主系统 SchemaLinker 在验证集上的字段召回率。

指标说明：
- gold 字段来自 问题模版 + 回答模版 占位符解析
- 召回命中采用比例分：|命中字段| / |gold字段|
  例如 gold=3 命中2，则得分 0.6667（不是 0）
- 分别评估：
  1) Top20候选召回
  2) 断崖/比例截断后的精简schema召回

输出：
- training/schema_recall_val_details.jsonl
- training/schema_recall_val_summary.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.schema_linker import SchemaLinker
from training.template_split import split_dataframe_by_template


def extract_cols_from_template(template_str: str) -> list[str]:
    if pd.isna(template_str):
        return []
    matches = re.findall(r"\{([^}]+)\}", str(template_str))
    cols: list[str] = []
    for m in matches:
        core = m.split("|")[0].strip()
        for sub in re.split(r"[,，]", core):
            c = sub.strip()
            if c:
                cols.append(c)
    return list(dict.fromkeys(cols))


def gold_columns(row: pd.Series, all_columns: set[str]) -> list[str]:
    q_cols = extract_cols_from_template(str(row.get("问题模版", "")))
    a_cols = extract_cols_from_template(str(row.get("回答模版", "")))
    gold = list(dict.fromkeys(q_cols + a_cols))
    return [c for c in gold if c in all_columns]


def recall_ratio(gold: set[str], pred: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & pred) / len(gold)


def _format_column_entry(item: dict) -> dict:
    """提取候选列的列名、相关性分数及关键 profile 信息，统一输出格式。"""
    col = item.get("列名") or ""
    entry: dict = {"列名": str(col), "相关性分数": float(item.get("相关性分数", 0.0))}
    for key in ("列描述", "字段类型", "是否枚举", "空值率", "唯一值数", "示例值", "格式", "范围"):
        if key in item:
            entry[key] = item[key]
    return entry


def _format_columns_list(items: list[dict] | None) -> list[dict]:
    """批量格式化候选列列表。"""
    return [_format_column_entry(x) for x in (items or []) if x.get("列名")]


def _iter_evidence_details(pack) -> dict:
    """按实体聚合证据详情，并保留每种匹配方式的结果与分数。"""
    evidence = getattr(pack, "证据详情", {}) or {}
    method_order = ("精确匹配", "模糊匹配", "向量匹配")
    by_entity: dict[str, dict] = {}

    for match_type, entity_map in evidence.items():
        for entity_text, hits in (entity_map or {}).items():
            entry = by_entity.setdefault(
                entity_text,
                {
                    "实体文本": entity_text,
                    "匹配明细": {
                        "精确匹配": [],
                        "模糊匹配": [],
                        "向量匹配": [],
                    },
                },
            )
            for hit in hits or []:
                entry["匹配明细"].setdefault(match_type, []).append(
                    {
                        "对应匹配值": hit.get("对应匹配值"),
                        "所在匹配列": hit.get("所在匹配列"),
                        "匹配方式": hit.get("匹配方式") or match_type,
                        "相关性分数": float(hit.get("相关性分数", 0.0)),
                    }
                )

    entity_items = []
    for entity_text in sorted(by_entity.keys()):
        item = by_entity[entity_text]
        item["匹配明细"] = {k: item["匹配明细"].get(k, []) for k in method_order}
        entity_items.append(item)
    return {
        "按实体明细": entity_items,
        "原始证据详情": evidence,
    }


def _gold_column_analysis(gold_set: set[str], top20_cols: set[str], compact_cols: set[str]) -> list[dict]:
    """对每个 gold 列做命中分析，标注 top20 命中但精简 schema 未命中的情况。"""
    analysis: list[dict] = []
    for col in sorted(gold_set):
        in_top20 = col in top20_cols
        in_compact = col in compact_cols
        entry = {"列名": col, "top20命中": in_top20, "精简schema命中": in_compact}
        if in_top20 and not in_compact:
            entry["标注"] = "⚠ top20召回命中但精简schema丢失"
        elif not in_top20:
            entry["标注"] = "✗ top20未召回"
        analysis.append(entry)
    return analysis


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Schema recall evaluation on validation split")
    parser.add_argument("--csv", type=str, default="", help="母表CSV，默认 settings.train_csv")
    parser.add_argument("--train-split", type=float, default=None, help="训练切分比例，默认 settings.train_split")
    parser.add_argument("--val-split", type=float, default=None, help="验证切分比例，默认 settings.val_split")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(settings.train_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"训练集文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if "SQL验证状态" in df.columns:
        df = df[df["SQL验证状态"] == "MATCH"].copy()

    train_split = args.train_split if args.train_split is not None else settings.train_split
    val_split = args.val_split if args.val_split is not None else settings.val_split

    df_train, df_val, df_test = split_dataframe_by_template(
        df,
        template_col="问题模版",
        train_split=train_split,
        val_split=val_split,
    )
    if df_val.empty:
        print("[WARN] 当前 val_split=0 或验证集为空，将退出。")
        return

    linker = SchemaLinker(str(settings.schema_json_path), str(settings.csv_path))
    all_columns = set(linker.column_names)

    details = []
    total_eval = 0
    failed_eval = 0
    top20_success = 0
    top20_fail_then_compact_fail = 0
    top20_success_compact_fail = 0

    for i, row in df_val.reset_index(drop=True).iterrows():
        question = str(row.get("生成问题", "")).strip()
        if not question:
            continue

        gold = gold_columns(row, all_columns)
        if not gold:
            continue

        pack = linker.retrieve(question, [])
        top20 = pack.Top20候选
        compact = pack.精简schema
        must_have_list = list(pack.必须列集合 or [])
        evidence_details = _iter_evidence_details(pack)

        top20_cols = {str(x.get("列名", "")).strip() for x in top20 if x.get("列名")}
        compact_cols = {str(x.get("列名", "")).strip() for x in compact if x.get("列名")}
        gold_set = set(gold)

        top20_r = recall_ratio(gold_set, top20_cols)
        final_r = recall_ratio(gold_set, compact_cols)
        total_eval += 1
        if top20_r == 1.0:
            top20_success += 1
        if final_r < 1.0:
            failed_eval += 1

        if top20_r < 1.0:
            top20_fail_then_compact_fail += 1
        elif final_r < 1.0:
            top20_success_compact_fail += 1

        if final_r == 1.0:
            continue

        top20_formatted = _format_columns_list(top20)
        compact_formatted = _format_columns_list(compact)
        col_analysis = _gold_column_analysis(gold_set, top20_cols, compact_cols)

        record = {
            "idx": int(i),
            "question": question,
            "gold_columns": sorted(gold_set),
            "top20_recall": round(top20_r, 4),
            "final_recall": round(final_r, 4),
            "top20_candidates": top20_formatted,
            "compact_schema": compact_formatted,
            "must_have_columns": must_have_list,
            "evidence_details": evidence_details,
            "gold_column_analysis": col_analysis,
        }
        details.append(record)

        top20_ok_compact_fail = top20_r == 1.0 and final_r < 1.0

        print("-" * 80)
        if top20_ok_compact_fail:
            print("⚠⚠⚠ [top20召回成功 但 精简schema召回失败] ⚠⚠⚠")
        print(f"idx: {i}")
        print(f"question: {question}")
        print(f"gold_columns: {sorted(gold_set)}")
        print(f"top20_recall: {round(top20_r, 4)}")
        print(f"final_recall (精简schema): {round(final_r, 4)}")
        print(f"\n[Top20 候选列] ({len(top20_formatted)} 列)")
        for c in top20_formatted:
            marker = " ★" if c["列名"] in gold_set else ""
            print(f"  {c['列名']} (score={c['相关性分数']:.4f}){marker}")
        print(f"\n[精简 Schema] ({len(compact_formatted)} 列, 含 must_have)")
        for c in compact_formatted:
            marker = " ★" if c["列名"] in gold_set else ""
            src = " [must_have]" if c["列名"] in set(must_have_list) and c["列名"] not in {
                x.get("列名") for x in (pack.Top20候选 or [])
            } else ""
            print(f"  {c['列名']} (score={c['相关性分数']:.4f}){src}{marker}")
        if must_have_list:
            print(f"\n[Must_have 列] {must_have_list}")
        entity_evidence = evidence_details.get("按实体明细", [])
        if entity_evidence:
            print(f"\n[证据实体匹配详情] ({len(entity_evidence)} 个实体)")
            for ev in entity_evidence:
                print(f"  实体: 「{ev['实体文本']}」")
                for match_type in ("精确匹配", "模糊匹配", "向量匹配"):
                    hits = ev.get("匹配明细", {}).get(match_type, [])
                    if not hits:
                        continue
                    print(f"    - {match_type}:")
                    for hit in hits:
                        print(
                            f"      列={hit.get('所在匹配列')} 值={hit.get('对应匹配值')} "
                            f"方式={hit.get('匹配方式')} score={float(hit.get('相关性分数', 0.0)):.4f}"
                        )
        print(f"\n[Gold 列命中分析]")
        for a in col_analysis:
            tag = a.get("标注", "✓ 已命中")
            print(f"  {a['列名']}: top20={'✓' if a['top20命中'] else '✗'}  "
                  f"精简schema={'✓' if a['精简schema命中'] else '✗'}  {tag}")

    out_dir = Path(__file__).parent
    detail_path = out_dir / "schema_recall_val_details.jsonl"
    summary_path = out_dir / "schema_recall_val_summary.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n = total_eval
    summary = {
        "dataset": str(csv_path),
        "split": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
        },
        "evaluated_samples": n,
        "failed_samples": failed_eval,
        "failed_ratio": round(failed_eval / n, 4) if n else 0.0,
        "top20_success_samples": top20_success,
        "top20_success_but_final_fail_samples": top20_success_compact_fail,
        "top20_fail_and_final_fail_samples": top20_fail_then_compact_fail,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Schema recall evaluation done (validation split)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"details: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
