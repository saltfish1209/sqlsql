from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

_EVAL_DB_LOCK: threading.Lock | None = None
_EVAL_RETRIEVE_LOCK: threading.Lock | None = None
_LLM_SEMAPHORE: asyncio.Semaphore | None = None
_PATH_PARALLEL: bool = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from pipeline.consensus_vote import annotate_candidate_risk, select_by_consensus
from pipeline.utils import TokenTracker, to_halfwidth


SUMMARY_FIELDNAMES = [
    "方案",
    "准确率",
    "正确题数(总题数)",
    "花费时间",
    "tokens平均消耗",
    "recall",
    "Pass@K",
    "selected_changed_by_refiner",
    "judge_trigger_count",
    "repair_success_count",
    "repair_harm_count",
]

# 模式1：不同 schema 构建策略对比
SCHEMA_COMPARE_CONFIGS = [
    ("全量schema+llm", "full_schema", "full_schema_recall"),
    ("topk截断schema+llm", "topk_schema", "topk_schema_recall"),
    ("断崖截取后schema+llm", "cliff_schema", "cliff_schema_recall"),
    ("主系统", "main_system", "main_system_schema_recall"),
]

# 模式2：主系统消融实验
ABLATION_CONFIGS = [
    ("主系统+全量schema", "main_full_schema", "main_full_schema_recall"),
    ("主系统", "main_system", "main_system_schema_recall"),
    ("主系统+topk截断schema", "main_topk_schema", "main_topk_schema_recall"),
    ("主系统+direct单路径", "main_direct_only", "main_direct_only_recall"),
    ("主系统+icl单路径", "main_icl_only", "main_icl_only_recall"),
    ("主系统+intent_plan单路径", "main_intent_plan_only", "main_intent_plan_only_recall"),
    ("主系统(无refiner)", "main_no_refiner", "main_no_refiner_schema_recall"),
    ("主系统(无实体提取)", "main_no_entity", "main_no_entity_schema_recall"),
    ("主系统(无审查judge, refiner+一致性投票)", "main_no_judge", "main_no_judge_recall"),
]


def _install_eval_runtime(system, *, llm_concurrency: int = 0) -> None:
    """评测期线程安全 + 可选全局 LLM 并发上限，提高 GPU 利用率。"""
    global _EVAL_DB_LOCK, _EVAL_RETRIEVE_LOCK, _LLM_SEMAPHORE
    _EVAL_DB_LOCK = threading.Lock()
    _EVAL_RETRIEVE_LOCK = threading.Lock()

    original_execute = system.db_engine.execute_sql

    def _locked_execute(sql: str):
        assert _EVAL_DB_LOCK is not None
        with _EVAL_DB_LOCK:
            return original_execute(sql)

    system.db_engine.execute_sql = _locked_execute  # type: ignore[method-assign]

    if llm_concurrency > 0:
        _LLM_SEMAPHORE = asyncio.Semaphore(llm_concurrency)
        original_create = system.client.chat.completions.create

        async def _limited_create(*args, **kwargs):
            assert _LLM_SEMAPHORE is not None
            async with _LLM_SEMAPHORE:
                return await original_create(*args, **kwargs)

        system.client.chat.completions.create = _limited_create  # type: ignore[method-assign]
    else:
        _LLM_SEMAPHORE = None


async def _retrieve_async(system, question: str, entities: list | None = None):
    """CrossEncoder 检索放到线程池，避免阻塞事件循环。"""
    assert _EVAL_RETRIEVE_LOCK is not None

    def _do_retrieve():
        with _EVAL_RETRIEVE_LOCK:
            return system.linker.retrieve(question, entities or [])

    return await asyncio.to_thread(_do_retrieve)


async def _prepare_single_pipeline_async(system, question: str, tracker: TokenTracker | None = None):
    tracker = tracker or TokenTracker()
    return await system._prepare_single_pipeline(question, tracker)


def _stages_from_candidate_pack(
    system,
    question: str,
    candidate_pack,
    *,
    cliff: bool = False,
) -> dict:
    question = to_halfwidth(question)
    norm_question = system._normalize_question(question)
    top20 = list(candidate_pack.Top20候选 or [])
    cliff_rows = system.linker._rank_candidates(top20, settings.candidate_top_k)
    cliff_schema_prompt = _build_schema_markdown(system, cliff_rows)
    if cliff:
        schema_rows = cliff_rows
        key = "cliff_schema"
    else:
        schema_rows = top20
        key = "topk_schema"
    schema_prompt = _build_schema_markdown(system, schema_rows)
    repair_schema_prompt = _build_schema_markdown(system, top20)
    return {
        "question": question,
        "norm_question": norm_question,
        "candidate_pack": candidate_pack,
        key: schema_rows,
        "cliff_schema_prompt": cliff_schema_prompt,
        "schema_prompt": schema_prompt,
        "repair_schema_prompt": repair_schema_prompt,
    }


async def _prepare_shared_retrieval_stages(system, question: str) -> tuple[dict, dict]:
    """一次 CrossEncoder 检索，同时产出 topk / cliff 两套路径所需 stages。"""
    question = to_halfwidth(question)
    norm_question = system._normalize_question(question)
    candidate_pack = await _retrieve_async(system, norm_question, [])
    topk_stages = _stages_from_candidate_pack(system, question, candidate_pack, cliff=False)
    cliff_stages = _stages_from_candidate_pack(system, question, candidate_pack, cliff=True)
    return topk_stages, cliff_stages


async def _run_paths(
    runners: list[Callable[[], Awaitable[Any]]],
    *,
    parallel: bool | None = None,
) -> list[Any]:
    use_parallel = _PATH_PARALLEL if parallel is None else parallel
    if use_parallel and len(runners) > 1:
        return list(await asyncio.gather(*(runner() for runner in runners)))
    outputs: list[Any] = []
    for runner in runners:
        outputs.append(await runner())
    return outputs


def _default_eval_path() -> Path:
    """兼容旧版 settings：优先 test_split_jsonl，否则回退到 data/test_split.jsonl。"""
    configured = getattr(settings, "test_split_jsonl", None)
    if configured not in (None, ""):
        return Path(configured)
    data_dir = getattr(settings, "data_dir", None)
    if data_dir in (None, ""):
        project_root = getattr(settings, "project_root", Path(__file__).resolve().parent.parent)
        data_dir = Path(project_root) / "data"
    return Path(data_dir) / "test_split.jsonl"


def _default_ablation_detail_path(out_dir: Path) -> Path:
    return out_dir / "ablation_outputs.jsonl"


def _load_eval_df(path: str | os.PathLike[str] | Path | None = None) -> list[dict]:
    eval_path = Path(path) if path else _default_eval_path()
    if not eval_path.is_file():
        raise FileNotFoundError(f"评测 test split 不存在: {eval_path}")
    if eval_path.suffix.lower() == ".csv":
        with eval_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return [dict(row) for row in rows]


def _parse_gold_columns(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    parts = re.split(r"[,，;；|]", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_cols_from_template(template_str: Any) -> list[str]:
    """解析问题/回答模版里的 {列名} / {列名|别名} / {列1,列2} 占位符。"""
    if template_str in (None, ""):
        return []
    cols: list[str] = []
    for m in re.findall(r"\{([^}]+)\}", str(template_str)):
        core = m.split("|")[0].strip()
        for sub in re.split(r"[,，]", core):
            c = sub.strip()
            if c:
                cols.append(c)
    return list(dict.fromkeys(cols))


def _gold_columns_from_row(row: dict, active_columns: list[str] | None = None) -> list[str]:
    """优先显式 gold 字段，否则由 问题模版 + 回答模版 占位符推导，并按有效列过滤。"""
    explicit = row.get("gold_columns") or row.get("Gold列") or row.get("相关列")
    if explicit not in (None, "", []):
        cols = _parse_gold_columns(explicit)
    else:
        q_cols = _extract_cols_from_template(row.get("问题模版"))
        a_cols = _extract_cols_from_template(row.get("回答模版"))
        cols = list(dict.fromkeys(q_cols + a_cols))
    if active_columns is not None:
        active = set(active_columns)
        cols = [c for c in cols if c in active]
    return cols


def _schema_names(schema_rows: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in schema_rows or []:
        col = str(item.get("列名") or item.get("column_name") or "").strip()
        if col and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _schema_recall(gold_columns: Any, selected_schema: list[dict]) -> dict:
    gold = set(_parse_gold_columns(gold_columns))
    selected = set(_schema_names(selected_schema))
    if not gold:
        return {
            "rerank_pruned_recall": "",
            "rerank_pruned_full_recall": "",
            "rerank_pruned_keep_count": len(selected),
        }
    hit = len(gold & selected)
    return {
        "rerank_pruned_recall": round(hit / len(gold), 6),
        "rerank_pruned_full_recall": int(gold.issubset(selected)),
        "rerank_pruned_keep_count": len(selected),
    }


def _schema_recall_value(gold_columns: Any, selected_schema: list[dict]) -> float | str:
    return _schema_recall(gold_columns, selected_schema)["rerank_pruned_recall"]


def _tokens(usage: dict | None) -> tuple[int, int, int]:
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def _avg(rows: list[dict], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        vals.append(float(value))
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def _sum_int(rows: list[dict], key: str) -> int:
    return int(sum(int(row.get(key) or 0) for row in rows))


def _build_summary_rows(
    rows: list[dict],
    configs: list[tuple[str, str, str]],
) -> list[dict]:
    total = len(rows)
    summary: list[dict] = []
    for label, prefix, recall_key in configs:
        correct = _sum_int(rows, f"{prefix}_correct")
        summary.append(
            {
                "方案": label,
                "准确率": round(correct / total, 6) if total else 0.0,
                "正确题数(总题数)": f"{correct}({total})",
                "花费时间": _avg(rows, f"{prefix}_time_seconds"),
                "tokens平均消耗": _avg(rows, f"{prefix}_total_tokens"),
                "recall": _avg(rows, recall_key),
                "Pass@K": _avg(rows, f"{prefix}_pass_at_k"),
                "selected_changed_by_refiner": _sum_int(
                    rows, f"{prefix}_selected_changed_by_refiner"
                ),
                "judge_trigger_count": _sum_int(rows, f"{prefix}_judge_trigger_count"),
                "repair_success_count": _sum_int(rows, f"{prefix}_repair_success_count"),
                "repair_harm_count": _sum_int(rows, f"{prefix}_repair_harm_count"),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _completed_labels(path: Path) -> set[str]:
    name_key = SUMMARY_FIELDNAMES[0]
    return {
        str(row.get(name_key) or "").strip()
        for row in _read_csv_rows(path)
        if str(row.get(name_key) or "").strip()
    }


def _print_resume_state(group_label: str, path: Path, configs: list[tuple[str, str, str]]) -> None:
    completed = _completed_labels(path)
    labels = [label for label, _prefix, _recall_key in configs]
    done = [label for label in labels if label in completed]
    pending = [label for label in labels if label not in completed]
    print(f"[Resume][{group_label}] CSV: {path}")
    print(f"[Resume][{group_label}] completed {len(done)}/{len(labels)}: {done or 'None'}")
    print(f"[Resume][{group_label}] pending: {pending or 'None'}")


def _load_extra_correct_records(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records: dict[str, dict] = {}
    baseline = data.get("baseline")
    if isinstance(baseline, dict):
        prefix = str(baseline.get("prefix") or "").strip()
        if prefix:
            records[prefix] = dict(baseline)
    for item in data.get("experiments") or []:
        prefix = str(item.get("prefix") or "").strip()
        if prefix:
            records[prefix] = dict(item)
    return records


def _correct_idx_record(
    *,
    label: str,
    prefix: str,
    rows: list[dict],
) -> dict:
    correct_idx = [
        int(row.get("idx"))
        for row in rows
        if int(row.get(f"{prefix}_correct") or 0) == 1 and row.get("idx") not in (None, "")
    ]
    details_by_idx = {
        str(int(row.get("idx"))): row["__detail__"]
        for row in rows
        if row.get("idx") not in (None, "") and isinstance(row.get("__detail__"), dict)
    }
    return {
        "label": label,
        "prefix": prefix,
        "correct_count": len(correct_idx),
        "correct_idx": correct_idx,
        "details_by_idx": details_by_idx,
    }


def _build_extra_correct_report(
    *,
    group_label: str,
    configs: list[tuple[str, str, str]],
    records: dict[str, dict],
    total_questions: int,
) -> dict:
    ordered = [
        records[prefix]
        for label, prefix, _recall_key in configs
        if prefix in records
    ]
    missing = [
        {"label": label, "prefix": prefix}
        for label, prefix, _recall_key in configs
        if prefix not in records
    ]
    use_main_system_baseline = group_label == "Ablation"
    baseline_prefix = "main_system" if use_main_system_baseline else None
    baseline = records.get(baseline_prefix) if baseline_prefix else None
    if baseline is None and not use_main_system_baseline:
        baseline = min(ordered, key=lambda item: item.get("correct_count", 0)) if ordered else None
    baseline_correct = set(baseline.get("correct_idx") or []) if baseline else set()
    baseline_details = baseline.get("details_by_idx") or {} if baseline else {}
    experiments = []
    for item in ordered:
        correct_idx = set(item.get("correct_idx") or [])
        extra = sorted(correct_idx - baseline_correct) if baseline else []
        details_by_idx = item.get("details_by_idx") or {}
        extra_details = []
        for idx in extra:
            idx_key = str(idx)
            detail = details_by_idx.get(idx_key)
            if not isinstance(detail, dict):
                detail = {"idx": idx}
            extra_details.append(
                {
                    "idx": idx,
                    "experiment": detail,
                    "baseline": baseline_details.get(idx_key),
                }
            )
        experiments.append(
            {
                "label": item.get("label"),
                "prefix": item.get("prefix"),
                "correct_count": item.get("correct_count", 0),
                "correct_idx": sorted(correct_idx),
                "extra_correct_count": len(extra),
                "extra_correct_idx": extra,
                "extra_correct_details": extra_details,
            }
        )
    return {
        "group": group_label,
        "total_questions": total_questions,
        "baseline": {
            "label": baseline.get("label"),
            "prefix": baseline.get("prefix"),
            "correct_count": baseline.get("correct_count", 0),
            "correct_idx": sorted(baseline_correct),
        } if baseline else None,
        "experiments": experiments,
        "missing_experiments": missing,
    }


def _write_extra_correct_report(
    path: Path,
    *,
    group_label: str,
    configs: list[tuple[str, str, str]],
    records: dict[str, dict],
    total_questions: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_extra_correct_report(
        group_label=group_label,
        configs=configs,
        records=records,
        total_questions=total_questions,
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _record_metrics(
    out: dict,
    *,
    prefix: str,
    raw_gt: Any,
    recall_schema: list[dict],
    gold_columns: list[str],
) -> dict:
    ok, _match = _compare_output(raw_gt, out)
    mechanism = _mechanism_metrics(raw_gt, out)
    _in, _out, total = _tokens(out.get("token_usage"))
    return {
        f"{prefix}_correct": int(ok),
        f"{prefix}_time_seconds": round(float(out.get("cost_time") or 0.0), 6),
        f"{prefix}_total_tokens": total,
        f"{prefix}_recall": _schema_recall_value(gold_columns, recall_schema),
        f"{prefix}_pass_at_k": mechanism["pass_at_k"],
        f"{prefix}_selected_changed_by_refiner": mechanism[
            "selected_changed_by_refiner"
        ],
        f"{prefix}_judge_trigger_count": mechanism["judge_trigger_count"],
        f"{prefix}_repair_success_count": mechanism["repair_success_count"],
        f"{prefix}_repair_harm_count": mechanism["repair_harm_count"],
    }


def _full_schema_rows(system) -> list[dict]:
    return [
        {"列名": str(meta.get("column_name") or "").strip()}
        for meta in system.linker.column_metadata
        if str(meta.get("column_name") or "").strip()
    ]


def _build_schema_markdown(system, schema_rows: list[dict]) -> str:
    from pipeline.schema_format import build_light_schema_markdown, enrich_schema_columns

    enriched = enrich_schema_columns(
        schema_rows,
        system.linker.column_metadata,
        system.linker.profile_detail_map,
    )
    return build_light_schema_markdown(enriched, settings.table_name)


def _full_schema_prompt(system) -> str:
    return _build_schema_markdown(system, _full_schema_rows(system))


def _build_plain_schema_sql_prompt(question: str, schema_prompt: str) -> str:
    """mode1 schema 对比：固定原始 LLM SQL prompt，仅替换 schema 内容。"""
    question = to_halfwidth(question)
    return (
        "你是一名SQL专家。请根据Schema为下列问题生成一条 SQLite SQL 查询。\n"
        "采用 sqlite，不需要加上数据库名，直接使用对应表名即可。\n\n"
        "以问题信息为生成SQL主要条件，Schema提供辅助。\n"
        "不要添加除问题所给信息外多余的约束。\n"
        f"[Schema]\n{schema_prompt}\n"
        f"[用户问题]\n{question}\n"
        "[硬性约束]\n"
        "1. SQL 关键字与列名、表名之间必须有空格分隔。\n"
        "2. 列名使用双引号包裹，如 `\"列名\"`。\n"
        "请直接输出SQL，用```sql ... ```包裹，不需要解释或其他内容。\n"
    )


async def _run_full_schema_once(system, question: str, schema_prompt: str) -> dict:
    from pipeline.generator import SQLGenerator

    start = time.time()
    tracker = TokenTracker()
    prompt = _build_plain_schema_sql_prompt(question, schema_prompt)
    sql = ""
    reason = "success"
    try:
        resp = await system.client.chat.completions.create(
            model=system.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.direct_temperature,
            timeout=settings.llm_request_timeout_sec,
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        tracker.track(resp)
        sql = SQLGenerator.extract_sql(resp.choices[0].message.content or "")
    except Exception as exc:
        reason = f"llm_error: {type(exc).__name__}: {exc}"

    result, error = (None, "EMPTY_SQL") if not sql else system.db_engine.execute_sql(sql)
    if error is not None:
        reason = error
    unique_rows: list[tuple] = []
    if result:
        seen_rows = set()
        for row in result:
            tup = tuple(row)
            if tup not in seen_rows:
                seen_rows.add(tup)
                unique_rows.append(tup)
    return {
        "final_sql": sql or None,
        "execution_result": unique_rows if result else result,
        "reason": reason,
        "cost_time": time.time() - start,
        "token_usage": tracker.get_report(),
    }


def _build_cliff_schema_markdown(system, cliff_schema: list[dict]) -> str:
    """仅断崖剪枝列 → 与主系统一致的富 schema markdown（不含 Must-have / 证据列）。"""
    return _build_schema_markdown(system, cliff_schema)


def _prepare_cliff_eval_stages(system, question: str) -> dict:
    """系统对比路径：只做 CrossEncoder 重排序 + top-k 断崖截断，不做实体提取 / must_have 补列。"""
    question = to_halfwidth(question)
    norm_question = system._normalize_question(question)
    candidate_pack = system.linker.retrieve(norm_question, [])
    cliff_schema = system.linker._rank_candidates(
        candidate_pack.Top20候选 or [],
        settings.candidate_top_k,
    )
    schema_prompt = _build_cliff_schema_markdown(system, cliff_schema)
    repair_schema_prompt = _build_schema_markdown(system, candidate_pack.Top20候选 or [])
    return {
        "question": question,
        "norm_question": norm_question,
        "candidate_pack": candidate_pack,
        "cliff_schema": cliff_schema,
        "cliff_schema_prompt": schema_prompt,
        "schema_prompt": schema_prompt,
        "repair_schema_prompt": repair_schema_prompt,
    }


def _prepare_topk_eval_stages(system, question: str) -> dict:
    """CrossEncoder 重排序后的 Top20，不做断崖截断。"""
    question = to_halfwidth(question)
    norm_question = system._normalize_question(question)
    candidate_pack = system.linker.retrieve(norm_question, [])
    topk_schema = list(candidate_pack.Top20候选 or [])
    cliff_schema = system.linker._rank_candidates(
        candidate_pack.Top20候选 or [],
        settings.candidate_top_k,
    )
    cliff_schema_prompt = _build_schema_markdown(system, cliff_schema)
    schema_prompt = _build_schema_markdown(system, topk_schema)
    return {
        "question": question,
        "norm_question": norm_question,
        "candidate_pack": candidate_pack,
        "topk_schema": topk_schema,
        "cliff_schema_prompt": cliff_schema_prompt,
        "schema_prompt": schema_prompt,
        "repair_schema_prompt": schema_prompt,
    }


async def _run_cliff_schema_once(system, question: str, *, stages: dict | None = None) -> dict:
    if stages is None:
        stages = _prepare_cliff_eval_stages(system, question)
    return await _run_full_schema_once(system, stages["question"], stages["schema_prompt"])


async def _run_topk_schema_once(system, question: str, *, stages: dict | None = None) -> dict:
    if stages is None:
        stages = _prepare_topk_eval_stages(system, question)
    return await _run_full_schema_once(system, stages["question"], stages["schema_prompt"])


def _prepare_main_no_entity_stages(system, question: str) -> dict:
    """主系统对比路径：重排序 + 断崖截断，跳过实体提取与 Must-have / 证据列。"""
    question = to_halfwidth(question)
    norm_question = system._normalize_question(question)
    candidate_pack = system.linker.retrieve(norm_question, [])
    plan_schema = system.linker._rank_candidates(
        candidate_pack.Top20候选 or [],
        settings.candidate_top_k,
    )
    final_schema = list(candidate_pack.Top20候选 or [])
    schema_prompt = _build_schema_markdown(system, plan_schema)
    repair_schema_prompt = _build_schema_markdown(system, final_schema)
    return {
        "question": question,
        "norm_question": norm_question,
        "candidate_pack": candidate_pack,
        "plan_schema": plan_schema,
        "final_schema": final_schema,
        "cliff_schema_prompt": schema_prompt,
        "schema_prompt": schema_prompt,
        "repair_schema_prompt": repair_schema_prompt,
    }


def _prepare_candidate_for_vote(system, cand: dict) -> dict:
    """无 refiner 时：执行 SQL 并补齐一致性投票所需的 status / result 字段。"""
    prepared = dict(cand)
    result, error = system.db_engine.execute_sql(prepared.get("sql") or "")
    prepared["execution_error"] = error
    if error is not None:
        prepared["status"] = "failed"
        prepared["result"] = None
        prepared["error_msg"] = error
    else:
        prepared["status"] = "success"
        prepared["result"] = result if result is not None else []
    annotate_candidate_risk(prepared, system.db_engine.check_literal_in_column)
    return prepared


def _preview_result(result, *, limit: int = 5):
    if result is None:
        return None
    preview = []
    for row in (result or [])[:limit]:
        if isinstance(row, tuple):
            preview.append(list(row))
        else:
            preview.append(row)
    return preview


def _candidate_path_record(candidate: dict) -> dict:
    debug = candidate.get("refiner_debug") or {}
    refiner_debug = debug.get("refiner") or {}
    judge_debug = debug.get("judge") or {"ran": False, "reason": ""}
    sql_before = refiner_debug.get("sql_before", candidate.get("sql"))
    return {
        "type": candidate.get("type"),
        "variant_id": candidate.get("variant_id"),
        "sql_before": sql_before,
        "sql_after": candidate.get("sql"),
        "sql": candidate.get("sql"),
        "status": candidate.get("status"),
        "execution_error": candidate.get("execution_error"),
        "error_msg": candidate.get("error_msg"),
        "result_preview": _preview_result(candidate.get("result")),
        "judge_status": candidate.get("judge_status"),
        "judge_risk": candidate.get("judge_risk"),
        "judge_reason": candidate.get("judge_reason"),
        "post_repair_judge_status": candidate.get("post_repair_judge_status"),
        "post_repair_judge_risk": candidate.get("post_repair_judge_risk"),
        "post_repair_judge_reason": candidate.get("post_repair_judge_reason"),
        "value_link_risk": candidate.get("value_link_risk"),
        "value_link_probe": candidate.get("value_link_probe") or {},
        "is_refined": bool(candidate.get("is_refined")),
        "refined_from": candidate.get("refined_from"),
        "refiner_round": candidate.get("refiner_round"),
        "source_sql": candidate.get("source_sql"),
        "repair_changed": candidate.get("repair_changed"),
        "judge_debug": judge_debug,
        "refiner_debug": refiner_debug,
    }


def _generated_candidate_record(candidate: dict) -> dict:
    return {
        "type": candidate.get("type"),
        "variant_id": candidate.get("variant_id"),
        "sql": candidate.get("sql"),
    }


def _pipeline_trace_fields(
    generated_candidates: list[dict],
    candidates: list[dict],
    selected: dict | None,
) -> dict:
    candidate_results = [
        {
            "type": item.get("type"),
            "variant_id": item.get("variant_id"),
            "status": item.get("status"),
            "execution_error": item.get("execution_error"),
            "result_preview": _preview_result(item.get("result")),
            "judge_status": item.get("judge_status"),
            "judge_risk": item.get("judge_risk"),
            "value_link_risk": item.get("value_link_risk"),
            "is_refined": bool(item.get("is_refined")),
            "refined_from": item.get("refined_from"),
        }
        for item in candidates
    ]
    return {
        "generated_candidates": [_generated_candidate_record(item) for item in generated_candidates],
        "refined_candidates": [_candidate_path_record(item) for item in candidates],
        "selected_candidate": _candidate_path_record(selected) if selected else None,
        "candidate_results": candidate_results,
        "candidate_sqls": [
            {
                "type": item.get("type"),
                "variant_id": item.get("variant_id"),
                "sql": item.get("sql"),
                "status": item.get("status"),
                "judge_status": item.get("judge_status"),
                "judge_risk": item.get("judge_risk"),
                "value_link_risk": item.get("value_link_risk"),
                "is_refined": bool(item.get("is_refined")),
                "refined_from": item.get("refined_from"),
            }
            for item in candidates
        ],
        "_candidate_pool": candidates,
        "_selected_candidate": selected,
        "_generated_candidate_count": len(generated_candidates),
    }


async def _run_main_no_entity_once(system, question: str) -> dict:
    """主系统路径（无实体提取）：不含 Must-have / 证据列，保留生成 + refiner。"""
    start = time.time()
    tracker = TokenTracker()
    stages = _prepare_main_no_entity_stages(system, question)
    candidate_pack = stages["candidate_pack"]
    schema_prompt = stages["schema_prompt"]
    repair_schema_prompt = stages["repair_schema_prompt"]

    candidates = await system.generator.generate_candidates_async(
        stages["question"],
        schema_prompt,
        tracker,
        intent_plan=stages.get("intent_plan"),
    )
    if not candidates:
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": "generation_failed",
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
        }

    confidence = (
        float(candidate_pack.Top20候选[0]["相关性分数"])
        if candidate_pack.Top20候选
        else 0.0
    )
    for cand in candidates:
        cand["confidence"] = confidence
    refined = await system.refiner.refine_async(
        schema_prompt,
        candidates,
        candidate_pack.Top20候选,
        tracker,
        repair_schema_prompt=repair_schema_prompt,
        question=stages["question"],
        judge_schema_prompt=stages.get("cliff_schema_prompt") or schema_prompt,
        intent_plan=stages.get("intent_plan"),
    )
    selected, reason, status = select_by_consensus(refined)
    if selected is None:
        trace_fields = _pipeline_trace_fields(candidates, refined, None)
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": reason,
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
            **trace_fields,
        }

    result = selected.get("result")
    unique_rows: list[tuple] = []
    if result:
        seen_rows = set()
        for row in result:
            tup = tuple(row)
            if tup not in seen_rows:
                seen_rows.add(tup)
                unique_rows.append(tup)
    trace_fields = _pipeline_trace_fields(candidates, refined, selected)
    return {
        "final_sql": selected.get("sql"),
        "execution_result": unique_rows if result else result,
        "reason": status,
        "cost_time": time.time() - start,
        "token_usage": tracker.get_report(),
        "stages": stages,
        **trace_fields,
    }


async def _run_main_no_refiner_once(system, question: str, *, stages: dict | None = None) -> dict:
    """主系统路径（无 refiner）：保留实体提取 + Must-have / 证据列，跳过修复。"""
    start = time.time()
    tracker = TokenTracker()
    if stages is None:
        stages = await _prepare_single_pipeline_async(system, question, tracker)
    candidate_pack = stages["candidate_pack"]
    schema_prompt = stages["schema_prompt"]

    candidates = await system.generator.generate_candidates_async(
        stages["question"],
        schema_prompt,
        tracker,
        intent_plan=stages.get("intent_plan"),
    )
    if not candidates:
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": "generation_failed",
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
        }

    confidence = (
        float(candidate_pack.Top20候选[0]["相关性分数"])
        if candidate_pack.Top20候选
        else 0.0
    )
    prepared = []
    for cand in candidates:
        cand["confidence"] = confidence
        prepared.append(_prepare_candidate_for_vote(system, cand))
    selected, reason, status = select_by_consensus(prepared)
    if selected is None:
        trace_fields = _pipeline_trace_fields(candidates, prepared, None)
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": reason,
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
            **trace_fields,
        }

    result = selected.get("result")
    unique_rows: list[tuple] = []
    if result:
        seen_rows = set()
        for row in result:
            tup = tuple(row)
            if tup not in seen_rows:
                seen_rows.add(tup)
                unique_rows.append(tup)
    trace_fields = _pipeline_trace_fields(candidates, prepared, selected)
    return {
        "final_sql": selected.get("sql"),
        "execution_result": unique_rows if result else result,
        "reason": status,
        "cost_time": time.time() - start,
        "token_usage": tracker.get_report(),
        "stages": stages,
        **trace_fields,
    }


async def _generate_single_path_cand(
    system,
    question: str,
    schema_prompt: str,
    tracker: TokenTracker,
    path_type: str,
    *,
    intent_plan: dict | None = None,
) -> dict | None:
    gen = system.generator
    fewshot_context = gen._build_fewshot_context(question)
    fewshot_block = f"[Few-shot示例]\n{fewshot_context}\n\n" if fewshot_context else ""
    intent_block = system.intent_planner.to_prompt_block(intent_plan)
    intent_section = f"[弱意图解析]\n{intent_block}\n\n" if intent_block else ""
    highlighted_question = gen.highlight_question_for_prompt(question)
    base_prompt = (
        "你是一名SQL专家。请只基于给定的 Schema 生成一条 SQLite SQL。\n\n"
        + fewshot_block
        + (intent_section if path_type == "intent_plan" else "")
        + f"[Schema]\n{schema_prompt}\n"
        + f"[用户问题]\n{highlighted_question}\n"
        + gen._sql_generation_rules()
    )
    if path_type == "direct":
        prompt = base_prompt + "\n[路径提示] 直接根据 Schema 与加粗问题原文生成最简洁 SQL。"
        temperature = settings.direct_temperature
    elif path_type == "icl":
        prompt = base_prompt + "\n[路径提示] 参考字段语义，过滤条件必须逐字来自加粗问题原文。"
        temperature = settings.icl_temperature
    elif path_type == "intent_plan":
        prompt = base_prompt + "\n[路径提示] 优先参考弱意图解析，但它不是硬约束；如果有冲突，以用户问题原文为准。"
        temperature = settings.direct_temperature
    else:
        raise ValueError(f"unsupported path_type: {path_type}")
    return await gen._call_llm_sql(prompt, tracker, temperature, path_type)


async def _finalize_main_pipeline(
    system,
    *,
    question: str,
    cand: dict | None,
    schema_prompt: str,
    repair_schema_prompt: str,
    cliff_schema_prompt: str,
    candidate_pack,
    tracker: TokenTracker,
    start: float,
    stages: dict,
    skip_refiner: bool = False,
    skip_judge: bool = False,
) -> dict:
    if cand is None:
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": "generation_failed",
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
            "generated_candidates": [],
            "refined_candidates": [],
            "selected_candidate": None,
            "candidate_results": [],
            "candidate_sqls": [],
        }

    generated_candidates = cand if isinstance(cand, list) else [cand]
    confidence = (
        float(candidate_pack.Top20候选[0]["相关性分数"])
        if candidate_pack.Top20候选
        else 0.0
    )
    for item in generated_candidates:
        item["confidence"] = confidence
    if skip_refiner:
        candidates = [_prepare_candidate_for_vote(system, item) for item in generated_candidates]
    else:
        candidates = await system.refiner.refine_async(
            schema_prompt,
            generated_candidates,
            candidate_pack.Top20候选,
            tracker,
            repair_schema_prompt=repair_schema_prompt,
            question=question if not skip_judge else None,
            judge_schema_prompt=cliff_schema_prompt,
            intent_plan=stages.get("intent_plan"),
        )
    selected, reason, status = select_by_consensus(candidates)
    if selected is None:
        trace_fields = _pipeline_trace_fields(generated_candidates, candidates, None)
        return {
            "final_sql": None,
            "execution_result": None,
            "reason": reason,
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "stages": stages,
            **trace_fields,
        }

    result = selected.get("result")
    unique_rows: list[tuple] = []
    if result:
        seen_rows = set()
        for row in result:
            tup = tuple(row)
            if tup not in seen_rows:
                seen_rows.add(tup)
                unique_rows.append(tup)

    trace_fields = _pipeline_trace_fields(generated_candidates, candidates, selected)
    return {
        "final_sql": selected.get("sql"),
        "execution_result": unique_rows if result else result,
        "reason": status,
        "cost_time": time.time() - start,
        "token_usage": tracker.get_report(),
        "stages": stages,
        **trace_fields,
    }


async def _run_main_system_once(system, question: str, *, stages: dict | None = None) -> dict:
    """完整主系统路径：实体提取 + must_have + 生成/修复/consistency judge。"""
    start = time.time()
    tracker = TokenTracker()
    if stages is None:
        stages = await _prepare_single_pipeline_async(system, question, tracker)
    candidate_pack = stages["candidate_pack"]
    schema_prompt = stages["schema_prompt"]
    repair_schema_prompt = stages["repair_schema_prompt"]
    cliff_schema_prompt = stages.get("cliff_schema_prompt") or schema_prompt

    cand = await system.generator.generate_candidates_async(
        stages["question"],
        schema_prompt,
        tracker,
        intent_plan=stages.get("intent_plan"),
    )
    return await _finalize_main_pipeline(
        system,
        question=stages["question"],
        cand=cand,
        schema_prompt=schema_prompt,
        repair_schema_prompt=repair_schema_prompt,
        cliff_schema_prompt=cliff_schema_prompt,
        candidate_pack=candidate_pack,
        tracker=tracker,
        start=start,
        stages=stages,
    )


async def _run_main_with_schema_once(
    system,
    question: str,
    schema_prompt: str,
    recall_schema: list[dict],
    *,
    stages: dict | None = None,
) -> dict:
    """主系统流程，但使用指定的 schema markdown 生成 SQL。"""
    start = time.time()
    tracker = TokenTracker()
    if stages is None:
        stages = await _prepare_single_pipeline_async(system, question, tracker)
    candidate_pack = stages["candidate_pack"]
    repair_schema_prompt = stages["repair_schema_prompt"]
    cliff_schema_prompt = stages.get("cliff_schema_prompt") or schema_prompt
    stages = dict(stages)
    stages["recall_schema"] = recall_schema

    cand = await system.generator.generate_candidates_async(
        stages["question"],
        schema_prompt,
        tracker,
        intent_plan=stages.get("intent_plan"),
    )
    return await _finalize_main_pipeline(
        system,
        question=stages["question"],
        cand=cand,
        schema_prompt=schema_prompt,
        repair_schema_prompt=repair_schema_prompt,
        cliff_schema_prompt=cliff_schema_prompt,
        candidate_pack=candidate_pack,
        tracker=tracker,
        start=start,
        stages=stages,
    )


async def _run_main_single_path_once(
    system,
    question: str,
    path_type: str,
    *,
    stages: dict | None = None,
) -> dict:
    """主系统流程，但生成阶段仅走 direct 或 icl 单路径。"""
    start = time.time()
    tracker = TokenTracker()
    if stages is None:
        stages = await _prepare_single_pipeline_async(system, question, tracker)
    candidate_pack = stages["candidate_pack"]
    schema_prompt = stages["schema_prompt"]
    repair_schema_prompt = stages["repair_schema_prompt"]
    cliff_schema_prompt = stages.get("cliff_schema_prompt") or schema_prompt

    cand = await _generate_single_path_cand(
        system,
        stages["question"],
        schema_prompt,
        tracker,
        path_type,
        intent_plan=stages.get("intent_plan"),
    )
    return await _finalize_main_pipeline(
        system,
        question=stages["question"],
        cand=cand,
        schema_prompt=schema_prompt,
        repair_schema_prompt=repair_schema_prompt,
        cliff_schema_prompt=cliff_schema_prompt,
        candidate_pack=candidate_pack,
        tracker=tracker,
        start=start,
        stages=stages,
    )


async def _run_main_no_judge_once(system, question: str, *, stages: dict | None = None) -> dict:
    """完整主系统，但跳过 LLM consistency judge，保留 refiner + 简单一致性投票。"""
    start = time.time()
    tracker = TokenTracker()
    if stages is None:
        stages = await _prepare_single_pipeline_async(system, question, tracker)
    candidate_pack = stages["candidate_pack"]
    schema_prompt = stages["schema_prompt"]
    repair_schema_prompt = stages["repair_schema_prompt"]
    cliff_schema_prompt = stages.get("cliff_schema_prompt") or schema_prompt

    cand = await system.generator.generate_candidates_async(
        stages["question"],
        schema_prompt,
        tracker,
        intent_plan=stages.get("intent_plan"),
    )
    return await _finalize_main_pipeline(
        system,
        question=stages["question"],
        cand=cand,
        schema_prompt=schema_prompt,
        repair_schema_prompt=repair_schema_prompt,
        cliff_schema_prompt=cliff_schema_prompt,
        candidate_pack=candidate_pack,
        tracker=tracker,
        start=start,
        stages=stages,
        skip_judge=True,
    )


def _normalize_eval_execution_result(result):
    from training.evaluate import normalize_execution_result

    return normalize_execution_result(result)


def _compare_output(raw_gt: Any, output: dict) -> tuple[bool, str]:
    from training.evaluate import _compare_results, normalize_execution_result, parse_ground_truth

    gt_parsed = parse_ground_truth(raw_gt)
    pred_parsed = normalize_execution_result(output.get("execution_result"))
    ok, _score, match_type = _compare_results(gt_parsed, pred_parsed)
    return bool(ok), str(match_type)


def _candidate_match(raw_gt: Any, candidate: dict) -> tuple[bool, str]:
    return _compare_output(raw_gt, {"execution_result": candidate.get("result")})


def _mechanism_metrics(raw_gt: Any, output: dict) -> dict:
    candidates = output.get("_candidate_pool") or []
    if not candidates:
        selected_ok, _match_type = _compare_output(raw_gt, output)
        return {
            "pass_at_k": int(selected_ok),
            "selected_changed_by_refiner": 0,
            "judge_trigger_count": 0,
            "repair_success_count": 0,
            "repair_harm_count": 0,
        }

    candidate_hits = [_candidate_match(raw_gt, item)[0] for item in candidates]
    selected = output.get("_selected_candidate") or {}
    selected_changed = bool(
        selected.get("is_refined")
        and selected.get("source_sql")
        and str(selected.get("source_sql")).strip() != str(selected.get("sql") or "").strip()
    )
    judge_trigger_count = sum(
        1
        for item in candidates
        if not item.get("is_refined") and item.get("judge_status") == "fail"
    )
    repair_success_count = 0
    repair_harm_count = 0
    for idx, item in enumerate(candidates):
        if not item.get("is_refined"):
            continue
        source_idx = item.get("refined_from")
        if not isinstance(source_idx, int) or source_idx < 0 or source_idx >= len(candidates):
            continue
        source_hit = candidate_hits[source_idx]
        repaired_hit = candidate_hits[idx]
        repair_success_count += int(repaired_hit and not source_hit)
        repair_harm_count += int(source_hit and not repaired_hit)

    return {
        "pass_at_k": int(any(candidate_hits)),
        "selected_changed_by_refiner": int(selected_changed),
        "judge_trigger_count": judge_trigger_count,
        "repair_success_count": repair_success_count,
        "repair_harm_count": repair_harm_count,
    }


def _detail_trace_fields(raw_gt: Any, output: dict) -> dict:
    candidates = output.get("_candidate_pool") or []
    candidate_matches = [
        {"candidate_correct": ok, "candidate_match_type": match_type}
        for ok, match_type in (_candidate_match(raw_gt, item) for item in candidates)
    ]

    def _enrich(records: list[dict], *, limit: int | None = None) -> list[dict]:
        enriched = []
        for idx, record in enumerate(records[:limit] if limit is not None else records):
            item = dict(record)
            if idx < len(candidate_matches):
                item.update(candidate_matches[idx])
            enriched.append(item)
        return enriched

    generated = output.get("generated_candidates") or []
    refined = output.get("refined_candidates") or []
    candidate_results = output.get("candidate_results") or []
    selected_record = output.get("selected_candidate")
    selected = output.get("_selected_candidate")
    if isinstance(selected_record, dict):
        selected_record = dict(selected_record)
        if isinstance(selected, dict):
            ok, match_type = _candidate_match(raw_gt, selected)
            selected_record.update(
                {"candidate_correct": ok, "candidate_match_type": match_type}
            )

    mechanism = _mechanism_metrics(raw_gt, output)
    return {
        "generated_candidates": _enrich(
            generated,
            limit=int(output.get("_generated_candidate_count") or len(generated)),
        ),
        "refined_candidates": _enrich(refined),
        "selected_candidate": selected_record,
        "candidate_results": _enrich(candidate_results),
        "candidate_pool_has_correct_sql": bool(mechanism["pass_at_k"]),
        "selected_from_refiner": bool(selected and selected.get("is_refined")),
        **mechanism,
    }


def _build_main_system_detail_record(
    *,
    idx: int,
    question: str,
    raw_gt: Any,
    output: dict,
) -> dict:
    from training.evaluate import parse_ground_truth

    ok, match_type = _compare_output(raw_gt, output)
    stages = output.get("stages") or {}
    trace_fields = _detail_trace_fields(raw_gt, output)
    return {
        "idx": idx,
        "question": question,
        "ground_truth_raw": raw_gt,
        "ground_truth_parsed": parse_ground_truth(raw_gt),
        "final_sql": output.get("final_sql"),
        "final_result": output.get("execution_result"),
        "final_result_parsed": _normalize_eval_execution_result(output.get("execution_result")),
        "correct": ok,
        "match_type": match_type,
        "reason": output.get("reason"),
        "cost_time": output.get("cost_time"),
        "token_usage": output.get("token_usage") or {},
        "candidate_sqls": output.get("candidate_sqls") or [],
        **trace_fields,
        "intent_plan": output.get("intent_plan") or stages.get("intent_plan") or {},
        "plan_schema": stages.get("plan_schema") or [],
        "repair_schema": output.get("repair_schema") or stages.get("final_schema") or [],
    }


def _build_experiment_detail_record(
    *,
    idx: int,
    label: str,
    prefix: str,
    question: str,
    raw_gt: Any,
    output: dict,
) -> dict:
    ok, match_type = _compare_output(raw_gt, output)
    trace_fields = _detail_trace_fields(raw_gt, output)
    return {
        "idx": idx,
        "label": label,
        "prefix": prefix,
        "question": question,
        "ground_truth_raw": raw_gt,
        "final_sql": output.get("final_sql"),
        "final_result": output.get("execution_result"),
        "final_result_parsed": _normalize_eval_execution_result(output.get("execution_result")),
        "correct": ok,
        "match_type": match_type,
        "reason": output.get("reason"),
        "cost_time": output.get("cost_time"),
        "token_usage": output.get("token_usage") or {},
        "candidate_sqls": output.get("candidate_sqls") or [],
        **trace_fields,
    }


async def _run_schema_compare_one(
    *,
    system,
    row: dict,
    idx: int,
    full_schema_prompt: str,
) -> dict:
    question = str(row.get("生成问题") or "").strip()
    raw_gt = row.get("生成结果")
    gold_columns = _gold_columns_from_row(row, system.linker.column_names)
    full_schema_row_list = _full_schema_rows(system)
    metrics: dict[str, Any] = {"idx": idx, "question": question}

    prep_task = asyncio.create_task(_prepare_single_pipeline_async(system, question))
    topk_stages, cliff_stages = await _prepare_shared_retrieval_stages(system, question)
    main_stages = await prep_task

    full_output, topk_output, cliff_output, main_output = await _run_paths(
        [
            lambda: _run_full_schema_once(system, question, full_schema_prompt),
            lambda: _run_topk_schema_once(system, question, stages=topk_stages),
            lambda: _run_cliff_schema_once(system, question, stages=cliff_stages),
            lambda: _run_main_system_once(system, question, stages=main_stages),
        ]
    )

    metrics.update(
        _record_metrics(
            full_output,
            prefix="full_schema",
            raw_gt=raw_gt,
            recall_schema=full_schema_row_list,
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            topk_output,
            prefix="topk_schema",
            raw_gt=raw_gt,
            recall_schema=topk_stages.get("topk_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            cliff_output,
            prefix="cliff_schema",
            raw_gt=raw_gt,
            recall_schema=cliff_stages.get("cliff_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_output,
            prefix="main_system",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics["main_system_schema_recall"] = metrics.pop("main_system_recall")
    return metrics


async def _run_ablation_one(
    *,
    system,
    row: dict,
    idx: int,
    full_schema_prompt: str,
) -> dict:
    question = str(row.get("生成问题") or "").strip()
    raw_gt = row.get("生成结果")
    gold_columns = _gold_columns_from_row(row, system.linker.column_names)
    full_schema_row_list = _full_schema_rows(system)
    metrics: dict[str, Any] = {"idx": idx, "question": question}

    no_entity_task = asyncio.create_task(asyncio.to_thread(_prepare_main_no_entity_stages, system, question))
    main_stages = await _prepare_single_pipeline_async(system, question)
    topk_stages = _stages_from_candidate_pack(
        system,
        question,
        main_stages["candidate_pack"],
        cliff=False,
    )
    no_entity_stages = await no_entity_task

    (
        main_full_output,
        main_output,
        main_topk_output,
        main_direct_output,
        main_icl_output,
        main_intent_plan_output,
        main_no_refiner_output,
        main_no_entity_output,
        main_no_judge_output,
    ) = await _run_paths(
        [
            lambda: _run_main_with_schema_once(
                system,
                question,
                full_schema_prompt,
                full_schema_row_list,
                stages=main_stages,
            ),
            lambda: _run_main_system_once(system, question, stages=main_stages),
            lambda: _run_main_with_schema_once(
                system,
                question,
                topk_stages["schema_prompt"],
                topk_stages["topk_schema"],
                stages=main_stages,
            ),
            lambda: _run_main_single_path_once(system, question, "direct", stages=main_stages),
            lambda: _run_main_single_path_once(system, question, "icl", stages=main_stages),
            lambda: _run_main_single_path_once(system, question, "intent_plan", stages=main_stages),
            lambda: _run_main_no_refiner_once(system, question, stages=main_stages),
            lambda: _run_main_no_entity_once(system, question),
            lambda: _run_main_no_judge_once(system, question, stages=main_stages),
        ]
    )

    metrics.update(
        _record_metrics(
            main_full_output,
            prefix="main_full_schema",
            raw_gt=raw_gt,
            recall_schema=full_schema_row_list,
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_output,
            prefix="main_system",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics["main_system_schema_recall"] = metrics.pop("main_system_recall")
    metrics.update(
        _record_metrics(
            main_topk_output,
            prefix="main_topk_schema",
            raw_gt=raw_gt,
            recall_schema=topk_stages["topk_schema"],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_direct_output,
            prefix="main_direct_only",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_icl_output,
            prefix="main_icl_only",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_intent_plan_output,
            prefix="main_intent_plan_only",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics.update(
        _record_metrics(
            main_no_refiner_output,
            prefix="main_no_refiner",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics["main_no_refiner_schema_recall"] = metrics.pop("main_no_refiner_recall")
    metrics.update(
        _record_metrics(
            main_no_entity_output,
            prefix="main_no_entity",
            raw_gt=raw_gt,
            recall_schema=no_entity_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    metrics["main_no_entity_schema_recall"] = metrics.pop("main_no_entity_recall")
    metrics.update(
        _record_metrics(
            main_no_judge_output,
            prefix="main_no_judge",
            raw_gt=raw_gt,
            recall_schema=main_stages.get("plan_schema") or [],
            gold_columns=gold_columns,
        )
    )
    return metrics


def _normalize_recall_key(metrics: dict, prefix: str, recall_key: str) -> None:
    raw_key = f"{prefix}_recall"
    if recall_key != raw_key and raw_key in metrics:
        metrics[recall_key] = metrics.pop(raw_key)


async def _run_schema_compare_config_one(
    *,
    system,
    row: dict,
    idx: int,
    full_schema_prompt: str,
    config: tuple[str, str, str],
) -> dict:
    _label, prefix, recall_key = config
    question = str(row.get("生成问题") or "").strip()
    raw_gt = row.get("生成结果")
    gold_columns = _gold_columns_from_row(row, system.linker.column_names)
    full_schema_row_list = _full_schema_rows(system)
    metrics: dict[str, Any] = {"idx": idx, "question": question}

    if prefix == "full_schema":
        output = await _run_full_schema_once(system, question, full_schema_prompt)
        recall_schema = full_schema_row_list
    elif prefix == "topk_schema":
        topk_stages, _cliff_stages = await _prepare_shared_retrieval_stages(system, question)
        output = await _run_topk_schema_once(system, question, stages=topk_stages)
        recall_schema = topk_stages.get("topk_schema") or []
    elif prefix == "cliff_schema":
        _topk_stages, cliff_stages = await _prepare_shared_retrieval_stages(system, question)
        output = await _run_cliff_schema_once(system, question, stages=cliff_stages)
        recall_schema = cliff_stages.get("cliff_schema") or []
    elif prefix == "main_system":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_system_once(system, question, stages=stages)
        recall_schema = stages.get("plan_schema") or []
    else:
        raise ValueError(f"unsupported schema compare config: {prefix}")

    metrics.update(
        _record_metrics(
            output,
            prefix=prefix,
            raw_gt=raw_gt,
            recall_schema=recall_schema,
            gold_columns=gold_columns,
        )
    )
    if prefix == "main_system":
        detail = _build_main_system_detail_record(
            idx=idx,
            question=question,
            raw_gt=raw_gt,
            output=output,
        )
        if not detail.get("correct"):
            metrics["__detail__"] = detail
    _normalize_recall_key(metrics, prefix, recall_key)
    return metrics


async def _run_ablation_config_one(
    *,
    system,
    row: dict,
    idx: int,
    full_schema_prompt: str,
    config: tuple[str, str, str],
) -> dict:
    label, prefix, recall_key = config
    question = str(row.get("生成问题") or "").strip()
    raw_gt = row.get("生成结果")
    gold_columns = _gold_columns_from_row(row, system.linker.column_names)
    full_schema_row_list = _full_schema_rows(system)
    metrics: dict[str, Any] = {"idx": idx, "question": question}

    if prefix == "main_full_schema":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_with_schema_once(
            system,
            question,
            full_schema_prompt,
            full_schema_row_list,
            stages=stages,
        )
        recall_schema = full_schema_row_list
    elif prefix == "main_system":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_system_once(system, question, stages=stages)
        recall_schema = stages.get("plan_schema") or []
    elif prefix == "main_topk_schema":
        stages = await _prepare_single_pipeline_async(system, question)
        topk_stages = _stages_from_candidate_pack(
            system,
            question,
            stages["candidate_pack"],
            cliff=False,
        )
        output = await _run_main_with_schema_once(
            system,
            question,
            topk_stages["schema_prompt"],
            topk_stages["topk_schema"],
            stages=stages,
        )
        recall_schema = topk_stages.get("topk_schema") or []
    elif prefix == "main_direct_only":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_single_path_once(system, question, "direct", stages=stages)
        recall_schema = stages.get("plan_schema") or []
    elif prefix == "main_icl_only":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_single_path_once(system, question, "icl", stages=stages)
        recall_schema = stages.get("plan_schema") or []
    elif prefix == "main_intent_plan_only":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_single_path_once(system, question, "intent_plan", stages=stages)
        recall_schema = stages.get("plan_schema") or []
    elif prefix == "main_no_refiner":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_no_refiner_once(system, question, stages=stages)
        recall_schema = stages.get("plan_schema") or []
    elif prefix == "main_no_entity":
        output = await _run_main_no_entity_once(system, question)
        recall_schema = (output.get("stages") or {}).get("plan_schema") or []
    elif prefix == "main_no_judge":
        stages = await _prepare_single_pipeline_async(system, question)
        output = await _run_main_no_judge_once(system, question, stages=stages)
        recall_schema = stages.get("plan_schema") or []
    else:
        raise ValueError(f"unsupported ablation config: {prefix}")

    metrics.update(
        _record_metrics(
            output,
            prefix=prefix,
            raw_gt=raw_gt,
            recall_schema=recall_schema,
            gold_columns=gold_columns,
        )
    )
    metrics["__detail__"] = _build_experiment_detail_record(
        idx=idx,
        label=label,
        prefix=prefix,
        question=question,
        raw_gt=raw_gt,
        output=output,
    )
    _normalize_recall_key(metrics, prefix, recall_key)
    return metrics


async def _run_parallel_eval(
    *,
    system,
    eval_rows: list[dict],
    full_schema_prompt: str,
    runner,
    label: str,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[dict | None] = [None] * len(eval_rows)

    async def _guarded_run(pos: int, row: dict) -> None:
        async with semaphore:
            idx = pos + 1
            try:
                results[pos] = await runner(
                    system=system,
                    row=row,
                    idx=idx,
                    full_schema_prompt=full_schema_prompt,
                )
                print(f"[{label}][{idx}/{len(eval_rows)}] done")
            except Exception as exc:
                print(f"[{label}][{idx}/{len(eval_rows)}] error: {type(exc).__name__}: {exc}")
                results[pos] = {"idx": idx, "question": str(row.get("生成问题") or "")}

    await asyncio.gather(*[_guarded_run(i, row) for i, row in enumerate(eval_rows)])
    return [r or {} for r in results]


async def _run_parallel_config_eval(
    *,
    system,
    eval_rows: list[dict],
    full_schema_prompt: str,
    runner,
    label: str,
    config: tuple[str, str, str],
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[dict | None] = [None] * len(eval_rows)
    _experiment_label, prefix, _recall_key = config

    async def _guarded_run(pos: int, row: dict) -> None:
        async with semaphore:
            idx = pos + 1
            try:
                results[pos] = await runner(
                    system=system,
                    row=row,
                    idx=idx,
                    full_schema_prompt=full_schema_prompt,
                    config=config,
                )
                print(f"[{label}:{prefix}][{idx}/{len(eval_rows)}] done")
            except Exception as exc:
                print(f"[{label}:{prefix}][{idx}/{len(eval_rows)}] error: {type(exc).__name__}: {exc}")
                results[pos] = {"idx": idx, "question": str(row.get("生成问题") or "")}

    await asyncio.gather(*[_guarded_run(i, row) for i, row in enumerate(eval_rows)])
    return [r or {} for r in results]


async def _run_config_group_with_resume(
    *,
    system,
    eval_rows: list[dict],
    full_schema_prompt: str,
    configs: list[tuple[str, str, str]],
    output_path: Path,
    detail_output_path: Path | None,
    extra_correct_path: Path | None,
    runner,
    label: str,
    concurrency: int,
) -> list[dict]:
    _print_resume_state(label, output_path, configs)
    completed = _completed_labels(output_path)
    extra_records = _load_extra_correct_records(extra_correct_path)

    for config in configs:
        experiment_label, prefix, _recall_key = config
        if experiment_label in completed:
            print(f"[Resume][{label}] skip completed: {experiment_label}")
            continue

        rows = await _run_parallel_config_eval(
            system=system,
            eval_rows=eval_rows,
            full_schema_prompt=full_schema_prompt,
            runner=runner,
            label=label,
            config=config,
            concurrency=concurrency,
        )
        detail_rows = [row["__detail__"] for row in rows if isinstance(row, dict) and "__detail__" in row]
        if detail_output_path and detail_rows:
            _append_jsonl_rows(detail_output_path, detail_rows)
        summary_row = _build_summary_rows(rows, [config])[0]
        _append_csv_row(output_path, summary_row, SUMMARY_FIELDNAMES)
        extra_records[prefix] = _correct_idx_record(
            label=experiment_label,
            prefix=prefix,
            rows=rows,
        )
        if extra_correct_path:
            _write_extra_correct_report(
                extra_correct_path,
                group_label=label,
                configs=configs,
                records=extra_records,
                total_questions=len(eval_rows),
            )
        completed.add(experiment_label)
        print(f"[Resume][{label}] saved: {experiment_label} ({prefix}) -> {output_path}")

    if extra_correct_path:
        _write_extra_correct_report(
            extra_correct_path,
            group_label=label,
            configs=configs,
            records=extra_records,
            total_questions=len(eval_rows),
        )
    return _read_csv_rows(output_path)


async def main_async(args: argparse.Namespace) -> None:
    from pipeline.system import TextToSQLSystem

    mode = str(args.mode or "both").strip().lower()
    if mode not in {"1", "2", "both"}:
        raise ValueError(f"unsupported --mode: {args.mode}")

    eval_rows = _load_eval_df(args.eval_path or None)
    if args.limit > 0:
        eval_rows = eval_rows[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_compare_path = (
        Path(args.schema_compare_output)
        if args.schema_compare_output
        else out_dir / "schema_compare_summary.csv"
    )
    ablation_path = (
        Path(args.ablation_output)
        if args.ablation_output
        else out_dir / "ablation_summary.csv"
    )
    main_system_detail_path = (
        Path(args.main_system_detail_output)
        if args.main_system_detail_output
        else out_dir / "main_system_outputs.jsonl"
    )
    ablation_detail_path = (
        Path(args.ablation_detail_output)
        if args.ablation_detail_output
        else _default_ablation_detail_path(out_dir)
    )
    schema_extra_correct_path = out_dir / "schema_compare_extra_correct.json"
    ablation_extra_correct_path = out_dir / "ablation_extra_correct.json"
    if args.overwrite:
        if mode in {"1", "both"} and schema_compare_path.exists():
            schema_compare_path.unlink()
            print(f"[Overwrite] removed: {schema_compare_path}")
        if mode in {"2", "both"} and ablation_path.exists():
            ablation_path.unlink()
            print(f"[Overwrite] removed: {ablation_path}")
        if mode in {"1", "both"} and main_system_detail_path.exists():
            main_system_detail_path.unlink()
            print(f"[Overwrite] removed: {main_system_detail_path}")
        if mode in {"1", "both"} and schema_extra_correct_path.exists():
            schema_extra_correct_path.unlink()
            print(f"[Overwrite] removed: {schema_extra_correct_path}")
        if mode in {"2", "both"} and ablation_extra_correct_path.exists():
            ablation_extra_correct_path.unlink()
            print(f"[Overwrite] removed: {ablation_extra_correct_path}")
        if mode in {"2", "both"} and ablation_detail_path.exists():
            ablation_detail_path.unlink()
            print(f"[Overwrite] removed: {ablation_detail_path}")

    global _PATH_PARALLEL
    _PATH_PARALLEL = not bool(args.no_path_parallel)

    start = time.time()
    system = TextToSQLSystem()
    _install_eval_runtime(system, llm_concurrency=int(args.llm_concurrency))
    full_prompt = _full_schema_prompt(system)
    concurrency = max(1, int(args.concurrency))
    report: dict[str, Any] = {
        "mode": mode,
        "total_questions": len(eval_rows),
        "concurrency": concurrency,
        "path_parallel": _PATH_PARALLEL,
        "llm_concurrency": int(args.llm_concurrency),
        "elapsed_seconds": 0.0,
    }

    if mode in {"1", "both"}:
        schema_summary = await _run_config_group_with_resume(
            system=system,
            eval_rows=eval_rows,
            full_schema_prompt=full_prompt,
            configs=SCHEMA_COMPARE_CONFIGS,
            output_path=schema_compare_path,
            detail_output_path=main_system_detail_path,
            extra_correct_path=schema_extra_correct_path,
            runner=_run_schema_compare_config_one,
            label="SchemaCompare",
            concurrency=concurrency,
        )
        report["schema_compare_summary"] = schema_summary
        report["schema_compare_csv"] = str(schema_compare_path)
        report["schema_compare_extra_correct_json"] = str(schema_extra_correct_path)
        if main_system_detail_path.exists():
            report["main_system_detail_jsonl"] = str(main_system_detail_path)
        print(f"schema compare summary CSV: {schema_compare_path}")
        print(f"schema compare extra-correct JSON: {schema_extra_correct_path}")

    if mode in {"2", "both"}:
        ablation_summary = await _run_config_group_with_resume(
            system=system,
            eval_rows=eval_rows,
            full_schema_prompt=full_prompt,
            configs=ABLATION_CONFIGS,
            output_path=ablation_path,
            detail_output_path=ablation_detail_path,
            extra_correct_path=ablation_extra_correct_path,
            runner=_run_ablation_config_one,
            label="Ablation",
            concurrency=concurrency,
        )
        report["ablation_summary"] = ablation_summary
        report["ablation_csv"] = str(ablation_path)
        report["ablation_detail_jsonl"] = str(ablation_detail_path)
        report["ablation_extra_correct_json"] = str(ablation_extra_correct_path)
        print(f"ablation summary CSV: {ablation_path}")
        print(f"ablation detail JSONL: {ablation_detail_path}")
        print(f"ablation extra-correct JSON: {ablation_extra_correct_path}")

    report["elapsed_seconds"] = round(time.time() - start, 6)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "并行评测 test split，仅输出汇总 CSV。\n"
            "模式1：不同 schema 策略（全量 / topk / 断崖 / 主系统）。\n"
            "模式2：主系统消融（全量schema、topk schema、direct/icl 单路径、无 refiner 等）。"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["1", "2", "both"],
        default="both",
        help="运行模式：1=schema对比，2=消融实验，both=两者都跑（默认）",
    )
    parser.add_argument(
        "--eval-path",
        default="",
        help="评测数据路径，默认 settings.test_split_jsonl 或 data/test_split.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="输出目录，默认 tests/results",
    )
    parser.add_argument(
        "--schema-compare-output",
        default="",
        help="模式1汇总 CSV 路径，默认 tests/results/schema_compare_summary.csv",
    )
    parser.add_argument(
        "--ablation-output",
        default="",
        help="模式2汇总 CSV 路径，默认 tests/results/ablation_summary.csv",
    )
    parser.add_argument(
        "--main-system-detail-output",
        default="",
        help="主系统逐题 JSONL 日志路径，默认 tests/results/main_system_outputs.jsonl",
    )
    parser.add_argument(
        "--ablation-detail-output",
        default="",
        help="模式2逐题 JSONL 日志路径，默认 tests/results/ablation_outputs.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有汇总 CSV；默认续写并跳过已完成的对比实验",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条，0 表示全部")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="并行处理的题目数，默认 8（GPU KV 利用率低时可继续加大）",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=16,
        help="全局同时在飞的 LLM 请求上限，默认 16；设为 0 表示不限制",
    )
    parser.add_argument(
        "--no-path-parallel",
        action="store_true",
        help="关闭单题内多路径并行（默认开启，与题目级并行叠加可提高 GPU 吞吐）",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
