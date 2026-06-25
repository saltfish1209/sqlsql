from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from pipeline.utils import strip_invisible, to_halfwidth
from evaluate_llm_full_schema_vs_system_plan import (
    _compare_output,
    _full_schema_prompt,
    _install_eval_runtime,
    _run_full_schema_once,
)


FIELDNAMES = [
    "问题模版",
    "基准原始填充问题",
    "基准生成问题",
    "基准答案",
    "基准是否正确",
    "基准match_type",
    "基准SQL",
    "基准执行结果JSON",
    "基准reason",
    "基准tokens",
    "改写序号",
    "改写生成问题",
    "改写答案",
    "改写是否正确",
    "改写match_type",
    "改写SQL",
    "改写执行结果JSON",
    "改写reason",
    "改写tokens",
    "改写组总数",
    "改写组正确数",
    "改写组错误数",
    "不一致类型",
]


@dataclass
class EvalResult:
    row: dict[str, Any]
    output: dict[str, Any]
    correct: bool
    match_type: str


def _normalize_key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return to_halfwidth(strip_invisible(text))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_template(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _normalize_key(row.get("问题模版"))
        if not key:
            continue
        groups.setdefault(key, []).append(row)
    return groups


def _tokens_total(output: dict[str, Any]) -> int:
    usage = output.get("token_usage") or {}
    return int(usage.get("total_tokens") or 0)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


async def _evaluate_row(
    *,
    system,
    full_schema_prompt: str,
    row: dict[str, Any],
) -> EvalResult:
    question = str(row.get("生成问题") or row.get("原始填充问题") or "").strip()
    answer = row.get("生成结果") or row.get("标准答案") or ""
    output = await _run_full_schema_once(system, question, full_schema_prompt)
    correct, match_type = _compare_output(answer, output)
    return EvalResult(row=row, output=output, correct=correct, match_type=match_type)


async def _evaluate_many(
    *,
    system,
    full_schema_prompt: str,
    jobs: list[tuple[str, int, dict[str, Any]]],
    concurrency: int,
) -> dict[tuple[str, int], EvalResult]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    results: dict[tuple[str, int], EvalResult] = {}

    async def _run_one(template_key: str, pos: int, row: dict[str, Any]) -> None:
        async with semaphore:
            results[(template_key, pos)] = await _evaluate_row(
                system=system,
                full_schema_prompt=full_schema_prompt,
                row=row,
            )
            print(f"[Eval][{len(results)}/{len(jobs)}] {row.get('生成问题') or row.get('原始填充问题')}")

    await asyncio.gather(*[_run_one(key, pos, row) for key, pos, row in jobs])
    return results


def _build_inconsistent_rows(
    *,
    template_key_order: list[str],
    template_only_by_key: dict[str, list[dict[str, Any]]],
    rewrite_by_key: dict[str, list[dict[str, Any]]],
    eval_results: dict[tuple[str, int], EvalResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template_key in template_key_order:
        base_result = eval_results.get((template_key, -1))
        rewrite_rows = rewrite_by_key.get(template_key) or []
        rewrite_results = [
            eval_results[(template_key, i)]
            for i in range(len(rewrite_rows))
            if (template_key, i) in eval_results
        ]
        if not base_result or not rewrite_results:
            continue

        correct_count = sum(1 for item in rewrite_results if item.correct)
        wrong_count = len(rewrite_results) - correct_count
        for i, rewrite_result in enumerate(rewrite_results, start=1):
            if rewrite_result.correct == base_result.correct:
                continue
            inconsistent_type = (
                "基准正确但改写错误"
                if base_result.correct
                else "基准错误但改写正确"
            )
            base_output = base_result.output
            rewrite_output = rewrite_result.output
            rows.append(
                {
                    "问题模版": template_only_by_key[template_key][0].get("问题模版") or rewrite_result.row.get("问题模版"),
                    "基准原始填充问题": base_result.row.get("原始填充问题", ""),
                    "基准生成问题": base_result.row.get("生成问题", ""),
                    "基准答案": base_result.row.get("生成结果") or base_result.row.get("标准答案") or "",
                    "基准是否正确": int(base_result.correct),
                    "基准match_type": base_result.match_type,
                    "基准SQL": base_output.get("final_sql") or "",
                    "基准执行结果JSON": _json_dumps(base_output.get("execution_result")),
                    "基准reason": base_output.get("reason") or "",
                    "基准tokens": _tokens_total(base_output),
                    "改写序号": i,
                    "改写生成问题": rewrite_result.row.get("生成问题", ""),
                    "改写答案": rewrite_result.row.get("生成结果") or rewrite_result.row.get("标准答案") or "",
                    "改写是否正确": int(rewrite_result.correct),
                    "改写match_type": rewrite_result.match_type,
                    "改写SQL": rewrite_output.get("final_sql") or "",
                    "改写执行结果JSON": _json_dumps(rewrite_output.get("execution_result")),
                    "改写reason": rewrite_output.get("reason") or "",
                    "改写tokens": _tokens_total(rewrite_output),
                    "改写组总数": len(rewrite_results),
                    "改写组正确数": correct_count,
                    "改写组错误数": wrong_count,
                    "不一致类型": inconsistent_type,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def main_async(args: argparse.Namespace) -> None:
    from pipeline.system import TextToSQLSystem

    template_only_path = Path(args.template_only_csv)
    rewrite_test_path = Path(args.rewrite_test_jsonl)
    output_path = Path(args.output)
    if not template_only_path.is_file():
        raise FileNotFoundError(f"template-only CSV 不存在: {template_only_path}")
    if not rewrite_test_path.is_file():
        raise FileNotFoundError(f"改写测试集 JSONL 不存在: {rewrite_test_path}")

    template_only_rows = _read_csv_rows(template_only_path)
    rewrite_test_rows = _read_jsonl_rows(rewrite_test_path)
    template_only_by_key = _group_by_template(template_only_rows)
    rewrite_by_key = _group_by_template(rewrite_test_rows)
    template_key_order = list(rewrite_by_key.keys())
    if args.limit_templates > 0:
        template_key_order = template_key_order[: args.limit_templates]

    missing = [key for key in template_key_order if key not in template_only_by_key]
    if missing:
        raise ValueError(f"有 {len(missing)} 个测试模板在 template-only 数据中找不到，示例: {missing[:3]}")

    jobs: list[tuple[str, int, dict[str, Any]]] = []
    for template_key in template_key_order:
        jobs.append((template_key, -1, template_only_by_key[template_key][0]))
        for pos, row in enumerate(rewrite_by_key[template_key]):
            jobs.append((template_key, pos, row))

    system = TextToSQLSystem()
    _install_eval_runtime(system, llm_concurrency=int(args.llm_concurrency))
    full_schema_prompt = _full_schema_prompt(system)
    eval_results = await _evaluate_many(
        system=system,
        full_schema_prompt=full_schema_prompt,
        jobs=jobs,
        concurrency=int(args.concurrency),
    )
    inconsistent_rows = _build_inconsistent_rows(
        template_key_order=template_key_order,
        template_only_by_key=template_only_by_key,
        rewrite_by_key=rewrite_by_key,
        eval_results=eval_results,
    )
    _write_csv(output_path, inconsistent_rows)

    print("=" * 60)
    print("template rewrite effectiveness check done")
    print(f"templates: {len(template_key_order)}")
    print(f"eval questions: {len(jobs)}")
    print(f"inconsistent rows: {len(inconsistent_rows)}")
    print(f"csv: {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用全量 schema SQL 生成结果检查 template-only 基准与语义改写问题是否一致。"
    )
    parser.add_argument(
        "--template-only-csv",
        default=str(settings.qa_template_csv),
        help="纯模板填充数据 CSV，默认 data/train_dataset_template_only.csv",
    )
    parser.add_argument(
        "--rewrite-test-jsonl",
        default=str(settings.test_split_jsonl),
        help="语义改写测试集 JSONL，默认 data/test_split.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "result" / "template_is_useful_or_not.csv"),
        help="不一致结果 CSV，默认 tests/result/template_is_useful_or_not.csv",
    )
    parser.add_argument(
        "--limit-templates",
        type=int,
        default=0,
        help="只评测前 N 个测试模板，0 表示全部",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="题目级并发数，默认 4",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=8,
        help="全局 LLM 请求并发上限，默认 8；设为 0 表示不限制",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
