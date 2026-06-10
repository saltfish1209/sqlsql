"""
Analyze whether min_ratio adds value beyond the score-cliff selector.

The important baseline is min_ratio=0: that is "cliff only".  When top_k
changes, this script intentionally sweeps the same min_ratio grid for every K
instead of scaling min_ratio by K.  min_ratio is relative to the best score, so
K only decides how much tail is exposed to the selector; the experiment should
show whether a non-zero floor is useful for each K.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings


TIER_ORDER = [
    "gold",
    "semantic_hard_negative",
    "string_hard_negative",
    "random_negative",
    "other",
]
TIER_PRIORITY = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
STRATEGIES = ("topk_only", "min_ratio_only", "cliff_only", "min_ratio_plus_cliff")


@dataclass(frozen=True)
class FieldScore:
    column: str
    score: float
    tier: str = "other"


def _round(value: float) -> float:
    return round(float(value), 6)


def _parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise ValueError("top-k list must contain positive integers")
    return sorted(dict.fromkeys(values))


def _parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values or any(x < 0.0 or x > 1.0 for x in values):
        raise ValueError("min-ratio list must contain values in [0, 1]")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(dict.fromkeys(values))


def _sorted_ranked(ranked: Iterable[FieldScore]) -> list[FieldScore]:
    return sorted(ranked, key=lambda x: (-float(x.score), x.column))


def _filter_by_min_ratio(items: list[FieldScore], min_ratio: float) -> list[FieldScore]:
    if not items:
        return []
    best = float(items[0].score)
    if best <= 0:
        return list(items)
    threshold = best * float(min_ratio)
    filtered = [item for item in items if float(item.score) >= threshold]
    return filtered or list(items)


def _apply_cliff(
    items: list[FieldScore],
    *,
    protect_ratio: float,
) -> list[FieldScore]:
    if not items:
        return []
    best = float(items[0].score)
    if best <= 0:
        return list(items)

    protect_score = best * float(protect_ratio)
    cutoff = len(items)
    for idx in range(len(items) - 1):
        cur = float(items[idx].score)
        nxt = float(items[idx + 1].score)
        if cur <= 0:
            continue
        if cur >= protect_score and nxt >= protect_score:
            continue
        cutoff = idx + 1
        break

    selected = items[:cutoff]
    return selected or list(items)


def _boundary_type(top_items: list[FieldScore], selected: list[FieldScore]) -> str:
    if len(selected) >= len(top_items):
        return "none"
    if not selected:
        return f"none->{top_items[0].tier}"
    return f"{selected[-1].tier}->{top_items[len(selected)].tier}"


def _tier_counts(items: Iterable[FieldScore]) -> dict[str, int]:
    counts = Counter(item.tier if item.tier in TIER_PRIORITY else "other" for item in items)
    return {tier: int(counts.get(tier, 0)) for tier in TIER_ORDER if counts.get(tier, 0)}


def apply_selection_strategy(
    ranked: list[FieldScore],
    *,
    gold: set[str],
    top_k: int,
    min_ratio: float,
    protect_ratio: float,
    strategy: str,
) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    top_items = _sorted_ranked(ranked)[:top_k]
    effective_min_ratio = 0.0 if strategy in {"topk_only", "cliff_only"} else float(min_ratio)

    if strategy == "topk_only":
        selected = list(top_items)
    elif strategy == "min_ratio_only":
        selected = _filter_by_min_ratio(top_items, effective_min_ratio)
    elif strategy == "cliff_only":
        selected = _apply_cliff(top_items, protect_ratio=protect_ratio)
    else:
        min_filtered = _filter_by_min_ratio(top_items, effective_min_ratio)
        selected = _apply_cliff(
            min_filtered,
            protect_ratio=protect_ratio,
        )

    selected_cols = [item.column for item in selected]
    selected_set = set(selected_cols)
    topk_gold = gold & {item.column for item in top_items}
    recall = len(gold & selected_set) / len(gold) if gold else 0.0

    return {
        "strategy": strategy,
        "top_k": int(top_k),
        "min_ratio": _round(effective_min_ratio),
        "selected_columns": selected_cols,
        "selected_count": len(selected_cols),
        "recall": _round(recall),
        "full_recall": int(bool(gold) and gold.issubset(selected_set)),
        "gold_dropped_from_topk": sorted(topk_gold - selected_set),
        "tier_totals_topk": _tier_counts(top_items),
        "tier_totals_selected": _tier_counts(selected),
        "boundary_type": _boundary_type(top_items, selected),
    }


def aggregate_sweep_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["strategy"], int(row["top_k"]), float(row["min_ratio"]))
        grouped[key].append(row)

    summaries: list[dict] = []
    for key in sorted(grouped.keys(), key=lambda x: (x[1], x[0], x[2])):
        strategy, top_k, min_ratio = key
        group = grouped[key]
        samples = len(group)
        topk_tier_totals = Counter()
        selected_tier_totals = Counter()
        boundary_counts = Counter()
        gold_drop_cases = 0

        for row in group:
            topk_tier_totals.update(row.get("tier_totals_topk") or {})
            selected_tier_totals.update(row.get("tier_totals_selected") or {})
            boundary_counts.update([row.get("boundary_type") or "none"])
            if row.get("gold_dropped_from_topk"):
                gold_drop_cases += 1

        tier_retention = {}
        for tier in TIER_ORDER:
            denom = int(topk_tier_totals.get(tier, 0))
            numer = int(selected_tier_totals.get(tier, 0))
            tier_retention[tier] = _round(numer / denom) if denom else 0.0

        summaries.append(
            {
                "strategy": strategy,
                "top_k": top_k,
                "min_ratio": _round(min_ratio),
                "samples": samples,
                "avg_selected": _round(sum(float(r["selected_count"]) for r in group) / samples),
                "avg_recall": _round(sum(float(r["recall"]) for r in group) / samples),
                "full_recall": _round(sum(int(r["full_recall"]) for r in group) / samples),
                "gold_drop_cases": gold_drop_cases,
                "tier_retention": tier_retention,
                "boundary_counts": dict(sorted(boundary_counts.items())),
            }
        )
    return summaries


def build_tier_map(
    gold: set[str],
    all_cols: list[str],
    *,
    seed: int | None = None,
) -> dict[str, str]:
    tier_map = {col: "gold" for col in gold}
    if seed is not None:
        state = random.getstate()
        random.seed(seed)
    else:
        state = None

    try:
        from training.prepare_data import (
            NUM_EASY_NEGATIVES,
            NUM_HARD_NEGATIVES_CHAR,
            NUM_HARD_NEGATIVES_SEMANTIC,
            generate_negatives,
        )

        groups = generate_negatives(
            sorted(gold),
            all_cols,
            num_hard_char=NUM_HARD_NEGATIVES_CHAR,
            num_hard_sem=NUM_HARD_NEGATIVES_SEMANTIC,
            num_easy=NUM_EASY_NEGATIVES,
        )
    finally:
        if state is not None:
            random.setstate(state)

    source_to_tier = {
        "semantic": "semantic_hard_negative",
        "string": "string_hard_negative",
        "random": "random_negative",
    }
    for group in groups:
        for col, source in zip(group.get("negative_columns") or [], group.get("negative_sources") or []):
            if col in gold:
                continue
            tier = source_to_tier.get(str(source), "other")
            existing = tier_map.get(col, "other")
            if TIER_PRIORITY[tier] < TIER_PRIORITY.get(existing, TIER_PRIORITY["other"]):
                tier_map[col] = tier
    return tier_map


def _load_records(split: str) -> list[dict]:
    if split == "train":
        path = Path(__file__).parent / "cross_encoder_train_eval_data.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing train eval data: {path}")
        return list(_read_jsonl(path))
    from training.cross_encoder_eval import load_eval_records

    return load_eval_records(split)


def _read_jsonl(path: str | os.PathLike[str]) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str | os.PathLike[str], records: Iterable[dict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def _write_summary_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "top_k",
        "min_ratio",
        "samples",
        "avg_selected",
        "avg_recall",
        "full_recall",
        "gold_drop_cases",
        "tier_retention",
        "boundary_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            out = dict(row)
            out["tier_retention"] = json.dumps(row["tier_retention"], ensure_ascii=False, sort_keys=True)
            out["boundary_counts"] = json.dumps(row["boundary_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)


def _recommend_by_topk(summaries: list[dict], *, recall_tolerance: float) -> list[dict]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in summaries:
        by_k[int(row["top_k"])].append(row)

    recommendations: list[dict] = []
    for top_k, rows in sorted(by_k.items()):
        baselines = [r for r in rows if r["strategy"] == "topk_only"]
        if not baselines:
            continue
        baseline = baselines[0]
        min_full = max(0.0, float(baseline["full_recall"]) - recall_tolerance)
        min_avg = max(0.0, float(baseline["avg_recall"]) - recall_tolerance)
        candidates = [
            r
            for r in rows
            if r["strategy"] in {"cliff_only", "min_ratio_plus_cliff"}
            and float(r["full_recall"]) >= min_full
            and float(r["avg_recall"]) >= min_avg
        ]
        if not candidates:
            recommendations.append(
                {
                    "top_k": top_k,
                    "recommendation": "keep_topk_or_relax_thresholds",
                    "reason": "no cliff/min_ratio setting stayed within recall tolerance",
                }
            )
            continue

        best = min(candidates, key=lambda r: (float(r["avg_selected"]), int(r["gold_drop_cases"]), float(r["min_ratio"])))
        cliff_only = [
            r
            for r in candidates
            if r["strategy"] == "cliff_only" and float(r["min_ratio"]) == 0.0
        ]
        cliff_only_best = cliff_only[0] if cliff_only else None
        if (
            best["strategy"] == "min_ratio_plus_cliff"
            and cliff_only_best is not None
            and float(cliff_only_best["avg_selected"]) - float(best["avg_selected"]) < 0.5
        ):
            recommendation = "prefer_cliff_only_min_ratio_0"
            reason = "non-zero min_ratio did not reduce at least 0.5 columns on average versus cliff-only"
        else:
            recommendation = best["strategy"]
            reason = "smallest average selected schema within recall tolerance"

        recommendations.append(
            {
                "top_k": top_k,
                "recommendation": recommendation,
                "strategy": best["strategy"],
                "min_ratio": best["min_ratio"],
                "avg_selected": best["avg_selected"],
                "avg_recall": best["avg_recall"],
                "full_recall": best["full_recall"],
                "gold_drop_cases": best["gold_drop_cases"],
                "reason": reason,
            }
        )
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze min_ratio vs cliff-only field filtering")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--top-k-list", default="8,10,15,20,30,50")
    parser.add_argument("--min-ratios", default="0,0.01,0.03,0.05,0.08,0.10,0.15,0.20")
    parser.add_argument("--protect-ratio", type=float, default=float(settings.candidate_cliff_protect_ratio))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-tolerance", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "min_ratio_analysis"))
    args = parser.parse_args()

    top_k_list = _parse_int_list(args.top_k_list)
    min_ratios = _parse_float_list(args.min_ratios)
    model_path = str(args.model_path).strip()
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    if not os.path.isfile(settings.csv_path):
        raise FileNotFoundError(f"CSV data file does not exist: {settings.csv_path}")

    from sentence_transformers import CrossEncoder
    from pipeline.cross_encoder_passage import build_column_passage_map

    print(
        f"[INFO] split={args.split} top_k={top_k_list} min_ratios={min_ratios} "
        f"protect={args.protect_ratio}"
    )
    model = CrossEncoder(model_path, trust_remote_code=True)
    records = _load_records(args.split)
    passage_map = build_column_passage_map(str(settings.schema_path), str(settings.csv_path))
    all_cols = list(passage_map.keys())

    sweep_rows: list[dict] = []
    detail_rows: list[dict] = []

    for idx, item in enumerate(records, start=1):
        question = str(item.get("question", "")).strip()
        gold = {str(c).strip() for c in (item.get("gold_columns") or []) if str(c).strip()}
        if not question or not gold:
            continue

        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=args.batch_size)
        tier_map = build_tier_map(gold, all_cols, seed=args.seed + idx)
        ranked = [
            FieldScore(column=col, score=float(scores[pos]), tier=tier_map.get(col, "other"))
            for pos, col in enumerate(all_cols)
        ]
        ranked_sorted = _sorted_ranked(ranked)

        case_detail = {
            "idx": idx,
            "question": question,
            "gold_columns": sorted(gold),
            "ranked_columns": [
                {
                    "column": item.column,
                    "score": _round(item.score),
                    "relative_score": _round(item.score / ranked_sorted[0].score) if ranked_sorted and ranked_sorted[0].score > 0 else 0.0,
                    "tier": item.tier,
                }
                for item in ranked_sorted[: max(top_k_list)]
            ],
            "selections": [],
        }

        for top_k in top_k_list:
            for min_ratio in min_ratios:
                for strategy in STRATEGIES:
                    if strategy in {"topk_only", "cliff_only"} and min_ratio != 0.0:
                        continue
                    row = apply_selection_strategy(
                        ranked,
                        gold=gold,
                        top_k=top_k,
                        min_ratio=min_ratio,
                        protect_ratio=args.protect_ratio,
                        strategy=strategy,
                    )
                    row["idx"] = idx
                    row["question"] = question
                    sweep_rows.append(row)
                    case_detail["selections"].append(
                        {
                            "strategy": row["strategy"],
                            "top_k": row["top_k"],
                            "min_ratio": row["min_ratio"],
                            "selected_count": row["selected_count"],
                            "recall": row["recall"],
                            "full_recall": row["full_recall"],
                            "gold_dropped_from_topk": row["gold_dropped_from_topk"],
                            "boundary_type": row["boundary_type"],
                        }
                    )
        detail_rows.append(case_detail)
        print(f"[Case {idx}] gold={len(gold)} scored={len(all_cols)}")

    summaries = aggregate_sweep_rows(sweep_rows)
    recommendations = _recommend_by_topk(summaries, recall_tolerance=float(args.recall_tolerance))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"min_ratio_analysis_{args.split}_summary.json"
    detail_path = out_dir / f"min_ratio_analysis_{args.split}_details.jsonl"
    csv_path = out_dir / f"min_ratio_analysis_{args.split}_summary.csv"
    summary_payload = {
        "params": {
            "split": args.split,
            "top_k_list": top_k_list,
            "min_ratios": min_ratios,
            "protect_ratio": args.protect_ratio,
            "model_path": model_path,
            "recall_tolerance": args.recall_tolerance,
        },
        "summary": summaries,
        "recommendations": recommendations,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(detail_path, detail_rows)
    _write_summary_csv(csv_path, summaries)

    print("=" * 70)
    print(json.dumps({"recommendations": recommendations}, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print(f"details: {detail_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
