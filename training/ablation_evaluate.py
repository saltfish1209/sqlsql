"""
主流程模块消融实验入口。

示例：
    # 跑内置核心消融配置
    python training/ablation_evaluate.py --eval-csv data/train_dataset_template_only.csv --use-full-data --concurrency 6

    # 自主调用：精简 schema + 实体提取 + A/B/C 路 + direct 单路径 + 不启用 refiner
    python training/ablation_evaluate.py --single --schema-mode pruned --use-entities --schema-routes ABC --paths direct --no-refiner

模块含义：
    schema-mode:
      full   = 全量 schema
      pruned = 精简 schema
    use-entities / no-entities:
      是否调用 EntityExtractor，并把实体交给 schema linker 与 generator prompt
    schema-routes:
      A   = CrossEncoder Top-K
      AB  = A + B路 exact match must-have
      AC  = A + C路 LSH fuzzy match must-have
      ABC = A + B + C
    paths:
      thinking / icl / direct / plan，可逗号组合，如 direct 或 thinking,icl,direct
    refiner:
      是否启用 SQLRefiner（执行反馈修复 + literal-column 校验）
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))

from config.settings import settings
from pipeline.db_engine import DBEngine
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.refiner import SQLRefiner
from pipeline.schema_linker import SchemaLinker
from pipeline.selector import SQLSelector
from pipeline.system import TextToSQLSystem
from pipeline.utils import TokenTracker, to_halfwidth
from training.evaluate import run_evaluation


@dataclass(frozen=True)
class AblationConfig:
    name: str
    schema_mode: str = "pruned"          # full | pruned
    use_entities: bool = True
    schema_routes: str = "ABC"           # A | AB | AC | ABC
    paths: tuple[str, ...] = ("thinking", "icl", "direct")
    use_refiner: bool = True


class AblationSystem:
    """轻量复用主流程模块，按配置开关做单题推理。"""

    def __init__(self, cfg: AblationConfig):
        self.cfg = cfg
        self.client = create_async_client()
        self.llm_model = get_model_name()
        self.db_engine = DBEngine(str(settings.csv_path), settings.table_name)
        self.linker = SchemaLinker(str(settings.schema_path), str(settings.csv_path))
        self.entity_extractor = EntityExtractor(self.client, self.llm_model)
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.selector = SQLSelector()
        profiler = DatabaseProfiler(csv_path=str(settings.csv_path))
        self._profile_map = profiler.get_profile_map(profiler.profile_all())
        self._all_columns = list(self.linker.column_names)

    async def run_pipeline(self, question: str) -> dict:
        return await self.run_pipeline_async(question)

    async def run_pipeline_async(self, question: str) -> dict:
        start = time.time()
        gen_start = start
        tracker = TokenTracker()
        question = to_halfwidth(question)
        entities: list[str] = []
        evidence: dict = {}
        k = settings.top_k_embed

        if self.cfg.schema_mode == "full":
            cols = self._all_columns
            if self.cfg.use_entities:
                pre_ranked, _, _ = self.linker.hybrid_retrieve(question, [], top_k_embed=k)
                entity_schema = self.linker.build_entity_schema([x[0] for x in pre_ranked[:k]])
                entities = await self.entity_extractor.extract(
                    question, entity_schema, tracker, schema_columns=self.linker.column_names
                )
        else:
            pre_ranked, _, _ = self.linker.hybrid_retrieve(question, [], top_k_embed=k)
            pre_top_cols = [x[0] for x in pre_ranked[:k]]
            if self.cfg.use_entities:
                entity_schema = self.linker.build_entity_schema(pre_top_cols)
                entities = await self.entity_extractor.extract(
                    question, entity_schema, tracker, schema_columns=self.linker.column_names
                )
            ranked, _must_have, evidence = self.linker.hybrid_retrieve(
                question, entities, top_k_embed=k
            )
            cols = self._select_columns(ranked, evidence, k)

        schema_prompt = self.generator.build_m_schema_prompt(
            cols,
            self.linker.column_metadata,
            table_name=settings.table_name,
            randomize=False,
            profile_map=self._profile_map,
        )
        gen_start = time.time()
        task_map = self.generator.start_candidate_tasks(
            question,
            schema_prompt,
            entities,
            evidence,
            tracker,
            paths=self.cfg.paths,
        )
        raw = await asyncio.gather(
            *[t for tasks in task_map.values() for t in tasks],
            return_exceptions=True,
        )
        candidates = TextToSQLSystem._normalize_candidate_sqls(
            [r for r in raw if isinstance(r, dict)]
        )
        first_inference_time = time.time() - gen_start

        if self.cfg.use_refiner:
            refined = await self.refiner.refine_async(
                question, schema_prompt, candidates, cols, tracker
            )
        else:
            refined = self._execute_without_refiner(candidates)

        selected, reason, status = self.selector.select_best(question, schema_prompt, refined)
        if selected is None:
            result = None
            final_sql = None
        elif status == "tie":
            selected = SQLRefiner.resolve_tie(selected)
            result = selected.get("result")
            final_sql = selected.get("sql")
            reason = "tie_resolved"
        else:
            result = selected.get("result")
            final_sql = selected.get("sql")

        unique_rows = self._dedupe_rows(result)
        return {
            "final_sql": final_sql,
            "execution_result": unique_rows if result else result,
            "unique_rows_count": len(unique_rows),
            "reason": status if status != "success" else reason,
            "first_inference_time": first_inference_time,
            "repair_times": [
                t for c in refined for t in (c.get("repair_times") or [])
            ],
            "cost_time": time.time() - start,
            "token_usage": tracker.get_report(),
            "entities": entities,
            "ablation_config": self.cfg.__dict__,
        }

    def _select_columns(
        self,
        ranked: list[tuple[str, float]],
        evidence: dict,
        k: int,
    ) -> list[str]:
        cols = {name for name, _ in ranked[:k]}
        routes = self.cfg.schema_routes.upper()
        if "B" in routes:
            for matched in evidence.get("exact_matches", {}).values():
                cols.update(matched)
        if "C" in routes:
            for col_dict in evidence.get("fuzzy_matches", {}).values():
                cols.update(col_dict.keys())
        return list(cols)

    def _execute_without_refiner(self, candidates: list[dict]) -> list[dict]:
        out = []
        for cand in candidates:
            result, error = self.db_engine.execute_sql(cand["sql"])
            cand = dict(cand)
            if error is None:
                cand["status"] = "success"
                cand["result"] = result if result is not None else []
            else:
                cand["status"] = "failed"
                cand["error_msg"] = error
                cand["result"] = None
            out.append(cand)
        return out

    @staticmethod
    def _dedupe_rows(result) -> list:
        if not result:
            return []
        rows = set(tuple(row) for row in result)
        return [
            row for row in rows
            if any(v is not None and str(v).strip() != "" for v in row)
        ]


def _default_configs() -> list[AblationConfig]:
    return [
        AblationConfig("full_schema_direct", schema_mode="full", use_entities=False, paths=("direct",), use_refiner=False),
        AblationConfig("pruned_A_no_entity_direct", use_entities=False, schema_routes="A", paths=("direct",), use_refiner=False),
        AblationConfig("pruned_ABC_entity_direct", use_entities=True, schema_routes="ABC", paths=("direct",), use_refiner=False),
        AblationConfig("schema_route_A", use_entities=True, schema_routes="A", paths=("direct",), use_refiner=False),
        AblationConfig("schema_route_AB", use_entities=True, schema_routes="AB", paths=("direct",), use_refiner=False),
        AblationConfig("schema_route_AC", use_entities=True, schema_routes="AC", paths=("direct",), use_refiner=False),
        AblationConfig("path_thinking_only", use_entities=True, schema_routes="ABC", paths=("thinking",), use_refiner=False),
        AblationConfig("path_icl_only", use_entities=True, schema_routes="ABC", paths=("icl",), use_refiner=False),
        AblationConfig("path_direct_only", use_entities=True, schema_routes="ABC", paths=("direct",), use_refiner=False),
        AblationConfig("paths_all_no_refiner", use_entities=True, schema_routes="ABC", paths=("thinking", "icl", "direct"), use_refiner=False),
        AblationConfig("paths_all_with_refiner", use_entities=True, schema_routes="ABC", paths=("thinking", "icl", "direct"), use_refiner=True),
        AblationConfig("paths_all_plan_with_refiner", use_entities=True, schema_routes="ABC", paths=("thinking", "icl", "direct", "plan"), use_refiner=True),
    ]


def _parse_paths(value: str) -> tuple[str, ...]:
    allowed = {"thinking", "icl", "direct", "plan"}
    paths = tuple(x.strip() for x in value.split(",") if x.strip())
    bad = [p for p in paths if p not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"未知路径: {bad}; 可选 {sorted(allowed)}")
    return paths or ("direct",)


async def main_async(args: argparse.Namespace) -> None:
    if args.eval_csv:
        settings.train_csv = Path(args.eval_csv)
    if args.use_full_data:
        settings.train_split = 0.0
        settings.val_split = 0.0
    if args.top_k:
        settings.top_k_embed = args.top_k

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        AblationConfig(
            name=args.name,
            schema_mode=args.schema_mode,
            use_entities=args.use_entities,
            schema_routes=args.schema_routes,
            paths=args.paths,
            use_refiner=args.use_refiner,
        )
    ] if args.single else _default_configs()

    rows = []
    for cfg in configs:
        print(f"\n[ablation] running {cfg.name}: {cfg}")
        system = AblationSystem(cfg)
        full_log = output_dir / f"{cfg.name}_run_report.json"
        error_log = output_dir / f"{cfg.name}_error_analysis.json"
        metrics = await run_evaluation(
            system,
            output_path=str(error_log),
            full_output_path=str(full_log),
            label=cfg.name,
            concurrency=args.concurrency,
        )
        rows.append({
            "name": cfg.name,
            "schema_mode": cfg.schema_mode,
            "use_entities": cfg.use_entities,
            "schema_routes": cfg.schema_routes,
            "paths": ",".join(cfg.paths),
            "use_refiner": cfg.use_refiner,
            "accuracy": metrics.get("accuracy", 0.0),
            "correct": metrics.get("correct", 0),
            "total": metrics.get("total", 0),
            "avg_first_inference_time": metrics.get("avg_first_inference_time", 0.0),
            "avg_total_time": metrics.get("avg_total_time", 0.0),
            "error_type_counts": json.dumps(metrics.get("error_type_counts", {}), ensure_ascii=False),
            "full_log_path": metrics.get("full_log_path", ""),
        })

    summary_path = output_dir / "ablation_metrics.csv"
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation] summary: {summary_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Text-to-SQL 主流程模块消融实验")
    p.add_argument("--eval-csv", default="", help="评测 CSV；缺省使用 settings.train_csv")
    p.add_argument("--use-full-data", action="store_true", help="使用全部 MATCH 数据")
    p.add_argument("--concurrency", type=int, default=1, help="并行题目数")
    p.add_argument("--top-k", type=int, default=0, help="覆盖 settings.top_k_embed")
    p.add_argument("--output-dir", default="baseline/ablation", help="输出目录")
    p.add_argument("--single", action="store_true", help="只运行命令行指定的一组配置")
    p.add_argument("--name", default="custom", help="single 模式下的配置名")
    p.add_argument("--schema-mode", choices=("full", "pruned"), default="pruned")
    p.add_argument("--use-entities", dest="use_entities", action="store_true", default=True)
    p.add_argument("--no-entities", dest="use_entities", action="store_false")
    p.add_argument("--schema-routes", choices=("A", "AB", "AC", "ABC"), default="ABC")
    p.add_argument("--paths", type=_parse_paths, default=("direct",), help="逗号分隔：thinking,icl,direct,plan")
    p.add_argument("--use-refiner", dest="use_refiner", action="store_true", default=True)
    p.add_argument("--no-refiner", dest="use_refiner", action="store_false")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
