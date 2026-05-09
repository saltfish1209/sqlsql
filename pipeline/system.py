"""
主编排器 —— 串联 Schema Linking → Entity Extraction → SQL Generation
→ Refinement → Selection 的完整 Text-to-SQL 推理流水线。

项目创新:
  1. A路预排序 → 精简 Schema 实体提取（避免 Examples 值污染）
  2. 三路混合召回 (CrossEncoder + ExactMatch + LSH)
  3. 扩展窗口梯队回退（保留高相关列）
  4. 自动 Profiling 注入 Prompt（论文新增）
  5. Literal-Column 校验（论文新增）
  6. Schema 字段随机化增强多样性（论文新增）
"""
from __future__ import annotations

import asyncio
import json
import re
import time

from config.settings import settings
from generation.multi_result_utils import MULTI_RESULT_SEP
from pipeline.db_engine import DBEngine
from pipeline.entity_extractor import EntityExtractor
from pipeline.generator import SQLGenerator
from pipeline.llm_client import create_async_client, get_model_name
from pipeline.profiler import DatabaseProfiler
from pipeline.refiner import SQLRefiner, majority_agreed
from pipeline.schema_linker import SchemaLinker
from pipeline.selector import SQLSelector
from pipeline.utils import to_halfwidth, debug_print, TokenTracker

_SQL_SINGLE_QUOTE_LITERAL_RE = re.compile(r"'([^']*)'")
_CJK_RE = r"\u4e00-\u9fff"
_ASCII_WORD_RE = r"A-Za-z0-9"


class TextToSQLSystem:
    """
    完整 Text-to-SQL 系统。

    初始化时加载数据库、构建索引、运行 Profiler。
    调用 run_pipeline_async(question) 返回推理结果字典。
    """

    def __init__(self):
        debug_print(">>> [System Init] 正在初始化 Text-to-SQL 各组件...")

        csv_path = str(settings.csv_path)
        schema_path = str(settings.schema_path)

        self.client = create_async_client()
        self.llm_model = get_model_name()

        self.db_engine = DBEngine(csv_path, settings.table_name)
        self.linker = SchemaLinker(schema_path, csv_path)
        self.entity_extractor = EntityExtractor(self.client, self.llm_model)
        self.generator = SQLGenerator(self.client, self.llm_model)
        self.refiner = SQLRefiner(self.client, self.llm_model, self.db_engine)
        self.selector = SQLSelector()

        # 自动 Profiling（论文新增）
        # profile_map：{列名: 内联摘要}，只对当前 selected_columns 中的列注入，
        # 拼接在各字段描述尾部，不做全量底部拼接。
        self.profiler = DatabaseProfiler(csv_path=csv_path)
        self._profiles = self.profiler.profile_all()
        self._profile_map = self.profiler.get_profile_map(self._profiles)

        debug_print(">>> [System Init] 初始化完成。\n")

    # ──────────── 主入口 ────────────

    async def run_pipeline_async(self, question: str) -> dict:
        """
        分发器：先判断是否多子问题，若是则逐题作答（agentic 风格）后合并；
        否则直接走单题流程。

        多题输出约定（与 evaluate.py、数据集生成端保持一致）：
          - ``execution_result``: ``list[list[tuple]]``  # 每个元素是一个子问题的结果集
          - ``final_sql``       : ``"sql1 ‖ sql2"``       # 子 SQL 用 MULTI_RESULT_SEP 拼
          - ``is_multi_question`` = True
          - ``sub_questions``  : list[str]
        """
        start = time.time()
        question = to_halfwidth(question)

        # —— Agentic 拆题：仅当问题里出现 ≥2 个 ?/？ 时才请 LLM 判一次 —— 
        sub_questions = await self._maybe_split_question(question)
        if len(sub_questions) <= 1:
            return await self._run_single_pipeline(question)

        debug_print(f"[Pipeline] 多子问题模式 → {sub_questions}")
        sub_outputs: list[dict] = []
        for i, sub_q in enumerate(sub_questions, 1):
            debug_print(f"[Pipeline] >>> 子问题 {i}/{len(sub_questions)}: {sub_q}")
            sub_out = await self._run_single_pipeline(sub_q)
            sub_outputs.append(sub_out)

        return self._merge_sub_outputs(question, sub_questions, sub_outputs, start)

    async def _run_single_pipeline(self, question: str) -> dict:
        """单题流程（原 run_pipeline_async 主体）。"""
        start = time.time()
        tracker = TokenTracker()
        K = settings.top_k_embed
        first_inference_time = 0.0
        selected_repair_times: list[float] = []

        # Step 1: A路预排序（无实体），构建精简 Schema 用于实体提取
        debug_print("[Pipeline] A路预排序，构建精简 Schema...")
        pre_ranked, _, _ = self.linker.hybrid_retrieve(question, [], top_k_embed=K)
        pre_top_cols = [x[0] for x in pre_ranked[:K]]
        entity_schema = self.linker.build_entity_schema(pre_top_cols)

        # Step 2: LLM + 规则联合实体提取
        entity_start = time.time()
        debug_print("[Pipeline] Step2 实体提取开始...")
        entities = await self.entity_extractor.extract(
            question, entity_schema, tracker,
            schema_columns=self.linker.column_names,
        )
        debug_print(
            f"[Pipeline] Step2 实体提取完成，耗时 {time.time() - entity_start:.2f}s"
        )

        # Step 3: 带实体的完整混合召回
        ranked, must_have, evidence = self.linker.hybrid_retrieve(
            question, entities, top_k_embed=K
        )
        full_list = [x[0] for x in ranked]

        # 梯队 1: Top-K + 必须命中列
        tier1 = list(set(full_list[:K]) | must_have)
        debug_print(f"[Pipeline] 梯队1(Top {K}): {tier1}")
        best = await self._try_flow(question, tier1, entities, evidence, tracker)
        if best:
            first_inference_time = best.get("first_inference_time", 0.0)
            selected_repair_times = best.get("repair_times", []) or []

        # 梯队 2: 扩展至 Top-2K（保留梯队1 + 新增列）
        if not best or best.get("status") != "success":
            tier2 = list(set(full_list[:K * 2]) | must_have)
            debug_print(f"[Pipeline] 梯队2 扩展至 Top {K*2}")
            best = await self._try_flow(question, tier2, entities, evidence, tracker)
            if best:
                selected_repair_times = best.get("repair_times", []) or []

        # 梯队 3: 全新列段 Top-2K+1 ~ 3K
        if not best or best.get("status") != "success":
            tier3 = list(set(full_list[K * 2:K * 3]) | must_have)
            debug_print(f"[Pipeline] 梯队3 兜底列")
            best = await self._try_flow(question, tier3, entities, evidence, tracker)
            if best:
                selected_repair_times = best.get("repair_times", []) or []

        total_time = time.time() - start

        if best is None:
            return {
                "final_sql": None,
                "execution_result": None,
                "unique_rows_count": 0,
                "reason": "all_paths_failed",
                "first_inference_time": first_inference_time,
                "repair_times": selected_repair_times,
                "cost_time": total_time,
                "token_usage": tracker.get_report(),
                "entities": entities,
                "is_multi_question": False,
            }

        final_res = best.get("result")
        unique_rows = []
        unique_count = 0
        if final_res:
            unique_set = set(tuple(row) for row in final_res)
            # 过滤全 NULL / 全空串行，保持与 evaluate 归一化逻辑一致
            unique_set = {
                row for row in unique_set
                if any(
                    v is not None and str(v).strip() != ""
                    for v in row
                )
            }
            unique_count = len(unique_set)
            unique_rows = list(unique_set)

        return {
            "final_sql": best.get("sql", "SELECT 1"),
            "execution_result": unique_rows if final_res else final_res,
            "unique_rows_count": unique_count,
            "reason": best.get("status", "success"),
            "first_inference_time": first_inference_time,
            "repair_times": selected_repair_times,
            "cost_time": total_time,
            "token_usage": tracker.get_report(),
            "entities": entities,
            "is_multi_question": False,
        }

    async def run_pipeline(self, question: str) -> dict:
        """别名，供 evaluate 脚本调用。"""
        return await self.run_pipeline_async(question)

    # ──────────── Agentic 多子问题拆分 ────────────

    async def _maybe_split_question(self, question: str) -> list[str]:
        """
        判断问题是否包含多个独立子问题，并返回子问题列表（单题时长度=1）。

        策略：
          1) 廉价过滤：仅当问号 ≥2 时才请 LLM 判断（节省 token）。
          2) LLM 输出 JSON 数组；解析失败或返回 ≤1 项时，回退为单题。
          3) 整个机制可由 ``settings.enable_question_split`` 关闭（缺省开启）。
        """
        if not getattr(settings, "enable_question_split", True):
            return [question]

        qmark_count = question.count("？") + question.count("?")
        if qmark_count < 2:
            return [question]

        prompt = (
            "你是一个 NL2SQL 助手。判断下面这条用户问题是否包含**多个相互独立**的子问题"
            "（即每个子问题都可以独立由一条 SQL 回答）。\n"
            "- 如果是，把它拆解为多个独立子问题，**不要丢失任何过滤条件 / 实体**，"
            "每个子问题都要继承原问题中的所有实体。\n"
            "- 如果不是（单一问题），直接返回只含一个元素的数组。\n"
            "只输出 JSON 字符串数组，不要其他文字。\n\n"
            "[示例]\n"
            "问题: 物料编码500116755的中标厂家有哪些？\n"
            "输出: [\"物料编码500116755的中标厂家有哪些？\"]\n\n"
            "问题: 许继电气共中标几个批次？这些批次的批次号分别是什么？\n"
            "输出: [\"许继电气共中标几个批次？\", \"许继电气中标的批次号分别是什么？\"]\n\n"
            "问题: 珠海许继的中标总金额是多少？平均单价又是多少？\n"
            "输出: [\"珠海许继的中标总金额是多少？\", \"珠海许继的中标平均单价是多少？\"]\n\n"
            f"[当前问题]\n{question}\n输出:"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
            arr = self._parse_str_array(text)
            arr = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
            if len(arr) >= 2:
                return arr
        except Exception as e:  # pragma: no cover - LLM 异常时回退
            debug_print(f"[Pipeline] 拆题失败，回退为单题: {e}")

        return [question]

    @staticmethod
    def _parse_str_array(text: str) -> list[str]:
        """从 LLM 文本响应中抽出 JSON 字符串数组。"""
        if not text:
            return []
        # 直接 JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        # 截取首尾方括号再 parse
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass
        return []

    @staticmethod
    def _merge_sub_outputs(
        original_question: str,
        sub_questions: list[str],
        sub_outputs: list[dict],
        start_ts: float,
    ) -> dict:
        """合并多子问题输出，与 evaluate 的多结果比较口径对齐。"""
        sub_sqls = [str(o.get("final_sql") or "") for o in sub_outputs]
        sub_results: list[list] = []
        for o in sub_outputs:
            sub_res = o.get("execution_result") or []
            sub_results.append(list(sub_res))

        # 实体：按子问题前缀汇总
        sub_entities = []
        for o in sub_outputs:
            sub_entities.append(o.get("entities") or [])

        first_inferences = [float(o.get("first_inference_time", 0.0) or 0.0) for o in sub_outputs]
        all_repairs: list[float] = []
        for o in sub_outputs:
            all_repairs.extend(o.get("repair_times", []) or [])

        return {
            "final_sql": MULTI_RESULT_SEP.join(sub_sqls),
            "execution_result": sub_results,        # list[list[tuple]]
            "unique_rows_count": sum(len(r) for r in sub_results),
            "reason": "multi_success" if all(o.get("reason") == "success" for o in sub_outputs) else "multi_partial",
            "first_inference_time": max(first_inferences) if first_inferences else 0.0,
            "repair_times": all_repairs,
            "cost_time": time.time() - start_ts,
            "token_usage": {},                       # 各子题 tracker 暂不合并
            "entities": [e for sub in sub_entities for e in sub],
            "is_multi_question": True,
            "sub_questions": sub_questions,
            "sub_outputs": sub_outputs,
        }

    # ──────────── 工具方法 ────────────

    @staticmethod
    def _normalize_sql_literals(sql: str) -> str:
        """轻量归一化生成 SQL：中文标点半角化 + 中英混排空格修正。"""
        if not sql:
            return sql
        # 先做整句半角化：中文标点/全角符号 -> 英文半角符号
        sql = to_halfwidth(sql)
        # 修正标识符里的中英混排空格：计划批次 ID -> 计划批次ID
        sql = re.sub(rf"(?<=[{_CJK_RE}])\s+(?=[{_ASCII_WORD_RE}])", "", sql)
        sql = re.sub(rf"(?<=[{_ASCII_WORD_RE}])\s+(?=[{_CJK_RE}])", "", sql)

        def _norm_match(m: re.Match[str]) -> str:
            lit = to_halfwidth(m.group(1))
            # 例：ECP 招标合同 -> ECP招标合同, II 型 -> II型
            lit = re.sub(rf"(?<=[{_ASCII_WORD_RE}])\s+(?=[{_CJK_RE}])", "", lit)
            lit = re.sub(rf"(?<=[{_CJK_RE}])\s+(?=[{_ASCII_WORD_RE}])", "", lit)
            return f"'{lit}'"

        return _SQL_SINGLE_QUOTE_LITERAL_RE.sub(_norm_match, sql)

    @classmethod
    def _normalize_candidate_sqls(cls, cands: list[dict]) -> list[dict]:
        """对候选 SQL 做轻量字面量归一化（原地修改，最小改动）。"""
        for c in cands:
            if isinstance(c, dict) and c.get("sql"):
                c["sql"] = cls._normalize_sql_literals(str(c["sql"]))
        return cands

    @staticmethod
    def _has_nonempty_result(cand: dict) -> bool:
        """判断 refined 候选的执行结果是否包含至少一行真实业务数据。

        把 `[]` / `[(0,)]` / `[(None,)]` / `[('',)]` / `[(0, '')]` 这类
        "聚合零 / 全空"的结果统一视作"空"，让早停继续等慢路兜底。
        """
        result = cand.get("result")
        if not result:
            return False
        for row in result:
            if not isinstance(row, (list, tuple)):
                if row is not None and str(row).strip() not in ("", "0"):
                    return True
                continue
            for v in row:
                if v is None:
                    continue
                s = str(v).strip()
                if s == "" or s == "0" or s == "0.0":
                    continue
                return True
        return False

    # ──────────── 单梯队尝试 ────────────

    async def _try_flow(
        self,
        question: str,
        cols: list[str],
        entities: list[str],
        evidence: dict,
        tracker: TokenTracker,
        randomize_schema: bool = False,
    ) -> dict | None:
        """
        单梯队三路生成 + 快路早停。

        编排逻辑：
          1) 同时启动 thinking / icl / direct 三路（thinking 路较慢）。
          2) **先等** ICL + Direct 两条快路；refine 后若结果一致 (majority_agreed)，
             立即取消 thinking 路返回 —— 用于减少单条问题的尾延迟。
          3) 否则等 thinking 路返回并加入 refine + 投票。
        """
        m_schema = self.generator.build_m_schema_prompt(
            cols, self.linker.column_metadata,
            randomize=randomize_schema,
            profile_map=self._profile_map,
        )

        gen_start = time.time()
        # paths：thinking + icl + direct + plan（agentic 多步抽取，受 settings 开关控制）
        active_paths: tuple[str, ...] = (
            ("thinking", "icl", "direct", "plan")
            if settings.enable_plan_path
            else ("thinking", "icl", "direct")
        )
        task_map = self.generator.start_candidate_tasks(
            question, m_schema, entities, evidence, tracker,
            paths=active_paths,
        )
        fast_tasks = task_map.get("icl", []) + task_map.get("direct", [])
        slow_tasks = task_map.get("thinking", []) + task_map.get("plan", [])

        # —— 阶段一：等两条快路 ——
        fast_raw = (
            await asyncio.gather(*fast_tasks, return_exceptions=True)
            if fast_tasks else []
        )
        fast_cands = self._normalize_candidate_sqls(
            [r for r in fast_raw if isinstance(r, dict)]
        )
        first_inference_time = time.time() - gen_start
        debug_print(f"[Pipeline] 快路完成 ({len(fast_cands)}个候选)")

        fast_refined = await self.refiner.refine_async(
            question, m_schema, fast_cands, cols, tracker
        )

        # 早停判定：两条快路投票一致 **且**结果非空。
        # 案例 [2]/[11]/[13]/[15] 都是快路一致但结果是空集 / `[(0,)]` 之类的"伪一致"，
        # 仅靠投票一致会过早杀掉慢路（thinking + plan）失去兜底机会。
        fast_success = [c for c in fast_refined if c.get("status") == "success"]
        fast_voted_nonempty = (
            majority_agreed(fast_success)
            and any(self._has_nonempty_result(c) for c in fast_success)
        )
        if fast_voted_nonempty and slow_tasks:
            debug_print(
                f"[Pipeline] 快路 {len(fast_success)} 路一致且结果非空 → 取消慢路"
            )
            for t in slow_tasks:
                t.cancel()
            await asyncio.gather(*slow_tasks, return_exceptions=True)
            all_refined = fast_refined
        elif slow_tasks:
            if majority_agreed(fast_success):
                debug_print("[Pipeline] 快路一致但结果为空 → 仍等慢路兜底")
            else:
                debug_print("[Pipeline] 快路未达成一致 → 等慢路兜底")
            slow_raw = await asyncio.gather(*slow_tasks, return_exceptions=True)
            slow_cands = self._normalize_candidate_sqls(
                [r for r in slow_raw if isinstance(r, dict)]
            )
            slow_refined = await self.refiner.refine_async(
                question, m_schema, slow_cands, cols, tracker
            )
            all_refined = fast_refined + slow_refined
            # 把"等到所有路全部生成完"的耗时也视作首次推理时间
            first_inference_time = time.time() - gen_start
        else:
            all_refined = fast_refined

        debug_print(f"[Pipeline] Refiner 修正后 ({len(all_refined)}个)")

        selected, reason, status = self.selector.select_best(
            question, m_schema, all_refined
        )
        if selected is None:
            debug_print(f"[Pipeline] 选择结果: {reason}")
            return None
        if status == "tie":
            debug_print("[Pipeline] 触发平票，按路径优先级决策")
            resolved = SQLRefiner.resolve_tie(selected)
            resolved["first_inference_time"] = first_inference_time
            resolved["repair_times"] = resolved.get("repair_times", []) or []
            return resolved

        debug_print(f"[Pipeline] 选择结果: {reason}")
        selected["first_inference_time"] = first_inference_time
        selected["repair_times"] = selected.get("repair_times", []) or []
        return selected


# ──────────── CLI 入口 ────────────

if __name__ == "__main__":
    system = TextToSQLSystem()

    test_cases = [
        {
            "question": "物料编码500116755的中标厂家有哪些？",
            "ground_truth": (
                "南京自强铁路车辆配件有限公司，天铂电力集团有限公司，苏州华源电气有限公司，"
                "江苏优家宁科技有限公司，江苏镇安电力设备有限公司，广蓝电气设备有限公司，"
                "河南平高通用电气有限公司，江苏一变电力装备有限公司，扬州电力设备修造厂有限公司，"
                "浙江聚弘凯电气有限公司，珠海沃顿电气有限公司，江苏大烨智能电气股份有限公司，"
                "北京合纵科技股份有限公司，江西环林集团股份有限公司，上海南华兰陵电气有限公司，"
                "扬州北辰电气集团有限公司，珠海许继电气有限公司，许继德理施尔电气有限公司，"
                "上海敬道电气有限公司，梵迩佳智能电气有限公司"
            ),
        }
    ]

    def _parse_gt(gt: str) -> set[str]:
        return {
            x.strip()
            for x in str(gt).replace("\n", "").split("，")
            if x.strip()
        }

    def _normalize_pred_rows(rows) -> set[str]:
        if not rows:
            return set()
        out = set()
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            vals = ["" if v is None else str(v).strip() for v in row]
            if all(v == "" for v in vals):
                continue
            out.add("|".join(vals))
        return out

    loop = asyncio.get_event_loop()
    for case in test_cases:
        q = case["question"]
        gt_set = _parse_gt(case["ground_truth"])
        print("\n" + "=" * 60)
        result = loop.run_until_complete(system.run_pipeline_async(q))
        print(f"FINAL OUTPUT (Time: {result['cost_time']:.2f}s)")
        print(f"SQL: {result['final_sql']}")
        tokens = result.get("token_usage", {})
        print(
            f"Token Usage: Input={tokens.get('input_tokens', 0)} | "
            f"Output={tokens.get('output_tokens', 0)} | "
            f"Total={tokens.get('total_tokens', 0)}"
        )
        print(f"Unique Rows: {result['unique_rows_count']}")
        res = result["execution_result"]
        print(f"Result (First 10): {res[:10] if res else 'Empty'}")
        pred_set = _normalize_pred_rows(res)
        ok = pred_set == gt_set
        print(f"Correctness: {'OK' if ok else 'FAIL'}")
        if not ok:
            missing = sorted(gt_set - pred_set)
            extra = sorted(pred_set - gt_set)
            print(f"Missing ({len(missing)}): {missing[:10]}")
            print(f"Extra ({len(extra)}): {extra[:10]}")
        print("=" * 60)
