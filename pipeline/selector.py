from __future__ import annotations

from collections import Counter
from typing import Any


class SQLSelector:
    def __init__(self, confidence_threshold: float = 0.6, tie_margin: float = 8.0):
        self.confidence_threshold = confidence_threshold
        self.tie_margin = tie_margin

    @staticmethod
    def _result_key(candidate: dict) -> tuple:
        rows = candidate.get("result") or []
        normalized = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                normalized.append(tuple(str(x) for x in row))
            else:
                normalized.append((str(row),))
        return tuple(normalized)

    @staticmethod
    def _non_empty(candidate: dict) -> bool:
        rows = candidate.get("result")
        if not rows:
            return False
        for row in rows:
            if isinstance(row, (list, tuple)):
                if any(v is not None and str(v).strip() != "" for v in row):
                    return True
            elif row is not None and str(row).strip() != "":
                return True
        return False

    @staticmethod
    def _issue_count(candidate: dict) -> int:
        return len(candidate.get("checker_issues") or [])

    @staticmethod
    def _sql_complexity(candidate: dict) -> float:
        sql = str(candidate.get("sql") or "")
        where_count = sql.upper().count(" WHERE ")
        and_count = sql.upper().count(" AND ")
        return len(sql) / 200.0 + where_count + and_count * 0.5

    def score_candidates(self, candidates: list[dict]) -> list[dict]:
        result_counts = Counter(
            self._result_key(c)
            for c in candidates
            if c.get("status") == "success" and self._non_empty(c)
        )
        scored: list[dict] = []
        for idx, candidate in enumerate(candidates):
            score = 0.0
            if candidate.get("status") == "success":
                score += 30.0
            if self._non_empty(candidate):
                score += 20.0
            result_key = self._result_key(candidate)
            consensus = result_counts.get(result_key, 0) if result_key else 0
            if consensus > 1:
                score += 15.0 + consensus
            score -= self._issue_count(candidate) * 8.0
            score += min(float(candidate.get("confidence") or 0.0), 1.0) * 5.0
            score -= self._sql_complexity(candidate)
            scored.append({"index": idx, "candidate": candidate, "score": round(score, 4), "consensus": consensus})
        return sorted(scored, key=lambda x: (-x["score"], x["index"]))

    def select_best(self, question: str, schema_prompt: str, candidates: list[dict]) -> tuple:
        if not candidates:
            return None, "no candidates", "failed"
        scored = self.score_candidates(candidates)
        best = scored[0]
        selected = best["candidate"]
        if selected.get("status") != "success":
            return selected, f"score={best['score']:.2f}; no successful candidates", "failed"
        reason = (
            f"score={best['score']:.2f}; "
            f"consensus={best['consensus']}; "
            f"type={selected.get('type', '')}; "
            f"variant={selected.get('variant_id', '')}"
        )
        return selected, reason, "success"
