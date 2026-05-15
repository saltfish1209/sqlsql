from __future__ import annotations


class SQLSelector:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def select_best(self, question: str, schema_prompt: str, candidates: list[dict]) -> tuple:
        valid = [c for c in candidates if c.get("status") == "success"]
        if not valid:
            return (candidates[0] if candidates else None), "所有路径均未成功", "failed"
        best = sorted(valid, key=lambda c: (-(c.get("confidence") or 0.0), c.get("sql", "")))[0]
        confidence = float(best.get("confidence") or 0.0)
        if confidence >= self.confidence_threshold:
            return best, f"confidence={confidence:.2f}", "success"
        return best, f"low confidence={confidence:.2f}", "low_confidence"
