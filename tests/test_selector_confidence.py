from pipeline.selector import SQLSelector


def _cand(path_type, sql, result):
    return {
        "type": path_type,
        "sql": sql,
        "status": "success",
        "result": result,
    }


def test_selector_returns_high_confidence_cluster_without_thinking_priority():
    selector = SQLSelector(confidence_threshold=0.6)
    candidates = [
        _cand("ICL_Path", "SELECT 'a'", [("a",)]),
        _cand("Direct_Path", "SELECT 'a'", [("a",)]),
        _cand("plan_path", "SELECT 'b'", [("b",)]),
    ]

    selected, reason, status = selector.select_best("q", "schema", candidates)

    assert status == "success"
    assert selected["result"] == [("a",)]
    assert selected["confidence"] == 2 / 3
    assert "confidence=0.67" in reason


def test_selector_uses_deterministic_non_thinking_tie_break():
    selector = SQLSelector(confidence_threshold=0.6)
    candidates = [
        _cand("Direct_Path", "SELECT 'b'", [("b",)]),
        _cand("ICL_Path", "SELECT 'a'", [("a",)]),
    ]

    selected, reason, status = selector.select_best("q", "schema", candidates)

    assert status == "low_confidence"
    assert selected["type"] == "ICL_Path"
    assert selected["confidence"] == 0.5
    assert "low confidence" in reason
