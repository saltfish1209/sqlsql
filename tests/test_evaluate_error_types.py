from training.evaluate import classify_error_type


def test_classify_empty_prediction():
    error_type, detail = classify_error_type({"a"}, set(), reason="success")
    assert error_type == "EMPTY_PRED"
    assert "empty" in detail


def test_classify_sql_execution_error():
    error_type, detail = classify_error_type({"a"}, set(), reason="no such column: foo")
    assert error_type == "SQL_EXEC_ERROR"
    assert "no such column" in detail


def test_classify_partial_match_for_sets():
    error_type, detail = classify_error_type({"a", "b"}, {"a"}, reason="success")
    assert error_type == "PARTIAL_MATCH"
    assert "overlap" in detail


def test_classify_shape_mismatch_for_multi_result():
    error_type, detail = classify_error_type([{"a"}], {"a"}, reason="success")
    assert error_type == "SHAPE_MISMATCH"
    assert "shape" in detail


def test_classify_value_mismatch_for_disjoint_sets():
    error_type, detail = classify_error_type({"a"}, {"b"}, reason="success")
    assert error_type == "VALUE_MISMATCH"
    assert "disjoint" in detail
