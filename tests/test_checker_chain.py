from pipeline.checkers import SQLCheckerChain


class FakeDB:
    def check_literal_in_column(self, column, literal):
        return column == "供应商" and literal == "真实供应商"


def _codes(issues):
    return {issue.code for issue in issues}


def test_checker_chain_reports_execution_error():
    issues = SQLCheckerChain(FakeDB()).check(
        'SELECT "未知列" FROM procurement_table',
        result=None,
        execution_error="NO_SUCH_COLUMN: no such column: 未知列",
    )

    assert "EXECUTION_ERROR" in _codes(issues)


def test_checker_chain_reports_literal_mismatch_and_quality_issues():
    sql = (
        "SELECT * FROM procurement_table "
        "WHERE \"供应商\" = '不存在供应商' "
        "ORDER BY \"中标金额\" DESC LIMIT 1"
    )

    issues = SQLCheckerChain(FakeDB()).check(sql, result=[], execution_error=None)

    assert {
        "LITERAL_MISMATCH",
        "EMPTY_RESULT",
        "SELECT_STAR",
        "NULL_GUARD_MISSING",
    }.issubset(_codes(issues))


def test_checker_chain_accepts_clean_executable_sql():
    sql = (
        "SELECT \"供应商\" FROM procurement_table "
        "WHERE \"供应商\" = '真实供应商' "
        "AND \"中标金额\" IS NOT NULL "
        "ORDER BY \"中标金额\" DESC LIMIT 1"
    )

    issues = SQLCheckerChain(FakeDB()).check(sql, result=[("真实供应商",)], execution_error=None)

    assert issues == []
