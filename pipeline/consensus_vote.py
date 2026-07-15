from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable


_ROUTE_PRIORITY = {
    "direct": 0,
    "icl": 1,
    "intent_plan": 2,
}

_CONDITION_RE = re.compile(
    r'(?:(?:"(?P<quoted_col>[^"]+)")|(?P<plain_col>[A-Za-z_][\w]*))'
    r"\s*(?P<operator>=|LIKE)\s*'(?P<literal>(?:''|[^'])*)'",
    re.IGNORECASE,
)


def _result_key(candidate: dict) -> tuple:
    rows = candidate.get("result") or []
    normalized = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            normalized.append(tuple(str(x) for x in row))
        else:
            normalized.append((str(row),))
    return tuple(sorted(set(normalized)))


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


def _row_count(candidate: dict) -> int:
    rows = candidate.get("result") or []
    return len(rows)


def build_value_link_probe(
    candidate: dict,
    literal_exists: Callable[[str, str], bool] | None = None,
    literal_probe: Callable[[str, str], dict] | None = None,
) -> dict:
    sql = str(candidate.get("sql") or "")
    if not sql:
        return {"risk": 0, "reasons": [], "literal_checks": []}

    where_match = re.search(r"\bWHERE\b(.+)", sql, flags=re.IGNORECASE | re.DOTALL)
    where_sql = where_match.group(1) if where_match else ""
    conditions = []
    for match in _CONDITION_RE.finditer(where_sql):
        conditions.append(
            {
                "column": match.group("quoted_col") or match.group("plain_col") or "",
                "operator": match.group("operator").upper(),
                "literal": match.group("literal").replace("''", "'").strip(),
            }
        )

    reasons: list[str] = []
    literal_checks: list[dict] = []
    if candidate.get("status") == "success" and not _non_empty(candidate):
        reasons.append("empty_result")
    if re.search(r"\bOR\b", where_sql, flags=re.IGNORECASE) and _row_count(candidate) > 1:
        reasons.append("broad_or_result")

    like_conditions = [item for item in conditions if item["operator"] == "LIKE"]
    if _row_count(candidate) > 1 and any("%" in item["literal"] for item in like_conditions):
        reasons.append("broad_like_result")

    literals_by_column: dict[str, list[str]] = defaultdict(list)
    for item in conditions:
        if len(item["literal"].strip("%_")) >= 4:
            literals_by_column[item["column"]].append(item["literal"])
        if item["operator"] != "=" or not item["literal"]:
            continue
        probe = literal_probe(item["column"], item["literal"]) if literal_probe else {}
        exists = bool(
            probe.get("exact_exists")
            if probe
            else literal_exists(item["column"], item["literal"])
            if literal_exists
            else True
        )
        literal_checks.append(
            {
                **item,
                "exists": exists,
                "like_exists": bool(probe.get("like_exists")),
                "candidate_values": list(probe.get("candidate_values") or []),
            }
        )
        if not exists:
            reasons.append(f"literal_not_found:{item['column']}={item['literal']}")

    if re.search(r"\b(?:AND|OR)\b", where_sql, flags=re.IGNORECASE):
        for column, literals in literals_by_column.items():
            if len(set(literals)) >= 2:
                reasons.append(f"same_column_multi_literal:{column}")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "risk": len(unique_reasons),
        "reasons": unique_reasons,
        "literal_checks": literal_checks,
    }


def annotate_candidate_risk(
    candidate: dict,
    literal_exists: Callable[[str, str], bool] | None = None,
    literal_probe: Callable[[str, str], dict] | None = None,
) -> dict:
    probe = build_value_link_probe(candidate, literal_exists, literal_probe)
    candidate["value_link_probe"] = probe
    candidate["value_link_risk"] = probe["risk"]
    return candidate


def _candidate_risk(candidate: dict) -> int:
    try:
        judge_risk = int(
            candidate.get("post_repair_judge_risk")
            if candidate.get("post_repair_judge_risk") is not None
            else candidate.get("judge_risk") or 0
        )
    except (TypeError, ValueError):
        judge_risk = 0
    try:
        value_link_risk = int(candidate.get("value_link_risk"))
    except (TypeError, ValueError):
        value_link_risk = int(build_value_link_probe(candidate)["risk"])
    return judge_risk + value_link_risk


def _route_priority(candidate: dict) -> int:
    route = str(candidate.get("type") or "").lower()
    return _ROUTE_PRIORITY.get(route, 99)


def _sort_key(candidate: dict, group_sizes: dict[tuple, int]) -> tuple:
    result_key = _result_key(candidate)
    return (
        0 if candidate.get("status") == "success" else 1,
        _candidate_risk(candidate),
        0 if _non_empty(candidate) else 1,
        -group_sizes.get(result_key, 1),
        1 if candidate.get("is_refined") else 0,
        _route_priority(candidate),
        int(candidate.get("variant_id") or 0),
    )


def select_by_consensus(candidates: list[dict]) -> tuple[dict | None, str, str]:
    if not candidates:
        return None, "no candidates", "failed"

    successful = [c for c in candidates if c.get("status") == "success"]
    if not successful:
        return candidates[0], "no successful candidates", "failed"

    non_empty = [c for c in successful if _non_empty(c)]
    vote_pool = non_empty or successful
    groups: dict[tuple, list[dict]] = {}
    for candidate in vote_pool:
        groups.setdefault(_result_key(candidate), []).append(candidate)

    group_sizes = {key: len(items) for key, items in groups.items()}
    selected = min(successful, key=lambda item: _sort_key(item, group_sizes))
    consensus_vote = group_sizes.get(_result_key(selected), 1)
    status = "success" if selected.get("status") == "success" else "failed"
    reason = (
        f"consensus_vote={consensus_vote}; "
        f"risk={_candidate_risk(selected)}; "
        f"value_link_risk={selected.get('value_link_risk', build_value_link_probe(selected)['risk'])}; "
        f"non_empty={_non_empty(selected)}; "
        f"rows={_row_count(selected)}; "
        f"is_refined={bool(selected.get('is_refined'))}; "
        f"type={selected.get('type', '')}; "
        f"variant={selected.get('variant_id', '')}"
    )
    return selected, reason, status
