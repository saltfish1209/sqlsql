"""
SQL 构造器 —— 根据条件和聚合配置程序化生成 SQL 语句。
用于训练数据生成阶段（非推理阶段）。
"""
import re


class SQLQueryBuilder:
    def __init__(self, table_name="procurement_table"):
        self.table_name = table_name

    def _fmt(self, val):
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return str(int(val)) if val.is_integer() else str(val)
        val_str = str(val).strip()
        if val_str.endswith(".0") and val_str[:-2].isdigit():
            val_str = val_str[:-2]
        val_str = val_str.replace("'", "''")
        return f"'{val_str}'"

    def _col(self, col):
        if isinstance(col, list):
            col = col[0] if col else ""
        return f'"{str(col).strip()}"'

    def generate(self, base_conditions, multi_conditions, agg_config):
        clauses = []
        for col, val in base_conditions.items():
            clauses.append(f"{self._col(col)} = {self._fmt(val)}")
        for col, vals in multi_conditions.items():
            val_str = ", ".join([self._fmt(v) for v in vals])
            clauses.append(f"{self._col(col)} IN ({val_str})")

        agg_type = agg_config.get("type")
        agg_col_raw = agg_config.get("col")
        if isinstance(agg_col_raw, list):
            agg_col_raw = agg_col_raw[0] if agg_col_raw else ""

        if agg_type == "count1" and agg_col_raw and "|" in str(agg_col_raw):
            target_col, expr = str(agg_col_raw).split("|", 1)
            match = re.match(r"(.+?)(==|>=|<=|!=|>|<)(.+)", expr.strip())
            if match:
                c, op, v = match.groups()
                clauses.append(f"{self._col(c.strip())} {op} {self._fmt(v.strip())}")
                agg_config["col"] = target_col.strip()

        target_filter_cols = []
        if agg_type == "select":
            target_filter_cols = agg_config.get("return_cols", [])
        elif agg_type in ("listdown", "listup"):
            target_filter_cols = agg_config.get("return_cols", [])
        elif agg_type == "count1":
            col_name = agg_config.get("col")
            if col_name:
                target_filter_cols = [col_name]

        for c in target_filter_cols:
            col_sql = self._col(c)
            clauses.append(f"{col_sql} != ''")
            clauses.append(f"{col_sql} IS NOT NULL")

        where_part = "WHERE " + " AND ".join(clauses) if clauses else ""
        select_sql = ""
        limit_sql = ""
        target_col = self._col(agg_config.get("col", "*"))

        if agg_type == "sum":
            select_sql = f"SELECT ROUND(SUM({target_col}), 2)"
        elif agg_type == "avg":
            select_sql = f"SELECT ROUND(AVG({target_col}), 2)"
        elif agg_type == "count":
            select_sql = "SELECT COUNT(*)"
        elif agg_type == "count1":
            select_sql = f"SELECT COUNT(DISTINCT {target_col})"
        elif agg_type in ("listdown", "listup"):
            sort = "DESC" if agg_type == "listdown" else "ASC"
            group_cols = [self._col(c) for c in agg_config["return_cols"]]
            group_str = ", ".join(group_cols)
            inner_agg = agg_config.get("agg_type", "sum")
            if inner_agg == "count":
                select_sql = f"SELECT {group_str}, COUNT(*) as _val"
            else:
                select_sql = f"SELECT {group_str}, ROUND(SUM({self._col(agg_config['agg_col'])}), 2) as _val"
            where_part += f" GROUP BY {group_str} ORDER BY _val {sort}"
            limit_sql = f"LIMIT {agg_config['top_n']}"
        else:
            cols_raw = agg_config.get("return_cols", [])
            cols = ", ".join([self._col(c) for c in cols_raw]) if cols_raw else "*"
            select_sql = f"SELECT DISTINCT {cols}"

        return f"{select_sql} FROM {self.table_name} {where_part} {limit_sql}"
