import re


class SQLQueryBuilder:
    def __init__(self, table_name="procurement_table"):
        self.table_name = table_name

    def _fmt(self, val):
        # 1. 如果是纯整数（int），直接返回
        if isinstance(val, int):
            return str(val)

        # 2. 如果是浮点数（float）
        if isinstance(val, float):
            # 检查是否是 450.0 这种“伪浮点数”
            if val.is_integer():
                return str(int(val))  # 变成 "450"
            return str(val)  # 变成 "450.5"

        # 3. 字符串处理
        val_str = str(val).strip()
        # 再次尝试检测字符串里的 ".0" (针对 Pandas 读取带来的 string "123.0")
        if val_str.endswith(".0") and val_str[:-2].isdigit():
            val_str = val_str[:-2]

        # SQL 转义，将一个单引号替换为两个单引号
        val_str = val_str.replace("'", "''")

        return f"'{val_str}'"



    def _col(self, col):
        """给列名加双引号，处理特殊字符，同时防御列表类型的输入"""
        if isinstance(col, list):
            col = col[0] if col else ""
        return f'"{str(col).strip()}"'

    def generate(self, base_conditions, multi_conditions, agg_config):
        """
        核心生成函数 - 修复了 count1 空字符串计数差异问题
        """
        # 1. 构建 WHERE 子句的基础部分
        clauses = []
        for col, val in base_conditions.items():
            clauses.append(f"{self._col(col)} = {self._fmt(val)}")

        for col, vals in multi_conditions.items():
            val_str = ", ".join([self._fmt(v) for v in vals])
            clauses.append(f"{self._col(col)} IN ({val_str})")

        agg_type = agg_config.get('type')
        agg_col_raw = agg_config.get('col')

        # --- 特殊处理 count1 的条件表达式 (如: 采购数量 > 1000) ---
        # 必须在确定 target_filter_cols 之前处理，以确保 agg_config['col'] 是纯净的列名
        if isinstance(agg_col_raw, list):
            agg_col_raw = agg_col_raw[0] if agg_col_raw else ""

        if agg_type == 'count1' and agg_col_raw and '|' in str(agg_col_raw):
            target_col, expr = str(agg_col_raw).split('|', 1)
            # 把长的操作符放在前面，避免 < 抢占了 <=
            match = re.match(r'(.+?)(==|>=|<=|!=|>|<)(.+)', expr.strip())
            if match:
                c, op, v = match.groups()
                clauses.append(f"{self._col(c.strip())} {op} {self._fmt(v.strip())}")
                # 更新为纯列名，供后续使用
                agg_config['col'] = target_col.strip()

                # --- 关键修复：确定需要排除空值的列 ---
        target_filter_cols = []

        if agg_type == 'select':
            target_filter_cols = agg_config.get('return_cols', [])
        elif agg_type in ['listdown', 'listup']:
            target_filter_cols = agg_config.get('return_cols', [])
        elif agg_type == 'count1':
            col_name = agg_config.get('col')
            if col_name:
                target_filter_cols = [col_name]

        # 将排除空值的条件加入 WHERE 子句
        for c in target_filter_cols:
            col_sql = self._col(c)
            # 既不为空串，也不是 NULL (虽然 SQLite 中空串和 NULL 是两码事，但加上更保险)
            clauses.append(f"{col_sql} != ''")
            clauses.append(f"{col_sql} IS NOT NULL")

        where_part = "WHERE " + " AND ".join(clauses) if clauses else ""

        # 2. 构建 SELECT 和 LIMIT 子句
        select_sql = ""
        limit_sql = ""

        target_col = self._col(agg_config.get('col', '*'))

        if agg_type == 'sum':
            select_sql = f"SELECT ROUND(SUM({target_col}), 2)"
        elif agg_type == 'avg':
            select_sql = f"SELECT ROUND(AVG({target_col}), 2)"
        elif agg_type == 'count':
            select_sql = "SELECT COUNT(*)"
        elif agg_type == 'count1':
            select_sql = f"SELECT COUNT(DISTINCT {target_col})"
        elif agg_type in ['listdown', 'listup']:
            sort = "DESC" if agg_type == 'listdown' else "ASC"
            group_cols = [self._col(c) for c in agg_config['return_cols']]
            group_str = ", ".join(group_cols)

            inner_agg = agg_config.get('agg_type', 'sum')
            if inner_agg == 'count':
                select_sql = f"SELECT {group_str}, COUNT(*) as _val"
            else:
                select_sql = f"SELECT {group_str}, ROUND(SUM({self._col(agg_config['agg_col'])}), 2) as _val"

            where_part += f" GROUP BY {group_str} ORDER BY _val {sort}"
            limit_sql = f"LIMIT {agg_config['top_n']}"
        else:
            cols_raw = agg_config.get('return_cols', [])
            if not cols_raw:
                cols = "*"
            else:
                cols = ", ".join([self._col(c) for c in cols_raw])
            select_sql = f"SELECT DISTINCT {cols}"
            limit_sql = ""

        # 3. 最终拼接
        return f"{select_sql} FROM {self.table_name} {where_part} {limit_sql}"