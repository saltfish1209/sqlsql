"""
数据库引擎 —— 在内存 SQLite 上加载 CSV 数据并提供安全的 SQL 执行接口。
"""
from __future__ import annotations

import os
import sqlite3
import pandas as pd

from pipeline.utils import to_halfwidth


class DBEngine:
    def __init__(self, csv_path: str, table_name: str = "procurement_table"):
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self.table_name = table_name
        csv_path = str(csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].apply(to_halfwidth)
        df.to_sql(self.table_name, self.conn, index=False, if_exists="replace")
        print(f"[DB] 数据已加载，表名: {self.table_name}, 行数: {len(df)}")

    @staticmethod
    def _extract_first_select(sql: str) -> str:
        """从可能包含多条语句 / 注释的文本中提取第一条 SELECT。"""
        sql = "\n".join(
            line for line in sql.strip().splitlines()
            if not line.strip().startswith("--")
        )
        for part in sql.split(";"):
            part = part.strip()
            if part and part.upper().lstrip().startswith("SELECT"):
                return part
        return sql.rstrip(";").strip()

    def execute_sql(self, sql: str):
        """执行 SQL 并返回 (rows, error)；error 为 None 表示成功。"""
        try:
            lower = sql.lower()
            if "drop" in lower or "delete" in lower or "update" in lower:
                return None, "Error: DML/DDL operations not allowed."
            clean_sql = self._extract_first_select(sql)
            self.cursor.execute(clean_sql)
            return self.cursor.fetchall(), None
        except sqlite3.OperationalError as e:
            tag = "NO_SUCH_COLUMN" if "no such column" in str(e) else "SQL_ERROR"
            return None, f"{tag}: {e}"
        except Exception as e:
            return None, f"UNEXPECTED: {e}"

    def check_literal_in_column(self, column: str, literal: str) -> bool:
        """论文 Literal-Column 校验：检查 WHERE 字面量是否真的存在于该列。"""
        try:
            self.cursor.execute(
                f'SELECT 1 FROM "{self.table_name}" WHERE "{column}" = ? LIMIT 1',
                (literal,),
            )
            return self.cursor.fetchone() is not None
        except Exception:
            return False

    def get_column_distinct_values(self, column: str, limit: int = 20) -> list[str]:
        """获取某列的去重值样本，用于 Profiling。"""
        try:
            self.cursor.execute(
                f'SELECT DISTINCT "{column}" FROM "{self.table_name}" '
                f'WHERE "{column}" IS NOT NULL AND "{column}" != \'\' '
                f'LIMIT ?',
                (limit,),
            )
            return [str(row[0]) for row in self.cursor.fetchall()]
        except Exception:
            return []

    def close(self):
        self.conn.close()
