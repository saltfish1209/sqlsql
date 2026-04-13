import sqlite3
import pandas as pd
from utils import to_halfwidth


class DBEngine:
    def __init__(self, csv_path):
        self.conn = sqlite3.connect(':memory:')  # 在内存中创建数据库，速度快
        self.cursor = self.conn.cursor()
        self.table_name = "procurement_table"  # 设定表名
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        # 加载CSV到pandas，再导入sqlite
        df = pd.read_csv(csv_path)
        # 字符归一化'
        str_cols = df.select_dtypes(include=['object']).columns

        for col in str_cols:
            df[col] =df[col].apply(to_halfwidth)

        # 确保列名没有特殊字符干扰SQL，虽然中文列名SQLite支持，但最好加上引号
        df.to_sql(self.table_name, self.conn, index=False, if_exists='replace')
        print(f"[DB] 数据已加载，表名: {self.table_name}, 行数: {len(df)}")

    def execute_sql(self, sql):
        """
        执行SQL并返回结果。
        如果出错，捕获异常并返回错误信息，这对 Refiner 至关重要。
        """
        try:
            # 简单的安全检查，防止删除表
            if "drop" in sql.lower() or "delete" in sql.lower():
                return None, "Error: DML operations not allowed."

            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results, None  # (数据, 无错误)
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                return None, f"NO_SUCH_COLUMN: {e}"
            else:
                return None, f"SQL_ERROR: {e}"  # (无数据, 错误信息)