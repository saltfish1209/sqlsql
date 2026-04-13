import pandas as pd
import re
from collections import Counter
from datetime import datetime
import numpy as np


def check_column_exists(df, column):
    if column not in df.columns:
        raise KeyError(f"Critical Error: Column '{column}' not found in dataset.")

def _normalize_val_internal(val):
    if val is None: return ""
    try:
        f = float(val)
        f_rounded = round(f, 2)
        if f_rounded.is_integer():
            return str(int(f_rounded))
        return str(f_rounded)
    except:
        return str(val).strip()


class EnhanceDataQueryBuilder:

    def __init__(self, df_raw):
        """
        df_raw原始数据集
        """
        self.df_raw = df_raw.copy()
        self.keyword_mapping={}

    def _update_keyword_mapping(self,fields):
        """
        更新映射
        """
        for field in fields:
            if field not in self.keyword_mapping:
                self.keyword_mapping[field]=field

    def _parse_condition(self, cond_str):
        """
        解析 '采购数量 > 1000' -> ('采购数量', '>', '1000')
        支持: >, <, >=, <=, ==, !=
        """
        match = re.match(r'^\s*([^\s<>!=]+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$', cond_str)
        if not match:
            raise ValueError(f"无法解析条件表达式: '{cond_str}'")
        col, op, val = match.groups()
        return col.strip(), op.strip(), val.strip()

    def _ranked_list(self, filtered_df, agg_col, return_cols, top_n=None, ascending=False, agg_type='sum'):
        """
        按 agg_col 对 return_cols 分组求和，排序后返回前 top_n 行
        (修复版：包含计算结果列，以便与 SQL 对齐)
        """
        df = filtered_df.copy()

        # 1. 基础清洗
        for col in return_cols:
            if col not in df.columns:
                raise KeyError(f"缺少返回列: '{col}'. 现有列: {list(df.columns)}")
            df = df[df[col].notna() & (df[col].astype(str).str.strip() != '')]

        # 2. 准备聚合数据
        sort_by = None
        if agg_type == 'sum':
            if agg_col not in df.columns:
                raise KeyError(f"缺少聚合列: '{agg_col}'.")
            df[agg_col] = pd.to_numeric(df[agg_col], errors='coerce')
            df = df.dropna(subset=[agg_col])
            # 分组求和
            grouped = df.groupby(return_cols, as_index=False)[agg_col].sum()
            sort_by = agg_col
        else:  # agg_type == 'count'
            grouped = df.groupby(return_cols, as_index=False).size()
            grouped = grouped.rename(columns={'size': '_count'})
            sort_by = '_count'

        if df.empty or grouped.empty:
            return []

        # 3. 排序和截取
        grouped = grouped.sort_values(by=sort_by, ascending=ascending)

        if top_n is not None:
            grouped = grouped.head(top_n)

        # 4. 格式化输出 (关键修改点！)
        result = []
        for _, row in grouped.iterrows():
            # A. 获取分组列 (如 '供应商描述')
            item = {col: str(row[col]).strip() for col in return_cols}

            # B. 获取计算值 (如 189810844.19) 并对齐 SQL 格式
            val = row[sort_by]

            # 格式化逻辑：
            # 1. 如果是 Count，转整数字符串
            # 2. 如果是 Sum，保留小数，且去除 .0 (为了和 SQL 的 ROUND(x,2) 近似)
            #    注意：Python 的 float 精度和 SQL 可能略有不同，但通常 2 位小数能对上
            if agg_type == 'count':
                val_str = str(int(val))
            else:
                # 处理 float
                val = float(val)
                # 简单模拟 SQL 的 ROUND(x, 2)
                # 如果是 100.0 -> 100; 100.123 -> 100.12
                # 这里使用 round(val, 2)
                val_rounded = round(val, 2)
                if val_rounded.is_integer():
                    val_str = str(int(val_rounded))
                else:
                    val_str = str(val_rounded)

            # C. 将计算值也加入字典
            # construct.py 会将字典所有的 value 拼起来，所以这里加入就会变成 "供应商|金额"
            item['_generated_metric'] = val_str

            result.append(item)

        return result

    def apply_template_filters(self,base_conditions, multi_conditions):
        """
        基于问题模版进行初筛，
        base_conditions:唯一重复字段条件
        multi_conditions:字符字段条件
        """
        mask = pd.Series([True] * len(self.df_raw), index=self.df_raw.index)

        for field ,value in base_conditions.items():
            if field not in self.df_raw.columns:
                print(1)
                continue
            col_vals = self.df_raw[field].astype(str).fillna('NaN').str.strip()
            mask &= (col_vals == str(value).strip())

        for field , values in multi_conditions.items():
            if field not in self.df_raw.columns:
                print(2)
                continue
            col_vals = self.df_raw[field].astype(str).fillna('NaN').str.strip()
            mask &= col_vals.isin([str(v).strip() for v in values])

        return self.df_raw[mask].copy()

    def sum(self, filtered_df, column_name):
        """计算指定列的总和"""
        check_column_exists(filtered_df, column_name)
        numeric_series = pd.to_numeric(filtered_df[column_name], errors='coerce')
        return float(numeric_series.sum())

    def average(self, filtered_df, column_name):
        """计算指定列的平均值"""
        check_column_exists(filtered_df, column_name)
        numeric_series = pd.to_numeric(filtered_df[column_name], errors='coerce')
        return float(numeric_series.mean())

    def count(self, filtered_df, column_name=None):
        """
        计算指定列的计数总和
        column_name 可忽略，仅用于接口统一
        """
        return int(len(filtered_df))

    def count1(self, filtered_df, column_name):
        """
        计算指定列中不重复关键词的数量

        参数:
        filtered_df: 过滤后的DataFrame
        column_name: 需要统计不重复值的列名

        返回:
        指定列中不重复值的数量（忽略NaN和空字符串）
        支持两种模式：
          1. 普通: column_name = "采购订单号"
          2. 带条件: column_name = "采购订单号 | 采购数量 > 1000"
        """
        df = filtered_df.copy()

        # ===== 新增：检查是否包含条件分隔符 =====
        if '|' in column_name:
            parts = [p.strip() for p in column_name.split('|', 1)]
            if len(parts) != 2:
                raise ValueError(f"count1 条件格式错误: {column_name}")
            target_col, condition_expr = parts
        else:
            target_col = column_name
            condition_expr = None
        # ======================================

        # 列存在性检查（用目标列）
        if target_col not in df.columns:
            raise KeyError(f"列 '{target_col}' 不存在")

        # ===== 新增：应用条件过滤 =====
        if condition_expr:
            try:
                cond_col, op, val_str = self._parse_condition(condition_expr)
            except Exception as e:
                raise ValueError(f"条件解析失败: {e}")

            if cond_col not in df.columns:
                raise KeyError(f"条件列 '{cond_col}' 不存在")

            # 转数值
            df[cond_col] = pd.to_numeric(df[cond_col], errors='coerce')
            try:
                threshold = float(val_str)
            except ValueError:
                raise ValueError(f"条件值 '{val_str}' 无法转为数字")

            # 过滤
            if op == '>':
                df = df[df[cond_col] > threshold]
            elif op == '<':
                df = df[df[cond_col] < threshold]
            elif op == '>=':
                df = df[df[cond_col] >= threshold]
            elif op == '<=':
                df = df[df[cond_col] <= threshold]
            elif op == '==':
                df = df[df[cond_col] == threshold]
            elif op == '!=':
                df = df[df[cond_col] != threshold]
            else:
                raise ValueError(f"不支持的操作符: {op}")

            if df.empty:
                return 0
        # ==============================

        # 原始去重计数逻辑（作用于 target_col）
        col_data = df[target_col].dropna().astype(str).str.strip()
        col_data = col_data[col_data != ""]
        return int(col_data.nunique())




    def listdown(self, filtered_df, params):
        """降序排名"""
        return self._ranked_list(
            filtered_df,
            agg_col=params['agg_col'],
            return_cols=params['return_cols'],
            top_n=params['top_n'],
            ascending=False,
            agg_type=params.get('agg_type','sum')
        )

    def listup(self, filtered_df, params):
        """升序排名"""
        return self._ranked_list(
            filtered_df,
            agg_col=params['agg_col'],
            return_cols=params['return_cols'],
            top_n=params['top_n'],
            ascending=True,
            agg_type=params.get('agg_type', 'sum')
        )


