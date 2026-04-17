"""
独立运行 Database Profiler，预览数据库统计信息。
用法: python scripts/profile_db.py
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from pipeline.profiler import DatabaseProfiler


def main():
    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV 文件不存在: {csv_path}")
        return
    print(f"正在分析: {csv_path}\n")
    profiler = DatabaseProfiler(csv_path=csv_path)
    profiles = profiler.profile_all()

    print("=" * 70)
    print("DATABASE PROFILE SUMMARY")
    print("=" * 70)
    for p in profiles:
        print(p.to_summary())
    print("=" * 70)

    cats = profiler.get_categorical_values(profiles)
    print(f"\n共 {len(cats)} 个分类列:")
    for col, vals in cats.items():
        print(f"  {col}: {vals[:5]}{'...' if len(vals) > 5 else ''}")


if __name__ == "__main__":
    main()
