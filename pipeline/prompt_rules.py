from __future__ import annotations


def aggregation_rule_text() -> str:
    return (
        "聚合规则：数据库以单个订单/物资明细为一行；用户问“多少个、几个、多少种、涉及多少、"
        "一共有多少订单/供应商/物料/类别”等个数问题时，优先使用 COUNT 或 COUNT(DISTINCT 目标字段)；"
        "只有用户明确询问金额合计、总价、采购数量合计、数量总和等数值累加问题时，才使用 SUM(数值字段)。"
    )
