"""与 generation/QA_Database_construction_template_only.py 保持一致的列类型定义。"""

NUMERIC_COLS: tuple[str, ...] = (
    "采购申请数量",
    "概算单价",
    "概算总价",
    "中标单价",
    "中标总价",
    "订单单价(含税)",
    "订单总价(含税)",
    "合同数量",
    "合同单价(含税)",
    "合同总价(含税)",
    "采购订单数量",
    "订单单价(不含税)",
    "订单总价(不含税)",
    "已付预付款金额(含税)",
    "已付到货款金额(含税)",
)

NUMERIC_COLS_SET = frozenset(NUMERIC_COLS)
