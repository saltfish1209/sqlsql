import pandas as pd

from training.prepare_data import process_data, resolve_active_columns


def test_resolve_active_columns_handles_full_halfwidth_parenthesis():
    schema_columns = ["订单总价(不含税）", "订单单价(不含税)"]
    raw_columns = ["订单总价(不含税)", "订单单价(不含税)"]
    profile_detail_map = {
        "订单总价(不含税)": {"空值率": "10%"},
        "订单单价(不含税)": {"空值率": "10%"},
    }

    active, report = resolve_active_columns(
        schema_columns=schema_columns,
        raw_columns=raw_columns,
        profile_detail_map=profile_detail_map,
        null_ratio_threshold=0.95,
    )

    assert "订单总价(不含税)" in active
    assert report["deprecated"]["schema_not_in_raw"] == []


def test_process_data_normalizes_template_column_names_against_passage_map():
    df = pd.DataFrame([
        {
            "生成问题": "订单总价是多少",
            "问题模版": "{订单总价(不含税）}",
            "回答模版": "{订单总价(不含税）}",
        }
    ])
    all_cols = ["订单总价(不含税)"]
    passage_map = {"订单总价(不含税)": "列名称: 订单总价(不含税)"}

    rows = process_data(df, all_cols, passage_map, is_training=True)

    positives = [r for r in rows if r.get("label") == 1]
    assert len(positives) == 1
    assert positives[0]["column_name"] == "订单总价(不含税)"


def test_resolve_active_columns_handles_nfkc_and_nbsp_variants():
    # schema 里混入全角括号、非断空格等变体，raw 列名为标准写法
    schema_columns = ["订单总价（不含税）\u00a0"]
    raw_columns = ["订单总价(不含税)"]
    profile_detail_map = {"订单总价(不含税)": {"空值率": "0%"}}

    active, report = resolve_active_columns(
        schema_columns=schema_columns,
        raw_columns=raw_columns,
        profile_detail_map=profile_detail_map,
        null_ratio_threshold=0.95,
    )

    assert "订单总价(不含税)" in active
    assert report["deprecated"]["schema_not_in_raw"] == []
