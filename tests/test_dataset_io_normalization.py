from training.dataset_io import (
    normalize_and_deduplicate_dataframe,
    normalize_cell_text,
    normalize_dataframe,
)


def test_normalize_cell_text_converts_fullwidth_punctuation():
    assert normalize_cell_text("订单总价(不含税）") == "订单总价(不含税)"
    assert normalize_cell_text("ＡＢＣ，１２３") == "ABC,123"


def test_normalize_dataframe_applies_to_columns_and_cells():
    import pandas as pd

    df = pd.DataFrame(
        {"列名（测试）": ["值Ａ", "值Ｂ"]},
        dtype=str,
    )
    out = normalize_dataframe(df)
    assert list(out.columns) == ["列名(测试)"]
    assert out.iloc[0, 0] == "值A"
    assert out.iloc[1, 0] == "值B"


def test_normalize_and_deduplicate_dataframe_with_subset():
    import pandas as pd

    df = pd.DataFrame(
        {
            "问题模版": ["{供应商描述}供应商编码是什么？", "{供应商描述}供应商编码是什么?"],
            "回答模版": ["{供应商编码}", "{供应商编码}"],
            "生成问题": ["a", "b"],
        },
        dtype=str,
    )
    out, removed = normalize_and_deduplicate_dataframe(df, subset=["问题模版", "回答模版"])
    assert removed == 1
    assert len(out) == 1
