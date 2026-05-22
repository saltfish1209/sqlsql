import pandas as pd

from training.run_full_split_pipeline import normalize_and_deduplicate


def test_normalize_then_deduplicate_by_generated_question():
    df = pd.DataFrame(
        [
            {"问题模版": "{供应商描述}供应商编码是什么？", "SQL验证状态": "MATCH", "生成问题": "A"},
            {"问题模版": "{供应商描述}供应商编码是什么?", "SQL验证状态": "MATCH", "生成问题": "A"},
            {"问题模版": "{供应商描述}供应商编码是什么?", "SQL验证状态": "MATCH", "生成问题": "B"},
        ],
        dtype=str,
    )

    out, removed = normalize_and_deduplicate(df)

    assert removed == 1
    assert len(out) == 2
    assert set(out["生成问题"].tolist()) == {"A", "B"}
