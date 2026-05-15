import asyncio

from pipeline.entity_extractor import EntityExtractor


class LLMOnlyExtractor(EntityExtractor):
    def __init__(self):
        super().__init__(client=None, model="dummy")

    async def _llm_extract(self, question, schema_text, tracker, max_retries):
        return ["ABC123", "不存在实体", "供应商编码", "JSON 输出", "ABC123"]


def test_entity_extractor_returns_only_filtered_llm_entities():
    asyncio.run(_assert_entity_extractor_returns_only_filtered_llm_entities())


async def _assert_entity_extractor_returns_only_filtered_llm_entities():
    extractor = LLMOnlyExtractor()

    entities = await extractor.extract(
        "请查询ABC123对应的供应商编码",
        "供应商编码：供应商编码",
        schema_columns=["供应商编码"],
    )

    assert entities == ["ABC123"]


def test_rule_extract_api_is_removed_from_entity_extractor():
    assert not hasattr(EntityExtractor, "_rule_extract")
