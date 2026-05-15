import asyncio

from pipeline.generator import SQLGenerator


class DummyTask:
    pass


def test_generator_path_specs_do_not_expose_thinking_path():
    generator = object.__new__(SQLGenerator)
    generator._get_top_k_examples = lambda question: ""

    specs = generator._build_path_specs(
        "查询供应商",
        "[Schema]",
        {"exact_matches": {}, "fuzzy_matches": {}},
        entities=["供应商A"],
    )

    assert set(specs) == {"icl", "direct"}
    assert "thinking" not in specs
    assert all(path_type != "thinking_path" for path_type, _, _ in specs.values())


def test_start_candidate_tasks_ignores_removed_thinking_path(monkeypatch):
    asyncio.run(_assert_start_candidate_tasks_ignore_thinking())


async def _assert_start_candidate_tasks_ignore_thinking():
    generator = object.__new__(SQLGenerator)
    generator._get_top_k_examples = lambda question: ""

    async def fake_call_llm(prompt, path_type, tracker, temperature):
        return {"type": path_type, "sql": "SELECT 1"}

    generator._call_llm = fake_call_llm

    task_map = generator.start_candidate_tasks(
        "查询供应商",
        "[Schema]",
        [],
        {"exact_matches": {}, "fuzzy_matches": {}},
        tracker=None,
        paths=("thinking", "icl", "direct"),
    )

    try:
        assert set(task_map) == {"icl", "direct"}
        assert "thinking" not in task_map
    finally:
        for tasks in task_map.values():
            for task in tasks:
                task.cancel()
        await asyncio.gather(
            *(task for tasks in task_map.values() for task in tasks),
            return_exceptions=True,
        )
