"""CrossEncoder 训练早停逻辑（无重依赖，便于单测）。"""


def early_stop_after_eval(
    *,
    current_recall: float,
    best_recall: float,
    patience_counter: int,
    patience: int,
) -> tuple[float, int, bool]:
    """val 未创新高则 patience_counter+1；达到 patience 则返回 should_stop=True。"""
    if current_recall > best_recall:
        return current_recall, 0, False
    new_counter = patience_counter + 1
    return best_recall, new_counter, new_counter >= patience
