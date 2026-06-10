"""CrossEncoder 训练早停逻辑（无重依赖，便于单测）。"""


def early_stop_after_eval(
    *,
    current_metric: float,
    best_metric: float,
    patience_counter: int,
    patience: int,
    # 向后兼容旧调用方式
    current_recall: float | None = None,
    best_recall: float | None = None,
) -> tuple[float, int, bool]:
    """val 指标未创新高则 patience_counter+1；达到 patience 则返回 should_stop=True。

    指标可以是 NDCG@K、MRR 等任何"越大越好"的值。
    """
    cur = current_metric if current_recall is None else current_recall
    best = best_metric if best_recall is None else best_recall
    if cur > best:
        return cur, 0, False
    new_counter = patience_counter + 1
    return best, new_counter, new_counter >= patience
