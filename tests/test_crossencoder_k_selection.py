from figure.crossencoder.k_selection import pick_top_k_from_curve


def test_pick_top_k_elbow():
    curve = [
        {"top_k": 6, "full_recall": 0.70},
        {"top_k": 10, "full_recall": 0.90},
        {"top_k": 15, "full_recall": 0.95},
        {"top_k": 20, "full_recall": 0.96},
    ]
    pick = pick_top_k_from_curve(curve, gold_p95=4, min_k=6, recall_ratio_of_max=0.95)
    # max=0.96, target=0.912; K>=6 中第一个达标是 K=15 (0.95)
    assert pick["chosen_k"] == 15


def test_pick_top_k_fallback_k_floor():
    curve = [{"top_k": 3, "full_recall": 0.5}, {"top_k": 8, "full_recall": 0.8}]
    pick = pick_top_k_from_curve(curve, gold_p95=6, min_k=6)
    assert pick["chosen_k"] == 8
