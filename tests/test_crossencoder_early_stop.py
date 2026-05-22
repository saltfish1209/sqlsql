from training.cross_encoder_early_stop import early_stop_after_eval


def test_early_stop_patience_one_no_improvement():
    best, counter, stop = early_stop_after_eval(
        current_recall=0.8,
        best_recall=0.85,
        patience_counter=0,
        patience=1,
    )
    assert best == 0.85
    assert counter == 1
    assert stop is True


def test_early_stop_resets_on_improvement():
    best, counter, stop = early_stop_after_eval(
        current_recall=0.9,
        best_recall=0.85,
        patience_counter=1,
        patience=1,
    )
    assert best == 0.9
    assert counter == 0
    assert stop is False


def test_early_stop_patience_two_requires_two_epochs():
    _, counter, stop = early_stop_after_eval(
        current_recall=0.8,
        best_recall=0.85,
        patience_counter=0,
        patience=2,
    )
    assert counter == 1
    assert stop is False

    _, counter, stop = early_stop_after_eval(
        current_recall=0.79,
        best_recall=0.85,
        patience_counter=1,
        patience=2,
    )
    assert counter == 2
    assert stop is True
