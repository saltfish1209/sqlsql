"""Helpers for deciding when to run validation during training."""


def should_run_mid_eval(step: int, steps_per_epoch: int, mid_eval_steps: int) -> bool:
    """Return whether a mid-epoch evaluation should run at current step.

    We skip very-late mid-evals that are too close to epoch end, because the
    epoch-end evaluation will run immediately afterwards.
    """
    if mid_eval_steps <= 0 or steps_per_epoch <= 0:
        return False
    if step <= 0 or step >= steps_per_epoch:
        return False
    if step % mid_eval_steps != 0:
        return False
    return (step + mid_eval_steps) <= steps_per_epoch
