from __future__ import annotations

import math


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # First, we linearly warmup for warmup_iters steps.
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Then, if it > cosine_cycle_iters, we return min learning rate.
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Else, we use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def learning_rate_schedule(lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int, current_iter: int) -> float:
    """
    Compute the learning rate based on a warmup and cosine decay schedule.

    Args:
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        warmup_iters (int): Number of iterations for linear warmup.
        cosine_cycle_iters (int): Number of iterations for one cosine cycle.
        current_iter (int): Current iteration number.

    Returns:
        float: Computed learning rate for the current iteration.
    """
    if current_iter < warmup_iters:
        return lr_max * current_iter / warmup_iters
    elif current_iter <= cosine_cycle_iters:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return lr_min + (lr_max - lr_min) * cosine_decay
    else:
        return lr_min