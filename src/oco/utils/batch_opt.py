"""
Batch optimization solver for computing regret baseline.
"""

import numpy as np
from typing import Tuple, List


def solve_batch_logreg(
        x_data: List[np.ndarray],
        y_data: List[int],
        B: float,
        max_iter: int = 5000,
        lr: float = 0.01,
        seed: int = 999
) -> Tuple[np.ndarray, float]:
    """
    Solve batch logistic regression:
    min_{||w||_2 <= B} sum_t log(1 + exp(-y_t * w^T x_t))

    Uses projected gradient descent.

    Returns:
        w_opt: Optimal weight vector
        opt_loss: Optimal loss value
    """
    rng = np.random.default_rng(seed)
    d = x_data[0].shape[0]
    T = len(x_data)

    # Initialize
    w = rng.standard_normal(d) * 0.01
    norm_w = np.linalg.norm(w)
    if norm_w > B:
        w = w * (B / norm_w)

    def compute_loss(w):
        total = 0.0
        for x_t, y_t in zip(x_data, y_data):
            margin = y_t * np.dot(w, x_t)
            if margin > 0:
                total += np.log1p(np.exp(-margin))
            else:
                total += -margin + np.log1p(np.exp(margin))
        return total

    def compute_grad(w):
        grad = np.zeros(d)
        for x_t, y_t in zip(x_data, y_data):
            margin = y_t * np.dot(w, x_t)
            if margin > 0:
                sigmoid = 1.0 / (1.0 + np.exp(-margin))
            else:
                exp_m = np.exp(margin)
                sigmoid = exp_m / (1.0 + exp_m)
            grad += -y_t * x_t * (1.0 - sigmoid)
        return grad

    # Projected GD
    for _ in range(max_iter):
        grad = compute_grad(w)
        w = w - lr * grad

        # Project onto ball
        norm_w = np.linalg.norm(w)
        if norm_w > B:
            w = w * (B / norm_w)

    opt_loss = compute_loss(w)
    return w.copy(), opt_loss
