"""
Batch optimization solver for computing regret baseline.
Vectorized implementation for efficiency.
"""

import numpy as np
from typing import Tuple, List


def solve_batch_logreg(
    x_data: List[np.ndarray],
    y_data: List[int],
    B: float,
    max_iter: int = 500,
    lr: float = 0.1,
    seed: int = 999
) -> Tuple[np.ndarray, float]:
    """
    Solve batch logistic regression:
    min_{||w||_2 <= B} sum_t log(1 + exp(-y_t * w^T x_t))

    Uses VECTORIZED projected gradient descent.

    Args:
        x_data: List of feature vectors (T, d)
        y_data: List of labels {-1, +1}
        B: Radius constraint
        max_iter: Number of GD iterations (reduced due to vectorization efficiency)
        lr: Learning rate
        seed: Random seed

    Returns:
        w_opt: Optimal weight vector
        opt_loss: Optimal loss value
    """
    rng = np.random.default_rng(seed)

    # Convert to numpy arrays for vectorization
    X = np.array(x_data)  # Shape: (T, d)
    y = np.array(y_data)  # Shape: (T,)

    T, d = X.shape

    # Initialize
    w = rng.standard_normal(d) * 0.01
    norm_w = np.linalg.norm(w)
    if norm_w > B:
        w = w * (B / norm_w)

    def compute_loss_vectorized(w: np.ndarray) -> float:
        """Vectorized loss computation."""
        margins = y * (X @ w)  # Shape: (T,)
        # Numerically stable log(1 + exp(-margin))
        losses = np.where(
            margins > 0,
            np.log1p(np.exp(-margins)),
            -margins + np.log1p(np.exp(margins))
        )
        return np.sum(losses)

    def compute_grad_vectorized(w: np.ndarray) -> np.ndarray:
        """Vectorized gradient computation."""
        margins = y * (X @ w)  # Shape: (T,)

        # Numerically stable sigmoid
        sigmoid = np.where(
            margins > 0,
            1.0 / (1.0 + np.exp(-margins)),
            np.exp(margins) / (1.0 + np.exp(margins))
        )

        # ∇f = sum_t (-y_t * x_t * (1 - sigmoid_t))
        # = -X^T @ (y * (1 - sigmoid))
        coeffs = y * (1.0 - sigmoid)  # Shape: (T,)
        grad = -X.T @ coeffs  # Shape: (d,)

        return grad

    # Projected GD with adaptive step size
    best_w = w.copy()
    best_loss = compute_loss_vectorized(w)

    for iteration in range(max_iter):
        grad = compute_grad_vectorized(w)

        # Gradient step
        w = w - lr * grad

        # Project onto ball
        norm_w = np.linalg.norm(w)
        if norm_w > B:
            w = w * (B / norm_w)

        # Track best
        current_loss = compute_loss_vectorized(w)
        if current_loss < best_loss:
            best_loss = current_loss
            best_w = w.copy()

        # Simple adaptive lr (optional, helps convergence)
        if iteration > 0 and iteration % 100 == 0:
            lr *= 0.9

    return best_w, best_loss


def solve_batch_quadratic(v_list: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Solve batch quadratic problem for toy benchmark.
    min_{x in [-b,b]^d} sum_t 3*||x - v_t||^2

    Closed-form solution: project mean(v_t) onto box.
    """
    v_mean = np.mean(v_list, axis=0)
    x_opt = np.clip(v_mean, -b, b)

    # Compute optimal loss
    diff = x_opt - v_list  # Shape: (T, d)
    opt_loss = 3.0 * np.sum(diff ** 2)

    return x_opt, opt_loss