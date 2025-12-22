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
    max_iter: int = 2000,
    lr: float = 0.05,
    seed: int = 999,
    tol: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Solve batch logistic regression:
    min_{||w||_2 <= B} sum_t log(1 + exp(-y_t * w^T x_t))

    Uses VECTORIZED projected gradient descent with:
    - Adaptive learning rate
    - Early stopping by gradient norm
    - Backtracking when loss increases

    Args:
        x_data: List of feature vectors
        y_data: List of labels {-1, +1}
        B: Radius constraint
        max_iter: Maximum iterations
        lr: Initial learning rate
        seed: Random seed
        tol: Gradient norm tolerance for early stopping

    Returns:
        w_opt: Optimal weight vector
        opt_loss: Optimal loss value
    """
    rng = np.random.default_rng(seed)

    # Convert to numpy arrays for vectorization
    X = np.array(x_data)  # Shape: (T, d)
    y = np.array(y_data)  # Shape: (T,)

    T, d = X.shape

    # Initialize near origin
    w = rng.standard_normal(d) * 0.01
    norm_w = np.linalg.norm(w)
    if norm_w > B:
        w = w * (B / norm_w)

    def compute_loss_vectorized(w: np.ndarray) -> float:
        """Vectorized loss computation."""
        margins = y * (X @ w)
        losses = np.where(
            margins > 0,
            np.log1p(np.exp(-margins)),
            -margins + np.log1p(np.exp(margins))
        )
        return np.sum(losses)

    def compute_grad_vectorized(w: np.ndarray) -> np.ndarray:
        """Vectorized gradient computation."""
        margins = y * (X @ w)
        sigmoid = np.where(
            margins > 0,
            1.0 / (1.0 + np.exp(-margins)),
            np.exp(margins) / (1.0 + np.exp(margins))
        )
        coeffs = y * (1.0 - sigmoid)
        grad = -X.T @ coeffs
        return grad

    def project_to_ball(w: np.ndarray) -> np.ndarray:
        """Project onto ball of radius B."""
        norm_w = np.linalg.norm(w)
        if norm_w > B:
            return w * (B / norm_w)
        return w

    # Track best solution
    best_w = w.copy()
    best_loss = compute_loss_vectorized(w)

    current_lr = lr
    no_improve_count = 0

    for iteration in range(max_iter):
        grad = compute_grad_vectorized(w)
        grad_norm = np.linalg.norm(grad)

        # Early stopping by gradient norm
        if grad_norm < tol:
            break

        # Gradient step with current lr
        w_new = project_to_ball(w - current_lr * grad)
        new_loss = compute_loss_vectorized(w_new)

        # Backtracking if loss increased
        if new_loss > best_loss:
            no_improve_count += 1
            if no_improve_count >= 10:
                current_lr *= 0.5
                no_improve_count = 0
        else:
            no_improve_count = 0

        w = w_new

        # Track best
        if new_loss < best_loss:
            best_loss = new_loss
            best_w = w.copy()

        # Scheduled lr decay
        if iteration > 0 and iteration % 500 == 0:
            current_lr *= 0.8

    return best_w, best_loss


def solve_batch_quadratic(v_list: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Solve batch quadratic problem for toy benchmark.
    min_{x in [-b,b]^d} sum_t 3*||x - v_t||^2

    Closed-form solution: project mean(v_t) onto box.

    Args:
        v_list: Array of v_t vectors, shape (T, d)
        b: Box constraint bound

    Returns:
        x_opt: Optimal point
        opt_loss: Optimal loss value
    """
    v_mean = np.mean(v_list, axis=0)
    x_opt = np.clip(v_mean, -b, b)

    # Compute optimal loss
    diff = x_opt - v_list  # Shape: (T, d)
    opt_loss = 3.0 * np.sum(diff ** 2)

    return x_opt, opt_loss