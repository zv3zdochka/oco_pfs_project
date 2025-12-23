"""
Subgradient functions for various norms.
"""

import numpy as np


def subgrad_linf_norm(x: np.ndarray) -> np.ndarray:
    """
    Subgradient of ||x||_inf.
    Returns e_i * sign(x_i) where i = argmax_j |x_j|
    """
    abs_x = np.abs(x)
    i = np.argmax(abs_x)  # Takes first index in case of ties

    u = np.zeros_like(x)
    if x[i] >= 0:
        u[i] = 1.0
    else:
        u[i] = -1.0

    # Handle x[i] = 0: use sign(0) = 1
    if x[i] == 0:
        u[i] = 1.0

    return u


def subgrad_l2_norm(x: np.ndarray) -> np.ndarray:
    """
    Subgradient of ||x||_2.
    Returns x / ||x||_2 if ||x||_2 > 0, else 0.
    """
    norm = np.linalg.norm(x)
    if norm > 1e-12:
        return x / norm
    return np.zeros_like(x)
