"""
Projection functions onto convex sets.
"""

import numpy as np


def project_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Project x onto Euclidean ball B(radius) = {x: ||x||_2 <= radius}
    """
    norm = np.linalg.norm(x)
    if norm <= radius:
        return x.copy()
    return x * (radius / norm)


def project_box(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Project x onto box [lower, upper]^d
    """
    return np.clip(x, lower, upper)