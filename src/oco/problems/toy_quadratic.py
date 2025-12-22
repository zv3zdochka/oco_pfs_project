"""
Toy Quadratic Problem (Benchmark A)
f_t(x) = 3 * ||x - v_t||^2, v_t ~ Unif([0,1]^d)
X_0 = B(R) (Euclidean ball)
g(x) = ||x||_inf - b
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.projections import project_ball, project_box
from ..utils.subgradients import subgrad_linf_norm


@dataclass
class ToyQuadraticProblem:
    """Toy quadratic problem from PFS paper 2025."""

    d: int = 2
    R: float = 1.0  # radius of X_0
    b: float = 0.51  # box constraint ||x||_inf <= b

    def __post_init__(self):
        self.name = "toy_quadratic"
        self._v_t: Optional[np.ndarray] = None

    def sample_loss_params(self, rng: np.random.Generator) -> None:
        """Sample v_t for current round."""
        self._v_t = rng.uniform(0, 1, size=self.d)

    def get_v_t(self) -> np.ndarray:
        """Get current v_t."""
        return self._v_t.copy()

    def loss(self, x: np.ndarray) -> float:
        """f_t(x) = 3 * ||x - v_t||^2"""
        diff = x - self._v_t
        return 3.0 * np.dot(diff, diff)

    def grad_loss(self, x: np.ndarray) -> np.ndarray:
        """∇f_t(x) = 6 * (x - v_t)"""
        return 6.0 * (x - self._v_t)

    def constraint(self, x: np.ndarray) -> float:
        """g(x) = ||x||_inf - b"""
        return np.max(np.abs(x)) - self.b

    def subgrad_constraint(self, x: np.ndarray) -> np.ndarray:
        """Subgradient of g(x) = ||x||_inf - b"""
        return subgrad_linf_norm(x)

    def project_X0(self, x: np.ndarray) -> np.ndarray:
        """Project onto X_0 = B(R)"""
        return project_ball(x, self.R)

    def project_X(self, x: np.ndarray) -> np.ndarray:
        """Project onto X = [-b, b]^d (box)"""
        return project_box(x, -self.b, self.b)

    def initial_point(self) -> np.ndarray:
        """Starting point x_1 = 0"""
        return np.zeros(self.d)