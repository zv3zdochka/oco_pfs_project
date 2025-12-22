"""
Online Logistic Regression Problem (Benchmark B)
f_t(w) = log(1 + exp(-y_t * w^T x_t))
X_0 = B(R0) (Euclidean ball of radius R0)
g(w) = ||w||_2 - B
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.projections import project_ball
from ..utils.subgradients import subgrad_l2_norm


@dataclass
class OnlineLogRegProblem:
    """Online Logistic Regression with norm constraint."""

    d: int = 20
    R0: float = 5.0  # radius of X_0
    B: float = 2.0   # true constraint ||w||_2 <= B
    w_star_seed: int = 123

    def __post_init__(self):
        self.name = "online_logreg"
        self._x_t: Optional[np.ndarray] = None
        self._y_t: Optional[int] = None
        self._w_star: Optional[np.ndarray] = None
        self._initialize_w_star()

    def _initialize_w_star(self):
        """Initialize true parameter w* with ||w*||_2 = 1"""
        rng = np.random.default_rng(self.w_star_seed)
        w = rng.standard_normal(self.d)
        self._w_star = w / np.linalg.norm(w)

    def sample_loss_params(self, rng: np.random.Generator) -> None:
        """Sample (x_t, y_t) for current round."""
        self._x_t = rng.standard_normal(self.d)
        prob = 1.0 / (1.0 + np.exp(-np.dot(self._w_star, self._x_t)))
        self._y_t = 1 if rng.random() < prob else -1

    def get_data_point(self) -> Tuple[np.ndarray, int]:
        """Get current (x_t, y_t)."""
        return self._x_t.copy(), self._y_t

    def loss(self, w: np.ndarray) -> float:
        """f_t(w) = log(1 + exp(-y_t * w^T x_t))"""
        margin = self._y_t * np.dot(w, self._x_t)
        # Numerically stable computation
        if margin > 0:
            return np.log1p(np.exp(-margin))
        else:
            return -margin + np.log1p(np.exp(margin))

    def grad_loss(self, w: np.ndarray) -> np.ndarray:
        """∇f_t(w) = -y_t * x_t / (1 + exp(y_t * w^T x_t))"""
        margin = self._y_t * np.dot(w, self._x_t)
        # Numerically stable sigmoid
        if margin > 0:
            sigmoid = 1.0 / (1.0 + np.exp(-margin))
        else:
            exp_margin = np.exp(margin)
            sigmoid = exp_margin / (1.0 + exp_margin)
        return -self._y_t * self._x_t * (1.0 - sigmoid)

    def constraint(self, w: np.ndarray) -> float:
        """g(w) = ||w||_2 - B"""
        return np.linalg.norm(w) - self.B

    def subgrad_constraint(self, w: np.ndarray) -> np.ndarray:
        """Subgradient of g(w) = ||w||_2 - B"""
        return subgrad_l2_norm(w)

    def project_X0(self, w: np.ndarray) -> np.ndarray:
        """Project onto X_0 = B(R0)"""
        return project_ball(w, self.R0)

    def project_X(self, w: np.ndarray) -> np.ndarray:
        """Project onto X = B(B)"""
        return project_ball(w, self.B)

    def initial_point(self) -> np.ndarray:
        """Starting point w_1 = 0"""
        return np.zeros(self.d)