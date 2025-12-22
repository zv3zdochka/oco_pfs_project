"""
POGD: Projected Online Gradient Descent
Classic baseline projecting onto the true feasible set X.
"""

import numpy as np
from typing import Dict, Any

from .base import Algorithm


class POGDAlgorithm(Algorithm):
    """
    POGD Algorithm:
    x_{t+1} = Π_X(x_t - η∇f_t(x_t))
    """

    def __init__(self, problem, T: int, config: Dict[str, Any]):
        super().__init__(problem, T, config)

        # Step size: η = η_const / √T
        eta_const = config.get("eta_const", 0.2)
        self.eta = eta_const / np.sqrt(T)

    @property
    def name(self) -> str:
        return "POGD"

    def step(self) -> np.ndarray:
        """Perform one POGD step."""
        self.t += 1
        x_t = self.x.copy()

        # Gradient step
        grad_f = self.problem.grad_loss(x_t)
        x_next = x_t - self.eta * grad_f

        # Project onto true feasible set X
        self.x = self.problem.project_X(x_next)

        return x_t