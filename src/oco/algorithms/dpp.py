"""
DPP: Drift-Plus-Penalty Algorithm
From: Yu et al., 2017 "Online Convex Optimization with Stochastic Constraints"
"""

import numpy as np
from typing import Dict, Any, Tuple

from .base import Algorithm


class DPPAlgorithm(Algorithm):
    """
    DPP Algorithm with virtual queue:
    1. d_t = V∇f_t(x_t) + Q_t * u_t
    2. x_{t+1} = Π_{X_0}(x_t - d_t / (2α))
    3. Q_{t+1} = max(Q_t + g(x_t) + u_t^T(x_{t+1} - x_t), 0)
    """

    def __init__(self, problem, T: int, config: Dict[str, Any]):
        super().__init__(problem, T, config)

        # Parameters: α = T, V = √T
        self.alpha = T
        self.V = np.sqrt(T)

        # Virtual queue (scalar for single constraint)
        self.Q = 0.0

    @property
    def name(self) -> str:
        return "DPP"

    def reset(self):
        """Reset algorithm state."""
        super().reset()
        self.Q = 0.0

    def step(self) -> Tuple[np.ndarray, float]:
        """Perform one DPP step."""
        self.t += 1
        x_t = self.x.copy()

        # Get gradients - ONE constraint query
        grad_f = self.problem.grad_loss(x_t)
        g_t = self.problem.constraint(x_t)
        u_t = self.problem.subgrad_constraint(x_t)

        # Primal step: d_t = V∇f_t + Q_t * u_t
        d_t = self.V * grad_f + self.Q * u_t

        # Update: x_{t+1} = Π_{X_0}(x_t - d_t / (2α))
        x_next = x_t - d_t / (2 * self.alpha)
        x_next = self.problem.project_X0(x_next)

        # Queue update: Q_{t+1} = max(Q_t + g(x_t) + u_t^T(x_{t+1} - x_t), 0)
        queue_update = self.Q + g_t + np.dot(u_t, x_next - x_t)
        self.Q = max(queue_update, 0.0)

        self.x = x_next
        return x_t, g_t