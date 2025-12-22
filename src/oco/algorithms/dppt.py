"""
DPP-T: Drift-Plus-Penalty with Tightened Constraint
Same as DPP but uses g_ρ(x) = g(x) + ρ
"""

import numpy as np
from typing import Dict, Any

from .base import Algorithm


class DPPTAlgorithm(Algorithm):
    """
    DPP-T Algorithm:
    Same as DPP but with tightened constraint g_ρ(x) = g(x) + ρ
    where ρ(T) = min(ε, √(c/T)), c = 20 by default
    """

    def __init__(self, problem, T: int, config: Dict[str, Any]):
        super().__init__(problem, T, config)

        # Parameters
        self.alpha = T
        self.V = np.sqrt(T)

        # Tightening: ρ = min(ε, √(c/T))
        self.epsilon = config.get("epsilon", 0.25)
        self.c = config.get("c", 20.0)
        self.rho = min(self.epsilon, np.sqrt(self.c / T))

        # Virtual queue
        self.Q = 0.0

    @property
    def name(self) -> str:
        return "DPP-T"

    def reset(self):
        """Reset algorithm state."""
        super().reset()
        self.Q = 0.0

    def step(self) -> np.ndarray:
        """Perform one DPP-T step."""
        self.t += 1
        x_t = self.x.copy()

        # Get gradients
        grad_f = self.problem.grad_loss(x_t)
        g_t = self.problem.constraint(x_t)  # Original g(x)
        u_t = self.problem.subgrad_constraint(x_t)

        # Tightened constraint value for queue update
        g_rho_t = g_t + self.rho

        # Primal step: d_t = V∇f_t + Q_t * u_t
        d_t = self.V * grad_f + self.Q * u_t

        # Update: x_{t+1} = Π_{X_0}(x_t - d_t / (2α))
        x_next = x_t - d_t / (2 * self.alpha)
        x_next = self.problem.project_X0(x_next)

        # Queue update with tightened constraint
        # Q_{t+1} = max(Q_t + g_ρ(x_t) + u_t^T(x_{t+1} - x_t), 0)
        queue_update = self.Q + g_rho_t + np.dot(u_t, x_next - x_t)
        self.Q = max(queue_update, 0.0)

        self.x = x_next
        return x_t