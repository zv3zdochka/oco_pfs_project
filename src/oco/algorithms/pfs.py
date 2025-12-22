"""
PFS: Online Gradient Descent + Polyak Feasibility Steps
From paper: "Constrained Online Convex Optimization with Polyak Feasibility Steps" (2025)

ВАЖНО: constraint query делается в x_t (не в y_t), затем используется
линейная аппроксимация g(y_t) ≈ g(x_t) + s_t^T(y_t - x_t)
"""

import numpy as np
from typing import Dict, Any

from .base import Algorithm
from ..utils.projections import project_ball


class PFSAlgorithm(Algorithm):
    """
    PFS Algorithm (Algorithm 1 from paper):
    1. Query constraint at x_t: g_t = g(x_t), s_t ∈ ∂g(x_t)
    2. Gradient step: y_t = x_t - η∇f_t(x_t)
    3. If linearized constraint g_t + s_t^T(y_t - x_t) + ρ > 0: Polyak step
    4. Project onto B(R)
    """

    def __init__(self, problem, T: int, config: Dict[str, Any]):
        super().__init__(problem, T, config)

        # Parameters
        self.epsilon = config.get("epsilon", 0.25)
        self.alpha = self.epsilon  # α = ε as per spec

        # ρ = min(ε, sqrt(α/T))
        self.rho = min(self.epsilon, np.sqrt(self.alpha / T))

        # η from config or default: η = ρ / (2√2) for toy, or η_const/√T for logreg
        if "eta_const" in config:
            self.eta = config["eta_const"] / np.sqrt(T)
        else:
            self.eta = self.rho / (2 * np.sqrt(2))

        # Radius for projection
        self.R = getattr(problem, 'R', getattr(problem, 'R0', 1.0))

    @property
    def name(self) -> str:
        return "PFS"

    def step(self) -> np.ndarray:
        """Perform one PFS step as in Algorithm 1."""
        self.t += 1
        x_t = self.x.copy()

        # 1. Query constraint at x_t (NOT at y_t!)
        g_t = self.problem.constraint(x_t)
        s_t = self.problem.subgrad_constraint(x_t)

        # 2. Get gradient of loss and do gradient step
        grad_f = self.problem.grad_loss(x_t)
        y_t = x_t - self.eta * grad_f

        # 3. Compute linearized constraint: g_t + s_t^T(y_t - x_t) + ρ
        linearized = g_t + np.dot(s_t, y_t - x_t) + self.rho

        if linearized > 0:
            s_norm_sq = np.dot(s_t, s_t)

            if s_norm_sq > 1e-12:
                # Polyak feasibility step
                step_size = linearized / s_norm_sq
                y_t = y_t - step_size * s_t

        # 4. Project onto B(R)
        self.x = project_ball(y_t, self.R)

        return x_t