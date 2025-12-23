"""
Base class for online optimization algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class Algorithm(ABC):
    """Abstract base class for OCO algorithms."""

    def __init__(self, problem, T: int, config: Dict[str, Any]):
        """
        Args:
            problem: Problem instance with loss/constraint functions
            T: Time horizon
            config: Algorithm-specific configuration
        """
        self.problem = problem
        self.T = T
        self.config = config
        self.x = problem.initial_point()
        self.t = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for logging."""
        pass

    @abstractmethod
    def step(self) -> Tuple[np.ndarray, float]:
        """
        Perform one step of the algorithm.
        Returns (x_t, g_t): current iterate before update and constraint value.
        This ensures only ONE constraint query per round.
        """
        pass

    def reset(self):
        """Reset algorithm state for new run."""
        self.x = self.problem.initial_point()
        self.t = 0
