"""
Base class for online optimization algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


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
    def step(self) -> np.ndarray:
        """
        Perform one step of the algorithm.
        Returns the current iterate x_t before update.
        """
        pass

    def reset(self):
        """Reset algorithm state for new run."""
        self.x = self.problem.initial_point()
        self.t = 0