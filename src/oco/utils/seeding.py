"""
Seeding utilities for reproducibility.
"""

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Set global seed and return a Generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def get_trial_seed(seed_base: int, trial: int, T: int) -> int:
    """Generate reproducible seed for a specific trial."""
    return seed_base + trial * 1000 + T
