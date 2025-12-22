"""
Metrics logging utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    benchmark: str
    algo: str
    trial: int
    T: int
    t: int
    loss_t: float
    g_t: float
    viol_t: float
    cum_loss: float
    cum_viol: float
    x_coords: np.ndarray


class MetricsLogger:
    """Logger for experiment metrics."""

    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        self.step_data: List[Dict[str, Any]] = []
        self.agg_data: List[Dict[str, Any]] = []

    def log_step(self, algo: str, trial: int, T: int, t: int,
                 loss_t: float, g_t: float, x: np.ndarray,
                 cum_loss: float, cum_viol: float):
        """Log metrics for a single step."""
        viol_t = max(g_t, 0.0)

        record = {
            "benchmark": self.benchmark,
            "algo": algo,
            "trial": trial,
            "T": T,
            "t": t,
            "loss_t": loss_t,
            "g_t": g_t,
            "viol_t": viol_t,
            "cum_loss": cum_loss,
            "cum_viol": cum_viol,
        }

        # Add coordinates (for toy) or norm (for logreg)
        if len(x) <= 2:
            for i, xi in enumerate(x):
                record[f"x{i + 1}"] = xi
        else:
            record["x_norm"] = np.linalg.norm(x)

        self.step_data.append(record)

    def log_aggregate(self, algo: str, T: int, trial: int,
                      regret: float, cum_viol: float, max_viol: float,
                      cum_loss: float):
        """Log aggregate metrics for a trial."""
        self.agg_data.append({
            "benchmark": self.benchmark,
            "algo": algo,
            "T": T,
            "trial": trial,
            "regret": regret,
            "cum_viol": cum_viol,
            "max_viol": max_viol,
            "cum_loss": cum_loss,
        })

    def get_step_df(self) -> pd.DataFrame:
        """Get step-level metrics as DataFrame."""
        return pd.DataFrame(self.step_data)

    def get_agg_df(self) -> pd.DataFrame:
        """Get aggregate metrics as DataFrame."""
        return pd.DataFrame(self.agg_data)

    def compute_summary(self) -> pd.DataFrame:
        """Compute mean and std across trials."""
        df = self.get_agg_df()
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby(["benchmark", "algo", "T"]).agg({
            "regret": ["mean", "std"],
            "cum_viol": ["mean", "std"],
            "max_viol": ["mean", "std"],
            "cum_loss": ["mean", "std"],
        }).reset_index()

        # Flatten column names
        summary.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        return summary