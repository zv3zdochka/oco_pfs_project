"""
Metrics logging utilities with streaming support for large experiments.
"""

import pandas as pd
import numpy as np
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class MetricsLogger:
    """
    Logger for experiment metrics.

    For large T (e.g., logreg with T=50000), uses:
    - Streaming writes to CSV (no memory accumulation for step data)
    - Downsampling for step-level metrics
    """

    def __init__(self, benchmark: str, output_dir: Optional[Path] = None,
                 step_subsample: int = 1):
        """
        Args:
            benchmark: Benchmark name
            output_dir: Directory for streaming output (if None, stores in memory)
            step_subsample: Log every N-th step (1 = log all, 10 = every 10th, etc.)
        """
        self.benchmark = benchmark
        self.output_dir = output_dir
        self.step_subsample = step_subsample

        # Aggregate data always stored in memory (small)
        self.agg_data: List[Dict[str, Any]] = []

        # Step data: either streaming or in-memory
        self._step_file = None
        self._step_writer = None
        self._step_header_written = False
        self._step_data_memory: List[Dict[str, Any]] = []

        if output_dir is not None:
            self._init_streaming(output_dir)

    def _init_streaming(self, output_dir: Path):
        """Initialize streaming CSV writer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._step_file = open(output_dir / "metrics_step.csv", "w", newline="")
        self._step_writer = csv.writer(self._step_file)

    def log_step(self, algo: str, trial: int, T: int, t: int,
                 loss_t: float, g_t: float, x: np.ndarray,
                 cum_loss: float, cum_viol: float):
        """Log metrics for a single step (with optional subsampling)."""

        # Subsample: only log every N-th step, but always log first and last
        if t % self.step_subsample != 0 and t != 1 and t != T:
            return

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
                record[f"x{i+1}"] = xi
        else:
            record["x_norm"] = np.linalg.norm(x)

        if self._step_writer is not None:
            # Streaming mode
            if not self._step_header_written:
                self._step_writer.writerow(record.keys())
                self._step_header_written = True
            self._step_writer.writerow(record.values())
        else:
            # In-memory mode
            self._step_data_memory.append(record)

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
        if self._step_file is not None:
            # Close file and read back
            self._step_file.close()
            self._step_file = None
            return pd.read_csv(self.output_dir / "metrics_step.csv")
        else:
            return pd.DataFrame(self._step_data_memory)

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

    def finalize(self):
        """Close any open file handles."""
        if self._step_file is not None:
            self._step_file.close()
            self._step_file = None

    def __del__(self):
        self.finalize()