"""
Plotting utilities for experiment results.
Usage: python -m oco.plot_results --input results/toy/<run_id>/

Features:
- Reads real optimal point from optimal_points.json
- Exact every-Nth-step subsampling for trajectories
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


# Style configuration
COLORS = {
    "PFS": "#2ecc71",      # Green
    "DPP": "#e74c3c",      # Red
    "DPP-T": "#3498db",    # Blue
    "POGD": "#9b59b6"      # Purple
}

MARKERS = {
    "PFS": "o",
    "DPP": "s",
    "DPP-T": "^",
    "POGD": "d"
}


def setup_plot_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "lines.linewidth": 2,
        "lines.markersize": 8,
    })


def plot_regret_vs_T(agg_df: pd.DataFrame, output_path: Path):
    """Plot A1: Regret vs Horizon T."""
    setup_plot_style()
    fig, ax = plt.subplots()

    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in agg_df["algo"].values:
            continue
        algo_data = agg_df[agg_df["algo"] == algo]
        grouped = algo_data.groupby("T")["regret"].agg(["mean", "std"])

        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            label=algo,
            color=COLORS.get(algo, "gray"),
            marker=MARKERS.get(algo, "o"),
            capsize=3
        )

    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Regret")
    ax.set_title("Regret vs Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "regret_vs_T.png", dpi=150)
    plt.close()


def plot_cumviol_vs_T(agg_df: pd.DataFrame, output_path: Path):
    """Plot A2: Cumulative Violation vs Horizon T."""
    setup_plot_style()
    fig, ax = plt.subplots()

    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in agg_df["algo"].values:
            continue
        algo_data = agg_df[agg_df["algo"] == algo]
        grouped = algo_data.groupby("T")["cum_viol"].agg(["mean", "std"])

        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            label=algo,
            color=COLORS.get(algo, "gray"),
            marker=MARKERS.get(algo, "o"),
            capsize=3
        )

    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Cumulative Violation")
    ax.set_title("Cumulative Constraint Violation vs Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "cumviol_vs_T.png", dpi=150)
    plt.close()


def plot_instviol_vs_t(step_df: pd.DataFrame, output_path: Path, T_plot: Optional[int] = None):
    """Plot A3: Instantaneous Violation vs t."""
    setup_plot_style()
    fig, ax = plt.subplots()

    if T_plot is None:
        T_plot = step_df["T"].max()

    data = step_df[step_df["T"] == T_plot]

    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in data["algo"].values:
            continue
        algo_data = data[data["algo"] == algo]
        grouped = algo_data.groupby("t")["viol_t"].agg(["mean", "std"])

        ax.plot(
            grouped.index,
            grouped["mean"],
            label=algo,
            color=COLORS.get(algo, "gray"),
            alpha=0.8
        )
        ax.fill_between(
            grouped.index,
            np.maximum(grouped["mean"] - grouped["std"], 0),
            grouped["mean"] + grouped["std"],
            color=COLORS.get(algo, "gray"),
            alpha=0.2
        )

    ax.set_xlabel("Step t")
    ax.set_ylabel("Instantaneous Violation [g(x_t)]+")
    ax.set_title(f"Instantaneous Violation vs Step (T={T_plot})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"instviol_vs_t_T{T_plot}.png", dpi=150)
    plt.close()


def plot_trajectory_2d(step_df: pd.DataFrame, output_path: Path,
                       b: float = 0.51, subsample: int = 300):
    """Plot A4: 2D trajectory plot for toy problem with REAL optimal action."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    T_max = step_df["T"].max()
    trial_1 = step_df[(step_df["T"] == T_max) & (step_df["trial"] == 1)]

    # Draw box constraint (feasible region X)
    rect = plt.Rectangle((-b, -b), 2*b, 2*b, fill=False,
                         edgecolor="black", linewidth=2, linestyle="--",
                         label="Feasible region X")
    ax.add_patch(rect)

    # Draw unit circle (X_0)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.5, label="X₀ boundary")

    # Load and plot REAL optimal action from saved file
    opt_file = output_path / "optimal_points.json"
    if opt_file.exists():
        with open(opt_file, "r") as f:
            optimal_points = json.load(f)
        key = f"T{int(T_max)}_trial1"
        if key in optimal_points:
            x_opt = np.array(optimal_points[key])
            ax.scatter(
                [x_opt[0]], [x_opt[1]],
                c="gold", marker="*", s=400,
                edgecolors="black", linewidths=2,
                label=f"x* = ({x_opt[0]:.3f}, {x_opt[1]:.3f})",
                zorder=10
            )

    # Plot trajectories with EXACT every-Nth-step subsampling
    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        algo_data = trial_1[trial_1["algo"] == algo]
        if algo_data.empty or "x1" not in algo_data.columns:
            continue

        # Sort by t and take every subsample-th step
        algo_data_sorted = algo_data.sort_values("t")
        subsampled = algo_data_sorted.iloc[::subsample]

        ax.scatter(
            subsampled["x1"],
            subsampled["x2"],
            c=COLORS.get(algo, "gray"),
            marker=MARKERS.get(algo, "o"),
            label=algo,
            alpha=0.6,
            s=30
        )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Trajectories (T={int(T_max)}, every {subsample}th action)")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "trajectory_2d.png", dpi=150)
    plt.close()


def plot_logreg_metrics(step_df: pd.DataFrame, agg_df: pd.DataFrame, output_path: Path):
    """Generate plots for logreg benchmark."""
    setup_plot_style()
    T = step_df["T"].max()
    data = step_df[step_df["T"] == T]

    # B1: Cumulative loss over time
    fig, ax = plt.subplots()

    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in data["algo"].values:
            continue
        algo_data = data[data["algo"] == algo]
        grouped = algo_data.groupby("t")["cum_loss"].agg(["mean", "std"])

        ax.plot(grouped.index, grouped["mean"], label=algo,
                color=COLORS.get(algo, "gray"))
        ax.fill_between(grouped.index,
                       grouped["mean"] - grouped["std"],
                       grouped["mean"] + grouped["std"],
                       color=COLORS.get(algo, "gray"), alpha=0.2)

    ax.set_xlabel("Step t")
    ax.set_ylabel("Cumulative Loss")
    ax.set_title("Cumulative Loss vs Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cumloss_vs_t.png", dpi=150)
    plt.close()

    # B2: Instantaneous violation
    fig, ax = plt.subplots()
    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in data["algo"].values:
            continue
        algo_data = data[data["algo"] == algo]
        grouped = algo_data.groupby("t")["viol_t"].agg(["mean", "std"])

        ax.plot(grouped.index, grouped["mean"], label=algo,
                color=COLORS.get(algo, "gray"))
        ax.fill_between(grouped.index,
                       np.maximum(grouped["mean"] - grouped["std"], 0),
                       grouped["mean"] + grouped["std"],
                       color=COLORS.get(algo, "gray"), alpha=0.2)

    ax.set_xlabel("Step t")
    ax.set_ylabel("Instantaneous Violation")
    ax.set_title("Instantaneous Violation vs Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "instviol_vs_t.png", dpi=150)
    plt.close()

    # B3: Cumulative violation
    fig, ax = plt.subplots()
    for algo in ["PFS", "DPP", "DPP-T", "POGD"]:
        if algo not in data["algo"].values:
            continue
        algo_data = data[data["algo"] == algo]
        grouped = algo_data.groupby("t")["cum_viol"].agg(["mean", "std"])

        ax.plot(grouped.index, grouped["mean"], label=algo,
                color=COLORS.get(algo, "gray"))
        ax.fill_between(grouped.index,
                       np.maximum(grouped["mean"] - grouped["std"], 0),
                       grouped["mean"] + grouped["std"],
                       color=COLORS.get(algo, "gray"), alpha=0.2)

    ax.set_xlabel("Step t")
    ax.set_ylabel("Cumulative Violation")
    ax.set_title("Cumulative Violation vs Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cumviol_vs_t.png", dpi=150)
    plt.close()

    # B4: Regret bar chart (final values)
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = agg_df.groupby("algo")["regret"].agg(["mean", "std"])

    # Order algorithms consistently
    algo_order = [a for a in ["PFS", "DPP", "DPP-T", "POGD"] if a in summary.index]
    summary = summary.loc[algo_order]

    x_pos = np.arange(len(algo_order))

    bars = ax.bar(x_pos, summary["mean"], yerr=summary["std"],
                  color=[COLORS.get(a, "gray") for a in algo_order],
                  capsize=5, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_order)
    ax.set_ylabel("Final Regret")
    ax.set_title(f"Final Regret Comparison (T={int(T)})")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "regret_comparison.png", dpi=150)
    plt.close()


def generate_all_plots(output_dir: Path, config: Dict[str, Any]):
    """Generate all plots based on benchmark type."""

    step_file = output_dir / "metrics_step.csv"
    if step_file.exists():
        step_df = pd.read_csv(step_file)
    else:
        print("Warning: metrics_step.csv not found, some plots will be skipped")
        step_df = pd.DataFrame()

    agg_file = output_dir / "metrics_agg.csv"
    if not agg_file.exists():
        print("Error: metrics_agg.csv not found")
        return

    agg_df = pd.read_csv(agg_file)

    benchmark = config["benchmark"]

    if benchmark == "toy":
        plot_regret_vs_T(agg_df, output_dir)
        plot_cumviol_vs_T(agg_df, output_dir)

        if not step_df.empty:
            T_max = int(step_df["T"].max())
            plot_instviol_vs_t(step_df, output_dir, T_plot=T_max)

            if "x1" in step_df.columns:
                b = config["problem"].get("b", 0.51)
                subsample = config.get("output", {}).get("trajectory_subsample", 300)
                plot_trajectory_2d(step_df, output_dir, b=b, subsample=subsample)

    elif benchmark == "logreg":
        if not step_df.empty:
            plot_logreg_metrics(step_df, agg_df, output_dir)
        else:
            # At least plot regret comparison from aggregates
            fig, ax = plt.subplots(figsize=(8, 5))
            summary = agg_df.groupby("algo")["regret"].agg(["mean", "std"])
            algo_order = [a for a in ["PFS", "DPP", "DPP-T", "POGD"] if a in summary.index]
            if algo_order:
                summary = summary.loc[algo_order]
                x_pos = np.arange(len(algo_order))
                ax.bar(x_pos, summary["mean"], yerr=summary["std"],
                      color=[COLORS.get(a, "gray") for a in algo_order],
                      capsize=5, alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(algo_order)
                ax.set_ylabel("Final Regret")
                ax.set_title("Final Regret Comparison")
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(output_dir / "regret_comparison.png", dpi=150)
                plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from results")
    parser.add_argument("--input", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Load config
    config_file = input_dir / "config_resolved.yaml"
    if not config_file.exists():
        print(f"Error: {config_file} not found")
        return

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    generate_all_plots(input_dir, config)


if __name__ == "__main__":
    main()