"""
Main experiment runner.
Usage: python -m oco.run_experiment --config configs/toy.yaml

Исправлено:
- Данные генерируются один раз для всех алгоритмов
- Стриминг логов для экономии памяти
- Правильный подсчёт progress bar
"""

import argparse
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from .problems import ToyQuadraticProblem, OnlineLogRegProblem
from .algorithms import PFSAlgorithm, DPPAlgorithm, DPPTAlgorithm, POGDAlgorithm
from .utils.logging import MetricsLogger
from .utils.seeding import get_trial_seed
from .utils.batch_opt import solve_batch_logreg, solve_batch_quadratic


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_problem(config: Dict[str, Any]):
    """Create problem instance from config."""
    prob_cfg = config["problem"]

    if prob_cfg["type"] == "toy_quadratic":
        return ToyQuadraticProblem(
            d=prob_cfg.get("d", 2),
            R=prob_cfg.get("R", 1.0),
            b=prob_cfg.get("b", 0.51)
        )
    elif prob_cfg["type"] == "online_logreg":
        return OnlineLogRegProblem(
            d=prob_cfg.get("d", 20),
            R0=prob_cfg.get("R0", 5.0),
            B=prob_cfg.get("B", 2.0),
            w_star_seed=prob_cfg.get("w_star_seed", 123)
        )
    else:
        raise ValueError(f"Unknown problem type: {prob_cfg['type']}")


def create_algorithms(problem, T: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create algorithm instances."""
    algo_configs = config["algorithms"]
    algorithms = {}

    if "PFS" in algo_configs:
        algorithms["PFS"] = PFSAlgorithm(problem, T, algo_configs["PFS"])

    if "DPP" in algo_configs:
        algorithms["DPP"] = DPPAlgorithm(problem, T, algo_configs["DPP"])

    if "DPP-T" in algo_configs:
        algorithms["DPP-T"] = DPPTAlgorithm(problem, T, algo_configs["DPP-T"])

    if "POGD" in algo_configs:
        algorithms["POGD"] = POGDAlgorithm(problem, T, algo_configs["POGD"])

    return algorithms


def pregenerate_data(problem, T: int, seed: int) -> Dict[str, Any]:
    """
    Pre-generate all random data for a trial.
    This ensures all algorithms see exactly the same sequence.
    """
    rng = np.random.default_rng(seed)

    data = {"T": T}

    if isinstance(problem, ToyQuadraticProblem):
        # Generate all v_t upfront
        data["v_list"] = rng.uniform(0, 1, size=(T, problem.d))

    elif isinstance(problem, OnlineLogRegProblem):
        # Generate all (x_t, y_t) upfront
        x_list = []
        y_list = []
        for _ in range(T):
            x_t = rng.standard_normal(problem.d)
            prob = 1.0 / (1.0 + np.exp(-np.dot(problem._w_star, x_t)))
            y_t = 1 if rng.random() < prob else -1
            x_list.append(x_t)
            y_list.append(y_t)

        data["x_list"] = x_list
        data["y_list"] = y_list

    return data


def set_problem_data_for_step(problem, data: Dict[str, Any], t: int):
    """Set problem's internal state to use pre-generated data for step t."""
    idx = t - 1  # t is 1-indexed

    if isinstance(problem, ToyQuadraticProblem):
        problem._v_t = data["v_list"][idx]

    elif isinstance(problem, OnlineLogRegProblem):
        problem._x_t = data["x_list"][idx]
        problem._y_t = data["y_list"][idx]


def compute_optimal_loss(problem, data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Compute optimal batch loss for regret calculation."""

    if isinstance(problem, ToyQuadraticProblem):
        _, opt_loss = solve_batch_quadratic(data["v_list"], problem.b)
        return opt_loss

    elif isinstance(problem, OnlineLogRegProblem):
        batch_cfg = config.get("batch_solver", {})
        _, opt_loss = solve_batch_logreg(
            data["x_list"],
            data["y_list"],
            B=problem.B,
            max_iter=batch_cfg.get("max_iter", 500),
            lr=batch_cfg.get("lr", 0.1),
            seed=batch_cfg.get("seed", 999)
        )
        return opt_loss

    return 0.0


def run_single_algo_trial(
    problem,
    algo,
    algo_name: str,
    data: Dict[str, Any],
    T: int,
    trial: int,
    logger: MetricsLogger,
    opt_loss: float
) -> Dict[str, float]:
    """Run a single algorithm for one trial."""

    algo.reset()

    cum_loss = 0.0
    cum_viol = 0.0
    max_viol = 0.0

    for t in range(1, T + 1):
        # Set problem data for this step
        set_problem_data_for_step(problem, data, t)

        # Algorithm step (returns x_t before update)
        x_t = algo.step()

        # Compute metrics at x_t
        loss_t = problem.loss(x_t)
        g_t = problem.constraint(x_t)
        viol_t = max(g_t, 0.0)

        cum_loss += loss_t
        cum_viol += viol_t
        max_viol = max(max_viol, viol_t)

        # Log step
        logger.log_step(
            algo=algo_name,
            trial=trial,
            T=T,
            t=t,
            loss_t=loss_t,
            g_t=g_t,
            x=x_t,
            cum_loss=cum_loss,
            cum_viol=cum_viol
        )

    # Compute regret
    regret = cum_loss - opt_loss

    # Log aggregate
    logger.log_aggregate(
        algo=algo_name,
        T=T,
        trial=trial,
        regret=regret,
        cum_viol=cum_viol,
        max_viol=max_viol,
        cum_loss=cum_loss
    )

    return {
        "cum_loss": cum_loss,
        "cum_viol": cum_viol,
        "max_viol": max_viol,
        "regret": regret
    }


def run_experiment(config: Dict[str, Any], output_dir: Path):
    """Run full experiment."""

    benchmark = config["benchmark"]
    exp_cfg = config["experiment"]
    horizons = exp_cfg["horizons"]
    trials = exp_cfg["trials"]
    seed_base = exp_cfg["seed_base"]

    # Determine step subsampling for large experiments
    max_T = max(horizons)
    step_subsample = 1 if max_T <= 20000 else max(1, max_T // 5000)

    # Initialize logger with streaming
    logger = MetricsLogger(
        benchmark=benchmark,
        output_dir=output_dir,
        step_subsample=step_subsample
    )

    problem = create_problem(config)

    # Calculate total runs for progress bar
    num_algorithms = len(config["algorithms"])
    total_runs = len(horizons) * trials * num_algorithms
    pbar = tqdm(total=total_runs, desc=f"Running {benchmark}")

    for T in horizons:
        algorithms = create_algorithms(problem, T, config)

        for trial in range(1, trials + 1):
            seed = get_trial_seed(seed_base, trial, T)

            # Pre-generate data ONCE for this trial
            data = pregenerate_data(problem, T, seed)

            # Compute optimal loss ONCE for this trial
            opt_loss = compute_optimal_loss(problem, data, config)

            # Run all algorithms on the SAME data
            for algo_name, algo in algorithms.items():
                run_single_algo_trial(
                    problem=problem,
                    algo=algo,
                    algo_name=algo_name,
                    data=data,
                    T=T,
                    trial=trial,
                    logger=logger,
                    opt_loss=opt_loss
                )
                pbar.update(1)

    pbar.close()

    # Finalize logger
    logger.finalize()

    # Save aggregates
    agg_df = logger.get_agg_df()
    summary_df = logger.compute_summary()

    agg_df.to_csv(output_dir / "metrics_agg.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    # Save resolved config
    with open(output_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\nResults saved to {output_dir}")

    # Generate plots
    from .plot_results import generate_all_plots
    generate_all_plots(output_dir, config)


def main():
    parser = argparse.ArgumentParser(description="Run OCO experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / config["benchmark"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(config, output_dir)


if __name__ == "__main__":
    main()