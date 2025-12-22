"""
Main experiment runner.
Usage: python -m oco.run_experiment --config configs/toy.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from .problems import ToyQuadraticProblem, OnlineLogRegProblem
from .algorithms import PFSAlgorithm, DPPAlgorithm, DPPTAlgorithm, POGDAlgorithm
from .utils.logging import MetricsLogger
from .utils.seeding import get_trial_seed
from .utils.batch_opt import solve_batch_logreg


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


def run_single_trial(
        problem,
        algorithms: Dict[str, Any],
        T: int,
        trial: int,
        seed: int,
        logger: MetricsLogger,
        save_data: bool = False
) -> Dict[str, Tuple[List, List]]:
    """Run a single trial for all algorithms."""

    # Data storage for batch optimization (logreg)
    data_points = {"x": [], "y": []}
    v_list = []  # For toy problem

    results = {}

    for algo_name, algo in algorithms.items():
        algo.reset()
        rng = np.random.default_rng(seed)

        cum_loss = 0.0
        cum_viol = 0.0
        max_viol = 0.0
        losses = []
        viols = []

        for t in range(1, T + 1):
            # Sample loss parameters
            problem.sample_loss_params(rng)

            # Store data for baseline computation
            if algo_name == list(algorithms.keys())[0]:  # Only first algo
                if hasattr(problem, 'get_v_t'):
                    v_list.append(problem.get_v_t())
                if hasattr(problem, 'get_data_point'):
                    x_t_data, y_t_data = problem.get_data_point()
                    data_points["x"].append(x_t_data)
                    data_points["y"].append(y_t_data)

            # Algorithm step
            x_t = algo.step()

            # Compute metrics
            loss_t = problem.loss(x_t)
            g_t = problem.constraint(x_t)
            viol_t = max(g_t, 0.0)

            cum_loss += loss_t
            cum_viol += viol_t
            max_viol = max(max_viol, viol_t)

            losses.append(loss_t)
            viols.append(viol_t)

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

        results[algo_name] = {
            "cum_loss": cum_loss,
            "cum_viol": cum_viol,
            "max_viol": max_viol,
            "losses": losses,
            "viols": viols
        }

    # Compute optimal baseline
    if hasattr(problem, 'compute_optimal_batch') and v_list:
        _, opt_loss = problem.compute_optimal_batch(np.array(v_list))
    elif data_points["x"]:
        # Batch solver for logreg
        batch_cfg = problem.__dict__.get("batch_solver", {})
        _, opt_loss = solve_batch_logreg(
            data_points["x"],
            data_points["y"],
            B=problem.B,
            max_iter=batch_cfg.get("max_iter", 5000),
            lr=batch_cfg.get("lr", 0.01),
            seed=batch_cfg.get("seed", 999)
        )
    else:
        opt_loss = 0.0

    # Log aggregates with regret
    for algo_name, res in results.items():
        regret = res["cum_loss"] - opt_loss
        logger.log_aggregate(
            algo=algo_name,
            T=T,
            trial=trial,
            regret=regret,
            cum_viol=res["cum_viol"],
            max_viol=res["max_viol"],
            cum_loss=res["cum_loss"]
        )

    return results


def run_experiment(config: Dict[str, Any], output_dir: Path):
    """Run full experiment."""

    benchmark = config["benchmark"]
    exp_cfg = config["experiment"]
    horizons = exp_cfg["horizons"]
    trials = exp_cfg["trials"]
    seed_base = exp_cfg["seed_base"]

    logger = MetricsLogger(benchmark)
    problem = create_problem(config)

    # Add batch solver config to problem if present
    if "batch_solver" in config:
        problem.batch_solver = config["batch_solver"]

    total_runs = len(horizons) * trials * 4  # 4 algorithms
    pbar = tqdm(total=total_runs, desc=f"Running {benchmark}")

    for T in horizons:
        algorithms = create_algorithms(problem, T, config)

        for trial in range(1, trials + 1):
            seed = get_trial_seed(seed_base, trial, T)

            run_single_trial(
                problem=problem,
                algorithms=algorithms,
                T=T,
                trial=trial,
                seed=seed,
                logger=logger
            )

            pbar.update(len(algorithms))

    pbar.close()

    # Save results
    step_df = logger.get_step_df()
    agg_df = logger.get_agg_df()
    summary_df = logger.compute_summary()

    step_df.to_csv(output_dir / "metrics_step.csv", index=False)
    agg_df.to_csv(output_dir / "metrics_agg.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    # Save resolved config
    with open(output_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Results saved to {output_dir}")

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