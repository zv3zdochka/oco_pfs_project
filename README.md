# Constrained Online Convex Optimization (OCO)  
## PFS (Polyak Feasibility Steps) vs Baseline Algorithms (POGD, DPP, DPP-T)

This repository is a reproducible testbed for comparing online convex optimization algorithms **with constraints** in the *constrained OCO* setting on two benchmarks:
1) Synthetic quadratic problem (Toy Quadratic),  
2) Online logistic regression with norm constraint (Online Logistic Regression).

The key idea is to compare:
- **POGD**: classical projected gradient descent (with projection onto the true feasible set $X$),
- **DPP**: primal–dual **Drift-Plus-Penalty** approach with a virtual queue,
- **DPP-T**: DPP with **tightening** (strengthened constraint),
- **PFS**: OGD + **Polyak Feasibility Steps** (in the style of the 2025 paper), where the constraint is controlled by a "Polyak step" with one constraint query per round.

---

## 1. Problem Statement (Constrained OCO)

At each step $t=1,\dots,T$, the algorithm chooses a solution $x_t \in X_0$ (a simple set onto which projection is easy). Then a convex loss $f_t(x_t)$ is incurred and the constraint violation $g(x_t)$ is measured, where the feasible region is defined as:

```math
X = \{x \in X_0 : g(x) \le 0\}
```

### Performance Metrics

**Regret** relative to the best fixed solution from the feasible region:

```math
\mathrm{Regret}_T = \sum_{t=1}^T f_t(x_t) - \min_{x \in X} \sum_{t=1}^T f_t(x)
```

In the code, the baseline $\min_{x \in X} \sum_{t=1}^T f_t(x)$ is computed via **batch optimization** (a separate solver) for fair regret computation.

### Constraint Satisfaction Metrics

Instantaneous violation:

```math
\mathrm{viol}_t = [g(x_t)]_+ = \max(g(x_t), 0)
```

Cumulative violation:

```math
\mathrm{CumViol}_T = \sum_{t=1}^T [g(x_t)]_+
```

Maximum violation:

```math
\mathrm{MaxViol}_T = \max_{t \le T} [g(x_t)]_+
```

---

## 2. Implemented Algorithms

All algorithms satisfy the engineering requirement of the experiment: **one constraint query per round** (in the code this is implemented by having `algo.step()` return $(x_t, g_t)$, where $g_t = g(x_t)$ is queried exactly once). For PFS/DPP, the subgradient of the constraint $u_t \in \partial g(x_t)$ is also used; within the access model, it is assumed to be available together with the constraint query.

### 2.1 POGD — Projected Online Gradient Descent

Classical projection onto the true feasible set $X$:

```math
x_{t+1} = \Pi_X\left(x_t - \eta \nabla f_t(x_t)\right), \quad \eta = \frac{\eta_{\mathrm{const}}}{\sqrt{T}}
```

**Strength:** almost always the best regret if projection onto $X$ is cheap.  
**Weakness:** in general, projection onto $X$ can be computationally expensive/unknown.

---

### 2.2 DPP — Drift-Plus-Penalty (Yu et al., 2017)

Primal–dual method with a virtual queue $Q_t \ge 0$ for constraint control:

1) Gradients at point $x_t$:  
   - $\nabla f_t(x_t)$  
   - $g_t = g(x_t)$  
   - $u_t \in \partial g(x_t)$

2) Primal step (projection only onto the simple set $X_0$):

```math
d_t = V \nabla f_t(x_t) + Q_t u_t
```

```math
x_{t+1} = \Pi_{X_0}\left(x_t - \frac{d_t}{2\alpha}\right)
```

3) Queue update:

```math
Q_{t+1} = \max\left(Q_t + g(x_t) + u_t^\top (x_{t+1} - x_t), 0\right)
```

The code uses parameters according to the classical scaling:

```math
\alpha = T, \quad V = \sqrt{T}
```

---

### 2.3 DPP-T — DPP with Tightened Constraint

Same as DPP, but the queue is updated using the **tightened** constraint:

```math
g_\rho(x) = g(x) + \rho
```

where tightening is taken as

```math
\rho(T) = \min\left(\varepsilon, \sqrt{\frac{c}{T}}\right)
```

Intuition: tightening should reduce the actual violation $g(x)$, but the cost is often worse regret.

---

### 2.4 PFS — OGD + Polyak Feasibility Steps (2025 Paper Style)

The step consists of two parts:

1) **Gradient step on loss**:

```math
y_t = x_t - \eta \nabla f_t(x_t)
```

2) **Polyak feasibility step** on the linear approximation of the constraint at point $x_t$:
   let $g_t = g(x_t)$ and $s_t \in \partial g(x_t)$. Consider the linear model at point $x_t$:

```math
\ell_t(y) = g_t + s_t^\top (y - x_t) + \rho
```

If $\ell_t(y_t) > 0$, perform a "Polyak step":

```math
y_t \leftarrow y_t - \frac{\ell_t(y_t)}{\|s_t\|^2} s_t
```

3) Projection onto the simple set $X_0$:

```math
x_{t+1} = \Pi_{X_0}(y_t)
```

In the code, tightening:

```math
\rho(T) = \min\left(\varepsilon, \sqrt{\frac{\alpha}{T}}\right), \quad \alpha = \varepsilon
```

and the step size:
- either $\eta = \dfrac{\eta_{\mathrm{const}}}{\sqrt{T}}$ (if specified in config),
- or by default $\eta = \dfrac{\rho}{2\sqrt{2}}$ (as a convenient scale for the toy problem).

---

## 3. Benchmarks (Problems)

### 3.1 Toy Quadratic (Benchmark A)

Losses:

```math
f_t(x) = 3\|x - v_t\|_2^2, \quad v_t \sim \mathrm{Unif}([0,1]^d)
```

Simple set:

```math
X_0 = B(R) = \{x : \|x\|_2 \le R\}
```

Constraint and feasible set:

```math
g(x) = \|x\|_\infty - b, \quad X = \{x : \|x\|_\infty \le b\} = [-b, b]^d
```

---

### 3.2 Online Logistic Regression (Benchmark B)

Losses (logistic):

```math
f_t(w) = \log\left(1 + \exp(-y_t \cdot w^\top x_t)\right), \quad y_t \in \{-1, +1\}
```

Data generation:
- $x_t \sim \mathcal{N}(0, I)$,
- a hidden vector $w^\star$ with $\|w^\star\|_2 = 1$ is fixed,
- then $y_t = +1$ with probability $\sigma\left((w^\star)^\top x_t\right)$, otherwise $-1$.

Simple set:

```math
X_0 = B(R_0) = \{w : \|w\|_2 \le R_0\}
```

Constraint and feasible set:

```math
g(w) = \|w\|_2 - B, \quad X = B(B) = \{w : \|w\|_2 \le B\}
```

---

## 4. How to Run

### 4.1 Installing Dependencies

Option via `requirements.txt`:
```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Or via `pyproject.toml` (if you prefer PEP-517/518 environment):

```bash
pip install -e .
```

---

### 4.2 Running an Experiment

Run via module:

```bash
python -m oco.run_experiment --config configs/toy.yaml
python -m oco.run_experiment --config configs/logreg.yaml
```

Or via console entrypoints:

```bash
oco-run --config configs/toy.yaml
oco-run --config configs/logreg.yaml
```

**Where results are saved:**
after running, a folder is created of the form:

```
results/<benchmark>/<YYYYMMDD_HHMMSS>/
```

The following is saved there:

* `config_resolved.yaml` — the fixed config,
* `metrics_step.csv` — step-by-step metrics (with subsampling for large T),
* `metrics_agg.csv` — aggregates for each trial,
* `metrics_summary.csv` — mean/std across trials,
* `optimal_points.json` — batch optimum points (for trajectories),
* a set of `.png` plots (if plotting is run).

---

### 4.3 Generating Plots from Results

```bash
python -m oco.plot_results --input results/toy/<TIMESTAMP>
python -m oco.plot_results --input results/logreg/<TIMESTAMP>
```

or:

```bash
oco-plot --input results/toy/<TIMESTAMP>
oco-plot --input results/logreg/<TIMESTAMP>
```

---

## 5. Configs and Hyperparameters

Configs are located in `configs/`.

### `configs/toy.yaml` (Toy Quadratic)

* `problem.d`, `problem.R`, `problem.b`
* `experiment.horizons` — list of horizons $T$
* `experiment.trials` — number of runs
* algorithms:

  * `PFS.epsilon`
  * `DPP` (no parameters, uses $\alpha = T$, $V = \sqrt{T}$)
  * `DPP-T.epsilon`, `DPP-T.c`
  * `POGD.eta_const`

### `configs/logreg.yaml` (Online Logistic Regression)

* `problem.d`, `problem.R0`, `problem.B`, `problem.w_star_seed`
* `experiment.horizons` (default `[50000]`)
* `batch_solver.*` — batch solver parameters for regret baseline
* algorithms:

  * `PFS.epsilon`, `PFS.eta_const`
  * `DPP`
  * `DPP-T.epsilon`, `DPP-T.c`
  * `POGD.eta_const`

---

## 6. Project Structure

```
configs/
  toy.yaml
  logreg.yaml

src/oco/
  run_experiment.py        # running experiments and logging metrics
  plot_results.py          # generating plots from results folder

  algorithms/
    base.py                # common Algorithm interface
    pfs.py                 # PFS (Polyak Feasibility Steps)
    pogd.py                # POGD (projection onto X)
    dpp.py                 # DPP (Drift-Plus-Penalty)
    dppt.py                # DPP-T (tightening)

  problems/
    toy_quadratic.py       # Toy Quadratic benchmark
    online_logreg.py       # Online Logistic Regression benchmark

  utils/
    logging.py             # MetricsLogger (step/agg/summary)
    projections.py         # project_ball, project_box
    subgradients.py        # norm subgradients
    batch_opt.py           # batch solvers for regret baseline
    seeding.py             # seeding/reproducibility

src/results/
  ...                      # example of already generated results (see below)
```

---

## 7. Example Results

Below are example plots from the `src/results/` folder (these are **demo artifacts** to make the README self-contained).
When you run experiments, similar files will appear in `results/...`.

---

### 7.1 Toy Quadratic (`src/results/toy/20251223_163822/`)

#### Regret vs Horizon

The plot shows typical behavior:

* **POGD** (projection onto $X$ is cheap — box) gives the minimum regret,
* **PFS** follows very close to POGD,
* **DPP** is slightly worse in regret,
* **DPP-T** noticeably loses in regret due to tightening.

![Toy: Regret vs T](src/results/toy/regret_vs_T.png)

#### Cumulative Constraint Violation

* **POGD** and **PFS** almost never violate the constraint (line at zero),
* **DPP** has substantial accumulated violation,
* **DPP-T** reduces violation compared to DPP, but doesn't make it zero.

![Toy: Cumulative violation vs T](src/results/toy/cumviol_vs_T.png)

#### Instantaneous Violation (T=20000)

The difference in "operating mode" is clearly visible:

* **DPP** violation stays at a non-zero level for almost the entire horizon,
* **DPP-T** violation is substantially lower and closer to zero,
* **POGD/PFS** violation is practically zero.

![Toy: Instantaneous violation](src/results/toy/instviol_vs_t_T20000.png)

#### Trajectories (2D)

On the toy problem, it's convenient to visually compare how algorithms "approach" the optimum and how often they exit the feasible box.

![Toy: Trajectories](src/results/toy/trajectory_2d.png)

---

### 7.2 Online Logistic Regression (`src/results/logreg/20251223_171211/`)

#### Final Regret (T=50000)

For this run:

* best regret is shown by **DPP**,
* **POGD** is close to it,
* **PFS** is worse in regret, but ensures zero violation,
* **DPP-T** loses substantially in regret.

![LogReg: Final regret comparison](src/results/logreg/regret_comparison.png)

#### Final Cumulative Violation (T=50000)

* **POGD** — strictly in $X$ (zero violation, since projection onto $X = B(B)$),
* **PFS** — in this run also gives zero violation,
* **DPP** accumulates large violation,
* **DPP-T** reduces violation compared to DPP, but doesn't eliminate it.

![LogReg: Final cumulative violation](src/results/logreg/cumviol_comparison.png)

#### Instantaneous Violation vs Steps

The plot emphasizes that DPP/DPP-T violation "pulsates" and doesn't disappear, while POGD/PFS stay at zero.

![LogReg: Instantaneous violation vs step](src/results/logreg/instviol_vs_t.png)

#### Relative Gap in Cumulative Loss (vs Best Algorithm)

In this run, DPP is the best in cumulative loss (so its line is around 0), while DPP-T accumulates a large gap.

![LogReg: Relative cumulative loss gap](src/results/logreg/loss_gap_vs_t.png)

#### Summary Table (mean/std across trials, T=50000)

(The table below is the exact transcription of the `summary_table.png` image.)

|  algo | regret_mean | regret_std | cum_viol_mean | cum_viol_std | max_viol_mean | max_viol_std |
| ----: | ----------: | ---------: | ------------: | -----------: | ------------: | -----------: |
|   PFS |      189.27 |       5.55 |          0.00 |         0.00 |          0.00 |         0.00 |
|   DPP |      176.24 |       5.78 |        521.37 |        71.85 |          0.07 |         0.01 |
| DPP-T |      243.76 |       5.79 |        174.14 |        47.47 |          0.05 |         0.01 |
|  POGD |      178.86 |       5.56 |          0.00 |         0.00 |          0.00 |         0.00 |


---

## 8. Interpretation of the Observed Trade-off

The practical picture (across both benchmarks) fits the expected **trade-off**:

* **When projection onto the true feasible $X$ is cheap** (toy problem, $X$ — box), **POGD** often becomes the strongest benchmark in regret while not violating the constraint.

* **PFS** aims to approach POGD quality while staying in the "simple projection onto $X_0$ + one constraint query" mode. On the toy problem, it is indeed close to POGD in regret and doesn't violate the constraint in practice.

* **DPP** and **DPP-T** demonstrate characteristic primal–dual behavior: one can win in regret (or be competitive), but the price is noticeable accumulated violation (especially for DPP). Tightening (DPP-T) reduces violation but usually worsens regret.

---