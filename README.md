# Constrained Online Convex Optimization (OCO) with constraints

## PFS (Polyak Feasibility Steps) vs baseline algorithms (POGD, DPP, DPP-T)

This repository is a reproducible testbed for comparing online convex optimization algorithms **with constraints** in the *constrained OCO* setting on two benchmarks:

1. a synthetic quadratic task (Toy Quadratic),
2. online logistic regression with a norm constraint (Online Logistic Regression).

The key idea is to compare:

* **POGD**: the classical projected method (with projection onto the truly feasible set $X$),
* **DPP**: the primal–dual **Drift-Plus-Penalty** approach with a virtual queue,
* **DPP-T**: DPP with **tightening** (a strengthened constraint),
* **PFS**: OGD + **Polyak Feasibility Steps** (in the style of the 2025 paper), where the constraint is controlled via a “Polyak step” with one constraint check per round.

---

## 1. Problem statement (constrained OCO)

At each step $t=1,\dots,T$ the algorithm chooses a decision $x_t \in X_0$ (a simple set onto which it is easy to project). Then a convex loss $f_t(x_t)$ is incurred and the constraint violation $g(x_t)$ is measured, where the feasible region is defined as:

$$
X ;=;{x \in X_0:; g(x)\le 0}.
$$

### Quality metrics

**Regret** with respect to the best fixed decision from the feasible region:
$$
\mathrm{Regret}*T ;=; \sum*{t=1}^T f_t(x_t);-;\min_{x\in X}\sum_{t=1}^T f_t(x).
$$

In the code, the baseline $\min_{x\in X}\sum_{t=1}^T f_t(x)$ is computed via **batch optimization** (a separate solver) for an honest regret calculation.

### Constraint satisfaction metrics

Instantaneous violation:
$$
\mathrm{viol}*t ;=;[g(x_t)]*+ ;=;\max(g(x_t),0).
$$

Cumulative violation:
$$
\mathrm{CumViol}*T ;=;\sum*{t=1}^T [g(x_t)]_+.
$$

Maximum violation:
$$
\mathrm{MaxViol}*T ;=;\max*{t\le T}[g(x_t)]_+.
$$

---

## 2. Implemented algorithms

All algorithms satisfy the engineering requirement of the experiment: **one constraint check per round** (in the code this is implemented by having `algo.step()` return $(x_t, g_t)$, and $g_t=g(x_t)$ is queried exactly once). For PFS/DPP, the constraint subgradient $u_t \in \partial g(x_t)$ is also used; within the access model it is assumed to be available together with the constraint query.

### 2.1 POGD — Projected Online Gradient Descent

Classic projection onto the truly feasible set $X$:

$$
x_{t+1} ;=; \Pi_X\Bigl(x_t - \eta \nabla f_t(x_t)\Bigr),
\qquad \eta=\frac{\eta_{\mathrm{const}}}{\sqrt{T}}.
$$

**Strength:** almost always the best regret if projection onto $X$ is cheap.
**Weakness:** in general, projection onto $X$ can be computationally expensive/unknown.

---

### 2.2 DPP — Drift-Plus-Penalty (Yu et al., 2017)

A primal–dual method with a virtual queue $Q_t\ge 0$ to control the constraint:

1. gradients at point $x_t$:

* $\nabla f_t(x_t)$
* $g_t=g(x_t)$
* $u_t \in \partial g(x_t)$

2. primal step (projection only onto the simple set $X_0$):
   $$
   d_t ;=; V\nabla f_t(x_t) + Q_t u_t,
   $$
   $$
   x_{t+1};=;\Pi_{X_0}\Bigl(x_t-\frac{d_t}{2\alpha}\Bigr).
   $$

3. queue update:
   $$
   Q_{t+1} ;=;\max\Bigl(Q_t + g(x_t) + u_t^\top(x_{t+1}-x_t),;0\Bigr).
   $$

In the code, parameters use the classic scaling:
$$
\alpha = T,\qquad V=\sqrt{T}.
$$

---

### 2.3 DPP-T — DPP with tightened constraint

Same as DPP, but the queue is updated using a **strengthened** constraint:
$$
g_\rho(x) ;=; g(x)+\rho,
$$
where tightening is chosen as
$$
\rho(T);=;\min\Bigl(\varepsilon,\sqrt{\tfrac{c}{T}}\Bigr).
$$

Intuition: tightening should reduce the real violation $g(x)$, but the cost is often worse regret.

---

### 2.4 PFS — OGD + Polyak Feasibility Steps (2025-paper style)

The step consists of two parts:

1. **gradient step on the loss**:
   $$
   y_t ;=; x_t - \eta \nabla f_t(x_t).
   $$

2. **Polyak feasibility step** for the linear approximation of the constraint at $x_t$:
   let $g_t=g(x_t)$ and $s_t\in\partial g(x_t)$. Consider the linear model at $x_t$:
   $$
   \ell_t(y);=;g_t + s_t^\top(y-x_t) + \rho.
   $$
   If $\ell_t(y_t)>0$, we perform a “Polyak step”:
   $$
   y_t \leftarrow y_t - \frac{\ell_t(y_t)}{|s_t|^2},s_t.
   $$

3. projection onto the simple set $X_0$:
   $$
   x_{t+1}=\Pi_{X_0}(y_t).
   $$

In the code, tightening is:
$$
\rho(T)=\min\Bigl(\varepsilon,\sqrt{\tfrac{\alpha}{T}}\Bigr),
\qquad \alpha=\varepsilon,
$$
and the step size is:

* either $\eta=\dfrac{\eta_{\mathrm{const}}}{\sqrt{T}}$ (if specified in the config),
* or by default $\eta=\dfrac{\rho}{2\sqrt{2}}$ (as a convenient scale for the toy task).

---

## 3. Benchmarks (tasks)

### 3.1 Toy Quadratic (Benchmark A)

Losses:
$$
f_t(x)=3|x-v_t|_2^2,\qquad v_t\sim \mathrm{Unif}([0,1]^d).
$$

Simple set:
$$
X_0 = B(R)={x:|x|_2\le R}.
$$

Constraint and feasible set:
$$
g(x)=|x|*\infty - b,\qquad
X={x:|x|*\infty\le b}=[-b,b]^d.
$$

---

### 3.2 Online Logistic Regression (Benchmark B)

Losses (logistic):
$$
f_t(w)=\log\bigl(1+\exp(-y_t,w^\top x_t)\bigr),\qquad y_t\in{-1,+1}.
$$

Data generation:

* $x_t\sim\mathcal N(0,I)$,
* a hidden vector $w^\star$ is fixed with $\lVert w^\star\rVert_2 = 1$,
* then $y_t = +1$ with probability $\sigma!\left((w^\star)^\top x_t\right)$, otherwise $-1$.

Simple set:
$$
X_0=B(R_0)={w:|w|_2\le R_0}.
$$

Constraint and feasible set:
$$
g(w)=|w|_2 - B,\qquad X=B(B)={w:|w|_2\le B}.
$$

---

## 4. How to run

### 4.1 Install dependencies

Option via `requirements.txt`:

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Or via `pyproject.toml` (if you prefer a PEP-517/518 environment):

```bash
pip install -e .
```

---

### 4.2 Run an experiment

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
after running, a folder of the form is created:

```
results/<benchmark>/<YYYYMMDD_HHMMSS>/
```

It contains:

* `config_resolved.yaml` — the resolved config,
* `metrics_step.csv` — per-step metrics (with subsampling for large T),
* `metrics_agg.csv` — aggregates per trial,
* `metrics_summary.csv` — mean/std over trials,
* `optimal_points.json` — batch-optimum points (for trajectories),
* a set of `.png` plots (if plotting is enabled).

---

### 4.3 Plot results

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

## 5. Configs and hyperparameters

Configs are in `configs/`.

### `configs/toy.yaml` (Toy Quadratic)

* `problem.d`, `problem.R`, `problem.b`
* `experiment.horizons` — list of horizons $T$
* `experiment.trials` — number of runs
* algorithms:

  * `PFS.epsilon`
  * `DPP` (no parameters, uses $\alpha=T$, $V=\sqrt{T}$)
  * `DPP-T.epsilon`, `DPP-T.c`
  * `POGD.eta_const`

### `configs/logreg.yaml` (Online Logistic Regression)

* `problem.d`, `problem.R0`, `problem.B`, `problem.w_star_seed`
* `experiment.horizons` (default `[50000]`)
* `batch_solver.*` — batch-solver parameters for the regret baseline
* algorithms:

  * `PFS.epsilon`, `PFS.eta_const`
  * `DPP`
  * `DPP-T.epsilon`, `DPP-T.c`
  * `POGD.eta_const`

---

## 6. Project structure

```
configs/
  toy.yaml
  logreg.yaml

src/oco/
  run_experiment.py        # runs experiments and logs metrics
  plot_results.py          # generates plots from a results folder

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
    batch_opt.py           # batch solvers for the regret baseline
    seeding.py             # seeding/reproducibility

src/results/
  ...                      # example already-generated results (see below)
```

---

## 7. Example results

Below are example plots from the `src/results/` folder (these are **demo artifacts**, so the README is self-contained).
In your runs, similar files will appear in `results/...`.

---

### 7.1 Toy Quadratic (`src/results/toy/20251223_163822/`)

#### Regret vs Horizon

The plot shows the typical behavior:

* **POGD** (projection onto $X$ is cheap — box) yields the smallest regret,
* **PFS** is very close to POGD,
* **DPP** is slightly worse in regret,
* **DPP-T** loses noticeably in regret due to tightening.

![Toy: Regret vs T](src/results/toy/20251223_163822/regret_vs_T.png)

#### Cumulative constraint violation

* **POGD** and **PFS** practically do not violate the constraint (line at zero),
* **DPP** has substantial accumulated violation,
* **DPP-T** reduces violation compared to DPP, but does not make it zero.

![Toy: Cumulative violation vs T](src/results/toy/20251223_163822/cumviol_vs_T.png)

#### Instantaneous violation (T=20000)

The difference in “operating regime” is clearly visible:

* for **DPP** the violation stays at a nonzero level for almost the whole horizon,
* for **DPP-T** the violation is much lower and closer to zero,
* for **POGD/PFS** the violation is almost zero.

![Toy: Instantaneous violation](src/results/toy/20251223_163822/instviol_vs_t_T20000.png)

#### Trajectories (2D)

On the toy task it is convenient to visually compare how algorithms “approach” the optimum and how often they go outside the feasible box.

![Toy: Trajectories](src/results/toy/20251223_163822/trajectory_2d.png)

---

### 7.2 Online Logistic Regression (`src/results/logreg/20251223_171211/`)

#### Final regret (T=50000)

For this run:

* **DPP** achieves the best regret,
* **POGD** is close to it,
* **PFS** is worse in regret, but achieves zero violation,
* **DPP-T** loses substantially in regret.

![LogReg: Final regret comparison](src/results/logreg/20251223_171211/regret_comparison.png)

#### Final cumulative violation (T=50000)

* **POGD** — strictly within $X$ (zero violation, since projection is onto $X=B(B)$),
* **PFS** — in this run also yields zero violation,
* **DPP** accumulates large violation,
* **DPP-T** reduces violation compared to DPP, but does not eliminate it.

![LogReg: Final cumulative violation](src/results/logreg/20251223_171211/cumviol_comparison.png)

#### Instantaneous violation over steps

The plot highlights that for DPP/DPP-T the violation “pulses” and does not disappear, whereas POGD/PFS stay at zero.

![LogReg: Instantaneous violation vs step](src/results/logreg/20251223_171211/instviol_vs_t.png)

#### Relative cumulative loss gap (vs the best algorithm)

In this run DPP is the best in cumulative loss (hence its line is near 0), while DPP-T accumulates a large gap.

![LogReg: Relative cumulative loss gap](src/results/logreg/20251223_171211/loss_gap_vs_t.png)

#### Summary table (mean/std over trials, T=50000)

(The table below is an exact transcription of the image `summary_table.png`.)

|  algo | regret_mean | regret_std | cum_viol_mean | cum_viol_std | max_viol_mean | max_viol_std |
| ----: | ----------: | ---------: | ------------: | -----------: | ------------: | -----------: |
|   PFS |      189.27 |       5.55 |          0.00 |         0.00 |          0.00 |         0.00 |
|   DPP |      176.24 |       5.78 |        521.37 |        71.85 |          0.07 |         0.01 |
| DPP-T |      243.76 |       5.79 |        174.14 |        47.47 |          0.05 |         0.01 |
|  POGD |      178.86 |       5.56 |          0.00 |         0.00 |          0.00 |         0.00 |

And the original image:

![LogReg: Summary table](src/results/logreg/20251223_171211/summary_table.png)

---

## 8. Interpretation of the observed trade-off

The practical picture (across the two benchmarks) fits the expected **trade-off**:

* **When projection onto the truly feasible set $X$ is cheap** (toy task, $X$ is a box), **POGD** often becomes the strongest reference in regret and does not violate the constraint.

* **PFS** aims to approach the quality of POGD, while staying in the regime “simple projection onto $X_0$ + one constraint check”. On the toy task it is indeed close to POGD in regret and does not violate the constraint in practice.

* **DPP** and **DPP-T** exhibit the characteristic primal–dual behavior: you can win in regret (or be competitive), but the price is noticeable accumulated violation (especially for DPP). Tightening (DPP-T) reduces violation, but usually worsens regret.

---
