# When Does Geometry-Aware Search Pay Off?

**DSA4212 Individual Project**  
*A controlled comparison of Bayesian optimisation and CMA-ES, with a GARCH(1,1) calibration application*

## Project Overview

This project studies how black-box optimisation methods behave when the number of objective evaluations is limited.

The primary experiment compares Bayesian optimisation (BO) and CMA-ES variants on aligned and randomly rotated ellipsoids with different condition numbers. The goal is to understand how rotation and conditioning affect finite-budget performance, particularly when an optimiser can or cannot learn correlated search directions.

A secondary application investigates whether the same ideas help explain optimiser behaviour when calibrating a GARCH(1,1) volatility model on rolling windows of WTI returns.

The synthetic experiment is the main contribution. The finance application is secondary and will be reduced or omitted if necessary to complete the core experiment reliably.

## Research Questions

**Primary question:** Under matched evaluation budgets, how do BO, full-covariance CMA-ES, and diagonal CMA-ES compare across aligned and rotated ellipsoids, and how does covariance adaptation interact with rotation and conditioning?

**Secondary question:** When fitting GARCH(1,1) models to rolling WTI return windows, does the conditioning of each likelihood surface help predict where full-covariance CMA-ES outperforms diagonal CMA-ES, and do optimisation differences affect out-of-sample volatility forecasts?

## Methods and Baselines

Three adaptive methods are compared against two reference arms.

### Adaptive methods

| Method | Implementation | Role |
|---|---|---|
| BO with an ARD Gaussian process | Matérn 5/2 ARD kernel with expected improvement | Surrogate-based search that learns coordinate-wise structure |
| Full-covariance CMA-ES | `pycma`, default full-covariance mode | Learns coordinate scales and correlations |
| Diagonal CMA-ES | `pycma`, `CMA_diagonal=True` | Learns coordinate-wise scales but not correlations |

### Reference arms

| Arm | Implementation | Role |
|---|---|---|
| Uniform random search | Independent uniform draws over the search box | Null baseline that does not use objective feedback to choose subsequent points |
| Isotropic ES | `pycma`, `CMA_on=0` | Ablation baseline that adapts step size without learning covariance structure |

All CMA-ES variants use the same `pycma` version, initialisation, population size, boundary handling, and stopping rules. The `CMA_diagonal` and `CMA_on` behaviours must be verified in the installed version before the main experiment: off-diagonal covariance entries must remain zero for the diagonal variant, and the covariance must remain the identity for the isotropic variant.

Conclusions apply to the implementations and configurations tested, not to every BO or evolutionary optimisation method.

## Experimental Design

### Main experiment: Ellipsoid

The core experiment uses a four-dimensional ellipsoid:

$$
f(x) = \sum_{i=1}^{d} \kappa^{(i-1)/(d-1)} z_i^2,
\qquad z = R(x-x_{\mathrm{opt}}).
$$

Here:

- $x$ is a candidate solution.
- $x_{\mathrm{opt}}$ is the known optimum.
- $R$ is an orthogonal rotation matrix.
- $\kappa$ is the condition number.

The orientation matrix $R$ is either the identity matrix (aligned) or a randomly generated orthogonal matrix (rotated). A new optimum and rotation are drawn for each replicate and saved. Every method within a replicate faces the same problem instance.

The experiment varies:

- **Condition number:** $\kappa \in \lbrace 1, 10^2, 10^4 \rbrace$.
- **Orientation:** aligned and randomly rotated.
- **Dimension:** $d=4$ for the main study.

At $\kappa=1$, the ellipsoid is a Sphere, so aligned and rotated objectives are mathematically identical. This case provides a control for the rotation implementation and seed handling.

### Validation: Sphere

The Sphere is used for unit tests, method-interface checks, and sanity checks at dimensions $d \in \lbrace 2, 4, 10 \rbrace$.

Rastrigin is not part of the planned experiments and is reserved for future work.

### Initialisation and Fairness

- Every method receives the same objective, search box, and total evaluation budget within each comparison.
- The CMA-ES starting point $x_0$ is sampled uniformly from the search box, independently of $x_{\mathrm{opt}}$.
- One initial step-size rule is used for all CMA-ES variants. The provisional candidate is $\sigma_0=0.3$ times the box width; the final rule will be frozen after the pilot.
- BO initial-design evaluations count towards its evaluation budget.
- The BO implementation, input scaling, initial-design policy, and other settings are frozen after the pilot.
- Every objective call is counted. The final CMA-ES batch is truncated if necessary to meet the exact budget.
- Problem seeds are shared across methods within a replicate; method seeds are derived separately.
- Boundary handling is documented, and clipping or penalty events are logged for each evaluation.
- Wall-clock time and optimiser overhead are reported separately from objective-evaluation counts.

The evaluation budget, replicate count, numerical floor for log-regret, and other pilot-dependent settings are documented in `decision_log.md` and will be frozen after the pilot.

## Evaluation and Statistical Plan

The primary outcome is final simple regret:

$$
r = f(\hat{x}) - f(x_{\mathrm{opt}}),
$$

where $\hat{x}$ is the best point actually evaluated by the method. The primary analysis uses $\log_{10}$ regret with a numerical floor selected during the pilot.

The primary comparison is a difference-in-differences in mean log-regret at $d=4$ and $\kappa=10^4$:

$$
\begin{aligned}
\Delta ={}&
\left(
\overline{\log_{10} r}_{\mathrm{rotated,\,diagonal}}
-
\overline{\log_{10} r}_{\mathrm{aligned,\,diagonal}}
\right)\\
&-
\left(
\overline{\log_{10} r}_{\mathrm{rotated,\,full}}
-
\overline{\log_{10} r}_{\mathrm{aligned,\,full}}
\right).
\end{aligned}
$$

A positive value means that rotation increases the diagonal CMA-ES method's mean log-regret more than it increases full-covariance CMA-ES's mean log-regret.

Replicates are paired by problem instance, and uncertainty is reported using bootstrap confidence intervals. The target is approximately 20 replicates per ellipsoid condition, subject to pilot runtime.

Other condition numbers, comparisons involving BO, intermediate checkpoints, and evaluations-to-target are exploratory. Mechanism diagnostics include the effective condition number of the learned CMA-ES covariance relative to the true Hessian, as well as the fitted BO ARD lengthscales. These diagnostics support interpretation but do not, by themselves, establish causation.

Uniform random search is checked against its theoretical success-probability law:

$$
P(\text{at least one success in }N\text{ evaluations})
= 1-(1-q)^N,
$$

where $q$ is the probability that a single uniform draw meets the defined success criterion. It is estimated using a large uniform sample.

## Finance Application: GARCH(1,1)

The secondary application calibrates a Gaussian quasi-maximum-likelihood GARCH(1,1) model on rolling windows of WTI returns.

The main optimisation arm uses the reparameterisation

$$
(\log\omega,p,s),
\qquad p=\alpha+\beta,
\qquad s=\frac{\alpha}{\alpha+\beta}.
$$

The original parameters are recovered using

$$
\alpha=ps,
\qquad
\beta=p(1-s),
\qquad
\omega=\exp(\log\omega).
$$

The bounds on $p$ and $s$ must enforce the intended parameter constraints. In particular, $p$ must remain strictly between zero and one for $\alpha+\beta<1$, and $s$ must remain strictly between zero and one if both $\alpha$ and $\beta$ are required to be positive. The exact numerical bounds will be documented and frozen before the main finance runs.

The proposed rolling design uses a 750-day estimation window and a 250-day step. The custom likelihood will be validated against the `arch` package on at least three windows before scaling up.

The analysis compares optimiser likelihood gaps, tests whether the Jacobi-scaled Hessian condition number of each window predicts the diagonal-versus-full CMA-ES gap, and compares out-of-sample volatility forecasts using QLIKE.

This is a model-calibration study, not a trading backtest, and it makes no profitability claim.

## Data

| Item | Detail |
|---|---|
| Series | FRED `DCOILWTICO`, daily WTI spot price in USD per barrel |
| Original data source | U.S. Energy Information Administration |
| Source URL | [FRED: DCOILWTICO](https://fred.stlouisfed.org/series/DCOILWTICO) |
| Intended frozen copy | `data/DCOILWTICO.csv` |
| Retrieval date | To be recorded when the CSV is downloaded and frozen |

Missing days are dropped. Returns are calculated as

$$
r_t=100\ln\left(\frac{P_t}{P_{t-1}}\right),
$$

where $P_t$ is the WTI spot price. Returns are demeaned using the estimation-window mean.

The series contains a non-positive price observation on 20 April 2020. Log returns are undefined for non-positive prices, so any estimation window containing that observation is excluded from the main analysis, and the exclusion is recorded.

## Repository Structure

```text
dsa4212-bo-cmaes/
├── code/
│   └── ...
├── configs/
│   └── experiment.yaml
├── data/
│   └── DCOILWTICO.csv
├── report/
│   └── figures/
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── README.md
├── authors.md
├── decision_log.md
└── requirements.txt
```

- `code/`: objective functions, optimisation methods, experiment runners, analysis scripts, and tests.
- `configs/`: experiment settings.
- `data/`: frozen input data and associated source metadata.
- `report/`: report source and figures selected for the final report.
- `results/raw/`: evaluation-level logs and raw outputs.
- `results/processed/`: aggregated results and analysis tables.
- `results/figures/`: generated result figures.
- `decision_log.md`: dated record of design choices, alternatives, rationale, and unresolved decisions.

## Environment Setup

Requires Python (version to be confirmed).

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Package versions, including `cma`, `arch`, and the selected BO library, are pinned in `requirements.txt`.

## Reproducing Results

The commands below are intended entry points. They should be treated as planned commands until the corresponding scripts have been implemented and tested.

```bash
# Unit tests
python -m pytest code/tests

# Synthetic study: ellipsoid and Sphere
python code/run_synthetic.py --config configs/experiment.yaml

# Finance application: GARCH(1,1)
python code/run_garch.py --config configs/experiment.yaml

# Regenerate tables and figures
python code/make_figures.py
```

## Reproducibility

The project preserves:

- Experiment configurations and software versions.
- Problem-instance and method seeds.
- The optimum and rotation matrix for each synthetic instance.
- Per-evaluation objective values, best-so-far values, and boundary-event flags.
- CMA-ES covariance snapshots and BO hyperparameter diagnostics.
- Raw results and the scripts used to generate tables and figures.

Main-run settings are frozen after the pilot. Unexpected results are investigated rather than silently rerun or omitted.

## Project Status

**Current stage:** Implementation and pilot preparation.

The immediate priority is to implement and test the ellipsoid and Sphere objectives, verify problem generation and evaluation logging, and run a small pilot. The GARCH application follows the priority order in the project proposal.

## Submission

The final report is a PDF of at most 10 pages, including references and appendices, due on 13 November 2026.