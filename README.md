# When Does Geometry-Aware Search Pay Off?

**DSA4212 Individual Project**  
*Bayesian optimisation versus evolution strategies on aligned and rotated ill-conditioned problems*

## Project Overview

This project investigates when Bayesian optimisation (BO) and CMA-ES succeed or fail on problems whose difficulty comes from geometry: ill-conditioning and rotation. It tests one explanation for a difference in their behaviour:

> Methods whose internal model is **axis-aligned** (BO with an ARD kernel, diagonal CMA-ES) should lose performance when an ill-conditioned problem is rotated. A method that can represent **any orientation** (full CMA-ES) should not, once it has had enough evaluations to adapt.

The study uses rotated ellipsoids, where conditioning and orientation can be varied independently while the optimum and Hessian are known exactly. A planned GARCH(1,1) application was removed from scope (decision log D31).

All design decisions, alternatives and sources are recorded in [`decision_log.md`](decision_log.md).

## Research Questions

**Primary:** Under matched evaluation budgets at $d=4$ and $\kappa=10^4$, does rotation hurt diagonal CMA-ES and BO with an ARD kernel more than it hurts full CMA-ES?

**Secondary (pre-registered):**

- Does full CMA-ES's rotation penalty shrink as the budget grows, while that of the axis-aligned methods does not?
- How does the rotation penalty change with conditioning?
- Across rotated instances, does the Jacobi-scaled condition number $\kappa_J$ predict the gap between diagonal and full CMA-ES?

## Methods

| Method | Model of geometry | Implementation |
|---|---|---|
| Full CMA-ES | Full covariance: scales and correlations | `pycma` 4.5.0, default settings |
| Diagonal CMA-ES | Per-coordinate scales only | `pycma`, `CMA_diagonal=True` |
| Isotropic ES | None (step size only) | `pycma`, `CMA_on=0` |
| BO, ARD kernel | Per-coordinate lengthscales | Matérn 5/2 ARD Gaussian process, expected improvement |
| BO, isotropic kernel | One lengthscale; rotation-invariant | Matérn 5/2 isotropic Gaussian process, expected improvement |
| Uniform random search | None; ignores feedback | Independent uniform draws over the box |

No method is tuned per problem. CMA-ES uses pycma defaults with $\sigma_0 = 0.3 \times$ box width and a uniform random start (Hansen, 2023); BO uses standard defaults with GP hyperparameters fitted by maximum marginal likelihood (decision log D30). The BO library and its numerical settings are fixed by a smoke test before the main runs.

## Experimental Design

The objective is a rotated ellipsoid, based on BBOB functions f2 and f10:

$$
f(x) = \sum_{i=1}^{d} \kappa^{(i-1)/(d-1)} z_i^2,
\qquad z = R(x-x_{\mathrm{opt}}).
$$

- **Orientation:** $R$ is the identity (aligned) or a Haar-random orthogonal matrix (rotated).
- **Conditioning:** $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$ for the CMA-ES variants and uniform search; $\kappa \in \lbrace 1, 10^4 \rbrace$ for BO. At $\kappa=1$ the ellipsoid is a sphere, so rotation changes nothing; this is the control.
- **Dimension:** $d=4$.
- **Domain:** $[-5,5]^4$, with the optimum drawn uniformly in $[-4,4]^4$.
- **Replicates:** 50 per condition. Every method faces the same instances; aligned and rotated instances share their optimum; the CMA-ES variants share their start point.
- **Budget:** objective evaluations, enforced exactly by one counting wrapper. The main budget is selected from pilot checkpoints (90, 250, 500, 1,000, 2,000) by a rule fixed before the pilot (D12). BO runs to a lower ceiling and all methods are compared at shared checkpoints.
- **Costs:** wall-clock time is reported separately from evaluations.

The Sphere is used only for unit tests and validation.

## Evaluation

The primary outcome is the final simple regret of the best evaluated point, $r = f(\hat{x}) - f(x_{\mathrm{opt}})$, analysed as $\log_{10}$ regret with a floor of $10^{-8}$ (D15).

Each primary comparison is a difference-in-differences against full CMA-ES at $d=4$, $\kappa=10^4$:

$$
\Delta_M =
\left(\bar L_{\mathrm{rot},M} - \bar L_{\mathrm{align},M}\right)
-
\left(\bar L_{\mathrm{rot,full}} - \bar L_{\mathrm{align,full}}\right),
$$

where $L$ is log regret and $M$ is diagonal CMA-ES or ARD BO. A positive value means rotation hurts $M$ more than full CMA-ES. Replicates are paired by instance and uncertainty is reported with bootstrap confidence intervals.

Mechanism diagnostics include $\kappa_{\mathrm{eff}} = \mathrm{cond}(\Sigma^{1/2} H \Sigma^{1/2})$ for CMA-ES (whether the search distribution matches the inverse Hessian) and the fitted ARD lengthscales for BO. Sensitivity analyses vary the floor ($10^{-6}$, $10^{-10}$) and, optionally, the initial step size and dimension.

## Repository Structure

```text
dsa4212-bo-cmaes/
├── code/
│   ├── objectives.py        # ellipsoid, Sphere, rotations, Hessian diagnostics
│   ├── config_loader.py     # loads and validates configs/experiment.yaml
│   ├── runner.py            # exact-budget wrapper, seeds, CMA-ES variants, uniform search
│   ├── bo.py                # Bayesian optimisation (ARD and isotropic Matérn 5/2, expected improvement)
│   ├── run_pilot.py         # pilot experiment
│   └── tests/               # unit and smoke tests
├── configs/
│   └── experiment.yaml
├── report/                  # LaTeX source, bibliography and figures of the submitted report
├── results/
│   ├── raw/                 # evaluation-level logs and problem instances
│   ├── processed/           # aggregated tables
│   └── figures/             # generated figures
├── readme.md
├── authors.md
├── decision_log.md
└── requirements.txt
```

## Environment Setup

Requires Python 3.12.4. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Package versions are pinned in `requirements.txt`.

## Data

No external data are used. All problem instances are generated from seeds recorded in `configs/experiment.yaml`; each instance (optimum, rotation matrix, $\kappa_H$, $\kappa_J$) is saved with the results.

## Reproducing Results

```bash
# Unit tests
python -m pytest code/tests

# Pilot (CMA-ES variants and uniform search); writes to results/raw/pilot/<run_id>/
python code/run_pilot.py
python code/run_pilot.py --quick   # 10-second smoke test
```

The following entry points are planned and will be confirmed once implemented and tested:

```bash
python code/run_main.py --config configs/experiment.yaml   # main runs
python code/make_figures.py                               # every table and figure in the report
```

## Reproducibility

The project preserves the experiment configuration, software versions, problem and method seeds, every problem instance, per-evaluation logs (point, objective value, best-so-far, bound-repair flag), CMA-ES covariance snapshots, BO hyperparameter diagnostics, and the scripts that generate every table and figure. Main-run settings are frozen after the pilot; unexpected results are investigated rather than silently rerun or omitted.

## Project Status

**Current stage:** pilot preparation.

Implemented and tested: objectives, configuration loader, exact-budget runner, CMA-ES variants, uniform search, Bayesian optimisation (ARD and isotropic kernels) and the pilot script. Next: run the pilot and freeze the budget and floor, then fix BO's settings with a smoke test.

## Submission

The final report is a PDF of at most 10 pages, including references and appendices, due on 13 November 2026.