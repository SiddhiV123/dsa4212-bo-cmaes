# When Does Geometry-Aware Search Pay Off?

**DSA4212 Individual Project**

*A controlled comparison of Bayesian optimisation and covariance matrix adaptation evolution strategy (CMA-ES), with an application to GARCH(1,1) volatility-model calibration.*

## 1. Project Overview

This project investigates how black-box optimisation methods perform when objective evaluations are limited, focusing on the interaction between search geometry, rotation, and conditioning.

The primary study compares five methods on synthetic objective functions:

- Bayesian optimisation (BO) using an ARD Gaussian process with a Matérn 5/2 kernel and expected improvement.
- Full-covariance CMA-ES.
- Diagonal-only (separable) CMA-ES.
- Isotropic CMA-ES.
- Uniform random search over the bounded domain.

The main experiment uses aligned and randomly rotated ellipsoid functions with dimension \(d=4\) and condition numbers of \(1\), \(10^2\), and \(10^4\). Sphere provides a sanity check, while Rastrigin provides a supporting comparison at dimensions 4 and 10.

We examine optimisation performance alongside diagnostics of covariance adaptation and Gaussian-process lengthscales.

The secondary application studies GARCH(1,1) quasi-maximum-likelihood calibration on rolling windows of West Texas Intermediate (WTI) crude-oil returns. It investigates whether the optimisation behaviour observed in synthetic experiments is informative for financial calibration and out-of-sample volatility forecasting.

The project aims to explain differences in optimiser behaviour, rather than establish a universally superior method or develop a trading system.

## 2. Repository Structure

```text
dsa4212-bo-cmaes/
├── README.md
├── authors.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── experiment.yaml
├── code/
├── results/
│   ├── raw/
│   └── processed/
└── report/
    └── figures/
```

- `configs/`: experiment configurations and frozen settings.
- `code/`: objective functions, optimiser implementations, experiments, analysis, and plotting scripts.
- `results/raw/`: raw per-run results, seeds, configurations, and diagnostic logs.
- `results/processed/`: aggregated summaries and analysis outputs.
- `report/`: report source, bibliography, and generated figures.

## 3. Environment and Dependencies

The project uses Python and a project-specific virtual environment named `.venv`.

### Create and activate the environment

From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

The environment uses NumPy, SciPy, Matplotlib, pandas, scikit-learn, and `cma`. Additional dependencies required for the finance application will be documented in `requirements.txt`.

Use the Python interpreter in `.venv` when running project scripts in VS Code.

## 4. Experimental Design and Reproducibility

All methods will be evaluated under matched objective-evaluation budgets. Bayesian optimisation's initial-design evaluations count towards its budget. Wall-clock runtime will be reported separately.

The primary outcome is **final simple regret**, defined as the best objective value evaluated within the budget minus the known global optimum value. Lower regret indicates better optimisation performance.

The primary comparison examines whether rotation changes the performance of diagonal CMA-ES relative to full-covariance CMA-ES on the four-dimensional ellipsoid with condition number \(10^4\).

Each run should record its method, problem configuration, random seed, evaluation count, evaluated points, objective values, best-so-far values, and relevant diagnostics. Raw results should be preserved separately from processed summaries.

Within each replicate, methods should use the same problem instance but separate method-specific random streams. Experiment settings will be finalised after pilot experiments. The final configurations, implementation choices, and documented deviations will be maintained in `configs/` and the project decision log.

## 5. Data

The finance application uses the FRED series **DCOILWTICO**, representing the daily spot price of West Texas Intermediate crude oil.

- **Source:** [FRED — DCOILWTICO](https://fred.stlouisfed.org/series/DCOILWTICO)
- **Local data file:** To be recorded in the repository once retrieved.
- **Retrieval date:** To be recorded when the data are downloaded.
- **Data processing:** Construct log returns from valid positive prices and explicitly exclude estimation windows containing the non-positive price observation around 20 April 2020.

The downloaded data file and the exact window definitions used in the final experiment will be documented to support reproducibility.

## 6. Running the Experiments

Commands will be added here as the experiment scripts are implemented and tested.

The final instructions will cover:

1. Running unit tests and smoke tests.
2. Running the synthetic benchmark experiments from saved configurations.
3. Running GARCH calibration and forecast evaluation, if completed.
4. Regenerating tables and figures in the report from saved results.

## 7. Project Status

The project is at the implementation and pilot stage. The synthetic study is the highest priority. The finance application is secondary and will be reduced in scope if necessary to complete the core experiments and reproducibility checks.

Experimental settings that depend on pilot results, including the evaluation budget and number of repetitions, will be finalised before the main runs.

## 8. Report

The final report will be provided as a PDF, with a maximum length of 10 pages including figures, references, and appendices, in accordance with the project requirements.
