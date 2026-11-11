# When Does Geometry-Aware Search Pay Off?

**DSA4212 Individual Project**  
*Bayesian optimisation versus evolution strategies on aligned and rotated ill-conditioned problems*

## Project Overview

This project investigates when Bayesian optimisation (BO) and CMA-ES succeed or fail on problems whose difficulty comes from geometry: ill-conditioning and rotation. It tests one explanation for a difference in their behaviour:

> Methods whose internal model is **axis-aligned** (BO with an ARD kernel, diagonal CMA-ES) should lose performance when an ill-conditioned problem is rotated. A method that can represent **any orientation** (full CMA-ES) should not, once it has had enough evaluations to adapt.

The study uses rotated ellipsoids, where conditioning and orientation can be varied independently while the optimum and Hessian are known exactly.

All design decisions, their sources, the pre-stated rules and their outcomes, and every change are recorded in [`decision_log.md`](decision_log.md) (entries D1–D22).

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
| BO, ARD kernel | Per-coordinate lengthscales | Matérn 5/2 ARD Gaussian process, expected improvement (scikit-learn) |
| BO, isotropic kernel | One lengthscale; rotation-invariant | Matérn 5/2 isotropic Gaussian process, expected improvement (scikit-learn) |
| Uniform random search | None; ignores feedback | Independent uniform draws over the box |

No method is tuned per problem. CMA-ES uses pycma defaults with $\sigma_0 = 0.3 \times$ box width and a uniform random start (Hansen, 2023). BO's settings follow Le Riche & Picheny (2021): an initial design of $d+4$ points (maximin Latin hypercube), lengthscale bounds scaled to the box, $2d$ likelihood starts, jitter added only after a failed fit, and EI maximised over space-filling candidates. A pre-registered runtime smoke test set BO's budget ceiling to 250 evaluations with the random-search EI setting (decision log D21).

## Experimental Design

The objective is a rotated ellipsoid, based on BBOB functions f2 and f10:

$$
f(x) = \sum_{i=1}^{d} \kappa^{(i-1)/(d-1)} z_i^2,
\qquad z = R(x-x_{\mathrm{opt}}).
$$

- **Orientation:** $R$ is the identity (aligned) or a Haar-random orthogonal matrix (rotated).
- **Conditioning:** $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$ for the CMA-ES variants and uniform search; $\kappa \in \lbrace 1, 10^4 \rbrace$ for BO. At $\kappa=1$ the ellipsoid is a sphere, so rotation changes nothing; this is the control.
- **Dimension:** $d=4$.
- **Domain:** $[-5,5]^4$ (the BBOB search domain), with the optimum drawn uniformly in $[-4,4]^4$.
- **Replicates:** 50 per condition. Every method faces the same instances; aligned and rotated instances share their optimum; the CMA-ES variants share their start point.
- **Budget:** objective evaluations, enforced exactly by one counting wrapper. Main budget 2,000 evaluations, selected by a rule fixed before the pilot (D11); BO ceiling 250. Results are read at the checkpoints 90, 250, 500, 1,000 and 2,000, and BO is compared with the other methods at shared checkpoints.
- **Costs:** wall-clock time and BO's model-fitting overhead are recorded separately from evaluations.

The Sphere is used only for unit tests and validation.

## Evaluation

The outcome is the simple regret of the best evaluated point, $r = f(\hat{x}) - f(x_{\mathrm{opt}})$ (the BBOB precision $\Delta f$), analysed as $L = \log_{10}(\max(r, 10^{-8}))$ (D14).

Each primary comparison is the **method × orientation interaction** against full CMA-ES at $d=4$, $\kappa=10^4$ (a 2 × 2 factorial interaction; Montgomery, Eq. 6.3, reported unhalved):

$$
\Delta_M =
\left(\bar L_{\mathrm{rot},M} - \bar L_{\mathrm{align},M}\right)
-
\left(\bar L_{\mathrm{rot,full}} - \bar L_{\mathrm{align,full}}\right),
$$

where $M$ is diagonal CMA-ES (at 2,000 evaluations) or ARD BO (at 250, with full CMA-ES read at 250). A positive value means rotation increases the regret of $M$ more than that of full CMA-ES. Replicates are paired by instance, and uncertainty is a 95% paired percentile bootstrap (10,000 replicates, seed 4212).

Mechanism diagnostics: $\kappa_{\mathrm{eff}} = \mathrm{cond}(\Sigma^{1/2} H \Sigma^{1/2})$ for CMA-ES (does the search distribution match the inverse Hessian?) and the fitted ARD lengthscales for BO. Robustness: the floor is varied to $10^{-6}$ and $10^{-10}$. The optional step-size and dimension checks were not run (D22).

## Main Results

| Comparison (κ = 10⁴) | Checkpoint | Δ [95% CI] |
|---|---|---|
| Diagonal CMA-ES vs full CMA-ES (primary) | 2,000 | +8.60 [8.32, 8.84] |
| ARD BO vs full CMA-ES (primary) | 250 | −0.04 [−0.27, 0.21] |
| Diagonal CMA-ES vs full CMA-ES (secondary, shared budget) | 250 | +0.91 [0.62, 1.20] |

The diagonal result holds at every floor (6.60 at $10^{-6}$, 10.60 at $10^{-10}$; its size is set by the floor). The full results, sanity checks and one recorded discrepancy are in the report and in decision log Section 7.

## Repository Structure

```text
dsa4212-bo-cmaes/
├── code/
│   ├── objectives.py        # ellipsoid, Sphere, Haar rotations, Hessian diagnostics (kappa_H, kappa_J)
│   ├── config_loader.py     # loads and validates configs/experiment.yaml for each stage
│   ├── runner.py            # exact-budget wrapper, seeds, CMA-ES variants, uniform search, kappa_eff
│   ├── bo.py                # BO: ARD and isotropic Matérn 5/2, expected improvement (Le Riche & Picheny settings)
│   ├── run_pilot.py         # pilot (sets the budget, floor and invariance check)
│   ├── bo_smoke_test.py     # BO smoke test (verifies BO; sets its ceiling by runtime)
│   ├── run_main.py          # main runs: parallel, resumable, consolidated output
│   ├── analysis.py          # every number in the report -> results/processed/
│   ├── make_figures.py      # every figure in the report -> results/figures/ and report/figures/
│   └── tests/               # 172 unit and end-to-end tests
├── configs/
│   └── experiment.yaml      # every setting, with decision-log references
├── report/                  # LaTeX source of the submitted report, references.bib and figures/
├── results/
│   ├── raw/
│   │   ├── pilot/           # pilot runs (official: 20261103T025402Z)
│   │   ├── bo_smoke/        # smoke test 20261110T095856Z (smoke_report.json)
│   │   └── main/main/       # main runs: summary.csv, evaluations/*.parquet, instances.json, ...
│   ├── processed/           # analysis tables (primary.csv, invariance.csv, kappa_eff.csv, ...)
│   └── figures/             # generated figures (PNG and PDF)
├── readme.md
├── authors.md
├── decision_log.md
└── requirements.txt
```

## Environment and Dependencies

Requires **Python 3.12.4**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used: numpy, scipy, pandas, matplotlib, scikit-learn, cma (pycma 4.5.0), pyyaml, pyarrow and pytest. The experiments were run on macOS on a 10-core Apple Silicon laptop. The run scripts limit each process to one BLAS thread, so parallel runs do not compete for cores.

## Data Preparation

No external data are used, so there is no download or preprocessing step. Every problem instance (optimum, rotation matrix, $\kappa_H$, $\kappa_J$) is generated deterministically from the base seed `4212` in `configs/experiment.yaml`, and is saved with the results (`instances.json`). The pilot, smoke test and main runs use separate seed streams, so their instances never overlap.

## Reproducing the Results

All commands run from the repository root with the environment activated.

### Option A: check the report from the saved results (about 5 minutes)

The raw main-run results are committed, so every table and figure in the report can be regenerated without re-running any optimiser:

```bash
python -m pytest code/tests        # 172 tests, about 1 minute
python code/analysis.py            # regenerates results/processed/; prints the two primary results
python code/make_figures.py        # regenerates results/figures/ and report/figures/
```

`analysis.py` should print:

```text
  DiD cma_es_diagonal vs cma_es_full at 2000: +8.60 [+8.32, +8.84]
  DiD bo_ard vs cma_es_full at 250: -0.04 [-0.27, +0.21]
```

### Option B: re-run everything from scratch (about 2 hours)

Each stage reads its settings from `configs/experiment.yaml`, and writes to a new folder, so earlier results are never overwritten.

```bash
python code/run_pilot.py           # pilot: about 15 s; writes results/raw/pilot/<run_id>/
python code/bo_smoke_test.py       # BO smoke test: about 15 min; writes results/raw/bo_smoke/<run_id>/
python code/run_main.py --run-id rerun    # main runs: about 1.5-2 h on 8 workers; resumable
python code/analysis.py --run-dir results/raw/main/rerun --out results/processed_rerun
python code/make_figures.py --processed results/processed_rerun --run-dir results/raw/main/rerun \
    --figures results/figures_rerun --report-figures results/figures_rerun_report
```

On macOS, prefix long runs with `caffeinate -i` to prevent sleep. If `run_main.py` is interrupted, run the same command again: finished tasks are skipped.

**Expected agreement:** CMA-ES and uniform-search results are deterministic given the seeds. BO fits Gaussian processes with floating-point linear algebra, so on a different machine or BLAS library its trajectories may differ slightly. The pilot was re-run with the final code and reproduced every result exactly (decision log Section 8). Wall-clock times depend on the machine.

## Where Each Part of the Report Comes From

| Report element | File | Produced by |
|---|---|---|
| Primary interactions and floor sensitivity | `results/processed/primary.csv`; `results/figures/fig1_primary_did.pdf` | `analysis.py`; `make_figures.py` |
| Regret curves | `results/processed/regret_curves.csv`; `fig2_regret_curves.pdf` | `analysis.py`; `make_figures.py` |
| Rotation penalty over the budget | `rotation_penalty_checkpoints.csv`; `fig3_penalty_checkpoints.pdf` | `analysis.py`; `make_figures.py` |
| Rotation penalty across κ | `rotation_penalty_kappa.csv`; `fig4_penalty_kappa.pdf` | `analysis.py`; `make_figures.py` |
| κ_J test | `kappa_j.csv`; `fig5_kappa_j.pdf` | `analysis.py`; `make_figures.py` |
| κ_eff diagnostic | `kappa_eff.csv`; `fig6_kappa_eff.pdf` | `analysis.py`; `make_figures.py` |
| BO lengthscales | `lengthscales.csv`; `fig7_lengthscales.pdf` | `analysis.py`; `make_figures.py` |
| κ = 1 invariance control | `invariance.csv` | `analysis.py` |
| Isotropic vs ARD BO | `bo_isotropic_vs_ard.csv` | `analysis.py` |
| Cost and break-even | `cost.csv`, `break_even.csv` | `analysis.py` |
| Stop categories and bound repairs | `stops.csv` | `analysis.py` |
| Uniform-search validation | `uniform_law.csv` | `analysis.py` |
| Headline numbers | `analysis_report.json` | `analysis.py` |
| Main budget, floor, κ = 1 pilot check | `results/raw/pilot/20261103T025402Z/` | `run_pilot.py` |
| BO ceiling and acquisition setting | `results/raw/bo_smoke/20261110T095856Z/smoke_report.json` | `bo_smoke_test.py` |
| Run metadata (software versions, stop counts) | `results/raw/main/main/run_info.json` | `run_main.py` |

Tables are in `results/processed/`; figures are in `results/figures/` and are copied to `report/figures/`, where the report includes them.

## Report

The `report/` folder contains the complete LaTeX source of the submitted report, its bibliography (`references.bib`) and its figures (`report/figures/`, generated by `make_figures.py`). The PDF is at most 10 pages, including references and appendices.

## Reproducibility Safeguards

- Every setting is in `configs/experiment.yaml`; the loader refuses to run a stage while a required setting is missing.
- Settings were frozen before each stage (pilot, smoke test, main runs, analysis) by rules recorded in advance (D20).
- Saved with the results: the resolved configuration, software versions, problem and method seeds, every problem instance, per-evaluation logs (point, value, best-so-far, bound-repair flag), CMA-ES covariance snapshots and BO diagnostics.
- No run was silently rerun or omitted; every change is recorded in the decision log (Section 8).

## Project Status

**Complete:** all code (172 tests), pilot, BO smoke test, main runs (2,400 runs), analysis and figures. 