# Experiment Decision Log

This file records methodological decisions made before running the main experiments. Decisions should be documented before inspecting final results. Settings that depend on pilot experiments will be finalised and recorded before the main runs.

## D001 — Optimisation budget

- **Decision:** The evaluation budget will be finalised after pilot experiments and before the main runs.
- **Rationale:** All methods must be compared under a consistent objective-evaluation budget.
- **Protocol:** Bayesian optimisation's initial design counts towards the budget. Every objective call counts as one evaluation, and the final CMA-ES batch must be truncated if necessary to respect the exact budget.
- **Status:** Pending pilot results.

## D002 — Random seeds and reproducibility

- **Decision:** Use a fixed base seed of `4212` and derive reproducible random seeds for problem instances and optimisation methods.
- **Rationale:** Ensure reproducibility and fair comparisons across methods.
- **Protocol:** Methods share the same problem instance within each replicate but use separate method-specific random streams. Record all seeds and save the experiment configuration, problem instances, and software versions.
- **Status:** Base seed and seed strategy specified; implementation and validation pending.

## D003 — Objective functions

- **Main experiment:** Aligned and randomly rotated ellipsoid functions.
- **Dimension:** \(d=4\).
- **Condition numbers:** \(\kappa\in\{1,100,10{,}000\}\).
- **Supporting experiments:** Sphere for implementation and optimisation sanity checks; Rastrigin at dimensions 4 and 10 as a supporting comparison.
- **Rationale:** The ellipsoid experiment examines the effects of rotation and conditioning on optimisation performance.
- **Status:** Main experimental conditions specified in the configuration; domain, optimum placement, and other pilot-dependent settings remain to be finalised.

## D004 — Optimisation methods

The following methods will be compared:

- Bayesian optimisation using a Gaussian process with an ARD Matérn 5/2 kernel and expected improvement.
- Full-covariance CMA-ES.
- Diagonal-covariance CMA-ES.
- Isotropic CMA-ES.
- Uniform random search.

- **Rationale:** Compare Bayesian optimisation and evolutionary/search baselines under the same objective functions and evaluation budgets.
- **Status:** Methods specified; implementation and validation pending.

## D005 — Evaluation metrics

- **Primary outcome:** Final simple regret, defined as the best evaluated objective value minus the known global optimum value.
- **Convention:** Lower objective values and lower simple regret are better.
- **Primary comparison:** Difference-in-differences comparing the rotation-related change in mean log-simple-regret for diagonal CMA-ES against the corresponding change for full CMA-ES at \(d=4\) and \(\kappa=10{,}000\).
- **Supporting metrics:** Best objective value, evaluations to target, and method-specific diagnostics.
- **Uncertainty:** Use paired comparisons and bootstrap confidence intervals.
- **Status:** Primary metric and comparison specified; log-regret floor and remaining implementation details to be finalised using pilot results.

## D006 — Pilot experiments and configuration freeze

- **Decision:** Run small pilot experiments before the main experimental runs.
- **Objectives:** Verify objective functions, rotation generation, evaluation-budget enforcement, random-seed reproducibility, method implementations, boundary handling, and runtime.
- **Constraint:** Freeze the experimental configuration before the main runs. Any subsequent methodological changes must be documented with a rationale.
- **Status:** Pending.

## D007 — Evaluation logging and raw results

- **Decision:** Save raw evaluation-level results and the configuration used for each run.
- **Protocol:** Record each evaluated point, objective value, best-so-far value, random seed, and relevant boundary-handling information. Save covariance snapshots and Gaussian-process hyperparameters as separate diagnostic records.
- **Rationale:** Support reproducibility, performance analysis, and investigation of optimisation behaviour.
- **Status:** Specified; implementation pending.
