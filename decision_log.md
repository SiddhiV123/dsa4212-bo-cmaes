# Experiment Decision Log

This file records methodological decisions made before running the main
experiments. Decisions should be documented before inspecting final results.

## D001 — Optimisation budget
- Decision: To be finalised before the pilot experiments.
- Rationale: All methods must be compared under a consistent evaluation budget.
- Status: Pending.

## D002 — Random seeds
- Decision: To be finalised before the pilot experiments.
- Rationale: Reproducibility and fair comparisons across methods.
- Status: Pending.

## D003 — Objective functions
- Main experiment: Rotated and unrotated ellipsoid functions with varying
  condition numbers.
- Supporting experiments: Sphere and Rastrigin functions.
- Status: Based on the project proposal.

## D004 — Optimisation methods
- Bayesian optimisation using an ARD Gaussian process.
- Full-covariance CMA-ES.
- Diagonal-only CMA-ES.
- Uniform random search.
- Gaussian random search.
- Status: Based on the project proposal.

## D005 — Evaluation metric
- Primary outcome: Best objective value found within the evaluation budget.
- Convention: Lower objective values are better.
- Status: To be confirmed against the final experimental protocol.

## D006 — Pilot experiments
- Decision: Run small pilot experiments to verify implementation and runtime.
- Constraint: Freeze the experimental configuration before the main runs.
- Status: Pending.