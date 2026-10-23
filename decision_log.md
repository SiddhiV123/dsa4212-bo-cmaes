# Decision Log — DSA4212 Project

**Project:** When Does Geometry-Aware Search Pay Off?  
**Proposal version:** v9, 22 October 2026  
**Last updated:** 24 October 2026  
**Status:** Implementation and pilot preparation

This document records the main experimental design decisions, their rationale, alternatives considered, and the settings that remain unresolved. Decisions marked *provisional* must be reviewed after the pilot and frozen before the main experiments.

---

## 1. Project Scope and Priorities

### D1. Use the ellipsoid study as the primary contribution

**Decision:** The primary study compares Bayesian optimisation (BO), full-covariance CMA-ES, diagonal CMA-ES, isotropic evolution strategy (ES), and uniform random search on controlled synthetic objectives.

**Rationale:** The ellipsoid provides a known optimum and allows conditioning and orientation to be varied independently. This makes it possible to study how the geometry of an objective interacts with an optimiser's search strategy.

The GARCH(1,1) finance application is secondary and will be reduced or cut first if implementation time becomes limited.

**Status:** Accepted.

### D2. Keep the finance application focused on GARCH(1,1) calibration

**Decision:** Use Gaussian quasi-maximum likelihood estimation (QMLE) on rolling windows of WTI returns rather than the original moving-average trading-strategy application.

**Alternatives:** Moving-average crossover strategy with a Sharpe-ratio objective and a grid-search reference.

**Rationale:** The moving-average problem has about two integer parameters, so an exhaustive grid is exact and instant, and it has no conditioning to study. GARCH estimation provides a continuous, constrained optimisation problem with potentially challenging curvature along the ARCH/GARCH parameter ridge. This is more directly related to the project's interest in conditioning and optimiser geometry.

The finance application is a model-calibration study, not a trading backtest, and makes no claim about trading profitability.

**Status:** Accepted.

---

## 2. Synthetic Objective Design

### D3. Use a four-dimensional ellipsoid as the primary benchmark

**Decision:** Set the primary dimension to $d=4$, with condition numbers $\kappa \in \lbrace 1, 100, 10000 \rbrace$. Compare aligned and randomly rotated versions of each condition.

The objective is

$$
f(x)=\sum_{i=1}^{d}
\kappa^{(i-1)/(d-1)}z_i^2,
\qquad z=R(x-x_{\mathrm{opt}}).
$$

Here, $R=I$ for aligned instances and $R$ is an orthogonal matrix for rotated instances. The known optimum is $x_{\mathrm{opt}}$, with objective value zero.

**Rationale:** This design varies conditioning and orientation while keeping the objective smooth, unimodal, and analytically understood. The $\kappa=1$ case acts as an orientation-invariance control because the objective is spherical regardless of the orthogonal rotation.

**Status:** Accepted.

### D4. Generate rotations using random orthogonal matrices

**Decision:** Generate rotated instances using QR decomposition with sign correction. Save the resulting rotation matrix for each problem instance.

**Rationale:** The rotation changes the orientation of the principal axes without changing the eigenvalues of the Hessian. Saving the matrix ensures that each instance can be reproduced exactly.

**Status:** Accepted; implementation and tests pending.

### D5. Randomise the optimum independently of optimiser initialisation

**Decision:** Sample the optimum uniformly within a configured sub-box of the search domain. Optimiser initial points must be generated independently of the optimum.

The sub-box fraction remains to be determined.

**Rationale:** A fixed optimum at the origin could unintentionally favour methods whose initialisation is centred or whose sampling scheme is symmetric around the origin. Independent randomisation reduces this source of bias.

**Status:** Design accepted; sub-box fraction provisional.

### D6. Use Sphere for supporting validation; cut other extensions

**Decision:** Retain Sphere objectives in dimensions $d \in \lbrace 2, 4, 10 \rbrace$ for unit tests, method-interface checks, and sanity checks.

The following are not run in the current scope: Rastrigin, the GARCH penalty arm, the noise study, and the optional $d=10$ ellipsoid run.

**Rationale:** Sphere is useful for validating objective evaluation, optimiser interfaces, and basic behaviour before running the main experiments. Rastrigin would add a multimodal study that is not essential to the primary rotation-and-conditioning question. The penalty arm, noise study, and $d=10$ ellipsoid run are optional extensions that are cut before the core study is touched.

**Status:** Accepted. Any reinstatement requires the core study to be complete and a recorded change.

---

## 3. Optimisation Methods

### D7. Compare five optimisation methods

**Decision:** Include the following methods:

1. Bayesian optimisation with an ARD Matérn 5/2 Gaussian process and expected improvement.
2. Full-covariance CMA-ES.
3. Diagonal CMA-ES.
4. Isotropic ES, implemented as CMA-ES with covariance adaptation disabled.
5. Uniform random search.

Every method reports its best *evaluated* point. BO may use its surrogate to choose the next evaluation, but its reported solution must always be a point that was actually evaluated.

**Alternatives:** A fixed Gaussian random-search arm centred on the CMA-ES starting point, proposed earlier and dropped because it had no published precedent.

**Rationale:** Full and diagonal CMA-ES provide the central comparison for understanding the role of covariance adaptation. Isotropic ES helps distinguish covariance learning from step-size adaptation, while uniform random search provides a baseline without objective-guided candidate selection. BO adds a surrogate-based optimisation approach. A common recommendation rule keeps the comparison about optimisation rather than about how each method reports its answer.

**Status:** Accepted; implementation details for BO and CMA-ES require smoke testing.

### D8. Use separable diagonal CMA-ES

**Decision:** Use a diagonal-only CMA-ES variant, rather than a method such as dd-CMA that retains a full covariance representation.

**Alternatives:** Diagonal-decoding CMA-ES (dd-CMA).

**Rationale:** The objective is to test how restricting covariance adaptation affects performance under rotation. A method that retains a full covariance representation would not isolate this distinction as directly.

**Status:** Accepted; installed library options must be verified.

### D9. Use common settings across CMA-ES variants

**Decision:** Define shared CMA-ES settings in `cma_es_common` and merge them into the full, diagonal, and isotropic variants through the configuration loader.

Use a common initial-point policy, a common provisional initial step-size rule, and matched evaluation budgets.

**Rationale:** Differences in performance should primarily reflect the covariance-adaptation mechanism rather than unrelated initialisation or budget differences.

The `inherits` field in the YAML file is a project convention, not native YAML inheritance. The implementation must explicitly merge the common settings and test the resolved configurations.

**Status:** Design accepted; loader and merge test pending.

### D10. Initialise optimisers without information about the optimum

**Decision:** Draw CMA-ES initial points uniformly from the search box, independently of the optimum. Use the same initial point across the three CMA-ES variants within a matched problem instance.

The initial step size is provisionally set to 0.3 times the search-box width.

**Rationale:** This supports paired comparisons and avoids giving one method privileged knowledge of the optimum. The initial step-size rule will be checked during the pilot.

**Status:** Initialisation policy accepted; numerical settings provisional.

### D11. Handle early termination transparently

**Decision:** Attempt to run methods to the prescribed evaluation budget. If a CMA-ES run terminates early despite the configured stopping policy, preserve its best evaluated point, flag the run as incomplete, and report the actual evaluation count. Incomplete runs are retained in the analysis, and counts are reported by method and condition.

**Rationale:** An early-terminated run must not be presented as if it consumed the full budget. On a noise-free objective, early termination often reflects convergence, so excluding these runs would bias results against the fastest-converging methods. Keeping the run and its status visible avoids silently discarding inconvenient outcomes.

**Status:** Accepted; stopping options must be verified against the installed `cma` version.

---

## 4. Evaluation Budget and Reproducibility

### D12. Enforce budgets at the objective-evaluation level

**Decision:** Route every objective evaluation through a runner-level counting wrapper. Count initial-design evaluations for BO and truncate a final CMA-ES batch when necessary to respect the budget exactly.

**Rationale:** Comparing methods by iterations rather than objective evaluations would not provide a fair comparison because one iteration can involve different numbers of evaluations across methods.

**Status:** Accepted; implementation pending.

### D13. Pair methods on the same problem instances

**Decision:** Use a fixed base seed of `4212`. Derive each problem seed, which determines $x_{\mathrm{opt}}$ and $R$, from the condition and replicate index only, so that all methods within a replicate face the same problem instance. Derive method-specific random seeds separately. Save the problem instances, seeds, software versions, and actual evaluation counts.

The provisional target is 20 replicates per condition.

**Rationale:** Pairing reduces variation attributable to differences between randomly generated problems. Separate method seeds preserve reproducibility without forcing every method to consume identical random-number streams.

**Status:** Design accepted; replicate count provisional.

### D14. Record evaluation-level diagnostics

**Decision:** Record candidate locations, objective values, best-so-far values, evaluation numbers, boundary events, method identifiers, and seeds.

Save CMA-ES covariance snapshots and BO hyperparameter diagnostics, including ARD lengthscales.

**Rationale:** Final regret alone describes performance but does not explain how the optimisers behave. These diagnostics support the project's mechanism analysis.

**Status:** Accepted; logging format and snapshot frequency pending implementation.

---

## 5. Primary Analysis

### D15. Use final simple regret as the primary outcome

**Decision:** Define final simple regret as

$$
r=f(x_{\mathrm{best}})-f(x_{\mathrm{opt}}),
$$

where $x_{\mathrm{best}}$ is the best evaluated point (D7). Analyse $\log_{10}(r)$ using a small positive floor selected during the control pilot.

**Rationale:** Simple regret measures the quality of the best solution found within the evaluation budget. A logarithmic scale helps compare performance across objectives and conditions with substantially different regret magnitudes.

The floor must be chosen before the main analysis and applied consistently.

**Status:** Outcome accepted; floor pending pilot.

### D16. Use a difference-in-differences comparison for the primary hypothesis

**Decision:** At $d=4$ and $\kappa=10000$, compare diagonal and full CMA-ES using

$$
\begin{aligned}
\Delta_{\mathrm{DiD}}
={}&(\bar L_{\mathrm{rot,diag}}
-\bar L_{\mathrm{align,diag}})\\
&-(\bar L_{\mathrm{rot,full}}
-\bar L_{\mathrm{align,full}}),
\end{aligned}
$$

where $L$ denotes log regret.

**Rationale:** This measures whether rotation increases log regret more for diagonal CMA-ES than for full-covariance CMA-ES, while accounting for the aligned-condition performance of each method.

A positive value indicates a larger rotation penalty for diagonal CMA-ES.

**Status:** Accepted.

### D17. Use paired uncertainty estimates and keep secondary comparisons exploratory

**Decision:** Pair observations by problem instance and report bootstrap confidence intervals. The bootstrap replicate count remains to be finalised.

Other condition numbers, BO comparisons, intermediate checkpoints, and evaluations-to-target are exploratory analyses.

**Rationale:** The primary comparison should remain clear and interpretable. Exploratory analyses can provide supporting evidence without being presented as equally central hypotheses.

**Status:** Analysis approach accepted; bootstrap settings pending.

### D18. Use the $\kappa=1$ control to assess orientation invariance

**Decision:** Estimate an invariance tolerance from the observed aligned-versus-rotated variation at $\kappa=1$, by method, during the control pilot.

**Rationale:** When $\kappa=1$, orthogonal rotation should not change the mathematical objective. Small numerical or sampling-related differences should be interpreted against a pre-specified tolerance rather than an arbitrary threshold.

**Status:** Concept accepted; tolerance pending pilot.

### D19. Use mechanism diagnostics to interpret results

**Decision:** Examine the effective condition number

$$
\kappa_{\mathrm{eff}}
=\mathrm{cond}(C^{1/2}HC^{1/2}),
$$

where $C$ is the learned CMA-ES covariance and $H$ is the objective Hessian. Also examine BO ARD lengthscales relative to curvature in aligned cases.

**Rationale:** These diagnostics help assess whether the learned search geometry is related to the objective geometry. They support interpretation but do not, by themselves, establish causation.

**Status:** Accepted; implementation and validation pending.

---

## 6. Finance Application: Data and Preprocessing

### D20. Use WTI spot prices from FRED

**Decision:** Use the FRED series `DCOILWTICO`, originally sourced from the U.S. Energy Information Administration. Store a frozen copy at `data/DCOILWTICO.csv` and record its retrieval date.

Source: https://fred.stlouisfed.org/series/DCOILWTICO

**Rationale:** A frozen dataset prevents later source revisions from silently changing the experiment.

**Status:** Source selected; download and freeze pending.

### D21. Define returns and handle non-positive prices explicitly

**Decision:** Calculate percentage log returns as

$$
r_t=100\log\left(\frac{P_t}{P_{t-1}}\right),
$$

then demean returns using the estimation-window mean. Drop missing observations according to the configured preprocessing policy.

Exclude any estimation window containing the non-positive price observation dated 20 April 2020, and record the exclusion.

**Rationale:** Log returns are undefined when a price is non-positive. Explicitly excluding affected windows avoids invalid calculations and makes the data-handling decision auditable.

**Status:** Accepted; data validation pending.

---

## 7. GARCH(1,1) Model and Analysis

### D22. Use Gaussian QMLE with a constrained reparameterisation

**Decision:** Estimate a GARCH(1,1) model using Gaussian quasi-maximum likelihood. Optimise over $(\log\omega, p, s)$, with

$$
p=\alpha+\beta,\qquad
s=\frac{\alpha}{\alpha+\beta},
$$

and recover the original parameters using

$$
\alpha=ps,\qquad
\beta=p(1-s),\qquad
\omega=e^{\log\omega}.
$$

Use $0<p_{\min}<p_{\max}<1$ and $s\in[0,1]$. Determine the numerical bounds for $p$ and $\log\omega$ using documented rules and freeze them before the main finance runs.

**Rationale:** This parameterisation supports nonnegative ARCH and GARCH coefficients and enforces the stationarity restriction $\alpha+\beta<1$. Allowing $s=0$ or $s=1$ permits one of the two coefficients to be zero.

**Status:** Parameterisation accepted; numerical bounds pending validation.

### D23. Use rolling estimation windows

**Decision:** Use 750 trading days per estimation window, moving forward by 250 trading days.

**Rationale:** Rolling windows create multiple calibration instances and allow the analysis to examine whether optimisation behaviour varies with the curvature of the fitted likelihood.

**Status:** Accepted; eligible windows to be determined after data preprocessing.

### D24. Validate the custom likelihood against `arch`

**Decision:** Compare the custom GARCH likelihood with the `arch` package on at least three estimation windows before scaling up the finance experiments.

Use the same initial-variance convention in both calculations: override `arch`'s default backcast (an exponentially weighted average of early squared residuals) with the window sample variance. Verify that the variance recursion agrees before comparing negative log-likelihood values.

The custom implementation must document and test the variance-bound rule used in the comparison.

**Rationale:** Differences in initialisation or variance bounds can produce different likelihood values even when the model equations are otherwise equivalent. These differences must be investigated before attributing discrepancies to optimisation performance.

**Status:** Required validation; not yet completed.

### D25. Define likelihood gaps within each estimation window

**Decision:** After all main runs are complete, define each window's best observed negative log-likelihood using the pre-specified set of completed main runs and the reference optimiser. Exclude pilot runs.

For a method's estimate $\hat\theta$, define

$$
g=\mathrm{NLL}(\hat\theta)
-\mathrm{NLL}_{\mathrm{best,window}}.
$$

Analyse $\log_{10}(g+\epsilon)$, with a positive $\epsilon$ frozen before final analysis. Record signed gaps to the L-BFGS-B reference separately and flag cases where a method outperforms that reference.

**Rationale:** A per-window reference avoids mixing likelihood levels across different data windows. A method that outperforms the reference optimiser should be flagged rather than hidden or automatically treated as an error.

**Status:** Analysis definition accepted; epsilon and reference-optimiser settings pending.

### D26. Test the relationship between conditioning and optimiser performance

**Decision:** Use the Jacobi-scaled Hessian condition number $\kappa_J$ as the primary predictor, with the ordinary Hessian condition number $\kappa_H$ as a secondary predictor. Estimate the Hessian by finite differences at the reference optimum in the main arm's unit-box coordinates.

Relate conditioning to the diagonal-minus-full CMA-ES log-gap using Spearman correlation and moving-block bootstrap uncertainty, with a block length of at least 3 windows.

**Rationale:** $\kappa_J$ measures the conditioning that remains after Jacobi (diagonal) rescaling, which is within a factor of $d$ of the best possible diagonal rescaling. This is the conditioning a diagonal model cannot remove. Aligned problems have $\kappa_J=1$, however large $\kappa$ is, so $\kappa_J$, not $\kappa_H$, is the quantity H3 predicts will drive the diagonal-minus-full gap. Consecutive windows overlap by two-thirds, so a block bootstrap is needed to respect the dependence.

Boundary optima will be handled with one-sided finite differences and flagged.

**Status:** Analysis approach accepted; implementation pending.

### D27. Keep the forecast analysis secondary and exploratory

**Decision:** Evaluate 250 trading days of one-step-ahead variance forecasts using QLIKE, holding estimated parameters fixed. Use squared demeaned returns as the variance proxy and compare against the reference-optimum QLIKE.

**Rationale:** This provides a secondary assessment of the practical implications of calibration differences without turning the project into a separate forecasting or trading study.

**Status:** Planned secondary analysis.

---

## 8. Configuration and Implementation Rules

### D28. Keep unresolved values explicit

**Decision:** Represent undecided settings as `null` in `configs/experiment.yaml` and mark provisional values clearly. The configuration loader must validate required settings before the relevant experiment can run.

**Rationale:** Explicit unresolved settings are safer than silently substituting defaults or treating missing values as zero.

**Status:** Accepted.

### D29. Freeze the main experiment settings after the pilot

**Decision:** Do not start the main runs until the pilot is complete, the configuration has been reviewed, and the required tests and validation checks have passed.

The freeze checklist includes:

- CMA-ES option and stopping-policy verification.
- Common CMA-ES configuration merge tests.
- Objective and rotation tests.
- Evaluation-budget and boundary-handling policies.
- Log-regret floor and invariance tolerance.
- Seed policy.
- GARCH likelihood validation before scaling up the finance study.

**Rationale:** Pilot results may be used to select numerical settings, but those settings must be documented and frozen before the main analysis to avoid outcome-driven changes.

**Status:** Accepted; freeze conditions not yet satisfied.

---

## 9. Outstanding Decisions

The following settings remain open and should be resolved in the order needed for implementation and pilot execution.

| Item | Outstanding decision | Required before |
|---|---|---|
| Synthetic domain | Boundary-handling policy | Synthetic pilot |
| Optimum placement | Sub-box fraction | Synthetic pilot |
| CMA-ES | Installed version, population size, stopping options, covariance verification | Main synthetic runs |
| CMA-ES | Initial step-size rule after pilot | Main synthetic runs |
| Bayesian optimisation | Library, initial design, restart counts, fixed jitter | BO smoke test and main runs |
| Evaluation budget | Evaluations per method and checkpoint schedule | Main synthetic runs |
| Replicates | Confirm or revise the provisional target of 20 | Main synthetic runs |
| Primary analysis | Log-regret floor, bootstrap replicate count, invariance tolerance | Main analysis |
| Finance data | Download, validate, freeze CSV, record retrieval date and column names | GARCH validation |
| GARCH | Numerical parameter bounds and initial-variance implementation | Likelihood validation |
| GARCH | Verify variance-bound compatibility with `arch` | Finance optimisation |
| Finance optimisation | Per-method budget, seed policy, CMA-ES initial-point rule, reference multi-start settings | Main finance runs |
| Finance analysis | Positive log-gap epsilon and moving-block bootstrap settings | Final finance analysis |

---

## 10. Change-Control Rule

Any change to a decision above should be recorded here with:

- **Date**
- **Decision changed**
- **Previous setting**
- **New setting**
- **Reason**
- **Whether the pilot or main runs had already started**
- **Effect on comparability or interpretation**

Changes made after the main runs begin must be documented explicitly. Runs should not be silently rerun or omitted because their results are inconvenient.

The final report should describe the settings actually used, including justified deviations from the original plan.