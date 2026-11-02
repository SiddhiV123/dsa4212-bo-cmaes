# Decision Log — DSA4212 Project

**Project:** When Does Geometry-Aware Search Pay Off?  
**Proposal version:** v9, 22 October 2026  
**Last updated:** 2 November 2026  
**Citation style:** APA 7th edition (author–date). Full references in Section 11.  
**Status:** Pilot preparation (synthetic study only; finance application removed, D31)

This document records the main experimental design decisions, their rationale, alternatives considered, and the settings that remain unresolved. Decisions marked *provisional* must be reviewed after the pilot and frozen before the main experiments.

---

## 1. Project Scope and Priorities

### D1. Use the ellipsoid study as the primary contribution

**Decision:** The primary study compares Bayesian optimisation (BO), full-covariance CMA-ES, diagonal CMA-ES, isotropic evolution strategy (ES), and uniform random search on controlled synthetic objectives.

**Rationale:** The ellipsoid provides a known optimum and allows conditioning and orientation to be varied independently. This makes it possible to study how the geometry of an objective interacts with an optimiser's search strategy.

The GARCH(1,1) finance application is secondary and will be reduced or cut first if implementation time becomes limited.

The finance application was later removed from scope (D31), so the synthetic study is the whole project.

**Status:** Accepted.

### D2. Keep the finance application focused on GARCH(1,1) calibration

**Decision:** Use Gaussian quasi-maximum likelihood estimation (QMLE) on rolling windows of WTI returns rather than the original moving-average trading-strategy application.

**Alternatives:** Moving-average crossover strategy with a Sharpe-ratio objective and a grid-search reference.

**Rationale:** The moving-average problem has about two integer parameters, so an exhaustive grid is exact and instant, and it has no conditioning to study. GARCH estimation provides a continuous, constrained optimisation problem with potentially challenging curvature along the ARCH/GARCH parameter ridge. This is more directly related to the project's interest in conditioning and optimiser geometry.

The finance application is a model-calibration study, not a trading backtest, and makes no claim about trading profitability.

**Status:** Cut (D31).

---

## 2. Synthetic Objective Design

### D3. Use a four-dimensional ellipsoid as the primary benchmark

**Decision:** Set the primary dimension to $d=4$, with condition numbers $\kappa \in \lbrace 1, 100, 10000 \rbrace$ (extended to a five-level ladder in D32). Compare aligned and randomly rotated versions of each condition.

The objective is

$$
f(x)=\sum_{i=1}^{d}
\kappa^{(i-1)/(d-1)}z_i^2,
\qquad z=R(x-x_{\mathrm{opt}}).
$$

Here, $R=I$ for aligned instances and $R$ is an orthogonal matrix for rotated instances. The known optimum is $x_{\mathrm{opt}}$, with objective value zero.

**Rationale:** This design varies conditioning and orientation while keeping the objective smooth, unimodal, and analytically understood. The $\kappa=1$ case acts as an orientation-invariance control because the objective is spherical regardless of the orthogonal rotation.

**Source and deviation:** The objective follows BBOB functions f2 and f10, the separable and rotated ellipsoids (Hansen et al., 2009, Sections 1.2 and 3.10, Equations 2 and 10). The fixed conditioning $10^6$ is generalised to a parameter $\kappa$, as in pycma's `cma.ff.elli(cond=...)` (Hansen et al., 2019), because the research question concerns how rotation interacts with conditioning. The oscillation transform $T_{\mathrm{osz}}$ is omitted so that the objective is an exact quadratic with a known Hessian. The implementation is cross-checked against `cma.ff.elli` in the unit tests.

**Status:** Accepted.

### D4. Generate rotations using random orthogonal matrices

**Decision:** Generate rotated instances using QR decomposition with sign correction. Save the resulting rotation matrix for each problem instance.

**Rationale:** The rotation changes the orientation of the principal axes without changing the eigenvalues of the Hessian. Saving the matrix ensures that each instance can be reproduced exactly. QR decomposition of a Gaussian matrix is not Haar-distributed on its own; multiplying the columns of $Q$ by the signs of the diagonal of $R$ makes it so (Mezzadri, 2007).

**Status:** Accepted; implemented and tested (orthogonality, reproducibility, and a check that the sign correction removes the bias).

### D5. Randomise the optimum independently of optimiser initialisation

**Decision:** Sample the optimum uniformly within a configured sub-box of the search domain. Optimiser initial points must be generated independently of the optimum.

**Pilot setting:** sub-box fraction 0.8, so $x_{\mathrm{opt}}$ is uniform in $[-4,4]^d$, following the BBOB convention (Long et al., 2022).

**Alternatives:** placing the optimum anywhere in the full domain $[-5,5]^d$, which avoids favouring centre-focused methods (Vermetten et al., 2023). The sub-box is kept because it also keeps the optimum away from the region where bound handling distorts the search; the centre-bias point is noted as a limitation.

**Rationale:** A fixed optimum at the origin could unintentionally favour methods whose initialisation is centred or whose sampling scheme is symmetric around the origin. Independent randomisation reduces this source of bias.

**Status:** Design accepted; sub-box fraction 0.8 for the pilot, to be confirmed at the freeze.

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

**Status:** Accepted; extended by D32 (BO with an isotropic kernel added; BO comparisons made co-primary).

### D8. Use separable diagonal CMA-ES

**Decision:** Use separable CMA-ES (Ros & Hansen, 2008), run as `CMA_diagonal=True` in pycma (Hansen et al., 2019), rather than a method that retains a full covariance representation.

**Alternatives:** Diagonal-decoding CMA-ES (dd-CMA; Akimoto & Hansen, 2020).

**Rationale:** The objective is to test how restricting covariance adaptation affects performance under rotation. A method that retains a full covariance representation would not isolate this distinction as directly.

**Verification:** In pycma 4.5.0 the diagonal variant samples from an identity matrix and adapts a per-coordinate scaling vector, so off-diagonal covariance entries are zero by construction. A unit test confirms this at every checkpoint.

**Status:** Accepted; verified on the installed version.

### D9. Use common settings across CMA-ES variants

**Decision:** Define shared CMA-ES settings in `cma_es_common` and merge them into the full, diagonal, and isotropic variants through the configuration loader.

Use a common initial-point policy, a common provisional initial step-size rule, and matched evaluation budgets.

**Rationale:** Differences in performance should primarily reflect the covariance-adaptation mechanism rather than unrelated initialisation or budget differences.

The `inherits` field in the YAML file is a project convention, not native YAML inheritance. The implementation must explicitly merge the common settings and test the resolved configurations.

**Verification:** The configuration loader rejects any variant that overrides a shared setting, and checks that `CMA_diagonal: true` and `CMA_on: 0` survive the merge. For isotropic ES, a unit test confirms the internal covariance stays at the identity (condition number below 1.001).

**Status:** Accepted; loader and merge tests implemented.

### D10. Initialise optimisers without information about the optimum

**Decision:** Draw CMA-ES initial points uniformly from the search box, independently of the optimum. Use the same initial point across the three CMA-ES variants within a matched problem instance.

The initial step size is $\sigma_0 = 0.3\,(b-a)$, following Hansen's rule for a uniform start in $[a,b]^n$ (Hansen, 2023, Appendix A, Figure 6, footnote 1). For the box $[-5,5]^4$ this gives $\sigma_0 = 3$.

**Rationale:** This supports paired comparisons and avoids giving one method privileged knowledge of the optimum. The step-size rule is fixed and is not changed on the basis of pilot results (D30).

**Status:** Accepted.

### D11. Handle early termination transparently

**Decision:** Attempt to run methods to the prescribed evaluation budget. If a CMA-ES run terminates early despite the configured stopping policy, preserve its best evaluated point, flag the run as incomplete, and report the actual evaluation count. Incomplete runs are retained in the analysis, and counts are reported by method and condition.

**Rationale:** An early-terminated run must not be presented as if it consumed the full budget. On a noise-free objective, early termination often reflects convergence, so excluding these runs would bias results against the fastest-converging methods. Keeping the run and its status visible avoids silently discarding inconvenient outcomes.

**Stop categories:** pycma's tolerance-based stops (`tolfun`, `tolfunrel`, `tolfunhist`, `tolx`, `tolstagnation`, `tolflatfitness`, `tolxstagnation`) are disabled. Two checks, `noeffectaxis` and `noeffectcoord`, cannot be disabled; they are numerical-stability criteria that fire when a step no longer changes the mean (Hansen, 2023, Appendix B.3). Runs that stop only on these are recorded with `stop_category = "numerical_precision"`, kept in the analysis, and their best value is carried to later checkpoints. Any other early stop is recorded as `stop_category = "other"` and investigated.

**Status:** Accepted; verified on pycma 4.5.0.

---

## 4. Evaluation Budget and Reproducibility

### D12. Enforce budgets at the objective-evaluation level

**Decision:** Route every objective evaluation through a runner-level counting wrapper. Count initial-design evaluations for BO and truncate a final CMA-ES batch when necessary to respect the budget exactly.

**Rationale:** Comparing methods by iterations rather than objective evaluations would not provide a fair comparison because one iteration can involve different numbers of evaluations across methods.

**Provisional pilot budget:** The original 500-evaluation ceiling was a pragmatic choice rather than a value derived from the literature. The pilot uses a provisional ceiling of 2,000 evaluations ($500d$ at $d=4$), with checkpoints at 90, 250, 500, 1,000 and 2,000. An informal pre-pilot check (10 seeds per condition, target $f(x) \le 10^{-8}$) found that full CMA-ES needed about 750–1,420 evaluations on the tested $\kappa=10^4$ instances, while diagonal CMA-ES did not reach the target on any rotated instance within 20,000; this is exploratory, not confirmatory. The ceiling lies within the pycma documentation's rule of thumb of $100n$ to $300n^2$ evaluations for hard problems (Hansen et al., 2019). The 90-evaluation checkpoint equals $10d+50$, the budget Santoni et al. (2024) used for their structural-mechanics applications, a domain-specific precedent.

**Selection rule (fixed before the pilot is analysed):**
1. Set the log-regret floor by the rule in D15 and freeze it, together with the rule for when a replicate counts as having reached it.
2. Select the smallest checkpoint at which full CMA-ES on the rotated $\kappa=10^4$, $d=4$ ellipsoid reaches the floor in at least 80% of pilot replicates. If none does, select 2,000.
3. At the selected checkpoint, check feasibility only: runtime, incomplete-run counts, and whether BO can be run. The size of the diagonal-versus-full difference is not inspected for this decision.
4. Record the final budget and rationale here before the confirmatory main runs. Pilot runs are excluded from confirmatory analysis.

**Status:** Enforcement implemented and tested; budget selected by the rule above.

### D13. Pair methods on the same problem instances

**Decision:** Use a fixed base seed of `4212`. Derive each problem seed, which determines $x_{\mathrm{opt}}$ and $R$, from the condition and replicate index only, so that all methods within a replicate face the same problem instance. Derive method-specific random seeds separately. Save the problem instances, seeds, software versions, and actual evaluation counts.

The optimum is drawn before the rotation, so the aligned and rotated instances in a replicate share $x_{\mathrm{opt}}$ and differ only in orientation.

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

**Log-regret floor rule (fixed before the pilot is run):** The floor is $10^{-8}$, the smallest of COCO's standard target precisions, which run from $10^2$ to $10^{-8}$ (Hansen et al., 2021). The $\kappa=1$ control runs are used only to confirm reachability: if every CMA-ES variant reaches $10^{-8}$ in all $\kappa=1$ pilot replicates, the floor is frozen at $10^{-8}$; otherwise it is the largest final regret attained across the $\kappa=1$ CMA-ES runs. A replicate has reached the floor when its best-so-far regret at a checkpoint is $\le$ the floor; regret below the floor is recorded as the floor.

**Sensitivity analysis:** the primary DiD is also reported at floors of $10^{-6}$ and $10^{-10}$.

**Status:** Outcome and floor rule accepted; floor confirmed at the pilot.

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

**Status:** Accepted; a second primary DiD for BO (ARD) is added by D32.

### D17. Use paired uncertainty estimates and keep secondary comparisons exploratory

**Decision:** Pair observations by problem instance and report bootstrap confidence intervals. The bootstrap replicate count remains to be finalised.

Other condition numbers, BO comparisons, intermediate checkpoints, and evaluations-to-target are exploratory analyses.

**Rationale:** The primary comparison should remain clear and interpretable. Exploratory analyses can provide supporting evidence without being presented as equally central hypotheses.

**Status:** Analysis approach accepted; bootstrap settings pending. BO's comparison with full CMA-ES at the primary condition is now primary (D32).

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

**Rationale:** These diagnostics help assess whether the learned search geometry is related to the objective geometry. They support interpretation but do not, by themselves, establish causation. Under an idealised model, the covariance of selected solutions becomes proportional to the inverse Hessian as the population grows (Shir & Yehudayoff, 2020), so $\kappa_{\mathrm{eff}} = 1$ is an idealised target rather than a guaranteed outcome.

**Implementation note:** With bound handling, pycma's internal covariance lives in a transformed space in which coordinates can be mirrored. $\kappa_{\mathrm{eff}}$ is therefore computed from the covariance of the evaluated points, obtained with the Jacobian of the bound transform; this matches the empirical covariance of pycma's own samples.

**Status:** Accepted; implemented and validated against sampled covariances.

---

## 6. Finance Application: Data and Preprocessing (cut, D31)

### D20. Use WTI spot prices from FRED

**Decision:** Use the FRED series `DCOILWTICO`, originally sourced from the U.S. Energy Information Administration. Store a frozen copy at `data/DCOILWTICO.csv` and record its retrieval date.

Source: https://fred.stlouisfed.org/series/DCOILWTICO

**Rationale:** A frozen dataset prevents later source revisions from silently changing the experiment.

**Status:** Cut (D31).

### D21. Define returns and handle non-positive prices explicitly

**Decision:** Calculate percentage log returns as

$$
r_t=100\log\left(\frac{P_t}{P_{t-1}}\right),
$$

then demean returns using the estimation-window mean. Drop missing observations according to the configured preprocessing policy.

Exclude any estimation window containing the non-positive price observation dated 20 April 2020, and record the exclusion.

**Rationale:** Log returns are undefined when a price is non-positive. Explicitly excluding affected windows avoids invalid calculations and makes the data-handling decision auditable.

**Status:** Cut (D31).

---

## 7. GARCH(1,1) Model and Analysis (cut, D31)

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

**Status:** Cut (D31).

### D23. Use rolling estimation windows

**Decision:** Use 750 trading days per estimation window, moving forward by 250 trading days.

**Rationale:** Rolling windows create multiple calibration instances and allow the analysis to examine whether optimisation behaviour varies with the curvature of the fitted likelihood.

**Status:** Cut (D31).

### D24. Validate the custom likelihood against `arch`

**Decision:** Compare the custom GARCH likelihood with the `arch` package on at least three estimation windows before scaling up the finance experiments.

Use the same initial-variance convention in both calculations: override `arch`'s default backcast (an exponentially weighted average of early squared residuals) with the window sample variance. Verify that the variance recursion agrees before comparing negative log-likelihood values.

The custom implementation must document and test the variance-bound rule used in the comparison.

**Rationale:** Differences in initialisation or variance bounds can produce different likelihood values even when the model equations are otherwise equivalent. These differences must be investigated before attributing discrepancies to optimisation performance.

**Status:** Cut (D31).

### D25. Define likelihood gaps within each estimation window

**Decision:** After all main runs are complete, define each window's best observed negative log-likelihood using the pre-specified set of completed main runs and the reference optimiser. Exclude pilot runs.

For a method's estimate $\hat\theta$, define

$$
g=\mathrm{NLL}(\hat\theta)
-\mathrm{NLL}_{\mathrm{best,window}}.
$$

Analyse $\log_{10}(g+\epsilon)$, with a positive $\epsilon$ frozen before final analysis. Record signed gaps to the L-BFGS-B reference separately and flag cases where a method outperforms that reference.

**Rationale:** A per-window reference avoids mixing likelihood levels across different data windows. A method that outperforms the reference optimiser should be flagged rather than hidden or automatically treated as an error.

**Status:** Cut (D31).

### D26. Test the relationship between conditioning and optimiser performance

**Decision:** Use the Jacobi-scaled Hessian condition number $\kappa_J$ as the primary predictor, with the ordinary Hessian condition number $\kappa_H$ as a secondary predictor. Estimate the Hessian by finite differences at the reference optimum in the main arm's unit-box coordinates.

Relate conditioning to the diagonal-minus-full CMA-ES log-gap using Spearman correlation and moving-block bootstrap uncertainty, with a block length of at least 3 windows.

**Rationale:** $\kappa_J$ measures the conditioning that remains after Jacobi (diagonal) rescaling, which is within a factor of $d$ of the best possible diagonal rescaling (van der Sluis, 1969). This is the conditioning a diagonal model cannot remove. Aligned problems have $\kappa_J=1$, however large $\kappa$ is, so $\kappa_J$, not $\kappa_H$, is the quantity H3 predicts will drive the diagonal-minus-full gap. Consecutive windows overlap by two-thirds, so a block bootstrap is needed to respect the dependence.

Boundary optima will be handled with one-sided finite differences and flagged.

**Status:** Cut (D31).

### D27. Keep the forecast analysis secondary and exploratory

**Decision:** Evaluate 250 trading days of one-step-ahead variance forecasts using QLIKE, holding estimated parameters fixed. Use squared demeaned returns as the variance proxy and compare against the reference-optimum QLIKE.

**Rationale:** This provides a secondary assessment of the practical implications of calibration differences without turning the project into a separate forecasting or trading study.

**Status:** Cut (D31).

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

Pilot settings live in a separate `pilot` block in `configs/experiment.yaml` and are applied only at the pilot stage. The main stage refuses to run while any required setting is null.

**Rationale:** Pilot results may be used to select numerical settings, but those settings must be documented and frozen before the main analysis to avoid outcome-driven changes.

**Status:** Accepted; freeze conditions not yet satisfied.

### D30. Optimiser settings and tuning policy

**Decision:** No hyperparameter tuning is performed for any method. Each method uses published default settings or settings fixed in this log before the pilot. Conclusions are therefore conditional on default configurations.

- **CMA-ES variants.** All three variants use the default strategy parameters of pycma 4.5.0 (Hansen et al., 2019), the reference Python implementation maintained by the author of CMA-ES. Hansen (2023, p. 32) describes the default parameters as a robust setting applicable to a wide range of functions and does not recommend changing them, apart from increasing the population size or, on noisy functions, decreasing $\alpha_{\mathrm{cov}}$; neither applies here. The population size is the default $\lambda = 4 + \lfloor 3\ln n \rfloor$ (Hansen, 2023, Table 1, Equation 48), giving $\lambda = 8$ at $d = 4$. The initial point and step size, which Hansen treats as problem dependent, follow his rule of a uniform start in $[a,b]^n$ with $\sigma_0 = 0.3\,(b-a)$ (Hansen, 2023, Appendix A, Figure 6, footnote 1), giving $\sigma_0 = 3$. The variants differ only in their covariance option (D8, D9).¹
- **Uniform random search** samples the same box and has no settings to tune.
- **BO** uses the ARD Matérn 5/2 kernel (Snoek et al., 2012, Section 3.1), plus an isotropic Matérn 5/2 variant (D32), a standard default supported by recent benchmarking (Le Riche & Picheny, 2021), and the expected improvement acquisition function (Jones et al., 1998). GP hyperparameters are fitted by maximising the marginal likelihood, that is, type-II maximum likelihood (Rasmussen & Williams, 2006, Section 5.4.1, Equation 5.8); this is part of the method, not tuning. Because the marginal likelihood can have multiple local optima, the fit uses several random restarts and keeps the best (Rasmussen & Williams, 2006, p. 115). BO's algorithm settings (initial design size, number of restarts, fixed jitter) are chosen once from published guidance, including a small initial design (Le Riche & Picheny, 2021), and a numerical-stability smoke test, then recorded here and frozen.

**What the pilot may change:** only (a) the budget, floor and invariance tolerance, by their pre-stated rules (D12, D15, D18); and (b) a numerical or implementation setting, and only if a failure trigger occurs: the budget is not enforced exactly; a test fails; the objective returns non-finite values; a run has `stop_category = "other"`; or, for BO, GP fitting or acquisition optimisation fails. Pilot regret values are never used to change $\sigma_0$, population size, kernel or acquisition function.

**Exploratory robustness (optional, pre-declared):** the primary DiD may also be reported at $\sigma_0 \in \lbrace 0.15, 0.5 \rbrace \times$ box width.

**Alternatives:** per-method tuning on each problem (e.g. grid search or hyperparameter optimisation). Bartz-Beielstein et al. (2020, Section 7.5) note that tuning every method on every problem compares methods at peak performance, and that comparing tuned with untuned methods is unfair. This study does not tune per problem, for two reasons: (i) the research question concerns a mechanism (covariance learning under rotation), and tuning each variant on the test ellipsoids would introduce parameter differences into that comparison and tune on the test instances themselves; (ii) the CMA-ES defaults are themselves designed as a robust setting across a wide range of functions (Hansen, 2023, p. 32), which corresponds to the option Bartz-Beielstein et al. (2020) describe of using defaults obtained from tuning over a broad problem set. Equal treatment is preserved because no method receives problem-specific tuning. Such a comparison deliberately abstains from setting good parameters for specific problems (Bartz-Beielstein et al., 2020); this is stated as a limitation, especially for BO, which is typically more sensitive to its settings.

**Status:** Accepted.

¹ pycma 4.5.0's defaults match Hansen (2023, Table 1) for the population size, recombination weights and covariance learning rates ($\lambda = 8$, $\mu = 4$, $\mu_{\mathrm{eff}} = 2.60$, $c_c = 0.500$, $c_1 = 0.0652$, $c_\mu = 0.0640$ at $d = 4$). Its step-size constants differ slightly ($c_\sigma = 0.479$ and $d_\sigma = 1.479$, versus $0.397$ and $1.397$ from Table 1); they are identical across all three variants, so they cannot bias the diagonal-versus-full comparison. Values verified on the installed version.

### D31. Remove the finance application from scope

**Date:** 2 November 2026

**Decision:** The GARCH(1,1) finance application (D2, D20–D27) is removed from the project scope. The project consists of the synthetic study only. The finance hypotheses are dropped.

**Previous setting:** A secondary application calibrating GARCH(1,1) on rolling WTI return windows, to be cut first if time ran short (D1).

**Reason:** Time. Completing the synthetic study reliably, including BO, the main runs and the analysis, takes priority before the 13 November deadline. This applies the priority order fixed in D1.

**Pilot or main runs started:** No. No finance code or data had been produced, so no results are affected.

**Effect on comparability or interpretation:** None for the synthetic study. The report no longer tests whether the synthetic findings transfer to a real calibration problem; this is stated as a limitation and future work.

**Status:** Accepted.

### D32. Make BO a co-primary comparison and strengthen the synthetic design

**Date:** 2 November 2026 (recorded before any pilot or main results).

**Decision:**

1. **BO becomes a co-primary comparison.** The topic brief asks for an explanation of a difference between BO and evolution strategies. The explanation tested is that axis-aligned models lose under rotation: diagonal CMA-ES and BO with an ARD kernel both model one scale per coordinate axis (Ros & Hansen, 2008; Rasmussen & Williams, 2006, p. 106), whereas full CMA-ES can represent any orientation (Hansen, 2023, Section 5, p. 24). On functions that vary along non-axis-aligned directions, an ARD kernel can fail to learn where a Mahalanobis kernel succeeds (Letham et al., 2020, supplement). A second primary DiD therefore compares BO (ARD) with full CMA-ES, using the same formula as D16.
2. **BO with an isotropic kernel is added** as an exploratory arm. An isotropic kernel depends only on distance and is invariant to all rigid motions (Rasmussen & Williams, 2006, Section 4.1, p. 80), but cannot learn different scales per direction. It mirrors isotropic ES, giving both families a rotation-invariant reference.
3. **Budget-dependent invariance.** Hansen (2023, p. 24, footnote 26) notes that invariance yields uniform performance only after the algorithm's state has adapted, and covariance learning takes roughly $n^2/\mu_{\mathrm{eff}}$ generations (Hansen, 2023, Section 3.2). A pre-registered secondary analysis tests whether the rotation penalty of full CMA-ES shrinks across checkpoints while that of diagonal CMA-ES and ARD BO does not.
4. **Conditioning ladder.** Main runs use $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$ for the CMA-ES variants and uniform search, so the rotation penalty can be traced against conditioning. BO runs at $\kappa \in \lbrace 1, 10^4 \rbrace$, the primary conditions, because it is the expensive method.
5. **Synthetic $\kappa_J$ test.** Across rotated instances, $\kappa_J$ varies with the random rotation. A pre-registered secondary analysis reports the Spearman correlation between per-instance $\kappa_J$ and the diagonal-minus-full log-regret gap (van der Sluis, 1969, for $\kappa_J$).
6. **Replicates:** 50 per condition (previously a target of 20).
7. **Cost.** Wall-clock time is reported alongside evaluations, with a break-even analysis of the evaluation cost at which BO's surrogate overhead pays off; exact GP fitting costs $n^3/6$ per Cholesky factorisation (Rasmussen & Williams, 2006, Algorithm 2.1).
8. **BO budget.** BO runs to a lower ceiling fixed by its smoke test, and all methods are compared at the shared checkpoints.
9. **Optional robustness:** dimension $d \in \lbrace 2, 8 \rbrace$ for the CMA-ES variants and uniform search.

**Labels:** the two DiDs (diagonal CMA-ES and ARD BO, each against full CMA-ES, at $d=4$, $\kappa=10^4$) are primary. Items 3–5 are pre-registered secondary analyses. Item 2, the cost analysis, and the BO-versus-ES comparisons beyond the primary DiD are exploratory. Items 9 and the sensitivity analyses in D15 and D30 are robustness checks.

**Alternatives:** keeping BO exploratory, rejected because the topic brief centres on BO versus evolution strategies; adding a second function family, rejected to keep the explanation testable within the time available.

**Pilot:** unchanged. It still uses $\kappa \in \lbrace 1, 10^4 \rbrace$ at $d = 4$ with the CMA-ES variants and uniform search, and only sets the budget, floor and invariance tolerance.

**Status:** Accepted.

---

## 9. Outstanding Decisions

These settings remain open, in the order they are needed.

| Item | Outstanding decision | Entry | Required before |
|---|---|---|---|
| Evaluation budget | Final budget by the pre-stated rule | D12 | Main runs |
| Log-regret floor | Confirm $10^{-8}$ is reachable at $\kappa=1$ | D15 | Main runs |
| Invariance tolerance | Set from the $\kappa=1$ pilot spread | D18 | Main runs |
| Search domain | Confirm boundary handling and sub-box fraction at the freeze | D5, D6 | Main runs |
| Bayesian optimisation | Library, initial design, restart counts, fixed jitter, BO ceiling | D30, D32 | BO smoke test |
| Primary analysis | Bootstrap replicate count | D17 | Main analysis |
| Diagnostics | Covariance snapshot frequency; uniform-search success criterion | D14, D19 | Main runs |

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

---

## 11. References --> ***VERFIY ALL SOURCES BEFORE SUBMISSION***

- Akimoto, Y., & Hansen, N. (2020). Diagonal acceleration for covariance matrix adaptation evolution strategies. *Evolutionary Computation, 28*(3), 405–435. https://arxiv.org/abs/1905.05885
- Bartz-Beielstein, T., Doerr, C., van den Berg, D., Bossek, J., Chandrasekaran, S., Eftimov, T., Fischbach, A., Kerschke, P., La Cava, W., López-Ibáñez, M., Malan, K. M., Moore, J. H., Naujoks, B., Orzechowski, P., Volz, V., Wagner, M., & Weise, T. (2020). *Benchmarking in optimization: Best practice and open issues* (arXiv:2007.03488). arXiv. https://doi.org/10.48550/arXiv.2007.03488
- Hansen, N. (2023). *The CMA evolution strategy: A tutorial* (arXiv:1604.00772; Version 2). arXiv. https://doi.org/10.48550/arXiv.1604.00772
- Hansen, N., Akimoto, Y., & Baudis, P. (2019). *CMA-ES/pycma on GitHub* (Version 4.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2559634
- Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software, 36*(1), 114–144. https://arxiv.org/abs/1603.08785
- Hansen, N., Finck, S., Ros, R., & Auger, A. (2009). *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions* (Research Report RR-6829). INRIA. https://hal-lara.archives-ouvertes.fr/inria-00362633v1
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization, 13*(4), 455–492. https://doi.org/10.1023/A:1008306431147
- Le Riche, R., & Picheny, V. (2021). Revisiting Bayesian optimization in the light of the COCO benchmark. *Structural and Multidisciplinary Optimization, 64*(5), 3063–3087. https://doi.org/10.1007/s00158-021-02977-1
- Letham, B., Calandra, R., Rai, A., & Bakshy, E. (2020). Re-examining linear embeddings for high-dimensional Bayesian optimization. In *Advances in neural information processing systems 33*. https://neurips.cc/virtual/2020/poster/18297
- Long, F. X., Vermetten, D., van Stein, B., & Kononova, A. V. (2022). *BBOB instance analysis: Landscape properties and algorithm performance across problem instances* (arXiv:2211.16318). arXiv. https://doi.org/10.48550/arXiv.2211.16318
- Mezzadri, F. (2007). How to generate random matrices from the classical compact groups. *Notices of the American Mathematical Society, 54*(5), 592–604. https://arxiv.org/abs/math-ph/0609050
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press. https://gaussianprocess.org/gpml/
- Ros, R., & Hansen, N. (2008). A simple modification in CMA-ES achieving linear time and space complexity. In G. Rudolph, T. Jansen, S. Lucas, C. Poloni, & N. Beume (Eds.), *Parallel problem solving from nature – PPSN X* (pp. 296–305). Springer.
- Santoni, M. L., Raponi, E., De Leone, R., & Doerr, C. (2024). Comparison of high-dimensional Bayesian optimization algorithms on BBOB. *ACM Transactions on Evolutionary Learning and Optimization, 4*(3). https://arxiv.org/abs/2303.00890
- Shir, O. M., & Yehudayoff, A. (2020). On the covariance-Hessian relation in evolution strategies. *Theoretical Computer Science, 801*, 157–174. https://cris.iucc.ac.il/en/publications/on-the-covariance-hessian-relation-in-evolution-strategies/
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In F. Pereira, C. J. C. Burges, L. Bottou, & K. Q. Weinberger (Eds.), *Advances in neural information processing systems 25* (pp. 2951–2959). Curran Associates. https://arxiv.org/abs/1206.2944
- van der Sluis, A. (1969). Condition numbers and equilibration of matrices. *Numerische Mathematik, 14*(1), 14–23. https://doi.org/10.1007/BF02165096
- Vermetten, D., López-Ibáñez, M., Mersmann, O., Allmendinger, R., & Kononova, A. V. (2023). *Analysis of modular CMA-ES on strict box-constrained problems in the SBOX-COST benchmarking suite* (arXiv:2305.15102). arXiv. https://doi.org/10.48550/arXiv.2305.15102