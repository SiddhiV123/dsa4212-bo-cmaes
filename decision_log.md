# Decision Log — DSA4212 Project

**Project:** When Does Geometry-Aware Search Pay Off?  
**Topic:** Bayesian optimisation versus evolution strategies  
**Last updated:** 3 November 2026  
**Citation style:** APA 7th edition (author–date). Full references in Section 9.  
**Status:** Pilot preparation

This document records the main experimental design decisions, their rationale, alternatives considered, and the settings that remain unresolved. Decisions marked *provisional* are reviewed after the pilot and frozen before the main experiments.

---

## 1. Project Scope and Research Question

### D1. Study BO versus evolution strategies through geometry: rotation and conditioning

**Decision:** The project compares Bayesian optimisation (BO) and CMA-ES variants on aligned and rotated ill-conditioned ellipsoids, and tests one explanation for a difference in their behaviour: methods whose internal model is axis-aligned should lose performance when an ill-conditioned problem is rotated, while a method that can represent any orientation should not, once it has had enough evaluations to adapt.

- Diagonal CMA-ES and BO with an ARD kernel both model one scale per coordinate axis (Ros & Hansen, 2008; Rasmussen & Williams, 2006, p. 106).
- Full CMA-ES can represent any orientation; Hansen lists invariance to rotations of the search space among its properties, "if the initial search point is transformed accordingly" (Hansen, 2023, Section 5, p. 24).
- On functions that vary along non-axis-aligned directions, an ARD kernel can fail to learn where a Mahalanobis kernel succeeds (Letham et al., 2020, supplement).
- A published precedent shows the effect for separable CMA-ES: it performs better on the axis-parallel ellipsoid than on its rotated version, while rotation-invariant variants perform almost identically on both (Loshchilov, 2014, Section 4.3).

**Rationale:** The topic brief asks for an explanation of a difference between BO and evolution strategies, and lists "geometry and parameterisation: conditioning, rotations … which theoretical invariances survive the practical implementation?" as a direction. Rotated ellipsoids isolate exactly this mechanism.

**Status:** Accepted.

---

## 2. Synthetic Objective Design

### D2. Use a four-dimensional ellipsoid as the benchmark

**Decision:** Use dimension $d=4$ and condition numbers $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$, each in an aligned and a randomly rotated version. BO is run at $\kappa \in \lbrace 1, 10^4 \rbrace$, the primary conditions, because it is the expensive method (D22).

The objective is

$$
f(x)=\sum_{i=1}^{d}
\kappa^{(i-1)/(d-1)}z_i^2,
\qquad z=R(x-x_{\mathrm{opt}}).
$$

Here, $R=I$ for aligned instances and $R$ is an orthogonal matrix for rotated instances. The known optimum is $x_{\mathrm{opt}}$, with objective value zero.

**Rationale:** This design varies conditioning and orientation while keeping the objective smooth, unimodal, and analytically understood. The $\kappa=1$ case acts as an orientation-invariance control because the objective is spherical regardless of the orthogonal rotation. The five-level ladder lets the rotation penalty be traced against conditioning.

**Source and deviation:** The objective follows BBOB functions f2 and f10, the separable and rotated ellipsoids (Hansen et al., 2009, Sections 1.2 and 3.10, Equations 2 and 10). The fixed conditioning $10^6$ is generalised to a parameter $\kappa$, as in pycma's `cma.ff.elli(cond=...)` (Hansen et al., 2019), because the research question concerns how rotation interacts with conditioning. The oscillation transform $T_{\mathrm{osz}}$ is omitted so that the objective is an exact quadratic with a known Hessian. The implementation is cross-checked against `cma.ff.elli` in the unit tests.

**Status:** Accepted.

### D3. Generate rotations using random orthogonal matrices

**Decision:** Generate rotated instances using QR decomposition with sign correction. Save the resulting rotation matrix for each problem instance.

**Rationale:** The rotation changes the orientation of the principal axes without changing the eigenvalues of the Hessian. Saving the matrix ensures that each instance can be reproduced exactly. QR decomposition of a Gaussian matrix is not Haar-distributed on its own; multiplying the columns of $Q$ by the signs of the diagonal of $R$ makes it so (Mezzadri, 2007).

**Status:** Accepted; implemented and tested (orthogonality, reproducibility, and a check that the sign correction removes the bias).

### D4. Randomise the optimum independently of optimiser initialisation

**Decision:** Sample the optimum uniformly within a configured sub-box of the search domain. Optimiser initial points must be generated independently of the optimum.

**Pilot setting:** sub-box fraction 0.8, so $x_{\mathrm{opt}}$ is uniform in $[-4,4]^d$, following the BBOB convention (Long et al., 2022).

**Alternatives:** placing the optimum anywhere in the full domain $[-5,5]^d$, which avoids favouring centre-focused methods (Vermetten et al., 2023). The sub-box is kept because it also keeps the optimum away from the region where bound handling distorts the search; the centre-bias point is noted as a limitation.

**Rationale:** A fixed optimum at the origin could unintentionally favour methods whose initialisation is centred or whose sampling scheme is symmetric around the origin. Independent randomisation reduces this source of bias.

**Status:** Design accepted; sub-box fraction 0.8 for the pilot, to be confirmed at the freeze.

### D5. Use Sphere for supporting validation; keep other functions out of scope

**Decision:** Retain Sphere objectives in dimensions $d \in \lbrace 2, 4, 10 \rbrace$ for unit tests, method-interface checks, and sanity checks. Rastrigin, noisy objectives and a $d=10$ ellipsoid study are not run.

**Rationale:** Sphere is useful for validating objective evaluation, optimiser interfaces, and basic behaviour before running the main experiments. Rastrigin results depend mainly on population size, which this study does not vary, and noise and higher dimensions would add questions the design is not built to answer.

**Status:** Accepted.

---

## 3. Optimisation Methods

### D6. Compare six optimisation methods

**Decision:** Include the following methods, which form a ladder from no geometry model to a full one in both families:

1. Full-covariance CMA-ES: learns scales and correlations.
2. Diagonal (separable) CMA-ES: learns per-coordinate scales only.
3. Isotropic ES: CMA-ES with covariance adaptation disabled, so step-size control only.
4. BO with an ARD Matérn 5/2 Gaussian process and expected improvement: per-coordinate lengthscales.
5. BO with an isotropic Matérn 5/2 Gaussian process and expected improvement: a single lengthscale.
6. Uniform random search: no model and no use of feedback.

Every method reports its best *evaluated* point. BO may use its surrogate to choose the next evaluation, but its reported solution must always be a point that was actually evaluated.

**Rationale:** Full and diagonal CMA-ES isolate the role of learning correlations. Isotropic ES separates covariance learning from step-size adaptation, which Hansen notes can be applied independently of the covariance matrix update (Hansen, 2023, Section 4, p. 19). ARD BO is the BO counterpart of diagonal CMA-ES, and isotropic BO, whose kernel depends only on distance and is invariant to all rigid motions (Rasmussen & Williams, 2006, Section 4.1, p. 80), is the BO counterpart of isotropic ES. Uniform random search is the standard null baseline for adaptive methods (Bergstra & Bengio, 2012) and is endorsed by COCO as additional baseline data (Hansen et al., 2016). A common recommendation rule keeps the comparison about optimisation rather than about how each method reports its answer.

**Alternatives:** A fixed Gaussian random-search arm centred on the CMA-ES starting point, rejected because it has no published precedent; isotropic ES answers the same question with an established method.

**Status:** Accepted; all six methods implemented and tested.

### D7. Use separable diagonal CMA-ES

**Decision:** Use separable CMA-ES (Ros & Hansen, 2008), run as `CMA_diagonal=True` in pycma (Hansen et al., 2019), rather than a method that retains a full covariance representation.

**Alternatives:** Diagonal-decoding CMA-ES (dd-CMA; Akimoto & Hansen, 2020).

**Rationale:** The objective is to test how restricting covariance adaptation affects performance under rotation. A method that retains a full covariance representation would not isolate this distinction as directly.

**Verification:** In pycma 4.5.0 the diagonal variant samples from an identity matrix and adapts a per-coordinate scaling vector, so off-diagonal covariance entries are zero by construction. A unit test confirms this at every checkpoint.

**Status:** Accepted; verified on the installed version.

### D8. Use common settings across CMA-ES variants

**Decision:** Define shared CMA-ES settings in `cma_es_common` and merge them into the full, diagonal, and isotropic variants through the configuration loader.

Use a common initial-point policy, a common initial step-size rule, and matched evaluation budgets.

**Rationale:** Differences in performance should primarily reflect the covariance-adaptation mechanism rather than unrelated initialisation or budget differences.

The `inherits` field in the YAML file is a project convention, not native YAML inheritance. The implementation must explicitly merge the common settings and test the resolved configurations.

**Verification:** The configuration loader rejects any variant that overrides a shared setting, and checks that `CMA_diagonal: true` and `CMA_on: 0` survive the merge. For isotropic ES, a unit test confirms the internal covariance stays at the identity (condition number below 1.001).

**Status:** Accepted; loader and merge tests implemented.

### D9. Initialise optimisers without information about the optimum

**Decision:** Draw CMA-ES initial points uniformly from the search box, independently of the optimum. Use the same initial point across the three CMA-ES variants within a matched problem instance.

The initial step size is $\sigma_0 = 0.3\,(b-a)$, following Hansen's rule for a uniform start in $[a,b]^n$ (Hansen, 2023, Appendix A, Figure 6, footnote 1). For the box $[-5,5]^4$ this gives $\sigma_0 = 3$.

**Rationale:** This supports paired comparisons and avoids giving one method privileged knowledge of the optimum. COCO likewise requires that an algorithm's initial state and parameters be based only on the problem's inputs, such as the search domain (Hansen et al., 2016). The step-size rule is fixed and is not changed on the basis of pilot results (D21).

**Status:** Accepted.

### D10. Handle early termination transparently

**Decision:** Attempt to run methods to the prescribed evaluation budget. If a CMA-ES run terminates early despite the configured stopping policy, preserve its best evaluated point, flag the run as incomplete, and report the actual evaluation count. Incomplete runs are retained in the analysis, and counts are reported by method and condition.

**Rationale:** An early-terminated run must not be presented as if it consumed the full budget. On a noise-free objective, early termination often reflects convergence, so excluding these runs would bias results against the fastest-converging methods. Keeping the run and its status visible avoids silently discarding inconvenient outcomes. COCO's only recommended tuning is checking that termination conditions suit the testbed (Hansen et al., 2016).

**Stop categories:** pycma's tolerance-based stops (`tolfun`, `tolfunrel`, `tolfunhist`, `tolx`, `tolstagnation`, `tolflatfitness`, `tolxstagnation`) are disabled. Two checks, `noeffectaxis` and `noeffectcoord`, cannot be disabled; they are numerical-stability criteria that fire when a step no longer changes the mean (Hansen, 2023, Appendix B.3). Runs that stop only on these are recorded with `stop_category = "numerical_precision"`, kept in the analysis, and their best value is carried to later checkpoints. Any other early stop is recorded as `stop_category = "other"` and investigated.

**Status:** Accepted; verified on pycma 4.5.0.

---

## 4. Evaluation Budget and Reproducibility

### D11. Enforce budgets at the objective-evaluation level

**Decision:** Route every objective evaluation through a runner-level counting wrapper. Count initial-design evaluations for BO and truncate a final CMA-ES batch when necessary to respect the budget exactly.

**Rationale:** Comparing methods by iterations rather than objective evaluations would not provide a fair comparison because one iteration can involve different numbers of evaluations across methods. The budget also changes which family wins: BO tends to do well at limited budgets and is often outperformed at larger ones (Raponi et al., 2023).

**Provisional pilot budget:** The pilot uses a provisional ceiling of 2,000 evaluations ($500d$ at $d=4$), with checkpoints at 90, 250, 500, 1,000 and 2,000. An informal pre-pilot check (10 seeds per condition, target $f(x) \le 10^{-8}$) found that full CMA-ES needed about 750–1,420 evaluations on the tested $\kappa=10^4$ instances, while diagonal CMA-ES did not reach the target on any rotated instance within 20,000; this is exploratory, not confirmatory. The ceiling allows for covariance learning, whose characteristic horizon is roughly $n^2/\mu_{\mathrm{eff}}$ generations (Hansen, 2023, Section 3.2), and lies within the pycma documentation's rule of thumb of $100n$ to $300n^2$ evaluations for hard problems (Hansen et al., 2019). The 90-evaluation checkpoint equals $10d+50$, the budget Santoni et al. (2024) used for their structural-mechanics applications, a domain-specific precedent.

**Selection rule (fixed before the pilot is analysed):**
1. Set the log-regret floor by the rule in D14 and freeze it, together with the rule for when a replicate counts as having reached it.
2. Select the smallest checkpoint at which full CMA-ES on the rotated $\kappa=10^4$, $d=4$ ellipsoid reaches the floor in at least 80% of pilot replicates. If none does, select 2,000.
3. At the selected checkpoint, check feasibility only: runtime, incomplete-run counts, and whether BO can be run. The size of the diagonal-versus-full difference is not inspected for this decision.
4. Record the final budget and rationale here before the confirmatory main runs. Pilot runs are excluded from confirmatory analysis.

**BO budget:** BO runs to a lower ceiling fixed by its smoke test, because exact GP fitting costs $n^3/6$ per Cholesky factorisation (Rasmussen & Williams, 2006, Algorithm 2.1). All methods are compared at the shared checkpoints, since algorithms are only comparable up to the smallest budget given to any of them (Hansen et al., 2016).

**Status:** Enforcement implemented and tested; budget selected by the rule above.

### D12. Pair methods on the same problem instances

**Decision:** Use a fixed base seed of `4212`. Derive each problem seed, which determines $x_{\mathrm{opt}}$ and $R$, from the condition and replicate index only, so that all methods within a replicate face the same problem instance. Derive method-specific random seeds separately. Run 50 replicates per condition. Save the problem instances, seeds, software versions, and actual evaluation counts.

The optimum is drawn before the rotation, so the aligned and rotated instances in a replicate share $x_{\mathrm{opt}}$ and differ only in orientation.

**Rationale:** Pairing reduces variation attributable to differences between randomly generated problems. Separate method seeds preserve reproducibility without forcing every method to consume identical random-number streams.

**Status:** Accepted; implemented.

### D13. Record evaluation-level diagnostics

**Decision:** Record candidate locations, objective values, best-so-far values, evaluation numbers, boundary events, method identifiers, and seeds.

Save CMA-ES covariance snapshots and BO hyperparameter diagnostics, including ARD lengthscales. Record wall-clock time and BO's surrogate overhead separately from evaluation counts.

**Rationale:** Final regret alone describes performance but does not explain how the optimisers behave. These diagnostics support the project's mechanism analysis, and the timing supports the cost analysis.

**Status:** Accepted; implemented in `runner.py`, `bo.py` and `run_pilot.py` (per-evaluation logs, covariance snapshots and BO lengthscales at every checkpoint).

---

## 5. Analysis

### D14. Use final simple regret as the primary outcome

**Decision:** Define final simple regret as

$$
r=f(x_{\mathrm{best}})-f(x_{\mathrm{opt}}),
$$

where $x_{\mathrm{best}}$ is the best evaluated point (D6). Analyse $\log_{10}(r)$ with a small positive floor.

**Rationale:** Simple regret measures the quality of the best solution found within the evaluation budget. A logarithmic scale helps compare performance across conditions with substantially different regret magnitudes.

**Log-regret floor rule (fixed before the pilot is run):** The floor is $10^{-8}$, the smallest of COCO's standard target precisions, which run from $10^2$ to $10^{-8}$ (Hansen et al., 2021). The $\kappa=1$ control runs are used only to confirm reachability: if every CMA-ES variant reaches $10^{-8}$ in all $\kappa=1$ pilot replicates, the floor is frozen at $10^{-8}$; otherwise it is the largest final regret attained across the $\kappa=1$ CMA-ES runs. A replicate has reached the floor when its best-so-far regret at a checkpoint is $\le$ the floor; regret below the floor is recorded as the floor.

**Sensitivity analysis:** the primary comparisons are also reported at floors of $10^{-6}$ and $10^{-10}$.

**Status:** Outcome and floor rule accepted; floor confirmed at the pilot.

### D15. Use difference-in-differences comparisons for the primary hypotheses

**Decision:** At $d=4$ and $\kappa=10^4$, compare each axis-aligned method $M$ with full CMA-ES using

$$
\Delta_M
=(\bar L_{\mathrm{rot},M}-\bar L_{\mathrm{align},M})
-(\bar L_{\mathrm{rot,full}}-\bar L_{\mathrm{align,full}}),
$$

where $L$ denotes log regret. There are two primary comparisons: $M$ = diagonal CMA-ES and $M$ = BO with an ARD kernel.

**Rationale:** Each comparison measures whether rotation increases log regret more for the axis-aligned method than for full CMA-ES, while accounting for each method's aligned performance. A positive value indicates a larger rotation penalty for $M$.

**Status:** Accepted.

### D16. Use paired uncertainty estimates and label every other analysis

**Decision:** Pair observations by problem instance and report bootstrap confidence intervals (Efron & Hastie, 2016). The bootstrap replicate count remains to be finalised.

Analyses beyond the two primary comparisons are labelled in advance (D22): pre-registered secondary analyses, exploratory analyses, and robustness checks.

**Rationale:** The primary comparisons should remain clear and interpretable. Labelled secondary and exploratory analyses can provide supporting evidence without being presented as equally central hypotheses.

**Status:** Analysis approach accepted; bootstrap settings pending.

### D17. Use the $\kappa=1$ control to assess orientation invariance

**Decision:** At $\kappa=1$, aligned and rotated problems are mathematically identical, so each method's aligned-minus-rotated difference should be zero apart from noise and box effects.

**Tolerance rule (fixed before the pilot is run):** A method is treated as invariant to rotation at $\kappa=1$ if the 95% paired bootstrap confidence interval for its mean aligned-minus-rotated log-regret difference lies entirely within $\pm 0.5$ (a factor of about 3 in regret). The margin is a design choice fixed in advance; the pilot is used only to check that the interval width at the planned replicate count makes the check informative. If an interval exceeds the margin, the difference is reported and investigated as a box or implementation effect, not hidden.

**Rationale:** Small numerical or sampling-related differences should be interpreted against a pre-specified tolerance rather than an arbitrary threshold. Because the start box is not rotated with the problem, exact invariance is not expected: in a similar check, the only systematic effect was initialisation in the non-rotated hypercube $[-5,5]^n$ (Loshchilov, 2017).

**Status:** Rule accepted; checked at the pilot and applied to the main runs.

### D18. Use mechanism diagnostics to interpret results

**Decision:** Examine the effective condition number

$$
\kappa_{\mathrm{eff}}
=\mathrm{cond}(C^{1/2}HC^{1/2}),
$$

where $C$ is the learned CMA-ES covariance and $H$ is the objective Hessian. Also examine BO ARD lengthscales relative to curvature in aligned cases.

**Rationale:** These diagnostics help assess whether the learned search geometry is related to the objective geometry. On a quadratic, setting the covariance to the inverse Hessian turns the ellipsoid into a sphere (Hansen, 2023, Section 0.4). Under an idealised model, the covariance of selected solutions becomes proportional to the inverse Hessian as the population grows (Shir & Yehudayoff, 2020), so $\kappa_{\mathrm{eff}} = 1$ is an idealised target rather than a guaranteed outcome. ARD lengthscales indicate the relevance BO assigns to each input (Rasmussen & Williams, 2006, p. 106). The diagnostics support interpretation but do not, by themselves, establish causation.

**Implementation note:** With bound handling, pycma's internal covariance lives in a transformed space in which coordinates can be mirrored. $\kappa_{\mathrm{eff}}$ is therefore computed from the covariance of the evaluated points, obtained with the Jacobian of the bound transform; this matches the empirical covariance of pycma's own samples.

**Status:** Accepted; implemented and validated against sampled covariances.

---

## 6. Configuration and Implementation Rules

### D19. Keep unresolved values explicit

**Decision:** Represent undecided settings as `null` in `configs/experiment.yaml` and mark provisional values clearly. The configuration loader must validate required settings before the relevant experiment can run.

**Rationale:** Explicit unresolved settings are safer than silently substituting defaults or treating missing values as zero.

**Status:** Accepted.

### D20. Freeze the main experiment settings after the pilot

**Decision:** Do not start the main runs until the pilot is complete, the configuration has been reviewed, and the required tests and validation checks have passed.

The freeze checklist includes:

- CMA-ES option and stopping-policy verification.
- Common CMA-ES configuration merge tests.
- Objective and rotation tests.
- Evaluation-budget and boundary-handling policies.
- Log-regret floor and invariance tolerance.
- Seed policy.
- BO settings fixed by the smoke test.

Pilot settings live in a separate `pilot` block in `configs/experiment.yaml` and are applied only at the pilot stage. The main stage refuses to run while any required setting is null.

**Rationale:** Pilot results may be used to select numerical settings, but those settings must be documented and frozen before the main analysis to avoid outcome-driven changes.

**Status:** Accepted; freeze conditions not yet satisfied.

### D21. Optimiser settings and tuning policy

**Decision:** No hyperparameter tuning is performed for any method. Each method uses published default settings or settings fixed in this log before the pilot. Conclusions are therefore conditional on default configurations.

- **CMA-ES variants.** All three variants use the default strategy parameters of pycma 4.5.0 (Hansen et al., 2019), the reference Python implementation maintained by the author of CMA-ES. Hansen (2023, p. 32) describes the default parameters as a robust setting applicable to a wide range of functions and does not recommend changing them, apart from increasing the population size or, on noisy functions, decreasing $\alpha_{\mathrm{cov}}$; neither applies here. The population size is the default $\lambda = 4 + \lfloor 3\ln n \rfloor$ (Hansen, 2023, Table 1, Equation 48), giving $\lambda = 8$ at $d = 4$. The initial point and step size, which Hansen treats as problem dependent, follow his rule of a uniform start in $[a,b]^n$ with $\sigma_0 = 0.3\,(b-a)$ (Hansen, 2023, Appendix A, Figure 6, footnote 1), giving $\sigma_0 = 3$. The variants differ only in their covariance option (D7, D8).¹
- **Uniform random search** samples the same box and has no settings to tune.
- **BO** uses the ARD Matérn 5/2 kernel (Snoek et al., 2012, Section 3.1) and an isotropic Matérn 5/2 variant, a standard default supported by recent benchmarking (Le Riche & Picheny, 2021), and the expected improvement acquisition function (Jones et al., 1998). It is implemented with scikit-learn's Gaussian process, whose Matérn kernel is isotropic for a scalar lengthscale and anisotropic for a vector one. GP hyperparameters are fitted by maximising the marginal likelihood, that is, type-II maximum likelihood (Rasmussen & Williams, 2006, Section 5.4.1, Equation 5.8); this is part of the method, not tuning. Because the marginal likelihood can have multiple local optima, the fit uses several random restarts and keeps the best (Rasmussen & Williams, 2006, p. 115). Lengthscale bounds are scikit-learn's defaults. BO's algorithm settings (initial design size, number of restarts, fixed jitter) are chosen once from published guidance, including a small initial design (Le Riche & Picheny, 2021), and a numerical-stability and runtime smoke test, then recorded here and frozen.

**What the pilot may change:** only (a) the budget, floor and invariance tolerance, by their pre-stated rules (D11, D14, D17); and (b) a numerical or implementation setting, and only if a failure trigger occurs: the budget is not enforced exactly; a test fails; the objective returns non-finite values; a run has `stop_category = "other"`; or, for BO, GP fitting or acquisition optimisation fails. Convergence warnings from GP hyperparameter fitting are counted and reported but are not failures. Pilot regret values are never used to change $\sigma_0$, population size, kernel or acquisition function.

**Exploratory robustness (optional, pre-declared):** the primary comparisons may also be reported at $\sigma_0 \in \lbrace 0.15, 0.5 \rbrace \times$ box width.

**Alternatives:** per-method tuning on each problem (e.g. grid search or hyperparameter optimisation). Bartz-Beielstein et al. (2020, Section 7.5) note that tuning every method on every problem compares methods at peak performance, and that comparing tuned with untuned methods is unfair. This study does not tune per problem, for two reasons: (i) the research question concerns a mechanism (geometry learning under rotation), and tuning each variant on the test ellipsoids would introduce parameter differences into that comparison and tune on the test instances themselves; (ii) the CMA-ES defaults are themselves designed as a robust setting across a wide range of functions (Hansen, 2023, p. 32), which corresponds to the option Bartz-Beielstein et al. (2020) describe of using defaults obtained from tuning over a broad problem set. COCO likewise requires the same parameter setting on all functions (Hansen et al., 2016). Equal treatment is preserved because no method receives problem-specific tuning. Such a comparison deliberately abstains from setting good parameters for specific problems (Bartz-Beielstein et al., 2020); this is stated as a limitation, especially for BO, which is typically more sensitive to its settings.

**Status:** Accepted.

¹ pycma 4.5.0's defaults match Hansen (2023, Table 1) for the population size, recombination weights and covariance learning rates ($\lambda = 8$, $\mu = 4$, $\mu_{\mathrm{eff}} = 2.60$, $c_c = 0.500$, $c_1 = 0.0652$, $c_\mu = 0.0640$ at $d = 4$). Its step-size constants differ slightly ($c_\sigma = 0.479$ and $d_\sigma = 1.479$, versus $0.397$ and $1.397$ from Table 1); they are identical across all three variants, so they cannot bias the diagonal-versus-full comparison. Values verified on the installed version.

### D22. Secondary, exploratory and robustness analyses

**Decision:** Beyond the two primary comparisons (D15), the following analyses are specified before any pilot or main results are seen:

1. **Budget-dependent invariance (secondary).** Hansen (2023, p. 24, footnote 26) notes that invariance yields uniform performance only after the algorithm's state has adapted. This analysis tests whether the rotation penalty of full CMA-ES shrinks across checkpoints while that of diagonal CMA-ES and ARD BO does not.
2. **Conditioning ladder (secondary).** The rotation penalty of each CMA-ES variant and uniform search is traced across $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$ (D2).
3. **$\kappa_J$ test (secondary).** $\kappa_J$, the condition number after Jacobi (diagonal) rescaling, measures the conditioning a diagonal model cannot remove; it is within a factor $d$ of the best possible diagonal rescaling (van der Sluis, 1969) and equals 1 for every aligned instance. Across rotated instances it varies with the random rotation, so this analysis reports the Spearman correlation between per-instance $\kappa_J$ and the diagonal-minus-full log-regret gap.
4. **Isotropic-kernel BO versus ARD BO (exploratory).**
5. **Cost (exploratory).** Wall-clock time is reported alongside evaluations, with a break-even analysis of the evaluation cost at which BO's surrogate overhead pays off; exact GP fitting costs $n^3/6$ per Cholesky factorisation (Rasmussen & Williams, 2006, Algorithm 2.1).
6. **Robustness (optional).** Dimension $d \in \lbrace 2, 8 \rbrace$ for the CMA-ES variants and uniform search; the floor and $\sigma_0$ sensitivity analyses in D14 and D21.

**Alternatives:** a second function family, rejected to keep the explanation testable within the time available.

**Rationale:** Labelling every analysis in advance prevents presenting the most favourable of many analyses as the main result.

**Status:** Accepted.

---

## 7. Outstanding Decisions

These settings remain open, in the order they are needed.

| Item | Outstanding decision | Entry | Required before |
|---|---|---|---|
| Evaluation budget | Final budget by the pre-stated rule | D11 | Main runs |
| Log-regret floor | Confirm $10^{-8}$ is reachable at $\kappa=1$ | D14 | Main runs |
| Invariance tolerance | Check the $\pm 0.5$ margin is informative at the pilot | D17 | Main runs |
| Search domain | Confirm boundary handling and sub-box fraction at the freeze | D4 | Main runs |
| Bayesian optimisation | Initial design size, restart counts, fixed jitter, BO ceiling | D11, D21 | BO smoke test |
| Primary analysis | Bootstrap replicate count | D16 | Main analysis |
| Diagnostics | Uniform-search success criterion | D6 | Main analysis |

---

## 8. Change-Control Rule

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

## 9. References ***VERIFY SOURCES***

*Verify every bibliographic detail before submission.*

- Akimoto, Y., & Hansen, N. (2020). Diagonal acceleration for covariance matrix adaptation evolution strategies. *Evolutionary Computation, 28*(3), 405–435. https://arxiv.org/abs/1905.05885
- Bartz-Beielstein, T., Doerr, C., van den Berg, D., Bossek, J., Chandrasekaran, S., Eftimov, T., Fischbach, A., Kerschke, P., La Cava, W., López-Ibáñez, M., Malan, K. M., Moore, J. H., Naujoks, B., Orzechowski, P., Volz, V., Wagner, M., & Weise, T. (2020). *Benchmarking in optimization: Best practice and open issues* (arXiv:2007.03488). arXiv. https://doi.org/10.48550/arXiv.2007.03488
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Efron, B., & Hastie, T. (2016). *Computer age statistical inference*. Cambridge University Press. https://web.stanford.edu/~hastie/CASI_files/PDF/casi_corrected_11102017.pdf
- Hansen, N. (2023). *The CMA evolution strategy: A tutorial* (arXiv:1604.00772; Version 2). arXiv. https://doi.org/10.48550/arXiv.1604.00772
- Hansen, N., Akimoto, Y., & Baudis, P. (2019). *CMA-ES/pycma on GitHub* (Version 4.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2559634
- Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software, 36*(1), 114–144. https://arxiv.org/abs/1603.08785
- Hansen, N., Finck, S., Ros, R., & Auger, A. (2009). *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions* (Research Report RR-6829). INRIA. https://hal-lara.archives-ouvertes.fr/inria-00362633v1
- Hansen, N., Tušar, T., Mersmann, O., Auger, A., & Brockhoff, D. (2016). *COCO: The experimental procedure* (arXiv:1603.08776). arXiv. https://arxiv.org/abs/1603.08776
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization, 13*(4), 455–492. https://doi.org/10.1023/A:1008306431147
- Le Riche, R., & Picheny, V. (2021). Revisiting Bayesian optimization in the light of the COCO benchmark. *Structural and Multidisciplinary Optimization, 64*(5), 3063–3087. https://doi.org/10.1007/s00158-021-02977-1
- Letham, B., Calandra, R., Rai, A., & Bakshy, E. (2020). Re-examining linear embeddings for high-dimensional Bayesian optimization. In *Advances in neural information processing systems 33*. https://neurips.cc/virtual/2020/poster/18297
- Long, F. X., Vermetten, D., van Stein, B., & Kononova, A. V. (2022). *BBOB instance analysis: Landscape properties and algorithm performance across problem instances* (arXiv:2211.16318). arXiv. https://doi.org/10.48550/arXiv.2211.16318
- Loshchilov, I. (2014). *A computationally efficient limited memory CMA-ES for large scale optimization* (arXiv:1404.5520). arXiv. https://arxiv.org/abs/1404.5520
- Loshchilov, I., Glasmachers, T., & Beyer, H.-G. (2017). *Limited-memory matrix adaptation for large scale black-box optimization* (arXiv:1705.06693). arXiv. https://arxiv.org/abs/1705.06693
- Mezzadri, F. (2007). How to generate random matrices from the classical compact groups. *Notices of the American Mathematical Society, 54*(5), 592–604. https://arxiv.org/abs/math-ph/0609050
- Raponi, E., et al. (2023). *Optimizing with low budgets: A comparison on the black-box optimization benchmarking suite and OpenAI Gym* (arXiv:2310.00077). arXiv. https://arxiv.org/abs/2310.00077
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press. https://gaussianprocess.org/gpml/
- Ros, R., & Hansen, N. (2008). A simple modification in CMA-ES achieving linear time and space complexity. In G. Rudolph, T. Jansen, S. Lucas, C. Poloni, & N. Beume (Eds.), *Parallel problem solving from nature – PPSN X* (pp. 296–305). Springer.
- Santoni, M. L., Raponi, E., De Leone, R., & Doerr, C. (2024). Comparison of high-dimensional Bayesian optimization algorithms on BBOB. *ACM Transactions on Evolutionary Learning and Optimization, 4*(3). https://arxiv.org/abs/2303.00890
- Shir, O. M., & Yehudayoff, A. (2020). On the covariance-Hessian relation in evolution strategies. *Theoretical Computer Science, 801*, 157–174. https://cris.iucc.ac.il/en/publications/on-the-covariance-hessian-relation-in-evolution-strategies/
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In F. Pereira, C. J. C. Burges, L. Bottou, & K. Q. Weinberger (Eds.), *Advances in neural information processing systems 25* (pp. 2951–2959). Curran Associates. https://arxiv.org/abs/1206.2944
- van der Sluis, A. (1969). Condition numbers and equilibration of matrices. *Numerische Mathematik, 14*(1), 14–23. https://doi.org/10.1007/BF02165096
- Vermetten, D., López-Ibáñez, M., Mersmann, O., Allmendinger, R., & Kononova, A. V. (2023). *Analysis of modular CMA-ES on strict box-constrained problems in the SBOX-COST benchmarking suite* (arXiv:2305.15102). arXiv. https://doi.org/10.48550/arXiv.2305.15102