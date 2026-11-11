# Decision Log — DSA4212 Project

**Project:** When Does Geometry-Aware Search Pay Off?  
**Topic:** Bayesian optimisation versus evolution strategies  
**Last updated:** 11 November 2026 (main runs and analysis complete)  
**Citation style:** APA 7th edition (author–date). Full references in Section 9.  
**Status:** All experiments complete; settings frozen before every stage; results recorded in Section 7.

This document records the experimental design decisions, their rationale and sources, the alternatives considered, the outcomes of the pre-stated rules, and every change made. Each decision was fixed before the results it could affect were seen.

---

## 1. Project Scope and Research Question

### D1. Study BO versus evolution strategies through geometry: rotation and conditioning

**Decision:** The project compares Bayesian optimisation (BO) and CMA-ES variants on aligned and rotated ill-conditioned ellipsoids, and tests one explanation for a difference in their behaviour: methods whose internal model is axis-aligned should lose performance when an ill-conditioned problem is rotated, while a method that can represent any orientation should not, once it has had enough evaluations to adapt.

- Diagonal CMA-ES and BO with an ARD kernel both model one scale per coordinate axis (Ros & Hansen, 2008; Rasmussen & Williams, 2006, p. 106).
- Full CMA-ES can represent any orientation; Hansen lists invariance to rotations of the search space among its properties, "if the initial search point is transformed accordingly" (Hansen, 2023, Section 5, p. 24).
- On functions that vary along non-axis-aligned directions, an ARD kernel can fail to learn where a Mahalanobis kernel succeeds (Letham et al., 2020, supplement).
- A published precedent shows the effect for separable CMA-ES: it performs better on the axis-parallel ellipsoid than on its rotated version, while rotation-invariant variants perform almost identically on both (Loshchilov, 2014, Section 4.3).
- BO is reported to underperform on unimodal functions with high conditioning (Le Riche & Picheny, 2021, Sections 6.3 and 7), which motivates studying exactly this problem class.

**Rationale:** The topic brief asks for an explanation of a difference between BO and evolution strategies, and lists "geometry and parameterisation: conditioning, rotations … which theoretical invariances survive the practical implementation?" as a direction. Rotated ellipsoids isolate exactly this mechanism.

**Status:** Accepted.

---

## 2. Synthetic Objective Design

### D2. Use a four-dimensional ellipsoid as the benchmark

**Decision:** Use dimension $d=4$ and condition numbers $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$, each in an aligned and a randomly rotated version. BO is run at $\kappa \in \lbrace 1, 10^4 \rbrace$, the primary conditions, because it is the expensive method.

The objective is

$$
f(x)=\sum_{i=1}^{d}
\kappa^{(i-1)/(d-1)}z_i^2,
\qquad z=R(x-x_{\mathrm{opt}}).
$$

Here, $R=I$ for aligned instances and $R$ is an orthogonal matrix for rotated instances. The known optimum is $x_{\mathrm{opt}}$, with objective value zero.

**Rationale:** This design varies conditioning and orientation while keeping the objective smooth, unimodal, and analytically understood. The $\kappa=1$ case acts as an orientation-invariance control because the objective is spherical regardless of the orthogonal rotation. The five-level ladder, one value per decade, lets the rotation penalty be traced against conditioning.

**Why $d=4$:** BO is typically applied in fewer than 20 dimensions with budgets of a few hundred evaluations (Frazier, 2018, abstract and Section 1); covariance learning time grows with $d^2$ (Hansen, 2023, Section 3.2), so full CMA-ES can adapt within the budget; GP cost (Rasmussen & Williams, 2006, Algorithm 2.1) keeps 400 BO runs feasible; and $d=4$ gives four distinct curvatures, so rotations mix several scales. Results may not extend to higher dimensions (limitation).

**Source and deviation:** The objective follows BBOB functions f2 and f10, the separable and rotated ellipsoids (Hansen et al., 2009a, Sections 1.2 and 3.10, Equations 2 and 10). The fixed conditioning $10^6$ is generalised to a parameter $\kappa$, as in pycma's `cma.ff.elli(cond=...)` (Hansen et al., 2019), because the research question concerns how rotation interacts with conditioning; the maximum $10^4$ lets methods separate within the budget. The oscillation transform $T_{\mathrm{osz}}$ is omitted so that the objective is an exact quadratic with a known Hessian. The implementation is cross-checked against `cma.ff.elli` in the unit tests.

**Status:** Accepted.

### D3. Generate rotations using random orthogonal matrices

**Decision:** Generate rotated instances using QR decomposition with sign correction. Save the resulting rotation matrix for each problem instance.

**Rationale:** The rotation changes the orientation of the principal axes without changing the eigenvalues of the Hessian. Saving the matrix ensures that each instance can be reproduced exactly. QR decomposition of a Gaussian matrix is not Haar-distributed on its own; multiplying the columns of $Q$ by the signs of the diagonal of $R$ makes it so (Mezzadri, 2007).

**Status:** Accepted; implemented and tested (orthogonality, reproducibility, and a check that the sign correction removes the bias).

### D4. Search domain and optimum placement

**Decision:** The search domain is $[-5,5]^d$, the BBOB search domain (Hansen et al., 2009b, Section 2.1; Hansen et al., 2009a). Sample the optimum uniformly within the sub-box $[-4,4]^d$ (sub-box fraction 0.8), following the BBOB convention (Long et al., 2022; Hansen et al., 2009b, Section 2.1). Optimiser initial points are generated independently of the optimum. Bound handling uses pycma's `BoundTransform` (pycma documentation; Brockhoff, 2023), with every repaired point logged, because repair can violate CMA-ES's assumptions on the solution distribution (Hansen, 2023, Appendix B.5).

**Alternatives:** placing the optimum anywhere in the full domain, which avoids favouring centre-focused methods (Vermetten et al., 2023, Section 2.1). The sub-box is kept because it also keeps the optimum away from the region where bound handling distorts the search; the centre-bias point is noted as a limitation.

**Rationale:** A fixed optimum at the origin could unintentionally favour methods whose initialisation is centred or whose sampling scheme is symmetric around the origin. Independent randomisation reduces this source of bias.

**Status:** Frozen after the pilot.

### D5. Use Sphere for supporting validation; keep other functions out of scope

**Decision:** Retain Sphere objectives in dimensions $d \in \lbrace 2, 4, 10 \rbrace$ for unit tests, method-interface checks, and sanity checks. Rastrigin, noisy objectives and a $d=10$ ellipsoid study are not run.

**Rationale:** Sphere is useful for validating objective evaluation, optimiser interfaces, and basic behaviour. Rastrigin results depend mainly on population size, which this study does not vary, and noise and higher dimensions would add questions the design is not built to answer.

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

**Rationale:** Full and diagonal CMA-ES isolate the role of learning correlations. Isotropic ES separates covariance learning from step-size adaptation, which can be applied independently of the covariance matrix update (Hansen, 2023, Section 4, p. 19). ARD BO is the BO counterpart of diagonal CMA-ES, and isotropic BO, whose kernel depends only on distance and is invariant to all rigid motions (Rasmussen & Williams, 2006, Section 4.1, p. 80), is the BO counterpart of isotropic ES. Uniform random search is the standard null baseline for adaptive methods (Bergstra & Bengio, 2012) and is endorsed by COCO as additional baseline data (Hansen et al., 2016a).

**Uniform search protocol:** uniform search is run like every other method: 50 independent runs per condition, each on the same instance as the other methods in its replicate (D12). Bergstra & Bengio's (2012, Section 2.2) single-long-run shortcut is not needed because uniform search is computationally negligible. Without a model, its rotation penalty also serves as a baseline for box effects.

**Uniform search validation:** observed success rates are compared with $1-(1-q)^N$, which holds for any success region with hitting probability $q$ (Bergstra & Bengio, 2012, Section 4; see also arXiv:2410.22854, Section 3.2.1). Success is regret $\le 1$, a standard COCO target (Hansen et al., 2021), chosen by the criterion that the check must be able to fail: at $\kappa=1$, $q=\pi^2\tau^2/(2\times10^4)$, and regret $\le 1$ is the only COCO decade target whose predicted success probability is neither near 0 nor near 1 across the checkpoints (4% at 90 to 63% at 2,000 evaluations; 0.1 gives at most 1%, 10 gives 99% from the first checkpoint). $q$ is estimated per instance from 100,000 uniform draws, for a Monte Carlo error (about ±0.05) below the sampling error of 50 runs (about ±0.07).

**Alternatives:** a fixed Gaussian random-search arm centred on the CMA-ES starting point, rejected because it has no published precedent; isotropic ES answers the same question with an established method.

**Status:** Accepted; criterion fixed before the analysis.

### D7. Use separable diagonal CMA-ES

**Decision:** Use separable CMA-ES (Ros & Hansen, 2008), run as `CMA_diagonal=True` in pycma (Hansen et al., 2019), rather than a method that retains a full covariance representation.

**Alternatives:** Diagonal-decoding CMA-ES (dd-CMA; Akimoto & Hansen, 2020), rejected because it keeps a full covariance and would not isolate correlation learning.

**Verification:** In pycma 4.5.0 the diagonal variant samples from an identity matrix and adapts a per-coordinate scaling vector, so off-diagonal covariance entries are zero by construction. A unit test confirms this at every checkpoint.

**Status:** Accepted; verified on the installed version.

### D8. Use common settings across CMA-ES variants

**Decision:** Define shared CMA-ES settings in `cma_es_common` and merge them into the full, diagonal, and isotropic variants through the configuration loader. Use a common initial-point policy, a common initial step-size rule, and matched evaluation budgets.

**Rationale:** Differences in performance should primarily reflect the covariance-adaptation mechanism rather than unrelated initialisation or budget differences. The `inherits` field in the YAML file is a project convention, not native YAML inheritance.

**Verification:** The configuration loader rejects any variant that overrides a shared setting, and checks that `CMA_diagonal: true` and `CMA_on: 0` survive the merge. For isotropic ES, a unit test confirms the internal covariance stays at the identity (condition number below 1.001).

**Status:** Accepted; loader and merge tests implemented.

### D9. Initialise optimisers without information about the optimum

**Decision:** Draw CMA-ES initial points uniformly from the search box, independently of the optimum. Use the same initial point across the three CMA-ES variants within a matched problem instance. The initial step size is $\sigma_0 = 0.3\,(b-a)$, following Hansen's rule for a uniform start in $[a,b]^n$ (Hansen, 2023, Appendix A, Figure 6, footnote 1). For $[-5,5]^4$ this gives $\sigma_0 = 3$.

**Rationale:** This supports paired comparisons and avoids giving one method privileged knowledge of the optimum. COCO requires that an algorithm's initial state and parameters be based only on the problem's inputs, such as the search domain (Hansen et al., 2016a). The step-size rule is fixed and was not changed on the basis of any results (D21).

**Status:** Accepted.

### D10. Handle early termination transparently

**Decision:** Attempt to run methods to the prescribed evaluation budget. If a CMA-ES run terminates early despite the configured stopping policy, preserve its best evaluated point, flag the run as incomplete, and report the actual evaluation count. Incomplete runs are retained in the analysis, and counts are reported by method and condition.

**Rationale:** An early-terminated run must not be presented as if it consumed the full budget. On a noise-free objective, early termination often reflects convergence, so excluding these runs would bias results against the fastest-converging methods. COCO's only recommended tuning is checking that termination conditions suit the testbed (Hansen et al., 2016a).

**Stop categories:** pycma's tolerance-based stops (`tolfun`, `tolfunrel`, `tolfunhist`, `tolx`, `tolstagnation`, `tolflatfitness`, `tolxstagnation`) are disabled. Two checks, `noeffectaxis` and `noeffectcoord`, cannot be disabled; they are numerical-stability criteria that fire when a step no longer changes the mean (Hansen, 2023, Appendix B.3). Runs that stop only on these are recorded with `stop_category = "numerical_precision"`, kept in the analysis, and their best value is carried to later checkpoints. Any other early stop is recorded as `stop_category = "other"` and investigated.

**Status:** Accepted; verified on pycma 4.5.0. Main runs: no `"other"` stops (Section 7).

---

## 4. Evaluation Budget and Reproducibility

### D11. Enforce budgets at the objective-evaluation level

**Decision:** Route every objective evaluation through a runner-level counting wrapper. Count initial-design evaluations for BO and truncate a final CMA-ES batch when necessary to respect the budget exactly.

**Rationale:** Comparing methods by iterations rather than objective evaluations would not be fair, because one iteration involves different numbers of evaluations across methods. The budget also changes which family wins: BO tends to do well at limited budgets and is often outperformed at larger ones (Raponi et al., 2023).

**Checkpoints:** 90, 250, 500, 1,000 and 2,000 evaluations. 90 equals $10d+50$, the budget Santoni et al. (2024, Section 1) used for their structural-mechanics applications; 250 is a BO ceiling option, so BO and ES can be compared there; 500, 1,000 and 2,000 double each time, equally spaced on a log budget axis. The provisional ceiling of 2,000 ($500d$) allows for covariance learning, whose characteristic horizon is roughly $n^2/\mu_{\mathrm{eff}}$ generations (Hansen, 2023, Section 3.2).

**Selection rule (fixed before the pilot was analysed):**
1. Set the log-regret floor by the rule in D14.
2. Select the smallest checkpoint at which full CMA-ES on the rotated $\kappa=10^4$, $d=4$ ellipsoid reaches the floor in at least 80% of pilot replicates. If none does, select 2,000.
3. At the selected checkpoint, check feasibility only: runtime, incomplete-run counts, and whether BO can be run. The size of the diagonal-versus-full difference is not inspected.
4. Pilot runs are excluded from confirmatory analysis.

**Pilot outcome (official run `20261103T025402Z`, see the change record in Section 8):** full CMA-ES on the rotated $\kappa=10^4$ ellipsoid reached the floor in 0/5, 0/5, 0/5, 2/5 and 5/5 replicates at 90, 250, 500, 1,000 and 2,000 evaluations, so the main budget is **2,000**. Feasibility: 0.12 s per CMA-ES run on average; 26 of 80 runs stopped early, all `"numerical_precision"`, none `"other"`.

**BO budget:** BO runs to a lower ceiling, 250 or 90 evaluations, chosen by the BO smoke test on runtime alone (D21), because exact GP fitting has cubic cost in the number of observations (Rasmussen & Williams, 2006, Algorithm 2.1). All BO-versus-ES comparisons use the shared checkpoints, since algorithms are only comparable up to the smallest budget given to any of them (Hansen et al., 2016a). **Outcome:** ceiling 250 (smoke test `20261110T095856Z`).

**Status:** Frozen: main budget 2,000; BO ceiling 250; checkpoints 90, 250, 500, 1,000, 2,000.

### D12. Pair methods on the same problem instances

**Decision:** Use a fixed base seed of `4212`. Derive each problem seed, which determines $x_{\mathrm{opt}}$ and $R$, from the condition and replicate index only, so that all methods within a replicate face the same problem instance. Derive method-specific random seeds separately. Run 50 replicates per condition. Main runs, pilot and smoke test use separate seed streams (offsets 0, $10^6$ and $2\times10^6$). Save the problem instances, seeds, software versions, and actual evaluation counts.

The optimum is drawn before the rotation, so the aligned and rotated instances in a replicate share $x_{\mathrm{opt}}$ and differ only in orientation.

**Rationale:** Each replicate is a block in which every method and orientation faces the same instance, so instance-to-instance difficulty cancels (Montgomery, Sections 2.5 and 7.2). Fifty replicates keep the paired bootstrap intervals stable while letting 400 BO runs fit the compute cap. Separate method seeds preserve reproducibility without forcing every method to consume identical random-number streams. The seed values are conventions; what matters is that they are fixed and separate.

**Status:** Accepted; implemented.

### D13. Record evaluation-level diagnostics

**Decision:** Record candidate locations, objective values, best-so-far values, evaluation numbers, boundary events, method identifiers, and seeds. Save CMA-ES covariance snapshots (with their true evaluation counts, since CMA-ES updates once per generation) and BO hyperparameter diagnostics, including ARD lengthscales logged from a GP fitted on exactly $c$ points at checkpoint $c$. Record wall-clock time and BO's surrogate overhead separately from evaluation counts.

**Rationale:** Final regret alone describes performance but does not explain how the optimisers behave. These diagnostics support the mechanism analysis, and the timing supports the cost analysis.

**Status:** Accepted; implemented in `runner.py`, `bo.py`, `run_pilot.py` and `run_main.py`.

---

## 5. Analysis

### D14. Use simple regret as the primary outcome

**Decision:** Define simple regret as

$$
r=f(x_{\mathrm{best}})-f(x_{\mathrm{opt}}),
$$

where $x_{\mathrm{best}}$ is the best point evaluated so far (D6), and analyse $L=\log_{10}(\max(r, 10^{-8}))$.

**Sources:** $r$ is the precision $\Delta f$ of BBOB/COCO, "a difference to the optimal function value" (Hansen et al., 2009b, Section 1.1, p. 2), recorded as best-so-far value minus the optimum (ibid., Appendix E.3.2). Precision is plotted on a $\log_{10}$ scale (ibid., Appendix D.5), so differences are ratios (Hansen et al., 2016b, Section 2).

**Log-regret floor rule (fixed before the pilot was run):** The floor is $10^{-8}$, the final BBOB target (Hansen et al., 2009b, Section 1.1) and the smallest of COCO's standard targets, which run from $10^2$ to $10^{-8}$ (Hansen et al., 2021). If every CMA-ES variant reaches $10^{-8}$ in all $\kappa=1$ pilot replicates, the floor is frozen at $10^{-8}$; otherwise it is the largest final regret attained across the $\kappa=1$ CMA-ES runs. Clipping at the floor is a design choice: CMA-ES converges to about $10^{-25}$ or exactly 0, where differences reflect machine precision rather than optimisation.

**Sensitivity analysis:** the primary comparisons are repeated at $10^{-6}$ and $10^{-10}$: two orders of magnitude either side on the log scale. Moving the floor by $k$ orders changes each log regret by at most $k$, so the interaction by at most $2k=4$ orders; both values stay far above where CMA-ES's precision stops fire.

**Limitation:** regret at a checkpoint is a fixed-cost measurement; Hansen et al. (2009b, Appendix D.4) prefer fixed-target measurement for benchmarking. It is used here because the research question concerns a fixed evaluation budget.

**Pilot outcome:** all 30 $\kappa=1$ CMA-ES runs reached $10^{-8}$ (worst final regret $7\times10^{-24}$), so the floor is frozen at $10^{-8}$.

**Status:** Frozen.

### D15. Primary comparisons: the method × orientation interaction

**Decision:** At $d=4$ and $\kappa=10^4$, compare each axis-aligned method $M$ with full CMA-ES using

$$
\Delta_M
=(\bar L_{\mathrm{rot},M}-\bar L_{\mathrm{align},M})
-(\bar L_{\mathrm{rot,full}}-\bar L_{\mathrm{align,full}}).
$$

There are two primary comparisons: $M$ = diagonal CMA-ES at 2,000 evaluations, and $M$ = ARD BO at the BO ceiling (250), with full CMA-ES read at the same checkpoint. Diagonal CMA-ES is also reported at the BO ceiling as a secondary comparison, so all three are compared at one budget.

**Source:** $\Delta_M$ is the method × orientation interaction in a 2 × 2 factorial design (Montgomery, Chapters 5–6, Equation 6.3). Montgomery's interaction effect $AB$ includes a factor of ½; $\Delta_M = 2AB$ is reported unhalved so it reads directly as the extra orders of magnitude of regret that rotation costs $M$. It is computed per replicate, $\Delta_i$, and averaged (each replicate is a block; Montgomery, Sections 2.5 and 7.2). The same arithmetic is called difference-in-differences in econometrics, but this is a controlled experiment, so no parallel-trends assumption is needed.

**Interpretation:** a positive value means rotation increases the regret of $M$ more than that of full CMA-ES. $\Delta_M$ measures differential sensitivity to rotation; it is consistent with, but does not by itself prove, the explanation that $M$ cannot model correlations, because rotation also changes the valley's position relative to the fixed box. The causal reading rests on D18 and D22 ($\kappa_J$, $\kappa_{\mathrm{eff}}$, the uniform-search box baseline).

**Status:** Accepted.

### D16. Paired uncertainty estimates and labelled analyses

**Decision:** Report 95% paired percentile bootstrap confidence intervals (Efron & Hastie, 2016), resampling whole replicates (so each block's four cells stay together), with 10,000 bootstrap replicates and seed 4212. At least about 1,000 replicates are recommended for confidence intervals (Efron & Tibshirani, 1993, as cited in arXiv:1807.03582), and Hesterberg recommends at least $10^4$ to minimise resampling variability (as cited in a secondary source; see Section 9); 10,000 meets the stricter recommendation at negligible cost.

Analyses beyond the two primary comparisons are labelled in advance (D22) as secondary, exploratory or robustness, following the principle that a study's goals shape its design (Bartz-Beielstein et al., 2020, Sections 1–2).

**Status:** Frozen before the analysis.

### D17. Use the $\kappa=1$ control to assess orientation invariance

**Decision:** At $\kappa=1$, aligned and rotated problems are mathematically identical, so each method's aligned-minus-rotated difference should be zero apart from noise and box effects.

**Tolerance rule (fixed before the pilot was run):** A method is treated as invariant to rotation at $\kappa=1$ if the 95% paired bootstrap confidence interval for its mean aligned-minus-rotated log-regret difference lies entirely within $\pm 0.5$ (a factor of about 3 in regret): small enough to detect implementation errors, large enough not to fail on floating-point differences.

**Rationale:** Because the start box is not rotated with the problem, exact invariance is not expected in general: in a similar check, the only systematic effect was initialisation in the non-rotated hypercube $[-5,5]^n$ (Loshchilov et al., 2017).

**Pilot outcome:** every difference at $\kappa=1$ was exactly zero. Because the rotated sphere is the same function and aligned and rotated runs share $x_{\mathrm{opt}}$, $x_0$ and seeds, the runs are identical. This confirms that rotation, seeding and pairing are implemented correctly, but it does not test box effects, which can only arise for $\kappa>1$; those are examined through the uniform-search penalty and bound-repair counts.

**Status:** Rule satisfied at the pilot and in the main runs (Section 7).

### D18. Use mechanism diagnostics to interpret results

**Decision:** Examine the effective condition number

$$
\kappa_{\mathrm{eff}}
=\mathrm{cond}(C^{1/2}HC^{1/2}),
$$

where $C$ is the CMA-ES search covariance and $H$ the objective Hessian, and BO ARD lengthscales relative to the axis curvatures.

**Rationale:** On a quadratic, setting the covariance to the inverse Hessian turns the ellipsoid into a sphere (Hansen, 2023, Section 0.4), so $\kappa_{\mathrm{eff}}$ measures how much of the geometry a method has learned. Under an idealised model, the covariance of selected solutions becomes proportional to the inverse Hessian as the population grows (Shir & Yehudayoff, 2020), so $\kappa_{\mathrm{eff}} = 1$ is an idealised target. ARD lengthscales indicate the relevance BO assigns to each input (Rasmussen & Williams, 2006, p. 106). The diagnostics support interpretation but do not, by themselves, establish causation.

**Implementation note:** With bound handling, pycma's internal covariance lives in a transformed space in which coordinates can be mirrored. $\kappa_{\mathrm{eff}}$ is therefore computed from the covariance of the evaluated points, obtained with the Jacobian of the bound transform; this matches the empirical covariance of pycma's own samples.

**Status:** Accepted; implemented and validated against sampled covariances.

---

## 6. Configuration and Implementation Rules

### D19. Keep unresolved values explicit

**Decision:** Represent undecided settings as `null` in `configs/experiment.yaml`. The configuration loader validates required settings before each stage: the main stage refuses to run until the BO ceiling and acquisition setting are set and the ceiling is a shared checkpoint; the analysis stage requires the bootstrap and uniform-search settings.

**Status:** Accepted; implemented and tested.

### D20. Freeze settings before each stage

**Decision:** Do not start a stage until the settings it depends on are frozen and the tests pass: pilot rules before the pilot; BO settings and smoke-test rules before the smoke test; all main-run settings before the main runs; analysis settings before the analysis. Pilot settings live in a separate `pilot` block and are applied only at the pilot stage.

**Rationale:** Settings may be selected from pilot or smoke-test data by pre-stated rules, but must be documented and frozen before the results they could affect, to avoid outcome-driven changes.

**Status:** Satisfied at every stage (Section 7).

### D21. Optimiser settings and tuning policy

**Decision:** No hyperparameter tuning is performed for any method. Each method uses published defaults or settings fixed in this log before the stage that uses them. Conclusions are therefore conditional on default configurations.

- **CMA-ES variants.** All three use the default strategy parameters of pycma 4.5.0 (Hansen et al., 2019). Hansen (2023, p. 32) describes the defaults as a robust setting applicable to a wide range of functions. The population size is $\lambda = 4 + \lfloor 3\ln n \rfloor = 8$ (Hansen, 2023, Table 1, Equation 48). The initial point and step size follow a uniform start with $\sigma_0 = 0.3\,(b-a) = 3$ (Hansen, 2023, Appendix A, Figure 6, footnote 1). The variants differ only in their covariance option (D7, D8).¹
- **Uniform random search** samples the same box and has no settings to tune.
- **BO** uses the ARD Matérn 5/2 kernel (Snoek et al., 2012, Section 3.1) and an isotropic Matérn 5/2 variant, with expected improvement (Jones et al., 1998), implemented with scikit-learn's Gaussian process. GP hyperparameters are fitted by type-II maximum likelihood (Rasmussen & Williams, 2006, Section 5.4.1, Equation 5.8), which is part of the method, not tuning. All other BO settings follow Le Riche & Picheny (2021, "L&P"), who tested them for BO on the COCO benchmark:
  - **Initial design:** $d+4 = 8$ points, the number of GP hyperparameters plus two, as a maximin Latin hypercube shared by both BO variants and counted in the budget (L&P, Section 3.2; McKay et al., 1979). L&P found no drawback over the first $30d$ evaluations and a substantial early gain (Section 6.1.1). The $10d$ convention (Jones et al., 1998; Loeppky et al., 2009) targets emulation accuracy and is not used.
  - **Hyperparameter fitting:** $2d = 8$ starting points (L&P, Section 4.1); the marginal likelihood can have multiple optima (Rasmussen & Williams, 2006, p. 115). scikit-learn draws restarts log-uniformly rather than from a Latin hypercube (minor difference).
  - **Lengthscale bounds:** $[(U-L)\sqrt d/100,\ (U-L)\sqrt d] = [0.02, 2]$ on the unit box (L&P, Section 4.1). Fits reaching a bound produce convergence warnings, counted but not failures.
  - **Jitter:** $10^{-10}$ (scikit-learn's default); after a failed fit it is multiplied by 100 and kept for the rest of the run, with every event recorded (L&P, Section 4.1: regularisation only after a failure; Gramacy & Lee, 2012: as small as stability allows). It is a numerical stabiliser, not a noise model.
  - **EI optimisation:** EI on $\min(2000, 500d) = 2000$ space-filling candidates, repeated $\min(10, d) = 4$ times (L&P, Section 4.2), either with L-BFGS refinement from the best candidate ("standard") or without it ("random", L&P's second alternative in Section 6.2.1, found almost comparable in dimensions 3 and 5). If EI is zero everywhere, a random point is evaluated; near-duplicate points are never re-evaluated (L&P, Section 4.2).
- **BO smoke test (rules fixed before it ran).** Own seed stream; never uses regret. Verifies that both GPs fit the initial design and that full BO runs complete, recording jitter events. Selects only the ceiling and acquisition setting, by runtime: the first of (250, standard), (250, random), (90, standard), (90, random) whose projected time for all BO main runs (mean run time × runs ÷ workers × 1.25) fits a 2-hour cap, with workers = CPU cores − 2. The cap, margin and worker count are project constraints.
- **Smoke-test outcome (run `20261110T095856Z`):** 40/40 initial-design fits and 8/8 BO runs succeeded, with no jitter escalations. 250 + standard: 118.0 s/run, projected 2.05 h (over the cap); 250 + random: 85.5 s/run, projected 1.49 h. **Ceiling 250, random acquisition.** ARD runs cost 2–3 times as much as isotropic runs.

**What the pilot may change:** only (a) the budget, floor and invariance tolerance, by their pre-stated rules (D11, D14, D17); and (b) a numerical or implementation setting, and only if a failure trigger occurs. Regret values are never used to change $\sigma_0$, population size, kernel or acquisition function.

**Alternatives:** per-method tuning on each problem. Bartz-Beielstein et al. (2020, Section 7.5) note that comparing tuned with untuned methods is unfair; tuning each variant on the test ellipsoids would also tune on the test instances. COCO likewise requires the same parameter setting on all functions (Hansen et al., 2016a). This is stated as a limitation, especially for BO.

**Status:** Frozen.

¹ pycma 4.5.0's defaults match Hansen (2023, Table 1) for $\lambda = 8$, $\mu = 4$, $\mu_{\mathrm{eff}} = 2.60$, $c_c = 0.500$, $c_1 = 0.0652$, $c_\mu = 0.0640$ at $d = 4$. Its step-size constants differ slightly ($c_\sigma = 0.479$, $d_\sigma = 1.479$, versus $0.397$ and $1.397$); they are identical across all three variants, so they cannot bias the comparison.

### D22. Secondary, exploratory and robustness analyses

**Decision:** Beyond the two primary comparisons (D15), the following analyses were specified before any results were seen:

1. **Budget-dependent invariance (secondary).** Invariance yields uniform performance only after the algorithm's state has adapted (Hansen, 2023, p. 24, footnote 26). Test whether full CMA-ES's rotation penalty shrinks across checkpoints while diagonal CMA-ES's and ARD BO's does not.
2. **Conditioning ladder (secondary).** Rotation penalty of each CMA-ES variant and uniform search across $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$.
3. **$\kappa_J$ test (secondary, pre-registered at $\kappa=10^4$).** $\kappa_J$, the condition number after Jacobi (diagonal) rescaling (Bradley & Murray, 2012, Section 2), is within a factor $d$ of the best diagonal rescaling (van der Sluis, 1969, as cited in Bradley & Murray, 2012; Jambulapati et al., 2021, Section 3 and Appendix B) and equals 1 for every aligned instance. Since a diagonal model can at best apply a diagonal rescaling, $\kappa_J/d$ bounds the conditioning it cannot remove. Report the Spearman correlation between per-instance $\kappa_J$ and the diagonal-minus-full log-regret gap on rotated instances.
4. **Isotropic-kernel BO versus ARD BO (exploratory).**
5. **Cost (exploratory).** Wall-clock time alongside evaluations, with a break-even analysis of the evaluation cost at which BO's overhead pays off (Rasmussen & Williams, 2006, Algorithm 2.1).
6. **Robustness.** Floor sensitivity (D14), performed. The optional $\sigma_0 \in \lbrace 0.15, 0.5 \rbrace \times$ width and $d \in \lbrace 2, 8 \rbrace$ runs were **not performed** for lack of time; this is stated as a limitation.

**Alternatives:** a second function family, rejected to keep the explanation testable within the time available.

**Status:** Accepted; items 1–5 and the floor sensitivity performed.

---

## 7. Experiment Record

There are no outstanding decisions. Every stage ran with frozen settings.

| Stage | Run | Outcome |
|---|---|---|
| Pilot | `20261103T025402Z` | Budget 2,000; floor $10^{-8}$; $\kappa=1$ control exactly zero (D11, D14, D17) |
| BO smoke test | `20261110T095856Z` | Ceiling 250, random acquisition; all fits and runs succeeded (D21) |
| Main runs | `main` (`results/raw/main/main/`) | 2,400 runs in 98.8 min on 8 workers (10-core Apple Silicon laptop, light background load); 1,749 budget stops, 651 numerical-precision stops, 0 other |
| Analysis | `results/processed/` | Settings confirmed before running (D16, D6) |

**Sanity checks (all passed):** no `"other"` stops; $\kappa=1$ aligned-minus-rotated differences zero for every method (CMA-ES to machine precision; BO within tolerance); isotropic ES $\kappa_{\mathrm{eff}}$ fixed at $\kappa$; full CMA-ES $\kappa_{\mathrm{eff}}$ on rotated $\kappa=10^4$ falling from about 4,900 to about 3, the level it reaches on $\kappa=1$; diagonal CMA-ES $\kappa_{\mathrm{eff}}$ staying around 6,500–8,400 when rotated; uniform search matching its success law at $\kappa=1$.

**Recorded discrepancy:** uniform search at $\kappa=10$, rotated, succeeded more often than predicted at 1,000 and 2,000 evaluations (11/50 vs about 5 expected at 2,000, about 3 standard deviations). It is one condition among about 20 informative comparisons, and cumulative checkpoints share runs; it is reported, not adjusted.

**Primary results (D15):** diagonal vs full CMA-ES at 2,000: $\Delta$ = +8.60 [8.32, 8.84]; at floors $10^{-6}$ / $10^{-10}$: 6.60 / 10.60 (three of four cells at the floor, so the size is floor-bound while sign and significance are robust). ARD BO vs full CMA-ES at 250: $\Delta$ = −0.04 [−0.27, 0.21], unchanged across floors. Secondary at 250: diagonal vs full $\Delta$ = +0.91 [0.62, 1.20]. The pre-registered $\kappa_J$ test at $\kappa=10^4$ was null ($\rho$ = −0.06 [−0.35, 0.24]); at $\kappa$ = 100 and 1,000, $\rho$ = 0.76 and 0.63 (supporting).

---

## 8. Change-Control Rule

Any change to a decision above is recorded here with its date, the decision changed, the previous and new setting, the reason, whether pilot or main runs had started, and the effect on comparability. Runs are not silently rerun or omitted because their results are inconvenient. The final report describes the settings actually used.

### Change record: pilot re-run with final code (3 November 2026)

- **Decision changed:** official pilot run for D11, D14 and D17.
- **Previous setting:** pilot run `20261103T022221Z`.
- **New setting:** pilot run `20261103T025402Z`, produced by the final code (commit `1c32a07`).
- **Reason:** after the first pilot, the code was updated (covariance snapshot labels, pilot plot floor, BO diagnostics, documentation). The pilot was re-run so the official pilot comes from the final code.
- **Runs already started:** pilot only.
- **Effect:** none. Regret at every checkpoint, evaluations used and stop categories are identical across the two runs. Both runs are retained in `results/raw/pilot/`.

### Change record: BO settings revised to Le Riche & Picheny (2021) (before any BO run)

- **Decision changed:** D21, BO settings.
- **Previous setting:** scikit-learn default lengthscale bounds ($10^{-5}$ to $10^{5}$); initial design size to be chosen by a stability rule over $\lbrace 2d, \ldots, 5d \rbrace$; fixed jitter chosen from $\lbrace 10^{-8}, 10^{-6}, 10^{-4} \rbrace$; an unsourced "light" acquisition setting.
- **New setting:** the L&P settings in D21 (initial design $d+4$, maximin Latin hypercube, $2d$ likelihood starts, bounds $[0.02, 2]$, failure-triggered jitter, L&P's EI optimiser and its random-search variant). The smoke test now verifies BO and chooses only the ceiling and acquisition setting.
- **Reason:** reading Le Riche & Picheny (2021) in full showed a single benchmarked study covering nearly every BO setting, replacing judgement calls with published choices.
- **Runs already started:** pilot only (no BO in the pilot); no BO run of any kind had been made.
- **Effect:** none on comparability.

### Change record: description-only updates (10–11 November 2026)

- D15 reworded as the method × orientation interaction (Montgomery) instead of an econometric difference-in-differences; the formula and code are unchanged.
- `experiment.yaml`: the uniform-search `curve_method` corrected to describe the independent paired runs the code has always performed; the optional robustness runs moved to a "not implemented" list.
- Bootstrap count (10,000) and uniform-search criterion (regret ≤ 1) confirmed, with their justifications, before the analysis was run.
- **Effect:** none on any run or result.

---

## 9. References

***Verify every bibliographic detail before submission.*** Entries marked [check] still need confirming.

- Akimoto, Y., & Hansen, N. (2020). Diagonal acceleration for covariance matrix adaptation evolution strategies. *Evolutionary Computation, 28*(3), 405–435. https://arxiv.org/abs/1905.05885
- Bartz-Beielstein, T., Doerr, C., van den Berg, D., Bossek, J., Chandrasekaran, S., Eftimov, T., Fischbach, A., Kerschke, P., La Cava, W., López-Ibáñez, M., Malan, K. M., Moore, J. H., Naujoks, B., Orzechowski, P., Volz, V., Wagner, M., & Weise, T. (2020). *Benchmarking in optimization: Best practice and open issues* (arXiv:2007.03488). arXiv. https://doi.org/10.48550/arXiv.2007.03488
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Bradley, A. M., & Murray, W. (2012). *Matrix-free approximate equilibration* (arXiv:1110.2805). arXiv. https://arxiv.org/abs/1110.2805
- Brockhoff, D. (2023). *Comparing boundary handling techniques of CMA-ES on the bbob and sbox-cost test suites.* https://researchportal.ip-paris.fr/en/publications/comparing-boundary-handling-techniques-of-cma-es-on-the-bbob-and-/ [check venue]
- Efron, B., & Hastie, T. (2016). *Computer age statistical inference*. Cambridge University Press. https://web.stanford.edu/~hastie/CASI_files/PDF/casi_corrected_11102017.pdf [check chapter numbers]
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall. [secondary citation, via arXiv:1807.03582]
- Frazier, P. I. (2018). *A tutorial on Bayesian optimization* (arXiv:1807.02811). arXiv. https://arxiv.org/abs/1807.02811
- Gramacy, R. B., & Lee, H. K. H. (2012). Cases for the nugget in modeling computer experiments. *Statistics and Computing, 22*(3), 713–722. https://arxiv.org/abs/1007.4580
- Hansen, N. (2023). *The CMA evolution strategy: A tutorial* (arXiv:1604.00772; Version 2). arXiv. https://doi.org/10.48550/arXiv.1604.00772
- Hansen, N., Akimoto, Y., & Baudis, P. (2019). *CMA-ES/pycma on GitHub* (Version 4.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2559634
- Hansen, N., Auger, A., Brockhoff, D., Tušar, D., & Tušar, T. (2016b). *COCO: Performance assessment* (arXiv:1605.03560). arXiv. https://arxiv.org/abs/1605.03560
- Hansen, N., Auger, A., Finck, S., & Ros, R. (2009b). *Real-parameter black-box optimization benchmarking 2009: Experimental setup* (Research Report RR-6828). INRIA. https://scispace.com/pdf/real-parameter-black-box-optimization-benchmarking-2009-9t7c9yda5e.pdf
- Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software, 36*(1), 114–144. https://arxiv.org/abs/1603.08785
- Hansen, N., Finck, S., Ros, R., & Auger, A. (2009a). *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions* (Research Report RR-6829). INRIA. https://hal-lara.archives-ouvertes.fr/inria-00362633v1
- Hansen, N., Tušar, T., Mersmann, O., Auger, A., & Brockhoff, D. (2016a). *COCO: The experimental procedure* (arXiv:1603.08776). arXiv. https://arxiv.org/abs/1603.08776
- *Hyperparameter optimization in machine learning* (2024) (arXiv:2410.22854). arXiv. https://arxiv.org/abs/2410.22854 [check authors]
- Jambulapati, A., Li, J., Musco, C., Sidford, A., & Tian, K. (2021). *Fast and near-optimal diagonal preconditioning* (arXiv:2008.01722; Version 2). arXiv. https://arxiv.org/abs/2008.01722
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization, 13*(4), 455–492. https://doi.org/10.1023/A:1008306431147
- Le Riche, R., & Picheny, V. (2021). Revisiting Bayesian optimization in the light of the COCO benchmark. *Structural and Multidisciplinary Optimization, 64*(5), 3063–3087. https://doi.org/10.1007/s00158-021-02977-1
- Letham, B., Calandra, R., Rai, A., & Bakshy, E. (2020). Re-examining linear embeddings for high-dimensional Bayesian optimization. In *Advances in neural information processing systems 33*. https://neurips.cc/virtual/2020/poster/18297
- Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the sample size of a computer experiment: A practical guide. *Technometrics, 51*(4), 366–376. https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/LoeppkySacksWelch_TNX2009.pdf
- Long, F. X., Vermetten, D., van Stein, B., & Kononova, A. V. (2022). *BBOB instance analysis: Landscape properties and algorithm performance across problem instances* (arXiv:2211.16318). arXiv. https://doi.org/10.48550/arXiv.2211.16318
- Loshchilov, I. (2014). *A computationally efficient limited memory CMA-ES for large scale optimization* (arXiv:1404.5520). arXiv. https://arxiv.org/abs/1404.5520
- Loshchilov, I., Glasmachers, T., & Beyer, H.-G. (2017). *Limited-memory matrix adaptation for large scale black-box optimization* (arXiv:1705.06693). arXiv. https://arxiv.org/abs/1705.06693
- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics, 21*(2), 239–245. https://doi.org/10.2307/1268522
- Mezzadri, F. (2007). How to generate random matrices from the classical compact groups. *Notices of the American Mathematical Society, 54*(5), 592–604. https://arxiv.org/abs/math-ph/0609050
- Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley. [check edition, chapter and equation numbers; course notes following it: https://ast301.netlify.app/dox/ch06]
- Raponi, E., et al. (2023). *Optimizing with low budgets: A comparison on the black-box optimization benchmarking suite and OpenAI Gym* (arXiv:2310.00077). arXiv. https://arxiv.org/abs/2310.00077 [check authors]
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press. https://gaussianprocess.org/gpml/
- Ros, R., & Hansen, N. (2008). A simple modification in CMA-ES achieving linear time and space complexity. In G. Rudolph, T. Jansen, S. Lucas, C. Poloni, & N. Beume (Eds.), *Parallel problem solving from nature – PPSN X* (pp. 296–305). Springer.
- Santoni, M. L., Raponi, E., De Leone, R., & Doerr, C. (2024). Comparison of high-dimensional Bayesian optimization algorithms on BBOB. *ACM Transactions on Evolutionary Learning and Optimization, 4*(3). https://arxiv.org/abs/2303.00890
- Shir, O. M., & Yehudayoff, A. (2020). On the covariance-Hessian relation in evolution strategies. *Theoretical Computer Science, 801*, 157–174. https://cris.iucc.ac.il/en/publications/on-the-covariance-hessian-relation-in-evolution-strategies/
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In F. Pereira, C. J. C. Burges, L. Bottou, & K. Q. Weinberger (Eds.), *Advances in neural information processing systems 25* (pp. 2951–2959). Curran Associates. https://arxiv.org/abs/1206.2944
- Vermetten, D., López-Ibáñez, M., Mersmann, O., Allmendinger, R., & Kononova, A. V. (2023). *Analysis of modular CMA-ES on strict box-constrained problems in the SBOX-COST benchmarking suite* (arXiv:2305.15102). arXiv. https://doi.org/10.48550/arXiv.2305.15102

*Secondary citations (originals not read; cited as "as cited in"):* van der Sluis (1969), via Bradley & Murray (2012) and Jambulapati et al. (2021); Efron & Tibshirani (1993), via arXiv:1807.03582; Hesterberg's recommendation of at least $10^4$ bootstrap samples, via a secondary source (https://www.dlib.si/stream/URN:NBN:SI:DOC-3JCGSX03/7b8eb174-43f5-45f7-9592-42ec301ccea9/PDF); Brooks (1958), via arXiv:2410.22854.