# Decision Log — DSA4212 Project

**Project:** When Does Geometry-Aware Search Pay Off? Bayesian Optimisation versus CMA-ES on Rotated Ill-Conditioned Problems  

This log records each experimental design decision, the reason for it, and its outcome. Every setting was fixed before the results it could affect were seen. Citations follow APA 7; references are listed at the end.

---

## 1. Research Question

### D1. Research question

**Decision:** Test whether rotating an ill-conditioned problem hurts methods with an axis-aligned model (diagonal CMA-ES, ARD BO) more than full CMA-ES. The motivation is given in the report (Sections 1–2).

---

## 2. Test Problems

### D2. A four-dimensional ellipsoid

**Decision:** Use the ellipsoid $f(x)=\sum_{i=1}^{d}\kappa^{(i-1)/(d-1)}z_i^2$, $z=R(x-x_{\mathrm{opt}})$, with $d=4$ and $\kappa \in \lbrace 1, 10, 10^2, 10^3, 10^4 \rbrace$, each aligned ($R=I$) and rotated. BO runs at $\kappa \in \lbrace 1, 10^4 \rbrace$ only, because of its cost.

**Reason:** This is the BBOB ellipsoid (functions f2 and f10; Hansen et al., 2009a) with the conditioning made a parameter, so conditioning and orientation vary independently while the optimum and Hessian are known exactly. At $\kappa=1$ rotation changes nothing, giving a control. $d=4$ lies in BO's usual range (Frazier, 2018), lets full CMA-ES adapt within the budget, and keeps 400 BO runs feasible.

### D3. Random rotations

**Decision:** Generate rotations by QR decomposition of a Gaussian matrix with sign correction, and save each matrix.

**Reason:** The sign correction makes the rotations Haar-distributed (Mezzadri, 2006).

### D4. Search domain and optimum

**Decision:** Search domain $[-5,5]^4$; optimum drawn uniformly in $[-4,4]^4$, independently of every method's start; bounds handled by pycma's `BoundTransform`, with repairs logged.

**Reason:** These are the BBOB conventions (Hansen et al., 2009b). Bound handling is pycma's default (Hansen et al., 2019).

### D5. Out of scope

**Decision:** The sphere is used only for tests. Multimodal, noisy and higher-dimensional problems are not studied.

**Reason:** They would add questions the design is not built to answer within the time available.

---

## 3. Methods

### D6. Six methods

**Decision:** Full CMA-ES, diagonal CMA-ES, isotropic ES, BO with an ARD Matérn 5/2 kernel, BO with an isotropic Matérn 5/2 kernel, and uniform random search. Every method reports its best evaluated point.

**Reason:** The methods form a ladder from no geometry model to a full one in each family. Isotropic ES keeps only step-size control, which can run without the covariance update (Hansen, 2016). An isotropic kernel is invariant to rotations (Rasmussen & Williams, 2006). Uniform search is a no-model baseline (Hansen et al., 2016), and is validated against its success law $1-(1-q)^N$ (Bergstra & Bengio, 2012) with success defined as regret $\le 1$, a standard COCO target (Hansen et al., 2021); this is the only decade target for which the check is informative at $\kappa=1$.

### D7. Separable CMA-ES as the diagonal variant

**Decision:** Use separable CMA-ES (`CMA_diagonal=True`; Ros & Hansen, 2008).

**Reason:** It restricts the covariance to be diagonal, isolating the effect of not learning correlations. Diagonal-decoding CMA-ES (Akimoto & Hansen, 2020) was rejected because it keeps a full covariance.

### D8. Common CMA-ES settings

**Decision:** The three CMA-ES variants share every setting except their covariance option; the configuration loader rejects any override.

**Reason:** Differences should reflect covariance adaptation only.

### D9. Initialisation

**Decision:** Start each CMA-ES variant from the same uniform random point in the box, with $\sigma_0 = 0.3 \times$ box width $= 3$.

**Reason:** This is Hansen's (2016) rule for a uniform start, and uses no knowledge of the optimum (Hansen et al., 2016).

### D10. Early termination

**Decision:** Disable pycma's tolerance-based stops. Runs ended by its numerical-precision checks keep their best value; any other early stop is investigated.

**Reason:** The precision checks fire when a step no longer changes the solution (Hansen, 2016); excluding such runs would bias results against the fastest methods.

---

## 4. Budget and Reproducibility

### D11. Budget, checkpoints and BO ceiling

**Decision:** Evaluations are counted exactly, and results recorded at 90, 250, 500, 1,000 and 2,000 evaluations. A pilot set the main budget to **2,000**, the first checkpoint at which full CMA-ES solved the rotated $\kappa=10^4$ problem in at least 80% of runs. A runtime smoke test set BO's ceiling to **250** (D21), where BO is compared with the other methods.

**Reason:** The budget changes which family wins (Raponi et al., 2023), and algorithms are only comparable up to the smallest budget given to any of them (Hansen et al., 2016).

### D12. Paired instances

**Decision:** Base seed 4212. All methods and both orientations in a replicate share the same optimum; method seeds are derived separately. 50 replicates per condition. Main runs, pilot and smoke test use separate seed streams.

**Reason:** Each replicate is a complete block, so instance-to-instance differences cancel (Montgomery, 2017). Fifty replicates give stable paired intervals within the compute cap.

### D13. Diagnostics

**Decision:** Log every evaluation, CMA-ES covariance snapshots, BO lengthscales, and wall-clock time.

**Reason:** Final regret shows *what* happened; the diagnostics test *why* (D18), and timing supports the cost analysis.

---

## 5. Analysis

Definitions, formulas and sources are given in the report (Section 3.4). The rules below were fixed before any results were seen.

- **D14. Outcome:** $\log_{10}$ simple regret, floored at $10^{-8}$ (confirmed by the pilot: all 30 $\kappa=1$ CMA-ES runs reached it), with sensitivity checks at floors of $10^{-6}$ and $10^{-10}$.
- **D15. Primary comparisons:** the method × orientation interaction at $\kappa=10^4$, for diagonal CMA-ES at 2,000 evaluations and ARD BO at 250, each against full CMA-ES at the same checkpoint.
- **D16. Uncertainty:** 95% paired percentile bootstrap intervals over replicates, 10,000 resamples, seed 4212.
- **D17. $\kappa=1$ control:** a method counts as rotation-invariant if the 95% interval of its aligned-minus-rotated difference lies within $\pm 0.5$.
- **D18. Mechanism diagnostics:** the effective condition number $\kappa_{\mathrm{eff}}$, the Jacobi-scaled condition number $\kappa_J$, and the correlation between ARD lengthscales and the axis curvatures.

---

## 6. Settings and Freezing

### D19. Explicit unresolved settings

**Decision:** Undecided settings are `null` in `configs/experiment.yaml`; the loader refuses to run a stage until its settings are set.

### D20. Freeze before each stage

**Decision:** Pilot rules before the pilot, BO rules before the smoke test, all settings before the main runs, and analysis settings before the analysis.

**Reason:** Settings chosen from pilot or smoke-test data must be fixed by stated rules, never by the results they could affect.

### D21. Optimiser settings; no tuning

**Decision:** No method is tuned per problem. CMA-ES uses pycma's defaults (Hansen et al., 2019); BO follows Le Riche and Picheny (2021). The full settings are listed in the report (Section 3.2). A BO smoke test chose only the ceiling and whether EI is refined locally, by runtime: the first of (250, refined), (250, not refined), (90, refined), (90, not refined) whose projected time for all BO main runs fitted a two-hour cap on 8 workers.

**Outcome:** ceiling 250, EI not refined (see Section 7).

### D22. Further analyses (fixed in advance)

Secondary: the rotation penalty across checkpoints and across $\kappa$, and the correlation of $\kappa_J$ with the diagonal-minus-full gap at $\kappa=10^4$. Exploratory: isotropic versus ARD BO, and BO's cost and break-even evaluation cost. Robustness: floor sensitivity. Optional checks over $\sigma_0$ and $d$ were not run.

---

## 7. Experiment Record

| Stage | Run | Outcome |
|---|---|---|
| Pilot | `20261103T025402Z` | Budget 2,000; floor $10^{-8}$; $\kappa=1$ control exactly zero |
| BO smoke test | `20261110T095856Z` | Ceiling 250, EI without refinement |
| Main runs | `results/raw/main/main/` | 2,400 runs in 98.8 min on 8 workers; 1,749 budget stops, 651 numerical-precision stops, none other |
| Analysis | `results/processed/` | All settings fixed before running |
| Report | `report/` | LaTeX source, bibliography and figures |

**Checks:** the $\kappa=1$ control gave identical results for all six methods (within $10^{-15}$); uniform search matched its success law at $\kappa=1$, but succeeded more often than predicted at $\kappa=10$, rotated (about three standard deviations), which is reported without adjustment.

---

## 8. Changes

- **Pilot re-run (3 November 2026):** the pilot was re-run with the final code (run `20261103T025402Z`, replacing `20261103T022221Z`); every result was identical.
- **BO settings (before any BO run):** earlier provisional BO settings were replaced by those of Le Riche and Picheny (2021), a benchmark study covering nearly every BO setting. No BO run had been made.
- **Wording (10–11 November 2026):** D15 was reframed as a factorial interaction (the formula and code are unchanged), and the bootstrap count and uniform-search criterion were confirmed before the analysis.

---

## References

Akimoto, Y., & Hansen, N. (2020). Diagonal acceleration for covariance matrix adaptation evolution strategies. *Evolutionary Computation, 28*(3), 405–435. https://arxiv.org/abs/1905.05885

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Frazier, P. I. (2018). *A tutorial on Bayesian optimization*. arXiv. https://doi.org/10.48550/arXiv.1807.02811

Hansen, N. (2016). *The CMA evolution strategy: A tutorial*. arXiv. https://doi.org/10.48550/arXiv.1604.00772

Hansen, N., Akimoto, Y., & Baudis, P. (2019). *CMA-ES/pycma on GitHub* (Version 4.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.2559634

Hansen, N., Auger, A., Finck, S., & Ros, R. (2009b). *Real-parameter black-box optimization benchmarking 2009: Experimental setup* (Research Report RR-6828). INRIA.

Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software, 36*(1), 114–144. https://arxiv.org/abs/1603.08785

Hansen, N., Finck, S., Ros, R., & Auger, A. (2009a). *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions* (Research Report RR-6829). INRIA. https://hal-lara.archives-ouvertes.fr/inria-00362633v1

Hansen, N., Tušar, T., Mersmann, O., Auger, A., & Brockhoff, D. (2016). *COCO: The experimental procedure*. arXiv. https://arxiv.org/abs/1603.08776

Le Riche, R., & Picheny, V. (2021). Revisiting Bayesian optimization in the light of the COCO benchmark. *Structural and Multidisciplinary Optimization, 64*(5), 3063–3087. https://doi.org/10.1007/s00158-021-02977-1

Mezzadri, F. (2006). *How to generate random matrices from the classical compact groups*. arXiv. https://doi.org/10.48550/arXiv.math-ph/0609050

Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). John Wiley & Sons.

Raponi, E., Carraz Rakotonirina, N., Rapin, J., Doerr, C., & Teytaud, O. (2023). *Optimizing with low budgets: A comparison on the black-box optimization benchmarking suite and OpenAI Gym*. arXiv. https://doi.org/10.48550/arXiv.2310.00077

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press.

Ros, R., & Hansen, N. (2008). A simple modification in CMA-ES achieving linear time and space complexity. In *Parallel problem solving from nature – PPSN X* (pp. 296–305). Springer. https://doi.org/10.1007/978-3-540-87700-4_30