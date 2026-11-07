"""Bayesian optimisation under an exact evaluation budget (decision log D21, D22).

Two variants that differ only in the kernel's lengthscale structure:

    kernel="ard"        Matern 5/2 with one lengthscale per input axis
                        (Snoek et al., 2012, Sec. 3.1; Rasmussen & Williams,
                        2006, p. 106). Axis-aligned model.
    kernel="isotropic"  Matern 5/2 with a single lengthscale. Depends only on
                        distance, so invariant to rotations (Rasmussen &
                        Williams, 2006, Sec. 4.1, p. 80).

Everything else is shared and follows Le Riche & Picheny (2021, "L&P"),
who tested these choices for BO on the COCO benchmark:

  * initial design: maximin Latin hypercube of size d + 4 (L&P Sec. 3.2,
    6.1), shared by both variants and counted in the budget (D11);
  * inputs scaled to the unit box; outputs standardised (L&P Sec. 4.1);
  * GP hyperparameters by maximum marginal likelihood (type-II ML;
    Rasmussen & Williams, 2006, Sec. 5.4.1) from 2d starting points
    (L&P Sec. 4.1);
  * lengthscale bounds [(U-L) sqrt(d)/100, (U-L) sqrt(d)], i.e.
    [sqrt(d)/100, sqrt(d)] on the unit box (L&P Sec. 4.1);
  * minimal jitter by default; if a GP fit fails, a larger jitter is added
    and kept for the rest of the run (L&P Sec. 4.1; Gramacy & Lee, 2012);
  * expected improvement (Jones et al., 1998), maximised by evaluating EI
    on a space-filling design of min(2000, 500d) points and running
    L-BFGS-B from the best point, repeated min(10, d) times (L&P Sec. 4.2).
    The "random" setting drops the L-BFGS step (L&P Sec. 6.2, EirandM).

Every objective call goes through runner.CountingObjective, so the budget
is exact and the reported solution is the best *evaluated* point (D6, D11).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from runner import CountingObjective, RunResult, _finish, derive_seed

__all__ = [
    "BO_METHODS",
    "ACQ_SETTINGS",
    "BORunResult",
    "default_n_init",
    "initial_design",
    "length_scale_bounds",
    "expected_improvement",
    "make_kernel",
    "run_bo",
]

BO_METHODS = {"bo_ard": "ard", "bo_isotropic": "isotropic"}
_BO_SEED_KEYS = {"ard": 2001, "isotropic": 2002}   # distinct from runner.METHOD_IDS

# Acquisition settings (D21). "standard" is L&P's EI optimiser (Sec. 4.2);
# "random" is their random-search-only variant EirandM (Sec. 6.2).
ACQ_SETTINGS = {"standard": {"local_search": True}, "random": {"local_search": False}}

BASE_JITTER = 1e-10          # scikit-learn's default; effectively interpolating
JITTER_ESCALATION = 100.0    # factor applied after each failed fit
MAX_JITTER = 1e-4            # give up (raise) beyond this
MAXIMIN_CANDIDATES = 100     # Latin hypercubes compared by the maximin criterion


@dataclass
class BORunResult(RunResult):
    """RunResult plus BO diagnostics (D13, D18, D22 item 5)."""

    # checkpoints, jitter events, overhead time, warning counts
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


# ----------------------------------------------------------------------
# Initial design (shared by both BO variants)
# ----------------------------------------------------------------------

def default_n_init(d: int) -> int:
    """Small initial design d + 4 (number of GP hyperparameters + 2; L&P Sec. 3.2)."""
    return d + 4


def initial_design(p_seed: int, d: int, n_init: int) -> np.ndarray:
    """Maximin Latin hypercube in the unit box, seeded from the PROBLEM.

    Seeding from the problem (not the method) gives both BO variants the
    same design on each instance, so their comparison is paired, as the
    CMA-ES variants share x0 (D9). Among MAXIMIN_CANDIDATES Latin
    hypercubes (McKay, Beckman & Conover, 1979), the one with the largest
    minimum pairwise distance is used (maximin criterion, L&P Sec. 3.2).
    """
    rng = np.random.default_rng(derive_seed(p_seed, 998))
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    best, best_dist = None, -np.inf
    for _ in range(MAXIMIN_CANDIDATES):
        U = sampler.random(n_init)
        dist = pdist(U).min() if n_init > 1 else 0.0
        if dist > best_dist:
            best, best_dist = U, dist
    return best


# ----------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------

def length_scale_bounds(d: int) -> tuple[float, float]:
    """[(U-L) sqrt(d)/100, (U-L) sqrt(d)] with U-L = 1 on the unit box (L&P Sec. 4.1)."""
    return (np.sqrt(d) / 100.0, np.sqrt(d))


def make_kernel(kind: str, d: int):
    """Constant * Matern(nu=2.5), ARD (vector lengthscale) or isotropic (scalar)."""
    if kind == "ard":
        length_scale = np.ones(d)
    elif kind == "isotropic":
        length_scale = 1.0
    else:
        raise ValueError(f"unknown kernel kind: {kind!r}")
    return ConstantKernel(1.0, (1e-5, 1e5)) * Matern(
        length_scale=length_scale, length_scale_bounds=length_scale_bounds(d), nu=2.5
    )


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    """Expected improvement for minimisation (Jones et al., 1998).

    EI = (y_best - mu) * Phi(z) + sigma * phi(z),  z = (y_best - mu) / sigma,
    and EI = max(y_best - mu, 0) where sigma = 0.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = y_best - mu
    ei = np.maximum(improvement, 0.0)
    pos = sigma > 1e-12
    z = improvement[pos] / sigma[pos]
    ei[pos] = improvement[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _maximise_ei(gp, y_best, d, rng, n_candidates, n_repeats, local_search, X_seen):
    """L&P's EI optimiser (Sec. 4.2).

    Repeated n_repeats times: evaluate EI on a space-filling (Latin hypercube)
    design of n_candidates points, then, if local_search, run L-BFGS-B from
    the best of them. The best point over all repeats is returned.
    """
    def neg_ei(u):
        m, s = gp.predict(u.reshape(1, -1), return_std=True)
        return -float(expected_improvement(m, s, y_best)[0])

    best_u, best_val = None, np.inf
    all_cand, all_ei = [], []
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    for _ in range(n_repeats):
        cand = sampler.random(n_candidates)
        mu, sd = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sd, y_best)
        all_cand.append(cand)
        all_ei.append(ei)
        u, val = cand[np.argmax(ei)], -float(ei.max())
        if local_search:
            res = minimize(neg_ei, u, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            if res.fun < val:
                u, val = np.clip(res.x, 0.0, 1.0), float(res.fun)
        if val < best_val:
            best_u, best_val = u, val

    cand = np.vstack(all_cand)
    ei = np.concatenate(all_ei)

    # EI flat at zero (a plateau): pick a random point (L&P Sec. 4.2).
    if not -best_val > 0.0:
        return rng.uniform(0.0, 1.0, size=d), "random_point_ei_zero"

    # Never re-evaluate an existing point: it adds no information in a
    # noise-free problem and makes the GP covariance singular (L&P Sec. 4.2
    # use a proximity check for the same reason).
    if np.min(np.linalg.norm(X_seen - best_u, axis=1)) < 1e-8:
        for idx in np.argsort(-ei):
            if np.min(np.linalg.norm(X_seen - cand[idx], axis=1)) >= 1e-8:
                return cand[idx], "replaced_near_duplicate"
    return best_u, "ok"


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def run_bo(
    f: Callable[[np.ndarray], float],
    method: str,
    d: int,
    lower: float,
    upper: float,
    budget: int,
    checkpoints: list[int],
    p_seed: int,
    n_init: int | None = None,
    jitter: float = BASE_JITTER,
    n_restarts_gp: int | None = None,
    n_candidates: int | None = None,
    n_acq_repeats: int | None = None,
    acq_setting: str = "standard",
    f_opt: float = 0.0,
    log_points: bool = True,
) -> BORunResult:
    """Run BO (method 'bo_ard' or 'bo_isotropic') to an exact budget.

    Defaults follow L&P: n_init = d + 4; 2d likelihood starts
    (n_restarts_gp = 2d - 1 random restarts after the initial one);
    n_candidates = min(2000, 500d); n_acq_repeats = min(10, d).
    """
    if method not in BO_METHODS:
        raise ValueError(f"method must be one of {sorted(BO_METHODS)}")
    if acq_setting not in ACQ_SETTINGS:
        raise ValueError(f"acq_setting must be one of {sorted(ACQ_SETTINGS)}")
    n_init = default_n_init(d) if n_init is None else n_init
    n_restarts_gp = 2 * d - 1 if n_restarts_gp is None else n_restarts_gp
    n_candidates = min(2000, 500 * d) if n_candidates is None else n_candidates
    n_acq_repeats = min(10, d) if n_acq_repeats is None else n_acq_repeats
    if not 1 <= n_init <= budget:
        raise ValueError("n_init must be between 1 and the budget")
    if jitter <= 0:
        raise ValueError("jitter must be positive (a fixed numerical stabiliser, not noise; D21)")
    if n_candidates < 1 or n_acq_repeats < 1:
        raise ValueError("n_candidates and n_acq_repeats must be at least 1")

    kind = BO_METHODS[method]
    local_search = ACQ_SETTINGS[acq_setting]["local_search"]
    m_seed = derive_seed(p_seed, _BO_SEED_KEYS[kind])
    rng = np.random.default_rng(m_seed)
    obj = CountingObjective(f, budget, checkpoints, f_opt, log_points)
    width = upper - lower
    to_x = lambda u: lower + width * np.asarray(u)          # unit box -> search box

    t0 = time.perf_counter()
    overhead_s = 0.0
    n_fit_warnings = 0
    acq_events: dict[str, int] = {}
    current_jitter = jitter
    diagnostics: dict[str, Any] = {
        "kernel": kind, "n_init": n_init, "base_jitter": jitter,
        "n_restarts_gp": n_restarts_gp, "n_candidates": n_candidates,
        "n_acq_repeats": n_acq_repeats, "acq_setting": acq_setting,
        "length_scale_bounds": list(length_scale_bounds(d)),
        "jitter_events": [], "checkpoints": {},
    }

    # Initial design: shared maximin Latin hypercube, counted in the budget.
    U = initial_design(p_seed, d, n_init)
    y = np.array([obj(to_x(u)) for u in U])

    def fit_gp(U, y):
        """Fit with the current jitter; on failure, escalate it and keep it (L&P Sec. 4.1)."""
        nonlocal n_fit_warnings, current_jitter
        while True:
            model = GaussianProcessRegressor(
                kernel=make_kernel(kind, d),
                alpha=current_jitter,
                normalize_y=True,
                n_restarts_optimizer=n_restarts_gp,
                random_state=int(rng.integers(2**31 - 1)),
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(U, y)
                n_fit_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                if np.all(np.isfinite(model.kernel_.theta)) and np.isfinite(model.log_marginal_likelihood_value_):
                    return model
                reason = "non-finite hyperparameters or likelihood"
            except (np.linalg.LinAlgError, ValueError) as exc:
                reason = f"{type(exc).__name__}"
            new_jitter = current_jitter * JITTER_ESCALATION
            if new_jitter > MAX_JITTER:
                raise RuntimeError(f"GP fit failed even with jitter {current_jitter:g}: {reason}")
            diagnostics["jitter_events"].append(
                {"n_points": len(y), "old": current_jitter, "new": new_jitter, "reason": reason})
            current_jitter = new_jitter

    # Checkpoints inside the initial design have no fitted model yet.
    for c in checkpoints:
        if c < n_init:
            diagnostics["checkpoints"][c] = None

    while obj.remaining > 0:
        t_model = time.perf_counter()
        gp = fit_gp(U, y)
        # This model uses every observation made so far, so it is the model
        # "at" checkpoint len(y): record it there (not one evaluation late).
        if len(y) in checkpoints:
            diagnostics["checkpoints"][len(y)] = _gp_state(gp, len(y))

        # With normalize_y=True, predict() returns mean and std in original
        # units, so the incumbent is simply the best observed value.
        u_next, event = _maximise_ei(gp, float(y.min()), d, rng, n_candidates,
                                     n_acq_repeats, local_search, U)
        if event != "ok":
            acq_events[event] = acq_events.get(event, 0) + 1
        overhead_s += time.perf_counter() - t_model

        y_next = obj(to_x(u_next))
        U = np.vstack([U, u_next])
        y = np.append(y, y_next)

    # The final checkpoint is reached by the last evaluation, after the last
    # fit, so fit one diagnostic model on all observations. It is not used to
    # choose any point and is excluded from the overhead timing (D13).
    if obj.n_evals in checkpoints and obj.n_evals not in diagnostics["checkpoints"]:
        diagnostics["checkpoints"][obj.n_evals] = _gp_state(fit_gp(U, y), len(y))
    diagnostics["overhead_s"] = overhead_s
    diagnostics["n_fit_convergence_warnings"] = n_fit_warnings
    diagnostics["final_jitter"] = current_jitter
    diagnostics["acquisition_events"] = acq_events

    base = _finish(obj, method, p_seed, m_seed, None, t0, {})
    return BORunResult(**{fl.name: getattr(base, fl.name) for fl in fields(RunResult)},
                       diagnostics=diagnostics)


def _gp_state(gp, n_points: int) -> dict[str, Any]:
    """Fitted hyperparameters, in unit-box coordinates (D18 lengthscale diagnostic)."""
    k = gp.kernel_
    return {
        "n_points": int(n_points),
        "length_scale": np.atleast_1d(k.k2.length_scale).tolist(),
        "signal_variance": float(k.k1.constant_value),
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
    }