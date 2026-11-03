"""Bayesian optimisation under an exact evaluation budget (decision log D21, D22).

Two variants that differ only in the kernel's lengthscale structure:

    kernel="ard"        Matern 5/2 with one lengthscale per input axis
                        (Snoek et al., 2012, Sec. 3.1; Rasmussen & Williams,
                        2006, p. 106). Axis-aligned model.
    kernel="isotropic"  Matern 5/2 with a single lengthscale. Depends only on
                        distance, so invariant to rotations (Rasmussen &
                        Williams, 2006, Sec. 4.1, p. 80).

Everything else is shared: inputs scaled to the unit box, outputs
standardised, GP hyperparameters fitted by maximising the log marginal
likelihood with random restarts (type-II ML; Rasmussen & Williams, 2006,
Sec. 5.4.1 and p. 115), a fixed small jitter, a Latin hypercube initial
design that counts towards the budget, and expected improvement (Jones et
al., 1998) maximised by multi-start L-BFGS-B. Every objective call goes
through runner.CountingObjective, so the budget is exact and the reported
solution is the best *evaluated* point (D6, D11).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from runner import CountingObjective, RunResult, _finish, derive_seed

__all__ = [
    "BO_METHODS",
    "initial_design",
    "BORunResult",
    "expected_improvement",
    "make_kernel",
    "run_bo",
]

BO_METHODS = {"bo_ard": "ard", "bo_isotropic": "isotropic"}
_BO_SEED_KEYS = {"ard": 2001, "isotropic": 2002}   # distinct from runner.METHOD_IDS


@dataclass
class BORunResult(RunResult):
    """RunResult plus BO diagnostics (D13, D18, D22 item 5)."""

    # checkpoint -> {"length_scale": [...], "signal_variance": float, ...}
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


# ----------------------------------------------------------------------
# Initial design (shared by both BO variants)
# ----------------------------------------------------------------------

def initial_design(p_seed: int, d: int, n_init: int) -> np.ndarray:
    """Latin hypercube in the unit box, seeded from the PROBLEM, not the method.

    Both BO variants therefore start from the same design on each instance,
    so their comparison is paired, as the CMA-ES variants share x0 (D9).
    Latin hypercube sampling: McKay, Beckman & Conover (1979).
    """
    rng = np.random.default_rng(derive_seed(p_seed, 998))
    return qmc.LatinHypercube(d=d, seed=rng).random(n_init)


# ----------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------

def make_kernel(kind: str, d: int, length_scale_bounds=(1e-5, 1e5)):
    """Constant * Matern(nu=2.5), ARD (vector lengthscale) or isotropic (scalar)."""
    if kind == "ard":
        length_scale = np.ones(d)
    elif kind == "isotropic":
        length_scale = 1.0
    else:
        raise ValueError(f"unknown kernel kind: {kind!r}")
    return ConstantKernel(1.0, (1e-5, 1e5)) * Matern(
        length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5
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


def _maximise_ei(gp, y_best, d, rng, n_candidates, n_starts, X_seen):
    """Multi-start L-BFGS-B on the unit box, started from the best random candidates."""
    cand = rng.uniform(0.0, 1.0, size=(n_candidates, d))
    mu, sd = gp.predict(cand, return_std=True)
    ei = expected_improvement(mu, sd, y_best)
    starts = cand[np.argsort(-ei)[:n_starts]]

    def neg_ei(u):
        m, s = gp.predict(u.reshape(1, -1), return_std=True)
        return -float(expected_improvement(m, s, y_best)[0])

    best_u, best_val = starts[0], neg_ei(starts[0])
    for s0 in starts:
        res = minimize(neg_ei, s0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if res.fun < best_val:
            best_u, best_val = np.clip(res.x, 0.0, 1.0), res.fun

    # Avoid re-evaluating an existing point (it adds no information and can
    # make the GP ill-conditioned); fall back to the best fresh candidate.
    if np.min(np.linalg.norm(X_seen - best_u, axis=1)) < 1e-8:
        order = np.argsort(-ei)
        for idx in order:
            if np.min(np.linalg.norm(X_seen - cand[idx], axis=1)) >= 1e-8:
                return cand[idx]
    return best_u


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
    n_init: int,
    jitter: float = 1e-6,
    n_restarts_gp: int = 5,
    n_candidates: int = 2000,
    n_starts_acq: int = 5,
    f_opt: float = 0.0,
    log_points: bool = True,
) -> BORunResult:
    """Run BO (method 'bo_ard' or 'bo_isotropic') to an exact budget."""
    if method not in BO_METHODS:
        raise ValueError(f"method must be one of {sorted(BO_METHODS)}")
    if not 1 <= n_init <= budget:
        raise ValueError("n_init must be between 1 and the budget")
    if jitter <= 0:
        raise ValueError("jitter must be positive (a fixed, non-learned noise term; D21)")

    kind = BO_METHODS[method]
    m_seed = derive_seed(p_seed, _BO_SEED_KEYS[kind])
    rng = np.random.default_rng(m_seed)
    obj = CountingObjective(f, budget, checkpoints, f_opt, log_points)
    width = upper - lower
    to_x = lambda u: lower + width * np.asarray(u)          # unit box -> search box

    t0 = time.perf_counter()
    overhead_s = 0.0
    n_fit_warnings = 0
    diagnostics: dict[str, Any] = {"kernel": kind, "n_init": n_init, "jitter": jitter,
                                   "n_restarts_gp": n_restarts_gp, "checkpoints": {}}

    # Initial design: shared Latin hypercube in the unit box, counted in the budget.
    U = initial_design(p_seed, d, n_init)
    y = np.array([obj(to_x(u)) for u in U])

    gp = None
    while obj.remaining > 0:
        t_model = time.perf_counter()
        gp = GaussianProcessRegressor(
            kernel=make_kernel(kind, d),
            alpha=jitter,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_gp,
            random_state=int(rng.integers(2**31 - 1)),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(U, y)
        n_fit_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)

        # With normalize_y=True, predict() returns mean and std in original
        # units, so the incumbent is simply the best observed value.
        u_next = _maximise_ei(gp, float(y.min()), d, rng, n_candidates, n_starts_acq, U)
        overhead_s += time.perf_counter() - t_model

        y_next = obj(to_x(u_next))
        U = np.vstack([U, u_next])
        y = np.append(y, y_next)

        if obj.n_evals in checkpoints:
            diagnostics["checkpoints"][obj.n_evals] = _gp_state(gp)

    # Checkpoints reached during the initial design have no fitted model yet.
    if gp is not None:
        for c in checkpoints:
            diagnostics["checkpoints"].setdefault(c, _gp_state(gp))
    diagnostics["overhead_s"] = overhead_s
    diagnostics["n_fit_convergence_warnings"] = n_fit_warnings

    base = _finish(obj, method, p_seed, m_seed, None, t0, {})
    return BORunResult(**{fl.name: getattr(base, fl.name) for fl in fields(RunResult)},
                       diagnostics=diagnostics)


def _gp_state(gp) -> dict[str, Any]:
    """Fitted hyperparameters, in unit-box coordinates (D18 lengthscale diagnostic)."""
    k = gp.kernel_
    return {
        "length_scale": np.atleast_1d(k.k2.length_scale).tolist(),
        "signal_variance": float(k.k1.constant_value),
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
    }