"""Run one optimiser on one problem instance under an exact evaluation budget.

Decision log references:
    D6   every method reports its best *evaluated* point
    D8   CMA-ES variants share every setting except their covariance option
    D9  x0 uniform in the box, independent of x_opt, shared by CMA variants
    D10  early termination is flagged, never presented as a full-budget run
    D11  one counting wrapper enforces the budget; the final CMA-ES
         generation is truncated to the first k candidates
    D12  problem seeds depend on condition and replicate only
    D13  per-evaluation logging; covariance snapshots for diagnostics
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "BudgetExhausted",
    "stop_category",
    "CountingObjective",
    "RunResult",
    "derive_seed",
    "problem_seed",
    "method_seed",
    "initial_point",
    "effective_covariance",
    "kappa_eff",
    "cma_options_from_config",
    "run_cma",
    "run_uniform",
]

METHOD_IDS = {
    "cma_es_full": 1,
    "cma_es_diagonal": 2,
    "isotropic_es": 3,
    "uniform_random_search": 4,
    "bayesian_optimisation": 5,
}

# pycma termination criteria disabled so runs go to budget (D10).
# Safety stops that signal genuine failure (tolconditioncov, tolupsigma,
# tolfacupx) are kept; if one fires, the run is flagged incomplete.
RUN_TO_BUDGET_OPTIONS = {
    "tolfun": 0,
    "tolfunrel": 0,
    "tolfunhist": 0,
    "tolx": 0,
    "tolstagnation": 10**12,
    "tolflatfitness": 10**12,
    "tolxstagnation": False,              # documented way to turn it off (pycma option docs)
    "maxfevals": np.inf,
    "maxiter": np.inf,
}


# pycma stops that cannot be disabled and fire only when steps no longer
# change x in float64 (i.e. the run has converged to machine precision).
NUMERICAL_PRECISION_STOPS = frozenset({"noeffectaxis", "noeffectcoord"})


def stop_category(evals_used: int, budget: int, early: dict | None) -> str:
    """'budget', 'numerical_precision' or 'other' (decision log D10)."""
    if evals_used == budget:
        return "budget"
    if early and set(early) <= NUMERICAL_PRECISION_STOPS:
        return "numerical_precision"
    return "other"


# ----------------------------------------------------------------------
# Seeds (D12)
# ----------------------------------------------------------------------

def derive_seed(*keys: int) -> int:
    """Deterministic positive 31-bit seed from integer keys."""
    state = np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0]
    return int(state % (2**31 - 2)) + 1


def problem_seed(base_seed: int, stage_offset: int, d: int, kappa: float, replicate: int) -> int:
    """Seed for x_opt and R. Orientation is deliberately NOT a key, so the
    aligned and rotated instances of a replicate share x_opt (D12)."""
    return derive_seed(base_seed, stage_offset, d, int(round(kappa)), replicate)


def method_seed(p_seed: int, method: str) -> int:
    """Seed for a method's internal randomness, separate from the problem."""
    return derive_seed(p_seed, 1000 + METHOD_IDS[method])


def initial_point(p_seed: int, d: int, lower: float, upper: float) -> np.ndarray:
    """CMA-ES start: uniform in the box, from its own stream (D9).

    Independent of x_opt (different stream), identical for all CMA variants
    and for the aligned and rotated instances of a replicate.
    """
    rng = np.random.default_rng(derive_seed(p_seed, 999))
    return rng.uniform(lower, upper, size=d)


# ----------------------------------------------------------------------
# Counting wrapper (D11)
# ----------------------------------------------------------------------

class BudgetExhausted(RuntimeError):
    """Raised if a method tries to evaluate beyond its budget."""


class CountingObjective:
    """The only path to the objective. Counts, caps, logs and checkpoints."""

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        budget: int,
        checkpoints: list[int],
        f_opt: float = 0.0,
        log_points: bool = True,
    ) -> None:
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
            raise TypeError("budget must be an integer")
        if budget <= 0:
            raise ValueError("budget must be positive")
        if not checkpoints or checkpoints[-1] != budget:
            raise ValueError("checkpoints must end at the budget")
        self._f = f
        self.budget = int(budget)
        self.checkpoints = list(checkpoints)
        self.f_opt = float(f_opt)
        self.log_points = log_points
        self.n_evals = 0
        self.n_repaired = 0
        self.best_f = np.inf
        self.best_x: np.ndarray | None = None
        self.checkpoint_best: dict[int, float] = {}
        self.log: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.n_evals

    def __call__(self, x, repaired: bool = False) -> float:
        if self.n_evals >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} evaluations exhausted")
        x = np.asarray(x, dtype=float).copy()
        value = float(self._f(x.copy()))
        if not np.isfinite(value):
            # Fail loudly: a NaN/inf would silently corrupt best-so-far.
            raise ValueError(
                f"objective returned a non-finite value at evaluation {self.n_evals + 1}"
            )
        self.n_evals += 1
        self.n_repaired += int(repaired)
        if value < self.best_f:
            self.best_f, self.best_x = value, x
        if self.n_evals in self.checkpoints:
            self.checkpoint_best[self.n_evals] = self.best_f
        row = {
            "eval": self.n_evals,
            "f": value,
            "best_so_far": self.best_f,
            "repaired": bool(repaired),
        }
        if self.log_points:
            row["x"] = x.tolist()
        self.log.append(row)
        return value

    def fill_unreached_checkpoints(self) -> None:
        """For an incomplete run, carry the best value to later checkpoints.

        The run stays flagged incomplete (budget_completed=False) so these
        values are never mistaken for full-budget results (D10).
        """
        for c in self.checkpoints:
            self.checkpoint_best.setdefault(c, self.best_f)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class RunResult:
    method: str
    problem_seed: int
    method_seed: int
    budget: int
    evals_used: int
    budget_completed: bool
    best_f: float
    best_x: list[float]
    regret: float
    checkpoint_regret: dict[int, float]
    n_repaired: int
    early_termination: dict[str, Any] | None
    stop_category: str
    wall_time_s: float
    eval_log: list[dict[str, Any]] = field(repr=False)
    # checkpoint -> {"internal": matrix, "evaluated": matrix}
    covariance_snapshots: dict[int, dict[str, list[list[float]]]] = field(default_factory=dict, repr=False)

    def summary(self) -> dict[str, Any]:
        """One row for results/processed (no per-evaluation log)."""
        row = {
            "method": self.method,
            "problem_seed": self.problem_seed,
            "method_seed": self.method_seed,
            "budget": self.budget,
            "evals_used": self.evals_used,
            "budget_completed": self.budget_completed,
            "best_f": self.best_f,
            "regret": self.regret,
            "n_repaired": self.n_repaired,
            "early_termination": None if self.early_termination is None else str(self.early_termination),
            "stop_category": self.stop_category,
            "wall_time_s": self.wall_time_s,
        }
        for c, r in self.checkpoint_regret.items():
            row[f"regret_at_{c}"] = r
        return row


def _finish(obj, method, p_seed, m_seed, early, t0, snaps) -> RunResult:
    obj.fill_unreached_checkpoints()
    return RunResult(
        method=method,
        problem_seed=p_seed,
        method_seed=m_seed,
        budget=obj.budget,
        evals_used=obj.n_evals,
        budget_completed=obj.n_evals == obj.budget,
        best_f=obj.best_f,
        best_x=obj.best_x.tolist(),
        regret=obj.best_f - obj.f_opt,
        checkpoint_regret={c: v - obj.f_opt for c, v in obj.checkpoint_best.items()},
        n_repaired=obj.n_repaired,
        early_termination=early,
        stop_category=stop_category(obj.n_evals, obj.budget, early),
        wall_time_s=time.perf_counter() - t0,
        eval_log=obj.log,
        covariance_snapshots=snaps,
    )


# ----------------------------------------------------------------------
# Covariance diagnostics (D18)
# ----------------------------------------------------------------------

def effective_covariance(es, evaluated: bool = True) -> np.ndarray:
    """Sampling covariance of a pycma strategy in the evaluated coordinates.

    Two pycma details matter (verified against the empirical covariance of
    es.ask() samples in pycma 4.5):
      * the diagonal variant is an identity sampler plus a per-coordinate
        scaling vector D, so the internal covariance is sigma^2 * D C D;
      * with `bounds`, C lives in pycma's genotype space. The bound transform
        is coordinate-wise and can mirror a coordinate, so the covariance of
        the evaluated points is J (sigma^2 D C D) J, with J the diagonal
        Jacobian of the transform at the mean.

    evaluated=False returns the internal covariance (used to verify that the
    diagonal / isotropic variants behave as specified, D7-D8); evaluated=True
    returns the covariance of the points actually evaluated (used for
    kappa_eff, D18). Near a bound the two differ even for isotropic ES.
    """
    d = es.N
    C = np.asarray(es.C, dtype=float)
    if C.ndim == 0:
        C = np.eye(d) * float(C)
    elif C.ndim == 1:
        C = np.diag(C)
    scaling = np.asarray(getattr(es.sigma_vec, "scaling", 1.0), dtype=float) * np.ones(d)
    Sigma = es.sigma**2 * (scaling[:, None] * C * scaling[None, :])
    if not evaluated:
        return Sigma  # internal (genotype) covariance: verifies the variant itself

    bh = getattr(es, "boundary_handler", None)
    if bh is not None and hasattr(bh, "transform") and getattr(bh, "has_bounds", lambda: False)():
        m = np.asarray(es.mean, dtype=float)
        h = 1e-6 * np.maximum(1.0, np.abs(m))
        J = np.empty(d)
        for i in range(d):
            e = np.zeros(d)
            e[i] = h[i]
            J[i] = (np.asarray(bh.transform(m + e))[i] - np.asarray(bh.transform(m - e))[i]) / (2 * h[i])
        Sigma = J[:, None] * Sigma * J[None, :]
    return Sigma


def kappa_eff(Sigma: np.ndarray, H: np.ndarray) -> float:
    """cond(Sigma^1/2 H Sigma^1/2); 1 means the search distribution matches
    the inverse Hessian up to scale."""
    w, V = np.linalg.eigh(Sigma)
    root = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    eig = np.linalg.eigvalsh(root @ H @ root)
    return float(eig[-1] / eig[0])


# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------

def cma_options_from_config(method_cfg: dict[str, Any], d: int, lower: float, upper: float) -> tuple[dict, float]:
    """Translate a resolved CMA-ES method block into pycma options and sigma0."""
    opts: dict[str, Any] = dict(RUN_TO_BUDGET_OPTIONS)
    opts["verbose"] = -9
    if method_cfg.get("boundary_handling") == "pycma_bound_transform":
        opts["bounds"] = [lower, upper]
    elif method_cfg.get("boundary_handling") is not None:
        raise ValueError(f"unsupported boundary_handling: {method_cfg['boundary_handling']}")
    popsize = method_cfg.get("population_size")
    if popsize not in (None, "pycma_default"):
        opts["popsize"] = int(popsize)
    opts.update(method_cfg.get("options") or {})
    sigma0 = float(method_cfg["initial_step_size_fraction_of_box_width"]) * (upper - lower)
    return opts, sigma0


def run_cma(
    f: Callable[[np.ndarray], float],
    method: str,
    options: dict[str, Any],
    sigma0: float,
    x0: np.ndarray,
    budget: int,
    checkpoints: list[int],
    p_seed: int,
    f_opt: float = 0.0,
    log_points: bool = True,
) -> RunResult:
    """Run one pycma variant to an exact budget (D10, D11)."""
    import cma

    m_seed = method_seed(p_seed, method)
    obj = CountingObjective(f, budget, checkpoints, f_opt, log_points)
    opts = dict(options, seed=m_seed)
    snaps: dict[int, dict[str, list[list[float]]]] = {}

    def snapshot(es):
        # CMA-ES updates once per generation, so a snapshot is taken at the
        # first generation boundary at or after the nominal checkpoint; the
        # true evaluation count is recorded with it.
        return {
            "evals_at_snapshot": obj.n_evals,
            "internal": effective_covariance(es, evaluated=False).tolist(),
            "evaluated": effective_covariance(es, evaluated=True).tolist(),
        }

    early = None
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = cma.CMAEvolutionStrategy(np.asarray(x0, float), sigma0, opts)
        bh = es.boundary_handler if "bounds" in opts else None
        next_snap = 0

        while obj.remaining > 0:
            stop = es.stop()
            if stop:
                early = dict(stop)
                break
            X = es.ask()
            k = min(len(X), obj.remaining)
            values = []
            for x in X[:k]:
                repaired = bh is not None and not np.allclose(bh.inverse(x), x)
                values.append(obj(x, repaired=repaired))
            if k < len(X):
                break  # truncated final generation: no update (D11)
            es.tell(X, values)
            while next_snap < len(checkpoints) and obj.n_evals >= checkpoints[next_snap]:
                snaps[checkpoints[next_snap]] = snapshot(es)
                next_snap += 1
        # Covariance at the end of the run, for any checkpoint not yet stored.
        for c in checkpoints[next_snap:]:
            snaps.setdefault(c, snapshot(es))

    return _finish(obj, method, p_seed, m_seed, early, t0, snaps)


def run_uniform(
    f: Callable[[np.ndarray], float],
    d: int,
    lower: float,
    upper: float,
    budget: int,
    checkpoints: list[int],
    p_seed: int,
    f_opt: float = 0.0,
    log_points: bool = True,
) -> RunResult:
    """Independent uniform draws over the box (null baseline)."""
    method = "uniform_random_search"
    m_seed = method_seed(p_seed, method)
    rng = np.random.default_rng(m_seed)
    obj = CountingObjective(f, budget, checkpoints, f_opt, log_points)
    t0 = time.perf_counter()
    while obj.remaining > 0:
        obj(rng.uniform(lower, upper, size=d))
    return _finish(obj, method, p_seed, m_seed, None, t0, {})