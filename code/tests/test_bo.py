"""Tests for code/bo.py (decision log D6, D11, D21, D22; L&P = Le Riche & Picheny, 2021)."""

import numpy as np
import pytest
from scipy.stats import norm

pytest.importorskip("sklearn")

import bo as bo_module
from bo import (
    BO_METHODS,
    default_n_init,
    expected_improvement,
    initial_design,
    length_scale_bounds,
    make_kernel,
    run_bo,
)
from objectives import make_ellipsoid_instance, sphere
from runner import run_uniform

BUDGET, N_INIT, CHECKPOINTS = 23, 5, [10, 23]
FAST = dict(n_restarts_gp=1, n_candidates=300, n_acq_repeats=2)


def _run(method, f=None, seed=7, **kw):
    f = f or make_ellipsoid_instance(4, 1e2, "rotated", seed)
    return run_bo(f, method, 4, -5, 5, BUDGET, CHECKPOINTS, seed, n_init=N_INIT, **{**FAST, **kw})


# ------------------------------------------------------ expected improvement

def test_ei_without_uncertainty_is_plain_improvement():
    ei = expected_improvement(np.array([1.0, 3.0]), np.array([0.0, 0.0]), y_best=2.0)
    assert np.allclose(ei, [1.0, 0.0])


def test_ei_at_incumbent_equals_sigma_phi0():
    ei = expected_improvement(np.array([2.0]), np.array([0.5]), y_best=2.0)
    assert ei[0] == pytest.approx(0.5 * norm.pdf(0.0))


def test_ei_is_nonnegative_and_grows_with_uncertainty():
    mu = np.full(5, 3.0)
    ei = expected_improvement(mu, np.linspace(0.1, 2.0, 5), y_best=2.0)
    assert np.all(ei >= 0) and np.all(np.diff(ei) > 0)


# -------------------------------------------------------------- kernels

def test_ard_kernel_has_one_lengthscale_per_axis():
    assert np.shape(make_kernel("ard", 4).k2.length_scale) == (4,)


def test_isotropic_kernel_has_a_single_lengthscale():
    assert np.ndim(make_kernel("isotropic", 4).k2.length_scale) == 0


def test_kernels_are_matern_five_halves():
    for kind in ("ard", "isotropic"):
        assert make_kernel(kind, 4).k2.nu == 2.5


def test_unknown_kernel_rejected():
    with pytest.raises(ValueError):
        make_kernel("rbf", 4)


# ------------------------------------------------------------- run_bo

@pytest.mark.parametrize("method", sorted(BO_METHODS))
def test_exact_budget_and_best_evaluated_point(method):
    inst = make_ellipsoid_instance(4, 1e2, "rotated", 7)
    res = _run(method, inst)
    assert res.evals_used == BUDGET and res.budget_completed
    assert len(res.eval_log) == BUDGET
    assert res.best_f == min(r["f"] for r in res.eval_log)
    assert inst(np.array(res.best_x)) == pytest.approx(res.best_f)
    assert set(res.checkpoint_regret) == set(CHECKPOINTS)


@pytest.mark.parametrize("method", sorted(BO_METHODS))
def test_points_stay_in_box(method):
    X = np.array([r["x"] for r in _run(method).eval_log])
    assert np.all(X >= -5) and np.all(X <= 5)


def test_initial_design_is_latin_hypercube():
    X = np.array([r["x"] for r in _run("bo_ard").eval_log[:N_INIT]])
    strata = np.floor((X + 5) / 10 * N_INIT).astype(int)
    for j in range(4):                      # one point in each stratum per axis
        assert sorted(strata[:, j]) == list(range(N_INIT))


def test_variants_share_initial_design():
    """ARD and isotropic BO start from the same points on an instance (paired)."""
    a = [r["x"] for r in _run("bo_ard").eval_log[:N_INIT]]
    b = [r["x"] for r in _run("bo_isotropic").eval_log[:N_INIT]]
    assert a == b


def test_initial_design_differs_across_instances():
    assert not np.allclose(initial_design(1, 4, N_INIT), initial_design(2, 4, N_INIT))


def test_reproducible():
    a, b = _run("bo_ard"), _run("bo_ard")
    assert [r["f"] for r in a.eval_log] == [r["f"] for r in b.eval_log]


def test_variants_use_different_seeds():
    assert _run("bo_ard").method_seed != _run("bo_isotropic").method_seed


@pytest.mark.parametrize("method, n_ls", [("bo_ard", 4), ("bo_isotropic", 1)])
def test_lengthscale_diagnostics_recorded(method, n_ls):
    res = _run(method)
    diag = res.diagnostics
    assert set(diag["checkpoints"]) == set(CHECKPOINTS)
    assert len(diag["checkpoints"][BUDGET]["length_scale"]) == n_ls
    assert diag["overhead_s"] > 0
    assert diag["kernel"] == BO_METHODS[method]


def test_bo_beats_uniform_search_on_sphere():
    f = lambda x: sphere(x, np.full(4, 1.0))
    bo = run_bo(f, "bo_ard", 4, -5, 5, BUDGET, CHECKPOINTS, 11, n_init=N_INIT, **FAST)
    rs = run_uniform(f, 4, -5, 5, BUDGET, CHECKPOINTS, 11)
    assert bo.regret < rs.regret


@pytest.mark.parametrize(
    "kwargs",
    [dict(method="bo_rbf"), dict(n_init=0), dict(n_init=BUDGET + 1), dict(jitter=0.0),
     dict(acq_setting="greedy"), dict(n_acq_repeats=0)],
)
def test_invalid_arguments_rejected(kwargs):
    args = dict(f=sphere, method="bo_ard", d=4, lower=-5, upper=5, budget=BUDGET,
                checkpoints=CHECKPOINTS, p_seed=1, n_init=N_INIT)
    args.update(kwargs)
    with pytest.raises(ValueError):
        run_bo(**args)


def test_lengthscale_snapshot_uses_all_observations_at_checkpoint():
    """The model logged at checkpoint c was fitted on exactly c observations."""
    res = _run("bo_ard")
    for c, state in res.diagnostics["checkpoints"].items():
        assert state["n_points"] == c


def test_checkpoints_inside_initial_design_have_no_model():
    res = run_bo(sphere, "bo_ard", 4, -5, 5, BUDGET, [3, BUDGET], 7, n_init=N_INIT, **FAST)
    assert res.diagnostics["checkpoints"][3] is None


# ------------------------------------------------- Le Riche & Picheny settings

def test_default_initial_design_is_d_plus_4():
    assert default_n_init(4) == 8                      # L&P Sec. 3.2: d + 4


def test_initial_design_is_maximin_among_candidates():
    """The chosen Latin hypercube has the largest minimum distance (L&P Sec. 3.2)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import qmc
    from runner import derive_seed
    U = initial_design(5, 4, 8)
    sampler = qmc.LatinHypercube(d=4, seed=np.random.default_rng(derive_seed(5, 998)))
    best = max(pdist(sampler.random(8)).min() for _ in range(bo_module.MAXIMIN_CANDIDATES))
    assert pdist(U).min() == pytest.approx(best)


def test_length_scale_bounds_follow_le_riche_picheny():
    lo, hi = length_scale_bounds(4)
    assert (lo, hi) == pytest.approx((0.02, 2.0))     # sqrt(4)/100, sqrt(4)
    for kind in ("ard", "isotropic"):
        assert np.allclose(make_kernel(kind, 4).k2.length_scale_bounds, (lo, hi))


@pytest.mark.parametrize("method", sorted(BO_METHODS))
def test_fitted_lengthscales_stay_within_bounds(method):
    lo, hi = length_scale_bounds(4)
    for state in _run(method).diagnostics["checkpoints"].values():
        assert all(lo * (1 - 1e-9) <= l <= hi * (1 + 1e-9) for l in state["length_scale"])


def test_defaults_follow_le_riche_picheny():
    inst = make_ellipsoid_instance(4, 1e2, "rotated", 7)
    res = run_bo(inst, "bo_ard", 4, -5, 5, 12, [12], 7, n_candidates=200)
    diag = res.diagnostics
    assert diag["n_init"] == 8 and diag["n_restarts_gp"] == 7 and diag["n_acq_repeats"] == 4


def test_random_acquisition_setting_runs():
    res = _run("bo_ard", acq_setting="random")
    assert res.evals_used == BUDGET and res.diagnostics["acq_setting"] == "random"


def test_failed_fit_escalates_jitter_and_keeps_it(monkeypatch):
    """A Cholesky failure triggers a larger jitter, recorded and kept (L&P Sec. 4.1)."""
    real_fit = bo_module.GaussianProcessRegressor.fit
    calls = {"n": 0}

    def flaky_fit(self, X, y):
        calls["n"] += 1
        if calls["n"] == 1:
            raise np.linalg.LinAlgError("simulated Cholesky failure")
        return real_fit(self, X, y)

    monkeypatch.setattr(bo_module.GaussianProcessRegressor, "fit", flaky_fit)
    res = _run("bo_ard")
    events = res.diagnostics["jitter_events"]
    assert len(events) == 1 and events[0]["new"] == pytest.approx(events[0]["old"] * 100)
    assert res.diagnostics["final_jitter"] == pytest.approx(events[0]["new"])
    assert res.evals_used == BUDGET


def test_repeated_failure_raises(monkeypatch):
    def always_fail(self, X, y):
        raise np.linalg.LinAlgError("simulated")
    monkeypatch.setattr(bo_module.GaussianProcessRegressor, "fit", always_fail)
    with pytest.raises(RuntimeError, match="GP fit failed"):
        _run("bo_ard")