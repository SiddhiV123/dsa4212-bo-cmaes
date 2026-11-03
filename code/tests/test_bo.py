"""Tests for code/bo.py (decision log D7, D12, D30, D32)."""

import numpy as np
import pytest
from scipy.stats import norm

pytest.importorskip("sklearn")

from bo import BO_METHODS, expected_improvement, make_kernel, run_bo
from objectives import make_ellipsoid_instance, sphere
from runner import run_uniform

BUDGET, N_INIT, CHECKPOINTS = 23, 5, [10, 23]
FAST = dict(n_restarts_gp=1, n_candidates=300, n_starts_acq=2)


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
    [dict(method="bo_rbf"), dict(n_init=0), dict(n_init=BUDGET + 1), dict(jitter=0.0)],
)
def test_invalid_arguments_rejected(kwargs):
    args = dict(f=sphere, method="bo_ard", d=4, lower=-5, upper=5, budget=BUDGET,
                checkpoints=CHECKPOINTS, p_seed=1, n_init=N_INIT)
    args.update(kwargs)
    with pytest.raises(ValueError):
        run_bo(**args)