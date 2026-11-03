"""Tests for code/runner.py (decision log D6, D8-D13)."""

import numpy as np
import pytest

from config_loader import load_config
from objectives import make_ellipsoid_instance, sphere
from runner import (
    BudgetExhausted,
    CountingObjective,
    cma_options_from_config,
    effective_covariance,
    initial_point,
    kappa_eff,
    method_seed,
    problem_seed,
    run_cma,
    run_uniform,
)

cma = pytest.importorskip("cma")

VARIANTS = ["cma_es_full", "cma_es_diagonal", "isotropic_es"]
BUDGET = 203                      # deliberately not a multiple of popsize 8
CHECKPOINTS = [50, 100, 203]


@pytest.fixture(scope="module")
def cfg():
    return load_config(stage="pilot")


def _setup(cfg, variant, kappa=1e4, orientation="rotated", rep=0):
    lo = cfg["synthetic"]["domain"]["lower_bound"]
    hi = cfg["synthetic"]["domain"]["upper_bound"]
    p = problem_seed(cfg["reproducibility"]["base_seed"], cfg["pilot"]["seed_offset"], 4, kappa, rep)
    inst = make_ellipsoid_instance(4, kappa, orientation, p, lo, hi, 0.8)
    opts, sigma0 = cma_options_from_config(cfg["methods"][variant], 4, lo, hi)
    return inst, opts, sigma0, initial_point(p, 4, lo, hi), p


# ------------------------------------------------------ counting wrapper

def test_counting_wrapper_caps_and_tracks():
    obj = CountingObjective(lambda x: float(np.sum(x**2)), 5, [2, 5])
    for v in [3.0, 1.0, 2.0, 0.5, 4.0]:
        obj(np.array([v]))
    assert obj.n_evals == 5 and obj.remaining == 0
    assert obj.best_f == pytest.approx(0.25)
    assert obj.checkpoint_best == {2: 1.0, 5: 0.25}
    with pytest.raises(BudgetExhausted):
        obj(np.array([0.0]))


@pytest.mark.parametrize("bad", [2.5, True, "10"])
def test_counting_wrapper_requires_integer_budget(bad):
    with pytest.raises(TypeError, match="integer"):
        CountingObjective(lambda x: 0.0, bad, [10])


@pytest.mark.parametrize("bad", [0, -3])
def test_counting_wrapper_requires_positive_budget(bad):
    with pytest.raises(ValueError, match="positive"):
        CountingObjective(lambda x: 0.0, bad, [1])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_counting_wrapper_rejects_non_finite_values(bad_value):
    obj = CountingObjective(lambda x: bad_value, 3, [3])
    with pytest.raises(ValueError, match="non-finite"):
        obj(np.array([1.0]))
    assert obj.n_evals == 0 and obj.best_x is None


def test_counting_wrapper_does_not_let_objective_mutate_logged_point():
    def mutating(x):
        x[0] = 99.0
        return 1.0
    obj = CountingObjective(mutating, 1, [1])
    obj(np.array([1.0]))
    assert obj.best_x[0] == 1.0 and obj.log[0]["x"] == [1.0]


def test_counting_wrapper_rejects_bad_checkpoints():
    with pytest.raises(ValueError):
        CountingObjective(lambda x: 0.0, 10, [5, 8])


# --------------------------------------------------------------- seeds

def test_problem_seed_ignores_orientation_and_method():
    a = make_ellipsoid_instance(4, 1e4, "aligned", problem_seed(4212, 0, 4, 1e4, 3))
    b = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed(4212, 0, 4, 1e4, 3))
    assert np.array_equal(a.x_opt, b.x_opt)


def test_pilot_and_main_seeds_differ():
    assert problem_seed(4212, 1_000_000, 4, 1e4, 0) != problem_seed(4212, 0, 4, 1e4, 0)


def test_method_seeds_distinct():
    seeds = {method_seed(123, m) for m in VARIANTS + ["uniform_random_search"]}
    assert len(seeds) == 4


def test_initial_point_in_box_and_independent_of_x_opt():
    p = problem_seed(4212, 0, 4, 1e4, 0)
    x0 = initial_point(p, 4, -5, 5)
    inst = make_ellipsoid_instance(4, 1e4, "rotated", p)
    assert np.all(np.abs(x0) <= 5)
    assert not np.allclose(x0, inst.x_opt)


# ------------------------------------------------------------- CMA-ES

@pytest.mark.parametrize("variant", VARIANTS)
def test_cma_uses_exact_budget(cfg, variant):
    inst, opts, sigma0, x0, p = _setup(cfg, variant)
    res = run_cma(inst, variant, opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    assert res.evals_used == BUDGET and res.budget_completed
    assert len(res.eval_log) == BUDGET
    assert res.early_termination is None


@pytest.mark.parametrize("variant", VARIANTS)
def test_cma_reports_best_evaluated_point(cfg, variant):
    inst, opts, sigma0, x0, p = _setup(cfg, variant)
    res = run_cma(inst, variant, opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    logged = min(r["f"] for r in res.eval_log)
    assert res.best_f == logged
    assert inst(np.array(res.best_x)) == pytest.approx(res.best_f)
    assert res.regret == pytest.approx(res.best_f)
    assert set(res.checkpoint_regret) == set(CHECKPOINTS)
    assert res.checkpoint_regret[203] == res.regret


def test_cma_points_stay_in_box(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    res = run_cma(inst, "cma_es_full", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    X = np.array([r["x"] for r in res.eval_log])
    assert np.all(X >= -5) and np.all(X <= 5)


def test_cma_reproducible(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    a = run_cma(inst, "cma_es_full", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    b = run_cma(inst, "cma_es_full", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    assert [r["f"] for r in a.eval_log] == [r["f"] for r in b.eval_log]


def test_cma_variants_share_x0(cfg):
    starts = {tuple(_setup(cfg, v)[3]) for v in VARIANTS}
    assert len(starts) == 1


def test_early_termination_is_flagged(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    opts = dict(opts, tolfun=1e6)          # force pycma to stop early
    res = run_cma(inst, "cma_es_full", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    assert not res.budget_completed and res.evals_used < BUDGET
    assert res.early_termination and res.stop_category == "other"
    assert set(res.checkpoint_regret) == set(CHECKPOINTS)


# ------------------------------------------- covariance (D7, D8, D18)

def test_diagonal_covariance_has_no_off_diagonals(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_diagonal")
    res = run_cma(inst, "cma_es_diagonal", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    for snap in res.covariance_snapshots.values():
        for kind in ("internal", "evaluated"):
            C = np.array(snap[kind])
            assert np.allclose(C - np.diag(np.diag(C)), 0)


def test_isotropic_covariance_stays_spherical(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "isotropic_es")
    res = run_cma(inst, "isotropic_es", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    # The algorithm itself stays isotropic; near a bound the evaluated-space
    # covariance can be stretched by the bound transform, so check internal.
    for snap in res.covariance_snapshots.values():
        assert np.linalg.cond(np.array(snap["internal"])) < 1.001


def test_full_cma_learns_rotated_geometry(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    res = run_cma(inst, "cma_es_full", opts, sigma0, x0, 1000, [500, 1000], p, log_points=False)
    k_end = kappa_eff(np.array(res.covariance_snapshots[1000]["evaluated"]), inst.hessian)
    assert k_end < inst.kappa_H / 100      # learned most of the 1e4 conditioning


def test_kappa_eff_is_one_for_inverse_hessian():
    H = np.diag([2.0, 20.0, 200.0])
    assert kappa_eff(np.linalg.inv(H), H) == pytest.approx(1.0)
    assert kappa_eff(np.eye(3), H) == pytest.approx(100.0)


# ------------------------------------------------------- uniform search

def test_uniform_exact_budget_in_box_reproducible():
    f = lambda x: sphere(x)
    a = run_uniform(f, 4, -5, 5, BUDGET, CHECKPOINTS, p_seed=7)
    b = run_uniform(f, 4, -5, 5, BUDGET, CHECKPOINTS, p_seed=7)
    assert a.evals_used == BUDGET and a.budget_completed
    X = np.array([r["x"] for r in a.eval_log])
    assert np.all(X >= -5) and np.all(X <= 5)
    assert [r["f"] for r in a.eval_log] == [r["f"] for r in b.eval_log]


# ---------------------------------------------------- config translation

def test_cma_options_from_config(cfg):
    opts, sigma0 = cma_options_from_config(cfg["methods"]["cma_es_diagonal"], 4, -5, 5)
    assert opts["bounds"] == [-5, 5]
    assert opts["CMA_diagonal"] is True
    assert "popsize" not in opts            # pycma default
    assert sigma0 == pytest.approx(3.0)
    iso, _ = cma_options_from_config(cfg["methods"]["isotropic_es"], 4, -5, 5)
    assert iso["CMA_on"] == 0


def test_effective_covariance_matches_samples(cfg):
    """Formula covariance agrees with pycma's own samples (bounds included)."""
    import warnings
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = cma.CMAEvolutionStrategy(inst.x_opt + 0.5, 3.0, dict(opts, seed=5))
        for _ in range(120):
            X = es.ask()
            es.tell(X, [inst(x) for x in X])
        S = np.cov(np.array(es.ask(number=40000)).T)
    E = effective_covariance(es)
    assert kappa_eff(E, inst.hessian) == pytest.approx(kappa_eff(S, inst.hessian), rel=0.1)


@pytest.mark.parametrize("variant", VARIANTS)
def test_converged_run_stops_only_for_budget_or_precision(cfg, variant):
    """Disabled tolerance stops never fire; the only early stops allowed are
    pycma's machine-precision checks, which fire at regret ~1e-30."""
    inst, opts, sigma0, x0, p = _setup(cfg, variant, kappa=1.0)
    res = run_cma(inst, variant, opts, sigma0, x0, 2000, [2000], p, log_points=False)
    assert res.stop_category in ("budget", "numerical_precision"), res.early_termination
    if res.stop_category == "numerical_precision":
        assert not res.budget_completed and res.regret < 1e-20


def test_stop_category():
    from runner import stop_category
    assert stop_category(100, 100, None) == "budget"
    assert stop_category(90, 100, {"noeffectaxis": None}) == "numerical_precision"
    assert stop_category(90, 100, {"noeffectcoord": [1], "noeffectaxis": None}) == "numerical_precision"
    assert stop_category(90, 100, {"tolconditioncov": 1e14}) == "other"
    assert stop_category(90, 100, {"noeffectaxis": None, "tolupsigma": 1}) == "other"

def test_covariance_snapshots_record_true_evaluation_count(cfg):
    inst, opts, sigma0, x0, p = _setup(cfg, "cma_es_full")
    res = run_cma(inst, "cma_es_full", opts, sigma0, x0, BUDGET, CHECKPOINTS, p)
    for c, snap in res.covariance_snapshots.items():
        assert snap["evals_at_snapshot"] >= min(c, res.evals_used)
        assert snap["evals_at_snapshot"] - c < 8 or c == BUDGET   # within one generation