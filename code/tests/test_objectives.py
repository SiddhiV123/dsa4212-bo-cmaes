"""Unit tests for code/objectives.py (decision log D2-D4, freeze checklist D20)."""

import numpy as np
import pytest

from objectives import (
    condition_number,
    ellipsoid,
    ellipsoid_hessian,
    ellipsoid_weights,
    haar_orthogonal,
    jacobi_condition_number,
    make_ellipsoid_instance,
    sphere,
)

KAPPAS = [1.0, 1e2, 1e4]
ORIENTATIONS = ["aligned", "rotated"]


# ---------------------------------------------------------------- weights

def test_weights_span_one_to_kappa():
    w = ellipsoid_weights(4, 1e4)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(1e4)
    assert np.all(np.diff(w) > 0)


def test_weights_kappa_one_are_all_one():
    assert np.allclose(ellipsoid_weights(4, 1.0), 1.0)


def test_weights_reject_bad_inputs():
    with pytest.raises(ValueError):
        ellipsoid_weights(0, 10)
    with pytest.raises(ValueError):
        ellipsoid_weights(4, 0.5)


# ---------------------------------------------------------- objective values

@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_optimum_value_is_zero(kappa, orientation):
    inst = make_ellipsoid_instance(4, kappa, orientation, problem_seed=1)
    assert inst(inst.x_opt) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_positive_away_from_optimum(kappa, orientation):
    inst = make_ellipsoid_instance(4, kappa, orientation, problem_seed=2)
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 4))
    assert np.all(inst(X) > 0)


def test_known_values_aligned():
    # Aligned, x_opt = 0: f(e_i) is the i-th weight.
    d, kappa = 4, 1e4
    w = ellipsoid_weights(d, kappa)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        assert ellipsoid(e, kappa, np.eye(d), np.zeros(d)) == pytest.approx(w[i])


def test_batch_matches_pointwise():
    inst = make_ellipsoid_instance(4, 1e2, "rotated", problem_seed=3)
    X = np.random.default_rng(1).uniform(-5, 5, size=(50, 4))
    batch = inst(X)
    loop = np.array([inst(x) for x in X])
    assert batch.shape == (50,)
    assert np.allclose(batch, loop)


def test_single_point_returns_float():
    inst = make_ellipsoid_instance(4, 1e2, "aligned", problem_seed=3)
    assert isinstance(inst(np.zeros(4)), float)


def test_dimension_mismatch_raises():
    inst = make_ellipsoid_instance(4, 1e2, "aligned", problem_seed=3)
    with pytest.raises(ValueError):
        inst(np.zeros(3))
    with pytest.raises(ValueError):
        ellipsoid(np.zeros(4), 1e2, np.eye(3), np.zeros(4))


# ------------------------------------------------------------------ sphere

@pytest.mark.parametrize("d", [2, 4, 10])
def test_sphere(d):
    x_opt = np.linspace(-1, 1, d)
    assert sphere(x_opt, x_opt) == pytest.approx(0.0)
    assert sphere(np.ones(d), np.zeros(d)) == pytest.approx(d)


def test_kappa_one_equals_sphere():
    inst = make_ellipsoid_instance(4, 1.0, "rotated", problem_seed=4)
    X = np.random.default_rng(2).uniform(-5, 5, size=(100, 4))
    assert np.allclose(inst(X), sphere(X, inst.x_opt))


# ---------------------------------------------------------------- rotation

@pytest.mark.parametrize("d", [2, 4, 10])
def test_rotation_is_orthogonal(d):
    R = haar_orthogonal(d, np.random.default_rng(5))
    assert np.allclose(R.T @ R, np.eye(d), atol=1e-12)
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-12


def test_rotation_reproducible_from_seed():
    a = haar_orthogonal(4, np.random.default_rng(42))
    b = haar_orthogonal(4, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_rotation_sign_correction_unbiased():
    # Without sign correction, diag(Q) is biased. With it, the mean of every
    # entry is ~0 under the Haar measure.
    rng = np.random.default_rng(7)
    mats = np.stack([haar_orthogonal(4, rng) for _ in range(4000)])
    assert np.all(np.abs(mats.mean(axis=0)) < 0.05)


def test_rotation_preserves_function_values_at_kappa_one():
    aligned = make_ellipsoid_instance(4, 1.0, "aligned", problem_seed=8)
    rotated = make_ellipsoid_instance(4, 1.0, "rotated", problem_seed=8)
    assert np.array_equal(aligned.x_opt, rotated.x_opt)
    X = np.random.default_rng(3).uniform(-5, 5, size=(100, 4))
    assert np.allclose(aligned(X), rotated(X))


# ------------------------------------------------------- Hessian diagnostics

@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_hessian_eigenvalues_and_kappa_H(kappa, orientation):
    inst = make_ellipsoid_instance(4, kappa, orientation, problem_seed=9)
    eig = np.linalg.eigvalsh(inst.hessian)
    assert np.allclose(np.sort(eig), np.sort(2 * ellipsoid_weights(4, kappa)))
    assert inst.kappa_H == pytest.approx(kappa)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_kappa_J_is_one_when_aligned(kappa):
    inst = make_ellipsoid_instance(4, kappa, "aligned", problem_seed=10)
    assert inst.kappa_J == pytest.approx(1.0)


def test_kappa_J_exceeds_one_when_rotated_and_conditioned():
    inst = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed=11)
    assert inst.kappa_J > 1.0
    assert inst.kappa_J <= inst.kappa_H * (1 + 1e-9) * 4  # van der Sluis bound


def test_hessian_matches_finite_differences():
    inst = make_ellipsoid_instance(4, 1e2, "rotated", problem_seed=12)
    h, d, x0 = 1e-3, 4, inst.x_opt + 0.1
    H_fd = np.zeros((d, d))
    E = np.eye(d) * h
    for i in range(d):
        for j in range(d):
            H_fd[i, j] = (
                inst(x0 + E[i] + E[j]) - inst(x0 + E[i] - E[j])
                - inst(x0 - E[i] + E[j]) + inst(x0 - E[i] - E[j])
            ) / (4 * h * h)
    assert np.allclose(H_fd, inst.hessian, rtol=1e-6, atol=1e-6)
    assert np.allclose(ellipsoid_hessian(1e2, inst.R), inst.hessian)


def test_condition_number_rejects_non_pd():
    with pytest.raises(ValueError):
        condition_number(np.diag([1.0, -1.0]))
    assert jacobi_condition_number(np.diag([3.0, 7.0])) == pytest.approx(1.0)


# --------------------------------------------------------------- instances

def test_x_opt_in_sub_box():
    for seed in range(200):
        inst = make_ellipsoid_instance(4, 1e2, "rotated", problem_seed=seed)
        assert np.all(np.abs(inst.x_opt) <= 4.0)


def test_instance_reproducible_and_serialisable():
    a = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed=13)
    b = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed=13)
    assert np.array_equal(a.x_opt, b.x_opt) and np.array_equal(a.R, b.R)
    rec = a.to_dict()
    assert rec["problem_seed"] == 13 and len(rec["R"]) == 4


def test_different_seeds_give_different_instances():
    a = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed=14)
    b = make_ellipsoid_instance(4, 1e4, "rotated", problem_seed=15)
    assert not np.allclose(a.x_opt, b.x_opt)


def test_instance_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_ellipsoid_instance(4, 1e2, "diagonal", problem_seed=0)
    with pytest.raises(ValueError):
        make_ellipsoid_instance(4, 1e2, "aligned", problem_seed=0, sub_box_fraction=0)

def test_matches_pycma_elli():
    """External cross-check of D2 against pycma's reference ellipsoid."""
    cma = pytest.importorskip("cma")
    X = np.random.default_rng(20).uniform(-5, 5, size=(20, 4))
    for kappa in KAPPAS:
        ours = ellipsoid(X, kappa, np.eye(4), np.zeros(4))
        ref = np.array([cma.ff.elli(x, cond=kappa) for x in X])
        assert np.allclose(ours, ref, rtol=1e-6)