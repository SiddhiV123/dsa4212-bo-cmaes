"""Unit tests for the statistics in code/analysis.py (decision log D14-D17, D22)."""

import numpy as np
import pandas as pd
import pytest

from analysis import (
    bootstrap_mean_ci,
    did,
    evals_to_target,
    log_regret,
    paired_L,
    rotation_penalty,
    spearman_with_ci,
    uniform_success_probability,
)


def _summary(effects: dict, n_rep: int = 20, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic summary: log10 regret = effects[(method, orientation)] (+ noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (method, orientation), L in effects.items():
        for r in range(n_rep):
            rows.append({"method": method, "kappa": 1e4, "orientation": orientation, "replicate": r,
                         "regret_at_100": 10 ** (L + noise * rng.standard_normal())})
    return pd.DataFrame(rows)


def test_log_regret_applies_floor():
    assert np.allclose(log_regret([1e-12, 1e-3, 1.0], 1e-8), [-8, -3, 0])


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(2.0, 1.0, 50)
    a, b = bootstrap_mean_ci(x, 2000, seed=3), bootstrap_mean_ci(x, 2000, seed=3)
    assert a == b and a[1] < a[0] < a[2]


def test_bootstrap_ci_is_degenerate_for_constant_data():
    assert bootstrap_mean_ci(np.zeros(10), 500, 0) == (0.0, 0.0, 0.0)


def test_rotation_penalty_recovers_known_value():
    s = _summary({("m", "aligned"): -6.0, ("m", "rotated"): -2.0})
    r = rotation_penalty(s, "m", 1e4, 100, 1e-8, 500, 0)
    assert r["estimate"] == pytest.approx(4.0)


def test_did_recovers_known_value():
    # diagonal: penalty +5; full: penalty +1  =>  DiD = +4
    s = _summary({("cma_es_diagonal", "aligned"): -7.0, ("cma_es_diagonal", "rotated"): -2.0,
                  ("cma_es_full", "aligned"): -7.0, ("cma_es_full", "rotated"): -6.0}, noise=0.1)
    r = did(s, "cma_es_diagonal", 1e4, 100, 1e-8, 2000, 0)
    assert r["did"] == pytest.approx(4.0, abs=0.15) and r["excludes_zero"]


def test_did_respects_floor():
    # both methods below the floor everywhere => DiD exactly 0
    s = _summary({("cma_es_diagonal", "aligned"): -20, ("cma_es_diagonal", "rotated"): -15,
                  ("cma_es_full", "aligned"): -25, ("cma_es_full", "rotated"): -12})
    assert did(s, "cma_es_diagonal", 1e4, 100, 1e-8, 200, 0)["did"] == 0.0


def test_paired_L_pairs_by_replicate():
    s = _summary({("m", "aligned"): -1.0, ("m", "rotated"): -3.0}, n_rep=5)
    t = paired_L(s, "m", 1e4, 100, 1e-8)
    assert list(t.index) == list(range(5)) and set(t.columns) == {"aligned", "rotated"}


def test_spearman_perfect_monotone():
    r = spearman_with_ci(np.arange(20), np.arange(20) ** 2, 300, 0)
    assert r["rho"] == pytest.approx(1.0) and r["ci_low"] > 0.9


def test_evals_to_target():
    assert evals_to_target(np.array([5, 3, 1, 0.5]), 1.0) == 3.0
    assert evals_to_target(np.array([5, 3]), 1.0) == np.inf


def test_uniform_success_probability():
    assert uniform_success_probability(0.0, 100) == 0.0
    assert uniform_success_probability(0.01, 100) == pytest.approx(1 - 0.99 ** 100)