"""Synthetic objective functions for the DSA4212 BO vs CMA-ES study.

Ellipsoid (main experiment) and Sphere (validation only), plus Haar-random
rotations and the Hessian diagnostics used as instance covariates.

Conventions (see decision log D2-D4):
    f(x) = sum_{i=1..d} kappa^((i-1)/(d-1)) * z_i^2,   z = R (x - x_opt)
    In code the index runs i = 0..d-1, so the weight is kappa^(i/(d-1)).
    Optimum value is 0 at x = x_opt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ellipsoid_weights",
    "ellipsoid",
    "sphere",
    "haar_orthogonal",
    "ellipsoid_hessian",
    "condition_number",
    "jacobi_condition_number",
    "EllipsoidInstance",
    "make_ellipsoid_instance",
]


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------

def ellipsoid_weights(d: int, kappa: float) -> np.ndarray:
    """Axis weights kappa^(i/(d-1)) for i = 0..d-1 (smallest 1, largest kappa)."""
    if d < 1:
        raise ValueError("d must be >= 1")
    if kappa < 1:
        raise ValueError("kappa must be >= 1")
    if d == 1:
        return np.ones(1)
    i = np.arange(d)
    return kappa ** (i / (d - 1))


def _as_batch(x: np.ndarray, d: int) -> tuple[np.ndarray, bool]:
    """Return x as an (n, d) array and whether the input was a single point."""
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    xb = x[None, :] if single else x
    if xb.ndim != 2 or xb.shape[1] != d:
        raise ValueError(f"expected shape ({d},) or (n, {d}), got {x.shape}")
    return xb, single


def ellipsoid(x, kappa: float, R: np.ndarray, x_opt: np.ndarray):
    """Rotated ellipsoid. Accepts a point (d,) or a batch (n, d)."""
    R = np.asarray(R, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)
    d = x_opt.shape[0]
    if R.shape != (d, d):
        raise ValueError(f"R must be ({d}, {d}), got {R.shape}")
    xb, single = _as_batch(x, d)
    z = (xb - x_opt) @ R.T          # row-wise z = R (x - x_opt)
    f = (z ** 2) @ ellipsoid_weights(d, kappa)
    return float(f[0]) if single else f


def sphere(x, x_opt=None):
    """Sphere sum_i (x_i - x_opt_i)^2. Accepts (d,) or (n, d)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    x_opt = np.zeros(d) if x_opt is None else np.asarray(x_opt, dtype=float)
    xb, single = _as_batch(x, d)
    f = np.sum((xb - x_opt) ** 2, axis=1)
    return float(f[0]) if single else f


# ----------------------------------------------------------------------
# Rotation (Mezzadri 2007: QR of a Gaussian matrix with sign correction)
# ----------------------------------------------------------------------

def haar_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal d x d matrix.

    Plain QR of a Gaussian matrix is NOT Haar-distributed; multiplying the
    columns of Q by sign(diag(R)) fixes this (Mezzadri, 2007).
    """
    Z = rng.standard_normal((d, d))
    Q, Rqr = np.linalg.qr(Z)
    signs = np.sign(np.diag(Rqr))
    signs[signs == 0] = 1.0
    return Q * signs


# ----------------------------------------------------------------------
# Hessian diagnostics (instance covariates; decision log D18, D22)
# ----------------------------------------------------------------------

def ellipsoid_hessian(kappa: float, R: np.ndarray) -> np.ndarray:
    """Exact Hessian H = 2 R^T diag(w) R (constant for a quadratic)."""
    R = np.asarray(R, dtype=float)
    w = ellipsoid_weights(R.shape[0], kappa)
    return 2.0 * R.T @ np.diag(w) @ R


def condition_number(H: np.ndarray) -> float:
    """kappa_H = lambda_max / lambda_min for a symmetric positive definite H."""
    eig = np.linalg.eigvalsh(np.asarray(H, dtype=float))
    if eig[0] <= 0:
        raise ValueError("H must be positive definite")
    return float(eig[-1] / eig[0])


def jacobi_condition_number(H: np.ndarray) -> float:
    """kappa_J = cond(D^-1/2 H D^-1/2), D = diag(H) (van der Sluis, 1969)."""
    H = np.asarray(H, dtype=float)
    dinv = 1.0 / np.sqrt(np.diag(H))
    return condition_number(H * np.outer(dinv, dinv))


# ----------------------------------------------------------------------
# Problem instances
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class EllipsoidInstance:
    """One saved problem instance. All methods in a replicate share it."""

    d: int
    kappa: float
    orientation: str          # "aligned" or "rotated"
    x_opt: np.ndarray
    R: np.ndarray
    lower: float
    upper: float
    problem_seed: int
    hessian: np.ndarray = field(repr=False)
    kappa_H: float = 0.0
    kappa_J: float = 0.0

    def __call__(self, x):
        return ellipsoid(x, self.kappa, self.R, self.x_opt)

    def to_dict(self) -> dict:
        """Serialisable record for results/raw (problem instance log)."""
        return {
            "d": self.d,
            "kappa": self.kappa,
            "orientation": self.orientation,
            "x_opt": self.x_opt.tolist(),
            "R": self.R.tolist(),
            "lower": self.lower,
            "upper": self.upper,
            "problem_seed": self.problem_seed,
            "kappa_H": self.kappa_H,
            "kappa_J": self.kappa_J,
        }


def make_ellipsoid_instance(
    d: int,
    kappa: float,
    orientation: str,
    problem_seed: int,
    lower: float = -5.0,
    upper: float = 5.0,
    sub_box_fraction: float = 0.8,
) -> EllipsoidInstance:
    """Draw x_opt uniformly in the centred sub-box and R per orientation.

    The problem seed must depend only on the condition and replicate index
    (decision log D12), never on the method, so every method sees the same
    instance. x_opt is drawn before R, so the aligned and rotated instances
    built from the same seed share x_opt.
    """
    if orientation not in ("aligned", "rotated"):
        raise ValueError("orientation must be 'aligned' or 'rotated'")
    if not 0 < sub_box_fraction <= 1:
        raise ValueError("sub_box_fraction must be in (0, 1]")
    if not lower < upper:
        raise ValueError("lower must be < upper")

    rng = np.random.default_rng(problem_seed)
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * sub_box_fraction
    x_opt = rng.uniform(centre - half, centre + half, size=d)
    R = haar_orthogonal(d, rng) if orientation == "rotated" else np.eye(d)

    H = ellipsoid_hessian(kappa, R)
    return EllipsoidInstance(
        d=d,
        kappa=float(kappa),
        orientation=orientation,
        x_opt=x_opt,
        R=R,
        lower=float(lower),
        upper=float(upper),
        problem_seed=int(problem_seed),
        hessian=H,
        kappa_H=condition_number(H),
        kappa_J=jacobi_condition_number(H),
    )