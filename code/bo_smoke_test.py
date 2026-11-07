"""BO smoke test: verify BO's fixed settings and choose its budget ceiling (decision log D21).

BO's settings are fixed by Le Riche & Picheny (2021) before this script runs
(initial design d + 4, lengthscale bounds, 2d likelihood starts, failure-
triggered jitter, EI optimiser). This script does NOT choose them. It:

  1. Verifies that the ARD and isotropic GPs fit the initial design on every
     smoke instance (failure = exception or non-finite result after the
     jitter policy has been applied).
  2. Verifies that full BO runs to 90 evaluations complete for both kernels
     on one instance per condition, and records any jitter events.
  3. Chooses the ceiling, the only setting it selects, by runtime alone:
     the first of (250, standard), (250, random), (90, standard),
     (90, random) whose projected time for all BO main runs,
     mean run time x runs / workers x 1.25, fits the cap.

Regret is never used. Smoke instances use their own seed stream.

Usage (from the repo root, on an otherwise quiet machine):
    caffeinate -i python code/bo_smoke_test.py            # cap 2 h, workers = cores - 2
    python code/bo_smoke_test.py --cap-hours 2 --workers 8

Outputs go to results/raw/bo_smoke/<run_id>/smoke_report.json.
"""

from __future__ import annotations

import os

# One BLAS thread per process: timings then match the main runs, where
# several BO runs execute in parallel (avoids thread oversubscription).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bo import BASE_JITTER, default_n_init, initial_design, make_kernel, run_bo  # noqa: E402
from config_loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config  # noqa: E402
from objectives import make_ellipsoid_instance  # noqa: E402
from runner import problem_seed  # noqa: E402

SMOKE_SEED_OFFSET = 2_000_000          # distinct from main (0) and pilot (1,000,000)
CEILING_OPTIONS = ((250, "standard"), (250, "random"), (90, "standard"), (90, "random"))
SAFETY_MARGIN = 1.25
BO_VARIANTS = ("bo_ard", "bo_isotropic")


# ----------------------------------------------------------------------
# Pure decision rules (unit-tested)
# ----------------------------------------------------------------------

def projected_hours(mean_run_s: float, n_runs: int, workers: int, margin: float = SAFETY_MARGIN) -> float:
    return mean_run_s * n_runs / max(workers, 1) * margin / 3600.0


def choose_ceiling(projections: dict[tuple[int, str], float], cap_hours: float):
    """First option in CEILING_OPTIONS whose projection fits the cap.

    Returns (ceiling, acq_setting, projected_hours, within_cap). If nothing
    fits, the last option is returned with within_cap=False, so the
    shortfall is visible rather than silently accepted.
    """
    for option in CEILING_OPTIONS:
        if option in projections and projections[option] <= cap_hours:
            return option[0], option[1], projections[option], True
    last = CEILING_OPTIONS[-1]
    return last[0], last[1], projections.get(last, float("nan")), False


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------

def initial_fit_ok(f, p_seed, d, lower, upper, n_init, kind):
    """Fit one GP to the shared initial design; return (ok, warnings, error)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor

    U = initial_design(p_seed, d, n_init)
    y = np.asarray(f(lower + (upper - lower) * U), dtype=float)
    try:
        gp = GaussianProcessRegressor(kernel=make_kernel(kind, d), alpha=BASE_JITTER, normalize_y=True,
                                      n_restarts_optimizer=2 * d - 1, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(U, y)
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        ok = bool(np.all(np.isfinite(gp.kernel_.theta)) and np.isfinite(gp.log_marginal_likelihood_value_))
        return ok, n_warn, "" if ok else "non-finite hyperparameters or likelihood"
    except Exception as exc:   # with base jitter; run_bo would escalate jitter here
        return False, 0, f"{type(exc).__name__}: {exc}"


def bo_run(f, method, d, lower, upper, budget, p_seed, acq_setting):
    """Run BO; return (ok, wall_time_s, diagnostics_summary, error_message)."""
    try:
        res = run_bo(f, method, d, lower, upper, budget, [budget], p_seed,
                     acq_setting=acq_setting, log_points=False)
        diag = res.diagnostics
        ok = bool(np.isfinite(res.best_f) and np.all(np.isfinite(diag["checkpoints"][budget]["length_scale"])))
        summary = {"jitter_events": diag["jitter_events"], "final_jitter": diag["final_jitter"],
                   "convergence_warnings": diag["n_fit_convergence_warnings"],
                   "acquisition_events": diag["acquisition_events"], "overhead_s": diag["overhead_s"]}
        return ok, res.wall_time_s, summary, "" if ok else "non-finite result"
    except Exception as exc:
        return False, float("nan"), {}, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------
# Main procedure
# ----------------------------------------------------------------------

def run_smoke(
    config_path: Path = DEFAULT_CONFIG_PATH,
    out_root: Path | None = None,
    run_id: str | None = None,
    cap_hours: float = 2.0,
    workers: int | None = None,
    quick: bool = False,
    log=print,
) -> dict:
    cfg = load_config(config_path, stage="main")
    base = cfg["reproducibility"]["base_seed"]
    lower = cfg["synthetic"]["domain"]["lower_bound"]
    upper = cfg["synthetic"]["domain"]["upper_bound"]
    sub_box = cfg["synthetic"]["ellipsoid"]["optimum"]["sub_box_fraction"]
    d = cfg["synthetic"]["ellipsoid"]["dimensions"][0]
    kappas = cfg["synthetic"]["ellipsoid"]["bo_condition_numbers"]
    orientations = cfg["synthetic"]["ellipsoid"]["orientations"]
    replicates = cfg["evaluation"]["target_replicates_per_condition"]
    n_main_runs = len(BO_VARIANTS) * len(kappas) * len(orientations) * replicates
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    n_init = default_n_init(d)

    smoke_reps = 2 if quick else 5
    stability_budget = 20 if quick else 90
    ceiling_budget = {250: 30, 90: 15} if quick else {250: 250, 90: 90}

    def instance(kappa, orientation, rep):
        p = problem_seed(base, SMOKE_SEED_OFFSET, d, kappa, rep)
        return make_ellipsoid_instance(d, kappa, orientation, p, lower, upper, sub_box), p

    report: dict = {"rules": "decision log D21", "quick": quick, "cap_hours": cap_hours,
                    "workers": workers, "n_main_bo_runs": n_main_runs, "n_init": n_init,
                    "machine": {"platform": platform.platform(), "python": platform.python_version(),
                                "cpu_count": os.cpu_count()}}

    # Check 1: GP fits on the initial design ------------------------------
    log(f"Check 1: GP fits on the initial design (n_init = d + 4 = {n_init})")
    fits = []
    for kind in ("ard", "isotropic"):
        for kappa in kappas:
            for orientation in orientations:
                for rep in range(smoke_reps):
                    inst, p = instance(kappa, orientation, rep)
                    ok, n_warn, err = initial_fit_ok(inst, p, d, lower, upper, n_init, kind)
                    fits.append({"kernel": kind, "kappa": kappa, "orientation": orientation,
                                 "replicate": rep, "ok": ok, "convergence_warnings": n_warn, "error": err})
    n_ok = sum(r["ok"] for r in fits)
    log(f"  {n_ok}/{len(fits)} fits succeeded with base jitter "
        f"(any failure is handled in BO runs by the jitter policy)")
    report["check1_initial_fits"] = {"succeeded": n_ok, "total": len(fits), "detail": fits}

    # Check 2: full BO runs complete ---------------------------------------
    log(f"Check 2: BO to {stability_budget} evaluations, both kernels, one instance per condition")
    runs = []
    for kappa in kappas:
        for orientation in orientations:
            inst, p = instance(kappa, orientation, 0)
            for method in BO_VARIANTS:
                ok, t, summary, err = bo_run(inst, method, d, lower, upper, stability_budget, p, "standard")
                runs.append({"kappa": kappa, "orientation": orientation, "method": method,
                             "ok": ok, "wall_time_s": t, "error": err, **summary})
    n_ok2 = sum(r["ok"] for r in runs)
    n_jitter = sum(len(r.get("jitter_events", [])) for r in runs)
    log(f"  {n_ok2}/{len(runs)} runs completed; jitter escalations: {n_jitter}")
    report["check2_bo_runs"] = {"completed": n_ok2, "total": len(runs),
                                "jitter_escalations": n_jitter, "detail": runs}
    if n_ok2 < len(runs):
        raise RuntimeError("Some BO runs failed (check 2); see the report and investigate before main runs.")

    # Rule: ceiling and acquisition setting --------------------------------
    log(f"Ceiling rule (cap {cap_hours} h, {workers} workers, {n_main_runs} BO main runs)")
    inst, p = instance(kappas[-1], orientations[-1], 0)      # hardest condition
    projections, t_detail = {}, {}
    for ceiling, setting in CEILING_OPTIONS:
        times = []
        for method in BO_VARIANTS:
            ok, t, _, err = bo_run(inst, method, d, lower, upper, ceiling_budget[ceiling], p, setting)
            if not ok:
                raise RuntimeError(f"BO failed during timing ({method}, {setting}): {err}")
            times.append(t)
        mean_t = float(np.mean(times))
        projections[(ceiling, setting)] = projected_hours(mean_t, n_main_runs, workers)
        t_detail[f"{ceiling}_{setting}"] = {"run_times_s": times, "mean_s": mean_t,
                                            "projected_hours": projections[(ceiling, setting)]}
        log(f"  ceiling {ceiling:3d}, {setting:8s}: mean {mean_t:6.1f} s/run -> "
            f"projected {projections[(ceiling, setting)]:.2f} h")
        if projections[(ceiling, setting)] <= cap_hours:
            break                                 # first option within the cap
    ceiling, setting, proj, within = choose_ceiling(projections, cap_hours)
    report["ceiling"] = {"chosen": ceiling, "acq_setting": setting, "projected_hours": proj,
                         "within_cap": within, "detail": t_detail}

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root or PROJECT_ROOT / "results" / "raw" / "bo_smoke") / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    report["run_id"] = run_id
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=1))
    report["out_dir"] = str(out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cap-hours", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="tiny budgets, for testing the script only")
    args = parser.parse_args()

    r = run_smoke(args.config, cap_hours=args.cap_hours, workers=args.workers, quick=args.quick)
    c = r["ceiling"]
    print("\n=== Smoke-test outcome (record in configs/experiment.yaml and D21) ===")
    print(f"  checks: initial fits {r['check1_initial_fits']['succeeded']}/{r['check1_initial_fits']['total']}, "
          f"BO runs {r['check2_bo_runs']['completed']}/{r['check2_bo_runs']['total']}, "
          f"jitter escalations {r['check2_bo_runs']['jitter_escalations']}")
    print(f"  budget_ceiling: {c['chosen']}")
    print(f"  acquisition setting: {c['acq_setting']}")
    print(f"  projected BO main-run time: {c['projected_hours']:.2f} h on {r['workers']} workers "
          f"(within cap: {c['within_cap']})")
    print(f"  report: {r['out_dir']}/smoke_report.json")


if __name__ == "__main__":
    main()