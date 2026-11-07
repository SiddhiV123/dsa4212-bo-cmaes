"""Run the confirmatory main experiment (decision log D11, D12, D13, D20).

Runs every method on every condition with the MAIN seed stream:
  * CMA-ES variants and uniform search: all condition numbers, both
    orientations, all replicates, to the main budget (2,000 evaluations);
  * BO (ARD and isotropic): the BO condition numbers, both orientations,
    all replicates, to the BO ceiling chosen by the smoke test (D21).

The run is parallel (one process per task, one BLAS thread each) and
resumable: every finished task is saved under parts/, and re-running the
same --run-id skips finished tasks. When all tasks are done, the parts are
consolidated into files small enough for GitHub (split by method and
condition number).

Usage (from the repo root):
    caffeinate -i python code/run_main.py                      # everything
    caffeinate -i python code/run_main.py --only bo            # BO only
    python code/run_main.py --only cma --workers 6
    python code/run_main.py --run-id main                      # resume / fixed folder

Outputs (results/raw/main/<run_id>/):
    summary.csv                     one row per run (all methods)
    evaluations/<method>_k<kappa>.parquet   per-evaluation logs
    covariance_snapshots.jsonl      CMA-ES covariance + kappa_eff per checkpoint
    bo_diagnostics.jsonl            BO lengthscales, jitter events, overhead
    instances.json, config_resolved.yaml, run_info.json
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  
import json  
import platform  
import sys  
import time  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from copy import deepcopy  
from pathlib import Path  

import numpy as np  
import pandas as pd  
import yaml  

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, check_ready, load_config  
from objectives import make_ellipsoid_instance  
from runner import (  
    cma_options_from_config,
    initial_point,
    kappa_eff,
    problem_seed,
    run_cma,
    run_uniform,
)

CMA_METHODS = ("cma_es_full", "cma_es_diagonal", "isotropic_es")
QUICK_BO_SETTINGS = dict(n_restarts_gp=1, n_candidates=200, n_acq_repeats=1)


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

def task_id(t: dict) -> str:
    return f"{t['method']}__d{t['d']}__k{int(t['kappa'])}__{t['orientation']}__r{t['replicate']:03d}"


def build_tasks(cfg: dict, only: str = "all") -> list[dict]:
    """All (method, d, kappa, orientation, replicate) combinations; BO first."""
    ell = cfg["synthetic"]["ellipsoid"]
    reps = cfg["evaluation"]["target_replicates_per_condition"]
    tasks = []
    groups = []
    if only in ("all", "bo"):
        groups.append((cfg["main"]["bo_methods"], ell["bo_condition_numbers"]))
    if only in ("all", "cma"):
        groups.append((cfg["main"]["cma_and_uniform_methods"], ell["condition_numbers"]))
    for methods, kappas in groups:
        for method in methods:
            for d in ell["dimensions"]:
                for kappa in kappas:
                    for orientation in ell["orientations"]:
                        for rep in range(reps):
                            tasks.append({"method": method, "d": d, "kappa": float(kappa),
                                          "orientation": orientation, "replicate": rep})
    return tasks


def run_task(t: dict, cfg: dict, quick: bool = False) -> dict:
    """Run one task; return a dict of everything to save (picklable)."""
    from bo import run_bo   # imported in the worker

    base = cfg["reproducibility"]["base_seed"]
    offset = cfg["main"]["seed_offset"]
    lower = cfg["synthetic"]["domain"]["lower_bound"]
    upper = cfg["synthetic"]["domain"]["upper_bound"]
    sub_box = cfg["synthetic"]["ellipsoid"]["optimum"]["sub_box_fraction"]
    budget = cfg["evaluation"]["budget_per_method"]
    checkpoints = list(cfg["evaluation"]["checkpoints"]["evaluation_counts"])
    method, d, kappa, orientation, rep = t["method"], t["d"], t["kappa"], t["orientation"], t["replicate"]

    p = problem_seed(base, offset, d, kappa, rep)
    inst = make_ellipsoid_instance(d, kappa, orientation, p, lower, upper, sub_box)
    out = {"task_id": task_id(t), "instance": {**inst.to_dict(), "replicate": rep},
           "covariance": [], "bo_diagnostics": None}

    if method in CMA_METHODS:
        opts, sigma0 = cma_options_from_config(cfg["methods"][method], d, lower, upper)
        x0 = initial_point(p, d, lower, upper)
        res = run_cma(inst, method, opts, sigma0, x0, budget, checkpoints, p)
        for c, snap in res.covariance_snapshots.items():
            out["covariance"].append({
                "method": method, "d": d, "kappa": kappa, "orientation": orientation, "replicate": rep,
                "checkpoint": c, "evals_at_snapshot": snap.get("evals_at_snapshot"),
                "kappa_eff": kappa_eff(np.array(snap["evaluated"]), inst.hessian),
                "evaluated": snap["evaluated"],
            })
        extra = {}
    elif method == "uniform_random_search":
        res = run_uniform(inst, d, lower, upper, budget, checkpoints, p)
        extra = {}
    else:
        bo = cfg["methods"]["bayesian_optimisation"]
        ceiling = int(bo["budget_ceiling"])
        bo_cps = [c for c in checkpoints if c <= ceiling]
        kw = dict(QUICK_BO_SETTINGS) if quick else {}
        res = run_bo(inst, method, d, lower, upper, ceiling, bo_cps, p,
                     n_init=int(bo["initial_design_size"]), acq_setting=bo["acquisition_setting"], **kw)
        diag = res.diagnostics
        out["bo_diagnostics"] = {"method": method, "d": d, "kappa": kappa, "orientation": orientation,
                                 "replicate": rep, **{k: v for k, v in diag.items()}}
        extra = {"bo_overhead_s": diag["overhead_s"],
                 "bo_convergence_warnings": diag["n_fit_convergence_warnings"],
                 "bo_jitter_escalations": len(diag["jitter_events"]),
                 "bo_final_jitter": diag["final_jitter"],
                 "bo_acq_events": sum(diag["acquisition_events"].values())}

    out["summary"] = {"stage": "main", "d": d, "kappa": kappa, "orientation": orientation,
                      "replicate": rep, "kappa_H": inst.kappa_H, "kappa_J": inst.kappa_J,
                      **res.summary(), **extra}
    out["evaluations"] = [{"method": method, "kappa": kappa, "orientation": orientation,
                           "replicate": rep, **row} for row in res.eval_log]
    return out


# ----------------------------------------------------------------------
# Saving, resuming, consolidating
# ----------------------------------------------------------------------

def save_part(parts_dir: Path, result: dict) -> None:
    tid = result["task_id"]
    pd.DataFrame(result["evaluations"]).to_parquet(parts_dir / f"{tid}.parquet", index=False)
    meta = {k: v for k, v in result.items() if k != "evaluations"}
    tmp = parts_dir / f"{tid}.json.tmp"
    tmp.write_text(json.dumps(meta, default=float))
    tmp.rename(parts_dir / f"{tid}.json")          # the .json marks the task as finished


def finished_ids(parts_dir: Path) -> set[str]:
    return {p.name[:-5] for p in parts_dir.glob("*.json")}


def consolidate(out_dir: Path, keep_parts: bool = False) -> pd.DataFrame:
    """Merge per-task parts into the final, GitHub-sized files."""
    parts_dir = out_dir / "parts"
    metas = [json.loads(p.read_text()) for p in sorted(parts_dir.glob("*.json"))]
    summary = pd.DataFrame([m["summary"] for m in metas]).sort_values(
        ["method", "d", "kappa", "orientation", "replicate"])
    summary.to_csv(out_dir / "summary.csv", index=False)

    instances = {}
    for m in metas:
        inst = m["instance"]
        instances[(inst["d"], inst["kappa"], inst["orientation"], inst["replicate"])] = inst
    (out_dir / "instances.json").write_text(json.dumps(
        [instances[k] for k in sorted(instances)], indent=1))

    with (out_dir / "covariance_snapshots.jsonl").open("w") as fh:
        for m in metas:
            for row in m["covariance"]:
                fh.write(json.dumps(row) + "\n")
    with (out_dir / "bo_diagnostics.jsonl").open("w") as fh:
        for m in metas:
            if m["bo_diagnostics"] is not None:
                fh.write(json.dumps(m["bo_diagnostics"], default=float) + "\n")

    ev_dir = out_dir / "evaluations"
    ev_dir.mkdir(exist_ok=True)
    for (method, kappa), g in summary.groupby(["method", "kappa"]):
        ids = [task_id({"method": method, "d": r.d, "kappa": kappa, "orientation": r.orientation,
                        "replicate": r.replicate}) for r in g.itertuples()]
        frames = [pd.read_parquet(parts_dir / f"{i}.parquet") for i in ids]
        pd.concat(frames, ignore_index=True).to_parquet(
            ev_dir / f"{method}_k{int(kappa)}.parquet", index=False, compression="zstd")

    if not keep_parts:
        for p in parts_dir.iterdir():
            p.unlink()
        parts_dir.rmdir()
    return summary


def software_versions() -> dict[str, str]:
    import cma
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "cma": cma.__version__}


def quick_config(cfg: dict) -> dict:
    """Tiny settings for testing the script only."""
    cfg = deepcopy(cfg)
    cfg["evaluation"]["budget_per_method"] = 100
    cfg["evaluation"]["checkpoints"]["evaluation_counts"] = [20, 50, 100]
    cfg["evaluation"]["target_replicates_per_condition"] = 3
    cfg["synthetic"]["ellipsoid"]["condition_numbers"] = [1, 10000]
    bo = cfg["methods"]["bayesian_optimisation"]
    bo["budget_ceiling"], bo["acquisition_setting"] = 20, "standard"
    return cfg


# ----------------------------------------------------------------------
# Main procedure
# ----------------------------------------------------------------------

def run_main(
    config_path: Path = DEFAULT_CONFIG_PATH,
    out_root: Path | None = None,
    run_id: str = "main",
    only: str = "all",
    workers: int | None = None,
    quick: bool = False,
    keep_parts: bool = False,
    log=print,
) -> Path:
    cfg = load_config(config_path, stage="main")
    if quick:
        cfg = quick_config(cfg)
    check_ready(cfg, "main")

    out_dir = (out_root or PROJECT_ROOT / "results" / "raw" / "main") / run_id
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.csv").exists() and not any(parts_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is already consolidated; use a new --run-id")

    tasks = build_tasks(cfg, only)
    done = finished_ids(parts_dir)
    todo = [t for t in tasks if task_id(t) not in done]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    log(f"{len(tasks)} tasks, {len(done)} already finished, {len(todo)} to run on {workers} workers")

    with (out_dir / "config_resolved.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    t0 = time.perf_counter()
    if workers == 1:
        for i, t in enumerate(todo, 1):
            save_part(parts_dir, run_task(t, cfg, quick))
            log(f"  [{i}/{len(todo)}] {task_id(t)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_task, t, cfg, quick): t for t in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                save_part(parts_dir, fut.result())        # a failure stops the run loudly
                if i % 25 == 0 or i == len(todo):
                    elapsed = time.perf_counter() - t0
                    log(f"  [{i}/{len(todo)}] {elapsed / 60:.1f} min elapsed")

    finished = finished_ids(parts_dir)
    all_ids = {task_id(t) for t in build_tasks(cfg, "all")}
    if all_ids <= finished:
        summary = consolidate(out_dir, keep_parts=keep_parts)
        info = {"stage": "main", "run_id": run_id, "quick": quick, "n_runs": len(summary),
                "budget": cfg["evaluation"]["budget_per_method"],
                "checkpoints": cfg["evaluation"]["checkpoints"]["evaluation_counts"],
                "bo_ceiling": cfg["methods"]["bayesian_optimisation"]["budget_ceiling"],
                "bo_acquisition_setting": cfg["methods"]["bayesian_optimisation"]["acquisition_setting"],
                "replicates": cfg["evaluation"]["target_replicates_per_condition"],
                "workers": workers, "machine": platform.platform(), "cpu_count": os.cpu_count(),
                "software": software_versions(),
                "stop_categories": summary["stop_category"].value_counts().to_dict()}
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))
        log(f"All tasks finished; consolidated into {out_dir}")
    else:
        log(f"{len(all_ids - finished)} tasks remain (run the other --only group or resume)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id", default="main")
    parser.add_argument("--only", choices=["all", "cma", "bo"], default="all")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--quick", action="store_true", help="tiny settings, for testing the script only")
    args = parser.parse_args()
    run_main(args.config, run_id=args.run_id, only=args.only, workers=args.workers,
             quick=args.quick, keep_parts=args.keep_parts)


if __name__ == "__main__":
    main()