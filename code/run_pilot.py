"""Run the synthetic pilot (decision log D11, D20).

Usage (from the repo root):
    python code/run_pilot.py                 # full pilot from configs/experiment.yaml
    python code/run_pilot.py --quick         # smoke test: 1 replicate, budget 100

Outputs go to results/raw/pilot/<run_id>/ and are labelled as pilot data.
Pilot runs use their own seed stream and are excluded from the
confirmatory analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config  # noqa: E402
from objectives import make_ellipsoid_instance  # noqa: E402
from runner import (  # noqa: E402
    cma_options_from_config,
    initial_point,
    kappa_eff,
    problem_seed,
    run_cma,
    run_uniform,
)

CMA_METHODS = ("cma_es_full", "cma_es_diagonal", "isotropic_es")


def software_versions() -> dict[str, str]:
    import cma

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cma": cma.__version__,
    }


def run_pilot(
    config_path: Path = DEFAULT_CONFIG_PATH,
    out_root: Path | None = None,
    quick: bool = False,
    run_id: str | None = None,
) -> Path:
    cfg = load_config(config_path, stage="pilot")
    pilot = cfg["pilot"]

    base_seed = cfg["reproducibility"]["base_seed"]
    offset = pilot["seed_offset"]
    lower = cfg["synthetic"]["domain"]["lower_bound"]
    upper = cfg["synthetic"]["domain"]["upper_bound"]
    sub_box = cfg["synthetic"]["ellipsoid"]["optimum"]["sub_box_fraction"]
    budget = cfg["evaluation"]["budget_per_method"]
    checkpoints = list(cfg["evaluation"]["checkpoints"]["evaluation_counts"])
    reps = pilot["replicates_per_condition"]
    methods = list(pilot["methods"])

    if quick:
        budget, checkpoints, reps = 100, [50, 100], 1

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root or PROJECT_ROOT / "results" / "raw" / "pilot") / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    summaries, evaluations, instances, covariances = [], [], [], []

    for d in pilot["conditions"]["dimensions"]:
        for kappa in pilot["conditions"]["condition_numbers"]:
            for orientation in pilot["conditions"]["orientations"]:
                for rep in range(reps):
                    p = problem_seed(base_seed, offset, d, kappa, rep)
                    inst = make_ellipsoid_instance(d, kappa, orientation, p, lower, upper, sub_box)
                    instances.append({**inst.to_dict(), "replicate": rep})
                    x0 = initial_point(p, d, lower, upper)
                    condition = {"d": d, "kappa": kappa, "orientation": orientation,
                                 "replicate": rep, "kappa_J": inst.kappa_J}

                    for method in methods:
                        if method in CMA_METHODS:
                            opts, sigma0 = cma_options_from_config(cfg["methods"][method], d, lower, upper)
                            res = run_cma(inst, method, opts, sigma0, x0, budget, checkpoints, p)
                        elif method == "uniform_random_search":
                            res = run_uniform(inst, d, lower, upper, budget, checkpoints, p)
                        else:
                            raise ValueError(f"method not supported in the pilot: {method}")

                        summaries.append({"stage": "pilot", **condition, **res.summary()})
                        for row in res.eval_log:
                            evaluations.append({"method": method, "problem_seed": p,
                                                "orientation": orientation, **row})
                        for c, snap in res.covariance_snapshots.items():
                            covariances.append({
                                "method": method, "problem_seed": p, "orientation": orientation,
                                "kappa": kappa, "checkpoint": c, "evals_at_snapshot": snap.get("evals_at_snapshot"),
                                "kappa_eff": kappa_eff(np.array(snap["evaluated"]), inst.hessian),
                                "internal": snap["internal"], "evaluated": snap["evaluated"],
                            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(evaluations).to_parquet(out_dir / "evaluations.parquet", index=False)
    (out_dir / "instances.json").write_text(json.dumps(instances, indent=1))
    (out_dir / "covariance_snapshots.json").write_text(json.dumps(covariances))
    with (out_dir / "config_resolved.yaml").open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    (out_dir / "run_info.json").write_text(json.dumps({
        "stage": "pilot",
        "run_id": run_id,
        "quick": quick,
        "budget": budget,
        "checkpoints": checkpoints,
        "replicates": reps,
        "n_runs": len(summary),
        "software": software_versions(),
        "stop_categories": summary["stop_category"].value_counts().to_dict(),
    }, indent=1))
    return out_dir


def plot_regret(out_dir: Path, floor: float = 1e-8) -> Path:
    """Median log10 regret against evaluations, one panel per condition.

    Regret below the pre-specified floor (D14) is shown at the floor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(out_dir / "summary.csv")
    cps = sorted(int(c.split("_")[-1]) for c in s.columns if c.startswith("regret_at_"))
    conditions = s[["kappa", "orientation"]].drop_duplicates().sort_values(["kappa", "orientation"])

    fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, cond) in zip(axes, conditions.iterrows()):
        sub = s[(s.kappa == cond.kappa) & (s.orientation == cond.orientation)]
        for method, g in sub.groupby("method"):
            med = [np.median(np.log10(np.maximum(g[f"regret_at_{c}"], floor))) for c in cps]
            ax.plot(cps, med, marker="o", label=method)
        ax.set_xscale("log")
        ax.set_title(f"kappa={cond.kappa:g}, {cond.orientation}")
        ax.set_xlabel("evaluations")
    for ax in axes:
        ax.axhline(np.log10(floor), color="grey", lw=0.8, ls="--")
    axes[0].set_ylabel(f"median log10 regret (floor {floor:g})")
    axes[-1].legend(fontsize=7)
    fig.suptitle("PILOT DATA (excluded from confirmatory analysis)", fontsize=9)
    fig.tight_layout()
    path = out_dir / "pilot_regret.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 replicate, budget 100")
    parser.add_argument("--replot", type=Path, help="regenerate the figure for an existing pilot folder only")
    args = parser.parse_args()

    if args.replot:
        print(f"Figure: {plot_regret(args.replot)}")
        return

    out_dir = run_pilot(args.config, quick=args.quick)
    fig = plot_regret(out_dir)
    info = json.loads((out_dir / "run_info.json").read_text())
    print(f"Pilot finished: {info['n_runs']} runs -> {out_dir}")
    print(f"Stop categories: {info['stop_categories']}")
    print(f"Figure: {fig}")


if __name__ == "__main__":
    main()