"""Analysis of the main runs: every number reported in the paper (D14-D18, D22).

Reads results/raw/main/<run_id>/ and writes tables to results/processed/.

Log regret L = log10(max(regret, floor)), floor 1e-8 (D14). Replicates are
paired: within a replicate, every method faces the same instance, and the
aligned and rotated instances share x_opt (D12). Uncertainty is a paired
percentile bootstrap over replicates (D16; Efron & Hastie, 2016).

Tables (results/processed/):
  primary.csv                 the two primary DiDs (D15), plus floor sensitivity
  rotation_penalty_checkpoints.csv  secondary: penalty per checkpoint at kappa = 1e4 (D22.1)
  rotation_penalty_kappa.csv  secondary: penalty and diagonal-vs-full DiD per kappa (D22.2)
  kappa_j.csv                 secondary: Spearman kappa_J vs diagonal-minus-full gap (D22.3)
  invariance.csv              kappa = 1 control (D17)
  bo_isotropic_vs_ard.csv     exploratory (D22.4)
  cost.csv                    exploratory: wall clock, BO overhead, break-even (D22.5)
  stops.csv                   stop categories and bound repairs per condition (D10)
  kappa_eff.csv               CMA-ES covariance diagnostic (D18)
  lengthscales.csv            BO ARD lengthscale diagnostic (D18)
  uniform_law.csv             uniform search vs 1 - (1 - q)^N (D6)
  regret_curves.csv           median log regret per method, condition, checkpoint
  analysis_report.json        headline numbers

Usage:
    python code/analysis.py                                   # results/raw/main/main
    python code/analysis.py --run-dir results/raw/main/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, check_ready, load_config  # noqa: E402
from objectives import ellipsoid, ellipsoid_weights  # noqa: E402

REFERENCE = "cma_es_full"
AXIS_ALIGNED = ("cma_es_diagonal", "bo_ard")


# ----------------------------------------------------------------------
# Core statistics (unit-tested)
# ----------------------------------------------------------------------

def log_regret(values, floor: float) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def bootstrap_mean_ci(x, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling the units in x."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_L(s: pd.DataFrame, method: str, kappa: float, checkpoint: int, floor: float) -> pd.DataFrame:
    """Log regret per replicate, columns aligned / rotated, for one method and kappa."""
    g = s[(s.method == method) & (s.kappa == kappa)]
    t = g.pivot(index="replicate", columns="orientation", values=f"regret_at_{checkpoint}")
    return t.apply(lambda col: log_regret(col, floor))


def rotation_penalty(s, method, kappa, checkpoint, floor, n_boot, seed, level=0.95) -> dict:
    """Mean of L_rotated - L_aligned over replicates, with bootstrap CI."""
    t = paired_L(s, method, kappa, checkpoint, floor).dropna()
    est, lo, hi = bootstrap_mean_ci(t["rotated"] - t["aligned"], n_boot, seed, level)
    return {"method": method, "kappa": kappa, "checkpoint": checkpoint, "floor": floor,
            "estimate": est, "ci_low": lo, "ci_high": hi, "n_replicates": len(t)}


def did(s, method, kappa, checkpoint, floor, n_boot, seed, level=0.95, reference=REFERENCE) -> dict:
    """Paired DiD: (rot - align)_M - (rot - align)_reference, per replicate (D15)."""
    m = paired_L(s, method, kappa, checkpoint, floor)
    r = paired_L(s, reference, kappa, checkpoint, floor)
    both = m.join(r, lsuffix="_m", rsuffix="_r", how="inner").dropna()
    contrib = (both.rotated_m - both.aligned_m) - (both.rotated_r - both.aligned_r)
    est, lo, hi = bootstrap_mean_ci(contrib, n_boot, seed, level)
    return {"method": method, "reference": reference, "kappa": kappa, "checkpoint": checkpoint,
            "floor": floor, "did": est, "ci_low": lo, "ci_high": hi, "n_replicates": len(both),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def spearman_with_ci(x, y, n_boot: int, seed: int, level: float = 0.95) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(x[i], y[i])[0])
    a = (1 - level) / 2
    lo, hi = (np.quantile(boots, a), np.quantile(boots, 1 - a)) if boots else (np.nan, np.nan)
    return {"rho": float(rho), "p_value": float(p), "ci_low": float(lo), "ci_high": float(hi), "n": len(x)}


def evals_to_target(best_so_far: np.ndarray, target: float) -> float:
    """First evaluation (1-based) at which best-so-far regret <= target; inf if never."""
    hit = np.nonzero(np.asarray(best_so_far) <= target)[0]
    return float(hit[0] + 1) if len(hit) else np.inf


def uniform_success_probability(q: float, n: int) -> float:
    """P(at least one success in n independent uniform draws) = 1 - (1 - q)^n (Bergstra & Bengio, 2012)."""
    return 1.0 - (1.0 - q) ** n


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def analyse(run_dir: Path, out_dir: Path, config_path: Path = DEFAULT_CONFIG_PATH,
            n_boot: int | None = None, log=print) -> dict:
    cfg = load_config(config_path, stage="main")
    check_ready(cfg, "analysis")
    info = json.loads((run_dir / "run_info.json").read_text())
    s = pd.read_csv(run_dir / "summary.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    floor = float(cfg["analysis"]["log_regret_floor"])
    sens_floors = [float(f) for f in cfg["analysis"]["floor_sensitivity"]]
    unc = cfg["analysis"]["uncertainty"]
    B = int(n_boot or unc["bootstrap_replicates"])
    seed = int(unc.get("bootstrap_seed", 4212))
    level = float(unc["confidence_level"])
    tol = float(cfg["analysis"]["invariance_tolerance"]["value"])
    budget = int(info["budget"])
    checkpoints = [int(c) for c in info["checkpoints"]]
    bo_ceiling = int(info["bo_ceiling"])
    kmax = float(cfg["analysis"]["primary_comparisons"]["condition_number"])
    methods = sorted(s.method.unique())
    bo_methods = [m for m in methods if m.startswith("bo_")]
    non_bo = [m for m in methods if not m.startswith("bo_")]
    kappas = sorted(s.kappa.unique())
    cps_for = lambda m: [c for c in checkpoints if (c <= bo_ceiling if m.startswith("bo_") else True)]
    report = {"run_id": info["run_id"], "floor": floor, "bootstrap_replicates": B,
              "budget": budget, "bo_ceiling": bo_ceiling}

    # Primary (D15) + floor sensitivity (D14) + diagonal at the BO ceiling (secondary)
    rows = []
    for f in [floor] + sens_floors:
        role = "primary" if f == floor else "floor_sensitivity"
        rows.append({"role": role, **did(s, "cma_es_diagonal", kmax, budget, f, B, seed, level)})
        if "bo_ard" in methods:
            rows.append({"role": role, **did(s, "bo_ard", kmax, bo_ceiling, f, B, seed, level)})
    rows.append({"role": "secondary_shared_checkpoint",
                 **did(s, "cma_es_diagonal", kmax, bo_ceiling, floor, B, seed, level)})
    primary = pd.DataFrame(rows)
    primary.to_csv(out_dir / "primary.csv", index=False)
    report["primary"] = primary[primary.role == "primary"].to_dict("records")

    # Secondary 1: penalty across checkpoints at kmax
    pen_cp = pd.DataFrame([rotation_penalty(s, m, kmax, c, floor, B, seed, level)
                           for m in methods for c in cps_for(m)])
    pen_cp.to_csv(out_dir / "rotation_penalty_checkpoints.csv", index=False)

    # Secondary 2: penalty and diagonal-vs-full DiD across kappa
    pen_k = []
    for m in methods:
        c = bo_ceiling if m.startswith("bo_") else budget
        for k in sorted(s[s.method == m].kappa.unique()):
            pen_k.append({"quantity": "rotation_penalty", **rotation_penalty(s, m, k, c, floor, B, seed, level)})
    for k in kappas:
        d_ = did(s, "cma_es_diagonal", k, budget, floor, B, seed, level)
        pen_k.append({"quantity": "did_diagonal_vs_full", "method": "cma_es_diagonal", "kappa": k,
                      "checkpoint": budget, "floor": floor, "estimate": d_["did"],
                      "ci_low": d_["ci_low"], "ci_high": d_["ci_high"], "n_replicates": d_["n_replicates"]})
    pd.DataFrame(pen_k).to_csv(out_dir / "rotation_penalty_kappa.csv", index=False)

    # Secondary 3: kappa_J vs diagonal-minus-full gap, rotated instances
    kj_rows = []
    for k in [k for k in kappas if k > 1]:
        dg = paired_L(s, "cma_es_diagonal", k, budget, floor)["rotated"]
        fl = paired_L(s, "cma_es_full", k, budget, floor)["rotated"]
        kj = s[(s.method == "cma_es_full") & (s.kappa == k) & (s.orientation == "rotated")] \
            .set_index("replicate")["kappa_J"]
        j = pd.DataFrame({"gap": dg - fl, "kappa_J": kj}).dropna()
        kj_rows.append({"kappa": k, "role": "pre-registered" if k == kmax else "supporting",
                        **spearman_with_ci(j.kappa_J, j.gap, min(B, 2000), seed, level)})
    pd.DataFrame(kj_rows).to_csv(out_dir / "kappa_j.csv", index=False)

    # Invariance control (D17)
    inv = []
    for m in methods:
        c = bo_ceiling if m.startswith("bo_") else budget
        r = rotation_penalty(s, m, 1.0, c, floor, B, seed, level)
        inv.append({**r, "aligned_minus_rotated": -r["estimate"],
                    "within_tolerance": bool(-r["ci_high"] >= -tol and -r["ci_low"] <= tol), "tolerance": tol})
    pd.DataFrame(inv).to_csv(out_dir / "invariance.csv", index=False)

    # Exploratory: isotropic vs ARD BO
    if {"bo_ard", "bo_isotropic"} <= set(methods):
        ex = [{"comparison": "rotation_penalty", **rotation_penalty(s, m, kmax, bo_ceiling, floor, B, seed, level)}
              for m in ("bo_ard", "bo_isotropic")]
        d_ = did(s, "bo_ard", kmax, bo_ceiling, floor, B, seed, level, reference="bo_isotropic")
        ex.append({"comparison": "did_ard_vs_isotropic", "method": "bo_ard", "kappa": kmax,
                   "checkpoint": bo_ceiling, "floor": floor, "estimate": d_["did"],
                   "ci_low": d_["ci_low"], "ci_high": d_["ci_high"], "n_replicates": d_["n_replicates"]})
        pd.DataFrame(ex).to_csv(out_dir / "bo_isotropic_vs_ard.csv", index=False)

    # Exploratory: cost and break-even (D22.5)
    cost = []
    for (m, k, o), g in s.groupby(["method", "kappa", "orientation"]):
        row = {"method": m, "kappa": k, "orientation": o, "median_wall_time_s": g.wall_time_s.median(),
               "median_wall_time_per_eval_s": (g.wall_time_s / g.evals_used).median()}
        if "bo_overhead_s" in g and g.bo_overhead_s.notna().any():
            row["median_bo_overhead_s"] = g.bo_overhead_s.median()
        cost.append(row)
    cost = pd.DataFrame(cost)
    be = []
    ev_dir = run_dir / "evaluations"
    for m in bo_methods:
        for k in sorted(s[s.method == m].kappa.unique()):
            ev_full = pd.read_parquet(ev_dir / f"cma_es_full_k{int(k)}.parquet")
            for o in ("aligned", "rotated"):
                gb = s[(s.method == m) & (s.kappa == k) & (s.orientation == o)]
                gf = s[(s.method == REFERENCE) & (s.kappa == k) & (s.orientation == o)]
                target = float(gb[f"regret_at_{bo_ceiling}"].median())   # what BO reaches at its ceiling
                ef = ev_full[ev_full.orientation == o]
                n_cma = np.median([evals_to_target(r.best_so_far.values, target)
                                   for _, r in ef.groupby("replicate")])
                o_bo = float(gb.bo_overhead_s.median())
                o_cma = float((gf.wall_time_s / gf.evals_used).median() * min(n_cma, budget))
                be_cost = (o_bo - o_cma) / (n_cma - bo_ceiling) if np.isfinite(n_cma) and n_cma > bo_ceiling else np.nan
                be.append({"method": m, "kappa": k, "orientation": o, "target_regret": target,
                           "bo_evals": bo_ceiling, "full_cma_median_evals_to_target": n_cma,
                           "bo_overhead_s": o_bo, "full_cma_overhead_s": o_cma,
                           "break_even_cost_per_eval_s": be_cost})
    cost.to_csv(out_dir / "cost.csv", index=False)
    pd.DataFrame(be).to_csv(out_dir / "break_even.csv", index=False)

    # Stops and bound repairs (D10)
    stops = s.groupby(["method", "kappa", "orientation", "stop_category"]).size().rename("n").reset_index()
    rep = s.assign(repair_fraction=s.n_repaired / s.evals_used).groupby(
        ["method", "kappa", "orientation"]).repair_fraction.median().rename("median_repair_fraction").reset_index()
    stops.merge(rep, on=["method", "kappa", "orientation"]).to_csv(out_dir / "stops.csv", index=False)

    # kappa_eff (D18)
    cov_path = run_dir / "covariance_snapshots.jsonl"
    if cov_path.exists() and cov_path.stat().st_size:
        cov = pd.read_json(cov_path, lines=True)
        ke = cov.groupby(["method", "kappa", "orientation", "checkpoint"]).agg(
            median_kappa_eff=("kappa_eff", "median"),
            q25=("kappa_eff", lambda v: v.quantile(0.25)), q75=("kappa_eff", lambda v: v.quantile(0.75)),
            median_evals_at_snapshot=("evals_at_snapshot", "median")).reset_index()
        ke.to_csv(out_dir / "kappa_eff.csv", index=False)

    # BO lengthscales (D18): Spearman(lengthscale, axis weight); aligned should be near -1
    bo_path = run_dir / "bo_diagnostics.jsonl"
    if bo_path.exists() and bo_path.stat().st_size:
        ls_rows = []
        for line in bo_path.read_text().splitlines():
            r = json.loads(line)
            if r["method"] != "bo_ard":
                continue
            state = r["checkpoints"].get(str(bo_ceiling)) or r["checkpoints"].get(bo_ceiling)
            if not state:
                continue
            w = ellipsoid_weights(r["d"], r["kappa"])
            rho = spearmanr(state["length_scale"], w)[0] if r["kappa"] > 1 else np.nan
            ls_rows.append({"kappa": r["kappa"], "orientation": r["orientation"], "replicate": r["replicate"],
                            "spearman_lengthscale_vs_weight": rho,
                            "lengthscales": state["length_scale"]})
        ls = pd.DataFrame(ls_rows)
        ls.to_csv(out_dir / "lengthscales.csv", index=False)
        report["lengthscale_median_rho"] = ls.groupby(["kappa", "orientation"]) \
            .spearman_lengthscale_vs_weight.median().dropna().to_dict()
        report["lengthscale_median_rho"] = {f"{k[0]:g}_{k[1]}": v for k, v in report["lengthscale_median_rho"].items()}

    # Uniform search vs 1 - (1 - q)^N (D6)
    tau = float(cfg["diagnostics"]["random_search"]["success_criterion"])
    n_mc = int(cfg["diagnostics"]["random_search"]["large_uniform_sample_size"])
    inst = json.loads((run_dir / "instances.json").read_text())
    lower, upper = cfg["synthetic"]["domain"]["lower_bound"], cfg["synthetic"]["domain"]["upper_bound"]
    rng = np.random.default_rng(seed)
    q_of = {}
    for i in inst:
        X = rng.uniform(lower, upper, size=(n_mc, i["d"]))
        q_of[(i["kappa"], i["orientation"], i["replicate"])] = float(
            np.mean(ellipsoid(X, i["kappa"], np.array(i["R"]), np.array(i["x_opt"])) <= tau))
    uni = []
    gu = s[s.method == "uniform_random_search"]
    for (k, o), g in gu.groupby(["kappa", "orientation"]):
        for c in checkpoints:
            pred = np.mean([uniform_success_probability(q_of[(k, o, r)], c) for r in g.replicate])
            uni.append({"kappa": k, "orientation": o, "checkpoint": c, "success_criterion": tau,
                        "observed_fraction": float((g[f"regret_at_{c}"] <= tau).mean()),
                        "predicted_probability": float(pred), "n_runs": len(g)})
    pd.DataFrame(uni).to_csv(out_dir / "uniform_law.csv", index=False)

    # Median log-regret curves (for figures)
    curves = []
    for (m, k, o), g in s.groupby(["method", "kappa", "orientation"]):
        for c in cps_for(m):
            L = log_regret(g[f"regret_at_{c}"], floor)
            curves.append({"method": m, "kappa": k, "orientation": o, "checkpoint": c,
                           "median_log_regret": float(np.median(L)),
                           "q25": float(np.quantile(L, 0.25)), "q75": float(np.quantile(L, 0.75))})
    pd.DataFrame(curves).to_csv(out_dir / "regret_curves.csv", index=False)

    report["n_runs"] = int(len(s))
    report["stop_categories"] = s.stop_category.value_counts().to_dict()
    (out_dir / "analysis_report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"Analysis written to {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "results" / "raw" / "main" / "main")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "processed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = analyse(args.run_dir, args.out, args.config)
    for r in report["primary"]:
        print(f"  DiD {r['method']} vs {r['reference']} at {r['checkpoint']}: "
              f"{r['did']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")


if __name__ == "__main__":
    main()