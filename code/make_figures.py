"""Make every figure in the report from results/processed/ (and the main-run summary).

Usage:
    python code/make_figures.py
    python code/make_figures.py --processed results/processed --run-dir results/raw/main/main

Writes PNG and PDF to results/figures/ and copies them to report/figures/.

Figures:
  fig1_primary_did            primary DiDs with 95% CIs, plus floor sensitivity (D15, D14)
  fig2_regret_curves          median log regret vs evaluations, kappa in {1, 1e4} x orientation
  fig3_penalty_checkpoints    rotation penalty vs evaluations at kappa = 1e4 (D22.1)
  fig4_penalty_kappa          rotation penalty vs kappa (D22.2)
  fig5_kappa_j                diagonal-minus-full gap vs kappa_J, rotated instances (D22.3)
  fig6_kappa_eff              CMA-ES kappa_eff vs evaluations (D18)
  fig7_lengthscales           BO ARD lengthscales by axis, aligned vs rotated (D18)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import PROJECT_ROOT  

LABELS = {"cma_es_full": "Full CMA-ES", "cma_es_diagonal": "Diagonal CMA-ES", "isotropic_es": "Isotropic ES",
          "bo_ard": "BO (ARD)", "bo_isotropic": "BO (isotropic)", "uniform_random_search": "Uniform search"}
COLOURS = {"cma_es_full": "#1b9e77", "cma_es_diagonal": "#d95f02", "isotropic_es": "#7570b3",
           "bo_ard": "#e7298a", "bo_isotropic": "#66a61e", "uniform_random_search": "#666666"}
ORDER = ["cma_es_full", "cma_es_diagonal", "isotropic_es", "bo_ard", "bo_isotropic", "uniform_random_search"]


def _save(fig, name: str, out_dirs: list[Path]) -> list[Path]:
    paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            p = d / f"{name}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            paths.append(p)
    plt.close(fig)
    return paths


def _methods(df: pd.DataFrame) -> list[str]:
    return [m for m in ORDER if m in set(df.method)]


def fig_primary(processed: Path, out_dirs):
    p = pd.read_csv(processed / "primary.csv")
    p = p[p.role.isin(["primary", "floor_sensitivity", "secondary_shared_checkpoint"])].reset_index(drop=True)
    labels = [f"{LABELS[r.method]} vs full, {r.checkpoint} evals, floor {r.floor:g}"
              + ("" if r.role == "primary" else f" ({r.role.replace('_', ' ')})") for r in p.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(p) + 1))
    y = np.arange(len(p))[::-1]
    for yi, r in zip(y, p.itertuples()):
        ax.errorbar(r.did, yi, xerr=[[r.did - r.ci_low], [r.ci_high - r.did]], fmt="o",
                    color=COLOURS[r.method], mfc="white" if r.role != "primary" else COLOURS[r.method], capsize=3)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("DiD in mean log10 regret (positive: rotation hurts the method more than full CMA-ES)")
    ax.set_title("Primary comparisons, kappa = 1e4 (95% paired bootstrap CI)", fontsize=9)
    return _save(fig, "fig1_primary_did", out_dirs)


def fig_regret_curves(processed: Path, out_dirs, floor: float):
    c = pd.read_csv(processed / "regret_curves.csv")
    kappas = [k for k in (1.0, c.kappa.max()) if k in set(c.kappa)]
    fig, axes = plt.subplots(len(kappas), 2, figsize=(9, 3.2 * len(kappas)), sharey=True, squeeze=False)
    for i, k in enumerate(kappas):
        for j, o in enumerate(("aligned", "rotated")):
            ax = axes[i, j]
            for m in _methods(c):
                g = c[(c.method == m) & (c.kappa == k) & (c.orientation == o)].sort_values("checkpoint")
                if g.empty:
                    continue
                ax.plot(g.checkpoint, g.median_log_regret, marker="o", ms=3, color=COLOURS[m], label=LABELS[m])
                ax.fill_between(g.checkpoint, g.q25, g.q75, color=COLOURS[m], alpha=0.15)
            ax.axhline(np.log10(floor), color="grey", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_title(f"kappa = {k:g}, {o}", fontsize=9)
            ax.set_xlabel("evaluations")
        axes[i, 0].set_ylabel("median log10 regret (IQR band)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig2_regret_curves", out_dirs)


def fig_penalty_checkpoints(processed: Path, out_dirs):
    p = pd.read_csv(processed / "rotation_penalty_checkpoints.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(_methods(p)):
        g = p[p.method == m].sort_values("checkpoint")
        x = g.checkpoint * (1 + 0.03 * (i - 2.5))           # small dodge
        ax.errorbar(x, g.estimate, yerr=[g.estimate - g.ci_low, g.ci_high - g.estimate],
                    marker="o", ms=4, capsize=2, color=COLOURS[m], label=LABELS[m])
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("rotation penalty: mean L_rotated - L_aligned")
    ax.set_title(f"Rotation penalty over the budget, kappa = {p.kappa.iloc[0]:g}", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, "fig3_penalty_checkpoints", out_dirs)


def fig_penalty_kappa(processed: Path, out_dirs):
    p = pd.read_csv(processed / "rotation_penalty_kappa.csv")
    p = p[p.quantity == "rotation_penalty"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in _methods(p):
        g = p[p.method == m].sort_values("kappa")
        ax.errorbar(g.kappa, g.estimate, yerr=[g.estimate - g.ci_low, g.ci_high - g.estimate],
                    marker="o", ms=4, capsize=2, color=COLOURS[m],
                    label=f"{LABELS[m]} ({int(g.checkpoint.iloc[0])} evals)")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("condition number kappa")
    ax.set_ylabel("rotation penalty: mean L_rotated - L_aligned")
    ax.set_title("Rotation penalty against conditioning", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, "fig4_penalty_kappa", out_dirs)


def fig_kappa_j(processed: Path, run_dir: Path, out_dirs, floor: float, budget: int):
    s = pd.read_csv(run_dir / "summary.csv")
    rho = pd.read_csv(processed / "kappa_j.csv").set_index("kappa")
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in [k for k in sorted(s.kappa.unique()) if k > 1]:
        r = s[(s.kappa == k) & (s.orientation == "rotated")]
        d = r[r.method == "cma_es_diagonal"].set_index("replicate")
        f = r[r.method == "cma_es_full"].set_index("replicate")
        gap = np.log10(np.maximum(d[f"regret_at_{budget}"], floor)) - np.log10(np.maximum(f[f"regret_at_{budget}"], floor))
        lab = f"kappa = {k:g} (rho = {rho.loc[k, 'rho']:.2f})" if k in rho.index else f"kappa = {k:g}"
        ax.scatter(f.loc[gap.index, "kappa_J"], gap, s=12, alpha=0.7, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("kappa_J of the rotated instance")
    ax.set_ylabel("log10 regret gap: diagonal - full CMA-ES")
    ax.set_title(f"Conditioning a diagonal model cannot remove ({budget} evals)", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, "fig5_kappa_j", out_dirs)


def fig_kappa_eff(processed: Path, out_dirs):
    path = processed / "kappa_eff.csv"
    if not path.exists():
        return []
    k = pd.read_csv(path)
    kmax = k.kappa.max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, o in zip(axes, ("aligned", "rotated")):
        for m in _methods(k):
            g = k[(k.method == m) & (k.kappa == kmax) & (k.orientation == o)].sort_values("checkpoint")
            ax.plot(g.median_evals_at_snapshot, g.median_kappa_eff, marker="o", ms=3, color=COLOURS[m], label=LABELS[m])
            ax.fill_between(g.median_evals_at_snapshot, g.q25, g.q75, color=COLOURS[m], alpha=0.15)
        ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"kappa = {kmax:g}, {o}", fontsize=9)
        ax.set_xlabel("evaluations (true snapshot count)")
    axes[0].set_ylabel("kappa_eff (1 = covariance matches inverse Hessian)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig6_kappa_eff", out_dirs)


def fig_lengthscales(processed: Path, out_dirs):
    path = processed / "lengthscales.csv"
    if not path.exists():
        return []
    ls = pd.read_csv(path)
    ls = ls[ls.kappa == ls.kappa.max()]
    if ls.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, o in zip(axes, ("aligned", "rotated")):
        vals = np.array([json.loads(v) if isinstance(v, str) else v for v in ls[ls.orientation == o].lengthscales])
        if len(vals):
            ax.boxplot([vals[:, i] for i in range(vals.shape[1])])
        ax.set_yscale("log")
        ax.set_xlabel("axis (weight increases left to right)")
        ax.set_title(f"BO (ARD) lengthscales, {o}", fontsize=9)
    axes[0].set_ylabel("fitted lengthscale (unit box)")
    fig.tight_layout()
    return _save(fig, "fig7_lengthscales", out_dirs)


def make_all(processed: Path, run_dir: Path, figures: Path, report_figures: Path, floor: float = 1e-8) -> list[Path]:
    info = json.loads((run_dir / "run_info.json").read_text())
    out_dirs = [figures, report_figures]
    paths = []
    paths += fig_primary(processed, out_dirs)
    paths += fig_regret_curves(processed, out_dirs, floor)
    paths += fig_penalty_checkpoints(processed, out_dirs)
    paths += fig_penalty_kappa(processed, out_dirs)
    paths += fig_kappa_j(processed, run_dir, out_dirs, floor, int(info["budget"]))
    paths += fig_kappa_eff(processed, out_dirs)
    paths += fig_lengthscales(processed, out_dirs)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "results" / "processed")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "results" / "raw" / "main" / "main")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--report-figures", type=Path, default=PROJECT_ROOT / "report" / "figures")
    args = parser.parse_args()
    paths = make_all(args.processed, args.run_dir, args.figures, args.report_figures)
    print(f"{len(paths)} files written, e.g. {paths[0]}")


if __name__ == "__main__":
    main()