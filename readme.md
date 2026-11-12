# When Does Geometry-Aware Search Pay Off?

DSA4212 individual project, Siddhi Verma. The report is in `report/`.

## Environment and Dependencies

Requires **Python 3.12.4**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used: numpy, scipy, pandas, matplotlib, scikit-learn, cma (pycma 4.5.0), pyyaml, pyarrow and pytest. The experiments were run on macOS on a 10-core Apple Silicon laptop; the run scripts limit each process to one BLAS thread so that parallel runs do not compete for cores.

The report (`report/main.tex`) compiles with pdfLaTeX and Biber, the default on Overleaf.

## Data Preparation

No external data are used, so there is no download or preprocessing step. Every problem instance (optimum and rotation matrix) is generated deterministically from the base seed `4212` in `configs/experiment.yaml`, and is saved with the results in `results/raw/main/main/instances.json`. The pilot, smoke test and main runs use separate seed streams, so their instances never overlap.

## Reproducing the Main Results

All commands run from the repository root with the environment activated.

### Option A: from the saved results (about 5 minutes)

The raw main-run results are committed, so the report's tables and figures can be regenerated without re-running any optimiser:

```bash
python -m pytest code/tests        # 172 tests, about 1 minute
python code/analysis.py            # writes the result tables to results/processed/
python code/make_figures.py        # writes the figures to results/figures/ and report/figures/
```

`analysis.py` prints the two primary results, which should read:

```text
  DiD cma_es_diagonal vs cma_es_full at 2000: +8.60 [+8.32, +8.84]
  DiD bo_ard vs cma_es_full at 250: -0.04 [-0.27, +0.21]
```

### Option B: from scratch (about 2 hours)

Each stage reads its settings from `configs/experiment.yaml` and writes to a new folder, so earlier results are never overwritten.

```bash
python code/run_pilot.py                  # pilot, about 15 s
python code/bo_smoke_test.py              # BO smoke test, about 15 min
python code/run_main.py --run-id rerun    # main runs, about 1.5-2 h on 8 workers; resumable
python code/analysis.py --run-dir results/raw/main/rerun --out results/processed_rerun
python code/make_figures.py --processed results/processed_rerun --run-dir results/raw/main/rerun \
    --figures results/figures_rerun --report-figures results/figures_rerun_report
```

On macOS, prefix long runs with `caffeinate -i` to prevent sleep. If `run_main.py` is interrupted, run the same command again: finished tasks are skipped.

CMA-ES and uniform-search results are deterministic given the seeds. BO fits Gaussian processes with floating-point linear algebra, so on a different machine its results may differ slightly. Wall-clock times depend on the machine.

### Where the report's results come from

| In the report | File in `results/processed/` | Figure file |
|---|---|---|
| Table 2 | `primary.csv` | — |
| Figure 1 | `regret_curves.csv` | `fig2_regret_curves.pdf` |
| Figure 2 | `rotation_penalty_checkpoints.csv` | `fig3_penalty_checkpoints.pdf` |
| Figure 3 | `kappa_eff.csv` | `fig6_kappa_eff.pdf` |
| Section 4.2 (conditioning) | `rotation_penalty_kappa.csv` | — |
| Section 4.3 | `kappa_j.csv`, `lengthscales.csv` | — |
| Section 4.4 | `invariance.csv`, `stops.csv`, `uniform_law.csv` | — |
| Section 5 | `bo_isotropic_vs_ard.csv`, `cost.csv`, `break_even.csv` | — |