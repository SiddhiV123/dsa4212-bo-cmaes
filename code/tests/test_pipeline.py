"""End-to-end test: quick main run -> analysis -> figures, in a temporary folder."""

import json

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from analysis import analyse
from config_loader import ConfigError, check_ready, load_config
from make_figures import make_all
from run_main import build_tasks, quick_config, run_main, task_id


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("main")
    run_main(out_root=root, run_id="q", quick=True, workers=2, only="cma", log=lambda *a: None)
    # Not yet consolidated: BO tasks remain. Resuming with BO completes it.
    assert not (root / "q" / "summary.csv").exists()
    return run_main(out_root=root, run_id="q", quick=True, workers=2, only="bo", log=lambda *a: None)


def test_main_stage_requires_bo_settings():
    with pytest.raises(ConfigError, match="not ready"):
        check_ready(load_config(stage="main"), "main")


def test_analysis_stage_ready():
    check_ready(load_config(stage="main"), "analysis")


def test_quick_config_is_ready():
    check_ready(quick_config(load_config(stage="main")), "main")


def test_tasks_are_unique_and_bo_first():
    tasks = build_tasks(quick_config(load_config(stage="main")))
    ids = [task_id(t) for t in tasks]
    assert len(ids) == len(set(ids)) and tasks[0]["method"].startswith("bo_")


def test_main_run_outputs(run_dir):
    s = pd.read_csv(run_dir / "summary.csv")
    assert len(s) == 72 and set(s.stage) == {"main"}
    assert not (run_dir / "parts").exists()
    for f in ["instances.json", "covariance_snapshots.jsonl", "bo_diagnostics.jsonl",
              "config_resolved.yaml", "run_info.json"]:
        assert (run_dir / f).exists(), f
    ev = list((run_dir / "evaluations").glob("*.parquet"))
    assert len(ev) == s.groupby(["method", "kappa"]).ngroups
    assert sum(len(pd.read_parquet(p)) for p in ev) == s.evals_used.sum()


def test_main_seeds_differ_from_pilot(run_dir):
    s = pd.read_csv(run_dir / "summary.csv")
    pilot_seed = __import__("runner").problem_seed(4212, 1_000_000, 4, 1.0, 0)
    assert pilot_seed not in set(s.problem_seed)


def test_bo_runs_stop_at_ceiling_and_share_instances(run_dir):
    s = pd.read_csv(run_dir / "summary.csv")
    bo = s[s.method.str.startswith("bo_")]
    assert (bo.evals_used == json.loads((run_dir / "run_info.json").read_text())["bo_ceiling"]).all()
    seeds = s.groupby(["kappa", "replicate"]).problem_seed.nunique()
    assert (seeds == 1).all()


def test_refuses_to_rerun_consolidated(run_dir):
    with pytest.raises(FileExistsError):
        run_main(out_root=run_dir.parent, run_id="q", quick=True, workers=1, log=lambda *a: None)


def test_analysis_and_figures(run_dir, tmp_path):
    report = analyse(run_dir, tmp_path / "processed", n_boot=200, log=lambda *a: None)
    assert {r["method"] for r in report["primary"]} == {"cma_es_diagonal", "bo_ard"}
    for f in ["primary.csv", "rotation_penalty_checkpoints.csv", "rotation_penalty_kappa.csv",
              "kappa_j.csv", "invariance.csv", "cost.csv", "break_even.csv", "stops.csv",
              "kappa_eff.csv", "lengthscales.csv", "uniform_law.csv", "regret_curves.csv"]:
        assert (tmp_path / "processed" / f).exists(), f
    inv = pd.read_csv(tmp_path / "processed" / "invariance.csv").set_index("method")
    # kappa = 1: rank-based methods are identical to machine precision; BO uses
    # function values, so last-digit rounding differences can shift it slightly.
    rank_based = ["cma_es_full", "cma_es_diagonal", "isotropic_es", "uniform_random_search"]
    assert (inv.loc[rank_based, "aligned_minus_rotated"].abs() < 1e-9).all()
    assert inv.within_tolerance.all()
    paths = make_all(tmp_path / "processed", run_dir, tmp_path / "fig", tmp_path / "rfig")
    assert len({p.stem for p in paths}) == 7