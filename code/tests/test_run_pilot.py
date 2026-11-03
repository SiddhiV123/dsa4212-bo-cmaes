"""Smoke test for code/run_pilot.py (quick mode, temporary output folder)."""

import json

import pandas as pd
import pytest

pytest.importorskip("cma")
pytest.importorskip("pyarrow")

from run_pilot import plot_regret, run_pilot


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return run_pilot(out_root=tmp_path_factory.mktemp("pilot"), quick=True, run_id="test")


def test_pilot_writes_all_outputs(out_dir):
    for name in ["summary.csv", "evaluations.parquet", "instances.json",
                 "covariance_snapshots.json", "config_resolved.yaml", "run_info.json"]:
        assert (out_dir / name).exists(), name


def test_pilot_runs_every_condition_and_method(out_dir):
    s = pd.read_csv(out_dir / "summary.csv")
    # 2 kappas x 2 orientations x 1 replicate x 4 methods in quick mode
    assert len(s) == 16
    assert set(s["stage"]) == {"pilot"}
    assert (s["evals_used"] <= s["budget"]).all()
    assert set(s["stop_category"]) <= {"budget", "numerical_precision"}


def test_pilot_evaluation_log_matches_summary(out_dir):
    s = pd.read_csv(out_dir / "summary.csv")
    e = pd.read_parquet(out_dir / "evaluations.parquet")
    assert len(e) == s["evals_used"].sum()


def test_pilot_aligned_and_rotated_share_instances(out_dir):
    s = pd.read_csv(out_dir / "summary.csv")
    seeds = s.groupby(["kappa", "replicate"])["problem_seed"].nunique()
    assert (seeds == 1).all()


def test_pilot_records_software_versions(out_dir):
    info = json.loads((out_dir / "run_info.json").read_text())
    assert info["stage"] == "pilot" and "cma" in info["software"]


def test_pilot_plot(out_dir):
    assert plot_regret(out_dir).exists()


def test_pilot_refuses_to_overwrite(out_dir):
    with pytest.raises(FileExistsError):
        run_pilot(out_root=out_dir.parent, quick=True, run_id="test")