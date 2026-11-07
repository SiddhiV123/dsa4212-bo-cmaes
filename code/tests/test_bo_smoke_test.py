"""Tests for code/bo_smoke_test.py (decision log D21 smoke-test rules)."""

import json
import math

import pytest

pytest.importorskip("sklearn")

from bo_smoke_test import CEILING_OPTIONS, SAFETY_MARGIN, choose_ceiling, projected_hours, run_smoke


def test_ceiling_options_order():
    assert CEILING_OPTIONS == ((250, "standard"), (250, "random"), (90, "standard"), (90, "random"))


def test_projection_formula():
    assert projected_hours(60, 400, 8) == pytest.approx(60 * 400 / 8 * SAFETY_MARGIN / 3600)


def test_choose_ceiling_takes_first_option_within_cap():
    proj = {(250, "standard"): 3.0, (250, "random"): 1.5, (90, "standard"): 0.4}
    assert choose_ceiling(proj, cap_hours=2.0)[:2] == (250, "random")
    assert choose_ceiling({(250, "standard"): 1.0}, cap_hours=2.0)[:2] == (250, "standard")


def test_choose_ceiling_flags_when_nothing_fits():
    proj = {opt: 9.0 for opt in CEILING_OPTIONS}
    ceiling, setting, hours, within = choose_ceiling(proj, cap_hours=2.0)
    assert (ceiling, setting) == CEILING_OPTIONS[-1] and within is False and hours == 9.0


def test_quick_smoke_run_end_to_end(tmp_path):
    r = run_smoke(out_root=tmp_path, run_id="t", quick=True, workers=2, log=lambda *a: None)
    assert r["n_init"] == 8
    assert r["check2_bo_runs"]["completed"] == r["check2_bo_runs"]["total"]
    assert r["ceiling"]["chosen"] in (250, 90)
    assert math.isfinite(r["ceiling"]["projected_hours"])
    saved = json.loads((tmp_path / "t" / "smoke_report.json").read_text())
    assert saved["quick"] is True and saved["n_main_bo_runs"] == 400