"""Tests for loading and resolving the experiment configuration."""

from copy import deepcopy

import pytest

from config_loader import (
    ConfigError,
    load_config,
    resolve_cma_variants,
    validate_config_structure,
)


def test_load_real_experiment_config():
    """The project's YAML loads and its project metadata is available."""
    config = load_config()

    assert config["project"]["course"] == "DSA4212"
    assert config["project"]["proposal_version"] == "v9"


def test_all_cma_variants_inherit_common_settings():
    """Shared CMA settings appear in all three resolved variants."""
    config = load_config()
    common = config["methods"]["cma_es_common"]

    for name in ("cma_es_full", "cma_es_diagonal", "isotropic_es"):
        variant = config["methods"][name]

        assert variant["library"] == common["library"]
        assert (
            variant["initial_point_policy"]
            == common["initial_point_policy"]
        )
        assert variant["stopping"] == common["stopping"]
        assert "inherits" not in variant


def test_cma_variants_keep_their_specific_settings():
    """Resolving inheritance must not overwrite variant-specific choices."""
    config = load_config()
    methods = config["methods"]

    assert methods["cma_es_full"]["covariance_mode"] == "full"

    assert methods["cma_es_diagonal"]["covariance_mode"] == "diagonal"
    assert methods["cma_es_diagonal"]["options"]["CMA_diagonal"] is True

    assert methods["isotropic_es"]["covariance_mode"] == "isotropic"
    assert methods["isotropic_es"]["options"]["CMA_on"] == 0


def test_nested_dictionaries_merge_recursively():
    """Child options override shared values without losing other keys."""
    config = {
        "methods": {
            "cma_es_common": {
                "library": "cma",
                "options": {
                    "shared_option": 10,
                    "overridden_option": 1,
                },
                "stopping": {
                    "policy": "run_to_budget",
                    "max_evaluations": 500,
                },
            },
            "cma_es_full": {
                "inherits": "cma_es_common",
                "covariance_mode": "full",
                "options": {
                    "overridden_option": 2,
                    "child_option": True,
                },
            },
            "cma_es_diagonal": {
                "inherits": "cma_es_common",
                "covariance_mode": "diagonal",
                "options": {"CMA_diagonal": True},
            },
            "isotropic_es": {
                "inherits": "cma_es_common",
                "covariance_mode": "isotropic",
                "options": {"CMA_on": 0},
            },
        }
    }

    original = deepcopy(config)
    resolved = resolve_cma_variants(config)
    full = resolved["methods"]["cma_es_full"]

    assert full["library"] == "cma"
    assert full["options"] == {
        "shared_option": 10,
        "overridden_option": 2,
        "child_option": True,
    }
    assert full["stopping"] == {
        "policy": "run_to_budget",
        "max_evaluations": 500,
    }

    # Resolving settings must not mutate the original configuration.
    assert config == original


def test_missing_top_level_section_is_rejected():
    """A configuration missing a required section should fail clearly."""
    config = {
        "project": {},
        "reproducibility": {},
        "synthetic": {},
        "methods": {
            "cma_es_common": {},
            "cma_es_full": {"inherits": "cma_es_common"},
            "cma_es_diagonal": {"inherits": "cma_es_common"},
            "isotropic_es": {"inherits": "cma_es_common"},
        },
        "evaluation": {},
        "analysis": {},
        "paths": {},
        # "freeze" intentionally omitted.
    }

    with pytest.raises(ConfigError, match="freeze"):
        validate_config_structure(config)


def test_invalid_inheritance_is_rejected():
    """A variant with the wrong inheritance target must fail validation."""
    config = {
        "project": {},
        "reproducibility": {},
        "synthetic": {},
        "methods": {
            "cma_es_common": {},
            "cma_es_full": {"inherits": "wrong_parent"},
            "cma_es_diagonal": {"inherits": "cma_es_common"},
            "isotropic_es": {"inherits": "cma_es_common"},
        },
        "evaluation": {},
        "analysis": {},
        "paths": {},
        "freeze": {},
    }

    with pytest.raises(ConfigError, match="cma_es_full"):
        validate_config_structure(config)


def test_pilot_overrides_apply_without_changing_main_config():
    """Pilot settings apply only when the pilot stage is requested."""
    main = load_config(stage="main")
    pilot = load_config(stage="pilot")

    # Main-run budget is frozen after the pilot (D11).
    assert main["evaluation"]["budget_per_method"] == 2000

    # The pilot receives the configured provisional budget.
    assert pilot["evaluation"]["budget_per_method"] == 2000

    # The pilot uses the configured optimum sub-box fraction.
    assert (
        pilot["synthetic"]["ellipsoid"]["optimum"]["sub_box_fraction"]
        == 0.8
    )


def test_invalid_pilot_override_path_is_rejected():
    """Typos in override paths should fail rather than be silently ignored."""
    from config_loader import apply_pilot_overrides, ConfigError

    config = load_config(stage="main")
    config["pilot"]["overrides"] = {
        "evaluation.this_setting_does_not_exist": 123
    }

    with pytest.raises(ConfigError, match="Invalid pilot override path"):
        apply_pilot_overrides(config)


def test_invalid_stage_is_rejected():
    """Only the explicitly supported configuration stages are accepted."""
    with pytest.raises(ConfigError, match="stage"):
        load_config(stage="production")


# ---------------------------------------------------------------------------
# Drift, required options and readiness checks (decision log D8, D19)
# ---------------------------------------------------------------------------

import yaml

from config_loader import (
    DEFAULT_CONFIG_PATH,
    apply_pilot_overrides,
    check_cma_consistency,
    check_ready,
)


@pytest.fixture
def raw():
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("variant", ["cma_es_full", "cma_es_diagonal", "isotropic_es"])
def test_every_common_setting_reaches_every_variant(variant):
    config = load_config(stage="pilot")
    common = config["methods"]["cma_es_common"]
    for key, value in common.items():
        assert config["methods"][variant][key] == value, key


def test_variant_overriding_shared_setting_is_caught(raw):
    raw["methods"]["cma_es_diagonal"]["initial_step_size_fraction_of_box_width"] = 0.5
    with pytest.raises(ConfigError, match="differs from cma_es_common"):
        check_cma_consistency(resolve_cma_variants(raw))


def test_missing_required_variant_option_is_caught(raw):
    raw["methods"]["isotropic_es"]["options"] = {}
    with pytest.raises(ConfigError, match="CMA_on"):
        check_cma_consistency(resolve_cma_variants(raw))


def test_main_stage_not_ready_while_settings_are_null():
    config = load_config(stage="main")
    config["methods"]["bayesian_optimisation"]["budget_ceiling"] = None
    with pytest.raises(ConfigError, match="not ready"):
        check_ready(config, "main")


def test_frozen_config_is_ready_for_main_runs():
    check_ready(load_config(stage="main"), "main")


def test_pilot_checkpoints_end_at_budget():
    config = load_config(stage="pilot")
    cps = config["evaluation"]["checkpoints"]["evaluation_counts"]
    assert cps[-1] == config["evaluation"]["budget_per_method"]


@pytest.mark.parametrize(
    "path, bad",
    [
        ("synthetic.domain.upper_bound", -10.0),
        ("synthetic.ellipsoid.optimum.sub_box_fraction", 1.5),
        ("evaluation.budget_per_method", -5),
        ("evaluation.checkpoints.evaluation_counts", [90, 250, 500]),
        ("synthetic.ellipsoid.condition_numbers", [0.5, 100]),
    ],
)
def test_invalid_values_rejected(raw, path, bad):
    raw["pilot"]["overrides"][path] = bad
    config = resolve_cma_variants(apply_pilot_overrides(raw))
    with pytest.raises(ConfigError, match="Invalid configuration"):
        check_ready(config, "pilot")