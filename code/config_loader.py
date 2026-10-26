"""Load and resolve the experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

CMA_VARIANTS = (
    "cma_es_full",
    "cma_es_diagonal",
    "isotropic_es",
)


class ConfigError(ValueError):
    """Raised when the experiment configuration is invalid."""


# Settings that must be non-null before a stage may run (decision log D28).
READY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "pilot": (
        "reproducibility.base_seed",
        "synthetic.domain.lower_bound",
        "synthetic.domain.upper_bound",
        "synthetic.domain.boundary_handling",
        "synthetic.ellipsoid.optimum.sub_box_fraction",
        "methods.cma_es_common.initial_step_size_fraction_of_box_width",
        "methods.cma_es_common.population_size",
        "methods.cma_es_common.boundary_handling",
        "evaluation.budget_per_method",
        "evaluation.checkpoints.evaluation_counts",
    ),
    "main": (
        "reproducibility.base_seed",
        "synthetic.domain.lower_bound",
        "synthetic.domain.upper_bound",
        "synthetic.domain.boundary_handling",
        "synthetic.ellipsoid.optimum.sub_box_fraction",
        "methods.cma_es_common.library_version",
        "methods.cma_es_common.initial_step_size_fraction_of_box_width",
        "methods.cma_es_common.population_size",
        "methods.cma_es_common.boundary_handling",
        "evaluation.budget_per_method",
        "evaluation.checkpoints.evaluation_counts",
        "analysis.log_regret_floor",
        "analysis.invariance_tolerance.value",
        "analysis.uncertainty.bootstrap_replicates",
    ),
}

# Options each variant must still carry after resolution (D8, D9).
REQUIRED_VARIANT_OPTIONS = {
    "cma_es_diagonal": ("CMA_diagonal", True),
    "isotropic_es": ("CMA_on", 0),
}

_MISSING = object()


def get_path(config: dict[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    """Return the value at a dotted path such as 'evaluation.budget_per_method'."""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            if default is _MISSING:
                raise ConfigError(f"Missing config key: {dotted}")
            return default
        node = node[key]
    return node


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Merge nested dictionaries, with override values taking precedence."""
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def validate_config_structure(config: Any) -> None:
    """Validate the basic structure without requiring pilot values to be frozen."""
    if not isinstance(config, dict):
        raise ConfigError("The YAML document must contain a mapping.")

    for section in ("project", "reproducibility", "synthetic",
                    "methods", "evaluation", "analysis", "paths", "freeze"):
        if section not in config:
            raise ConfigError(f"Missing required top-level section: {section}")

    methods = config["methods"]
    if not isinstance(methods, dict):
        raise ConfigError("'methods' must be a mapping.")

    common = methods.get("cma_es_common")
    if not isinstance(common, dict):
        raise ConfigError("'methods.cma_es_common' must be a mapping.")

    for name in CMA_VARIANTS:
        variant = methods.get(name)

        if not isinstance(variant, dict):
            raise ConfigError(f"'methods.{name}' must be a mapping.")

        if variant.get("inherits") != "cma_es_common":
            raise ConfigError(
                f"'methods.{name}.inherits' must be 'cma_es_common'."
            )


def resolve_cma_variants(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with shared CMA-ES settings merged into each variant."""
    resolved = deepcopy(config)
    methods = resolved["methods"]
    common = methods["cma_es_common"]

    for name in CMA_VARIANTS:
        variant = methods[name]

        # Child-specific values override common values, including nested
        # settings such as options.
        child_settings = {
            key: value
            for key, value in variant.items()
            if key != "inherits"
        }

        merged = _deep_merge(common, child_settings)
        merged.pop("inherits", None)
        methods[name] = merged

    return resolved


def apply_pilot_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply configured pilot overrides to a copy of the configuration."""
    resolved = deepcopy(config)
    pilot = resolved.get("pilot")

    if not isinstance(pilot, dict):
        raise ConfigError("Pilot stage requested, but 'pilot' is missing.")

    overrides = pilot.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigError("'pilot.overrides' must be a mapping.")

    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ConfigError("Each pilot override must have a non-empty path.")

        parts = dotted_path.split(".")
        target = resolved

        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ConfigError(
                    f"Invalid pilot override path: {dotted_path}"
                )
            target = target[part]

        leaf = parts[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise ConfigError(
                f"Invalid pilot override path: {dotted_path}"
            )

        target[leaf] = deepcopy(value)

    return resolved


def check_cma_consistency(resolved: dict[str, Any]) -> None:
    """Every cma_es_common setting must be identical in each variant (D9).

    Variants may add keys (covariance_mode, options) but must not override
    a shared key. Required variant options must survive the merge.
    """
    methods = resolved["methods"]
    common = methods["cma_es_common"]
    errors = []
    for name in CMA_VARIANTS:
        for key, value in common.items():
            if methods[name].get(key) != value:
                errors.append(f"methods.{name}.{key} differs from cma_es_common")
    for name, (option, expected) in REQUIRED_VARIANT_OPTIONS.items():
        if methods[name].get("options", {}).get(option) != expected:
            errors.append(f"methods.{name}.options.{option} must be {expected!r}")
    if errors:
        raise ConfigError("Inconsistent CMA-ES settings:\n  " + "\n  ".join(errors))


def check_ready(config: dict[str, Any], stage: str) -> None:
    """Refuse to run a stage while required settings are null or invalid (D28).

    load_config(stage="pilot") calls this automatically. The main-run entry
    point must call check_ready(config, "main") before launching any run.
    """
    if stage not in READY_REQUIREMENTS:
        raise ConfigError(f"Unknown stage: {stage!r}")

    missing = [p for p in READY_REQUIREMENTS[stage] if get_path(config, p, None) is None]
    if missing:
        raise ConfigError(
            f"Stage {stage!r} is not ready; these settings are still null or absent:\n  "
            + "\n  ".join(missing)
        )

    errors = []
    if not get_path(config, "synthetic.domain.lower_bound") < get_path(config, "synthetic.domain.upper_bound"):
        errors.append("synthetic.domain.lower_bound must be < upper_bound")
    if not 0 < get_path(config, "synthetic.ellipsoid.optimum.sub_box_fraction") <= 1:
        errors.append("sub_box_fraction must be in (0, 1]")
    kappas = get_path(config, "synthetic.ellipsoid.condition_numbers")
    if not kappas or any(k < 1 for k in kappas):
        errors.append("condition_numbers must be non-empty and all >= 1")
    if not 0 < get_path(config, "methods.cma_es_common.initial_step_size_fraction_of_box_width") <= 1:
        errors.append("initial_step_size_fraction_of_box_width must be in (0, 1]")
    budget = get_path(config, "evaluation.budget_per_method")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        errors.append("evaluation.budget_per_method must be a positive integer")
    else:
        cps = list(get_path(config, "evaluation.checkpoints.evaluation_counts"))
        if cps != sorted(set(cps)) or cps[0] <= 0 or cps[-1] != budget:
            errors.append("checkpoints must be strictly increasing, positive, and end at the budget")
    if errors:
        raise ConfigError("Invalid configuration:\n  " + "\n  ".join(errors))


def load_config(
    config_path: str | Path | None = None,
    *,
    stage: str = "main",
) -> dict[str, Any]:
    """Load configuration for the main experiment or pilot stage."""
    if stage not in {"main", "pilot"}:
        raise ConfigError("stage must be either 'main' or 'pilot'.")

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML file: {path}") from exc

    validate_config_structure(config)

    if stage == "pilot":
        config = apply_pilot_overrides(config)

    resolved = resolve_cma_variants(config)
    check_cma_consistency(resolved)

    if stage == "pilot":
        check_ready(resolved, "pilot")

    return resolved