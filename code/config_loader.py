
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

    return resolve_cma_variants(config)
