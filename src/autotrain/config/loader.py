"""Configuration loading with cascading priority: CLI > project > global > defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autotrain.config.schema import RunConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if missing or empty."""
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    return yaml.safe_load(text) or {}


def load_config(
    repo_path: Path,
    cli_overrides: dict[str, Any] | None = None,
    config_file: Path | None = None,
) -> RunConfig:
    """Load RunConfig with cascading priority.

    Priority (highest to lowest):
    1. CLI overrides
    2. Project config (autotrain.yaml in repo root, or explicit config_file)
    3. Global config (~/.config/autotrain/config.yaml)
    4. Pydantic defaults
    """
    config_data: dict[str, Any] = {}

    # Global defaults
    global_path = Path.home() / ".config" / "autotrain" / "config.yaml"
    config_data = _deep_merge(config_data, _load_yaml(global_path))

    # Project config
    if config_file:
        config_data = _deep_merge(config_data, _load_yaml(config_file))
    else:
        project_path = repo_path / "autotrain.yaml"
        config_data = _deep_merge(config_data, _load_yaml(project_path))

    # CLI overrides
    if cli_overrides:
        config_data = _deep_merge(config_data, cli_overrides)

    # Ensure repo_path is set
    config_data["repo_path"] = str(repo_path)

    return RunConfig(**config_data)
