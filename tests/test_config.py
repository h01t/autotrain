"""Tests for configuration schema and loader."""

from __future__ import annotations

import pytest
import yaml

from autotrain.config.loader import load_config
from autotrain.config.schema import (
    BudgetConfig,
    ExecutionConfig,
    MetricConfig,
    RunConfig,
    _parse_duration,
)


class TestParseDuration:
    def test_seconds(self):
        assert _parse_duration("30s") == 30

    def test_minutes(self):
        assert _parse_duration("15m") == 900

    def test_hours(self):
        assert _parse_duration("4h") == 14400

    def test_days(self):
        assert _parse_duration("1d") == 86400

    def test_combined(self):
        assert _parse_duration("2h30m") == 9000

    def test_bare_int(self):
        assert _parse_duration("3600") == 3600

    def test_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_duration("abc")


class TestMetricConfig:
    def test_valid_json_mode(self):
        m = MetricConfig(name="val_auc", target=0.85)
        assert m.extraction_mode == "json"

    def test_regex_requires_pattern(self):
        with pytest.raises(ValueError, match="extraction_pattern required"):
            MetricConfig(name="val_auc", target=0.85, extraction_mode="regex")

    def test_regex_with_pattern(self):
        m = MetricConfig(
            name="val_auc",
            target=0.85,
            extraction_mode="regex",
            extraction_pattern=r"val_auc[:\s=]+([\d.]+)",
        )
        assert m.extraction_pattern is not None

class TestBudgetConfig:
    def test_time_string_parsed(self):
        b = BudgetConfig(time_seconds="4h")
        assert b.time_seconds == 14400

    def test_experiment_timeout_string(self):
        b = BudgetConfig(experiment_timeout_seconds="10m")
        assert b.experiment_timeout_seconds == 600


class TestExecutionConfig:
    def test_local_defaults(self):
        e = ExecutionConfig()
        assert e.mode == "local"

    def test_ssh_requires_host(self):
        with pytest.raises(ValueError, match="ssh_host required"):
            ExecutionConfig(mode="ssh")

    def test_ssh_requires_remote_dir(self):
        with pytest.raises(ValueError, match="ssh_remote_dir required"):
            ExecutionConfig(mode="ssh", ssh_host="gpu-box")

    def test_ssh_valid(self):
        e = ExecutionConfig(mode="ssh", ssh_host="gpu-box", ssh_remote_dir="~/project")
        assert e.ssh_host == "gpu-box"


class TestRunConfig:
    def test_minimal(self, tmp_dir):
        repo = tmp_dir / "repo"
        repo.mkdir()
        config = RunConfig(
            repo_path=repo,
            metric=MetricConfig(name="val_auc", target=0.85),
        )
        assert config.repo_path == repo
        assert config.budget.experiment_timeout_seconds == 1800

    def test_path_resolved(self, tmp_dir):
        repo = tmp_dir / "repo"
        repo.mkdir()
        config = RunConfig(
            repo_path=str(repo),
            metric=MetricConfig(name="val_auc", target=0.85),
        )
        assert config.repo_path.is_absolute()


class TestConfigLoader:
    def test_load_from_yaml(self, sample_repo):
        config_data = {
            "metric": {"name": "mAP@0.5", "target": 0.80, "direction": "maximize"},
            "budget": {"time_seconds": "2h", "max_iterations": 30},
            "execution": {"train_command": "python train.py --epochs 10"},
        }
        config_file = sample_repo / "autotrain.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(sample_repo)
        assert config.metric.name == "mAP@0.5"
        assert config.budget.time_seconds == 7200
        assert config.budget.max_iterations == 30

    def test_cli_overrides(self, sample_repo):
        config_data = {
            "metric": {"name": "val_auc", "target": 0.80},
        }
        config_file = sample_repo / "autotrain.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(
            sample_repo,
            cli_overrides={"metric": {"target": 0.90}},
        )
        assert config.metric.target == 0.90

    def test_missing_config_uses_defaults(self, sample_repo):
        config = load_config(
            sample_repo,
            cli_overrides={
                "metric": {"name": "val_auc", "target": 0.85},
            },
        )
        assert config.execution.mode == "local"
        assert config.agent.model == "claude-sonnet-4-20250514"
