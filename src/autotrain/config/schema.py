"""Pydantic configuration models for AutoTrain."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _parse_duration(value: str) -> int:
    """Parse a human-readable duration string to seconds.

    Supports: '30s', '15m', '4h', '1d', '2h30m', or bare int (seconds).
    """
    if isinstance(value, int | float):
        return int(value)
    total = 0
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])", re.IGNORECASE)
    matches = pattern.findall(value)
    if not matches:
        # Try bare integer
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot parse duration: {value!r}")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    for amount, unit in matches:
        total += int(float(amount) * multipliers[unit.lower()])
    return total


class MetricConfig(BaseModel):
    """Target metric definition."""

    name: str
    target: float
    direction: Literal["maximize", "minimize"] = "maximize"
    extraction_mode: Literal["json", "regex"] = "json"
    extraction_pattern: str | None = None

    @model_validator(mode="after")
    def validate_extraction(self):
        if self.extraction_mode == "regex" and not self.extraction_pattern:
            raise ValueError("extraction_pattern required when extraction_mode='regex'")
        return self


class BudgetConfig(BaseModel):
    """Resource budget limits."""

    time_seconds: int | None = None
    api_dollars: float | None = None
    max_iterations: int | None = None
    experiment_timeout_seconds: int = 900  # 15 min default

    @field_validator("time_seconds", mode="before")
    @classmethod
    def parse_time(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return _parse_duration(v)
        return int(v)

    @field_validator("experiment_timeout_seconds", mode="before")
    @classmethod
    def parse_experiment_timeout(cls, v):
        if isinstance(v, str):
            return _parse_duration(v)
        return int(v)


class ExecutionConfig(BaseModel):
    """Training execution configuration."""

    mode: Literal["local", "ssh"] = "local"
    train_command: str = "python train.py"
    gpu_device: str | None = None
    venv_activate: str | None = None
    working_dir: str | None = None

    # SSH-specific
    ssh_host: str | None = None
    ssh_remote_dir: str | None = None
    ssh_key: str | None = None
    ssh_port: int = 22
    ssh_setup_command: str | None = None  # e.g. "uv sync", runs once before first training
    rsync_excludes: list[str] = Field(
        default_factory=lambda: [
            ".venv",
            "data",
            "__pycache__",
            ".git",
            ".autotrain",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.onnx",
            "outputs",
            "runs",
        ]
    )
    checkpoint_patterns: list[str] = Field(
        default_factory=lambda: [
            "outputs/training/*/weights/last.pt",
            "outputs/training/*/weights/best.pt",
            "**/checkpoint-*",
        ]
    )

    @model_validator(mode="after")
    def validate_ssh(self):
        if self.mode == "ssh":
            if not self.ssh_host:
                raise ValueError("ssh_host required when mode='ssh'")
            if not self.ssh_remote_dir:
                raise ValueError("ssh_remote_dir required when mode='ssh'")
        return self


class SandboxConfig(BaseModel):
    """Code sandboxing constraints."""

    writable_files: list[str] = Field(default_factory=lambda: ["train.py", "config.yaml"])
    forbidden_patterns: list[str] = Field(
        default_factory=lambda: [
            r"subprocess\.",
            r"os\.system",
            r"exec\(",
            r"eval\(",
            r"__import__",
            r"shutil\.rmtree",
            r"os\.remove",
            r"os\.unlink",
        ]
    )
    max_file_size_bytes: int = 100_000
    max_changes_per_iteration: int = 5


class AgentConfig(BaseModel):
    """LLM agent configuration."""

    provider: str = "anthropic"  # "anthropic" | "ollama" | "deepseek"
    api_base: str = "http://localhost:11434"  # Base URL for Ollama
    model: str = "claude-sonnet-4-20250514"
    max_retries: int = 5
    retry_base_seconds: float = 2.0
    hard_timeout_seconds: int = 120
    max_context_tokens: int = 16000
    stagnation_threshold: int = 5
    temperature: float = 0.3


class WatchdogConfig(BaseModel):
    """Training watchdog configuration."""

    enabled: bool = True
    check_interval_seconds: int = 30
    gpu_memory_min_mb: int = 100
    disk_space_min_gb: float = 1.0
    stdout_stagnation_minutes: int = 30
    process_hung_minutes: int = 60


class NotifyConfig(BaseModel):
    """Notification configuration."""

    webhook_url: str | None = None
    webhook_events: list[str] = Field(
        default_factory=lambda: [
            "target_hit",
            "budget_exhausted",
            "stuck",
            "failed",
            "error",
        ]
    )
    terminal: bool = True


class RunConfig(BaseModel):
    """Top-level run configuration."""

    repo_path: Path
    metric: MetricConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)

    @field_validator("repo_path", mode="before")
    @classmethod
    def resolve_path(cls, v):
        return Path(v).resolve()
