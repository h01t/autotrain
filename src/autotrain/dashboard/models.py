"""Pydantic request/response models for dashboard run creation and control.

These models define the API contract for creating, starting, stopping,
and validating ML training runs from the dashboard UI.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class RunControlAction(StrEnum):
    """Actions the dashboard can perform on a run."""

    START = "start"
    STOP = "stop"
    RESTART = "restart"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ValidateConfigRequest(BaseModel):
    """Request to validate a run configuration without creating a run."""

    config_yaml: str = Field(
        ...,
        description="Full autotrain.yaml content as a string.",
        min_length=1,
    )


class ConfigValidationError(BaseModel):
    """A single configuration validation error."""

    field: str = Field(..., description="The config field path, e.g. 'metric.name'.")
    message: str = Field(..., description="Human-readable error message.")


class ValidateConfigResponse(BaseModel):
    """Response after validating a run configuration."""

    valid: bool
    errors: list[ConfigValidationError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU preflight
# ---------------------------------------------------------------------------


class PreflightGpuInfo(BaseModel):
    """Information about a single GPU discovered during preflight."""

    index: int
    name: str | None = None
    memory_total_mb: float | None = None
    memory_free_mb: float | None = None
    utilization_pct: float | None = None


class PreflightRequest(BaseModel):
    """Request to check if a target environment is ready for training."""

    repo_path: str = Field(..., description="Path to the training repository.")
    mode: Literal["local", "ssh"] = "local"
    ssh_host: str | None = None
    ssh_port: int = 22
    gpu_device: str | None = None
    venv_activate: str | None = None
    train_command: str = "python train.py"


class PreflightResult(BaseModel):
    """Individual check result within a preflight report."""

    check: str = Field(..., description="Name of the check, e.g. 'gpu_available'.")
    passed: bool
    message: str = ""
    detail: str | None = Field(
        default=None,
        description="Expanded diagnostic detail for 'Why this failed?'.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Actionable fix suggestion.",
    )


class PreflightResponse(BaseModel):
    """Full preflight check response."""

    passed: bool = Field(
        ...,
        description="True if all checks passed, false if any failed.",
    )
    checks: list[PreflightResult] = Field(default_factory=list)
    gpus: list[PreflightGpuInfo] = Field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Run creation
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Request to create (and optionally start) a new training run."""

    config_yaml: str = Field(
        ...,
        description="Full autotrain.yaml content as a string.",
        min_length=1,
    )
    start_immediately: bool = Field(
        default=True,
        description="Whether to start the run immediately after creation.",
    )


class CreateRunResponse(BaseModel):
    """Response after creating a run.

    * success — run created and optionally started
    * invalid_config — config validation errors (caller should return 400)
    * conflict — another active run exists on the same repo (caller should return 409)
    """

    run_id: str
    status: str  # "running", "stopped", "invalid_config", "conflict"
    message: str
    config_errors: list[ConfigValidationError] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run resume
# ---------------------------------------------------------------------------


class ResumeRunRequest(BaseModel):
    """Request to create a new run that resumes from a prior run's state."""

    run_id: str = Field(
        ...,
        description="The ID of the prior run to resume from.",
    )
    start_immediately: bool = Field(
        default=True,
        description="Whether to start the new run immediately after creation.",
    )


class ResumeRunResponse(BaseModel):
    """Response after creating a resume run."""

    new_run_id: str
    prior_run_id: str
    status: str
    message: str
    resumed_from_checkpoint: bool = False


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------


class RunConfigResponse(BaseModel):
    """Current configuration for a run (parsed from config_snapshot)."""

    run_id: str
    config_yaml: str | None = None
    config_json: dict | None = None


# ---------------------------------------------------------------------------
# Logs endpoint
# ---------------------------------------------------------------------------


class RunLogsResponse(BaseModel):
    """Training log output for a run."""

    run_id: str
    lines: list[str] = Field(default_factory=list)
    total_lines: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Artifacts endpoint
# ---------------------------------------------------------------------------


class ArtifactInfo(BaseModel):
    """Information about a single artifact file."""

    name: str
    path: str
    size_bytes: int
    modified: str | None = None


class ArtifactsListResponse(BaseModel):
    """List of artifacts for a run."""

    run_id: str
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    total_bytes: int = 0


# ---------------------------------------------------------------------------
# Defaults endpoint
# ---------------------------------------------------------------------------


class DefaultsResponse(BaseModel):
    """Default configuration template for new runs."""

    config_yaml: str


# ---------------------------------------------------------------------------
# Save-config endpoint
# ---------------------------------------------------------------------------


class SaveConfigRequest(BaseModel):
    """Request to save a configuration to a repo."""

    repo_path: str = Field(..., description="Path to the training repository.")
    config_yaml: str = Field(..., min_length=1, description="Full autotrain.yaml content.")


class SaveConfigResponse(BaseModel):
    """Response after saving a configuration."""

    success: bool
    path: str
    message: str


# ---------------------------------------------------------------------------
# Run control responses
# ---------------------------------------------------------------------------


class RunActionResponse(BaseModel):
    """Generic response for start/stop/restart actions."""

    run_id: str
    action: str
    success: bool
    message: str
    previous_status: str | None = None
    new_status: str | None = None


class RunStatusResponse(BaseModel):
    """Current status of a run."""

    run_id: str
    status: str
    is_active: bool
    pid: int | None = None
    uptime_seconds: float | None = None
