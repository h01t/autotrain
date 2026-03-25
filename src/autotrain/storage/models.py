"""Data models for storage layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    STOPPED = "stopped"


class IterationOutcome(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    NO_CHANGE = "no_change"
    CRASHED = "crashed"
    SANDBOX_REJECTED = "sandbox_rejected"
    TIMEOUT = "timeout"


@dataclass
class Run:
    id: str  # UUID
    repo_path: str
    metric_name: str
    metric_target: float
    metric_direction: str  # "maximize" | "minimize"
    status: RunStatus
    best_metric_value: float | None = None
    best_iteration: int | None = None
    total_iterations: int = 0
    total_api_cost: float = 0.0
    git_branch: str | None = None
    config_snapshot: str | None = None  # JSON dump of RunConfig
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass
class Iteration:
    id: int | None = None  # Auto-increment
    run_id: str = ""
    iteration_num: int = 0
    state: str = ""  # Current AgentState name
    outcome: IterationOutcome | None = None
    metric_value: float | None = None
    commit_hash: str | None = None
    agent_reasoning: str | None = None
    agent_hypothesis: str | None = None
    changes_summary: str | None = None
    duration_seconds: float | None = None
    api_cost: float | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class MetricSnapshot:
    id: int | None = None
    run_id: str = ""
    iteration_num: int = 0
    metric_name: str = ""
    value: float = 0.0
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class JournalEntry:
    id: int | None = None
    run_id: str = ""
    iteration_num: int = 0
    state: str = ""
    data: str | None = None  # JSON payload
    timestamp: datetime = field(default_factory=_utcnow)
