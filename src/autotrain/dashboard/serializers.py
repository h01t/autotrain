"""Convert storage dataclasses to JSON-safe dicts for the API layer."""

from __future__ import annotations

import json
from datetime import datetime

from autotrain.storage.models import (
    EpochMetric,
    GpuSnapshot,
    Iteration,
    MetricSnapshot,
    Run,
)


def _dt(val: datetime | None) -> str | None:
    if val is None:
        return None
    return val.isoformat()


def serialize_run(run: Run) -> dict:
    return {
        "id": run.id,
        "repo_path": run.repo_path,
        "metric_name": run.metric_name,
        "metric_target": run.metric_target,
        "metric_direction": run.metric_direction,
        "status": run.status.value if run.status else None,
        "best_metric_value": run.best_metric_value,
        "best_iteration": run.best_iteration,
        "total_iterations": run.total_iterations,
        "total_api_cost": run.total_api_cost,
        "git_branch": run.git_branch,
        "config_snapshot": run.config_snapshot,
        "resumed_from_run_id": run.resumed_from_run_id,
        "created_at": _dt(run.created_at),
        "updated_at": _dt(run.updated_at),
    }


def serialize_iteration(it: Iteration) -> dict:
    return {
        "id": it.id,
        "run_id": it.run_id,
        "iteration_num": it.iteration_num,
        "state": it.state,
        "outcome": it.outcome.value if it.outcome else None,
        "metric_value": it.metric_value,
        "commit_hash": it.commit_hash,
        "agent_reasoning": it.agent_reasoning,
        "agent_hypothesis": it.agent_hypothesis,
        "changes_summary": it.changes_summary,
        "duration_seconds": it.duration_seconds,
        "api_cost": it.api_cost,
        "error_message": it.error_message,
        "checkpoint_path": it.checkpoint_path,
        "resumed_from_checkpoint": it.resumed_from_checkpoint,
        "created_at": _dt(it.created_at),
    }


def serialize_metric_snapshot(ms: MetricSnapshot) -> dict:
    return {
        "id": ms.id,
        "run_id": ms.run_id,
        "iteration_num": ms.iteration_num,
        "metric_name": ms.metric_name,
        "value": ms.value,
        "timestamp": _dt(ms.timestamp),
    }


def serialize_epoch_metric(em: EpochMetric) -> dict:
    # Parse the JSON metrics string into a dict for the frontend
    try:
        metrics = json.loads(em.metrics) if isinstance(em.metrics, str) else em.metrics
    except (json.JSONDecodeError, TypeError):
        metrics = {}

    return {
        "id": em.id,
        "run_id": em.run_id,
        "iteration_num": em.iteration_num,
        "epoch": em.epoch,
        "metrics": metrics,
        "timestamp": _dt(em.timestamp),
    }


def serialize_gpu_snapshot(gs: GpuSnapshot) -> dict:
    return {
        "id": gs.id,
        "run_id": gs.run_id,
        "gpu_index": gs.gpu_index,
        "utilization_pct": gs.utilization_pct,
        "memory_used_mb": gs.memory_used_mb,
        "memory_total_mb": gs.memory_total_mb,
        "temperature_c": gs.temperature_c,
        "timestamp": _dt(gs.timestamp),
    }
