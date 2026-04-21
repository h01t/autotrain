"""Named query functions for the storage layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from autotrain.storage.models import (
    EpochMetric,
    GpuSnapshot,
    Iteration,
    IterationOutcome,
    JournalEntry,
    MetricSnapshot,
    Run,
    RunStatus,
)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# --- Runs ---


def create_run(conn: sqlite3.Connection, run: Run) -> None:
    """Insert a new run."""
    conn.execute(
        """INSERT INTO runs (id, repo_path, metric_name, metric_target, metric_direction,
           status, best_metric_value, best_iteration, total_iterations, total_api_cost,
           git_branch, config_snapshot, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run.id,
            run.repo_path,
            run.metric_name,
            run.metric_target,
            run.metric_direction,
            run.status.value,
            run.best_metric_value,
            run.best_iteration,
            run.total_iterations,
            run.total_api_cost,
            run.git_branch,
            run.config_snapshot,
            run.created_at.isoformat(),
            run.updated_at.isoformat(),
        ),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> Run | None:
    """Get a run by ID."""
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    return Run(
        id=row["id"],
        repo_path=row["repo_path"],
        metric_name=row["metric_name"],
        metric_target=row["metric_target"],
        metric_direction=row["metric_direction"],
        status=RunStatus(row["status"]),
        best_metric_value=row["best_metric_value"],
        best_iteration=row["best_iteration"],
        total_iterations=row["total_iterations"],
        total_api_cost=row["total_api_cost"],
        git_branch=row["git_branch"],
        config_snapshot=row["config_snapshot"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def get_all_runs(conn: sqlite3.Connection) -> list[Run]:
    """Get all runs, newest first."""
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC"
    ).fetchall()
    return [
        Run(
            id=row["id"],
            repo_path=row["repo_path"],
            metric_name=row["metric_name"],
            metric_target=row["metric_target"],
            metric_direction=row["metric_direction"],
            status=RunStatus(row["status"]),
            best_metric_value=row["best_metric_value"],
            best_iteration=row["best_iteration"],
            total_iterations=row["total_iterations"],
            total_api_cost=row["total_api_cost"],
            git_branch=row["git_branch"],
            config_snapshot=row["config_snapshot"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        for row in rows
    ]


def get_latest_run(conn: sqlite3.Connection, repo_path: str | None = None) -> Run | None:
    """Get the most recent run, optionally filtered by repo."""
    if repo_path:
        row = conn.execute(
            "SELECT * FROM runs WHERE repo_path = ? ORDER BY created_at DESC LIMIT 1",
            (repo_path,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return get_run(conn, row["id"])


def update_run_status(conn: sqlite3.Connection, run_id: str, status: RunStatus) -> None:
    """Update run status."""
    conn.execute(
        "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
        (status.value, _utcnow_iso(), run_id),
    )
    conn.commit()


def update_run_best(
    conn: sqlite3.Connection,
    run_id: str,
    metric_value: float,
    iteration_num: int,
) -> None:
    """Update the best metric value for a run."""
    conn.execute(
        "UPDATE runs SET best_metric_value = ?, best_iteration = ?, updated_at = ? WHERE id = ?",
        (metric_value, iteration_num, _utcnow_iso(), run_id),
    )
    conn.commit()


def increment_run_iterations(conn: sqlite3.Connection, run_id: str) -> None:
    """Increment the iteration counter for a run."""
    conn.execute(
        "UPDATE runs SET total_iterations = total_iterations + 1, updated_at = ? WHERE id = ?",
        (_utcnow_iso(), run_id),
    )
    conn.commit()


def add_run_api_cost(conn: sqlite3.Connection, run_id: str, cost: float) -> None:
    """Add API cost to the run total."""
    conn.execute(
        "UPDATE runs SET total_api_cost = total_api_cost + ?, updated_at = ? WHERE id = ?",
        (cost, _utcnow_iso(), run_id),
    )
    conn.commit()


# --- Iterations ---


def create_iteration(conn: sqlite3.Connection, iteration: Iteration) -> int:
    """Insert a new iteration, returning its ID."""
    cursor = conn.execute(
        """INSERT INTO iterations (run_id, iteration_num, state, outcome, metric_value,
           commit_hash, agent_reasoning, agent_hypothesis, changes_summary,
           duration_seconds, api_cost, error_message, checkpoint_path,
           resumed_from_checkpoint, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            iteration.run_id,
            iteration.iteration_num,
            iteration.state,
            iteration.outcome.value if iteration.outcome else None,
            iteration.metric_value,
            iteration.commit_hash,
            iteration.agent_reasoning,
            iteration.agent_hypothesis,
            iteration.changes_summary,
            iteration.duration_seconds,
            iteration.api_cost,
            iteration.error_message,
            iteration.checkpoint_path,
            int(iteration.resumed_from_checkpoint),
            iteration.created_at.isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_iteration(
    conn: sqlite3.Connection,
    iteration_id: int,
    **kwargs,
) -> None:
    """Update specific fields on an iteration."""
    allowed = {
        "state", "outcome", "metric_value", "commit_hash", "agent_reasoning",
        "agent_hypothesis", "changes_summary", "duration_seconds", "api_cost",
        "error_message", "checkpoint_path", "resumed_from_checkpoint",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    # Convert enums to values
    if "outcome" in updates and isinstance(updates["outcome"], IterationOutcome):
        updates["outcome"] = updates["outcome"].value

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values())
    values.append(iteration_id)
    conn.execute(f"UPDATE iterations SET {set_clause} WHERE id = ?", values)
    conn.commit()


def get_recent_iterations(
    conn: sqlite3.Connection,
    run_id: str,
    limit: int = 20,
) -> list[Iteration]:
    """Get the most recent iterations for a run."""
    rows = conn.execute(
        "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration_num DESC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return [_row_to_iteration(row) for row in reversed(rows)]


def get_best_iterations(
    conn: sqlite3.Connection,
    run_id: str,
    direction: str = "maximize",
    limit: int = 5,
) -> list[Iteration]:
    """Get the best iterations by metric value."""
    order = "DESC" if direction == "maximize" else "ASC"
    rows = conn.execute(
        f"""SELECT * FROM iterations WHERE run_id = ? AND metric_value IS NOT NULL
            AND outcome = 'improved' ORDER BY metric_value {order} LIMIT ?""",
        (run_id, limit),
    ).fetchall()
    return [_row_to_iteration(row) for row in rows]


def _row_to_iteration(row: sqlite3.Row) -> Iteration:
    keys = row.keys()
    return Iteration(
        id=row["id"],
        run_id=row["run_id"],
        iteration_num=row["iteration_num"],
        state=row["state"],
        outcome=IterationOutcome(row["outcome"]) if row["outcome"] else None,
        metric_value=row["metric_value"],
        commit_hash=row["commit_hash"],
        agent_reasoning=row["agent_reasoning"],
        agent_hypothesis=row["agent_hypothesis"],
        changes_summary=row["changes_summary"],
        duration_seconds=row["duration_seconds"],
        api_cost=row["api_cost"],
        error_message=row["error_message"],
        checkpoint_path=row["checkpoint_path"] if "checkpoint_path" in keys else None,
        resumed_from_checkpoint=bool(
            row["resumed_from_checkpoint"]
        ) if "resumed_from_checkpoint" in keys else False,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# --- Metrics ---


def record_metric(conn: sqlite3.Connection, snapshot: MetricSnapshot) -> None:
    """Record a metric snapshot."""
    conn.execute(
        """INSERT INTO metric_snapshots (run_id, iteration_num, metric_name, value, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (
            snapshot.run_id,
            snapshot.iteration_num,
            snapshot.metric_name,
            snapshot.value,
            snapshot.timestamp.isoformat(),
        ),
    )
    conn.commit()


def get_all_metric_snapshots(
    conn: sqlite3.Connection, run_id: str,
) -> list[MetricSnapshot]:
    """Get all metric snapshots for a run, ordered by iteration."""
    rows = conn.execute(
        """SELECT * FROM metric_snapshots WHERE run_id = ?
           ORDER BY iteration_num ASC""",
        (run_id,),
    ).fetchall()
    return [
        MetricSnapshot(
            id=row["id"],
            run_id=row["run_id"],
            iteration_num=row["iteration_num"],
            metric_name=row["metric_name"],
            value=row["value"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        for row in rows
    ]


# --- Epoch Metrics ---


def record_epoch_metric(
    conn: sqlite3.Connection,
    run_id: str,
    iteration_num: int,
    epoch: int,
    metrics_json: str,
) -> None:
    """Record a per-epoch metric snapshot."""
    conn.execute(
        """INSERT INTO epoch_metrics (run_id, iteration_num, epoch, metrics, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, iteration_num, epoch, metrics_json, _utcnow_iso()),
    )
    conn.commit()


def get_epoch_metrics(
    conn: sqlite3.Connection,
    run_id: str,
    iteration_num: int | None = None,
) -> list[EpochMetric]:
    """Get epoch metrics for a run, optionally filtered by iteration."""
    if iteration_num is not None:
        rows = conn.execute(
            """SELECT * FROM epoch_metrics WHERE run_id = ? AND iteration_num = ?
               ORDER BY epoch ASC""",
            (run_id, iteration_num),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM epoch_metrics WHERE run_id = ?
               ORDER BY iteration_num ASC, epoch ASC""",
            (run_id,),
        ).fetchall()
    return [
        EpochMetric(
            id=row["id"],
            run_id=row["run_id"],
            iteration_num=row["iteration_num"],
            epoch=row["epoch"],
            metrics=row["metrics"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        for row in rows
    ]


# --- GPU Snapshots ---


def record_gpu_snapshot(conn: sqlite3.Connection, snapshot: GpuSnapshot) -> None:
    """Record a GPU metrics snapshot."""
    conn.execute(
        """INSERT INTO gpu_snapshots
           (run_id, gpu_index, utilization_pct, memory_used_mb, memory_total_mb,
            temperature_c, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot.run_id,
            snapshot.gpu_index,
            snapshot.utilization_pct,
            snapshot.memory_used_mb,
            snapshot.memory_total_mb,
            snapshot.temperature_c,
            snapshot.timestamp.isoformat(),
        ),
    )
    conn.commit()


def get_gpu_snapshots(
    conn: sqlite3.Connection, run_id: str, limit: int = 500,
) -> list[GpuSnapshot]:
    """Get recent GPU snapshots for a run, ordered by timestamp."""
    rows = conn.execute(
        """SELECT * FROM (
               SELECT * FROM gpu_snapshots WHERE run_id = ?
               ORDER BY timestamp DESC LIMIT ?
           ) ORDER BY timestamp ASC""",
        (run_id, limit),
    ).fetchall()
    return [_row_to_gpu_snapshot(row) for row in rows]


def get_latest_gpu_snapshot(
    conn: sqlite3.Connection, run_id: str,
) -> GpuSnapshot | None:
    """Get the most recent GPU snapshot for a run."""
    row = conn.execute(
        "SELECT * FROM gpu_snapshots WHERE run_id = ? ORDER BY timestamp DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return _row_to_gpu_snapshot(row)


def _row_to_gpu_snapshot(row: sqlite3.Row) -> GpuSnapshot:
    return GpuSnapshot(
        id=row["id"],
        run_id=row["run_id"],
        gpu_index=row["gpu_index"],
        utilization_pct=row["utilization_pct"],
        memory_used_mb=row["memory_used_mb"],
        memory_total_mb=row["memory_total_mb"],
        temperature_c=row["temperature_c"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )


# --- Journal ---


def write_journal(conn: sqlite3.Connection, entry: JournalEntry) -> None:
    """Write a journal entry for crash recovery."""
    conn.execute(
        """INSERT INTO journal (run_id, iteration_num, state, data, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (
            entry.run_id,
            entry.iteration_num,
            entry.state,
            entry.data,
            entry.timestamp.isoformat(),
        ),
    )
    conn.commit()


def get_latest_journal(conn: sqlite3.Connection, run_id: str) -> JournalEntry | None:
    """Get the most recent journal entry for a run."""
    row = conn.execute(
        "SELECT * FROM journal WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return JournalEntry(
        id=row["id"],
        run_id=row["run_id"],
        iteration_num=row["iteration_num"],
        state=row["state"],
        data=row["data"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )
