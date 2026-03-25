"""Tests for storage layer (SQLite + queries)."""

from __future__ import annotations

from autotrain.storage.db import init_db
from autotrain.storage.models import (
    Iteration,
    IterationOutcome,
    MetricSnapshot,
    Run,
    RunStatus,
)
from autotrain.storage.queries import (
    add_run_api_cost,
    create_iteration,
    create_run,
    get_best_iterations,
    get_latest_run,
    get_recent_iterations,
    get_run,
    increment_run_iterations,
    record_metric,
    update_iteration,
    update_run_best,
    update_run_status,
)


class TestDatabase:
    def test_init_creates_tables(self, db_conn):
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "runs" in tables
        assert "iterations" in tables
        assert "metric_snapshots" in tables
        assert "journal" in tables

    def test_wal_mode(self, db_conn):
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_idempotent_init(self, tmp_dir):
        db_path = tmp_dir / ".autotrain" / "state.db"
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)  # Should not fail
        conn2.close()


class TestRunQueries:
    def test_create_and_get_run(self, db_conn):
        run = Run(
            id="run-123",
            repo_path="/path/to/repo",
            metric_name="val_auc",
            metric_target=0.85,
            metric_direction="maximize",
            status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)

        fetched = get_run(db_conn, "run-123")
        assert fetched is not None
        assert fetched.metric_name == "val_auc"
        assert fetched.metric_target == 0.85
        assert fetched.status == RunStatus.RUNNING

    def test_get_nonexistent_run(self, db_conn):
        assert get_run(db_conn, "nope") is None

    def test_update_status(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="m",
            metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)
        update_run_status(db_conn, "run-1", RunStatus.COMPLETED)

        fetched = get_run(db_conn, "run-1")
        assert fetched.status == RunStatus.COMPLETED

    def test_update_best(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="m",
            metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)
        update_run_best(db_conn, "run-1", 0.82, 5)

        fetched = get_run(db_conn, "run-1")
        assert fetched.best_metric_value == 0.82
        assert fetched.best_iteration == 5

    def test_increment_iterations(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="m",
            metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)
        increment_run_iterations(db_conn, "run-1")
        increment_run_iterations(db_conn, "run-1")

        fetched = get_run(db_conn, "run-1")
        assert fetched.total_iterations == 2

    def test_add_api_cost(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="m",
            metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)
        add_run_api_cost(db_conn, "run-1", 0.05)
        add_run_api_cost(db_conn, "run-1", 0.03)

        fetched = get_run(db_conn, "run-1")
        assert abs(fetched.total_api_cost - 0.08) < 1e-9

    def test_get_latest_run(self, db_conn):
        for i in range(3):
            run = Run(
                id=f"run-{i}", repo_path="/r", metric_name="m",
                metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
            )
            create_run(db_conn, run)

        latest = get_latest_run(db_conn)
        assert latest is not None
        assert latest.id == "run-2"


class TestIterationQueries:
    def _setup_run(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="val_auc",
            metric_target=0.85, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)

    def test_create_iteration(self, db_conn):
        self._setup_run(db_conn)
        it = Iteration(run_id="run-1", iteration_num=1, state="executing")
        row_id = create_iteration(db_conn, it)
        assert row_id > 0

    def test_update_iteration(self, db_conn):
        self._setup_run(db_conn)
        it = Iteration(run_id="run-1", iteration_num=1, state="executing")
        row_id = create_iteration(db_conn, it)

        update_iteration(db_conn, row_id, outcome=IterationOutcome.IMPROVED, metric_value=0.78)

        rows = get_recent_iterations(db_conn, "run-1")
        assert len(rows) == 1
        assert rows[0].outcome == IterationOutcome.IMPROVED
        assert rows[0].metric_value == 0.78

    def test_recent_iterations_ordered(self, db_conn):
        self._setup_run(db_conn)
        for i in range(5):
            create_iteration(
                db_conn,
                Iteration(run_id="run-1", iteration_num=i + 1, state="done"),
            )

        recent = get_recent_iterations(db_conn, "run-1", limit=3)
        assert len(recent) == 3
        # Should be in ascending order (oldest first within the limit)
        assert recent[0].iteration_num == 3
        assert recent[2].iteration_num == 5

    def test_best_iterations(self, db_conn):
        self._setup_run(db_conn)
        for i, val in enumerate([0.70, 0.82, 0.75, 0.85, 0.80]):
            it = Iteration(
                run_id="run-1", iteration_num=i + 1, state="done",
                outcome=IterationOutcome.IMPROVED, metric_value=val,
            )
            create_iteration(db_conn, it)

        best = get_best_iterations(db_conn, "run-1", direction="maximize", limit=3)
        assert len(best) == 3
        assert best[0].metric_value == 0.85


class TestMetricQueries:
    def test_record_metric(self, db_conn):
        run = Run(
            id="run-1", repo_path="/r", metric_name="m",
            metric_target=0.5, metric_direction="maximize", status=RunStatus.RUNNING,
        )
        create_run(db_conn, run)

        snapshot = MetricSnapshot(
            run_id="run-1", iteration_num=1, metric_name="val_auc", value=0.72,
        )
        record_metric(db_conn, snapshot)

        row = db_conn.execute(
            "SELECT * FROM metric_snapshots WHERE run_id = 'run-1'"
        ).fetchone()
        assert row is not None
        assert row["value"] == 0.72
