"""Tests for the FastAPI dashboard API endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrain.dashboard.server import create_app
from autotrain.storage.db import init_db
from autotrain.storage.models import (
    GpuSnapshot,
    Iteration,
    IterationOutcome,
    MetricSnapshot,
    Run,
    RunStatus,
)
from autotrain.storage.queries import (
    create_iteration,
    create_run,
    record_epoch_metric,
    record_gpu_snapshot,
    record_metric,
    update_iteration,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create and seed a test database."""
    db_file = tmp_path / ".autotrain" / "state.db"
    conn = init_db(db_file)

    # Create a run
    run = Run(
        id="test-run",
        repo_path="/test",
        metric_name="mAP",
        metric_target=0.9,
        metric_direction="maximize",
        status=RunStatus.RUNNING,
    )
    create_run(conn, run)

    # Create iterations
    it = Iteration(run_id="test-run", iteration_num=1, state="evaluating")
    it_id = create_iteration(conn, it)
    update_iteration(
        conn, it_id,
        outcome=IterationOutcome.IMPROVED,
        metric_value=0.75,
        commit_hash="abc1234",
        agent_hypothesis="Increase lr to 0.01",
    )

    it2 = Iteration(run_id="test-run", iteration_num=2, state="evaluating")
    it2_id = create_iteration(conn, it2)
    update_iteration(
        conn, it2_id,
        outcome=IterationOutcome.REGRESSED,
        metric_value=0.70,
        commit_hash="def5678",
        agent_hypothesis="Decrease batch size",
    )

    # Record metrics
    record_metric(conn, MetricSnapshot(
        run_id="test-run", iteration_num=1, metric_name="mAP", value=0.75,
    ))
    record_metric(conn, MetricSnapshot(
        run_id="test-run", iteration_num=2, metric_name="mAP", value=0.70,
    ))

    # Record epoch metrics
    record_epoch_metric(conn, "test-run", 1, 1, '{"loss": 0.5, "mAP": 0.6}')
    record_epoch_metric(conn, "test-run", 1, 2, '{"loss": 0.3, "mAP": 0.75}')

    # Record GPU snapshot
    record_gpu_snapshot(conn, GpuSnapshot(
        run_id="test-run", gpu_index=0,
        utilization_pct=85.0, memory_used_mb=4000.0,
        memory_total_mb=8192.0, temperature_c=72.0,
    ))

    conn.close()
    return db_file


@pytest.fixture
def client(db_path: Path) -> TestClient:
    app = create_app(db_path)
    return TestClient(app)


# -- Health -------------------------------------------------------------------

def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# -- Runs ---------------------------------------------------------------------

def test_list_runs(client):
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) == 1
    assert runs[0]["id"] == "test-run"
    assert runs[0]["metric_name"] == "mAP"
    assert runs[0]["status"] == "running"


def test_get_run(client):
    resp = client.get("/api/v1/runs/test-run")
    assert resp.status_code == 200
    run = resp.json()
    assert run["id"] == "test-run"
    assert run["metric_target"] == 0.9


def test_get_run_not_found(client):
    resp = client.get("/api/v1/runs/nonexistent")
    assert resp.status_code == 404


# -- Iterations ---------------------------------------------------------------

def test_list_iterations(client):
    resp = client.get("/api/v1/runs/test-run/iterations")
    assert resp.status_code == 200
    iters = resp.json()
    assert len(iters) == 2
    assert iters[0]["iteration_num"] == 1
    assert iters[0]["outcome"] == "improved"
    assert iters[0]["metric_value"] == 0.75
    assert iters[1]["iteration_num"] == 2
    assert iters[1]["outcome"] == "regressed"


def test_list_iterations_with_limit(client):
    resp = client.get("/api/v1/runs/test-run/iterations?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_list_best_iterations(client):
    resp = client.get("/api/v1/runs/test-run/iterations/best?direction=maximize&limit=1")
    assert resp.status_code == 200
    best = resp.json()
    assert len(best) == 1
    assert best[0]["metric_value"] == 0.75


# -- Metrics ------------------------------------------------------------------

def test_list_metrics(client):
    resp = client.get("/api/v1/runs/test-run/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert len(metrics) == 2
    assert metrics[0]["value"] == 0.75
    assert metrics[1]["value"] == 0.70


# -- Epoch Metrics ------------------------------------------------------------

def test_list_epoch_metrics(client):
    resp = client.get("/api/v1/runs/test-run/epochs?iteration_num=1")
    assert resp.status_code == 200
    epochs = resp.json()
    assert len(epochs) == 2
    assert epochs[0]["epoch"] == 1
    assert epochs[0]["metrics"]["loss"] == 0.5
    assert epochs[1]["metrics"]["mAP"] == 0.75


def test_list_all_epochs(client):
    resp = client.get("/api/v1/runs/test-run/epochs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# -- GPU Snapshots ------------------------------------------------------------

def test_list_gpu_snapshots(client):
    resp = client.get("/api/v1/runs/test-run/gpu")
    assert resp.status_code == 200
    snaps = resp.json()
    assert len(snaps) == 1
    assert snaps[0]["utilization_pct"] == 85.0
    assert snaps[0]["memory_total_mb"] == 8192.0


def test_get_gpu_latest(client):
    resp = client.get("/api/v1/runs/test-run/gpu/latest")
    assert resp.status_code == 200
    snap = resp.json()
    assert snap["temperature_c"] == 72.0


def test_get_gpu_latest_no_data(client):
    resp = client.get("/api/v1/runs/nonexistent/gpu/latest")
    assert resp.status_code == 404


# -- Future placeholders ------------------------------------------------------

def test_diff_placeholder(client):
    resp = client.get("/api/v1/runs/test-run/diff/1/2")
    assert resp.status_code == 501


def test_analytics_placeholder(client):
    resp = client.get("/api/v1/runs/analytics")
    assert resp.status_code == 501


def test_early_stop_placeholder(client):
    resp = client.post("/api/v1/runs/test-run/early-stop")
    assert resp.status_code == 501


def test_files_placeholder(client):
    resp = client.get("/api/v1/runs/test-run/files")
    assert resp.status_code == 501


# -- Agent WebSocket ----------------------------------------------------------

def test_agent_ws_gpu_metrics(client, db_path):
    """Remote agent pushes GPU metrics via WebSocket and they land in the DB."""
    import json
    import sqlite3 as _sqlite3
    from autotrain.storage.queries import get_gpu_snapshots

    with client.websocket_connect("/ws/agent/test-run") as ws:
        ws.send_json({
            "type": "gpu_metrics",
            "ts": "2026-03-28T12:00:00Z",
            "gpus": [{
                "gpu_index": 0,
                "utilization_pct": 95.0,
                "memory_used_mb": 7000.0,
                "memory_total_mb": 8192.0,
                "temperature_c": 80.0,
            }],
        })
        # Send heartbeat to verify connection stays alive
        ws.send_json({"type": "heartbeat", "ts": "2026-03-28T12:00:01Z", "uptime_s": 1})

    # Verify GPU data was written to DB
    conn = _sqlite3.connect(str(db_path))
    conn.row_factory = _sqlite3.Row
    snapshots = get_gpu_snapshots(conn, "test-run", limit=100)
    conn.close()

    # Should have the original fixture snapshot + the one from agent
    assert len(snapshots) >= 2
    latest = snapshots[-1]
    assert latest.utilization_pct == 95.0
    assert latest.temperature_c == 80.0


def test_agent_ws_connection_tracking(db_path):
    """Agent connection manager correctly tracks connected agents."""
    app = create_app(db_path)
    agent_mgr = app.state.agent_manager

    # No agent connected initially
    assert not agent_mgr.is_agent_connected("test-run")

    client = TestClient(app)
    with client.websocket_connect("/ws/agent/test-run") as ws:
        ws.send_json({"type": "heartbeat", "ts": "2026-03-28T12:00:00Z", "uptime_s": 0})
        # Agent should be tracked as connected
        assert agent_mgr.is_agent_connected("test-run")

    # After disconnect, agent should no longer be tracked
    assert not agent_mgr.is_agent_connected("test-run")
