"""Tests for dashboard run creation and control endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrain.dashboard.server import create_app
from autotrain.storage.db import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create and seed a test database."""
    db_file = tmp_path / ".autotrain" / "state.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def client(db_path: Path) -> TestClient:
    """Create a test client with the dashboard app."""
    app = create_app(db_path)
    return TestClient(app)


def _yaml_config(repo_path: str, **overrides: str) -> str:
    """Build a minimal valid YAML config."""
    return f"""repo_path: {repo_path}
metric:
  name: val_loss
  target: 0.1
  direction: minimize
{chr(10).join(f"{k}: {v}" for k, v in overrides.items())}
"""


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_validate_config_valid(client, tmp_path: Path):
    """Valid config should pass validation."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": _yaml_config(str(repo)),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert len(data["errors"]) == 0


def test_validate_config_invalid_yaml(client):
    """Garbage YAML should fail validation."""
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": "{{{ bad yaml [[[",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) >= 1
    assert "YAML" in data["errors"][0]["message"]


def test_validate_config_missing_repo_path(client):
    """Config without repo_path should fail."""
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": "metric:\n  name: loss\n  target: 0.5",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert any("repo_path" in e["field"] for e in data["errors"])


def test_validate_config_nonexistent_repo(client, tmp_path: Path):
    """Config with nonexistent repo should fail."""
    repo = tmp_path / "nonexistent"
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": _yaml_config(str(repo)),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert any("exist" in e["message"].lower() for e in data["errors"])


def test_validate_config_ssh_missing_fields(client, tmp_path: Path):
    """SSH mode without host should fail."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    yaml_str = f"""repo_path: {repo}
metric:
  name: loss
  target: 0.5
execution:
  mode: ssh
  ssh_remote_dir: /remote
"""
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": yaml_str,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False


def test_validate_config_empty_yaml(client):
    """Empty YAML should fail."""
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": "",
    })
    assert resp.status_code == 422  # Pydantic validation: min_length=1


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_repo_exists(client, tmp_path: Path):
    """Preflight with existing repo should pass basic checks."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": str(repo),
        "mode": "local",
        "train_command": "python train.py",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "passed" in data
    assert "checks" in data
    assert len(data["checks"]) >= 3  # repo, train_script, python, writable


def test_preflight_repo_not_found(client, tmp_path: Path):
    """Preflight with nonexistent repo should report failure."""
    repo = tmp_path / "nonexistent"
    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": str(repo),
        "mode": "local",
    })
    assert resp.status_code == 200
    data = resp.json()
    repo_check = [c for c in data["checks"] if c["check"] == "repo_exists"][0]
    assert repo_check["passed"] is False
    assert repo_check["detail"] is not None
    assert repo_check["suggestion"] is not None


def test_preflight_train_script_missing(client, tmp_path: Path):
    """Preflight should detect missing train script."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": str(repo),
        "mode": "local",
        "train_command": "python train.py",
    })
    assert resp.status_code == 200
    data = resp.json()
    script_check = [c for c in data["checks"] if c["check"] == "train_script"][0]
    assert script_check["passed"] is False


def test_preflight_custom_train_command(client, tmp_path: Path):
    """Custom train commands should be accepted."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": str(repo),
        "mode": "local",
        "train_command": ".venv/bin/python -m trainer.main",
    })
    assert resp.status_code == 200
    data = resp.json()
    cmd_check = [c for c in data["checks"] if c["check"] == "train_command"]
    assert len(cmd_check) == 1
    assert cmd_check[0]["passed"] is True


def test_preflight_response_structure(client, tmp_path: Path):
    """Preflight response should have all expected fields."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": str(repo),
        "mode": "local",
        "train_command": "python train.py",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "passed" in data
    assert "checks" in data
    assert "gpus" in data
    assert "duration_seconds" in data
    assert isinstance(data["duration_seconds"], (int, float))
    for check in data["checks"]:
        assert "check" in check
        assert "passed" in check
        assert "message" in check
        # detail and suggestion are optional but should be present in failures


# ---------------------------------------------------------------------------
# Run creation
# ---------------------------------------------------------------------------


def test_create_run_valid(client, tmp_path: Path):
    """Creating a run with valid config should succeed."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"]
    assert data["status"] == "stopped"
    assert "created" in data["message"]


def test_create_run_invalid_yaml(client):
    """Creating a run with invalid YAML should return errors."""
    resp = client.post("/api/v1/runs", json={
        "config_yaml": "not: valid: yaml: :::",
        "start_immediately": False,
    })
    assert resp.status_code == 201  # 201 even on error — response body has details
    data = resp.json()
    assert data["status"] == "invalid_config"
    assert len(data["config_errors"]) >= 1


def test_create_run_missing_repo_path(client):
    """Creating a run without repo_path should fail."""
    resp = client.post("/api/v1/runs", json={
        "config_yaml": "metric:\n  name: loss\n  target: 0.5",
        "start_immediately": False,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "invalid_config"


def test_create_run_start_immediately(client, tmp_path: Path):
    """start_immediately=True should work without errors."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"]
    assert data["status"] == "running"


def test_create_run_persisted(client, tmp_path: Path):
    """Created run should appear in GET /runs."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    # Verify it appears in the list
    resp2 = client.get("/api/v1/runs")
    runs = resp2.json()
    assert any(r["id"] == run_id for r in runs)


# ---------------------------------------------------------------------------
# Run control — start/stop/restart
# ---------------------------------------------------------------------------


def test_run_status_endpoint(client, tmp_path: Path, db_path: Path):
    """GET /runs/{run_id}/status should return status info."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create a run
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    # Check status
    resp2 = client.get(f"/api/v1/runs/{run_id}/status")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["run_id"] == run_id
    assert data["status"] in ("stopped", "running")
    assert "is_active" in data


def test_run_status_not_found(client):
    """Status for nonexistent run should return not_found."""
    resp = client.get("/api/v1/runs/nonexistent/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"


def test_start_nonexistent_run(client):
    """Starting a nonexistent run should fail."""
    resp = client.post("/api/v1/runs/nonexistent/start")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "not found" in data["message"].lower()


def test_stop_nonexistent_run(client):
    """Stopping a nonexistent run should fail."""
    resp = client.post("/api/v1/runs/nonexistent/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "not found" in data["message"].lower()


def test_restart_nonexistent_run(client):
    """Restarting a nonexistent run should fail."""
    resp = client.post("/api/v1/runs/nonexistent/restart")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False


def test_start_stopped_run(client, tmp_path: Path):
    """Starting a stopped run should succeed."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create stopped run
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    # Start it
    resp2 = client.post(f"/api/v1/runs/{run_id}/start")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["success"] is True
    assert data["new_status"] == "running"


def test_stop_running_run(client, tmp_path: Path):
    """Stopping a running run should succeed."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create started run
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    run_id = resp.json()["run_id"]

    # Stop it
    resp2 = client.post(f"/api/v1/runs/{run_id}/stop")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["success"] is True
    assert data["new_status"] == "stopped"


def test_restart_run(client, tmp_path: Path):
    """Restarting a stopped run should succeed."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create stopped run
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    # Restart it
    resp2 = client.post(f"/api/v1/runs/{run_id}/restart")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["success"] is True
    assert data["action"] == "restart"
    assert data["new_status"] == "running"


def test_start_already_running(client, tmp_path: Path):
    """Starting an already running run should return gracefully."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create started run
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    run_id = resp.json()["run_id"]

    # Try to start again — may succeed or fail depending on timing
    resp2 = client.post(f"/api/v1/runs/{run_id}/start")
    assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# Request validation (Pydantic)
# ---------------------------------------------------------------------------


def test_create_run_empty_config(client):
    """Empty config_yaml should be rejected by Pydantic."""
    resp = client.post("/api/v1/runs", json={
        "config_yaml": "",
        "start_immediately": False,
    })
    assert resp.status_code == 422


def test_validate_config_empty(client):
    """Empty config_yaml should be rejected by Pydantic."""
    resp = client.post("/api/v1/runs/validate-config", json={
        "config_yaml": "",
    })
    assert resp.status_code == 422


def test_preflight_empty_repo_path(client):
    """Empty repo_path should result in a failed preflight check."""
    resp = client.post("/api/v1/runs/preflight", json={
        "repo_path": "",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Empty path should fail repo_exists check
    assert data["passed"] is False


def test_create_run_missing_start_immediately(client, tmp_path: Path):
    """start_immediately defaults to True."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "running"  # default is True


def test_run_action_response_structure(client, tmp_path: Path):
    """Verify RunActionResponse has all expected fields."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.post(f"/api/v1/runs/{run_id}/start")
    data = resp2.json()
    assert "run_id" in data
    assert "action" in data
    assert "success" in data
    assert "message" in data
    assert "previous_status" in data
    assert "new_status" in data
