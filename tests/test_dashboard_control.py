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
    """Creating a run with invalid YAML should return 400."""
    resp = client.post("/api/v1/runs", json={
        "config_yaml": "not: valid: yaml: :::",
        "start_immediately": False,
    })
    assert resp.status_code == 400
    data = resp.json()["detail"]
    assert data["status"] == "invalid_config"
    assert len(data["config_errors"]) >= 1


def test_create_run_missing_repo_path(client):
    """Creating a run without repo_path should return 400."""
    resp = client.post("/api/v1/runs", json={
        "config_yaml": "metric:\n  name: loss\n  target: 0.5",
        "start_immediately": False,
    })
    assert resp.status_code == 400
    data = resp.json()["detail"]
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


# ============================================================================
# Milestone #2 revisions — new tests
# ============================================================================


# ---------------------------------------------------------------------------
# Same-repo active-run guard (409 Conflict)
# ---------------------------------------------------------------------------


def test_create_run_conflict_same_repo(client, tmp_path: Path):
    """Creating a run on a repo with an active run should return 409."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp1 = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    assert resp1.status_code == 201
    run1_id = resp1.json()["run_id"]

    resp2 = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    assert resp2.status_code == 409
    detail = resp2.json()["detail"]
    assert detail["status"] == "conflict"
    assert run1_id in detail["message"]


# ---------------------------------------------------------------------------
# Stop preserves terminal status
# ---------------------------------------------------------------------------


def test_stop_preserves_terminal_status(client, tmp_path: Path):
    """Stopping a FAILED run should not change its status."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    # Manually set to FAILED via DB
    import sqlite3

    from autotrain.storage.models import RunStatus
    from autotrain.storage.queries import update_run_status as _update_status
    db_file = tmp_path / ".autotrain" / "state.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    _update_status(conn, run_id, RunStatus.FAILED)
    conn.close()

    resp2 = client.post(f"/api/v1/runs/{run_id}/stop")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["success"] is True
    assert data["new_status"] == "failed"
    assert "terminal state" in data["message"].lower()


# ---------------------------------------------------------------------------
# Resume endpoint
# ---------------------------------------------------------------------------


def test_resume_creates_new_run(client, tmp_path: Path):
    """Resume should create a new run record linked to prior run."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.post(f"/api/v1/runs/{run_id}/resume", json={
        "run_id": run_id,
        "start_immediately": False,
    })
    assert resp2.status_code == 201
    data = resp2.json()
    assert data["new_run_id"]
    assert data["new_run_id"] != run_id
    assert data["prior_run_id"] == run_id
    assert data["status"] == "stopped"


def test_resume_nonexistent_run(client):
    """Resuming a nonexistent run should return 404."""
    resp = client.post("/api/v1/runs/nonexistent/resume", json={
        "run_id": "nonexistent",
        "start_immediately": False,
    })
    assert resp.status_code == 404


def test_resume_persists_link(client, tmp_path: Path):
    """Resumed run should store resumed_from_run_id in DB."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.post(f"/api/v1/runs/{run_id}/resume", json={
        "run_id": run_id,
        "start_immediately": False,
    })
    new_run_id = resp2.json()["new_run_id"]

    resp3 = client.get(f"/api/v1/runs/{new_run_id}")
    assert resp3.status_code == 200
    data = resp3.json()
    assert data["resumed_from_run_id"] == run_id


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------


def test_get_run_config(client, tmp_path: Path):
    """GET /runs/{run_id}/config should return config as YAML and JSON."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(f"/api/v1/runs/{run_id}/config")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["run_id"] == run_id
    assert data["config_yaml"] is not None
    assert data["config_json"] is not None
    assert data["config_json"]["repo_path"] == str(repo)


def test_config_not_found(client):
    """Config for nonexistent run should 404."""
    resp = client.get("/api/v1/runs/nonexistent/config")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Logs endpoint
# ---------------------------------------------------------------------------


def test_get_run_logs(client, tmp_path: Path):
    """GET /runs/{run_id}/logs should return log lines."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")
    log_dir = repo / ".autotrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_output.log").write_text("epoch 1\nepoch 2\nepoch 3\n")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(f"/api/v1/runs/{run_id}/logs")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["run_id"] == run_id
    assert len(data["lines"]) == 3
    assert data["total_lines"] == 3
    assert data["truncated"] is False


def test_get_run_logs_empty(client, tmp_path: Path):
    """Logs for run without log file should return empty."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(f"/api/v1/runs/{run_id}/logs")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["lines"] == []


def test_get_logs_truncated(client, tmp_path: Path):
    """Logs should be truncated when exceeding tail limit."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")
    log_dir = repo / ".autotrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_output.log").write_text(
        "\n".join(f"line {i}" for i in range(300))
    )

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(f"/api/v1/runs/{run_id}/logs?tail=50")
    assert resp2.status_code == 200
    data = resp2.json()
    assert len(data["lines"]) == 50
    assert data["total_lines"] == 300
    assert data["truncated"] is True


# ---------------------------------------------------------------------------
# Artifacts endpoint (safety-hardened)
# ---------------------------------------------------------------------------


def test_list_artifacts_empty(client, tmp_path: Path):
    """GET /runs/{run_id}/artifacts with no artifacts should return empty."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["artifacts"] == []


def test_list_artifacts_with_files(client, tmp_path: Path):
    """Artifacts endpoint should list artifact files."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    artifacts_dir = repo / ".autotrain" / "artifacts" / run_id
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "model.pt").write_text("checkpoint data")
    (artifacts_dir / "config.yaml").write_text("lr: 0.001")

    resp2 = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert resp2.status_code == 200
    data = resp2.json()
    assert len(data["artifacts"]) == 2
    names = {a["name"] for a in data["artifacts"]}
    assert names == {"model.pt", "config.yaml"}


def test_download_artifact(client, tmp_path: Path):
    """Download should serve an artifact file."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    artifacts_dir = repo / ".autotrain" / "artifacts" / run_id
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "model.pt").write_text("checkpoint data")

    resp2 = client.get(f"/api/v1/runs/{run_id}/artifacts/model.pt")
    assert resp2.status_code == 200
    assert resp2.text == "checkpoint data"


def test_artifact_path_traversal_blocked(client, tmp_path: Path):
    """Path traversal in artifact download should be blocked (403).

    Uses URL-encoded '..' which bypasses Starlette's path normalization
    and reaches our handler with literal traversal segments.
    """
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    run_id = resp.json()["run_id"]

    resp2 = client.get(
        f"/api/v1/runs/{run_id}/artifacts/%2e%2e/%2e%2e/%2e%2e/etc/passwd"
    )
    assert resp2.status_code == 403, f"Got {resp2.status_code}: {resp2.text}"


# ---------------------------------------------------------------------------
# Defaults endpoint
# ---------------------------------------------------------------------------


def test_get_defaults(client):
    """GET /defaults should return a YAML template."""
    resp = client.get("/api/v1/defaults")
    assert resp.status_code == 200
    data = resp.json()
    assert data["config_yaml"]
    assert "repo_path:" in data["config_yaml"]
    assert "metric:" in data["config_yaml"]


# ---------------------------------------------------------------------------
# Save config endpoint
# ---------------------------------------------------------------------------


def test_save_config(client, tmp_path: Path):
    """POST /save-config should write autotrain.yaml to repo."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    resp = client.post("/api/v1/save-config", json={
        "repo_path": str(repo),
        "config_yaml": _yaml_config(str(repo)),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert (repo / "autotrain.yaml").exists()
    assert "val_loss" in (repo / "autotrain.yaml").read_text()


def test_save_config_bad_yaml(client, tmp_path: Path):
    """Invalid YAML should be rejected."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    resp = client.post("/api/v1/save-config", json={
        "repo_path": str(repo),
        "config_yaml": "{{{ bad yaml",
    })
    assert resp.status_code == 400


def test_save_config_nonexistent_repo(client):
    """Nonexistent repo should be rejected."""
    resp = client.post("/api/v1/save-config", json={
        "repo_path": "/nonexistent/repo",
        "config_yaml": "metric:\n  name: loss",
    })
    assert resp.status_code == 400


def test_save_config_schema_validation(client, tmp_path: Path):
    """Config with missing required fields should fail schema validation."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    resp = client.post("/api/v1/save-config", json={
        "repo_path": str(repo),
        "config_yaml": f"repo_path: {repo}\nmetric:\n  name: loss\n",
    })
    # Valid config with metric name but missing target should fail
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Same-repo conflict guarding for start and resume
# ---------------------------------------------------------------------------


def test_start_conflict_same_repo(client, tmp_path: Path):
    """Starting a run on a repo with another active run should return 409."""
    repo1 = tmp_path / "test-repo"
    repo1.mkdir()
    (repo1 / "train.py").write_text("print('train1')")

    # Create and start run on repo1
    resp1 = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo1)),
        "start_immediately": True,
    })
    assert resp1.status_code == 201

    # Create a stopped run also on repo1 (should succeed — guard only fires on start)
    resp2 = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo1)),
        "start_immediately": False,
    })
    assert resp2.status_code == 201, f"Got {resp2.status_code}: {resp2.json()}"
    run2_id = resp2.json()["run_id"]

    # Try to start run2 — should be blocked because run1 is active on same repo
    resp3 = client.post(f"/api/v1/runs/{run2_id}/start")
    assert resp3.status_code == 409, f"Got {resp3.status_code}: {resp3.json()}"
    detail = resp3.json()["detail"]
    assert "already active" in detail["message"].lower()


def test_resume_conflict_same_repo(client, tmp_path: Path):
    """Resuming a run on a repo with another active run should return 409."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create and start a run
    resp1 = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": True,
    })
    assert resp1.status_code == 201
    run1_id = resp1.json()["run_id"]

    # Try to resume another run on same repo (the resume itself creates
    # a new run and try to start it on the same repo)
    resp2 = client.post(f"/api/v1/runs/{run1_id}/resume", json={
        "run_id": run1_id,
        "start_immediately": True,
    })
    assert resp2.status_code == 409, f"Got {resp2.status_code}: {resp2.json()}"


# ---------------------------------------------------------------------------
# Duplicate run row prevention
# ---------------------------------------------------------------------------


def test_no_duplicate_run_rows(client, tmp_path: Path):
    """AgentLoop must not create a second run record when created via dashboard."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')")

    # Create via dashboard (should create exactly 1 row)
    resp = client.post("/api/v1/runs", json={
        "config_yaml": _yaml_config(str(repo)),
        "start_immediately": False,
    })
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    # Query the DB directly to count rows
    import sqlite3
    db_file = tmp_path / ".autotrain" / "state.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT COUNT(*) as cnt FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.close()
    assert rows["cnt"] == 1, f"Expected 1 row for run_id={run_id}, got {rows['cnt']}"
