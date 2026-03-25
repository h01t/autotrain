"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from autotrain.storage.db import init_db
from autotrain.storage.models import Run, RunStatus
from autotrain.storage.queries import create_run


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Provide an initialized SQLite connection."""
    db_path = tmp_path / ".autotrain" / "state.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def db_conn_with_run(db_conn: sqlite3.Connection) -> sqlite3.Connection:
    """Provide a db connection with a pre-created run (for FK constraints)."""
    run = Run(
        id="run-1",
        repo_path="/test/repo",
        metric_name="val_auc",
        metric_target=0.85,
        metric_direction="maximize",
        status=RunStatus.RUNNING,
    )
    create_run(db_conn, run)
    return db_conn


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a train.py."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('training...')\n")
    (repo / "config.yaml").write_text("lr: 0.001\n")
    return repo
