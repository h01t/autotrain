"""SQLite database connection and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

log = structlog.get_logger()

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    repo_path TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_target REAL NOT NULL,
    metric_direction TEXT NOT NULL DEFAULT 'maximize',
    status TEXT NOT NULL DEFAULT 'running',
    best_metric_value REAL,
    best_iteration INTEGER,
    total_iterations INTEGER NOT NULL DEFAULT 0,
    total_api_cost REAL NOT NULL DEFAULT 0.0,
    git_branch TEXT,
    config_snapshot TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    iteration_num INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT '',
    outcome TEXT,
    metric_value REAL,
    commit_hash TEXT,
    agent_reasoning TEXT,
    agent_hypothesis TEXT,
    changes_summary TEXT,
    duration_seconds REAL,
    api_cost REAL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    iteration_num INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    iteration_num INTEGER NOT NULL,
    state TEXT NOT NULL,
    data TEXT,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id, iteration_num);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metric_snapshots(run_id, iteration_num);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal(run_id, iteration_num);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and foreign keys enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with schema, returning the connection."""
    conn = get_connection(db_path)

    # Check if schema exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cursor.fetchone() is None:
        log.info("initializing_database", path=str(db_path))
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    else:
        # Check version for future migrations
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row and row["version"] < SCHEMA_VERSION:
            log.info("migrating_database", from_version=row["version"], to_version=SCHEMA_VERSION)
            # Future: run migration scripts here
            conn.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
            conn.commit()

    return conn
