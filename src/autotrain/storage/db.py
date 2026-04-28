"""SQLite database connection and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

log = structlog.get_logger()

SCHEMA_VERSION = 5

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
    resumed_from_run_id TEXT,
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
    checkpoint_path TEXT,
    resumed_from_checkpoint INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS epoch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    iteration_num INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    gpu_index INTEGER NOT NULL DEFAULT 0,
    utilization_pct REAL,
    memory_used_mb REAL,
    memory_total_mb REAL,
    temperature_c REAL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id, iteration_num);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metric_snapshots(run_id, iteration_num);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal(run_id, iteration_num);
CREATE INDEX IF NOT EXISTS idx_epoch_metrics ON epoch_metrics(run_id, iteration_num, epoch);
CREATE INDEX IF NOT EXISTS idx_gpu_snapshots_run ON gpu_snapshots(run_id, timestamp);
"""

MIGRATION_V2 = """
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    iteration_num INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_epoch_metrics ON epoch_metrics(run_id, iteration_num, epoch);
"""

MIGRATION_V3 = """
ALTER TABLE iterations ADD COLUMN checkpoint_path TEXT;
ALTER TABLE iterations ADD COLUMN resumed_from_checkpoint INTEGER NOT NULL DEFAULT 0;
"""

MIGRATION_V4 = """
CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    gpu_index INTEGER NOT NULL DEFAULT 0,
    utilization_pct REAL,
    memory_used_mb REAL,
    memory_total_mb REAL,
    temperature_c REAL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gpu_snapshots_run ON gpu_snapshots(run_id, timestamp);
"""

MIGRATION_V5 = """
ALTER TABLE runs ADD COLUMN resumed_from_run_id TEXT;
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
            current = row["version"]
            log.info("migrating_database", from_version=current, to_version=SCHEMA_VERSION)
            if current < 2:
                conn.executescript(MIGRATION_V2)
            if current < 3:
                conn.executescript(MIGRATION_V3)
            if current < 4:
                conn.executescript(MIGRATION_V4)
            if current < 5:
                conn.executescript(MIGRATION_V5)
            conn.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
            conn.commit()

    return conn
