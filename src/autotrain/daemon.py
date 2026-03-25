"""Daemon mode — PID file management and background execution."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from autotrain.config.defaults import AUTOTRAIN_DIR, PID_FILE

log = structlog.get_logger()


def write_pid(repo_path: Path) -> Path:
    """Write current PID to file. Returns PID file path."""
    pid_path = repo_path / AUTOTRAIN_DIR / PID_FILE
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    return pid_path


def read_pid(repo_path: Path) -> int | None:
    """Read PID from file. Returns None if not found."""
    pid_path = repo_path / AUTOTRAIN_DIR / PID_FILE
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def remove_pid(repo_path: Path) -> None:
    """Remove PID file."""
    pid_path = repo_path / AUTOTRAIN_DIR / PID_FILE
    pid_path.unlink(missing_ok=True)


def is_running(repo_path: Path) -> bool:
    """Check if an autotrain process is running for this repo."""
    pid = read_pid(repo_path)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        # Stale PID file
        remove_pid(repo_path)
        return False
