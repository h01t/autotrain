"""Executor protocol — abstract interface for local and SSH execution."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ExecutionResult:
    """Result of a training execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    was_timeout: bool = False
    was_killed: bool = False


@runtime_checkable
class Executor(Protocol):
    """Interface for training execution backends."""

    def setup(self) -> None:
        """One-time setup: validate connectivity, create dirs, etc."""
        ...

    def sync_files(self, repo_path: Path, files: list[str] | None = None) -> None:
        """Push files to the execution environment."""
        ...

    def execute(
        self,
        command: str,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Execute a command, yielding stdout lines as they arrive."""
        ...

    def get_result(self) -> ExecutionResult:
        """Get the final result after execute() completes."""
        ...

    def fetch_results(self, patterns: list[str], dest: Path) -> None:
        """Pull result files back from the execution environment."""
        ...

    def is_process_alive(self) -> bool:
        """Check if the training process is still running."""
        ...

    def kill(self) -> None:
        """Kill the running training process."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...
