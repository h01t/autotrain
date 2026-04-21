"""Local subprocess executor for training on the same machine."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import structlog

from autotrain.execution.base import ExecutionResult

log = structlog.get_logger()


class LocalExecutor:
    """Execute training commands as local subprocesses."""

    def __init__(self, working_dir: Path | None = None) -> None:
        self._working_dir = working_dir
        self._process: subprocess.Popen | None = None
        self._result: ExecutionResult | None = None
        self._start_time: float = 0
        self._output_lines: list[str] = []

    def setup(self, repo_path: Path | None = None) -> None:
        """Validate working directory exists."""
        if self._working_dir and not self._working_dir.exists():
            raise FileNotFoundError(f"Working directory not found: {self._working_dir}")

    def sync_files(self, repo_path: Path, files: list[str] | None = None) -> None:
        """No-op for local execution — files are already local."""
        pass

    def execute(
        self,
        command: str,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Execute command, yielding stdout lines in real-time."""
        full_env = {**os.environ, **(env or {})}

        self._start_time = time.monotonic()
        self._result = None
        self._output_lines = []

        log.info("local_exec_start", command=command, timeout=timeout_seconds)

        self._process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
            cwd=self._working_dir,
            preexec_fn=os.setsid,  # Process group for clean kill
        )

        try:
            from autotrain.util.signals import is_shutting_down

            deadline = time.monotonic() + timeout_seconds
            for line in iter(self._process.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self._output_lines.append(decoded)
                yield decoded

                if is_shutting_down():
                    log.info("local_exec_shutdown_signal")
                    self._kill_process()
                    self._result = ExecutionResult(
                        exit_code=-15,
                        stdout=self._captured_stdout(),
                        stderr="Killed: shutdown signal",
                        duration_seconds=time.monotonic() - self._start_time,
                        was_killed=True,
                    )
                    return

                if time.monotonic() > deadline:
                    log.warning("local_exec_timeout", timeout=timeout_seconds)
                    self._kill_process()
                    self._result = ExecutionResult(
                        exit_code=-1,
                        stdout=self._captured_stdout(),
                        stderr="Killed: timeout exceeded",
                        duration_seconds=time.monotonic() - self._start_time,
                        was_timeout=True,
                    )
                    return

            self._process.wait()
            duration = time.monotonic() - self._start_time

            self._result = ExecutionResult(
                exit_code=self._process.returncode,
                stdout=self._captured_stdout(),
                stderr="",
                duration_seconds=duration,
            )

            log.info(
                "local_exec_done",
                exit_code=self._process.returncode,
                duration=f"{duration:.1f}s",
            )

        except Exception:
            self._kill_process()
            raise

    def get_result(self) -> ExecutionResult:
        """Get the execution result."""
        if self._result is None:
            raise RuntimeError("No execution result available. Call execute() first.")
        return self._result

    def fetch_results(self, patterns: list[str], dest: Path) -> None:
        """No-op for local — results are already on disk."""
        pass

    def is_process_alive(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def kill(self) -> None:
        """Kill the running process."""
        self._kill_process()
        if self._result is None:
            self._result = ExecutionResult(
                exit_code=-9,
                stdout=self._captured_stdout(),
                stderr="Killed by watchdog/user",
                duration_seconds=time.monotonic() - self._start_time if self._start_time else 0,
                was_killed=True,
            )

    def detect_checkpoint(self, patterns: list[str]) -> str | None:
        """Find the most recent checkpoint file locally."""

        base = self._working_dir or Path(".")
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(base.glob(pattern))
        if not candidates:
            return None
        # Return most recently modified
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        log.info("local_checkpoint_detected", path=str(newest))
        return str(newest)

    def cleanup(self) -> None:
        """Ensure process is terminated."""
        self._kill_process()

    def _captured_stdout(self) -> str:
        """Return last 200 lines of captured output."""
        return "\n".join(self._output_lines[-200:])

    def _kill_process(self) -> None:
        """Kill the process group (SIGTERM then SIGKILL)."""
        if self._process is None or self._process.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._process.wait(timeout=5)
            log.info("local_process_killed", pid=self._process.pid)
        except (ProcessLookupError, OSError):
            pass  # Already dead
