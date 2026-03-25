"""SSH executor — run training on a remote GPU via SSH + rsync.

Key resilience pattern: training runs with nohup on the remote, writing to a log file.
If SSH drops, training continues. We reconnect and resume tailing.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import structlog

from autotrain.execution.base import ExecutionResult

log = structlog.get_logger()


class SSHExecutor:
    """Execute training on a remote machine via SSH."""

    def __init__(
        self,
        host: str,
        remote_dir: str,
        ssh_key: str | None = None,
        ssh_port: int = 22,
        rsync_excludes: list[str] | None = None,
    ) -> None:
        self._host = host
        self._remote_dir = remote_dir
        self._ssh_port = ssh_port
        self._rsync_excludes = rsync_excludes or []
        self._result: ExecutionResult | None = None
        self._start_time: float = 0

        self._ssh_opts = [
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-p", str(ssh_port),
        ]
        if ssh_key:
            self._ssh_opts.extend(["-i", ssh_key])

    def setup(self) -> None:
        """Verify SSH connectivity and create remote directory."""
        try:
            self._ssh_run("echo ok", timeout=15)
        except subprocess.CalledProcessError as e:
            raise ConnectionError(
                f"Cannot connect to {self._host}: {e.stderr}"
            ) from e

        self._ssh_run(f"mkdir -p {self._remote_dir}/.autotrain", timeout=10)
        log.info("ssh_setup_complete", host=self._host, remote_dir=self._remote_dir)

    def sync_files(self, repo_path: Path, files: list[str] | None = None) -> None:
        """rsync project files to the remote machine."""
        cmd = ["rsync", "-avz", "--checksum", "--delete"]
        for exc in self._rsync_excludes:
            cmd.extend(["--exclude", exc])

        src = f"{repo_path}/"
        dst = f"{self._host}:{self._remote_dir}/"

        if self._ssh_port != 22:
            cmd.extend(["-e", f"ssh -p {self._ssh_port}"])

        cmd.extend([src, dst])

        log.info("ssh_rsync_start", src=src, dst=dst)
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        log.info("ssh_rsync_complete")

    def execute(
        self,
        command: str,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Execute command remotely with setsid for resilience.

        1. Kill any leftover process from a previous iteration.
        2. Start training with setsid (new process group), writing to a log file.
        3. Tail the remote log via a separate SSH connection.
        4. If SSH drops, training continues. We can reconnect.
        """
        self._start_time = time.monotonic()
        self._result = None

        remote_log = f"{self._remote_dir}/.autotrain/train_output.log"
        remote_pid = f"{self._remote_dir}/.autotrain/train.pid"

        # Kill any leftover process from a previous iteration
        if self._is_remote_process_alive(remote_pid):
            log.info("ssh_killing_previous_process")
            self._kill_process_group(remote_pid)

        # Build env prefix
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"{k}={v}" for k, v in env.items()) + " "

        # Start training with setsid (new process group) + nohup for full detach
        start_cmd = (
            f"cd {self._remote_dir} && "
            f"setsid bash -c '{env_prefix}{command}' > {remote_log} 2>&1 </dev/null & "
            f"echo $! > {remote_pid} && disown"
        )
        self._ssh_run(start_cmd, timeout=60)
        log.info("ssh_training_started", command=command)

        # Tail the remote log
        yield from self._tail_remote_log(remote_log, remote_pid, timeout_seconds)

    def _tail_remote_log(
        self,
        remote_log: str,
        remote_pid: str,
        timeout_seconds: int,
    ) -> Iterator[str]:
        """Tail remote log with reconnect on SSH drop."""
        deadline = time.monotonic() + timeout_seconds
        lines_seen = 0

        while time.monotonic() < deadline:
            try:
                # tail from where we left off
                skip_cmd = f"tail -n +{lines_seen + 1} -f {remote_log}"
                proc = subprocess.Popen(
                    ["ssh"] + self._ssh_opts + [self._host, skip_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                for line in iter(proc.stdout.readline, b""):
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    lines_seen += 1
                    yield decoded

                    if time.monotonic() > deadline:
                        proc.terminate()
                        self._finalize_result(remote_pid, was_timeout=True)
                        return

                proc.wait()

                # tail -f ended — check if training is done
                if not self._is_remote_process_alive(remote_pid):
                    break

            except (subprocess.SubprocessError, OSError) as e:
                log.warning("ssh_tail_disconnected", error=str(e))
                time.sleep(5)  # Wait before reconnect
                continue

        self._finalize_result(remote_pid, was_timeout=time.monotonic() >= deadline)

    def _finalize_result(self, remote_pid: str, was_timeout: bool = False) -> None:
        """Build the execution result from remote state."""
        duration = time.monotonic() - self._start_time

        # Get exit code from remote
        exit_code = -1
        try:
            result = self._ssh_run(
                f"wait $(cat {remote_pid} 2>/dev/null) 2>/dev/null; echo $?",
                timeout=10,
                check=False,
            )
            exit_code = int(result.stdout.strip()) if result.stdout.strip().isdigit() else -1
        except Exception:
            pass

        self._result = ExecutionResult(
            exit_code=exit_code,
            stdout="",
            stderr="",
            duration_seconds=duration,
            was_timeout=was_timeout,
        )
        log.info("ssh_exec_done", exit_code=exit_code, duration=f"{duration:.1f}s")

    def get_result(self) -> ExecutionResult:
        if self._result is None:
            raise RuntimeError("No execution result. Call execute() first.")
        return self._result

    def fetch_results(self, patterns: list[str], dest: Path) -> None:
        """rsync result files back from remote."""
        dest.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            cmd = ["rsync", "-avz"]
            if self._ssh_port != 22:
                cmd.extend(["-e", f"ssh -p {self._ssh_port}"])
            cmd.extend([
                f"{self._host}:{self._remote_dir}/{pattern}",
                str(dest) + "/",
            ])
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            except subprocess.CalledProcessError:
                log.debug("ssh_fetch_pattern_missing", pattern=pattern)

    def is_process_alive(self) -> bool:
        """Check if remote training process is alive."""
        remote_pid = f"{self._remote_dir}/.autotrain/train.pid"
        return self._is_remote_process_alive(remote_pid)

    def kill(self) -> None:
        """Kill the remote training process group."""
        remote_pid = f"{self._remote_dir}/.autotrain/train.pid"
        self._kill_process_group(remote_pid)

        if self._result is None:
            self._result = ExecutionResult(
                exit_code=-9, stdout="", stderr="Killed",
                duration_seconds=time.monotonic() - self._start_time,
                was_killed=True,
            )

    def cleanup(self) -> None:
        """Kill remote process if still running."""
        self.kill()

    def _ssh_run(
        self, command: str, timeout: int = 30, check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command over SSH."""
        return subprocess.run(
            ["ssh"] + self._ssh_opts + [self._host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def _kill_process_group(self, remote_pid: str) -> None:
        """Kill the entire process group rooted at the PID in remote_pid file."""
        try:
            # Kill entire process group (setsid makes PID == PGID),
            # then also kill the individual PID as fallback
            self._ssh_run(
                f"PID=$(cat {remote_pid} 2>/dev/null) && "
                f"kill -- -$PID 2>/dev/null; kill $PID 2>/dev/null",
                timeout=10,
                check=False,
            )
            log.info("ssh_process_group_killed")
        except Exception:
            pass

    def _is_remote_process_alive(self, remote_pid: str) -> bool:
        """Check if remote PID is still running."""
        try:
            result = self._ssh_run(
                f"kill -0 $(cat {remote_pid} 2>/dev/null) 2>/dev/null && echo alive",
                timeout=10,
                check=False,
            )
            return "alive" in result.stdout
        except Exception:
            return False
