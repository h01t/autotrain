"""SSH executor — run training on a remote GPU via SSH + rsync.

Key resilience pattern: training runs with setsid on the remote (new process group),
writing to a log file. If SSH drops, training continues. We reconnect and resume tailing.
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
        setup_command: str | None = None,
    ) -> None:
        self._host = host
        self._remote_dir = remote_dir
        self._ssh_port = ssh_port
        self._rsync_excludes = rsync_excludes or []
        self._setup_command = setup_command
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

    def setup(self, repo_path: Path | None = None) -> None:
        """Verify SSH connectivity, create remote directory, and run setup command."""
        try:
            self._ssh_run("echo ok", timeout=15)
        except subprocess.CalledProcessError as e:
            raise ConnectionError(
                f"Cannot connect to {self._host}: {e.stderr}"
            ) from e

        self._ssh_run(f"mkdir -p {self._remote_dir}/.autotrain", timeout=10)

        # Sync files first so requirements.txt/pyproject.toml are on the remote
        if repo_path and self._setup_command:
            self.sync_files(repo_path)
            log.info("ssh_running_setup_command", command=self._setup_command)
            self._ssh_run(
                f"cd {self._remote_dir} && {self._setup_command}",
                timeout=300,
            )
            log.info("ssh_setup_command_complete")

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

        # Start training in a fully detached process group.
        # We write a small launcher script to avoid SSH fd-inheritance hangs:
        # setsid creates a new process group, nohup prevents SIGHUP,
        # and the script writes the PID then exits — SSH returns immediately.
        launcher = f"{self._remote_dir}/.autotrain/_launch.sh"
        remote_exit = f"{self._remote_dir}/.autotrain/train.exit"
        script = (
            f"#!/bin/bash\n"
            f"cd {self._remote_dir}\n"
            f"rm -f {remote_exit}\n"
            f"nohup setsid bash -c '{env_prefix}{command}; echo $? > {remote_exit}' "
            f"> {remote_log} 2>&1 </dev/null &\n"
            f"echo $! > {remote_pid}\n"
        )
        self._ssh_run(f"cat > {launcher} << 'LAUNCH_EOF'\n{script}LAUNCH_EOF", timeout=10)
        self._ssh_run(f"chmod +x {launcher} && bash {launcher}", timeout=15)
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
                # tail from where we left off, using --pid so tail exits
                # when the training process dies
                skip_cmd = (
                    f"tail -n +{lines_seen + 1} -f "
                    f"--pid=$(cat {remote_pid} 2>/dev/null || echo 1) "
                    f"{remote_log}"
                )
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

        # Read exit code from file written by launcher script
        remote_exit = f"{self._remote_dir}/.autotrain/train.exit"
        exit_code = -1
        try:
            result = self._ssh_run(
                f"cat {remote_exit} 2>/dev/null",
                timeout=10,
                check=False,
            )
            code_str = result.stdout.strip()
            exit_code = int(code_str) if code_str.isdigit() else -1
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

    def detect_checkpoint(self, patterns: list[str]) -> str | None:
        """Find the most recent checkpoint file on the remote."""
        # Build a single ls command for all patterns
        paths = " ".join(
            f"{self._remote_dir}/{p}" for p in patterns
        )
        try:
            result = self._ssh_run(
                f"ls -t {paths} 2>/dev/null | head -1",
                timeout=10,
                check=False,
            )
            path = result.stdout.strip()
            if path:
                log.info("ssh_checkpoint_detected", path=path)
                return path
        except Exception:
            pass
        return None

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
