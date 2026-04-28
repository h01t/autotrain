"""Dashboard control service — manages run lifecycle from the UI.

Handles run creation, starting, stopping, restarting, preflight checks,
and config validation. Runs are managed in a background thread pool so the
API remains responsive.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import yaml

from autotrain.config.loader import load_config
from autotrain.config.schema import RunConfig
from autotrain.core.agent_loop import AgentLoop
from autotrain.storage.db import init_db
from autotrain.storage.models import Run, RunStatus
from autotrain.storage.queries import create_run as create_run_record
from autotrain.storage.queries import get_run, update_run_status

from .models import (
    ConfigValidationError,
    CreateRunRequest,
    CreateRunResponse,
    PreflightGpuInfo,
    PreflightRequest,
    PreflightResponse,
    PreflightResult,
    ResumeRunResponse,
    RunActionResponse,
    RunStatusResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)

log = structlog.get_logger()


def _parse_yaml_safe(yaml_str: str) -> dict[str, Any] | None:
    """Parse YAML string, returning None on failure."""
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            return None
        return data
    except yaml.YAMLError:
        return None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_config(request: ValidateConfigRequest) -> ValidateConfigResponse:
    """Validate a run configuration YAML without creating or starting a run."""
    errors: list[ConfigValidationError] = []
    warnings: list[str] = []

    data = _parse_yaml_safe(request.config_yaml)
    if data is None:
        return ValidateConfigResponse(
            valid=False,
            errors=[ConfigValidationError(
                field="(root)",
                message="Invalid YAML: could not parse configuration.",
            )],
        )

    # Check required top-level field: repo_path
    repo_path_str = data.get("repo_path")
    if not repo_path_str:
        errors.append(ConfigValidationError(
            field="repo_path",
            message="repo_path is required.",
        ))
        return ValidateConfigResponse(valid=False, errors=errors, warnings=warnings)

    repo_path = Path(repo_path_str)
    if not repo_path.exists():
        errors.append(ConfigValidationError(
            field="repo_path",
            message=f"Directory does not exist: {repo_path}",
        ))

    # Validate metric config
    metric = data.get("metric", {})
    if not metric.get("name"):
        errors.append(ConfigValidationError(
            field="metric.name",
            message="metric.name is required.",
        ))
    if metric.get("target") is None:
        errors.append(ConfigValidationError(
            field="metric.target",
            message="metric.target is required.",
        ))

    # Check for train.py or custom train_command
    execution = data.get("execution", {})
    train_cmd = execution.get("train_command", "python train.py")
    if train_cmd == "python train.py" and not (repo_path / "train.py").exists():
        py_files = list(repo_path.glob("*.py")) if repo_path.exists() else []
        warnings.append(
            f"Default train_command 'python train.py' but train.py not found. "
            f"Available .py files: {[f.name for f in py_files] or 'none'}.",
        )

    # Validate SSH config if mode is ssh
    if execution.get("mode") == "ssh":
        if not execution.get("ssh_host"):
            errors.append(ConfigValidationError(
                field="execution.ssh_host",
                message="ssh_host is required when mode='ssh'.",
            ))
        if not execution.get("ssh_remote_dir"):
            errors.append(ConfigValidationError(
                field="execution.ssh_remote_dir",
                message="ssh_remote_dir is required when mode='ssh'.",
            ))

    # Try full Pydantic validation
    try:
        load_config(repo_path, cli_overrides=data)
    except Exception as e:
        errors.append(ConfigValidationError(
            field="(validation)",
            message=f"Config validation failed: {e}",
        ))

    return ValidateConfigResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# GPU preflight
# ---------------------------------------------------------------------------


def run_preflight(request: PreflightRequest) -> PreflightResponse:
    """Run GPU and environment preflight checks."""
    start = time.monotonic()
    checks: list[PreflightResult] = []
    gpus: list[PreflightGpuInfo] = []

    repo_path = Path(request.repo_path).resolve()

    # Check 1: repo exists
    if repo_path.exists():
        checks.append(PreflightResult(
            check="repo_exists",
            passed=True,
            message=f"Repository found: {repo_path}",
        ))
    else:
        checks.append(PreflightResult(
            check="repo_exists",
            passed=False,
            message=f"Repository not found: {repo_path}",
            detail="The specified repo_path does not exist on disk. "
                   "Create the directory and add your training script first.",
            suggestion=f"mkdir -p {repo_path} && touch {repo_path}/train.py",
        ))

    # Check 2: train command file exists
    train_parts = request.train_command.split()
    train_script = (
        train_parts[-1]
        if len(train_parts) > 1 and train_parts[0] in ("python", "python3")
        else None
    )
    if train_script:
        script_path = repo_path / train_script
        if script_path.exists():
            checks.append(PreflightResult(
                check="train_script",
                passed=True,
                message=f"Training script found: {train_script}",
            ))
        else:
            checks.append(PreflightResult(
                check="train_script",
                passed=False,
                message=f"Training script not found: {train_script}",
                detail=f"The file '{train_script}' was not found in {repo_path}. "
                       f"Make sure your training script exists.",
                suggestion=f"Create {script_path} with your training code.",
            ))
    else:
        checks.append(PreflightResult(
            check="train_command",
            passed=True,
            message=f"Custom train command: {request.train_command}",
        ))

    # Check 3: GPU availability (local mode)
    if request.mode == "local":
        gpus, gpu_check = _check_local_gpu(request.gpu_device)
        checks.append(gpu_check)

    # Check 4: Python environment
    python_check = _check_python(request.venv_activate)
    checks.append(python_check)

    # Check 5: writable directory
    write_check = _check_writable(repo_path)
    checks.append(write_check)

    # Check 6: SSH connectivity (if applicable)
    if request.mode == "ssh" and request.ssh_host:
        ssh_check = _check_ssh(request.ssh_host, request.ssh_port)
        checks.append(ssh_check)

    duration = time.monotonic() - start
    all_passed = all(c.passed for c in checks)

    return PreflightResponse(
        passed=all_passed,
        checks=checks,
        gpus=gpus,
        duration_seconds=round(duration, 2),
    )


def _check_local_gpu(gpu_device: str | None) -> tuple[list[PreflightGpuInfo], PreflightResult]:
    """Check for local GPU availability."""
    gpus: list[PreflightGpuInfo] = []

    # Try nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return gpus, PreflightResult(
            check="gpu_available",
            passed=False,
            message="nvidia-smi not found — no NVIDIA GPU driver detected.",
            detail="The nvidia-smi command was not found on your PATH. "
                   "This means either no NVIDIA GPU is installed, or the "
                   "NVIDIA drivers are not set up correctly.",
            suggestion="Install NVIDIA drivers: https://www.nvidia.com/download/",
        )

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return gpus, PreflightResult(
                check="gpu_available",
                passed=False,
                message=f"nvidia-smi failed: {result.stderr.strip()}",
                detail="The GPU driver is present but nvidia-smi returned an error. "
                       "This can happen if the GPU is in use by another process "
                       "or if the driver is in a bad state.",
                suggestion="Run 'nvidia-smi' manually to diagnose.",
            )

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(PreflightGpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=float(parts[2]),
                    memory_free_mb=float(parts[3]),
                    utilization_pct=float(parts[4]),
                ))

        if gpu_device is not None:
            try:
                dev_idx = int(gpu_device)
                gpus = [g for g in gpus if g.index == dev_idx]
            except ValueError:
                pass

        if gpus:
            free_mb = sum(g.memory_free_mb or 0 for g in gpus)
            return gpus, PreflightResult(
                check="gpu_available",
                passed=True,
                message=f"{len(gpus)} GPU(s) found, {free_mb:.0f} MB free.",
            )
        else:
            return gpus, PreflightResult(
                check="gpu_available",
                passed=False,
                message="No GPUs detected by nvidia-smi.",
                detail="nvidia-smi ran successfully but returned zero GPUs. "
                       "This is unusual — check if GPUs are visible to the system.",
                suggestion="Run 'lspci | grep -i nvidia' to verify hardware.",
            )

    except FileNotFoundError:
        return gpus, PreflightResult(
            check="gpu_available",
            passed=False,
            message="nvidia-smi not found.",
            detail="GPU driver tools are not installed.",
            suggestion="Install NVIDIA CUDA toolkit.",
        )
    except subprocess.TimeoutExpired:
        return gpus, PreflightResult(
            check="gpu_available",
            passed=False,
            message="nvidia-smi timed out after 15 seconds.",
            detail="The GPU driver is not responding. This can indicate "
                   "a hung GPU or driver issue.",
            suggestion="Try rebooting or resetting the GPU.",
        )


def _check_python(venv_activate: str | None) -> PreflightResult:
    """Check Python environment availability."""
    if venv_activate:
        activate_path = Path(venv_activate)
        if activate_path.exists():
            return PreflightResult(
                check="python_env",
                passed=True,
                message=f"Virtual environment found: {venv_activate}",
            )
        else:
            return PreflightResult(
                check="python_env",
                passed=False,
                message=f"Virtual env activate script not found: {venv_activate}",
                detail=f"The file '{venv_activate}' does not exist. "
                       f"Make sure your virtual environment is set up.",
                suggestion="python -m venv .venv && .venv/bin/pip install -r requirements.txt",
            )

    # Check system python
    python = shutil.which("python3") or shutil.which("python")
    if python:
        try:
            result = subprocess.run(
                [python, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            version = result.stdout.strip() or result.stderr.strip()
            return PreflightResult(
                check="python_env",
                passed=True,
                message=f"Python found: {version}",
            )
        except Exception:
            pass

    return PreflightResult(
        check="python_env",
        passed=False,
        message="No Python interpreter found on PATH.",
        detail="Neither 'python3' nor 'python' was found. "
               "Install Python 3.12+ to continue.",
        suggestion="Install Python: https://www.python.org/downloads/",
    )


def _check_writable(repo_path: Path) -> PreflightResult:
    """Check that the repository directory is writable."""
    autotrain_dir = repo_path / ".autotrain"
    try:
        if repo_path.exists():
            autotrain_dir.mkdir(parents=True, exist_ok=True)
            test_file = autotrain_dir / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            return PreflightResult(
                check="directory_writable",
                passed=True,
                message="Repository directory is writable.",
            )
        else:
            return PreflightResult(
                check="directory_writable",
                passed=False,
                message="Repository directory does not exist.",
                detail=f"The path '{repo_path}' doesn't exist yet.",
                suggestion=f"mkdir -p {repo_path}",
            )
    except (OSError, PermissionError) as e:
        return PreflightResult(
            check="directory_writable",
            passed=False,
            message=f"Cannot write to repository: {e}",
            detail="AutoTrain needs write access to create .autotrain/ "
                   "for its database and working state.",
            suggestion=f"chmod -R u+w {repo_path}",
        )


def _check_ssh(host: str, port: int) -> PreflightResult:
    """Check SSH connectivity using a simple echo command."""
    ssh = shutil.which("ssh")
    if not ssh:
        return PreflightResult(
            check="ssh_connectivity",
            passed=False,
            message="ssh command not found on PATH.",
            detail="OpenSSH client is required for remote execution.",
            suggestion=(
                "Install OpenSSH: brew install openssh (macOS) "
                "or apt install openssh-client (Linux)."
            ),
        )

    try:
        result = subprocess.run(
            [
                ssh,
                "-p", str(port),
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                host,
                "echo ok",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return PreflightResult(
                check="ssh_connectivity",
                passed=True,
                message=f"SSH connection to {host}:{port} successful.",
            )
        else:
            return PreflightResult(
                check="ssh_connectivity",
                passed=False,
                message=f"SSH connection to {host}:{port} failed.",
                detail=f"Exit code: {result.returncode}. Stderr: {result.stderr.strip()[:500]}",
                suggestion=f"Verify SSH key is set up: ssh-copy-id {host}",
            )
    except subprocess.TimeoutExpired:
        return PreflightResult(
            check="ssh_connectivity",
            passed=False,
            message=f"SSH connection to {host}:{port} timed out.",
            detail="The server did not respond within 15 seconds.",
            suggestion="Check firewall rules and network connectivity.",
        )
    except Exception as e:
        return PreflightResult(
            check="ssh_connectivity",
            passed=False,
            message=f"SSH check error: {e}",
            detail=str(e),
            suggestion="Verify SSH configuration and try again.",
        )


# ---------------------------------------------------------------------------
# Run manager
# ---------------------------------------------------------------------------


class _RunHandle:
    """Internal handle for a running training task."""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.loop: AgentLoop | None = None
        self.start_time: float = 0.0
        self.pid: int | None = None
        self.stop_requested: bool = False


class RunManager:
    """Manages active training runs in background threads."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._active_runs: dict[str, _RunHandle] = {}
        self._lock = threading.Lock()

    # -- Shared helpers -------------------------------------------------------

    def _find_active_run_on_repo(self, repo_path: Path) -> str | None:
        """Return the run_id of an active run on the same repo, or None."""
        resolved = str(repo_path.resolve())
        with self._lock:
            for active_id, handle in self._active_runs.items():
                if handle.thread is not None and handle.thread.is_alive():
                    try:
                        conn = init_db(self._db_path)
                        active_run = get_run(conn, active_id)
                        conn.close()
                    except Exception:
                        continue
                    if (
                        active_run is not None
                        and str(Path(active_run.repo_path).resolve()) == resolved
                    ):
                        return active_id
        return None

    def _run_quick_preflight(
        self, repo_path: Path, train_command: str = "python train.py",
    ) -> list[PreflightResult]:
        """Run essential preflight checks and return failures (empty = all passed)."""
        failures: list[PreflightResult] = []
        resolved = repo_path.resolve()
        if not resolved.is_dir():
            failures.append(PreflightResult(
                check="repo_exists", passed=False,
                message=f"Repository not found: {resolved}",
                suggestion=f"mkdir -p {resolved}",
            ))
            return failures  # Can't do further checks without a repo

        # Writable check
        autotrain_dir = resolved / ".autotrain"
        try:
            autotrain_dir.mkdir(parents=True, exist_ok=True)
            test = autotrain_dir / ".write_test"
            test.write_text("ok")
            test.unlink()
        except (OSError, PermissionError) as e:
            failures.append(PreflightResult(
                check="directory_writable", passed=False,
                message=f"Cannot write to repository: {e}",
                suggestion=f"chmod -R u+w {resolved}",
            ))

        # Train script check
        train_parts = train_command.split()
        if len(train_parts) > 1 and train_parts[0] in ("python", "python3"):
            script_path = resolved / train_parts[-1]
            if not script_path.exists():
                failures.append(PreflightResult(
                    check="train_script", passed=False,
                    message=f"Training script not found: {train_parts[-1]}",
                    suggestion=f"Create {script_path} with training code.",
                ))

        return failures

    @property
    def active_run_ids(self) -> list[str]:
        with self._lock:
            return list(self._active_runs.keys())

    def get_run_status(self, run_id: str) -> RunStatusResponse:
        """Get the current status of a run."""
        with self._lock:
            handle = self._active_runs.get(run_id)

        conn = init_db(self._db_path)
        try:
            run = get_run(conn, run_id)
            if run is None:
                return RunStatusResponse(
                    run_id=run_id, status="not_found", is_active=False,
                )

            is_active = (
                handle is not None
                and handle.thread is not None
                and handle.thread.is_alive()
            )
            return RunStatusResponse(
                run_id=run_id,
                status=run.status.value,
                is_active=is_active,
                pid=handle.pid if handle else None,
                uptime_seconds=(
                    time.monotonic() - handle.start_time
                    if handle and handle.start_time else None
                ),
            )
        finally:
            conn.close()

    def create_and_start(
        self, request: CreateRunRequest,
    ) -> CreateRunResponse:
        """Create a run record and optionally start it in the background.

        Returns a CreateRunResponse with status 'invalid_config' for
        validation errors.  Callers must check the status and return
        HTTP 400 for 'invalid_config'.
        """
        # Parse and validate config
        data = _parse_yaml_safe(request.config_yaml)
        if data is None:
            return CreateRunResponse(
                run_id="",
                status="invalid_config",
                message="Invalid YAML: could not parse configuration.",
                config_errors=[
                    ConfigValidationError(
                        field="(root)",
                        message="Invalid YAML: could not parse configuration.",
                    ),
                ],
            )

        repo_path_str = data.get("repo_path", "")
        if not repo_path_str:
            return CreateRunResponse(
                run_id="",
                status="invalid_config",
                message="repo_path is required.",
                config_errors=[
                    ConfigValidationError(
                        field="repo_path",
                        message="repo_path is required.",
                    ),
                ],
            )

        repo_path = Path(repo_path_str).resolve()

        # ── Same-repo active-run guard (only when starting) ──
        if request.start_immediately:
            conflict_id = self._find_active_run_on_repo(repo_path)
            if conflict_id is not None:
                return CreateRunResponse(
                    run_id="",
                    status="conflict",
                    message=(
                        f"Another run (id={conflict_id}) is already active "
                        f"on repository {repo_path}. Stop it before "
                        f"starting a new run."
                    ),
                    config_errors=[
                        ConfigValidationError(
                            field="repo_path",
                            message=(
                                f"Another run (id={conflict_id}) is already "
                                "active on this repository."
                            ),
                        ),
                    ],
                )

        # Load full config
        try:
            config = load_config(repo_path, cli_overrides=data)
        except Exception as e:
            return CreateRunResponse(
                run_id="",
                status="invalid_config",
                message=f"Config validation failed: {e}",
                config_errors=[
                    ConfigValidationError(
                        field="(validation)",
                        message=str(e),
                    ),
                ],
            )

        # Create storage record
        run_id = str(uuid.uuid4())[:8]
        conn = init_db(self._db_path)
        try:
            run = Run(
                id=run_id,
                repo_path=str(repo_path),
                metric_name=config.metric.name,
                metric_target=config.metric.target,
                metric_direction=config.metric.direction,
                status=RunStatus.RUNNING if request.start_immediately else RunStatus.STOPPED,
                config_snapshot=config.model_dump_json(),
            )
            create_run_record(conn, run)
        finally:
            conn.close()

        if request.start_immediately:
            # ── Quick preflight gate ─────────────────────────────────
            failures = self._run_quick_preflight(
                repo_path, config.execution.train_command,
            )
            if failures:
                return CreateRunResponse(
                    run_id=run_id,
                    status="invalid_config",
                    message=f"Preflight failed: {failures[0].message}",
                    config_errors=[
                        ConfigValidationError(field="preflight", message=f.message)
                        for f in failures
                    ],
                )
            self._start_run_async(run_id, config)

        return CreateRunResponse(
            run_id=run_id,
            status="running" if request.start_immediately else "stopped",
            message=f"Run {run_id} created{' and started' if request.start_immediately else ''}.",
        )

    def start_run(self, run_id: str) -> RunActionResponse:
        """Start a previously created or stopped run."""
        conn = init_db(self._db_path)
        try:
            run = get_run(conn, run_id)
            if run is None:
                return RunActionResponse(
                    run_id=run_id, action="start",
                    success=False, message="Run not found.",
                )

            # ── Same-repo active-run guard ──────────────────────
            repo_path = Path(run.repo_path).resolve()
            conflict_id = self._find_active_run_on_repo(repo_path)
            if conflict_id is not None and conflict_id != run_id:
                return RunActionResponse(
                    run_id=run_id, action="start",
                    success=False,
                    message=(
                        f"Another run (id={conflict_id}) is already active "
                        f"on this repository. Stop it first."
                    ),
                    previous_status=run.status.value,
                    new_status=run.status.value,
                )

            previous_status = run.status.value

            if run.status == RunStatus.RUNNING:
                with self._lock:
                    if run_id in self._active_runs:
                        handle = self._active_runs[run_id]
                        if handle.thread is not None and handle.thread.is_alive():
                            return RunActionResponse(
                                run_id=run_id, action="start",
                                success=False, message="Run is already running.",
                                previous_status=previous_status,
                                new_status=previous_status,
                            )

            # Load config from snapshot
            config_data = json.loads(run.config_snapshot) if run.config_snapshot else {}
            repo_path = Path(run.repo_path)
            config = load_config(repo_path, cli_overrides=config_data)

            # ── Quick preflight gate ─────────────────────────────────
            failures = self._run_quick_preflight(
                Path(run.repo_path), config.execution.train_command,
            )
            if failures:
                return RunActionResponse(
                    run_id=run_id, action="start",
                    success=False,
                    message=f"Preflight failed: {failures[0].message}",
                    previous_status=previous_status,
                    new_status=previous_status,
                )

            update_run_status(conn, run_id, RunStatus.RUNNING)
            self._start_run_async(run_id, config)

            return RunActionResponse(
                run_id=run_id, action="start",
                success=True, message=f"Run {run_id} started.",
                previous_status=previous_status,
                new_status="running",
            )
        finally:
            conn.close()

    def stop_run(self, run_id: str) -> RunActionResponse:
        """Stop a running run.

        Preserves terminal statuses (FAILED, COMPLETED, BUDGET_EXHAUSTED).
        Only transitions RUNNING -> STOPPED.
        """
        conn = init_db(self._db_path)
        try:
            run = get_run(conn, run_id)
            if run is None:
                return RunActionResponse(
                    run_id=run_id, action="stop",
                    success=False, message="Run not found.",
                )

            previous_status = run.status.value

            # Preserve terminal statuses — don't overwrite them
            if run.status in (
                RunStatus.FAILED,
                RunStatus.COMPLETED,
                RunStatus.BUDGET_EXHAUSTED,
            ):
                return RunActionResponse(
                    run_id=run_id, action="stop",
                    success=True,
                    message=(
                        f"Run is already in terminal state "
                        f"'{run.status.value}' — not modified."
                    ),
                    previous_status=previous_status,
                    new_status=previous_status,
                )

            with self._lock:
                handle = self._active_runs.get(run_id)
                if handle is None:
                    update_run_status(conn, run_id, RunStatus.STOPPED)
                    return RunActionResponse(
                        run_id=run_id, action="stop",
                        success=True,
                        message="Run was not active. Status set to stopped.",
                        previous_status=previous_status,
                        new_status="stopped",
                    )

                # Signal the agent loop to stop
                handle.stop_requested = True
                if handle.loop is not None:
                    handle.loop._executor.kill()

            # Wait briefly for thread to finish
            if handle.thread is not None:
                handle.thread.join(timeout=5.0)
            update_run_status(conn, run_id, RunStatus.STOPPED)

            with self._lock:
                self._active_runs.pop(run_id, None)

            return RunActionResponse(
                run_id=run_id, action="stop",
                success=True, message=f"Run {run_id} stopped.",
                previous_status=previous_status,
                new_status="stopped",
            )
        finally:
            conn.close()

    def restart_run(self, run_id: str) -> RunActionResponse:
        """Restart a failed, stopped, or completed run."""
        conn = init_db(self._db_path)
        try:
            run = get_run(conn, run_id)
            if run is None:
                return RunActionResponse(
                    run_id=run_id, action="restart",
                    success=False, message="Run not found.",
                )

            previous_status = run.status.value

            # Stop if running
            if run.status == RunStatus.RUNNING:
                stop_result = self.stop_run(run_id)
                if not stop_result.success:
                    return RunActionResponse(
                        run_id=run_id, action="restart",
                        success=False,
                        message=f"Failed to stop running run: {stop_result.message}",
                        previous_status=previous_status,
                    )

            # Now start
            start_result = self.start_run(run_id)
            start_result.action = "restart"
            start_result.previous_status = previous_status
            return start_result
        finally:
            conn.close()

    def resume_run(
        self, prior_run_id: str, start_immediately: bool = True,
    ) -> ResumeRunResponse:
        """Create a new run that resumes from a prior run's state.

        The new run:
        - Gets a unique run_id
        - Links to prior_run_id via resumed_from_run_id
        - Copies the prior run's config
        - Optionally starts immediately

        Returns 409 status if another run is active on the same repo.
        """
        conn = init_db(self._db_path)
        try:
            prior_run = get_run(conn, prior_run_id)
            if prior_run is None:
                return ResumeRunResponse(
                    new_run_id="", prior_run_id=prior_run_id,
                    status="not_found",
                    message=f"Prior run {prior_run_id} not found.",
                )

            # ── Same-repo active-run guard ──────────────────────
            if start_immediately:
                repo_path = Path(prior_run.repo_path).resolve()
                conflict_id = self._find_active_run_on_repo(repo_path)
                if conflict_id is not None:
                    return ResumeRunResponse(
                        new_run_id="",
                        prior_run_id=prior_run_id,
                        status="conflict",
                        message=(
                            f"Another run (id={conflict_id}) is already active "
                            f"on this repository. Stop it first."
                        ),
                    )

            # Determine if there's a checkpoint to resume from
            has_checkpoint = False
            checkpoint_path = (
                Path(prior_run.repo_path) / ".autotrain" / "checkpoints"
                / prior_run_id
            )
            if checkpoint_path.is_dir():
                pth_files = list(checkpoint_path.glob("*.pt")) + list(
                    checkpoint_path.glob("*.pth")
                )
                if pth_files:
                    has_checkpoint = True

            # Load prior config
            config_data = (
                json.loads(prior_run.config_snapshot)
                if prior_run.config_snapshot else {}
            )
            repo_path = Path(prior_run.repo_path)
            config = load_config(repo_path, cli_overrides=config_data)

            # Create new run record
            new_run_id = str(uuid.uuid4())[:8]
            new_run = Run(
                id=new_run_id,
                repo_path=prior_run.repo_path,
                metric_name=prior_run.metric_name,
                metric_target=prior_run.metric_target,
                metric_direction=prior_run.metric_direction,
                status=(
                    RunStatus.RUNNING if start_immediately
                    else RunStatus.STOPPED
                ),
                config_snapshot=prior_run.config_snapshot,
                resumed_from_run_id=prior_run_id,
            )
            create_run_record(conn, new_run)
        finally:
            conn.close()

        if start_immediately:
            # ── Quick preflight gate ─────────────────────────────────
            failures = self._run_quick_preflight(
                repo_path, config.execution.train_command,
            )
            if failures:
                return ResumeRunResponse(
                    new_run_id=new_run_id,
                    prior_run_id=prior_run_id,
                    status="invalid_config",
                    message=f"Preflight failed: {failures[0].message}",
                )
            self._start_run_async(new_run_id, config)

        return ResumeRunResponse(
            new_run_id=new_run_id,
            prior_run_id=prior_run_id,
            status="running" if start_immediately else "stopped",
            message=(
                f"Resume run {new_run_id} created from {prior_run_id}"
                f"{' and started' if start_immediately else ''}."
            ),
            resumed_from_checkpoint=has_checkpoint,
        )

    def _start_run_async(self, run_id: str, config: RunConfig) -> None:
        """Launch the agent loop in a background thread."""
        handle = _RunHandle()
        handle.stop_requested = False

        def _run_target() -> None:
            try:
                loop = AgentLoop(config, run_id=run_id)
                handle.loop = loop
                final_status = loop.run()

                conn = init_db(self._db_path)
                try:
                    update_run_status(conn, run_id, final_status)
                finally:
                    conn.close()

                log.info("run_completed", run_id=run_id, status=final_status.value)
            except Exception as e:
                log.error("run_thread_error", run_id=run_id, error=str(e))
                conn = init_db(self._db_path)
                try:
                    update_run_status(conn, run_id, RunStatus.FAILED)
                finally:
                    conn.close()
            finally:
                with self._lock:
                    self._active_runs.pop(run_id, None)

        handle.thread = threading.Thread(
            target=_run_target,
            name=f"autotrain-run-{run_id}",
            daemon=True,
        )
        handle.start_time = time.monotonic()
        handle.pid = os.getpid()

        with self._lock:
            self._active_runs[run_id] = handle

        handle.thread.start()
        log.info("run_started_async", run_id=run_id)
