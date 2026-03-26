"""GPU metrics collection via nvidia-smi — local and SSH."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import structlog

log = structlog.get_logger()

_NVIDIA_SMI_CMD = (
    "nvidia-smi "
    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
    "--format=csv,noheader,nounits"
)


def query_gpu_local() -> list[dict]:
    """Query GPU metrics locally via nvidia-smi."""
    try:
        result = subprocess.run(
            _NVIDIA_SMI_CMD,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        return _parse_nvidia_smi(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def query_gpu_ssh(ssh_run_fn: Callable) -> list[dict]:
    """Query GPU metrics on a remote host via an SSH runner function.

    ssh_run_fn should match SSHExecutor._ssh_run signature:
        ssh_run_fn(command: str, timeout: int, check: bool) -> CompletedProcess
    """
    try:
        result = ssh_run_fn(_NVIDIA_SMI_CMD, timeout=10, check=False)
        if result.returncode != 0:
            return []
        return _parse_nvidia_smi(result.stdout)
    except Exception:
        return []


def _parse_nvidia_smi(output: str) -> list[dict]:
    """Parse nvidia-smi CSV output into a list of GPU metric dicts.

    Expected format (one line per GPU):
        0, 87, 4521, 8192, 72
    """
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "gpu_index": int(parts[0]),
                "utilization_pct": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
                "temperature_c": float(parts[4]),
            })
        except (ValueError, IndexError):
            continue
    return gpus
