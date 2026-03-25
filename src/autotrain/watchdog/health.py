"""Health checks for GPU, disk, and process monitoring."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

log = structlog.get_logger()


def check_disk_space(path: Path, min_gb: float = 1.0) -> bool:
    """Check if enough disk space is available."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        log.warning("disk_space_low", free_gb=round(free_gb, 2), min_gb=min_gb)
        return False
    return True


def check_gpu_memory(min_free_mb: int = 100) -> bool | None:
    """Check GPU memory. Returns None if no GPU / nvidia-smi not available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        free_mb = int(result.stdout.strip().split("\n")[0])
        if free_mb < min_free_mb:
            log.warning("gpu_memory_low", free_mb=free_mb, min_mb=min_free_mb)
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def check_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        import os
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
