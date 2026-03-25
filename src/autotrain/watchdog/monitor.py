"""Watchdog monitor thread — polls health checks periodically."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import structlog

from autotrain.config.schema import WatchdogConfig
from autotrain.util.signals import is_shutting_down
from autotrain.watchdog.health import check_disk_space, check_gpu_memory

log = structlog.get_logger()


class WatchdogMonitor:
    """Background thread that monitors training health."""

    def __init__(
        self,
        config: WatchdogConfig,
        repo_path: Path,
        on_alert: callable | None = None,
    ) -> None:
        self._config = config
        self._repo_path = repo_path
        self._on_alert = on_alert
        self._thread: threading.Thread | None = None
        self._last_stdout_time = time.monotonic()
        self._alerts: list[str] = []

    def start(self) -> None:
        if not self._config.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("watchdog_started", interval=self._config.check_interval_seconds)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def report_stdout_activity(self) -> None:
        """Called when training produces output — resets stagnation timer."""
        self._last_stdout_time = time.monotonic()

    @property
    def alerts(self) -> list[str]:
        return list(self._alerts)

    def _run(self) -> None:
        while not is_shutting_down():
            try:
                self._check_all()
            except Exception as e:
                log.error("watchdog_check_error", error=str(e))
            time.sleep(self._config.check_interval_seconds)

    def _check_all(self) -> None:
        # Disk space
        if not check_disk_space(self._repo_path, self._config.disk_space_min_gb):
            self._alert("Disk space critically low")

        # GPU memory
        gpu_ok = check_gpu_memory(self._config.gpu_memory_min_mb)
        if gpu_ok is False:
            self._alert("GPU memory critically low")

        # Stdout stagnation
        minutes_silent = (
            (time.monotonic() - self._last_stdout_time) / 60
        )
        if minutes_silent > self._config.stdout_stagnation_minutes:
            self._alert(
                f"Training silent for {minutes_silent:.0f} minutes"
            )

    def _alert(self, message: str) -> None:
        log.warning("watchdog_alert", message=message)
        self._alerts.append(message)
        if self._on_alert:
            self._on_alert(message)
