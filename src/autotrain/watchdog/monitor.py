"""Watchdog monitor thread — polls health checks periodically."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from autotrain.config.schema import WatchdogConfig
from autotrain.storage.models import GpuSnapshot
from autotrain.storage.queries import record_gpu_snapshot
from autotrain.util.signals import is_shutting_down
from autotrain.watchdog.health import check_disk_space

log = structlog.get_logger()


class WatchdogMonitor:
    """Background thread that monitors training health."""

    def __init__(
        self,
        config: WatchdogConfig,
        repo_path: Path,
        on_alert: Callable | None = None,
        gpu_query_fn: Callable[[], list[dict]] | None = None,
        db_conn: sqlite3.Connection | None = None,
        run_id: str | None = None,
    ) -> None:
        self._config = config
        self._repo_path = repo_path
        self._on_alert = on_alert
        self._gpu_query_fn = gpu_query_fn
        self._db_conn = db_conn
        self._run_id = run_id
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

        # GPU metrics collection + memory alert
        self._collect_gpu_metrics()

        # Stdout stagnation
        minutes_silent = (
            (time.monotonic() - self._last_stdout_time) / 60
        )
        if minutes_silent > self._config.stdout_stagnation_minutes:
            self._alert(
                f"Training silent for {minutes_silent:.0f} minutes"
            )

    def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics, store to DB, and check memory thresholds."""
        if not self._gpu_query_fn:
            return

        try:
            gpus = self._gpu_query_fn()
        except Exception as e:
            log.debug("gpu_query_failed", error=str(e))
            return

        for gpu in gpus:
            # Check memory threshold
            free_mb = gpu.get("memory_total_mb", 0) - gpu.get("memory_used_mb", 0)
            if free_mb < self._config.gpu_memory_min_mb:
                self._alert(
                    f"GPU {gpu.get('gpu_index', 0)} memory critically low: "
                    f"{free_mb:.0f}MB free"
                )

            # Store to DB if available
            if self._db_conn and self._run_id:
                try:
                    snapshot = GpuSnapshot(
                        run_id=self._run_id,
                        gpu_index=gpu.get("gpu_index", 0),
                        utilization_pct=gpu.get("utilization_pct"),
                        memory_used_mb=gpu.get("memory_used_mb"),
                        memory_total_mb=gpu.get("memory_total_mb"),
                        temperature_c=gpu.get("temperature_c"),
                    )
                    record_gpu_snapshot(self._db_conn, snapshot)
                except Exception as e:
                    log.debug("gpu_snapshot_write_failed", error=str(e))

    def _alert(self, message: str) -> None:
        log.warning("watchdog_alert", message=message)
        self._alerts.append(message)
        if self._on_alert:
            self._on_alert(message)
