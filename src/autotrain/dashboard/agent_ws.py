"""Server-side handler for remote agent WebSocket connections.

Receives GPU metrics and log lines from the remote agent, writes to SQLite
for persistence, and relays immediately to browser clients via the browser
ConnectionManager.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from autotrain.storage.models import GpuSnapshot
from autotrain.storage.queries import record_gpu_snapshot

from .serializers import serialize_gpu_snapshot
from .ws import ConnectionManager as BrowserManager

log = structlog.get_logger()


class AgentConnectionManager:
    """Manages WebSocket connections from remote metrics agents."""

    def __init__(self, browser_manager: BrowserManager, db_path: Path) -> None:
        self._browser = browser_manager
        self._db_path = db_path
        self._agents: dict[str, WebSocket] = {}  # run_id -> agent ws
        self._last_heartbeat: dict[str, float] = {}

    def is_agent_connected(self, run_id: str) -> bool:
        """Check if a remote agent is actively connected for this run."""
        if run_id not in self._agents:
            return False
        # Consider stale if no heartbeat for 30s
        last = self._last_heartbeat.get(run_id, 0)
        return (time.monotonic() - last) < 30

    async def handle_agent_ws(self, websocket: WebSocket, run_id: str) -> None:
        """Accept and handle an agent WebSocket connection."""
        await websocket.accept()
        self._agents[run_id] = websocket
        self._last_heartbeat[run_id] = time.monotonic()
        log.info("agent_ws_connected", run_id=run_id)

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "gpu_metrics":
                    await self._handle_gpu_metrics(run_id, data)
                elif msg_type == "log_line":
                    await self._handle_log_line(run_id, data)
                elif msg_type == "heartbeat":
                    self._last_heartbeat[run_id] = time.monotonic()

        except WebSocketDisconnect:
            log.info("agent_ws_disconnected", run_id=run_id)
        except Exception as e:
            log.warning("agent_ws_error", run_id=run_id, error=str(e))
        finally:
            self._agents.pop(run_id, None)
            self._last_heartbeat.pop(run_id, None)

    async def _handle_gpu_metrics(self, run_id: str, data: dict) -> None:
        """Process GPU metrics: write to DB + relay to browsers."""
        gpus = data.get("gpus", [])

        for gpu_data in gpus:
            snapshot = GpuSnapshot(
                run_id=run_id,
                gpu_index=gpu_data.get("gpu_index", 0),
                utilization_pct=gpu_data.get("utilization_pct"),
                memory_used_mb=gpu_data.get("memory_used_mb"),
                memory_total_mb=gpu_data.get("memory_total_mb"),
                temperature_c=gpu_data.get("temperature_c"),
            )

            # Write to DB in a thread to avoid blocking the event loop
            await asyncio.to_thread(self._write_gpu_snapshot, snapshot)

            # Relay to browser clients immediately
            await self._browser.broadcast(run_id, {
                "type": "gpu_snapshot",
                "data": serialize_gpu_snapshot(snapshot),
            })

    def _write_gpu_snapshot(self, snapshot: GpuSnapshot) -> None:
        """Write GPU snapshot to SQLite (runs in thread)."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            record_gpu_snapshot(conn, snapshot)
            conn.close()
        except Exception as e:
            log.debug("agent_gpu_write_failed", error=str(e))

    async def _handle_log_line(self, run_id: str, data: dict) -> None:
        """Relay training log line to browser clients."""
        await self._browser.broadcast(run_id, {
            "type": "log_line",
            "data": {
                "line": data.get("line", ""),
                "ts": data.get("ts", datetime.now(timezone.utc).isoformat()),
            },
        })
