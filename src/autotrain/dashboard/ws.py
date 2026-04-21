"""WebSocket manager — polls SQLite for changes and broadcasts to clients."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from autotrain.storage.queries import (
    get_latest_gpu_snapshot,
    get_recent_iterations,
    get_run,
)

from .serializers import (
    serialize_gpu_snapshot,
    serialize_iteration,
    serialize_run,
)

log = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections and polls DB for changes."""

    def __init__(self, db_path: Path, poll_interval: float = 2.0) -> None:
        self._db_path = db_path
        self._poll_interval = poll_interval
        self._connections: dict[str, set[WebSocket]] = {}  # run_id -> clients
        self._last_state: dict[str, dict] = {}  # run_id -> cached state
        self._poll_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        if run_id not in self._connections:
            self._connections[run_id] = set()
        self._connections[run_id].add(websocket)

        # Start polling if not already running
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

        log.debug("ws_client_connected", run_id=run_id)

    async def disconnect(self, websocket: WebSocket, run_id: str) -> None:
        self._connections.get(run_id, set()).discard(websocket)
        if run_id in self._connections and not self._connections[run_id]:
            del self._connections[run_id]
            self._last_state.pop(run_id, None)

        # Stop polling if no active connections
        if not self._connections and self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Poll DB for changes across all active run subscriptions."""
        try:
            while self._connections:
                for run_id in list(self._connections.keys()):
                    try:
                        await self._check_run(run_id)
                    except Exception as e:
                        log.debug("ws_poll_error", run_id=run_id, error=str(e))
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass

    async def _check_run(self, run_id: str) -> None:
        """Check a single run for changes and broadcast if found."""
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            run = get_run(conn, run_id)
            if run is None:
                return

            # Build current state fingerprint
            iterations = get_recent_iterations(conn, run_id, limit=1)
            max_iter_id = iterations[-1].id if iterations else 0
            gpu = get_latest_gpu_snapshot(conn, run_id)
            gpu_ts = gpu.timestamp.isoformat() if gpu else None

            current = {
                "updated_at": run.updated_at.isoformat() if run.updated_at else "",
                "max_iter_id": max_iter_id,
                "total_iterations": run.total_iterations,
                "status": run.status.value,
                "gpu_ts": gpu_ts,
            }

            prev = self._last_state.get(run_id)
            if prev == current:
                return  # No change

            self._last_state[run_id] = current

            if prev is None:
                return  # First check — establish baseline, don't spam

            # Determine what changed and broadcast
            if current["status"] != prev.get("status"):
                await self.broadcast(run_id, {
                    "type": "run_updated",
                    "data": serialize_run(run),
                })
                if current["status"] in ("completed", "budget_exhausted", "failed", "stopped"):
                    await self.broadcast(run_id, {"type": "run_completed", "data": {}})

            if current["max_iter_id"] != prev.get("max_iter_id"):
                new_iters = get_recent_iterations(conn, run_id, limit=3)
                await self.broadcast(run_id, {
                    "type": "iteration_added",
                    "data": [serialize_iteration(it) for it in new_iters],
                })

            if current["total_iterations"] != prev.get("total_iterations"):
                await self.broadcast(run_id, {
                    "type": "metric_added",
                    "data": serialize_run(run),
                })

            if current["gpu_ts"] != prev.get("gpu_ts") and gpu:
                await self.broadcast(run_id, {
                    "type": "gpu_snapshot",
                    "data": serialize_gpu_snapshot(gpu),
                })

        finally:
            conn.close()

    async def broadcast(self, run_id: str, message: dict) -> None:
        """Send a message to all clients subscribed to a run."""
        clients = self._connections.get(run_id, set()).copy()
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.get(run_id, set()).discard(ws)


async def websocket_endpoint(
    websocket: WebSocket, run_id: str, manager: ConnectionManager
) -> None:
    """WebSocket endpoint handler for real-time run updates."""
    await manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, run_id)
