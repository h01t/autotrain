"""Remote metrics agent — runs on GPU machine, pushes to dashboard via WebSocket.

Usage:
    python -m autotrain.remote_agent.agent \
        --server ws://localhost:8000/ws/agent/RUN_ID \
        [--log-path /path/to/train_output.log] \
        [--interval 1.0] \
        [--token SECRET]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from autotrain.remote_agent.collector import collect_gpu_metrics


async def _gpu_loop(ws, interval: float) -> None:
    """Push GPU metrics at fixed interval."""
    while True:
        try:
            gpus = collect_gpu_metrics()
            if gpus:
                msg = json.dumps({
                    "type": "gpu_metrics",
                    "ts": datetime.now(UTC).isoformat(),
                    "gpus": gpus,
                })
                await ws.send(msg)
        except Exception as e:
            print(f"[agent] gpu error: {e}", file=sys.stderr)
        await asyncio.sleep(interval)


async def _log_loop(ws, log_path: Path) -> None:
    """Tail training log and push lines."""
    from autotrain.remote_agent.tailer import tail_file

    try:
        async for line in tail_file(log_path):
            msg = json.dumps({
                "type": "log_line",
                "ts": datetime.now(UTC).isoformat(),
                "line": line,
            })
            await ws.send(msg)
    except Exception as e:
        print(f"[agent] tailer error: {e}", file=sys.stderr)


async def _heartbeat_loop(ws) -> None:
    """Send periodic heartbeat."""
    start = time.monotonic()
    while True:
        await asyncio.sleep(10)
        try:
            msg = json.dumps({
                "type": "heartbeat",
                "ts": datetime.now(UTC).isoformat(),
                "uptime_s": round(time.monotonic() - start),
            })
            await ws.send(msg)
        except Exception:
            break


async def run_agent(
    server_url: str,
    log_path: Path | None = None,
    interval: float = 1.0,
) -> None:
    """Connect to dashboard server and push metrics."""
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets import connect  # type: ignore[assignment]

    backoff = 1.0
    max_backoff = 30.0

    while True:
        try:
            print(f"[agent] connecting to {server_url}")
            async with connect(server_url, ping_interval=20, ping_timeout=10) as ws:
                print("[agent] connected")
                backoff = 1.0

                tasks = [
                    asyncio.create_task(_gpu_loop(ws, interval)),
                    asyncio.create_task(_heartbeat_loop(ws)),
                ]
                if log_path:
                    tasks.append(asyncio.create_task(_log_loop(ws, log_path)))

                # Wait until any task fails (usually means WS closed)
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_EXCEPTION,
                )
                for t in pending:
                    t.cancel()
                # Re-raise if there was an exception
                for t in done:
                    if t.exception():
                        raise t.exception()

        except (OSError, Exception) as e:
            print(f"[agent] disconnected: {e}. Reconnecting in {backoff:.0f}s...", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoTrain remote metrics agent")
    parser.add_argument("--server", required=True, help="WebSocket URL (ws://host:port/ws/agent/RUN_ID)")
    parser.add_argument("--log-path", default=None, help="Training log file to tail")
    parser.add_argument("--interval", type=float, default=1.0, help="GPU poll interval in seconds")
    args = parser.parse_args()

    log_path = Path(args.log_path) if args.log_path else None
    asyncio.run(run_agent(args.server, log_path, args.interval))


if __name__ == "__main__":
    main()
