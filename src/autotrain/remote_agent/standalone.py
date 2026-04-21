#!/usr/bin/env python3
"""Standalone remote metrics agent — no autotrain package dependency.

Deploy this single file to the GPU machine and run:
    python standalone.py --server ws://dashboard:8000/ws/agent/RUN_ID [--interval 1.0]

Requires: websockets (pip install websockets)
Optional: pynvml / nvidia-ml-py3 (falls back to nvidia-smi subprocess)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# GPU collector
# ---------------------------------------------------------------------------

def collect_gpu_metrics() -> list[dict]:
    """Query GPUs via pynvml (fast) or nvidia-smi (fallback)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            gpus.append({
                "gpu_index": i,
                "utilization_pct": float(util.gpu),
                "memory_used_mb": round(mem.used / 1048576, 1),
                "memory_total_mb": round(mem.total / 1048576, 1),
                "temperature_c": float(temp) if temp is not None else None,
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "gpu_index": int(parts[0]),
                    "utilization_pct": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "temperature_c": float(parts[4]),
                })
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Log tailer
# ---------------------------------------------------------------------------

async def tail_file(path: Path, poll_interval: float = 0.1):
    """Async generator that yields new lines appended to a file."""
    inode = None
    offset = 0
    fp = None
    try:
        while True:
            if not path.exists():
                await asyncio.sleep(poll_interval)
                continue
            cur_inode = os.stat(path).st_ino
            if inode != cur_inode:
                if fp:
                    fp.close()
                fp = open(path)
                inode = cur_inode
                if offset == 0:
                    fp.seek(0, 2)
                    offset = fp.tell()
                else:
                    offset = 0
            fp.seek(offset)
            data = fp.read()
            if data:
                offset = fp.tell()
                for line in data.splitlines():
                    if line:
                        yield line
            else:
                await asyncio.sleep(poll_interval)
    finally:
        if fp:
            fp.close()


# ---------------------------------------------------------------------------
# Agent loops
# ---------------------------------------------------------------------------

async def _gpu_loop(ws, interval: float):
    while True:
        try:
            gpus = collect_gpu_metrics()
            if gpus:
                await ws.send(json.dumps({
                    "type": "gpu_metrics",
                    "ts": datetime.now(UTC).isoformat(),
                    "gpus": gpus,
                }))
        except Exception as e:
            print(f"[agent] gpu error: {e}", file=sys.stderr)
        await asyncio.sleep(interval)


async def _log_loop(ws, log_path: Path):
    try:
        async for line in tail_file(log_path):
            await ws.send(json.dumps({
                "type": "log_line",
                "ts": datetime.now(UTC).isoformat(),
                "line": line,
            }))
    except Exception as e:
        print(f"[agent] tailer error: {e}", file=sys.stderr)


async def _heartbeat_loop(ws):
    start = time.monotonic()
    while True:
        await asyncio.sleep(10)
        try:
            await ws.send(json.dumps({
                "type": "heartbeat",
                "ts": datetime.now(UTC).isoformat(),
                "uptime_s": round(time.monotonic() - start),
            }))
        except Exception:
            break


async def run_agent(server_url: str, log_path: Path | None = None, interval: float = 1.0):
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets import connect

    backoff = 1.0
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
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
        except Exception as e:
            print(f"[agent] disconnected: {e}. Reconnecting in {backoff:.0f}s...", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main():
    p = argparse.ArgumentParser(description="AutoTrain remote metrics agent")
    p.add_argument("--server", required=True, help="WebSocket URL")
    p.add_argument("--log-path", default=None, help="Training log to tail")
    p.add_argument("--interval", type=float, default=1.0, help="GPU poll interval (seconds)")
    args = p.parse_args()
    log_path = Path(args.log_path) if args.log_path else None
    asyncio.run(run_agent(args.server, log_path, args.interval))


if __name__ == "__main__":
    main()
