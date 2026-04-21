"""REST API endpoints — thin wrappers around storage/queries.py."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from autotrain import __version__
from autotrain.storage.queries import (
    get_all_metric_snapshots,
    get_all_runs,
    get_best_iterations,
    get_epoch_metrics,
    get_gpu_snapshots,
    get_latest_gpu_snapshot,
    get_recent_iterations,
    get_run,
)

from .serializers import (
    serialize_epoch_metric,
    serialize_gpu_snapshot,
    serialize_iteration,
    serialize_metric_snapshot,
    serialize_run,
)

router = APIRouter(prefix="/api/v1")


def get_db(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Yield a read-only DB connection per request."""
    db_path: Path = request.app.state.db_path
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# -- Health -------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok", "version": __version__}


# -- Runs ---------------------------------------------------------------------

@router.get("/runs")
def list_runs(conn: sqlite3.Connection = Depends(get_db)):
    runs = get_all_runs(conn)
    return [serialize_run(r) for r in runs]


# Future placeholder — must come before /runs/{run_id} to avoid route conflict
@router.get("/runs/analytics")
def get_analytics():
    raise HTTPException(status_code=501, detail="Not implemented — Phase 2")


@router.get("/runs/{run_id}")
def get_run_detail(run_id: str, conn: sqlite3.Connection = Depends(get_db)):
    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return serialize_run(run)


# -- Iterations ---------------------------------------------------------------

@router.get("/runs/{run_id}/iterations")
def list_iterations(
    run_id: str,
    limit: int = Query(100, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
):
    iterations = get_recent_iterations(conn, run_id, limit=limit)
    return [serialize_iteration(it) for it in iterations]


@router.get("/runs/{run_id}/iterations/best")
def list_best_iterations(
    run_id: str,
    direction: str = Query("maximize"),
    limit: int = Query(5, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_db),
):
    iterations = get_best_iterations(conn, run_id, direction=direction, limit=limit)
    return [serialize_iteration(it) for it in iterations]


# -- Metrics ------------------------------------------------------------------

@router.get("/runs/{run_id}/metrics")
def list_metrics(run_id: str, conn: sqlite3.Connection = Depends(get_db)):
    snapshots = get_all_metric_snapshots(conn, run_id)
    return [serialize_metric_snapshot(ms) for ms in snapshots]


# -- Epoch Metrics ------------------------------------------------------------

@router.get("/runs/{run_id}/epochs")
def list_epoch_metrics(
    run_id: str,
    iteration_num: int | None = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
):
    epochs = get_epoch_metrics(conn, run_id, iteration_num=iteration_num)
    return [serialize_epoch_metric(em) for em in epochs]


# -- GPU Snapshots ------------------------------------------------------------

@router.get("/runs/{run_id}/gpu")
def list_gpu_snapshots(
    run_id: str,
    limit: int = Query(500, ge=1, le=5000),
    conn: sqlite3.Connection = Depends(get_db),
):
    snapshots = get_gpu_snapshots(conn, run_id, limit=limit)
    return [serialize_gpu_snapshot(gs) for gs in snapshots]


@router.get("/runs/{run_id}/gpu/latest")
def get_gpu_latest(run_id: str, conn: sqlite3.Connection = Depends(get_db)):
    snapshot = get_latest_gpu_snapshot(conn, run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No GPU data")
    return serialize_gpu_snapshot(snapshot)


# -- Future placeholders (Phase 2/3) -----------------------------------------

@router.get("/runs/{run_id}/diff/{iter_a}/{iter_b}")
def get_diff(run_id: str, iter_a: int, iter_b: int):
    raise HTTPException(status_code=501, detail="Not implemented — Phase 2")


@router.post("/runs/{run_id}/early-stop")
def early_stop(run_id: str):
    raise HTTPException(status_code=501, detail="Not implemented — Phase 2")


@router.get("/runs/{run_id}/files")
def get_files(run_id: str):
    raise HTTPException(status_code=501, detail="Not implemented — Phase 3")
