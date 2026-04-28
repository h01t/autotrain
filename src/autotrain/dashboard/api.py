"""REST API endpoints — thin wrappers around storage/queries.py."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from autotrain import __version__
from autotrain.config.defaults import create_default_config
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

from .models import (
    ArtifactInfo,
    ArtifactsListResponse,
    CreateRunRequest,
    DefaultsResponse,
    PreflightRequest,
    PreflightResponse,
    ResumeRunRequest,
    ResumeRunResponse,
    RunActionResponse,
    RunConfigResponse,
    RunLogsResponse,
    SaveConfigRequest,
    SaveConfigResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
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


def get_run_manager(request: Request):
    """Get the RunManager from app state."""
    return request.app.state.run_manager


# -- Health -------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok", "version": __version__}


# -- Runs ---------------------------------------------------------------------

@router.get("/runs")
def list_runs(conn: sqlite3.Connection = Depends(get_db)):
    runs = get_all_runs(conn)
    return [serialize_run(r) for r in runs]


# -- Run Creation & Control ---------------------------------------------------

@router.post("/runs", status_code=201)
def create_run(
    body: CreateRunRequest,
    manager=Depends(get_run_manager),
):
    """Create a new training run, optionally starting it immediately.

    Returns:
        201 — Run created successfully.
        400 — Config validation failed (status='invalid_config').
        409 — Another run is already active on the same repository (status='conflict').
    """
    result = manager.create_and_start(body)
    if result.status == "invalid_config":
        raise HTTPException(status_code=400, detail=result.model_dump())
    if result.status == "conflict":
        raise HTTPException(status_code=409, detail=result.model_dump())
    return result


@router.post("/runs/{run_id}/start")
def start_run(
    run_id: str,
    manager=Depends(get_run_manager),
) -> RunActionResponse:
    """Start a previously created or stopped run.

    Returns 409 if another run is active on the same repository.
    """
    result = manager.start_run(run_id)
    if not result.success and "already active" in result.message.lower():
        raise HTTPException(status_code=409, detail=result.model_dump())
    return result


@router.post("/runs/{run_id}/stop")
def stop_run(
    run_id: str,
    manager=Depends(get_run_manager),
) -> RunActionResponse:
    """Stop a running run."""
    return manager.stop_run(run_id)


@router.post("/runs/{run_id}/restart")
def restart_run(
    run_id: str,
    manager=Depends(get_run_manager),
) -> RunActionResponse:
    """Restart a failed, stopped, or completed run."""
    return manager.restart_run(run_id)


@router.get("/runs/{run_id}/status")
def get_run_status(
    run_id: str,
    manager=Depends(get_run_manager),
):
    """Get the current runtime status of a run (active, pid, uptime)."""
    return manager.get_run_status(run_id)


# -- Preflight & Validation ---------------------------------------------------

@router.post("/runs/preflight")
def run_preflight(body: PreflightRequest) -> PreflightResponse:
    """Run GPU and environment preflight checks."""
    from .control import run_preflight as _run_preflight
    return _run_preflight(body)


@router.post("/runs/validate-config")
def validate_config(body: ValidateConfigRequest) -> ValidateConfigResponse:
    """Validate a run configuration YAML without creating a run."""
    from .control import validate_config as _validate_config
    return _validate_config(body)


# -- Resume -------------------------------------------------------------------


@router.post("/runs/{run_id}/resume", status_code=201)
def resume_run(
    run_id: str,
    body: ResumeRunRequest,
    manager=Depends(get_run_manager),
) -> ResumeRunResponse:
    """Create a new run that resumes from a prior run's state.

    Returns a new run record with resumed_from_run_id linking to the prior run.
    """
    result = manager.resume_run(run_id, body.start_immediately)
    if result.status == "not_found":
        raise HTTPException(status_code=404, detail=result.model_dump())
    if result.status == "conflict":
        raise HTTPException(status_code=409, detail=result.model_dump())
    if result.status == "invalid_config":
        raise HTTPException(status_code=400, detail=result.model_dump())
    return result


# -- Config -------------------------------------------------------------------


@router.get("/runs/{run_id}/config")
def get_run_config(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> RunConfigResponse:
    """Return the full configuration for a run as both YAML and JSON."""
    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    config_json = (
        json.loads(run.config_snapshot) if run.config_snapshot else None
    )
    config_yaml_str = (
        yaml.dump(config_json, default_flow_style=False)
        if config_json else None
    )
    return RunConfigResponse(
        run_id=run_id,
        config_yaml=config_yaml_str,
        config_json=config_json,
    )


# -- Logs ---------------------------------------------------------------------


@router.get("/runs/{run_id}/logs")
def get_run_logs(
    run_id: str,
    tail: int = Query(200, ge=1, le=2000),
    conn: sqlite3.Connection = Depends(get_db),
    request: Request = None,
) -> RunLogsResponse:
    """Return the most recent training log lines for a run."""
    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    repo_path = Path(run.repo_path)
    log_path = repo_path / ".autotrain" / "train_output.log"

    lines: list[str] = []
    truncated = False
    if log_path.exists():
        try:
            all_lines = log_path.read_text().splitlines()
            total = len(all_lines)
            if total > tail:
                truncated = True
                lines = all_lines[-tail:]
            else:
                lines = all_lines
            return RunLogsResponse(
                run_id=run_id,
                lines=lines,
                total_lines=total,
                truncated=truncated,
            )
        except Exception:
            pass

    return RunLogsResponse(run_id=run_id, lines=[], total_lines=0, truncated=False)


# -- Artifacts ----------------------------------------------------------------


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> ArtifactsListResponse:
    """List artifact files for a run (checkpoints, model files, etc.)."""
    import os as _os

    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    repo_path = Path(run.repo_path)
    artifacts_dir = repo_path / ".autotrain" / "artifacts" / run_id

    artifacts: list[ArtifactInfo] = []
    total_bytes = 0

    if artifacts_dir.is_dir():
        for root, dirs, files in _os.walk(artifacts_dir):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    real = fpath.resolve(strict=False)
                    # Safety: ensure resolved path stays within artifacts_dir
                    real.relative_to(artifacts_dir.resolve())
                    stat = fpath.stat()
                    size = stat.st_size
                    artifacts.append(ArtifactInfo(
                        name=fname,
                        path=str(fpath.relative_to(artifacts_dir)),
                        size_bytes=size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    ))
                    total_bytes += size
                except (ValueError, OSError):
                    # Symlink escape or broken file — skip
                    continue

    return ArtifactsListResponse(
        run_id=run_id,
        artifacts=artifacts,
        total_bytes=total_bytes,
    )


@router.get("/runs/{run_id}/artifacts/{artifact_path:path}")
def download_artifact(
    run_id: str,
    artifact_path: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Download a specific artifact file."""

    from fastapi.responses import FileResponse

    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    repo_path = Path(run.repo_path)
    artifacts_dir = (repo_path / ".autotrain" / "artifacts" / run_id).resolve()

    # Safety: check for path traversal patterns in the request
    if ".." in artifact_path or artifact_path.startswith("/"):
        raise HTTPException(
            status_code=403,
            detail="Artifact path contains traversal segments.",
        )
    # Also verify resolved path stays in artifact root
    raw_path = artifacts_dir / artifact_path
    try:
        resolved = raw_path.resolve(strict=False)
        resolved.relative_to(artifacts_dir)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Artifact path escapes the artifact root.",
        )

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(resolved, filename=resolved.name)


# -- Defaults -----------------------------------------------------------------


@router.get("/defaults")
def get_defaults() -> DefaultsResponse:
    """Return a default configuration template for new runs."""
    yaml_str = create_default_config()
    return DefaultsResponse(config_yaml=yaml_str)


# -- Save Config --------------------------------------------------------------


@router.post("/save-config")
def save_config(body: SaveConfigRequest) -> SaveConfigResponse:
    """Save a configuration YAML to a repository's autotrain.yaml file."""

    repo_path = Path(body.repo_path).resolve()

    # Safety: reject paths with null bytes or control chars
    if "\x00" in str(repo_path):
        raise HTTPException(status_code=400, detail="Invalid repo_path")

    # Validate repo exists
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Repository directory does not exist: {repo_path}",
        )

    # Validate YAML is parseable
    try:
        data = yaml.safe_load(body.config_yaml)
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Config YAML must parse to a mapping/dict.",
            )
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid YAML: {e}",
        )

    # Validate config schema via load_config
    try:
        from autotrain.config.loader import load_config as _load_config
        _load_config(repo_path, cli_overrides=data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Config validation failed: {e}",
        )

    config_path = repo_path / "autotrain.yaml"
    try:
        config_path.write_text(body.config_yaml)
    except (OSError, PermissionError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write config: {e}",
        )

    return SaveConfigResponse(
        success=True,
        path=str(config_path),
        message="Configuration saved successfully.",
    )


# -- Analytics placeholder — must come before /runs/{run_id} to avoid route conflict
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
