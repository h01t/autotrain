"""FastAPI application factory — serves API, WebSocket, and React SPA."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent_ws import AgentConnectionManager
from .api import router as api_router
from .control import RunManager
from .ws import ConnectionManager, websocket_endpoint

# React build output directory
_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def create_app(db_path: Path) -> FastAPI:
    """Create a configured FastAPI app for the dashboard."""

    browser_manager = ConnectionManager(db_path, poll_interval=1.0)
    agent_manager = AgentConnectionManager(browser_manager, db_path)
    run_manager = RunManager(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Cleanup: cancel polling task if running
        if browser_manager._poll_task and not browser_manager._poll_task.done():
            browser_manager._poll_task.cancel()

    app = FastAPI(
        title="AutoTrain Dashboard",
        lifespan=lifespan,
    )
    app.state.db_path = db_path
    app.state.agent_manager = agent_manager
    app.state.run_manager = run_manager

    # REST API
    app.include_router(api_router)

    # Agent WebSocket — must be registered before /ws/{run_id} to avoid catch-all conflict
    @app.websocket("/ws/agent/{run_id}")
    async def agent_ws_route(websocket: WebSocket, run_id: str):
        await agent_manager.handle_agent_ws(websocket, run_id)

    # Browser WebSocket (server pushes updates to dashboard UI)
    @app.websocket("/ws/{run_id}")
    async def ws_route(websocket: WebSocket, run_id: str):
        await websocket_endpoint(websocket, run_id, browser_manager)

    # Serve React SPA (static files + catch-all fallback)
    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=_FRONTEND_DIST / "assets"),
            name="assets",
        )

        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            # Serve the actual file if it exists in dist, otherwise index.html
            file_path = _FRONTEND_DIST / path
            if path and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(_FRONTEND_DIST / "index.html")
    else:
        @app.get("/")
        async def no_frontend():
            return {
                "message": "AutoTrain API is running. Frontend not built yet.",
                "hint": "Run 'npm run build' in src/autotrain/dashboard/frontend/",
                "api_docs": "/docs",
            }

    return app
