from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router
from .lab import ResearchLab
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="MoE Tools Test Suite",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.lab = ResearchLab(settings)
    app.include_router(router)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": settings.mode}

    _mount_frontend(app, settings.frontend_dist)
    return app


def _mount_frontend(app: FastAPI, dist_path: Path) -> None:
    index_path = dist_path / "index.html"
    assets_path = dist_path / "assets"
    if assets_path.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/", response_model=None)
    def index() -> FileResponse | HTMLResponse:
        if index_path.is_file():
            return FileResponse(index_path)
        return HTMLResponse(
            "<h1>MoE Tools Test Suite</h1>"
            "<p>Frontend build not found. Use the Vite development server.</p>"
        )

    @app.get("/{path:path}", response_model=None)
    def frontend_fallback(path: str) -> FileResponse | HTMLResponse:
        del path
        if index_path.is_file():
            return FileResponse(index_path)
        return HTMLResponse("Frontend build not found", status_code=404)


app = create_app()
