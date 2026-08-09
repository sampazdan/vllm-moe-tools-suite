from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from . import __version__
from .api import router
from .lab import ResearchLab
from .security import SessionMiddleware, create_session_router, create_session_signer
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        await application.state.lab.shutdown()

    app = FastAPI(
        title="MoE Tools Test Suite",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.lab = ResearchLab(settings)
    app.state.session_signer = create_session_signer(settings)
    app.add_middleware(
        SessionMiddleware,
        settings=settings,
        signer=app.state.session_signer,
    )
    app.include_router(create_session_router(settings))
    app.include_router(router)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": settings.mode}

    @app.get("/readyz", response_model=None)
    def readiness() -> dict[str, str] | JSONResponse:
        try:
            with app.state.lab.store.engine.connect() as database:
                database.execute(text("SELECT 1"))
        except (OSError, SQLAlchemyError):
            return JSONResponse(
                {"status": "not_ready", "database": "unavailable"},
                status_code=503,
            )
        return {"status": "ready", "database": "ok", "mode": settings.mode}

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
