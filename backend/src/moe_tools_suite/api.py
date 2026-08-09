from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from . import __version__
from .domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkRun,
    CreateModelSessionRequest,
    ExpertProfile,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileValidation,
    RoutingSummary,
    RunRequest,
    SystemStatus,
)
from .lab import ResearchLab

router = APIRouter(prefix="/api")


def _lab(request: Request) -> ResearchLab:
    return request.app.state.lab


@router.get("/system/status", response_model=SystemStatus)
def system_status(request: Request) -> SystemStatus:
    lab = _lab(request)
    return SystemStatus(
        mode=lab.settings.mode,
        version=__version__,
        model_state=lab.session.state if lab.session else ModelState.UNLOADED,
        data_dir=str(lab.settings.data_dir),
    )


@router.get("/models", response_model=list[ModelRegistryEntry])
def list_models(request: Request) -> list[ModelRegistryEntry]:
    return _lab(request).models


@router.post(
    "/model-sessions",
    response_model=ModelSession,
    status_code=status.HTTP_201_CREATED,
)
def create_model_session(
    payload: CreateModelSessionRequest, request: Request
) -> ModelSession:
    try:
        return _lab(request).create_model_session(payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/model-sessions/current", response_model=ModelSession)
def current_model_session(request: Request) -> ModelSession:
    session = _lab(request).session
    if session is None:
        raise HTTPException(status_code=404, detail="no model session")
    return session


@router.get("/benchmarks", response_model=list[BenchmarkInfo])
def list_benchmarks(request: Request) -> list[BenchmarkInfo]:
    return _lab(request).list_benchmarks()


@router.get(
    "/benchmarks/{benchmark_id}/items",
    response_model=list[BenchmarkItem],
)
def list_benchmark_items(
    benchmark_id: str, request: Request
) -> list[BenchmarkItem]:
    lab = _lab(request)
    if benchmark_id != "fixture-arithmetic":
        raise HTTPException(status_code=404, detail="benchmark not found")
    return lab.items


@router.post("/runs", response_model=BenchmarkRun, status_code=status.HTTP_201_CREATED)
async def create_run(payload: RunRequest, request: Request) -> BenchmarkRun:
    try:
        return await _lab(request).run_benchmark(payload)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/runs/{run_id}", response_model=BenchmarkRun)
def get_run(run_id: str, request: Request) -> BenchmarkRun:
    artifacts = _lab(request).runs.get(run_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="run not found")
    return artifacts.run


@router.get("/runs/{run_id}/routing", response_model=RoutingSummary)
def get_run_routing(run_id: str, request: Request) -> RoutingSummary:
    artifacts = _lab(request).runs.get(run_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="run not found")
    return artifacts.routing


@router.post("/profiles/validate", response_model=ProfileValidation)
def validate_expert_profile(
    payload: ExpertProfile, request: Request
) -> ProfileValidation:
    return _lab(request).validate_profile(payload)


@router.post("/profiles/propose", response_model=ProfileProposal)
def propose_expert_profile(
    payload: ProfileProposalRequest, request: Request
) -> ProfileProposal:
    try:
        return _lab(request).propose_profile(payload)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
