from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse

from . import __version__
from .domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkRun,
    ComparisonRecord,
    CreateComparisonRequest,
    CreateExpertProfileRequest,
    CreateModelSessionRequest,
    ExpertProfile,
    JobRecord,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileValidation,
    RoutingSummary,
    RunDetail,
    RunRequest,
    RuntimeStatus,
    SavedExpertProfile,
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


@router.get("/runtime/status", response_model=RuntimeStatus)
def runtime_status(request: Request) -> RuntimeStatus:
    return _lab(request).runtime_status()


@router.post(
    "/model-sessions",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_model_session(
    payload: CreateModelSessionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_model_session(payload)
        background_tasks.add_task(lab.execute_job, job.id)
        return job
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/model-sessions/current", response_model=ModelSession)
def current_model_session(request: Request) -> ModelSession:
    session = _lab(request).session
    if session is None:
        raise HTTPException(status_code=404, detail="no model session")
    return session


@router.get("/model-sessions", response_model=list[ModelSession])
def list_model_sessions(request: Request) -> list[ModelSession]:
    return _lab(request).list_model_sessions()


@router.get("/model-sessions/{session_id}", response_model=ModelSession)
def get_model_session(session_id: str, request: Request) -> ModelSession:
    model_session = _lab(request).model_sessions.get(session_id)
    if model_session is None:
        raise HTTPException(status_code=404, detail="model session not found")
    return model_session


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


@router.post(
    "/runs",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_run(
    payload: RunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_benchmark(payload)
        background_tasks.add_task(lab.execute_job, job.id)
        return job
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/runs", response_model=list[BenchmarkRun])
def list_runs(request: Request) -> list[BenchmarkRun]:
    return _lab(request).list_runs()


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


@router.get("/runs/{run_id}/detail", response_model=RunDetail)
def get_run_detail(run_id: str, request: Request) -> RunDetail:
    try:
        return _lab(request).get_run_detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/jobs", response_model=list[JobRecord])
def list_jobs(request: Request) -> list[JobRecord]:
    return _lab(request).list_jobs()


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    artifacts = _lab(request).jobs.get(job_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="job not found")
    return artifacts.record


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


@router.post(
    "/profiles",
    response_model=SavedExpertProfile,
    status_code=status.HTTP_201_CREATED,
)
def create_expert_profile(
    payload: CreateExpertProfileRequest, request: Request
) -> SavedExpertProfile:
    try:
        return _lab(request).create_expert_profile(payload)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="profile provenance not found"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/profiles", response_model=list[SavedExpertProfile])
def list_expert_profiles(request: Request) -> list[SavedExpertProfile]:
    return _lab(request).list_expert_profiles()


@router.get("/profiles/{profile_id}", response_model=SavedExpertProfile)
def get_expert_profile(profile_id: str, request: Request) -> SavedExpertProfile:
    profile = _lab(request).profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="expert profile not found")
    return profile


@router.get("/profiles/{profile_id}/export", response_model=ExpertProfile)
def export_expert_profile(
    profile_id: str, request: Request
) -> JSONResponse:
    profile = _lab(request).profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="expert profile not found")
    return JSONResponse(
        content=profile.profile.model_dump(mode="json"),
        headers={
            "Content-Disposition": (
                f'attachment; filename="expert-profile-{profile.id}.json"'
            )
        },
    )


@router.post(
    "/comparisons",
    response_model=ComparisonRecord,
    status_code=status.HTTP_201_CREATED,
)
def create_comparison(
    payload: CreateComparisonRequest, request: Request
) -> ComparisonRecord:
    try:
        return _lab(request).create_comparison(payload)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="comparison run not found"
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/comparisons", response_model=list[ComparisonRecord])
def list_comparisons(request: Request) -> list[ComparisonRecord]:
    return _lab(request).list_comparisons()


@router.get("/comparisons/{comparison_id}", response_model=ComparisonRecord)
def get_comparison(comparison_id: str, request: Request) -> ComparisonRecord:
    comparison = _lab(request).comparisons.get(comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    return comparison
