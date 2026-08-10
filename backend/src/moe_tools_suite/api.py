from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import JSONResponse, Response

from . import __version__
from .agentic.atif import trajectory_to_json
from .agentic.domain import (
    AgentDefinition,
    AgentRunExport,
    AgentRunView,
    AgentTaskInfo,
    AgentTaskPackInfo,
    AgentTrajectoryView,
    AgentTrialArtifactsView,
    AgentTrialSummary,
    CreateAgentRunRequest,
    ProviderPreflight,
    SandboxProviderInfo,
    TrialRoutingSummary,
)
from .domain import (
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItemPage,
    BenchmarkRun,
    ComparisonRecord,
    CreateComparisonRequest,
    CreateCustomBenchmarkRequest,
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


@router.get("/agents", response_model=list[AgentDefinition])
def list_agents(request: Request) -> list[AgentDefinition]:
    return _lab(request).agentic.list_agents()


@router.get("/sandbox-providers", response_model=list[SandboxProviderInfo])
def list_sandbox_providers(request: Request) -> list[SandboxProviderInfo]:
    return _lab(request).agentic.list_providers()


@router.post(
    "/sandbox-providers/{provider_id}/preflight",
    response_model=ProviderPreflight,
)
async def preflight_sandbox_provider(
    provider_id: str, request: Request
) -> ProviderPreflight:
    try:
        return await _lab(request).agentic.preflight_provider(provider_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="provider not found") from error


@router.get("/agent-task-packs", response_model=list[AgentTaskPackInfo])
def list_agent_task_packs(request: Request) -> list[AgentTaskPackInfo]:
    return _lab(request).agentic.list_task_packs()


@router.get(
    "/agent-task-packs/{pack_id}/tasks",
    response_model=list[AgentTaskInfo],
)
def list_agent_tasks(pack_id: str, request: Request) -> list[AgentTaskInfo]:
    try:
        return [
            AgentTaskInfo(
                id=task.id,
                title=task.title,
                instruction=task.instruction,
                language=task.language,
                tags=task.tags,
                timeout_seconds=task.timeout_seconds,
            )
            for task in _lab(request).agentic.list_tasks(pack_id)
        ]
    except KeyError as error:
        raise HTTPException(status_code=404, detail="task pack not found") from error


@router.post(
    "/agent-runs",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_agent_run(
    payload: CreateAgentRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_agent_run(payload)
        background_tasks.add_task(lab.execute_job, job.id)
        return job
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/agent-runs", response_model=list[AgentRunView])
def list_agent_runs(request: Request) -> list[AgentRunView]:
    return _lab(request).agentic.list_run_views()


@router.get("/agent-runs/{run_id}", response_model=AgentRunView)
def get_agent_run(run_id: str, request: Request) -> AgentRunView:
    try:
        return _lab(request).agentic.run_view(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="agent run not found") from error


@router.post("/agent-runs/{run_id}/cancel", response_model=AgentRunView)
def cancel_agent_run(run_id: str, request: Request) -> AgentRunView:
    lab = _lab(request)
    try:
        run = lab.agentic.runs[run_id]
        lab.cancel_job(run.job_id)
        return lab.agentic.run_view(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="agent run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/agent-runs/{run_id}/export", response_model=AgentRunExport)
def export_agent_run(run_id: str, request: Request) -> JSONResponse:
    try:
        exported = _lab(request).agentic.export_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="agent run not found") from error
    return JSONResponse(
        content=exported.model_dump(mode="json", exclude_none=True),
        headers={
            "Content-Disposition": (f'attachment; filename="agent-run-{run_id}.json"')
        },
    )


@router.get("/trials/{trial_id}", response_model=AgentTrialSummary)
def get_agent_trial(trial_id: str, request: Request) -> AgentTrialSummary:
    try:
        return _lab(request).agentic.get_trial(trial_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="trial not found") from error


@router.get("/trials/{trial_id}/trajectory", response_model=AgentTrajectoryView)
def get_agent_trajectory(trial_id: str, request: Request) -> AgentTrajectoryView:
    try:
        return _lab(request).agentic.get_trajectory(trial_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="trajectory not found") from error


@router.get("/trials/{trial_id}/export/atif", response_model=None)
def export_agent_trajectory(trial_id: str, request: Request) -> Response:
    trajectory = _lab(request).agentic.trajectories.get(trial_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail="trajectory not found")
    return Response(
        content=trajectory_to_json(trajectory),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="trajectory-{trial_id}.atif.json"'
            )
        },
    )


@router.get("/trials/{trial_id}/routing", response_model=TrialRoutingSummary)
def get_agent_routing(trial_id: str, request: Request) -> TrialRoutingSummary:
    try:
        return _lab(request).agentic.get_routing(trial_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="routing not found") from error


@router.get(
    "/trials/{trial_id}/artifacts",
    response_model=AgentTrialArtifactsView,
)
def get_agent_artifacts(trial_id: str, request: Request) -> AgentTrialArtifactsView:
    try:
        return _lab(request).agentic.get_artifacts(trial_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="trial not found") from error


@router.post("/trials/{trial_id}/profile-proposal", response_model=ProfileProposal)
def propose_agent_profile(
    trial_id: str,
    request: Request,
    keep_per_layer: int = 64,
    metric: Literal["routing_mass", "selection_count"] = "routing_mass",
) -> ProfileProposal:
    try:
        return _lab(request).propose_agent_profile(
            trial_id,
            keep_per_layer=keep_per_layer,
            metric=metric,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="routing not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
    response_model=BenchmarkItemPage,
)
def list_benchmark_items(
    benchmark_id: str,
    request: Request,
    search: str = "",
    category: str | None = None,
    offset: int = 0,
    limit: int = 200,
) -> BenchmarkItemPage:
    try:
        return _lab(request).list_benchmark_items(
            benchmark_id,
            search=search,
            category=category,
            offset=offset,
            limit=limit,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post(
    "/benchmarks/custom",
    response_model=BenchmarkDatasetRecord,
    status_code=status.HTTP_201_CREATED,
)
def import_custom_benchmark(
    payload: CreateCustomBenchmarkRequest,
    request: Request,
) -> BenchmarkDatasetRecord:
    try:
        return _lab(request).import_custom_benchmark(payload)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post(
    "/benchmarks/{benchmark_id}/prepare",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def prepare_benchmark(
    benchmark_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_prepare_benchmark(benchmark_id)
        background_tasks.add_task(lab.execute_job, job.id)
        return job
    except KeyError as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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
    except KeyError as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
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


@router.get("/runs/{run_id}/export", response_model=None)
def export_run(
    run_id: str,
    request: Request,
    export_format: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
) -> JSONResponse | Response:
    lab = _lab(request)
    try:
        detail = lab.get_run_detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    artifacts = lab.runs[run_id]
    filename = f"benchmark-run-{run_id}.{export_format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if export_format == "json":
        comparisons = [
            comparison
            for comparison in lab.comparisons.values()
            if run_id in {comparison.baseline_run_id, comparison.candidate_run_id}
        ]
        return JSONResponse(
            content={
                "schema_version": 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "detail": detail.model_dump(mode="json"),
                "routing": artifacts.routing.model_dump(mode="json"),
                "comparisons": [
                    comparison.model_dump(mode="json") for comparison in comparisons
                ],
            },
            headers=headers,
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "item_id",
            "passed",
            "scoring",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "error",
            "prompt",
            "expected",
            "output",
        ]
    )
    for item in detail.run.items:
        writer.writerow(
            [
                item.item_id,
                item.passed,
                item.scoring.value,
                item.latency_ms,
                item.prompt_tokens,
                item.completion_tokens,
                item.error or "",
                item.prompt,
                item.expected,
                item.output,
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@router.get("/jobs", response_model=list[JobRecord])
def list_jobs(request: Request) -> list[JobRecord]:
    return _lab(request).list_jobs()


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    artifacts = _lab(request).jobs.get(job_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="job not found")
    return artifacts.record


@router.post("/jobs/{job_id}/cancel", response_model=JobRecord)
def cancel_job(job_id: str, request: Request) -> JobRecord:
    try:
        return _lab(request).cancel_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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
def export_expert_profile(profile_id: str, request: Request) -> JSONResponse:
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
