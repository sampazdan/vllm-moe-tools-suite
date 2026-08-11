from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .agentic.atif import trajectory_to_json
from .agentic.comparison import (
    AgentRunComparison,
    CreateAgentComparisonRequest,
)
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
    ActiveJobConflictDetail,
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItemPage,
    BenchmarkProblemDetail,
    BenchmarkRunPage,
    BenchmarkRunSummary,
    ComparisonRecord,
    CreateComparisonRequest,
    CreateCustomBenchmarkRequest,
    CreateExpertProfileRequest,
    CreateModelSessionRequest,
    EvaluationCapabilities,
    ExpertProfile,
    JobRecord,
    JudgeEvaluationRequest,
    LLMJudgeResult,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileValidation,
    RoutingSummary,
    RunDetail,
    RunItemResult,
    RunItemResultPage,
    RunRequest,
    RuntimeStatus,
    SavedExpertProfile,
    SystemStatus,
)
from .expert_explorer import RoutingExploreRequest, RoutingExploreResponse
from .judges import JudgeProviderError
from .lab import ActiveJobConflict, ResearchLab

router = APIRouter(prefix="/api")


def _lab(request: Request) -> ResearchLab:
    return request.app.state.lab


def _job_conflict(error: ActiveJobConflict) -> HTTPException:
    active_job = error.active_job
    detail = ActiveJobConflictDetail(
        message=str(error),
        active_job=active_job,
        recovery_url=f"/api/jobs/{active_job.id}",
    )
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=detail.model_dump(mode="json"),
    )


@router.get("/system/status", response_model=SystemStatus)
def system_status(request: Request) -> SystemStatus:
    lab = _lab(request)
    current_session = lab.current_model_session()
    return SystemStatus(
        mode=lab.settings.mode,
        version=__version__,
        model_state=(current_session.state if current_session else ModelState.UNLOADED),
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
                success_criteria=task.success_criteria,
                verifier_fingerprint=task.verifier_fingerprint(),
                hidden_verifier_file_count=len(task.verifier_file_paths),
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
async def create_agent_run(
    payload: CreateAgentRunRequest,
    request: Request,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_agent_run(payload)
        lab.start_job(job.id)
        return job
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ActiveJobConflict as error:
        raise _job_conflict(error) from error
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


@router.post("/agent-comparisons", response_model=AgentRunComparison)
def create_agent_comparison(
    payload: CreateAgentComparisonRequest,
    request: Request,
) -> AgentRunComparison:
    try:
        return _lab(request).compare_agent_runs(payload)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="agent comparison run not found"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
async def create_model_session(
    payload: CreateModelSessionRequest,
    request: Request,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_model_session(payload)
        lab.start_job(job.id)
        return job
    except ActiveJobConflict as error:
        raise _job_conflict(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/model-sessions/current", response_model=ModelSession)
def current_model_session(request: Request) -> ModelSession:
    session = _lab(request).current_model_session()
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
    "/benchmarks/{benchmark_id}/items/{item_id}",
    response_model=BenchmarkProblemDetail,
)
def get_benchmark_problem(
    benchmark_id: str,
    item_id: str,
    request: Request,
) -> BenchmarkProblemDetail:
    try:
        return _lab(request).get_benchmark_problem(benchmark_id, item_id)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="benchmark item not found"
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/evaluation/capabilities", response_model=EvaluationCapabilities)
def evaluation_capabilities(request: Request) -> EvaluationCapabilities:
    return _lab(request).evaluation_capabilities()


@router.post("/evaluation/judge", response_model=LLMJudgeResult)
async def judge_evaluation(
    payload: JudgeEvaluationRequest,
    request: Request,
) -> LLMJudgeResult:
    try:
        return await _lab(request).judge(payload)
    except JudgeProviderError as error:
        message = str(error)
        status_code = 409 if "not configured" in message else 502
        raise HTTPException(status_code=status_code, detail=message) from error


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
async def prepare_benchmark(
    benchmark_id: str,
    request: Request,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_prepare_benchmark(benchmark_id)
        lab.start_job(job.id)
        return job
    except KeyError as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    except ActiveJobConflict as error:
        raise _job_conflict(error) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/runs",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    payload: RunRequest,
    request: Request,
) -> JobRecord:
    try:
        lab = _lab(request)
        job = lab.submit_benchmark(payload)
        lab.start_job(job.id)
        return job
    except ActiveJobConflict as error:
        raise _job_conflict(error) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/runs", response_model=BenchmarkRunPage)
def list_runs(
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> BenchmarkRunPage:
    return _lab(request).list_runs(offset=offset, limit=limit)


@router.get("/runs/{run_id}", response_model=BenchmarkRunSummary)
def get_run(run_id: str, request: Request) -> BenchmarkRunSummary:
    try:
        return _lab(request).get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error


@router.get("/runs/{run_id}/items", response_model=RunItemResultPage)
def get_run_items(
    run_id: str,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=250)] = 100,
) -> RunItemResultPage:
    try:
        return _lab(request).get_run_items(run_id, offset=offset, limit=limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error


@router.get("/runs/{run_id}/routing", response_model=RoutingSummary)
def get_run_routing(run_id: str, request: Request) -> RoutingSummary:
    artifacts = _lab(request).runs.get(run_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail="run not found")
    return artifacts.routing


@router.post("/routing/explore", response_model=RoutingExploreResponse)
def explore_routing(
    payload: RoutingExploreRequest,
    request: Request,
) -> RoutingExploreResponse:
    try:
        return _lab(request).explore_routing(payload)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="routing source not found"
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
) -> StreamingResponse:
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
        return StreamingResponse(
            _run_json_export_chunks(
                detail=detail,
                items=artifacts.run.items,
                routing=artifacts.routing,
                comparisons=comparisons,
                cohort_item_ids=(
                    lab.cohorts[artifacts.run.cohort_id].item_ids
                    if artifacts.run.cohort_id is not None
                    and artifacts.run.cohort_id in lab.cohorts
                    else None
                ),
            ),
            media_type="application/json",
            headers=headers,
        )
    return StreamingResponse(
        _run_csv_export_chunks(artifacts.run.items),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


def _run_json_export_chunks(
    *,
    detail: RunDetail,
    items: list[RunItemResult],
    routing: RoutingSummary,
    comparisons: list[ComparisonRecord],
    cohort_item_ids: list[str] | None,
) -> Iterator[str]:
    detail_data = detail.model_dump(mode="json")
    run_data = detail_data.pop("run")
    run_json = _compact_json(run_data)
    yield (
        '{"schema_version":1,"exported_at":'
        f'{_compact_json(datetime.now(UTC).isoformat())},"detail":{{"run":'
        f'{run_json[:-1]},"items":['
    )
    for index, item in enumerate(items):
        if index:
            yield ","
        yield item.model_dump_json()
    yield "]}"
    for key, value in detail_data.items():
        yield f",{_compact_json(key)}:"
        if key == "cohort" and value is not None and cohort_item_ids is not None:
            cohort_json = _compact_json(value)
            yield f'{cohort_json[:-1]},"item_ids":['
            for index, item_id in enumerate(cohort_item_ids):
                if index:
                    yield ","
                yield _compact_json(item_id)
            yield "]}"
        else:
            yield _compact_json(value)
    yield f'}},"routing":{routing.model_dump_json()},"comparisons":['
    for index, comparison in enumerate(comparisons):
        if index:
            yield ","
        yield comparison.model_dump_json()
    yield "]}"


_RUN_CSV_HEADER = [
    "item_id",
    "passed",
    "scoring",
    "attempt",
    "deterministic_score",
    "judge_score",
    "combined_score",
    "latency_ms",
    "prompt_tokens",
    "reasoning_tokens",
    "completion_tokens",
    "total_tokens",
    "ttft_ms",
    "decode_ms",
    "tokens_per_second",
    "inference_cost_usd",
    "judge_cost_usd",
    "judge_equivalent_cost_usd",
    "judge_budget_debit_usd",
    "judge_cost_uncertain",
    "error",
    "prompt",
    "expected",
    "output",
]


def _run_csv_export_chunks(items: list[RunItemResult]) -> Iterator[str]:
    yield _csv_row(_RUN_CSV_HEADER)
    for item in items:
        evaluation = item.evaluation
        performance = item.performance
        judge = evaluation.judge if evaluation is not None else None
        yield _csv_row(
            [
                item.item_id,
                item.passed,
                item.scoring.value,
                item.attempt,
                evaluation.deterministic_score if evaluation is not None else "",
                evaluation.judge_score if evaluation is not None else "",
                evaluation.combined_score if evaluation is not None else "",
                item.latency_ms,
                item.prompt_tokens,
                item.reasoning_tokens if item.reasoning_tokens is not None else "",
                item.completion_tokens,
                item.total_tokens,
                performance.ttft_ms if performance is not None else "",
                performance.decode_ms if performance is not None else "",
                performance.tokens_per_second if performance is not None else "",
                (
                    performance.estimated_cost_usd
                    if performance is not None
                    and performance.estimated_cost_usd is not None
                    else ""
                ),
                (
                    judge.usage.incurred_cost_usd
                    if judge is not None and judge.usage.incurred_cost_usd is not None
                    else ""
                ),
                (
                    judge.usage.estimated_cost_usd
                    if judge is not None and judge.usage.estimated_cost_usd is not None
                    else ""
                ),
                item.judge_budget_debit_usd,
                item.judge_cost_uncertain,
                item.error or "",
                item.prompt,
                item.expected,
                item.output,
            ]
        )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv_row(values: list[object]) -> str:
    output = io.StringIO(newline="")
    csv.writer(output).writerow([_csv_safe_cell(value) for value in values])
    return output.getvalue()


def _csv_safe_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


@router.get("/jobs", response_model=list[JobRecord])
def list_jobs(request: Request) -> list[JobRecord]:
    return _lab(request).list_jobs()


@router.get("/jobs/active", response_model=JobRecord | None)
def get_active_job(request: Request) -> JobRecord | None:
    return _lab(request).active_job()


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    try:
        return _lab(request).get_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="job not found") from error


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
