from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np

from . import __version__
from .agentic.comparison import (
    AgentRunComparison,
    CreateAgentComparisonRequest,
    compare_agent_runs,
)
from .agentic.controller import AgenticController
from .agentic.domain import CreateAgentRunRequest
from .benchmarks import BenchmarkAdapter, BenchmarkCatalog
from .domain import (
    BenchmarkCohort,
    BenchmarkCohortView,
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkItemPage,
    BenchmarkProblemDetail,
    BenchmarkRun,
    BenchmarkRunPage,
    BenchmarkRunSummary,
    ComparisonRecord,
    CreateComparisonRequest,
    CreateCustomBenchmarkRequest,
    CreateExpertProfileRequest,
    CreateModelSessionRequest,
    CriterionVisibility,
    DeterministicScorerKind,
    EvaluationCapabilities,
    EvaluationContract,
    EvaluationCriterion,
    ExecutionPolicy,
    ExpertProfile,
    GenerationConfig,
    InferencePerformance,
    JobKind,
    JobRecord,
    JobStatus,
    JudgeEvaluationRequest,
    JudgeMode,
    LLMJudgeResult,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ModelTopology,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileSourceKind,
    ProfileSourceRef,
    ProfileValidation,
    RoutingSummary,
    RunDetail,
    RunItemResult,
    RunItemResultPage,
    RunPerformance,
    RunProvenance,
    RunProvenanceView,
    RunRequest,
    RuntimeStatus,
    SavedExpertProfile,
    ScoringMode,
    SuccessCriteriaView,
)
from .evaluation import evaluate_output, validate_cost_policy_pricing
from .expert_explorer import (
    ResolvedRoutingSource,
    RoutingExploreRequest,
    RoutingExploreResponse,
    RoutingFilterCapabilities,
    RoutingSourceKind,
    RoutingSourceReference,
    build_routing_explorer_response,
)
from .judges import (
    JudgeBudgetExceeded,
    JudgeProviderError,
    JudgeService,
    judge_request_cost_upper_bound,
)
from .persistence import SqliteStore
from .process_manager import ManagedVllmServer
from .profiles import (
    calculate_observed_mass_retained,
    propose_fixed_budget_profile,
    validate_profile,
)
from .runtime import MockModelRuntime, ModelRuntime, VllmRuntime
from .settings import Settings
from .telemetry import AggregatedRouting, aggregate_routing

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
FIXTURE_BENCHMARK_ID = "fixture-arithmetic"
logger = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    run: BenchmarkRun
    routing: RoutingSummary
    provenance: RunProvenance | None = None


@dataclass
class JobArtifacts:
    record: JobRecord
    payload: dict[str, object]


@dataclass
class BenchmarkItemExecution:
    result: RunItemResult
    routing: AggregatedRouting | None
    evaluation_time_ms: float = 0
    budget_debit_usd: float = 0
    cost_budget_exhausted: bool = False
    token_budget_exhausted: bool = False


class ActiveJobConflict(RuntimeError):
    """A submission conflicts with the process-wide active job."""

    def __init__(self, active_job: JobRecord) -> None:
        self.active_job = active_job.model_copy(deep=True)
        super().__init__(
            f"{active_job.kind.value} job {active_job.id} is already "
            f"{active_job.status.value}"
        )


class ResearchLab:
    """Application service for the first model-to-profile vertical slice."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.topology = ModelTopology(
            num_layers=40,
            num_experts=256,
            top_k=8,
            routed_layer_ids=list(range(40)),
        )
        self.models = [
            ModelRegistryEntry(
                id=MODEL_ID,
                display_name="Qwen 3.6 35B A3B FP8",
                enabled=True,
                topology=self.topology,
                notes=(
                    "Validated target for one RTX PRO 6000 Blackwell. "
                    "Mock mode synthesizes its 40 × 256 routing topology."
                ),
            )
        ]
        self.session: ModelSession | None = None
        self.store = SqliteStore(settings.data_dir)
        self.store.reconcile_interrupted_state()
        self.datasets = self.store.load_benchmark_datasets()
        self.benchmark_catalog = BenchmarkCatalog(
            settings.data_dir,
            self.datasets,
            max_custom_dataset_bytes=settings.max_custom_dataset_bytes,
        )
        self.items = self.benchmark_catalog.get_adapter(FIXTURE_BENCHMARK_ID).items()
        self.cohorts = self.store.load_cohorts()
        self.model_sessions = self.store.load_model_sessions()
        self.runs = {
            run_id: RunArtifacts(
                run=run,
                routing=routing,
                provenance=provenance,
            )
            for run_id, (run, routing, provenance) in self.store.load_runs().items()
        }
        self.jobs = {
            job_id: JobArtifacts(record=job, payload=payload)
            for job_id, (job, payload) in self.store.load_jobs().items()
        }
        self.profiles = self.store.load_expert_profiles()
        self.comparisons = self.store.load_comparisons()
        self.judges = JudgeService(settings=settings, cache=self.store)
        self._job_guard = Lock()
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self._sandbox_cleanup_task: asyncio.Task[None] | None = None
        self.runtime: ModelRuntime = self._create_runtime()
        self.server = self._create_server()
        self.agentic = AgenticController(
            settings=settings,
            store=self.store,
            runtime=self.runtime,
            topology=self.topology,
            model_id=MODEL_ID,
            judge_service=self.judges,
        )

    def _create_runtime(self) -> ModelRuntime:
        if self.settings.mode == "vllm":
            return VllmRuntime(
                base_url=self.settings.vllm_base_url,
                model_id=MODEL_ID,
                topology=self.topology,
            )
        return MockModelRuntime(self.topology)

    def _create_server(self) -> ManagedVllmServer | None:
        if self.settings.mode != "vllm":
            return None
        return ManagedVllmServer(
            command=self.settings.vllm_command,
            base_url=self.settings.vllm_base_url,
            model_id=MODEL_ID,
            data_dir=self.settings.data_dir,
            startup_timeout_seconds=self.settings.vllm_startup_timeout_seconds,
            shutdown_timeout_seconds=self.settings.vllm_shutdown_timeout_seconds,
            poll_interval_seconds=self.settings.vllm_poll_interval_seconds,
        )

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        return self.benchmark_catalog.list_benchmarks()

    async def startup(self) -> None:
        if self.server is not None:
            await self.server.recover_interrupted_process()
        await self.agentic.cleanup_interrupted_sandboxes()
        if self._sandbox_cleanup_task is None or self._sandbox_cleanup_task.done():
            self._sandbox_cleanup_task = asyncio.create_task(
                self._sandbox_cleanup_loop(),
                name="moe-tools-sandbox-cleanup",
            )

    async def _sandbox_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.agent_cleanup_retry_seconds)
            try:
                await self.agentic.cleanup_interrupted_sandboxes()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("periodic sandbox cleanup failed")

    def list_benchmark_items(
        self,
        benchmark_id: str,
        *,
        search: str = "",
        category: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> BenchmarkItemPage:
        return self.benchmark_catalog.list_items(
            benchmark_id,
            search=search,
            category=category,
            offset=offset,
            limit=limit,
        )

    def get_benchmark_problem(
        self, benchmark_id: str, item_id: str
    ) -> BenchmarkProblemDetail:
        adapter = self.benchmark_catalog.get_adapter(benchmark_id)
        item = next((item for item in adapter.items() if item.id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        contract = _default_evaluation_contract(adapter.info, item)
        public_criteria = [
            criterion
            for criterion in contract.criteria
            if criterion.visibility is CriterionVisibility.PUBLIC
        ]
        hidden_criteria = [
            criterion
            for criterion in contract.criteria
            if criterion.visibility is CriterionVisibility.HIDDEN
        ]
        hidden_hash = (
            hashlib.sha256(
                json.dumps(
                    [
                        criterion.model_dump(mode="json")
                        for criterion in hidden_criteria
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if hidden_criteria
            else None
        )
        public_contract = (
            EvaluationContract(
                name=contract.name,
                description=contract.description,
                criteria=public_criteria,
                aggregation=contract.aggregation,
                pass_threshold=contract.pass_threshold,
                judge=contract.judge,
                judge_weight=contract.judge_weight,
                judge_can_override_deterministic_failure=(
                    contract.judge_can_override_deterministic_failure
                ),
            )
            if public_criteria
            else None
        )
        return BenchmarkProblemDetail(
            benchmark=adapter.info,
            item=item,
            rendered_prompt=adapter.render_prompt(item),
            success_criteria=SuccessCriteriaView(
                summary=_criterion_summary(item),
                public_criteria=public_criteria,
                hidden_criteria_count=len(hidden_criteria),
                hidden_criteria_hash=hidden_hash,
                evaluation_contract_fingerprint=contract.fingerprint,
                public_contract=public_contract,
            ),
            default_execution_policy=ExecutionPolicy(),
        )

    def evaluation_capabilities(self) -> EvaluationCapabilities:
        return self.judges.capabilities()

    async def judge(self, request: JudgeEvaluationRequest) -> LLMJudgeResult:
        return await self.judges.evaluate(request)

    def import_custom_benchmark(
        self, request: CreateCustomBenchmarkRequest
    ) -> BenchmarkDatasetRecord:
        record = self.benchmark_catalog.import_custom(request)
        self.datasets[record.id] = record
        self.store.save_benchmark_dataset(record)
        return record

    async def create_model_session(
        self,
        request: CreateModelSessionRequest,
        *,
        session_id: str | None = None,
    ) -> ModelSession:
        request = self._resolve_model_session_request(request)
        previous_session = self.session
        if previous_session is not None:
            previous_session.state = ModelState.STOPPING
            self.store.save_model_session(previous_session)
            try:
                if self.server is not None:
                    await self.server.stop()
            except Exception:
                previous_session.state = ModelState.FAILED
                self.store.save_model_session(previous_session)
                raise
            previous_session.state = ModelState.STOPPED
            self.store.save_model_session(previous_session)

        matched_profile = (
            self._find_saved_profile(request.model_id, request.profile)
            if request.profile is not None
            else None
        )
        profile_id = request.profile_id or (
            matched_profile.id if matched_profile is not None else None
        )
        planned_session = (
            self.model_sessions.get(session_id) if session_id is not None else None
        )
        if planned_session is not None:
            if (
                planned_session.model_id != request.model_id
                or planned_session.profile != request.profile
                or planned_session.profile_id != profile_id
            ):
                raise RuntimeError("model-load job does not match its planned session")
            planned_session.state = ModelState.STARTING
            self.session = planned_session
        else:
            self.session = ModelSession(
                id=session_id or str(uuid4()),
                model_id=request.model_id,
                state=ModelState.STARTING,
                mode=self.settings.mode,
                profile=request.profile,
                profile_id=profile_id,
            )
        self.model_sessions[self.session.id] = self.session
        self.store.save_model_session(self.session)
        try:
            if self.server is not None:
                await self.server.start(
                    profile=request.profile,
                    session_id=self.session.id,
                )
        except Exception:
            self.session.state = ModelState.FAILED
            self.store.save_model_session(self.session)
            raise
        self.session.state = ModelState.READY
        self.store.save_model_session(self.session)
        return self.session

    def list_model_sessions(self) -> list[ModelSession]:
        return sorted(
            (
                model_session.model_copy(deep=True)
                for model_session in self.model_sessions.values()
            ),
            key=lambda model_session: model_session.created_at,
            reverse=True,
        )

    def current_model_session(self) -> ModelSession | None:
        with self._job_guard:
            active = self._active_job_artifacts_unlocked()
            if (
                active is not None
                and active.record.kind is JobKind.MODEL_LOAD
                and active.record.result_id is not None
            ):
                planned = self.model_sessions.get(active.record.result_id)
                if planned is not None:
                    return planned.model_copy(deep=True)
            return self.session.model_copy(deep=True) if self.session else None

    def runtime_status(self) -> RuntimeStatus:
        current_session = self.current_model_session()
        if self.server is None:
            return RuntimeStatus(
                managed=False,
                model_id=MODEL_ID,
                session_id=current_session.id if current_session else None,
            )
        return RuntimeStatus(
            managed=True,
            model_id=MODEL_ID,
            pid=self.server.pid,
            session_id=current_session.id if current_session else None,
            started_at=self.server.started_at,
            log_path=str(self.server.log_path) if self.server.log_path else None,
            profile_path=(
                str(self.server.profile_path) if self.server.profile_path else None
            ),
            log_tail=self.server.read_log_tail(),
        )

    def submit_model_session(self, request: CreateModelSessionRequest) -> JobRecord:
        request = self._resolve_model_session_request(request)
        payload = request.model_dump(mode="json")
        with self._job_guard:
            active = self._active_job_artifacts_unlocked()
            if active is not None:
                if (
                    active.record.kind is JobKind.MODEL_LOAD
                    and active.payload == payload
                ):
                    return active.record.model_copy(deep=True)
                raise ActiveJobConflict(active.record)

            matched_profile = (
                self._find_saved_profile(request.model_id, request.profile)
                if request.profile is not None
                else None
            )
            model_session = ModelSession(
                id=str(uuid4()),
                model_id=request.model_id,
                state=ModelState.STARTING,
                mode=self.settings.mode,
                profile=request.profile,
                profile_id=request.profile_id
                or (matched_profile.id if matched_profile is not None else None),
            )
            job = JobRecord(
                id=str(uuid4()),
                kind=JobKind.MODEL_LOAD,
                status=JobStatus.QUEUED,
                progress_total=1,
                result_id=model_session.id,
            )
            self.store.save_model_load_submission(model_session, job, payload)
            self.model_sessions[model_session.id] = model_session
            self.jobs[job.id] = JobArtifacts(record=job, payload=payload)
            return job.model_copy(deep=True)

    def submit_benchmark(self, request: RunRequest) -> JobRecord:
        self._ensure_no_active_job()
        adapter, selected, generation, contract, policy = self._select_benchmark_items(
            request
        )
        cohort = self._create_cohort(
            adapter,
            selected,
            generation,
            contract,
            policy,
        )
        normalized_request = request.model_copy(
            update={
                "generation": generation,
                "evaluation_contract": contract,
                "execution_policy": policy,
            }
        )
        return self._queue_job(
            kind=JobKind.BENCHMARK_RUN,
            payload={
                "request": normalized_request.model_dump(mode="json"),
                "cohort_id": cohort.id,
            },
            progress_total=len(selected) * policy.attempts,
        )

    def submit_prepare_benchmark(self, benchmark_id: str) -> JobRecord:
        self._ensure_no_active_job()
        self.benchmark_catalog.get_info(benchmark_id)
        return self._queue_job(
            kind=JobKind.DATASET_PREPARE,
            payload={"benchmark_id": benchmark_id},
            progress_total=1,
        )

    def submit_agent_run(self, request: CreateAgentRunRequest) -> JobRecord:
        self._ensure_no_active_job()
        request = self.agentic.normalize_run_request(request)
        model_session = self.model_sessions.get(request.model_session_id)
        self.agentic.validate_run_request(request, model_session)
        run_id = str(uuid4())
        job_id = str(uuid4())
        job = self._queue_job(
            kind=JobKind.AGENT_RUN,
            payload={"run_id": run_id},
            progress_total=len(request.task_ids) * request.attempts,
            job_id=job_id,
            result_id=run_id,
        )
        assert model_session is not None
        self.agentic.create_run(
            run_id=run_id,
            job_id=job_id,
            request=request,
            model_session=model_session,
        )
        return job

    async def execute_job(self, job_id: str) -> None:
        with self._job_guard:
            artifacts = self.jobs[job_id]
            job = artifacts.record
            if job.status is not JobStatus.QUEUED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            self.store.save_job(job, artifacts.payload)

        try:
            if job.kind is JobKind.MODEL_LOAD:
                model_request = CreateModelSessionRequest.model_validate(
                    artifacts.payload
                )
                result = await self.create_model_session(
                    model_request,
                    session_id=job.result_id,
                )
                self._update_job_progress(job_id, 1, 1)
            elif job.kind is JobKind.BENCHMARK_RUN:
                run_request = RunRequest.model_validate(artifacts.payload["request"])
                result = await self.run_benchmark(
                    run_request,
                    job_id=job_id,
                    cohort_id=str(artifacts.payload["cohort_id"]),
                )
            elif job.kind is JobKind.DATASET_PREPARE:
                result = await self.benchmark_catalog.prepare(
                    str(artifacts.payload["benchmark_id"]),
                    on_progress=lambda current, total: self._update_job_progress(
                        job_id, current, total
                    ),
                    should_cancel=lambda: self._job_cancelled(job_id),
                )
            else:
                run_id = str(artifacts.payload["run_id"])
                agent_run = self.agentic.runs[run_id]
                model_session = self.model_sessions.get(agent_run.model_session_id)
                if model_session is None:
                    raise RuntimeError("agent run references a missing model session")
                result = await self.agentic.execute_run(
                    run_id,
                    model_session=model_session,
                    on_progress=lambda current, total: self._update_job_progress(
                        job_id, current, total
                    ),
                    should_cancel=lambda: self._job_cancelled(job_id),
                )
        except asyncio.CancelledError:
            with self._job_guard:
                if job.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                    job.status = JobStatus.CANCELLED
                    job.error = None
                    job.completed_at = job.completed_at or datetime.now(UTC)
                    self.store.save_job(job, artifacts.payload)
                elif job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    job.status = JobStatus.FAILED
                    job.error = "job execution was interrupted during shutdown"
                    job.completed_at = datetime.now(UTC)
                    self._fail_planned_model_session_unlocked(job)
                    self.store.save_job(job, artifacts.payload)
            raise
        except Exception as error:
            with self._job_guard:
                if job.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                    job.status = JobStatus.CANCELLED
                    job.error = None
                    job.completed_at = job.completed_at or datetime.now(UTC)
                    self.store.save_job(job, artifacts.payload)
                    return
                logger.exception("job %s failed", job_id)
                job.status = JobStatus.FAILED
                job.error = str(error) or error.__class__.__name__
                job.completed_at = datetime.now(UTC)
                self._fail_planned_model_session_unlocked(job)
                self.store.save_job(job, artifacts.payload)
            return

        with self._job_guard:
            job.result_id = result.id
            if job.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                job.status = JobStatus.CANCELLED
                job.completed_at = job.completed_at or datetime.now(UTC)
                self.store.save_job(job, artifacts.payload)
                return
            job.status = JobStatus.COMPLETED
            job.progress_current = job.progress_total
            job.completed_at = datetime.now(UTC)
            self.store.save_job(job, artifacts.payload)

    def start_job(self, job_id: str) -> asyncio.Task[None]:
        """Start and track a persisted job before its API response is sent."""
        loop = asyncio.get_running_loop()
        with self._job_guard:
            task = self._job_tasks.get(job_id)
            if task is None or task.done():
                task = loop.create_task(
                    self.execute_job(job_id),
                    name=f"moe-tools-job-{job_id}",
                )
                self._job_tasks[job_id] = task
                task.add_done_callback(
                    lambda completed, tracked_id=job_id: self._discard_job_task(
                        tracked_id,
                        completed,
                    )
                )
            return task

    async def dispatch_job(self, job_id: str) -> None:
        """Wait for tracked execution without propagating waiter cancellation."""
        task = self.start_job(job_id)
        await asyncio.shield(task)

    def list_jobs(self) -> list[JobRecord]:
        return sorted(
            (
                artifacts.record.model_copy(deep=True)
                for artifacts in self.jobs.values()
            ),
            key=lambda job: job.created_at,
            reverse=True,
        )

    def active_job(self) -> JobRecord | None:
        with self._job_guard:
            active = self._active_job_artifacts_unlocked()
            return active.record.model_copy(deep=True) if active else None

    def get_job(self, job_id: str) -> JobRecord:
        with self._job_guard:
            artifacts = self.jobs.get(job_id)
            if artifacts is None:
                raise KeyError(job_id)
            return artifacts.record.model_copy(deep=True)

    def cancel_job(self, job_id: str) -> JobRecord:
        with self._job_guard:
            artifacts = self.jobs.get(job_id)
            if artifacts is None:
                raise KeyError(job_id)
            job = artifacts.record
            if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                return job.model_copy(deep=True)
            if job.kind is JobKind.MODEL_LOAD:
                raise ValueError("model-load cancellation is not implemented")
            if job.status is JobStatus.RUNNING:
                job.status = JobStatus.CANCELLING
                if job.kind is JobKind.AGENT_RUN:
                    self.agentic.request_cancel(str(artifacts.payload["run_id"]))
            else:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(UTC)
                if job.kind is JobKind.AGENT_RUN:
                    self.agentic.cancel_queued_run(str(artifacts.payload["run_id"]))
            self.store.save_job(job, artifacts.payload)
            return job.model_copy(deep=True)

    async def shutdown(self) -> None:
        cleanup_task = self._sandbox_cleanup_task
        self._sandbox_cleanup_task = None
        if cleanup_task is not None:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        with self._job_guard:
            active_job_tasks = list(self._job_tasks.values())
        for task in active_job_tasks:
            task.cancel()
        if active_job_tasks:
            await asyncio.gather(*active_job_tasks, return_exceptions=True)
        if self.session is not None and self.session.state is ModelState.READY:
            self.session.state = ModelState.STOPPING
            self.store.save_model_session(self.session)
        try:
            await self.agentic.cleanup_interrupted_sandboxes()
            if self.server is not None:
                await self.server.aclose()
        finally:
            try:
                await self.agentic.aclose()
            finally:
                try:
                    await self.judges.aclose()
                finally:
                    await self.runtime.aclose()
        if self.session is not None and self.session.state is ModelState.STOPPING:
            self.session.state = ModelState.STOPPED
            self.store.save_model_session(self.session)

    def _resolve_model_session_request(
        self, request: CreateModelSessionRequest
    ) -> CreateModelSessionRequest:
        if request.model_id != MODEL_ID:
            raise ValueError(f"unsupported model {request.model_id!r}")
        profile = request.profile
        if request.profile_id is not None:
            saved_profile = self.profiles.get(request.profile_id)
            if saved_profile is None:
                raise ValueError("profile_id does not reference a saved profile")
            if saved_profile.model_id != request.model_id:
                raise ValueError("saved profile belongs to a different model")
            if profile is not None and saved_profile.profile != profile:
                raise ValueError("profile does not match saved profile_id")
            profile = saved_profile.profile.model_copy(deep=True)
        if profile is not None:
            validation = validate_profile(profile, self.topology)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
        return request.model_copy(update={"profile": profile})

    async def run_benchmark(
        self,
        request: RunRequest,
        *,
        job_id: str | None = None,
        cohort_id: str | None = None,
    ) -> BenchmarkRun:
        adapter, selected, generation, contract, policy = self._select_benchmark_items(
            request
        )
        expected_cohort = self._create_cohort(
            adapter,
            selected,
            generation,
            contract,
            policy,
        )
        cohort = expected_cohort
        if cohort_id is not None:
            cohort = self.cohorts.get(cohort_id)
            if cohort is None:
                raise ValueError("benchmark job references a missing cohort")
            if cohort.fingerprint != expected_cohort.fingerprint:
                raise ValueError("benchmark request no longer matches its cohort")
        session = self.session
        if session is None:
            raise RuntimeError("load a model before starting a benchmark")

        run_id = str(uuid4())
        counts = np.zeros(
            (self.topology.num_layers, self.topology.num_experts), dtype=np.int64
        )
        mass = np.zeros(counts.shape, dtype=np.float64)
        total_slots = 0
        results: list[RunItemResult] = []
        run_started = time.perf_counter()
        evaluation_time_ms = 0.0
        work = [
            (item, attempt)
            for item in selected
            for attempt in range(1, policy.attempts + 1)
        ]
        total_work = len(work)
        stop = False
        termination_reason: str | None = None
        accounted_cost_usd = 0.0
        for batch_start in range(0, total_work, policy.concurrency):
            if stop or (job_id is not None and self._job_cancelled(job_id)):
                break
            elapsed = time.perf_counter() - run_started
            if elapsed >= policy.timeout_seconds:
                termination_reason = "execution_policy_timeout"
                break
            consumed_tokens = sum(result.total_tokens for result in results)
            remaining_token_budget = (
                policy.max_tokens - consumed_tokens
                if policy.max_tokens is not None
                else None
            )
            if remaining_token_budget is not None and remaining_token_budget <= 0:
                termination_reason = "max_tokens"
                break
            remaining_cost_budget = (
                max(0.0, policy.max_cost_usd - accounted_cost_usd)
                if policy.max_cost_usd is not None
                else None
            )
            if remaining_cost_budget is not None and remaining_cost_budget <= 0:
                termination_reason = "max_cost_usd"
                break
            batch = work[batch_start : batch_start + policy.concurrency]
            remaining_timeout = max(0.001, policy.timeout_seconds - elapsed)
            try:
                async with asyncio.timeout(remaining_timeout):
                    calls = [
                        self._execute_benchmark_item(
                            adapter=adapter,
                            item=item,
                            attempt=attempt,
                            generation=generation,
                            contract=contract,
                            policy=policy,
                            session=session,
                            run_id=run_id,
                            token_budget=remaining_token_budget,
                            cost_budget_usd=remaining_cost_budget,
                        )
                        for item, attempt in batch
                    ]
                    executions = (
                        [await calls[0]]
                        if len(calls) == 1
                        else await asyncio.gather(*calls)
                    )
            except TimeoutError:
                termination_reason = "execution_policy_timeout"
                break
            for execution in executions:
                results.append(execution.result)
                evaluation_time_ms += execution.evaluation_time_ms
                accounted_cost_usd += execution.budget_debit_usd
                if execution.routing is not None:
                    counts += execution.routing.selection_counts
                    mass += execution.routing.routing_mass
                    total_slots += execution.routing.total_routed_slots
                if policy.fail_fast and execution.result.passed is False:
                    stop = True
                    termination_reason = "fail_fast"
                if policy.max_tokens is not None and (
                    execution.token_budget_exhausted
                    or sum(result.total_tokens for result in results)
                    >= policy.max_tokens
                ):
                    stop = True
                    termination_reason = "max_tokens"
                if policy.max_cost_usd is not None and (
                    execution.cost_budget_exhausted
                    or accounted_cost_usd >= policy.max_cost_usd
                ):
                    stop = True
                    termination_reason = "max_cost_usd"
            if job_id is not None:
                self._update_job_progress(job_id, len(results), total_work)

        scored = [item for item in results if item.passed is not None]
        passed = sum(item.passed is True for item in scored)
        cancelled = job_id is not None and self._job_cancelled(job_id)
        if cancelled:
            termination_reason = "cancelled"
        performance = _run_performance(
            results,
            wall_time_ms=(time.perf_counter() - run_started) * 1000,
            evaluation_time_ms=evaluation_time_ms,
        )
        run = BenchmarkRun(
            id=run_id,
            benchmark_id=request.benchmark_id,
            model_session_id=session.id,
            status="cancelled" if cancelled else "completed",
            score=passed / len(scored) if scored else None,
            scored_items=len(scored),
            completed_items=len(results),
            total_items=total_work,
            items=results,
            cohort_id=cohort.id,
            evaluation_contract_fingerprint=contract.fingerprint,
            execution_policy_fingerprint=policy.fingerprint,
            performance=performance,
            termination_reason=termination_reason,
        )
        routing = RoutingSummary(
            run_id=run_id,
            layer_ids=self.topology.routed_layer_ids,
            selection_counts=counts.tolist(),
            routing_mass=mass.tolist(),
            total_routed_slots=total_slots,
        )
        provenance = RunProvenance(
            run_id=run_id,
            cohort_id=cohort.id,
            cohort_fingerprint=cohort.fingerprint,
            model_id=session.model_id,
            model_session_id=session.id,
            profile_id=session.profile_id,
            profile_fingerprint=(
                _profile_fingerprint(session.profile)
                if session.profile is not None
                else None
            ),
            benchmark_id=adapter.info.id,
            benchmark_revision=adapter.info.revision,
            dataset_content_hash=adapter.content_hash,
            prompt_template_version=adapter.info.prompt_template_version,
            scoring_version=adapter.scoring_version,
            generation=generation,
            evaluation_contract=contract,
            execution_policy=policy,
            app_version=__version__,
        )
        self.runs[run_id] = RunArtifacts(
            run=run,
            routing=routing,
            provenance=provenance,
        )
        self.store.save_run(run, routing, provenance)
        return run

    async def _execute_benchmark_item(
        self,
        *,
        adapter: BenchmarkAdapter,
        item: BenchmarkItem,
        attempt: int,
        generation: GenerationConfig,
        contract: EvaluationContract,
        policy: ExecutionPolicy,
        session: ModelSession,
        run_id: str,
        token_budget: int | None,
        cost_budget_usd: float | None,
    ) -> BenchmarkItemExecution:
        prompt = adapter.render_prompt(item)
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        output = ""
        prompt_token_reserve = _prompt_token_upper_bound(prompt)
        if token_budget is not None and token_budget <= prompt_token_reserve:
            return BenchmarkItemExecution(
                result=RunItemResult(
                    item_id=item.id,
                    attempt=attempt,
                    prompt=prompt,
                    expected=item.expected,
                    output="",
                    passed=(None if item.scoring is ScoringMode.UNGRADED else False),
                    scoring=item.scoring,
                    error=(
                        "inference skipped: conservative prompt-token reserve "
                        "does not fit inside the remaining run token cap"
                    ),
                    latency_ms=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    performance=InferencePerformance(
                        started_at=started_at,
                        completed_at=started_at,
                    ),
                ),
                routing=None,
                token_budget_exhausted=True,
            )
        try:
            output_token_budget = (
                token_budget - prompt_token_reserve
                if token_budget is not None
                else None
            )
            attempt_generation = generation.model_copy(
                update={
                    "seed": (
                        generation.seed + attempt - 1
                        if generation.seed is not None
                        else None
                    ),
                    "max_tokens": min(
                        generation.max_tokens,
                        output_token_budget,
                    )
                    if output_token_budget is not None
                    else generation.max_tokens,
                }
            )
            request_key = item.id if attempt == 1 else f"{item.id}:attempt-{attempt}"
            async with asyncio.timeout(policy.per_item_timeout_seconds):
                completion = await self.runtime.complete(
                    prompt,
                    request_key=request_key,
                    profile=session.profile,
                    generation=attempt_generation,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            routing = aggregate_routing(completion.routing, self.topology)
            output = completion.content.strip()
            evaluation_started = time.perf_counter()
            judge_error = None
            judge_budget_debit = 0.0
            judge_cost_uncertain = False
            cost_budget_exhausted = False
            judge_request = self._benchmark_judge_request(
                contract,
                item=item,
                prompt=prompt,
                output=output,
            )
            try:
                judge_result = (
                    await self.judges.evaluate(
                        judge_request,
                        max_incurred_cost_usd=cost_budget_usd,
                    )
                    if judge_request is not None
                    else None
                )
                if judge_result is not None:
                    judge_budget_debit, judge_cost_uncertain = _judge_budget_accounting(
                        judge_result
                    )
            except JudgeBudgetExceeded as error:
                judge_result = None
                judge_error = str(error)[:1000]
                cost_budget_exhausted = True
            except JudgeProviderError as error:
                judge_result = None
                judge_error = str(error)[:1000]
                judge_cost_uncertain = True
                judge_budget_debit = (
                    cost_budget_usd
                    if cost_budget_usd is not None
                    else (
                        judge_request_cost_upper_bound(judge_request) or 0
                        if judge_request is not None
                        else 0
                    )
                )
                if cost_budget_usd is not None:
                    cost_budget_exhausted = True
            evaluation = evaluate_output(
                contract,
                output=output,
                expected=item.expected,
                benchmark_scorer=lambda candidate: adapter.score(item, candidate),
                judge=judge_result,
            )
            if judge_error is not None:
                evaluation.passed = False
                evaluation.error = judge_error
            evaluation_time_ms = (time.perf_counter() - evaluation_started) * 1000
            runtime_performance = completion.performance
            tokens_per_second = runtime_performance.tokens_per_second
            total_tokens = completion.prompt_tokens + completion.completion_tokens
            performance = InferencePerformance(
                prompt_tokens=completion.prompt_tokens,
                reasoning_tokens=completion.reasoning_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                ttft_ms=runtime_performance.time_to_first_token_ms,
                decode_ms=runtime_performance.generation_time_ms,
                queue_time_ms=runtime_performance.queue_time_ms,
                mean_inter_token_latency_ms=(
                    runtime_performance.mean_inter_token_latency_ms
                ),
                tokens_per_second=tokens_per_second,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            return BenchmarkItemExecution(
                result=RunItemResult(
                    item_id=item.id,
                    attempt=attempt,
                    prompt=prompt,
                    expected=item.expected,
                    output=output,
                    passed=evaluation.passed,
                    scoring=item.scoring,
                    latency_ms=latency_ms,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    reasoning_tokens=completion.reasoning_tokens,
                    total_tokens=total_tokens,
                    judge_budget_debit_usd=judge_budget_debit,
                    judge_cost_uncertain=judge_cost_uncertain,
                    evaluation=evaluation,
                    performance=performance,
                ),
                routing=routing,
                evaluation_time_ms=evaluation_time_ms,
                budget_debit_usd=judge_budget_debit,
                cost_budget_exhausted=cost_budget_exhausted,
            )
        except Exception as error:
            logger.exception(
                "benchmark item %s attempt %s failed in run %s",
                item.id,
                attempt,
                run_id,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            return BenchmarkItemExecution(
                result=RunItemResult(
                    item_id=item.id,
                    attempt=attempt,
                    prompt=prompt,
                    expected=item.expected,
                    output=output,
                    passed=(None if item.scoring is ScoringMode.UNGRADED else False),
                    scoring=item.scoring,
                    error=(str(error) or error.__class__.__name__)[:1000],
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    performance=InferencePerformance(
                        latency_ms=latency_ms,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    ),
                ),
                routing=None,
            )

    def _benchmark_judge_request(
        self,
        contract: EvaluationContract,
        *,
        item: BenchmarkItem,
        prompt: str,
        output: str,
    ) -> JudgeEvaluationRequest | None:
        if contract.judge is None:
            return None
        reference = (
            item.expected if contract.judge.mode is JudgeMode.REFERENCE else None
        )
        request = JudgeEvaluationRequest(
            config=contract.judge,
            task=prompt,
            candidate=output,
            reference=reference,
            criteria=[
                criterion.description or criterion.label
                for criterion in contract.criteria
                if criterion.visibility is CriterionVisibility.PUBLIC
            ],
        )
        return request

    def _select_benchmark_items(
        self, request: RunRequest
    ) -> tuple[
        BenchmarkAdapter,
        list[BenchmarkItem],
        GenerationConfig,
        EvaluationContract,
        ExecutionPolicy,
    ]:
        if self.session is None or self.session.state is not ModelState.READY:
            raise RuntimeError("load a model before starting a benchmark")
        adapter = self.benchmark_catalog.get_adapter(request.benchmark_id)
        items = adapter.items()
        raw_ids = (
            [item.id for item in items]
            if request.item_ids is None
            else request.item_ids
        )
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("benchmark item IDs cannot contain duplicates")
        requested_ids = set(raw_ids)
        selected = [item for item in items if item.id in requested_ids]
        missing = requested_ids - {item.id for item in selected}
        if missing:
            raise ValueError(f"unknown benchmark items: {sorted(missing)}")
        if not selected:
            raise ValueError("select at least one benchmark item")
        generation = request.generation or adapter.info.default_generation
        contract = request.evaluation_contract or _default_evaluation_contract(
            adapter.info,
            selected[0],
        )
        policy = request.execution_policy or ExecutionPolicy()
        if contract.judge is not None and contract.judge.mode is JudgeMode.PAIRWISE:
            raise ValueError(
                "pairwise judging requires the explicit evaluation/judge endpoint"
            )
        validate_cost_policy_pricing(contract, policy)
        if policy.concurrency > 1 and (
            policy.max_tokens is not None
            or policy.max_cost_usd is not None
            or policy.fail_fast
        ):
            raise ValueError(
                "concurrency must be 1 when using token, cost, or fail-fast caps"
            )
        return adapter, selected, generation, contract, policy

    def _create_cohort(
        self,
        adapter: BenchmarkAdapter,
        selected: list[BenchmarkItem],
        generation: GenerationConfig,
        evaluation_contract: EvaluationContract,
        execution_policy: ExecutionPolicy,
    ) -> BenchmarkCohort:
        item_ids = [item.id for item in selected]
        fingerprint = _cohort_fingerprint(
            adapter,
            item_ids,
            generation,
            evaluation_contract,
            execution_policy,
        )
        existing = next(
            (
                cohort
                for cohort in self.cohorts.values()
                if cohort.fingerprint == fingerprint
            ),
            None,
        )
        if existing is not None:
            return existing
        cohort = BenchmarkCohort(
            id=str(uuid4()),
            benchmark_id=adapter.info.id,
            benchmark_revision=adapter.info.revision,
            dataset_content_hash=adapter.content_hash,
            prompt_template_version=adapter.info.prompt_template_version,
            scoring_version=adapter.scoring_version,
            item_ids=item_ids,
            fingerprint=fingerprint,
            generation=generation,
            evaluation_contract=evaluation_contract,
            execution_policy=execution_policy,
        )
        self.cohorts[cohort.id] = cohort
        self.store.save_cohort(cohort)
        return cohort

    def _queue_job(
        self,
        *,
        kind: JobKind,
        payload: dict[str, object],
        progress_total: int,
        job_id: str | None = None,
        result_id: str | None = None,
    ) -> JobRecord:
        with self._job_guard:
            active = self._active_job_artifacts_unlocked()
            if active is not None:
                raise ActiveJobConflict(active.record)
            job = JobRecord(
                id=job_id or str(uuid4()),
                kind=kind,
                status=JobStatus.QUEUED,
                progress_total=progress_total,
                result_id=result_id,
            )
            self.jobs[job.id] = JobArtifacts(record=job, payload=payload)
            self.store.save_job(job, payload)
            return job.model_copy(deep=True)

    def _ensure_no_active_job(self) -> None:
        with self._job_guard:
            active = self._active_job_artifacts_unlocked()
            if active is not None:
                raise ActiveJobConflict(active.record)

    def _active_job_artifacts_unlocked(self) -> JobArtifacts | None:
        active = [
            artifacts
            for artifacts in self.jobs.values()
            if artifacts.record.status
            in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING}
        ]
        return max(
            active,
            key=lambda artifacts: artifacts.record.created_at,
            default=None,
        )

    def _fail_planned_model_session_unlocked(self, job: JobRecord) -> None:
        if job.kind is not JobKind.MODEL_LOAD or job.result_id is None:
            return
        model_session = self.model_sessions.get(job.result_id)
        if model_session is None or model_session.state is not ModelState.STARTING:
            return
        model_session.state = ModelState.FAILED
        self.store.save_model_session(model_session)

    def _discard_job_task(
        self,
        job_id: str,
        task: asyncio.Task[None],
    ) -> None:
        with self._job_guard:
            if self._job_tasks.get(job_id) is task:
                self._job_tasks.pop(job_id, None)

    def _update_job_progress(self, job_id: str, current: int, total: int) -> None:
        with self._job_guard:
            artifacts = self.jobs[job_id]
            if artifacts.record.status is JobStatus.CANCELLED:
                return
            artifacts.record.progress_current = current
            artifacts.record.progress_total = total
            self.store.save_job(artifacts.record, artifacts.payload)

    def _job_cancelled(self, job_id: str) -> bool:
        with self._job_guard:
            return self.jobs[job_id].record.status in {
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            }

    def list_runs(self, *, offset: int, limit: int) -> BenchmarkRunPage:
        runs = sorted(
            (artifacts.run for artifacts in self.runs.values()),
            key=lambda run: run.created_at,
            reverse=True,
        )
        return BenchmarkRunPage(
            items=[_run_summary(run) for run in runs[offset : offset + limit]],
            total=len(runs),
            offset=offset,
            limit=limit,
        )

    def get_run(self, run_id: str) -> BenchmarkRunSummary:
        artifacts = self.runs.get(run_id)
        if artifacts is None:
            raise KeyError(run_id)
        return _run_summary(artifacts.run)

    def get_run_items(
        self,
        run_id: str,
        *,
        offset: int,
        limit: int,
    ) -> RunItemResultPage:
        artifacts = self.runs.get(run_id)
        if artifacts is None:
            raise KeyError(run_id)
        items = artifacts.run.items
        return RunItemResultPage(
            run_id=run_id,
            items=items[offset : offset + limit],
            total=len(items),
            offset=offset,
            limit=limit,
        )

    def get_run_detail(self, run_id: str) -> RunDetail:
        artifacts = self.runs.get(run_id)
        if artifacts is None:
            raise KeyError(run_id)
        model_session = self.model_sessions.get(artifacts.run.model_session_id)
        if model_session is None:
            raise RuntimeError("run references a missing model session")
        saved_profile = (
            self.profiles.get(model_session.profile_id)
            if model_session.profile_id is not None
            else (
                self._find_saved_profile(model_session.model_id, model_session.profile)
                if model_session.profile is not None
                else None
            )
        )
        return RunDetail(
            run=_run_summary(artifacts.run),
            model_session=model_session,
            saved_profile=saved_profile,
            cohort=(
                _public_cohort(self.cohorts[artifacts.run.cohort_id])
                if artifacts.run.cohort_id is not None
                and artifacts.run.cohort_id in self.cohorts
                else None
            ),
            provenance=(
                _public_run_provenance(artifacts.provenance)
                if artifacts.provenance is not None
                else None
            ),
        )

    def create_expert_profile(
        self, request: CreateExpertProfileRequest
    ) -> SavedExpertProfile:
        if request.model_id != MODEL_ID:
            raise ValueError(f"unsupported model {request.model_id!r}")
        validation = validate_profile(request.profile, self.topology)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

        source_refs = list(request.source_refs)
        legacy_refs: list[ProfileSourceRef] = []
        if request.source_run_id is not None:
            legacy_refs.append(
                ProfileSourceRef(
                    kind=ProfileSourceKind.BENCHMARK_RUN,
                    id=request.source_run_id,
                )
            )
        if request.source_trial_id is not None:
            legacy_refs.append(
                ProfileSourceRef(
                    kind=ProfileSourceKind.AGENT_TRIAL,
                    id=request.source_trial_id,
                )
            )
        known_refs = {(ref.kind, ref.id) for ref in source_refs}
        source_refs.extend(
            ref for ref in legacy_refs if (ref.kind, ref.id) not in known_refs
        )
        source_routings: list[tuple[Any, float]] = []
        for source_ref in source_refs:
            if source_ref.kind is ProfileSourceKind.BENCHMARK_RUN:
                source = self.runs.get(source_ref.id)
                if source is None:
                    raise KeyError(source_ref.id)
                source_session = self.model_sessions.get(source.run.model_session_id)
                source_routing = source.routing
            else:
                trial = self.agentic.trials.get(source_ref.id)
                if trial is None:
                    raise KeyError(source_ref.id)
                source_session = self.model_sessions.get(trial.model_session_id)
                try:
                    source_routing = self.agentic.get_routing(trial.id)
                except KeyError as error:
                    raise ValueError(
                        f"source trial {trial.id} has no routing telemetry"
                    ) from error
            if source_session is None or source_session.model_id != request.model_id:
                raise ValueError("all profile sources must use the same model")
            source_routings.append((source_routing, source_ref.weight))

        source_fingerprint = None
        if source_refs:
            source_fingerprint = self.explore_routing(
                RoutingExploreRequest(
                    sources=[
                        RoutingSourceReference(
                            kind=RoutingSourceKind(source.kind.value),
                            id=source.id,
                            weight=source.weight,
                        )
                        for source in source_refs
                    ],
                    metric=request.metric or "routing_mass",
                )
            ).fingerprint
            if (
                request.source_fingerprint is not None
                and request.source_fingerprint != source_fingerprint
            ):
                raise ValueError(
                    "source_fingerprint does not match authoritative routing sources"
                )

        if request.parent_profile_id is not None:
            parent = self.profiles.get(request.parent_profile_id)
            if parent is None:
                raise KeyError(request.parent_profile_id)
            if parent.model_id != request.model_id:
                raise ValueError("parent profile belongs to a different model")

        saved = SavedExpertProfile(
            id=str(uuid4()),
            name=request.name,
            description=request.description,
            model_id=request.model_id,
            profile=request.profile,
            profile_fingerprint=_profile_fingerprint(request.profile),
            source=request.source,
            source_run_id=request.source_run_id,
            source_trial_id=request.source_trial_id,
            source_refs=source_refs,
            source_fingerprint=source_fingerprint,
            parent_profile_id=request.parent_profile_id,
            metric=request.metric,
            selection_strategy=request.selection_strategy,
            selection_config=request.selection_config,
            validation=validation,
            observed_mass_retained=(
                _weighted_observed_mass_retained(source_routings, request.profile)
                if source_routings
                else None
            ),
        )
        self.profiles[saved.id] = saved
        self.store.save_expert_profile(saved)
        return saved

    def explore_routing(self, request: RoutingExploreRequest) -> RoutingExploreResponse:
        primary = self._resolve_routing_sources(request.sources, request)
        comparisons = self._resolve_routing_sources(
            request.comparison_sources,
            request,
        )
        return build_routing_explorer_response(request, primary, comparisons)

    def _resolve_routing_sources(
        self,
        references: list[RoutingSourceReference],
        request: RoutingExploreRequest,
    ) -> list[ResolvedRoutingSource]:
        resolved: list[ResolvedRoutingSource] = []
        for reference in references:
            source = self._resolve_routing_source(reference)
            if (
                reference.kind is RoutingSourceKind.AGENT_TRIAL
                and request.filters.trial_ids
                and reference.id not in request.filters.trial_ids
            ):
                continue
            if (
                reference.kind is RoutingSourceKind.AGENT_TRIAL
                and request.filters.passed is not None
                and (source.status == "passed") is not request.filters.passed
            ):
                continue
            resolved.append(source)
        return resolved

    def _resolve_routing_source(
        self, reference: RoutingSourceReference
    ) -> ResolvedRoutingSource:
        if reference.kind is RoutingSourceKind.BENCHMARK_RUN:
            artifacts = self.runs.get(reference.id)
            if artifacts is None:
                raise KeyError(reference.id)
            session = self.model_sessions.get(artifacts.run.model_session_id)
            if session is None:
                raise RuntimeError("routing run references a missing model session")
            summary = artifacts.routing
            benchmark_id = (
                artifacts.provenance.benchmark_id
                if artifacts.provenance is not None
                else artifacts.run.benchmark_id
            )
            revision = (
                artifacts.provenance.benchmark_revision
                if artifacts.provenance is not None
                else None
            )
            label = benchmark_id if revision is None else f"{benchmark_id} @ {revision}"
            return ResolvedRoutingSource(
                reference=reference,
                label=f"{label} · {artifacts.run.id[:8]}",
                status=artifacts.run.status,
                model_id=session.model_id,
                topology=self.topology,
                profile_id=session.profile_id,
                profile_fingerprint=(
                    artifacts.provenance.profile_fingerprint
                    if artifacts.provenance is not None
                    else (
                        _profile_fingerprint(session.profile)
                        if session.profile is not None
                        else None
                    )
                ),
                selection_counts=np.asarray(summary.selection_counts),
                routing_mass=np.asarray(summary.routing_mass),
                total_routed_slots=summary.total_routed_slots,
                served_tokens=(
                    artifacts.run.performance.total_tokens
                    if artifacts.run.performance is not None
                    else None
                ),
                capabilities=RoutingFilterCapabilities(),
            )

        trial = self.agentic.trials.get(reference.id)
        if trial is None:
            raise KeyError(reference.id)
        run = self.agentic.runs.get(trial.run_id)
        if run is None:
            raise RuntimeError("routing trial references a missing agent run")
        session = self.model_sessions.get(trial.model_session_id)
        if session is None:
            raise RuntimeError("routing trial references a missing model session")
        summary = self.agentic.get_routing(trial.id)
        task_title = trial.task_title_snapshot or (
            f"Unknown archived task ({trial.task_id})"
        )
        return ResolvedRoutingSource(
            reference=reference,
            label=f"{task_title} · attempt {trial.attempt}",
            status=trial.status.value,
            model_id=session.model_id,
            topology=self.topology,
            profile_id=summary.profile_id,
            profile_fingerprint=summary.profile_fingerprint,
            selection_counts=np.asarray(summary.selection_counts),
            routing_mass=np.asarray(summary.routing_mass),
            total_routed_slots=summary.total_routed_slots,
            captured_inference_calls=summary.captured_inference_calls,
            total_inference_calls=summary.inference_calls,
            served_tokens=summary.served_tokens,
            capabilities=RoutingFilterCapabilities(trial=True, outcome=True),
        )

    def list_expert_profiles(self) -> list[SavedExpertProfile]:
        return sorted(
            self.profiles.values(),
            key=lambda profile: profile.created_at,
            reverse=True,
        )

    def create_comparison(self, request: CreateComparisonRequest) -> ComparisonRecord:
        if request.baseline_run_id == request.candidate_run_id:
            raise ValueError("comparison requires two different runs")
        baseline = self.runs.get(request.baseline_run_id)
        candidate = self.runs.get(request.candidate_run_id)
        if baseline is None:
            raise KeyError(request.baseline_run_id)
        if candidate is None:
            raise KeyError(request.candidate_run_id)
        if baseline.run.status != "completed" or candidate.run.status != "completed":
            raise ValueError("only completed runs can be compared")

        baseline_session = self.model_sessions.get(baseline.run.model_session_id)
        candidate_session = self.model_sessions.get(candidate.run.model_session_id)
        if baseline_session is None or candidate_session is None:
            raise RuntimeError("comparison run references a missing model session")
        if baseline_session.model_id != candidate_session.model_id:
            raise ValueError("comparison runs use different models")
        if candidate_session.profile is None:
            raise ValueError("candidate run does not use an expert profile")
        if baseline.run.benchmark_id != candidate.run.benchmark_id:
            raise ValueError("comparison runs use different benchmarks")
        if (baseline.provenance is None) != (candidate.provenance is None):
            raise ValueError("comparison runs have incompatible provenance")
        if baseline.provenance is None:
            baseline_cohort = (
                self.cohorts.get(baseline.run.cohort_id)
                if baseline.run.cohort_id is not None
                else None
            )
            candidate_cohort = (
                self.cohorts.get(candidate.run.cohort_id)
                if candidate.run.cohort_id is not None
                else None
            )
            if (
                baseline_cohort is None
                or candidate_cohort is None
                or baseline_cohort.fingerprint != candidate_cohort.fingerprint
            ):
                raise ValueError(
                    "legacy run comparability cannot be verified without matching "
                    "persisted cohort fingerprints"
                )
        elif (
            baseline.provenance.cohort_fingerprint
            != candidate.provenance.cohort_fingerprint
        ):
            raise ValueError(
                "comparison runs use different dataset, prompt, scorer, "
                "generation, or cohort fingerprints"
            )

        baseline_item_ids = [item.item_id for item in baseline.run.items]
        candidate_item_ids = [item.item_id for item in candidate.run.items]
        if baseline_item_ids != candidate_item_ids:
            raise ValueError(
                "comparison runs must use the same ordered benchmark cohort"
            )

        paired_items = zip(baseline.run.items, candidate.run.items, strict=True)
        transitions = [
            (baseline_item.passed, candidate_item.passed)
            for baseline_item, candidate_item in paired_items
        ]
        saved_profile = (
            self.profiles.get(candidate_session.profile_id)
            if candidate_session.profile_id is not None
            else self._find_saved_profile(
                candidate_session.model_id, candidate_session.profile
            )
        )
        fingerprint = _profile_fingerprint(candidate_session.profile)
        default_name = (
            f"{saved_profile.name if saved_profile else 'Masked profile'} · "
            f"{request.baseline_run_id[:8]} → {request.candidate_run_id[:8]}"
        )
        comparison = ComparisonRecord(
            id=str(uuid4()),
            name=request.name or default_name,
            baseline_run_id=request.baseline_run_id,
            candidate_run_id=request.candidate_run_id,
            profile_id=saved_profile.id if saved_profile else None,
            profile_fingerprint=fingerprint,
            benchmark_id=baseline.run.benchmark_id,
            cohort_item_ids=baseline_item_ids,
            baseline_score=baseline.run.score,
            candidate_score=candidate.run.score,
            score_delta=(
                candidate.run.score - baseline.run.score
                if candidate.run.score is not None and baseline.run.score is not None
                else None
            ),
            regressions=sum(
                base is True and masked is False for base, masked in transitions
            ),
            recoveries=sum(
                base is False and masked is True for base, masked in transitions
            ),
            retained_passes=sum(
                base is True and masked is True for base, masked in transitions
            ),
            retained_failures=sum(
                base is False and masked is False for base, masked in transitions
            ),
            unscored_items=sum(
                base is None or masked is None for base, masked in transitions
            ),
        )
        self.comparisons[comparison.id] = comparison
        self.store.save_comparison(comparison)
        return comparison

    def list_comparisons(self) -> list[ComparisonRecord]:
        return sorted(
            self.comparisons.values(),
            key=lambda comparison: comparison.created_at,
            reverse=True,
        )

    def compare_agent_runs(
        self, request: CreateAgentComparisonRequest
    ) -> AgentRunComparison:
        baseline = self.agentic.runs.get(request.baseline_run_id)
        if baseline is None:
            raise KeyError(request.baseline_run_id)
        candidate = self.agentic.runs.get(request.candidate_run_id)
        if candidate is None:
            raise KeyError(request.candidate_run_id)
        return compare_agent_runs(
            request,
            baseline,
            candidate,
            self.agentic._run_trials(baseline.id),
            self.agentic._run_trials(candidate.id),
        )

    def _find_saved_profile(
        self, model_id: str, profile: ExpertProfile
    ) -> SavedExpertProfile | None:
        fingerprint = _profile_fingerprint(profile)
        matching = [
            saved
            for saved in self.profiles.values()
            if saved.model_id == model_id and saved.profile_fingerprint == fingerprint
        ]
        return max(matching, key=lambda saved: saved.created_at, default=None)

    def validate_profile(self, profile) -> ProfileValidation:
        return validate_profile(profile, self.topology)

    def propose_profile(self, request: ProfileProposalRequest) -> ProfileProposal:
        artifacts = self.runs.get(request.run_id)
        if artifacts is None:
            raise KeyError(request.run_id)
        return propose_fixed_budget_profile(
            artifacts.routing,
            self.topology,
            request.keep_per_layer,
            request.metric,
        )

    def propose_agent_profile(
        self,
        trial_id: str,
        *,
        keep_per_layer: int,
        metric: str,
    ) -> ProfileProposal:
        routing = self.agentic.get_routing(trial_id)
        return propose_fixed_budget_profile(
            routing,
            self.topology,
            keep_per_layer,
            metric,
        )


def _profile_fingerprint(profile: ExpertProfile) -> str:
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _weighted_observed_mass_retained(
    source_routings: list[tuple[Any, float]],
    profile: ExpertProfile,
) -> float:
    total_weight = sum(weight for _, weight in source_routings)
    return (
        sum(
            calculate_observed_mass_retained(routing, profile) * weight
            for routing, weight in source_routings
        )
        / total_weight
    )


def _public_contract(
    contract: EvaluationContract,
) -> tuple[EvaluationContract | None, int, str | None]:
    public_criteria = [
        criterion
        for criterion in contract.criteria
        if criterion.visibility is CriterionVisibility.PUBLIC
    ]
    hidden_criteria = [
        criterion
        for criterion in contract.criteria
        if criterion.visibility is CriterionVisibility.HIDDEN
    ]
    public_contract = (
        EvaluationContract(
            name=contract.name,
            description=contract.description,
            criteria=public_criteria,
            aggregation=contract.aggregation,
            pass_threshold=contract.pass_threshold,
            judge=contract.judge,
            judge_weight=contract.judge_weight,
            judge_can_override_deterministic_failure=(
                contract.judge_can_override_deterministic_failure
            ),
        )
        if public_criteria
        else None
    )
    hidden_hash = (
        hashlib.sha256(
            json.dumps(
                [criterion.model_dump(mode="json") for criterion in hidden_criteria],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if hidden_criteria
        else None
    )
    return public_contract, len(hidden_criteria), hidden_hash


def _public_cohort(cohort: BenchmarkCohort) -> BenchmarkCohortView:
    if cohort.evaluation_contract is None:
        public_contract, hidden_count, hidden_hash = None, 0, None
    else:
        public_contract, hidden_count, hidden_hash = _public_contract(
            cohort.evaluation_contract
        )
    return BenchmarkCohortView(
        id=cohort.id,
        benchmark_id=cohort.benchmark_id,
        benchmark_revision=cohort.benchmark_revision,
        dataset_content_hash=cohort.dataset_content_hash,
        prompt_template_version=cohort.prompt_template_version,
        scoring_version=cohort.scoring_version,
        item_count=len(cohort.item_ids),
        fingerprint=cohort.fingerprint,
        generation=cohort.generation,
        evaluation_contract=public_contract,
        evaluation_contract_fingerprint=(cohort.evaluation_contract_fingerprint),
        hidden_criteria_count=hidden_count,
        hidden_criteria_hash=hidden_hash,
        execution_policy=cohort.execution_policy,
        execution_policy_fingerprint=cohort.execution_policy_fingerprint,
        created_at=cohort.created_at,
    )


def _public_run_provenance(provenance: RunProvenance) -> RunProvenanceView:
    contract = provenance.evaluation_contract
    if contract is None:
        public_contract, hidden_count, hidden_hash = None, 0, None
    else:
        public_contract, hidden_count, hidden_hash = _public_contract(contract)
    return RunProvenanceView(
        run_id=provenance.run_id,
        cohort_id=provenance.cohort_id,
        cohort_fingerprint=provenance.cohort_fingerprint,
        model_id=provenance.model_id,
        model_session_id=provenance.model_session_id,
        profile_id=provenance.profile_id,
        profile_fingerprint=provenance.profile_fingerprint,
        benchmark_id=provenance.benchmark_id,
        benchmark_revision=provenance.benchmark_revision,
        dataset_content_hash=provenance.dataset_content_hash,
        prompt_template_version=provenance.prompt_template_version,
        scoring_version=provenance.scoring_version,
        generation=provenance.generation,
        evaluation_contract=public_contract,
        evaluation_contract_fingerprint=(provenance.evaluation_contract_fingerprint),
        hidden_criteria_count=hidden_count,
        hidden_criteria_hash=hidden_hash,
        execution_policy=provenance.execution_policy,
        execution_policy_fingerprint=provenance.execution_policy_fingerprint,
        app_version=provenance.app_version,
    )


def _cohort_fingerprint(
    adapter: BenchmarkAdapter,
    item_ids: list[str],
    generation: GenerationConfig,
    evaluation_contract: EvaluationContract,
    execution_policy: ExecutionPolicy,
) -> str:
    canonical = json.dumps(
        {
            "benchmark_id": adapter.info.id,
            "benchmark_revision": adapter.info.revision,
            "dataset_content_hash": adapter.content_hash,
            "prompt_template_version": adapter.info.prompt_template_version,
            "scoring_version": adapter.scoring_version,
            "item_ids": item_ids,
            "generation": generation.model_dump(mode="json"),
            "evaluation_contract_fingerprint": evaluation_contract.fingerprint,
            "execution_policy_fingerprint": execution_policy.fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _default_evaluation_contract(
    benchmark: BenchmarkInfo,
    item: BenchmarkItem,
) -> EvaluationContract:
    return EvaluationContract(
        name=f"{benchmark.name} default evaluation",
        description=(
            "Uses the benchmark adapter's pinned deterministic scorer and "
            "success criteria."
        ),
        criteria=[
            EvaluationCriterion(
                id="benchmark-correctness",
                label="Benchmark correctness",
                description=_criterion_summary(item),
                kind=DeterministicScorerKind.BENCHMARK_DEFAULT,
                required=item.scoring is not ScoringMode.UNGRADED,
            )
        ],
    )


def _criterion_summary(item: BenchmarkItem) -> str:
    summaries = {
        ScoringMode.EXACT: (
            "The trimmed response must exactly match the expected answer."
        ),
        ScoringMode.GSM8K: (
            "The final numeric answer must exactly match the reference value."
        ),
        ScoringMode.MULTIPLE_CHOICE: (
            "The pinned MMLU-Pro scorer extracts the last explicit final answer "
            "letter A–J and compares that option identity with the reference."
        ),
        ScoringMode.IFEVAL: (
            "All verifiable instruction-following constraints must pass."
        ),
        ScoringMode.LIVEBENCH: (
            "The release-specific LiveBench answer extractor must match the "
            "reference answer."
        ),
        ScoringMode.REGEX: "The response must match the configured regular expression.",
        ScoringMode.CONTAINS: "The response must contain the expected value.",
        ScoringMode.UNGRADED: "This problem records output without assigning a score.",
    }
    return summaries[item.scoring]


def _run_summary(run: BenchmarkRun) -> BenchmarkRunSummary:
    return BenchmarkRunSummary.model_validate(
        run.model_dump(mode="python", exclude={"items"})
    )


def _run_performance(
    results: list[RunItemResult],
    *,
    wall_time_ms: float,
    evaluation_time_ms: float,
) -> RunPerformance:
    performances = [
        result.performance for result in results if result.performance is not None
    ]
    rates = [
        performance.tokens_per_second
        for performance in performances
        if performance.tokens_per_second is not None
    ]
    known_costs = [
        performance.estimated_cost_usd
        for performance in performances
        if performance.estimated_cost_usd is not None
    ]
    judge_costs = [
        result.evaluation.judge.usage.incurred_cost_usd
        for result in results
        if result.evaluation is not None
        and result.evaluation.judge is not None
        and result.evaluation.judge.usage.incurred_cost_usd is not None
    ]
    judge_equivalent_costs = [
        result.evaluation.judge.usage.estimated_cost_usd
        for result in results
        if result.evaluation is not None
        and result.evaluation.judge is not None
        and result.evaluation.judge.usage.estimated_cost_usd is not None
    ]
    prompt_tokens = sum(performance.prompt_tokens for performance in performances)
    known_reasoning_tokens = [
        performance.reasoning_tokens
        for performance in performances
        if performance.reasoning_tokens is not None
    ]
    completion_tokens = sum(
        performance.completion_tokens for performance in performances
    )
    return RunPerformance(
        wall_time_ms=wall_time_ms,
        model_time_ms=sum(performance.latency_ms for performance in performances),
        evaluation_time_ms=evaluation_time_ms,
        prompt_tokens=prompt_tokens,
        reasoning_tokens=(
            sum(known_reasoning_tokens) if known_reasoning_tokens else None
        ),
        completion_tokens=completion_tokens,
        total_tokens=sum(performance.total_tokens for performance in performances),
        mean_tokens_per_second=(float(np.mean(rates)) if rates else None),
        p50_tokens_per_second=(float(np.percentile(rates, 50)) if rates else None),
        p95_tokens_per_second=(float(np.percentile(rates, 95)) if rates else None),
        inference_cost_usd=(sum(known_costs) if known_costs else None),
        judge_cost_usd=(sum(judge_costs) if judge_costs else None),
        judge_equivalent_cost_usd=(
            sum(judge_equivalent_costs) if judge_equivalent_costs else None
        ),
        judge_budget_debit_usd=sum(result.judge_budget_debit_usd for result in results),
        judge_cost_uncertain=any(result.judge_cost_uncertain for result in results),
        estimated_cost_usd=(
            sum(known_costs) + sum(judge_costs) if known_costs or judge_costs else None
        ),
    )


def _judge_budget_accounting(result: LLMJudgeResult) -> tuple[float, bool]:
    incurred = result.usage.incurred_cost_usd
    if incurred is not None:
        return incurred, False
    upper_bound = result.usage.cost_upper_bound_usd
    return (upper_bound or 0), upper_bound is not None


def _prompt_token_upper_bound(prompt: str) -> int:
    return len(prompt.encode("utf-8")) + 4096
