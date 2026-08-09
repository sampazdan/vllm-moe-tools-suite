from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

import numpy as np

from .domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkRun,
    CreateModelSessionRequest,
    JobKind,
    JobRecord,
    JobStatus,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ModelTopology,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileValidation,
    RoutingSummary,
    RunItemResult,
    RunRequest,
    RuntimeStatus,
)
from .persistence import SqliteStore
from .process_manager import ManagedVllmServer
from .profiles import propose_fixed_budget_profile, validate_profile
from .runtime import MockModelRuntime, ModelRuntime, VllmRuntime
from .settings import Settings
from .telemetry import aggregate_routing

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
FIXTURE_BENCHMARK_ID = "fixture-arithmetic"
logger = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    run: BenchmarkRun
    routing: RoutingSummary


@dataclass
class JobArtifacts:
    record: JobRecord
    payload: dict[str, object]


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
        self.items = _fixture_items()
        self.session: ModelSession | None = None
        self.store = SqliteStore(settings.data_dir)
        self.store.reconcile_interrupted_sessions()
        self.store.reconcile_interrupted_jobs()
        self.model_sessions = self.store.load_model_sessions()
        self.runs = {
            run_id: RunArtifacts(run=run, routing=routing)
            for run_id, (run, routing) in self.store.load_runs().items()
        }
        self.jobs = {
            job_id: JobArtifacts(record=job, payload=payload)
            for job_id, (job, payload) in self.store.load_jobs().items()
        }
        self._job_guard = Lock()
        self.runtime: ModelRuntime = self._create_runtime()
        self.server = self._create_server()

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
        return [
            BenchmarkInfo(
                id=FIXTURE_BENCHMARK_ID,
                name="Arithmetic routing fixture",
                description=(
                    "Deterministic GPU-free prompts used to exercise scoring, "
                    "routing aggregation, and expert-profile creation."
                ),
                item_count=len(self.items),
                categories=sorted({item.category for item in self.items}),
            )
        ]

    async def create_model_session(
        self, request: CreateModelSessionRequest
    ) -> ModelSession:
        self._validate_model_session_request(request)
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

        self.session = ModelSession(
            id=str(uuid4()),
            model_id=request.model_id,
            state=ModelState.STARTING,
            mode=self.settings.mode,
            profile=request.profile,
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
            self.model_sessions.values(),
            key=lambda model_session: model_session.created_at,
            reverse=True,
        )

    def runtime_status(self) -> RuntimeStatus:
        if self.server is None:
            return RuntimeStatus(
                managed=False,
                model_id=MODEL_ID,
                session_id=self.session.id if self.session else None,
            )
        return RuntimeStatus(
            managed=True,
            model_id=MODEL_ID,
            pid=self.server.pid,
            session_id=self.session.id if self.session else None,
            started_at=self.server.started_at,
            log_path=str(self.server.log_path) if self.server.log_path else None,
            profile_path=(
                str(self.server.profile_path) if self.server.profile_path else None
            ),
            log_tail=self.server.read_log_tail(),
        )

    def submit_model_session(self, request: CreateModelSessionRequest) -> JobRecord:
        self._validate_model_session_request(request)
        return self._queue_job(
            kind=JobKind.MODEL_LOAD,
            payload=request.model_dump(mode="json"),
            progress_total=1,
        )

    def submit_benchmark(self, request: RunRequest) -> JobRecord:
        selected = self._select_benchmark_items(request)
        return self._queue_job(
            kind=JobKind.BENCHMARK_RUN,
            payload=request.model_dump(mode="json"),
            progress_total=len(selected),
        )

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
                result = await self.create_model_session(model_request)
                self._update_job_progress(job_id, 1, 1)
            else:
                run_request = RunRequest.model_validate(artifacts.payload)
                result = await self.run_benchmark(run_request, job_id=job_id)
        except Exception as error:
            logger.exception("job %s failed", job_id)
            with self._job_guard:
                job.status = JobStatus.FAILED
                job.error = str(error) or error.__class__.__name__
                job.completed_at = datetime.now(UTC)
                self.store.save_job(job, artifacts.payload)
            return

        with self._job_guard:
            job.status = JobStatus.COMPLETED
            job.progress_current = job.progress_total
            job.result_id = result.id
            job.completed_at = datetime.now(UTC)
            self.store.save_job(job, artifacts.payload)

    def list_jobs(self) -> list[JobRecord]:
        return sorted(
            (artifacts.record for artifacts in self.jobs.values()),
            key=lambda job: job.created_at,
            reverse=True,
        )

    async def shutdown(self) -> None:
        if self.session is not None and self.session.state is ModelState.READY:
            self.session.state = ModelState.STOPPING
            self.store.save_model_session(self.session)
        try:
            if self.server is not None:
                await self.server.aclose()
        finally:
            await self.runtime.aclose()
        if self.session is not None and self.session.state is ModelState.STOPPING:
            self.session.state = ModelState.STOPPED
            self.store.save_model_session(self.session)

    def _validate_model_session_request(
        self, request: CreateModelSessionRequest
    ) -> None:
        if request.model_id != MODEL_ID:
            raise ValueError(f"unsupported model {request.model_id!r}")
        if request.profile is not None:
            validation = validate_profile(request.profile, self.topology)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))

    async def run_benchmark(
        self, request: RunRequest, *, job_id: str | None = None
    ) -> BenchmarkRun:
        selected = self._select_benchmark_items(request)

        run_id = str(uuid4())
        counts = np.zeros(
            (self.topology.num_layers, self.topology.num_experts), dtype=np.int64
        )
        mass = np.zeros(counts.shape, dtype=np.float64)
        total_slots = 0
        results: list[RunItemResult] = []
        for item_number, item in enumerate(selected, start=1):
            started = time.perf_counter()
            completion = await self.runtime.complete(
                item.prompt,
                request_key=item.id,
                profile=self.session.profile,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            aggregate = aggregate_routing(completion.routing, self.topology)
            counts += aggregate.selection_counts
            mass += aggregate.routing_mass
            total_slots += aggregate.total_routed_slots
            output = completion.content.strip()
            results.append(
                RunItemResult(
                    item_id=item.id,
                    prompt=item.prompt,
                    expected=item.expected,
                    output=output,
                    passed=output == item.expected,
                    latency_ms=latency_ms,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                )
            )
            if job_id is not None:
                self._update_job_progress(job_id, item_number, len(selected))

        passed = sum(item.passed for item in results)
        run = BenchmarkRun(
            id=run_id,
            benchmark_id=request.benchmark_id,
            model_session_id=self.session.id,
            status="completed",
            score=passed / len(results),
            completed_items=len(results),
            total_items=len(results),
            items=results,
        )
        routing = RoutingSummary(
            run_id=run_id,
            layer_ids=self.topology.routed_layer_ids,
            selection_counts=counts.tolist(),
            routing_mass=mass.tolist(),
            total_routed_slots=total_slots,
        )
        self.runs[run_id] = RunArtifacts(run=run, routing=routing)
        self.store.save_run(run, routing)
        return run

    def _select_benchmark_items(self, request: RunRequest) -> list[BenchmarkItem]:
        if self.session is None or self.session.state is not ModelState.READY:
            raise RuntimeError("load a model before starting a benchmark")
        if request.benchmark_id != FIXTURE_BENCHMARK_ID:
            raise ValueError(f"unknown benchmark {request.benchmark_id!r}")
        requested_ids = set(request.item_ids or [item.id for item in self.items])
        selected = [item for item in self.items if item.id in requested_ids]
        missing = requested_ids - {item.id for item in selected}
        if missing:
            raise ValueError(f"unknown benchmark items: {sorted(missing)}")
        if not selected:
            raise ValueError("select at least one benchmark item")
        return selected

    def _queue_job(
        self,
        *,
        kind: JobKind,
        payload: dict[str, object],
        progress_total: int,
    ) -> JobRecord:
        with self._job_guard:
            active = any(
                artifacts.record.status in {JobStatus.QUEUED, JobStatus.RUNNING}
                for artifacts in self.jobs.values()
            )
            if active:
                raise RuntimeError("another model or benchmark job is already active")
            job = JobRecord(
                id=str(uuid4()),
                kind=kind,
                status=JobStatus.QUEUED,
                progress_total=progress_total,
            )
            self.jobs[job.id] = JobArtifacts(record=job, payload=payload)
            self.store.save_job(job, payload)
            return job.model_copy(deep=True)

    def _update_job_progress(self, job_id: str, current: int, total: int) -> None:
        with self._job_guard:
            artifacts = self.jobs[job_id]
            artifacts.record.progress_current = current
            artifacts.record.progress_total = total
            self.store.save_job(artifacts.record, artifacts.payload)

    def list_runs(self) -> list[BenchmarkRun]:
        return sorted(
            (artifacts.run for artifacts in self.runs.values()),
            key=lambda run: run.created_at,
            reverse=True,
        )

    def validate_profile(self, profile) -> ProfileValidation:
        return validate_profile(profile, self.topology)

    def propose_profile(
        self, request: ProfileProposalRequest
    ) -> ProfileProposal:
        artifacts = self.runs.get(request.run_id)
        if artifacts is None:
            raise KeyError(request.run_id)
        return propose_fixed_budget_profile(
            artifacts.routing,
            self.topology,
            request.keep_per_layer,
            request.metric,
        )


def _fixture_items() -> list[BenchmarkItem]:
    expressions = [
        ("03", "12 + 7", "19", "addition"),
        ("04", "42 - 19", "23", "subtraction"),
        ("05", "9 * 8", "72", "multiplication"),
        ("06", "31 + 46", "77", "addition"),
        ("07", "100 - 37", "63", "subtraction"),
        ("08", "13 * 6", "78", "multiplication"),
        ("09", "-4 + 15", "11", "addition"),
        ("10", "81 - 99", "-18", "subtraction"),
        ("11", "17 * 5", "85", "multiplication"),
        ("12", "128 + 64", "192", "addition"),
        ("13", "72 - 18", "54", "subtraction"),
        ("14", "21 * 4", "84", "multiplication"),
    ]
    return [
        BenchmarkItem(
            id=f"arith-{item_id}",
            prompt=f"Return only the integer result of {expression}.",
            expected=expected,
            category=category,
        )
        for item_id, expression, expected, category in expressions
    ]
