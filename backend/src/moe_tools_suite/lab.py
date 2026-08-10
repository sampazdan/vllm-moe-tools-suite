from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

import numpy as np

from . import __version__
from .agentic.controller import AgenticController
from .agentic.domain import CreateAgentRunRequest
from .benchmarks import BenchmarkAdapter, BenchmarkCatalog
from .domain import (
    BenchmarkCohort,
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkItemPage,
    BenchmarkRun,
    ComparisonRecord,
    CreateComparisonRequest,
    CreateCustomBenchmarkRequest,
    CreateExpertProfileRequest,
    CreateModelSessionRequest,
    ExpertProfile,
    GenerationConfig,
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
    RunDetail,
    RunItemResult,
    RunProvenance,
    RunRequest,
    RuntimeStatus,
    SavedExpertProfile,
    ScoringMode,
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
from .telemetry import aggregate_routing

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
        self.store.reconcile_interrupted_sessions()
        self.store.reconcile_interrupted_jobs()
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
        self._job_guard = Lock()
        self.runtime: ModelRuntime = self._create_runtime()
        self.server = self._create_server()
        self.agentic = AgenticController(
            settings=settings,
            store=self.store,
            runtime=self.runtime,
            topology=self.topology,
            model_id=MODEL_ID,
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
        await self.agentic.cleanup_interrupted_sandboxes()

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

    def import_custom_benchmark(
        self, request: CreateCustomBenchmarkRequest
    ) -> BenchmarkDatasetRecord:
        record = self.benchmark_catalog.import_custom(request)
        self.datasets[record.id] = record
        self.store.save_benchmark_dataset(record)
        return record

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

        matched_profile = (
            self._find_saved_profile(request.model_id, request.profile)
            if request.profile is not None
            else None
        )
        self.session = ModelSession(
            id=str(uuid4()),
            model_id=request.model_id,
            state=ModelState.STARTING,
            mode=self.settings.mode,
            profile=request.profile,
            profile_id=request.profile_id
            or (matched_profile.id if matched_profile is not None else None),
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
        adapter, selected, generation = self._select_benchmark_items(request)
        cohort = self._create_cohort(adapter, selected, generation)
        normalized_request = request.model_copy(update={"generation": generation})
        return self._queue_job(
            kind=JobKind.BENCHMARK_RUN,
            payload={
                "request": normalized_request.model_dump(mode="json"),
                "cohort_id": cohort.id,
            },
            progress_total=len(selected),
        )

    def submit_prepare_benchmark(self, benchmark_id: str) -> JobRecord:
        self.benchmark_catalog.get_info(benchmark_id)
        return self._queue_job(
            kind=JobKind.DATASET_PREPARE,
            payload={"benchmark_id": benchmark_id},
            progress_total=1,
        )

    def submit_agent_run(self, request: CreateAgentRunRequest) -> JobRecord:
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
                result = await self.create_model_session(model_request)
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
        except Exception as error:
            with self._job_guard:
                if job.status is JobStatus.CANCELLED:
                    return
                logger.exception("job %s failed", job_id)
                job.status = JobStatus.FAILED
                job.error = str(error) or error.__class__.__name__
                job.completed_at = datetime.now(UTC)
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

    def list_jobs(self) -> list[JobRecord]:
        return sorted(
            (artifacts.record for artifacts in self.jobs.values()),
            key=lambda job: job.created_at,
            reverse=True,
        )

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
            if job.kind is JobKind.AGENT_RUN and job.status is JobStatus.RUNNING:
                job.status = JobStatus.CANCELLING
                self.agentic.request_cancel(str(artifacts.payload["run_id"]))
            else:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(UTC)
                if job.kind is JobKind.AGENT_RUN:
                    self.agentic.cancel_queued_run(str(artifacts.payload["run_id"]))
            self.store.save_job(job, artifacts.payload)
            return job.model_copy(deep=True)

    async def shutdown(self) -> None:
        if self.session is not None and self.session.state is ModelState.READY:
            self.session.state = ModelState.STOPPING
            self.store.save_model_session(self.session)
        try:
            if self.server is not None:
                await self.server.aclose()
        finally:
            try:
                await self.agentic.aclose()
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
        if request.profile_id is not None:
            saved_profile = self.profiles.get(request.profile_id)
            if saved_profile is None:
                raise ValueError("profile_id does not reference a saved profile")
            if saved_profile.model_id != request.model_id:
                raise ValueError("saved profile belongs to a different model")
            if saved_profile.profile != request.profile:
                raise ValueError("profile does not match saved profile_id")

    async def run_benchmark(
        self,
        request: RunRequest,
        *,
        job_id: str | None = None,
        cohort_id: str | None = None,
    ) -> BenchmarkRun:
        adapter, selected, generation = self._select_benchmark_items(request)
        expected_cohort = self._create_cohort(adapter, selected, generation)
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
        for item_number, item in enumerate(selected, start=1):
            if job_id is not None and self._job_cancelled(job_id):
                break
            prompt = adapter.render_prompt(item)
            started = time.perf_counter()
            try:
                completion = await self.runtime.complete(
                    prompt,
                    request_key=item.id,
                    profile=session.profile,
                    generation=generation,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                aggregate = aggregate_routing(completion.routing, self.topology)
                counts += aggregate.selection_counts
                mass += aggregate.routing_mass
                total_slots += aggregate.total_routed_slots
                output = completion.content.strip()
                passed = adapter.score(item, output)
                result = RunItemResult(
                    item_id=item.id,
                    prompt=prompt,
                    expected=item.expected,
                    output=output,
                    passed=passed,
                    scoring=item.scoring,
                    latency_ms=latency_ms,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                )
            except Exception as error:
                logger.exception("benchmark item %s failed in run %s", item.id, run_id)
                result = RunItemResult(
                    item_id=item.id,
                    prompt=prompt,
                    expected=item.expected,
                    output="",
                    passed=(None if item.scoring is ScoringMode.UNGRADED else False),
                    scoring=item.scoring,
                    error=(str(error) or error.__class__.__name__)[:1000],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            results.append(result)
            if job_id is not None:
                self._update_job_progress(job_id, item_number, len(selected))

        scored = [item for item in results if item.passed is not None]
        passed = sum(item.passed is True for item in scored)
        cancelled = job_id is not None and self._job_cancelled(job_id)
        run = BenchmarkRun(
            id=run_id,
            benchmark_id=request.benchmark_id,
            model_session_id=session.id,
            status="cancelled" if cancelled else "completed",
            score=passed / len(scored) if scored else None,
            scored_items=len(scored),
            completed_items=len(results),
            total_items=len(selected),
            items=results,
            cohort_id=cohort.id,
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
            app_version=__version__,
        )
        self.runs[run_id] = RunArtifacts(
            run=run,
            routing=routing,
            provenance=provenance,
        )
        self.store.save_run(run, routing, provenance)
        return run

    def _select_benchmark_items(
        self, request: RunRequest
    ) -> tuple[BenchmarkAdapter, list[BenchmarkItem], GenerationConfig]:
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
        return adapter, selected, generation

    def _create_cohort(
        self,
        adapter: BenchmarkAdapter,
        selected: list[BenchmarkItem],
        generation: GenerationConfig,
    ) -> BenchmarkCohort:
        item_ids = [item.id for item in selected]
        fingerprint = _cohort_fingerprint(
            adapter,
            item_ids,
            generation,
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
            active = any(
                artifacts.record.status
                in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING}
                for artifacts in self.jobs.values()
            )
            if active:
                raise RuntimeError("another job is already active")
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

    def list_runs(self) -> list[BenchmarkRun]:
        return sorted(
            (artifacts.run for artifacts in self.runs.values()),
            key=lambda run: run.created_at,
            reverse=True,
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
            run=artifacts.run,
            model_session=model_session,
            saved_profile=saved_profile,
            cohort=(
                self.cohorts.get(artifacts.run.cohort_id)
                if artifacts.run.cohort_id is not None
                else None
            ),
            provenance=artifacts.provenance,
        )

    def create_expert_profile(
        self, request: CreateExpertProfileRequest
    ) -> SavedExpertProfile:
        if request.model_id != MODEL_ID:
            raise ValueError(f"unsupported model {request.model_id!r}")
        validation = validate_profile(request.profile, self.topology)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

        source_run = None
        source_trial = None
        source_routing = None
        if request.source_run_id is not None:
            source_run = self.runs.get(request.source_run_id)
            if source_run is None:
                raise KeyError(request.source_run_id)
            source_session = self.model_sessions.get(source_run.run.model_session_id)
            if source_session is None or source_session.model_id != request.model_id:
                raise ValueError("source run belongs to a different model")
            source_routing = source_run.routing
        if request.source_trial_id is not None:
            source_trial = self.agentic.trials.get(request.source_trial_id)
            if source_trial is None:
                raise KeyError(request.source_trial_id)
            source_session = self.model_sessions.get(source_trial.model_session_id)
            if source_session is None or source_session.model_id != request.model_id:
                raise ValueError("source trial belongs to a different model")
            try:
                source_routing = self.agentic.get_routing(source_trial.id)
            except KeyError as error:
                raise ValueError("source trial has no routing telemetry") from error

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
            parent_profile_id=request.parent_profile_id,
            metric=request.metric,
            validation=validation,
            observed_mass_retained=(
                calculate_observed_mass_retained(source_routing, request.profile)
                if source_routing is not None
                else None
            ),
        )
        self.profiles[saved.id] = saved
        self.store.save_expert_profile(saved)
        return saved

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
        if (
            baseline.provenance is not None
            and candidate.provenance is not None
            and baseline.provenance.cohort_fingerprint
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


def _cohort_fingerprint(
    adapter: BenchmarkAdapter,
    item_ids: list[str],
    generation: GenerationConfig,
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
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()
