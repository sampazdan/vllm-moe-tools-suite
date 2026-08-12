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
from .agentic.task_packs import load_external_task_pack_registry
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
    JobPhaseRecord,
    JobRecord,
    JobStatus,
    JudgeEvaluationRequest,
    JudgeMode,
    LLMJudgeResult,
    ModelLoadPhase,
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
from .experiments import (
    ExperimentAdapterRequest,
    ExperimentAdapterResult,
    ExperimentCancellationRequested,
    ExperimentService,
)
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
from .model_registry import QUALIFIED_MODEL_ID, model_registry
from .persistence import SqliteStore
from .process_manager import ManagedVllmServer
from .profiles import (
    EXPERT_PROFILE_FINGERPRINT_VERSION,
    calculate_observed_mass_retained,
    canonical_profile_layer_map,
    expert_profile_fingerprint,
    legacy_profile_fingerprint,
    propose_fixed_budget_profile,
    validate_profile,
)
from .runtime import MockModelRuntime, ModelRuntime, VllmRuntime
from .settings import Settings
from .telemetry import AggregatedRouting, aggregate_routing
from .v2_domain import (
    ContextActivationResult,
    CreateExperimentRequest,
    EvaluationResultRecord,
    Experiment,
    ExperimentDetail,
    ExperimentStatus,
    InterventionContextRef,
    InterventionKind,
    PerformanceSnapshot,
    RunEventPage,
    UpdateProfileMetadataRequest,
    WorkloadDescriptor,
    WorkloadKind,
    canonical_fingerprint,
)

MODEL_ID = QUALIFIED_MODEL_ID
FIXTURE_BENCHMARK_ID = "fixture-arithmetic"
logger = logging.getLogger(__name__)


class _ModelLoadCleanupError(RuntimeError):
    """A failed model load could not prove its managed process was stopped."""


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
        self.models = model_registry()
        self._active_model = next(
            model for model in self.models if model.id == MODEL_ID
        )
        assert self._active_model.topology is not None
        self.topology = self._active_model.topology.model_copy(deep=True)
        self.session: ModelSession | None = None
        self.store = SqliteStore(settings.data_dir)
        self.store.reconcile_interrupted_state()
        self.store.reconcile_interrupted_experiments()
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
        self.intervention_contexts = self.store.load_intervention_contexts()
        self.active_context: InterventionContextRef | None = None
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
            model_id=self._active_model.id,
            judge_service=self.judges,
        )
        self.experiments = ExperimentService(
            store=self.store,
            runtime=self.runtime,
            topology=self.topology,
            model_id=self._active_model.id,
            mode=self.settings.mode,
            profiles=self.profiles,
            activate_context=lambda profile_id: self.activate_expert_context(
                profile_id, internal=True
            ),
            context_builder=self._context_reference,
            current_context=self.current_intervention_context,
            current_session=self.current_model_session,
            answer_executor=self._execute_v2_answer_unit,
            coding_executor=self._execute_v2_coding_unit,
        )

    def _create_runtime(self, model: ModelRegistryEntry | None = None) -> ModelRuntime:
        selected = model or self._active_model
        if selected.topology is None:
            raise RuntimeError("enabled model is missing its manifest topology")
        if self.settings.mode == "vllm":
            return VllmRuntime(
                base_url=self.settings.vllm_base_url,
                model_id=selected.id,
                topology=selected.topology,
            )
        return MockModelRuntime(selected.topology)

    def _create_server(self) -> ManagedVllmServer | None:
        if self.settings.mode != "vllm":
            return None
        return ManagedVllmServer(
            command=self.settings.vllm_command,
            base_url=self.settings.vllm_base_url,
            model_id=self._active_model.id,
            data_dir=self.settings.data_dir,
            startup_timeout_seconds=self.settings.vllm_startup_timeout_seconds,
            shutdown_timeout_seconds=self.settings.vllm_shutdown_timeout_seconds,
            poll_interval_seconds=self.settings.vllm_poll_interval_seconds,
        )

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        return self.benchmark_catalog.list_benchmarks()

    def list_workloads(self) -> list[WorkloadDescriptor]:
        descriptors: list[WorkloadDescriptor] = []
        for benchmark in self.benchmark_catalog.list_benchmarks():
            blocked_reason = None
            try:
                adapter = self.benchmark_catalog.get_adapter(benchmark.id)
                items = adapter.items()
                content_fingerprint = adapter.content_hash
            except RuntimeError as error:
                items = []
                content_fingerprint = canonical_fingerprint(
                    benchmark.model_dump(mode="json")
                )
                blocked_reason = str(error)
            for item in items:
                descriptors.append(
                    WorkloadDescriptor(
                        id=f"answer:{benchmark.id}:{item.id}",
                        kind=WorkloadKind.ANSWER,
                        name=f"{benchmark.name} · {item.id}",
                        description=item.prompt,
                        source=benchmark.source,
                        revision=benchmark.revision,
                        content_fingerprint=canonical_fingerprint(
                            {
                                "dataset": content_fingerprint,
                                "item": item.model_dump(mode="json"),
                            }
                        ),
                        ready=benchmark.ready,
                        blocked_reason=blocked_reason,
                        unit_ids=[item.id],
                        parent_id=benchmark.id,
                        task_id=item.id,
                        difficulty=str(item.metadata.get("difficulty", "unspecified")),
                        expected_horizon=1,
                        tools=[],
                        runtime="model completion + deterministic scorer",
                        preparation_status=(
                            "validated" if benchmark.ready else "not_prepared"
                        ),
                        public_problem_statement=item.prompt,
                        public_success_criteria=[
                            f"Pass the benchmark's {benchmark.scoring.value} scorer."
                        ],
                        metadata={
                            "category": item.category,
                            "benchmark_name": benchmark.name,
                            "license": benchmark.license,
                        },
                    )
                )
        packs = {pack.id: pack for pack in self.agentic.list_task_packs()}
        for pack_id, pack in packs.items():
            for task in self.agentic.list_tasks(pack_id):
                ready = pack.ready and pack.oracle_passed and pack.noop_failed
                descriptors.append(
                    WorkloadDescriptor(
                        id=f"coding:{pack_id}:{task.id}",
                        kind=WorkloadKind.CODING,
                        name=task.title,
                        description=task.instruction,
                        source=pack.source,
                        revision=pack.revision,
                        content_fingerprint=canonical_fingerprint(
                            {
                                "pack": pack.fingerprint,
                                "task_id": task.id,
                                "verifier": task.verifier_fingerprint(),
                            }
                        ),
                        ready=ready,
                        blocked_reason=(
                            None
                            if ready
                            else "task pack has not passed oracle/no-op eligibility"
                        ),
                        unit_ids=[task.id],
                        parent_id=pack_id,
                        task_id=task.id,
                        language=task.language,
                        difficulty=_task_difficulty(task.tags),
                        expected_horizon=_task_expected_horizon(task.tags),
                        tools=["bash"],
                        runtime=(
                            f"{task.image_ref}@{task.image_digest}; "
                            f"{task.timeout_seconds:g}s timeout"
                        ),
                        preparation_status=("validated" if ready else "not_eligible"),
                        public_problem_statement=task.instruction,
                        public_success_criteria=list(task.success_criteria),
                        metadata={
                            "category": task.category,
                            "network_policy": task.network_policy.value,
                            "tags": task.tags,
                        },
                    )
                )
        registry = load_external_task_pack_registry()
        bundled_pack_ids = set(packs)
        for pack in registry.task_packs:
            if pack.id in bundled_pack_ids or pack.launchable:
                continue
            task_ids = pack.curated_task_ids or [f"{pack.id}:unselected-cohort"]
            blocker = "; ".join(pack.limitations)
            for task_id in task_ids:
                descriptors.append(
                    WorkloadDescriptor(
                        id=f"coding:{pack.id}:{task_id}",
                        kind=WorkloadKind.CODING,
                        name=f"{pack.name} · {task_id}",
                        description=pack.curated_selection,
                        source=str(pack.source_url),
                        revision=pack.source_revision,
                        content_fingerprint=canonical_fingerprint(
                            {
                                "registry": registry.content_hash,
                                "pack": pack.id,
                                "task": task_id,
                            }
                        ),
                        ready=False,
                        blocked_reason=blocker,
                        unit_ids=[task_id],
                        parent_id=pack.id,
                        task_id=task_id,
                        language=_external_language(task_id),
                        difficulty="unqualified",
                        expected_horizon=None,
                        tools=["bash"],
                        runtime=pack.distribution,
                        preparation_status=pack.preparation_state.value,
                        public_problem_statement=(
                            "Public task content becomes available only after the "
                            "pinned assets are prepared."
                        ),
                        public_success_criteria=[pack.success_criteria],
                        metadata={
                            "family": pack.family,
                            "license": pack.license,
                            "assets_prepared": pack.assets_prepared,
                            "oracle_passed": pack.oracle_passed,
                            "noop_failed": pack.noop_failed,
                        },
                    )
                )
        descriptors.append(self.experiments.state_drift_descriptor())
        return sorted(descriptors, key=lambda item: (item.kind.value, item.name))

    def get_workload_descriptor(self, workload_id: str) -> WorkloadDescriptor:
        descriptor = next(
            (item for item in self.list_workloads() if item.id == workload_id),
            None,
        )
        if descriptor is None:
            raise KeyError(workload_id)
        return descriptor

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
        job_id: str | None = None,
    ) -> ModelSession:
        self._set_model_load_phase(
            job_id,
            ModelLoadPhase.RESOLVING_MODEL,
            detail="Resolving the enabled pinned model manifest",
        )
        request = self._resolve_model_session_request(request)
        assert request.model_id is not None
        model = self._resolve_manifest_model(request.model_id)
        matched_profile = (
            self._find_saved_profile(request.model_id, request.profile)
            if request.profile is not None
            else None
        )
        profile_id = request.profile_id or (
            matched_profile.id if matched_profile is not None else None
        )
        load_session = (
            self.model_sessions.get(session_id) if session_id is not None else None
        )
        if load_session is not None:
            if (
                load_session.model_id != request.model_id
                or load_session.profile != request.profile
                or load_session.profile_id != profile_id
                or load_session.model_revision != model.revision
                or load_session.topology != model.topology
                or load_session.runtime_recipe != model.runtime_recipe
            ):
                raise RuntimeError("model-load job does not match its planned session")
        else:
            load_session = ModelSession(
                id=session_id or str(uuid4()),
                model_id=request.model_id,
                state=ModelState.STARTING,
                mode=self.settings.mode,
                profile=request.profile,
                profile_id=profile_id,
                model_revision=model.revision,
                topology=model.topology,
                runtime_recipe=model.runtime_recipe,
            )
            self.model_sessions[load_session.id] = load_session
            self.store.save_model_session(load_session)
        previous_session = self.session
        try:
            if previous_session is not None:
                self._set_model_load_phase(
                    job_id,
                    ModelLoadPhase.STOPPING_PREVIOUS,
                    detail=f"Stopping model session {previous_session.id}",
                )
                previous_session.state = ModelState.STOPPING
                self.store.save_model_session(previous_session)
                if self.server is not None:
                    await self.server.stop()
                previous_session.state = ModelState.STOPPED
                self.store.save_model_session(previous_session)
                self.active_context = None
                self.runtime.set_active_context(None)
                self.session = None

            self._set_model_load_phase(
                job_id,
                ModelLoadPhase.CONFIGURING_RUNTIME,
                detail=(
                    f"Configuring application runtime for {model.id} "
                    f"at {model.revision}"
                ),
            )
            await self._configure_runtime_for_model(model)

            load_session.state = ModelState.STARTING
            load_session.model_revision = model.revision
            load_session.topology = model.topology
            load_session.runtime_recipe = model.runtime_recipe
            self.session = load_session
            self.store.save_model_session(load_session)
            if self.server is not None:
                await self.server.start(
                    profile=request.profile,
                    session_id=load_session.id,
                    model_id=model.id,
                    model_revision=model.revision,
                    runtime_recipe=model.runtime_recipe,
                    on_phase=lambda phase, detail: self._set_model_load_phase(
                        job_id,
                        phase,
                        detail=detail,
                    ),
                )
                if self.settings.mode == "vllm":
                    self._set_model_load_phase(
                        job_id,
                        ModelLoadPhase.VERIFYING_READY_CONTEXT,
                        detail=(
                            "Verifying model identity and active expert-context "
                            "provenance"
                        ),
                    )
                    context = await self._synchronize_server_context(
                        profile_id,
                        profile=request.profile,
                    )
            if self.settings.mode != "vllm":
                saved_profile = (
                    self.profiles.get(profile_id) if profile_id is not None else None
                )
                context = self._context_reference(saved_profile)
            self.active_context = context
            self.intervention_contexts[context.context_id] = context
            self.store.save_intervention_context(context)
            self.runtime.set_active_context(context)
            load_session.state = ModelState.READY
            self.store.save_model_session(load_session)
            self._set_model_load_phase(
                job_id,
                ModelLoadPhase.READY,
                detail="Model runtime is ready",
                terminal=True,
            )
            return load_session
        except BaseException as error:
            cleanup_error = await self._stop_incomplete_model_load()
            cancelled = (
                isinstance(error, asyncio.CancelledError) and cleanup_error is None
            )
            load_session.state = ModelState.STOPPED if cancelled else ModelState.FAILED
            self.store.save_model_session(load_session)
            if previous_session is not None and (
                previous_session.state is ModelState.STOPPING
            ):
                previous_session.state = (
                    ModelState.STOPPED if cleanup_error is None else ModelState.FAILED
                )
                self.store.save_model_session(previous_session)
            self.active_context = None
            self.runtime.set_active_context(None)
            self.session = None if cancelled else load_session
            if cleanup_error is not None:
                raise _ModelLoadCleanupError(
                    "model load did not become ready and managed process cleanup "
                    f"failed: {cleanup_error}"
                ) from cleanup_error
            raise

    async def _stop_incomplete_model_load(self) -> BaseException | None:
        if self.server is None:
            return None
        cleanup = asyncio.create_task(self.server.stop())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:
                break
        if cleanup.cancelled():
            return cancellation or RuntimeError("managed process cleanup was cancelled")
        return cleanup.exception()

    async def _synchronize_server_context(
        self,
        profile_id: str | None,
        *,
        profile: ExpertProfile | None,
    ) -> InterventionContextRef:
        if self.server is None:
            raise RuntimeError("managed vLLM server is unavailable")
        saved_profile = (
            self.profiles.get(profile_id) if profile_id is not None else None
        )
        if profile is not None and saved_profile is None:
            raise RuntimeError(
                "masked managed startup requires saved expert-profile provenance"
            )
        context = self._context_reference(saved_profile)
        current = await self.server.current_context()
        observed_id = current.get("active_context_id")
        observed_fingerprint = current.get("active_context_fingerprint")
        observed_profile_fingerprint = current.get("profile_fingerprint")
        observed_topology = current.get("topology_fingerprint")
        observed_process_id = current.get("process_id")
        if not isinstance(observed_id, str) or not observed_id:
            raise RuntimeError(
                "vLLM current expert context omitted its active context ID"
            )
        if not isinstance(observed_fingerprint, str) or not observed_fingerprint:
            raise RuntimeError(
                "vLLM current expert context omitted its active fingerprint"
            )
        if not isinstance(observed_topology, str) or not observed_topology:
            raise RuntimeError(
                "vLLM current expert context omitted its topology fingerprint"
            )
        if observed_process_id != self.server.pid:
            raise RuntimeError("vLLM current expert context came from another process")
        if profile is not None:
            expected_profile_fingerprint = expert_profile_fingerprint(
                profile,
                self.topology,
            )
            if not isinstance(observed_profile_fingerprint, str):
                raise RuntimeError(
                    "vLLM current expert context omitted its profile fingerprint"
                )
            if observed_profile_fingerprint != expected_profile_fingerprint:
                raise RuntimeError(
                    "vLLM current expert context uses a different canonical expert mask"
                )
        return _validated_context_update(
            context,
            context_id=observed_id,
            context_fingerprint=observed_fingerprint,
            topology_fingerprint=observed_topology,
        )

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
                model_id=(
                    current_session.model_id
                    if current_session is not None
                    else self._active_model.id
                ),
                model_revision=(
                    current_session.model_revision
                    if current_session is not None
                    else self._active_model.revision
                ),
                session_id=current_session.id if current_session else None,
            )
        return RuntimeStatus(
            managed=True,
            model_id=self.server.model_id,
            model_revision=self.server.model_revision,
            pid=self.server.pid,
            session_id=current_session.id if current_session else None,
            started_at=self.server.started_at,
            log_path=str(self.server.log_path) if self.server.log_path else None,
            profile_path=(
                str(self.server.profile_path) if self.server.profile_path else None
            ),
            log_tail=self.server.read_log_tail(),
        )

    def current_intervention_context(self) -> InterventionContextRef | None:
        return (
            self.active_context.model_copy(deep=True)
            if self.active_context is not None
            else None
        )

    async def activate_expert_context(
        self,
        profile_id: str | None,
        *,
        internal: bool = False,
    ) -> ContextActivationResult:
        if self.session is None or self.session.state is not ModelState.READY:
            raise RuntimeError("load a model before activating an expert context")
        model = self._resolve_manifest_model(self.session.model_id)
        if model.masking_status == "unsupported":
            raise RuntimeError("the ready model does not support expert contexts")
        if not internal:
            with self._job_guard:
                active = self._active_job_artifacts_unlocked()
                if active is not None:
                    raise ActiveJobConflict(active.record)
        profile = None
        if profile_id is not None:
            profile = self.profiles.get(profile_id)
            if profile is None:
                raise KeyError(profile_id)
            if profile.model_id != self.session.model_id:
                raise ValueError("expert profile belongs to a different model")

        previous = self.active_context
        target = self._context_reference(profile)
        if previous is not None and previous.context_fingerprint == (
            target.context_fingerprint
        ):
            return ContextActivationResult(
                old_context_id=previous.context_id,
                old_context_fingerprint=previous.context_fingerprint,
                new_context_id=previous.context_id,
                new_context_fingerprint=previous.context_fingerprint,
                topology_fingerprint=previous.topology_fingerprint,
                duration_ms=0,
                process_id=self.server.pid if self.server is not None else None,
                weights_reloaded=False,
            )

        if self.server is None:
            receipt = ContextActivationResult(
                old_context_id=previous.context_id if previous else None,
                old_context_fingerprint=(
                    previous.context_fingerprint if previous else None
                ),
                new_context_id=target.context_id,
                new_context_fingerprint=target.context_fingerprint,
                topology_fingerprint=target.topology_fingerprint,
                duration_ms=0,
                process_id=None,
                weights_reloaded=False,
            )
        else:
            capabilities = await self.server.context_capabilities()
            if capabilities.get("supported") is not True:
                reason = capabilities.get("unsupported_reason") or "unknown reason"
                raise RuntimeError(f"hot expert contexts are unsupported: {reason}")
            discovered_topology = capabilities.get("topology_fingerprint")
            if not isinstance(discovered_topology, str) or not discovered_topology:
                raise RuntimeError(
                    "vLLM expert-context capability response omitted topology identity"
                )
            target = _validated_context_update(
                target,
                topology_fingerprint=discovered_topology,
            )
            if profile is None:
                activation_started = time.perf_counter()
                try:
                    receipt = await self.server.reset_context()
                except RuntimeError as error:
                    target, receipt = await self._reconcile_activation_error(
                        previous=previous,
                        target=target,
                        error=error,
                        activation_started=activation_started,
                    )
            else:
                assert model.topology is not None
                expected_layers = canonical_profile_layer_map(
                    profile.profile,
                    model.topology,
                )
                expected_profile_fingerprint = expert_profile_fingerprint(
                    profile.profile,
                    model.topology,
                )
                registered = await self.server.register_context(
                    context_id=target.context_id,
                    layers=expected_layers,
                    creation_source=target.creation_source,
                    metadata={
                        "profile_id": profile.id,
                        "profile_fingerprint": expected_profile_fingerprint,
                        "saved_profile_fingerprint": profile.profile_fingerprint,
                        "saved_profile_fingerprint_version": (
                            profile.profile_fingerprint_version
                        ),
                    },
                )
                registration_error = _registration_identity_error(
                    registered,
                    expected_context_id=target.context_id,
                    expected_profile_fingerprint=expected_profile_fingerprint,
                    expected_topology_fingerprint=discovered_topology,
                    expected_layers=expected_layers,
                )
                if registration_error is not None:
                    raise registration_error
                registered_fingerprint = registered.get("context_fingerprint")
                if not isinstance(registered_fingerprint, str):
                    raise RuntimeError(
                        "vLLM expert-context registration omitted its fingerprint"
                    )
                target = _validated_context_update(
                    target,
                    context_fingerprint=registered_fingerprint,
                )
                activation_started = time.perf_counter()
                try:
                    receipt = await self.server.activate_context(target.context_id)
                except RuntimeError as error:
                    target, receipt = await self._reconcile_activation_error(
                        previous=previous,
                        target=target,
                        error=error,
                        activation_started=activation_started,
                    )
            receipt_error = None
            if receipt.topology_fingerprint != target.topology_fingerprint:
                receipt_error = RuntimeError(
                    "vLLM activated an expert context for a different topology"
                )
            elif profile is None and receipt.new_context_id != "baseline":
                receipt_error = RuntimeError(
                    "vLLM reset activated a non-baseline expert context"
                )
            elif profile is not None and (
                receipt.new_context_id != target.context_id
                or receipt.new_context_fingerprint != target.context_fingerprint
            ):
                receipt_error = RuntimeError(
                    "vLLM activated a different expert context"
                )
            if receipt_error is not None:
                target, receipt = await self._reconcile_activation_error(
                    previous=previous,
                    target=target,
                    error=receipt_error,
                    activation_started=activation_started,
                )
            elif profile is None:
                target = _validated_context_update(
                    target,
                    context_id=receipt.new_context_id,
                    context_fingerprint=receipt.new_context_fingerprint,
                )

        self.active_context = target
        self.intervention_contexts[target.context_id] = target
        self.store.save_intervention_context(target)
        self.runtime.set_active_context(target)
        return receipt

    async def _reconcile_activation_error(
        self,
        *,
        previous: InterventionContextRef | None,
        target: InterventionContextRef,
        error: RuntimeError,
        activation_started: float,
    ) -> tuple[InterventionContextRef, ContextActivationResult]:
        if self.server is None:
            raise error
        try:
            current = await self.server.current_context()
            observed_id = current.get("active_context_id")
            observed_fingerprint = current.get("active_context_fingerprint")
            observed_topology = current.get("topology_fingerprint")
            process_id = current.get("process_id")
            if not all(
                isinstance(value, str) and value
                for value in (
                    observed_id,
                    observed_fingerprint,
                    observed_topology,
                )
            ):
                raise RuntimeError(
                    "vLLM reconciliation response omitted active context identity"
                )
        except Exception as reconciliation_error:
            self._disable_ambiguous_runtime_context()
            raise RuntimeError(
                "expert-context activation outcome is ambiguous and current "
                "context reconciliation failed; the model session was disabled"
            ) from reconciliation_error

        assert isinstance(observed_id, str)
        assert isinstance(observed_fingerprint, str)
        assert isinstance(observed_topology, str)
        serving_safe = current.get("serving_safe")
        recovery_required = current.get("recovery_required")
        if serving_safe is not True or recovery_required is not False:
            try:
                recovery = (
                    await self.server.reset_context()
                    if observed_id == "baseline"
                    else await self.server.activate_context(observed_id)
                )
            except Exception as recovery_error:
                self._disable_ambiguous_runtime_context()
                raise RuntimeError(
                    "expert-context activation committed but serving recovery "
                    "failed; the model session was disabled"
                ) from recovery_error
            if (
                recovery.new_context_id != observed_id
                or recovery.new_context_fingerprint != observed_fingerprint
                or recovery.topology_fingerprint != observed_topology
            ):
                self._disable_ambiguous_runtime_context()
                raise RuntimeError(
                    "expert-context recovery changed the committed context; "
                    "the model session was disabled"
                ) from error
        if process_id != self.server.pid:
            self._disable_ambiguous_runtime_context()
            raise RuntimeError(
                "expert-context activation reconciled to another process; the "
                "model session was disabled"
            ) from error
        if observed_topology != target.topology_fingerprint:
            self._disable_ambiguous_runtime_context()
            raise RuntimeError(
                "expert-context activation reconciled to a different topology; "
                "the model session was disabled"
            ) from error
        target_committed = observed_fingerprint == target.context_fingerprint or (
            target.kind is InterventionKind.BASELINE and observed_id == "baseline"
        )
        if target_committed:
            reconciled = _validated_context_update(
                target,
                context_id=observed_id,
                context_fingerprint=observed_fingerprint,
                topology_fingerprint=observed_topology,
            )
            return reconciled, ContextActivationResult(
                old_context_id=previous.context_id if previous else None,
                old_context_fingerprint=(
                    previous.context_fingerprint if previous else None
                ),
                new_context_id=observed_id,
                new_context_fingerprint=observed_fingerprint,
                topology_fingerprint=observed_topology,
                duration_ms=(time.perf_counter() - activation_started) * 1000,
                process_id=process_id if isinstance(process_id, int) else None,
                weights_reloaded=False,
            )
        if previous is not None and (
            observed_fingerprint == previous.context_fingerprint
        ):
            raise error
        known = next(
            (
                context
                for context in self.intervention_contexts.values()
                if context.context_fingerprint == observed_fingerprint
            ),
            None,
        )
        if known is not None:
            self.active_context = known
            self.runtime.set_active_context(known)
            raise RuntimeError(
                "expert-context activation failed; vLLM reconciled to a known "
                f"context {known.context_id!r}"
            ) from error
        self._disable_ambiguous_runtime_context()
        raise RuntimeError(
            "expert-context activation reached an unknown context; the model "
            "session was disabled"
        ) from error

    def _disable_ambiguous_runtime_context(self) -> None:
        self.active_context = None
        self.runtime.set_active_context(None)
        if self.session is not None:
            self.session.state = ModelState.FAILED
            self.store.save_model_session(self.session)

    def _context_reference(
        self, profile: SavedExpertProfile | None
    ) -> InterventionContextRef:
        model_id = profile.model_id if profile is not None else self._active_model.id
        model = self._resolve_manifest_model(model_id)
        topology = model.topology
        if topology is None:
            raise RuntimeError("enabled model is missing its manifest topology")
        topology_fingerprint = canonical_fingerprint(topology)
        if profile is None:
            identity = {
                "model_id": model.id,
                "model_revision": model.revision,
                "topology_fingerprint": topology_fingerprint,
                "kind": InterventionKind.BASELINE.value,
            }
            fingerprint = canonical_fingerprint(identity)
            return InterventionContextRef(
                context_id="baseline",
                kind=InterventionKind.BASELINE,
                model_id=model.id,
                model_revision=model.revision,
                topology_fingerprint=topology_fingerprint,
                context_fingerprint=fingerprint,
                creation_source="model-baseline",
            )
        runtime_profile_fingerprint = expert_profile_fingerprint(
            profile.profile,
            topology,
        )
        identity = {
            "model_id": profile.model_id,
            "model_revision": model.revision,
            "topology_fingerprint": topology_fingerprint,
            "profile_fingerprint": runtime_profile_fingerprint,
            "kind": InterventionKind.EXPERT_MASK.value,
        }
        fingerprint = canonical_fingerprint(identity)
        return InterventionContextRef(
            context_id=f"expert-mask:{fingerprint[:24]}",
            kind=InterventionKind.EXPERT_MASK,
            model_id=profile.model_id,
            model_revision=model.revision,
            topology_fingerprint=topology_fingerprint,
            profile_id=profile.id,
            profile_fingerprint=runtime_profile_fingerprint,
            context_fingerprint=fingerprint,
            creation_source="saved-expert-profile",
            metadata={
                "profile_name": profile.name,
                "saved_profile_fingerprint": profile.profile_fingerprint,
                "saved_profile_fingerprint_version": (
                    profile.profile_fingerprint_version
                ),
            },
        )

    def submit_model_session(self, request: CreateModelSessionRequest) -> JobRecord:
        request = self._resolve_model_session_request(request)
        assert request.model_id is not None
        model = self._resolve_manifest_model(request.model_id)
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
                model_revision=model.revision,
                topology=model.topology,
                runtime_recipe=model.runtime_recipe,
            )
            job = JobRecord(
                id=str(uuid4()),
                kind=JobKind.MODEL_LOAD,
                status=JobStatus.QUEUED,
                progress_total=1,
                result_id=model_session.id,
                phase_history=[
                    JobPhaseRecord(
                        phase=ModelLoadPhase.QUEUED,
                        detail="Model-load request persisted and queued",
                    )
                ],
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

    def submit_experiment(self, request: CreateExperimentRequest) -> Experiment:
        self._ensure_no_active_job()
        descriptor = self.get_workload_descriptor(request.workload_id)
        self.experiments.validate_request(request, descriptor)
        experiment_id = str(uuid4())
        job_id = str(uuid4())
        dimensions = (
            len(request.scenario_ids)
            * len(request.conditions)
            * len(request.horizons)
            * len(request.seeds)
            if descriptor.kind is WorkloadKind.STATE_DRIFT
            else len(request.seeds)
        )
        lane_count = 2 if request.candidate_profile_id is not None else 1
        self._queue_job(
            kind=JobKind.EXPERIMENT_RUN,
            payload={
                "experiment_id": experiment_id,
                "request": request.model_dump(mode="json"),
            },
            progress_total=dimensions * lane_count,
            job_id=job_id,
            result_id=experiment_id,
        )
        try:
            return self.experiments.create(
                request,
                job_id=job_id,
                descriptor=descriptor,
                experiment_id=experiment_id,
            )
        except Exception:
            with self._job_guard:
                artifacts = self.jobs[job_id]
                artifacts.record.status = JobStatus.FAILED
                artifacts.record.error = "experiment graph creation failed"
                artifacts.record.completed_at = datetime.now(UTC)
                self.store.save_job(artifacts.record, artifacts.payload)
            raise

    def list_experiments(self) -> list[Experiment]:
        return self.experiments.list()

    def get_experiment(self, experiment_id: str) -> ExperimentDetail:
        return self.experiments.detail(experiment_id)

    def get_experiment_events(
        self,
        experiment_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> RunEventPage:
        return self.experiments.events(
            experiment_id,
            after=after,
            limit=limit,
        )

    def cancel_experiment(self, experiment_id: str) -> Experiment:
        experiment = self.experiments.get(experiment_id)
        self.cancel_job(experiment.job_id)
        return self.experiments.get(experiment_id)

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
                    job_id=job_id,
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
            elif job.kind is JobKind.EXPERIMENT_RUN:
                result = await self.experiments.execute(
                    str(artifacts.payload["experiment_id"]),
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
                    if job.kind is JobKind.MODEL_LOAD:
                        self._cancel_model_session_unlocked(job)
                        self._cancel_model_load_phase_unlocked(
                            job,
                            detail="Model-load execution stopped and cleanup completed",
                        )
                    job.status = JobStatus.CANCELLED
                    job.error = None
                    job.completed_at = job.completed_at or datetime.now(UTC)
                    self.store.save_job(job, artifacts.payload)
                elif job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    job.status = JobStatus.FAILED
                    job.error = "job execution was interrupted during shutdown"
                    job.completed_at = datetime.now(UTC)
                    self._fail_planned_model_session_unlocked(job)
                    self._fail_model_load_phase_unlocked(job, job.error)
                    self.store.save_job(job, artifacts.payload)
            raise
        except Exception as error:
            with self._job_guard:
                terminal_experiment = (
                    self.experiments.get(str(artifacts.payload["experiment_id"]))
                    if job.kind is JobKind.EXPERIMENT_RUN
                    else None
                )
                if terminal_experiment is not None and terminal_experiment.status in {
                    ExperimentStatus.COMPLETED,
                    ExperimentStatus.CANCELLED,
                    ExperimentStatus.FAILED,
                }:
                    job.status = (
                        JobStatus.COMPLETED
                        if terminal_experiment.status is ExperimentStatus.COMPLETED
                        else (
                            JobStatus.CANCELLED
                            if terminal_experiment.status is ExperimentStatus.CANCELLED
                            else JobStatus.FAILED
                        )
                    )
                    job.error = terminal_experiment.error
                    job.progress_current = terminal_experiment.completed_units
                    job.completed_at = terminal_experiment.completed_at or datetime.now(
                        UTC
                    )
                    self.store.save_job(job, artifacts.payload)
                    return
                if job.status in {
                    JobStatus.CANCELLING,
                    JobStatus.CANCELLED,
                } and not isinstance(error, _ModelLoadCleanupError):
                    if job.kind is JobKind.MODEL_LOAD:
                        self._cancel_model_session_unlocked(job)
                        self._cancel_model_load_phase_unlocked(
                            job,
                            detail="Model-load execution stopped and cleanup completed",
                        )
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
                self._fail_model_load_phase_unlocked(job, job.error)
                self.store.save_job(job, artifacts.payload)
            return

        with self._job_guard:
            job.result_id = result.id
            cancelling_after_result = job.status in {
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            }
            result_is_terminal = (
                job.kind is JobKind.EXPERIMENT_RUN
                and isinstance(result, Experiment)
                and result.status
                in {
                    ExperimentStatus.COMPLETED,
                    ExperimentStatus.CANCELLED,
                    ExperimentStatus.FAILED,
                }
            )
            if result_is_terminal:
                job.status = (
                    JobStatus.COMPLETED
                    if result.status is ExperimentStatus.COMPLETED
                    else (
                        JobStatus.CANCELLED
                        if result.status is ExperimentStatus.CANCELLED
                        else JobStatus.FAILED
                    )
                )
                job.error = result.error
                job.progress_current = result.completed_units
                job.completed_at = result.completed_at or datetime.now(UTC)
                self.store.save_job(job, artifacts.payload)
                return
            if not cancelling_after_result:
                job.status = JobStatus.COMPLETED
                job.progress_current = job.progress_total
                job.completed_at = datetime.now(UTC)
                self.store.save_job(job, artifacts.payload)
                return

        cleanup_error = (
            await self._stop_incomplete_model_load()
            if job.kind is JobKind.MODEL_LOAD
            else None
        )
        with self._job_guard:
            if cleanup_error is not None:
                job.status = JobStatus.FAILED
                job.error = (
                    "model load completed during cancellation but managed process "
                    f"cleanup failed: {cleanup_error}"
                )
                job.completed_at = datetime.now(UTC)
                self._fail_planned_model_session_unlocked(job)
                self._fail_model_load_phase_unlocked(job, job.error)
                self.active_context = None
                self.runtime.set_active_context(None)
            else:
                if job.kind is JobKind.MODEL_LOAD:
                    self._cancel_model_session_unlocked(job)
                    self._cancel_model_load_phase_unlocked(
                        job,
                        detail="Model-load execution stopped and cleanup completed",
                    )
                job.status = JobStatus.CANCELLED
                job.error = None
                job.completed_at = job.completed_at or datetime.now(UTC)
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
            if job.status is JobStatus.RUNNING:
                if job.kind is JobKind.EXPERIMENT_RUN:
                    experiment = self.experiments.get(
                        str(artifacts.payload["experiment_id"])
                    )
                    if experiment.status in {
                        ExperimentStatus.COMPLETED,
                        ExperimentStatus.CANCELLED,
                        ExperimentStatus.FAILED,
                    }:
                        job.status = (
                            JobStatus.COMPLETED
                            if experiment.status is ExperimentStatus.COMPLETED
                            else (
                                JobStatus.CANCELLED
                                if experiment.status is ExperimentStatus.CANCELLED
                                else JobStatus.FAILED
                            )
                        )
                        job.error = experiment.error
                        job.progress_current = experiment.completed_units
                        job.completed_at = experiment.completed_at or datetime.now(UTC)
                        self.store.save_job(job, artifacts.payload)
                        return job.model_copy(deep=True)
                if job.kind is JobKind.MODEL_LOAD:
                    task = self._job_tasks.get(job_id)
                    if task is None or task.done():
                        raise ValueError(
                            "model-load execution is not owned by this app process"
                        )
                job.status = JobStatus.CANCELLING
                if job.kind is JobKind.MODEL_LOAD:
                    self._cancel_owned_task(task)
                elif job.kind is JobKind.AGENT_RUN:
                    self.agentic.request_cancel(str(artifacts.payload["run_id"]))
                elif job.kind is JobKind.EXPERIMENT_RUN:
                    self.experiments.request_cancel(
                        str(artifacts.payload["experiment_id"])
                    )
            else:
                if job.kind is JobKind.MODEL_LOAD:
                    self._cancel_model_session_unlocked(job)
                    self._cancel_model_load_phase_unlocked(
                        job,
                        detail="Model-load request cancelled before runtime launch",
                    )
                elif job.kind is JobKind.AGENT_RUN:
                    self.agentic.cancel_queued_run(str(artifacts.payload["run_id"]))
                elif job.kind is JobKind.EXPERIMENT_RUN:
                    self.experiments.cancel_queued(
                        str(artifacts.payload["experiment_id"])
                    )
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(UTC)
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
        model = self._resolve_manifest_model(request.model_id)
        model_id = model.id
        profile = request.profile
        if request.profile_id is not None:
            saved_profile = self.profiles.get(request.profile_id)
            if saved_profile is None:
                raise ValueError("profile_id does not reference a saved profile")
            if saved_profile.model_id != model_id:
                raise ValueError("saved profile belongs to a different model")
            if profile is not None and saved_profile.profile != profile:
                raise ValueError("profile does not match saved profile_id")
            profile = saved_profile.profile.model_copy(deep=True)
        if profile is not None:
            assert model.topology is not None
            validation = validate_profile(profile, model.topology)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
        return request.model_copy(update={"model_id": model_id, "profile": profile})

    def _resolve_manifest_model(self, model_id: str | None) -> ModelRegistryEntry:
        selected_id = model_id or MODEL_ID
        model = next((item for item in self.models if item.id == selected_id), None)
        if model is None:
            raise ValueError(f"unknown model {selected_id!r}")
        if not model.enabled or model.qualification_status == "manifest_only":
            reason = model.failure_reason or "model is not enabled for live loading"
            raise ValueError(f"model {selected_id!r} is unavailable: {reason}")
        if (
            model.revision is None
            or model.topology is None
            or model.runtime_recipe is None
        ):
            raise ValueError(
                f"enabled model {selected_id!r} lacks a pinned runnable manifest"
            )
        return model

    async def _configure_runtime_for_model(self, model: ModelRegistryEntry) -> None:
        if model.topology is None:
            raise RuntimeError("enabled model is missing its manifest topology")
        if (
            self._active_model.id == model.id
            and self.topology == model.topology
            and (
                (
                    self.settings.mode == "mock"
                    and isinstance(self.runtime, MockModelRuntime)
                )
                or (
                    self.settings.mode == "vllm"
                    and isinstance(self.runtime, VllmRuntime)
                )
            )
        ):
            return
        previous_runtime = self.runtime
        self._active_model = model
        self.topology = model.topology.model_copy(deep=True)
        self.runtime = self._create_runtime(model)
        self.agentic.runtime = self.runtime
        self.agentic.topology = self.topology
        self.agentic.model_id = model.id
        self.agentic.gateway.runtime = self.runtime
        self.agentic.gateway.topology = self.topology
        self.experiments.runtime = self.runtime
        self.experiments.topology = self.topology
        self.experiments.model_id = model.id
        await previous_runtime.aclose()

    async def _execute_v2_answer_unit(
        self, request: ExperimentAdapterRequest
    ) -> ExperimentAdapterResult:
        if request.should_cancel():
            raise ExperimentCancellationRequested
        descriptor = request.experiment.workload
        if descriptor.parent_id is None or descriptor.task_id is None:
            raise RuntimeError("answer workload descriptor is missing its source item")
        adapter = self.benchmark_catalog.get_adapter(descriptor.parent_id)
        item = next(
            (item for item in adapter.items() if item.id == descriptor.task_id),
            None,
        )
        if item is None:
            raise RuntimeError("answer workload source item is no longer available")
        session = self.session
        if session is None or session.state is not ModelState.READY:
            raise RuntimeError(
                "answer workload requires the current ready model session"
            )
        current_context = self.current_intervention_context()
        if current_context is None or (
            current_context.context_fingerprint != request.context.context_fingerprint
        ):
            raise RuntimeError("answer workload context changed before inference")
        generation = adapter.info.default_generation.model_copy(
            update={"seed": request.unit.seed}
        )
        contract = _default_evaluation_contract(adapter.info, item)
        execution = await self._execute_benchmark_item(
            adapter=adapter,
            item=item,
            attempt=1,
            generation=generation,
            contract=contract,
            policy=ExecutionPolicy(attempts=1, concurrency=1),
            session=session,
            profile=request.profile.profile if request.profile is not None else None,
            run_id=request.unit.id,
            token_budget=None,
            cost_budget_usd=None,
        )
        if request.should_cancel():
            raise ExperimentCancellationRequested
        item_result = execution.result
        deterministic = item_result.evaluation
        score = (
            None
            if item_result.passed is None
            else (
                deterministic.combined_score
                if deterministic is not None
                else (1.0 if item_result.passed else 0.0)
            )
        )
        routing = None
        if execution.routing is not None:
            routing = {
                "layer_ids": self.topology.routed_layer_ids,
                "selection_counts": execution.routing.selection_counts.tolist(),
                "routing_mass": execution.routing.routing_mass.tolist(),
                "total_routed_slots": execution.routing.total_routed_slots,
            }
        inference = item_result.performance or InferencePerformance(
            prompt_tokens=item_result.prompt_tokens,
            completion_tokens=item_result.completion_tokens,
            total_tokens=item_result.total_tokens,
            latency_ms=item_result.latency_ms,
        )
        evaluation = EvaluationResultRecord(
            id=str(uuid4()),
            workload_run_id=request.workload_run.id,
            run_unit_id=request.unit.id,
            passed=item_result.passed,
            score=score,
            metrics={
                "passed": item_result.passed,
                "scored": item_result.passed is not None,
            },
            formula_fingerprint=contract.fingerprint,
            deterministic=deterministic,
        )
        performance = PerformanceSnapshot(
            id=str(uuid4()),
            workload_run_id=request.workload_run.id,
            run_unit_id=request.unit.id,
            prompt_tokens=item_result.prompt_tokens,
            completion_tokens=item_result.completion_tokens,
            total_tokens=item_result.total_tokens,
            latency_ms=item_result.latency_ms,
            tokens_per_second=inference.tokens_per_second,
            inference=inference,
        )
        return ExperimentAdapterResult(
            passed=item_result.passed,
            score=score,
            result={
                "kind": WorkloadKind.ANSWER.value,
                "benchmark_id": descriptor.parent_id,
                "benchmark_revision": adapter.info.revision,
                "dataset_content_hash": adapter.content_hash,
                "scoring_version": adapter.scoring_version,
                "evaluation_contract_fingerprint": contract.fingerprint,
                "generation": generation.model_dump(mode="json"),
                "model_session_id": session.id,
                "context": request.context.model_dump(mode="json"),
                "prompt": item_result.prompt,
                "output": item_result.output,
                "expected": item_result.expected,
                "error": item_result.error,
                "scoring": item_result.scoring.value,
                "item": item_result.model_dump(mode="json"),
                "routing": routing,
            },
            evaluation=evaluation,
            performance=performance,
        )

    async def _execute_v2_coding_unit(
        self, request: ExperimentAdapterRequest
    ) -> ExperimentAdapterResult:
        if request.should_cancel():
            raise ExperimentCancellationRequested
        descriptor = request.experiment.workload
        if descriptor.parent_id is None or descriptor.task_id is None:
            raise RuntimeError("coding workload descriptor is missing its source task")
        session = self.session
        if session is None or session.state is not ModelState.READY:
            raise RuntimeError(
                "coding workload requires the current ready model session"
            )
        current_context = self.current_intervention_context()
        if current_context is None or (
            current_context.context_fingerprint != request.context.context_fingerprint
        ):
            raise RuntimeError("coding workload context changed before execution")
        agent_id = str(
            request.experiment.execution_config.get("agent_id", "bash-json-v1")
        )
        sandbox_provider_id = str(
            request.experiment.execution_config.get("sandbox_provider_id", "fake")
        )
        agent_request = CreateAgentRunRequest(
            task_pack_id=descriptor.parent_id,
            task_ids=[descriptor.task_id],
            agent_id=agent_id,
            sandbox_provider_id=sandbox_provider_id,
            model_session_id=session.id,
            attempts=1,
            seed=request.unit.seed if request.unit.seed is not None else 0,
        )
        agent_run_id = str(uuid4())
        self.agentic.create_run(
            run_id=agent_run_id,
            job_id=request.experiment.job_id,
            request=agent_request,
            model_session=session,
            intervention_context=request.context,
        )
        trial_id = next(
            trial.id
            for trial in self.agentic.trials.values()
            if trial.run_id == agent_run_id
        )
        execution = asyncio.create_task(
            self.agentic.execute_run(
                agent_run_id,
                model_session=session,
                on_progress=lambda _current, _total: None,
                should_cancel=request.should_cancel,
                intervention_context=request.context,
                profile=(
                    request.profile.profile if request.profile is not None else None
                ),
            )
        )
        emitted_steps = 0

        def emit_new_steps() -> None:
            nonlocal emitted_steps
            try:
                trajectory = self.agentic.get_trajectory(trial_id)
            except KeyError:
                return
            for step in trajectory.steps[emitted_steps:]:
                request.emit_trajectory(step.model_dump(mode="json", exclude_none=True))
            emitted_steps = len(trajectory.steps)

        try:
            while not execution.done():
                emit_new_steps()
                await asyncio.sleep(0.02)
            agent_run = await execution
        except BaseException:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            raise
        emit_new_steps()
        trials = [
            trial
            for trial in self.agentic.trials.values()
            if trial.run_id == agent_run_id
        ]
        if agent_run.status.value == "cancelled" or request.should_cancel():
            raise ExperimentCancellationRequested
        if agent_run.status.value == "failed":
            raise RuntimeError(agent_run.error or "coding workload execution failed")
        if len(trials) != 1:
            raise RuntimeError("coding workload did not produce exactly one trial")
        trial = trials[0]
        trajectory = self.agentic.get_trajectory(trial.id)
        routing = self.agentic.routings.get(trial.id)
        evaluation = trial.evaluation
        score = evaluation.combined_score if evaluation is not None else trial.reward
        passed = evaluation.passed if evaluation is not None else None
        if passed is None:
            passed = trial.status.value == "passed"
        trial_performance = trial.performance
        inference_performance = InferencePerformance(
            prompt_tokens=trial.prompt_tokens,
            completion_tokens=trial.completion_tokens,
            total_tokens=trial.prompt_tokens + trial.completion_tokens,
            latency_ms=(
                trial_performance.model_time_ms if trial_performance is not None else 0
            ),
            tokens_per_second=(
                trial_performance.mean_tps if trial_performance is not None else None
            ),
            started_at=trial.started_at,
            completed_at=trial.completed_at,
        )
        evaluation_record = EvaluationResultRecord(
            id=str(uuid4()),
            workload_run_id=request.workload_run.id,
            run_unit_id=request.unit.id,
            passed=passed,
            score=score,
            metrics={
                "reward": trial.reward,
                "turns": trial.turns,
                "commands": trial.commands,
            },
            formula_fingerprint=agent_run.evaluation_contract.fingerprint,
            deterministic=evaluation,
        )
        performance = PerformanceSnapshot(
            id=str(uuid4()),
            workload_run_id=request.workload_run.id,
            run_unit_id=request.unit.id,
            prompt_tokens=trial.prompt_tokens,
            completion_tokens=trial.completion_tokens,
            total_tokens=trial.prompt_tokens + trial.completion_tokens,
            latency_ms=(
                trial_performance.wall_time_ms if trial_performance is not None else 0
            ),
            tokens_per_second=(
                trial_performance.mean_tps if trial_performance is not None else None
            ),
            inference=inference_performance,
        )
        return ExperimentAdapterResult(
            passed=passed,
            score=score,
            result={
                "kind": WorkloadKind.CODING.value,
                "agent_run_id": agent_run.id,
                "trial_id": trial.id,
                "agent_run_status": agent_run.status.value,
                "trial_status": trial.status.value,
                "termination_cause": (
                    trial.termination_cause.value
                    if trial.termination_cause is not None
                    else None
                ),
                "task_pack_revision": agent_run.task_pack_revision,
                "task_pack_content_hash": agent_run.task_pack_content_hash,
                "contract_fingerprint": agent_run.contract_fingerprint,
                "evaluation_contract_fingerprint": (
                    agent_run.evaluation_contract.fingerprint
                ),
                "model_session_id": session.id,
                "context": request.context.model_dump(mode="json"),
                "trajectory": [
                    step.model_dump(mode="json", exclude_none=True)
                    for step in trajectory.steps
                ],
                "routing": (
                    routing.model_dump(mode="json") if routing is not None else None
                ),
                "artifact_endpoints": {
                    "trajectory": f"/api/trials/{trial.id}/trajectory",
                    "routing": f"/api/trials/{trial.id}/routing",
                    "artifacts": f"/api/trials/{trial.id}/artifacts",
                    "export": f"/api/trials/{trial.id}/export",
                },
            },
            evaluation=evaluation_record,
            performance=performance,
        )

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
                            profile=session.profile,
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
                expert_profile_fingerprint(
                    session.profile,
                    session.topology or self.topology,
                )
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
        profile: ExpertProfile | None,
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
                    profile=profile,
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
        if model_session is None or model_session.state not in {
            ModelState.STARTING,
            ModelState.READY,
        }:
            return
        model_session.state = ModelState.FAILED
        self.store.save_model_session(model_session)

    def _cancel_model_session_unlocked(self, job: JobRecord) -> None:
        if job.kind is not JobKind.MODEL_LOAD or job.result_id is None:
            return
        model_session = self.model_sessions.get(job.result_id)
        if model_session is None or model_session.state not in {
            ModelState.STARTING,
            ModelState.READY,
            ModelState.FAILED,
            ModelState.STOPPED,
        }:
            return
        if model_session.state is not ModelState.STOPPED:
            model_session.state = ModelState.STOPPED
            self.store.save_model_session(model_session)
        if self.session is model_session:
            self.session = None
            self.active_context = None
            self.runtime.set_active_context(None)

    def _set_model_load_phase(
        self,
        job_id: str | None,
        phase: ModelLoadPhase,
        *,
        detail: str,
        terminal: bool = False,
    ) -> None:
        if job_id is None:
            return
        with self._job_guard:
            artifacts = self.jobs[job_id]
            now = datetime.now(UTC)
            for record in reversed(artifacts.record.phase_history):
                if record.status == "active":
                    record.status = "completed"
                    record.completed_at = now
                    break
            artifacts.record.phase_history.append(
                JobPhaseRecord(
                    phase=phase,
                    status="completed" if terminal else "active",
                    started_at=now,
                    completed_at=now if terminal else None,
                    detail=detail,
                )
            )
            self.store.save_job(artifacts.record, artifacts.payload)

    @staticmethod
    def _fail_model_load_phase_unlocked(job: JobRecord, detail: str | None) -> None:
        if job.kind is not JobKind.MODEL_LOAD:
            return
        now = datetime.now(UTC)
        for record in reversed(job.phase_history):
            if record.status == "active":
                record.status = "failed"
                record.completed_at = now
                break
        job.phase_history.append(
            JobPhaseRecord(
                phase=ModelLoadPhase.FAILED,
                status="failed",
                started_at=now,
                completed_at=now,
                detail=detail,
            )
        )

    @staticmethod
    def _cancel_model_load_phase_unlocked(job: JobRecord, *, detail: str) -> None:
        if job.kind is not JobKind.MODEL_LOAD:
            return
        now = datetime.now(UTC)
        for record in reversed(job.phase_history):
            if record.status == "active":
                record.status = "cancelled"
                record.completed_at = now
                break
        job.phase_history.append(
            JobPhaseRecord(
                phase=ModelLoadPhase.CANCELLED,
                status="cancelled",
                started_at=now,
                completed_at=now,
                detail=detail,
            )
        )

    def _discard_job_task(
        self,
        job_id: str,
        task: asyncio.Task[None],
    ) -> None:
        with self._job_guard:
            if self._job_tasks.get(job_id) is task:
                self._job_tasks.pop(job_id, None)

    @staticmethod
    def _cancel_owned_task(task: asyncio.Task[None]) -> None:
        task_loop = task.get_loop()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is task_loop:
            task.cancel()
        else:
            task_loop.call_soon_threadsafe(task.cancel)

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
        model = self._resolve_manifest_model(request.model_id)
        assert model.topology is not None
        validation = validate_profile(request.profile, model.topology)
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
            profile_fingerprint=expert_profile_fingerprint(
                request.profile,
                model.topology,
            ),
            profile_fingerprint_version=EXPERT_PROFILE_FINGERPRINT_VERSION,
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
                        expert_profile_fingerprint(
                            session.profile,
                            session.topology or self.topology,
                        )
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

    def update_profile_metadata(
        self,
        profile_id: str,
        update: UpdateProfileMetadataRequest,
    ) -> SavedExpertProfile:
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        model = self._resolve_manifest_model(profile.model_id)
        assert model.topology is not None
        immutable_fingerprint = _stored_profile_fingerprint(profile, model.topology)
        if immutable_fingerprint != profile.profile_fingerprint:
            raise RuntimeError("stored expert profile fingerprint is inconsistent")
        changes: dict[str, str] = {}
        if update.name is not None:
            changes["name"] = update.name
        if update.description is not None:
            changes["description"] = update.description
        renamed = profile.model_copy(update=changes, deep=True)
        if (
            _stored_profile_fingerprint(renamed, model.topology)
            != immutable_fingerprint
        ):
            raise RuntimeError("profile metadata update changed the immutable mask")
        self.profiles[profile_id] = renamed
        self.store.save_expert_profile(renamed)
        return renamed.model_copy(deep=True)

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
        fingerprint = expert_profile_fingerprint(
            candidate_session.profile,
            candidate_session.topology or self.topology,
        )
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
        model = self._resolve_manifest_model(model_id)
        assert model.topology is not None
        canonical_fingerprint = expert_profile_fingerprint(profile, model.topology)
        legacy_fingerprint = legacy_profile_fingerprint(profile)
        matching = [
            saved
            for saved in self.profiles.values()
            if saved.model_id == model_id
            and saved.profile_fingerprint
            == (
                canonical_fingerprint
                if saved.profile_fingerprint_version
                == EXPERT_PROFILE_FINGERPRINT_VERSION
                else legacy_fingerprint
            )
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


def _stored_profile_fingerprint(
    profile: SavedExpertProfile,
    topology: ModelTopology,
) -> str:
    if profile.profile_fingerprint_version == EXPERT_PROFILE_FINGERPRINT_VERSION:
        return expert_profile_fingerprint(profile.profile, topology)
    return legacy_profile_fingerprint(profile.profile)


def _validated_context_update(
    context: InterventionContextRef,
    **updates: object,
) -> InterventionContextRef:
    try:
        return InterventionContextRef.model_validate(
            {**context.model_dump(mode="python"), **updates}
        )
    except ValueError as error:
        raise RuntimeError(
            "vLLM expert-context control returned invalid identity provenance"
        ) from error


def _registration_identity_error(
    registered: dict[str, object],
    *,
    expected_context_id: str,
    expected_profile_fingerprint: str,
    expected_topology_fingerprint: str,
    expected_layers: dict[str, dict[str, list[int]]],
) -> RuntimeError | None:
    if registered.get("context_id") != expected_context_id:
        return RuntimeError("vLLM registered an unexpected expert context ID")
    if registered.get("profile_fingerprint") != expected_profile_fingerprint:
        return RuntimeError("vLLM registered a different canonical expert mask")
    if registered.get("topology_fingerprint") != expected_topology_fingerprint:
        return RuntimeError("vLLM registered an expert mask for a different topology")
    if registered.get("layers") != expected_layers:
        return RuntimeError("vLLM registered different canonical expert-mask layers")
    return None


def _task_difficulty(tags: list[str]) -> str:
    for candidate in ("hard", "medium", "easy", "smoke", "canary"):
        if candidate in tags:
            return candidate
    return "unspecified"


def _task_expected_horizon(tags: list[str]) -> int:
    if "repo-engineering" in tags or "hard" in tags:
        return 16
    if "aider-polyglot" in tags or "medium" in tags:
        return 8
    return 4


def _external_language(task_id: str) -> str | None:
    marker = "polyglot_"
    if marker not in task_id:
        return None
    return task_id.split(marker, 1)[1].split("_", 1)[0]


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
