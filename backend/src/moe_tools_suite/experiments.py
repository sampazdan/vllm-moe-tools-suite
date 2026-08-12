from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any
from uuid import uuid4

from .domain import (
    EvaluationResult,
    ExpertProfile,
    GenerationConfig,
    ModelSession,
    ModelState,
    ModelTopology,
    SavedExpertProfile,
)
from .driftbench import (
    DRIFT_FORMULA_FINGERPRINT,
    DRIFT_FORMULAS,
    SCENARIOS,
    DriftRunResult,
    DriftToolCall,
    DriftTurnInput,
    reliability_horizon,
    run_drift_scenario,
)
from .persistence import SqliteStore
from .runtime import CompletionResult, ModelRuntime
from .telemetry import aggregate_routing
from .v2_domain import (
    ContextActivationResult,
    CreateExperimentRequest,
    DriftCondition,
    EvaluationResultRecord,
    Experiment,
    ExperimentDetail,
    ExperimentLane,
    ExperimentLaneRole,
    ExperimentStatus,
    InterventionContextRef,
    PerformanceSnapshot,
    RunEvent,
    RunEventKind,
    RunEventPage,
    RunUnit,
    RunUnitStatus,
    WorkloadDescriptor,
    WorkloadKind,
    WorkloadRun,
    WorkloadRunStatus,
    canonical_fingerprint,
    utc_now,
)

STATE_DRIFT_WORKLOAD_ID = "state-drift-v1"

ActivateContext = Callable[[str | None], Awaitable[ContextActivationResult]]
ContextBuilder = Callable[[SavedExpertProfile | None], InterventionContextRef]
CurrentContext = Callable[[], InterventionContextRef | None]
CurrentSession = Callable[[], ModelSession | None]


@dataclass
class _ProviderTelemetry:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    routing: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentAdapterRequest:
    experiment: Experiment
    lane: ExperimentLane
    workload_run: WorkloadRun
    unit: RunUnit
    profile: SavedExpertProfile | None
    context: InterventionContextRef
    should_cancel: Callable[[], bool]
    emit_trajectory: Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ExperimentAdapterResult:
    passed: bool | None
    score: float | None
    result: dict[str, Any]
    evaluation: EvaluationResultRecord
    performance: PerformanceSnapshot


AdapterExecutor = Callable[
    [ExperimentAdapterRequest], Awaitable[ExperimentAdapterResult]
]


class ExperimentCancellationRequested(RuntimeError):
    """A workload adapter observed a durable experiment cancellation."""


class ExperimentService:
    """Durable V2 orchestration for synchronized workload lanes."""

    def __init__(
        self,
        *,
        store: SqliteStore,
        runtime: ModelRuntime,
        topology: ModelTopology,
        model_id: str,
        mode: str,
        profiles: dict[str, SavedExpertProfile],
        activate_context: ActivateContext,
        context_builder: ContextBuilder,
        current_context: CurrentContext,
        current_session: CurrentSession,
        answer_executor: AdapterExecutor | None = None,
        coding_executor: AdapterExecutor | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.topology = topology
        self.model_id = model_id
        self.mode = mode
        self.profiles = profiles
        self._activate_context = activate_context
        self._context_builder = context_builder
        self._current_context = current_context
        self._current_session = current_session
        self._answer_executor = answer_executor
        self._coding_executor = coding_executor
        self.experiments = store.load_experiments()
        self.workload_runs = store.load_workload_runs()
        self.units = store.load_run_units()
        self._cancel_requested: set[str] = set()

    @staticmethod
    def state_drift_descriptor() -> WorkloadDescriptor:
        scenarios = [SCENARIOS[key] for key in sorted(SCENARIOS)]
        identity = {
            "id": STATE_DRIFT_WORKLOAD_ID,
            "revision": "1.0.0",
            "formula_fingerprint": DRIFT_FORMULA_FINGERPRINT,
            "scenarios": [
                {
                    "id": scenario.id,
                    "revision": scenario.revision,
                    "name": scenario.name,
                }
                for scenario in scenarios
            ],
        }
        return WorkloadDescriptor(
            id=STATE_DRIFT_WORKLOAD_ID,
            kind=WorkloadKind.STATE_DRIFT,
            name="State Drift Bench",
            description=(
                "Deterministic state-machine scenarios with exact checkpoint "
                "scoring, rollback, and paired baseline/mask lanes."
            ),
            source="first-party",
            revision="1.0.0",
            content_fingerprint=canonical_fingerprint(identity),
            ready=True,
            unit_ids=[scenario.id for scenario in scenarios],
            difficulty="parameterized",
            expected_horizon=24,
            tools=[
                "typed deterministic reducers",
                "snapshot",
                "rollback",
            ],
            runtime="in-process deterministic state machine",
            preparation_status="validated",
            public_problem_statement=(
                "Maintain exact application state across scripted changes, "
                "branches, snapshots, and rollbacks."
            ),
            public_success_criteria=[
                "Apply valid typed tools for every transition.",
                "Preserve scenario invariants and unrelated state.",
                "Match hidden canonical checkpoint state exactly.",
            ],
            metadata={
                "conditions": [condition.value for condition in DriftCondition],
                "horizons": [2, 4, 8, 12, 16, 24],
                "formulas": DRIFT_FORMULAS,
                "formula_fingerprint": DRIFT_FORMULA_FINGERPRINT,
                "scenarios": [
                    {
                        "id": scenario.id,
                        "name": scenario.name,
                        "description": scenario.description,
                        "revision": scenario.revision,
                    }
                    for scenario in scenarios
                ],
            },
        )

    def validate_request(
        self,
        request: CreateExperimentRequest,
        descriptor: WorkloadDescriptor,
    ) -> None:
        if request.model_id != self.model_id:
            raise ValueError(f"unsupported experiment model {request.model_id!r}")
        if descriptor.id != request.workload_id:
            raise ValueError("resolved workload does not match the experiment request")
        if not descriptor.ready:
            reason = descriptor.blocked_reason or "workload is not prepared"
            raise ValueError(f"workload is not executable: {reason}")
        if descriptor.kind is WorkloadKind.STATE_DRIFT:
            unknown = set(request.scenario_ids) - SCENARIOS.keys()
            if unknown:
                raise ValueError(f"unknown state-drift scenarios: {sorted(unknown)}")
        elif descriptor.kind is WorkloadKind.ANSWER:
            if self._answer_executor is None:
                raise ValueError("answer workload execution is unavailable")
        elif descriptor.kind is WorkloadKind.CODING:
            if self._coding_executor is None:
                raise ValueError("coding workload execution is unavailable")
        session = self._current_session()
        if session is None or session.state is not ModelState.READY:
            raise RuntimeError("load a model before starting an experiment")
        if session.model_id != request.model_id:
            raise ValueError("the ready model does not match the experiment model")
        if request.candidate_profile_id is not None:
            profile = self.profiles.get(request.candidate_profile_id)
            if profile is None:
                raise KeyError(request.candidate_profile_id)
            if profile.model_id != request.model_id:
                raise ValueError("candidate profile belongs to a different model")

    def create(
        self,
        request: CreateExperimentRequest,
        *,
        job_id: str,
        descriptor: WorkloadDescriptor,
        experiment_id: str | None = None,
    ) -> Experiment:
        self.validate_request(request, descriptor)
        experiment_id = experiment_id or str(uuid4())
        baseline_context = self._context_builder(None)
        candidate_profile = (
            self.profiles[request.candidate_profile_id]
            if request.candidate_profile_id is not None
            else None
        )
        contexts = [(ExperimentLaneRole.BASELINE, baseline_context)]
        if candidate_profile is not None:
            contexts.append(
                (
                    ExperimentLaneRole.CANDIDATE,
                    self._context_builder(candidate_profile),
                )
            )
        units_per_lane = (
            len(request.scenario_ids)
            * len(request.conditions)
            * len(request.horizons)
            * len(request.seeds)
            if descriptor.kind is WorkloadKind.STATE_DRIFT
            else len(request.seeds)
        )
        lanes: list[ExperimentLane] = []
        runs: list[WorkloadRun] = []
        for role, context in contexts:
            lane_id = str(uuid4())
            run_id = str(uuid4())
            lanes.append(
                ExperimentLane(
                    id=lane_id,
                    experiment_id=experiment_id,
                    role=role,
                    label=(
                        "Baseline"
                        if role is ExperimentLaneRole.BASELINE
                        else candidate_profile.name
                    ),
                    context=context,
                    workload_run_id=run_id,
                )
            )
            runs.append(
                WorkloadRun(
                    id=run_id,
                    experiment_id=experiment_id,
                    lane_id=lane_id,
                    workload=descriptor,
                    context=context,
                    total_units=units_per_lane,
                )
            )
        units: list[RunUnit] = []
        ordinal = 0
        if descriptor.kind is WorkloadKind.STATE_DRIFT:
            dimensions = (
                (scenario_id, condition, horizon, seed)
                for scenario_id in request.scenario_ids
                for condition in request.conditions
                for horizon in request.horizons
                for seed in request.seeds
            )
        else:
            dimensions = ((None, None, None, seed) for seed in request.seeds)
        for scenario_id, condition, horizon, seed in dimensions:
            for lane, run in zip(lanes, runs, strict=True):
                unit_dimension = (
                    f"{scenario_id}:{condition.value}:h{horizon}:s{seed}"
                    if scenario_id is not None and condition is not None
                    else f"{descriptor.task_id or descriptor.id}:s{seed}"
                )
                units.append(
                    RunUnit(
                        id=str(uuid4()),
                        experiment_id=experiment_id,
                        lane_id=lane.id,
                        workload_run_id=run.id,
                        ordinal=ordinal,
                        unit_key=f"{unit_dimension}:{lane.role.value}",
                        scenario_id=scenario_id,
                        condition=condition,
                        horizon=horizon,
                        seed=seed,
                    )
                )
                ordinal += 1
        experiment = Experiment(
            id=experiment_id,
            job_id=job_id,
            name=request.name or f"{descriptor.name} · {utc_now():%Y-%m-%d %H:%M}",
            model_id=request.model_id,
            workload=descriptor,
            execution_config={
                "agent_id": request.agent_id,
                "sandbox_provider_id": request.sandbox_provider_id,
            },
            lanes=lanes,
            total_units=len(units),
            latest_event_sequence=1,
        )
        initial_event = RunEvent(
            experiment_id=experiment_id,
            sequence=1,
            kind=RunEventKind.QUEUED,
            phase="queued",
            message="Experiment graph persisted and queued",
            data={"total_units": len(units), "lane_count": len(lanes)},
        )
        self.store.save_experiment_bundle(experiment, runs, units, initial_event)
        self.experiments[experiment.id] = experiment
        self.workload_runs.update({run.id: run for run in runs})
        self.units.update({unit.id: unit for unit in units})
        return experiment.model_copy(deep=True)

    def list(self) -> list[Experiment]:
        return sorted(
            (item.model_copy(deep=True) for item in self.experiments.values()),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def get(self, experiment_id: str) -> Experiment:
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        return experiment.model_copy(deep=True)

    def detail(self, experiment_id: str) -> ExperimentDetail:
        experiment = self.get(experiment_id)
        runs = sorted(
            (
                run.model_copy(deep=True)
                for run in self.workload_runs.values()
                if run.experiment_id == experiment_id
            ),
            key=lambda run: run.created_at,
        )
        units = sorted(
            (
                unit.model_copy(deep=True)
                for unit in self.units.values()
                if unit.experiment_id == experiment_id
            ),
            key=lambda unit: unit.ordinal,
        )
        return ExperimentDetail(
            experiment=experiment,
            workload_runs=runs,
            units=units,
        )

    def events(
        self, experiment_id: str, *, after: int = 0, limit: int = 500
    ) -> RunEventPage:
        if experiment_id not in self.experiments:
            raise KeyError(experiment_id)
        events, latest, has_more = self.store.load_run_events(
            experiment_id, after=after, limit=limit
        )
        return RunEventPage(
            experiment_id=experiment_id,
            events=events,
            after=after,
            latest_sequence=latest,
            has_more=has_more,
        )

    def request_cancel(self, experiment_id: str) -> Experiment:
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        if experiment.status not in {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
        }:
            return experiment.model_copy(deep=True)
        self._cancel_requested.add(experiment_id)
        experiment.status = ExperimentStatus.CANCELLING
        self._save_with_event(
            experiment,
            RunEventKind.CANCELLING,
            phase="cancelling",
            message="Cancellation requested",
        )
        return experiment.model_copy(deep=True)

    def cancel_queued(self, experiment_id: str) -> Experiment:
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        if experiment.status is not ExperimentStatus.QUEUED:
            return self.request_cancel(experiment_id)
        ordered_units = [
            unit for unit in self.units.values() if unit.experiment_id == experiment_id
        ]
        return self._finish_cancelled(experiment, ordered_units)

    async def execute(
        self,
        experiment_id: str,
        *,
        on_progress: Callable[[int, int], None],
        should_cancel: Callable[[], bool],
    ) -> Experiment:
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = utc_now()
        for lane in experiment.lanes:
            lane.status = WorkloadRunStatus.RUNNING
            run = self.workload_runs[lane.workload_run_id]
            run.status = WorkloadRunStatus.RUNNING
            run.started_at = experiment.started_at
            self.store.save_workload_run(run)
        self._save_with_event(
            experiment,
            RunEventKind.PREPARING,
            phase="preparing",
            message=f"Preparing synchronized {experiment.workload.kind.value} lanes",
        )
        ordered_units = sorted(
            (
                unit
                for unit in self.units.values()
                if unit.experiment_id == experiment_id
            ),
            key=lambda unit: unit.ordinal,
        )
        active_unit: RunUnit | None = None
        try:
            for unit in ordered_units:
                if self._cancelled(experiment_id, should_cancel):
                    return self._finish_cancelled(experiment, ordered_units)
                active_unit = unit
                await self._execute_unit(
                    experiment,
                    unit,
                    should_cancel=lambda: self._cancelled(experiment_id, should_cancel),
                )
                experiment.completed_units += 1
                if unit.status is RunUnitStatus.PASSED:
                    experiment.passed_units += 1
                run = self.workload_runs[unit.workload_run_id]
                run.completed_units += 1
                if unit.status is RunUnitStatus.PASSED:
                    run.passed_units += 1
                terminal_kind = (
                    RunEventKind.UNIT_PASSED
                    if unit.status is RunUnitStatus.PASSED
                    else (
                        RunEventKind.UNIT_FAILED
                        if unit.status is RunUnitStatus.FAILED
                        else RunEventKind.UNIT_UNSCORED
                    )
                )
                self._save_with_event(
                    experiment,
                    terminal_kind,
                    phase="evaluating",
                    message=(
                        "Workload unit passed"
                        if unit.status is RunUnitStatus.PASSED
                        else (
                            "Workload unit failed"
                            if unit.status is RunUnitStatus.FAILED
                            else "Workload unit completed without a score"
                        )
                    ),
                    lane_id=unit.lane_id,
                    workload_run_id=run.id,
                    run_unit_id=unit.id,
                    data={
                        "score": (unit.evaluation.score if unit.evaluation else None)
                    },
                )
                on_progress(experiment.completed_units, experiment.total_units)
                active_unit = None
        except ExperimentCancellationRequested:
            return self._finish_cancelled(experiment, ordered_units)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            experiment.status = ExperimentStatus.FAILED
            experiment.error = str(error) or error.__class__.__name__
            experiment.completed_at = utc_now()
            for lane in experiment.lanes:
                if lane.status is WorkloadRunStatus.RUNNING:
                    lane.status = WorkloadRunStatus.FAILED
                    run = self.workload_runs[lane.workload_run_id]
                    run.status = WorkloadRunStatus.FAILED
                    run.error = experiment.error
                    run.completed_at = experiment.completed_at
            failed_run = None
            if active_unit is not None:
                active_unit.status = RunUnitStatus.ERROR
                active_unit.completed_at = experiment.completed_at
                active_unit.result = {
                    **(active_unit.result or {}),
                    "error": experiment.error,
                }
                failed_run = self.workload_runs[active_unit.workload_run_id]
            self._save_with_event(
                experiment,
                RunEventKind.FAILED,
                phase="failed",
                message=experiment.error,
                lane_id=active_unit.lane_id if active_unit else None,
                workload_run_id=(failed_run.id if failed_run else None),
                run_unit_id=active_unit.id if active_unit else None,
            )
            raise

        for lane in experiment.lanes:
            run = self.workload_runs[lane.workload_run_id]
            run.status = WorkloadRunStatus.COMPLETED
            run.completed_at = utc_now()
            run.aggregate_metrics = self._aggregate_lane(experiment, lane.id)
            lane.status = WorkloadRunStatus.COMPLETED
            self.store.save_workload_run(run)
        experiment.comparison = self._comparison(experiment)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = utc_now()
        self._save_with_event(
            experiment,
            RunEventKind.COMPLETED,
            phase="completed",
            message="All experiment units completed",
            data={
                "completed_units": experiment.completed_units,
                "passed_units": experiment.passed_units,
            },
        )
        return experiment.model_copy(deep=True)

    async def _execute_unit(
        self,
        experiment: Experiment,
        unit: RunUnit,
        *,
        should_cancel: Callable[[], bool],
    ) -> None:
        lane = next(lane for lane in experiment.lanes if lane.id == unit.lane_id)
        profile_id = lane.context.profile_id
        activation_started = time.perf_counter()
        receipt = await self._activate_context(profile_id)
        activation_ms = (time.perf_counter() - activation_started) * 1000
        active_context = self._current_context()
        if active_context is None:
            raise RuntimeError("context activation did not establish active provenance")
        lane.context = active_context
        run = self.workload_runs[unit.workload_run_id]
        run.context = active_context
        self.store.save_workload_run(run)
        self._save_with_event(
            experiment,
            RunEventKind.CONTEXT_SWITCHED,
            phase="context_switch",
            message=f"Activated {lane.label}",
            lane_id=lane.id,
            workload_run_id=run.id,
            run_unit_id=unit.id,
            data=receipt.model_dump(mode="json"),
        )
        unit.status = RunUnitStatus.RUNNING
        unit.started_at = utc_now()
        self.store.save_run_unit(unit)
        self._save_with_event(
            experiment,
            RunEventKind.CURRENT_UNIT,
            phase="running",
            message=f"Running {unit.unit_key}",
            lane_id=lane.id,
            workload_run_id=run.id,
            run_unit_id=unit.id,
            data={
                "ordinal": unit.ordinal,
                "total_units": experiment.total_units,
                "scenario_id": unit.scenario_id,
                "condition": unit.condition.value if unit.condition else None,
                "horizon": unit.horizon,
                "seed": unit.seed,
            },
        )
        profile = self.profiles.get(profile_id) if profile_id else None
        if experiment.workload.kind is not WorkloadKind.STATE_DRIFT:
            executor = (
                self._answer_executor
                if experiment.workload.kind is WorkloadKind.ANSWER
                else self._coding_executor
            )
            if executor is None:
                raise RuntimeError(
                    f"{experiment.workload.kind.value} workload executor is unavailable"
                )
            self._save_with_event(
                experiment,
                RunEventKind.MODEL_REQUEST_STARTED,
                phase="executing_workload",
                message=f"Executing {experiment.workload.kind.value} workload",
                lane_id=lane.id,
                workload_run_id=run.id,
                run_unit_id=unit.id,
            )
            adapter_result = await executor(
                ExperimentAdapterRequest(
                    experiment=experiment,
                    lane=lane,
                    workload_run=run,
                    unit=unit,
                    profile=profile,
                    context=active_context,
                    should_cancel=should_cancel,
                    emit_trajectory=lambda step: self._save_with_event(
                        experiment,
                        RunEventKind.TRAJECTORY_STEP,
                        phase=str(step.get("phase") or "trajectory"),
                        message=str(step.get("title") or "Agent trajectory update"),
                        lane_id=lane.id,
                        workload_run_id=run.id,
                        run_unit_id=unit.id,
                        data={"trajectory_step": step},
                    ),
                )
            )
            self._save_with_event(
                experiment,
                RunEventKind.MODEL_REQUEST_COMPLETED,
                phase="executing_workload",
                message=f"Finished {experiment.workload.kind.value} workload",
                lane_id=lane.id,
                workload_run_id=run.id,
                run_unit_id=unit.id,
                data={
                    "prompt_tokens": adapter_result.performance.prompt_tokens,
                    "completion_tokens": (adapter_result.performance.completion_tokens),
                    "context_fingerprint": active_context.context_fingerprint,
                },
            )
            unit.result = adapter_result.result
            unit.evaluation = (
                adapter_result.evaluation.model_copy(
                    update={"passed": None, "score": None}
                )
                if adapter_result.passed is None
                else adapter_result.evaluation
            )
            unit.performance = adapter_result.performance.model_copy(
                update={
                    "context_activation_ms": max(receipt.duration_ms, activation_ms)
                }
            )
            unit.status = (
                RunUnitStatus.PASSED
                if adapter_result.passed is True
                else (
                    RunUnitStatus.FAILED
                    if adapter_result.passed is False
                    else RunUnitStatus.UNSCORED
                )
            )
            unit.completed_at = utc_now()
            self.store.save_run_unit(unit)
            return
        if unit.scenario_id is None or unit.condition is None:
            raise RuntimeError("state-drift unit is missing scenario dimensions")
        if unit.horizon is None or unit.seed is None:
            raise RuntimeError("state-drift unit is missing horizon or seed")
        telemetry = _ProviderTelemetry()
        provider = self._action_provider(
            lane=lane,
            unit=unit,
            profile=profile.profile if profile else None,
            telemetry=telemetry,
            should_cancel=should_cancel,
        )

        async def checkpoint(current: int, total: int) -> None:
            if should_cancel():
                raise ExperimentCancellationRequested
            self._save_with_event(
                experiment,
                RunEventKind.CURRENT_CHECKPOINT,
                phase="checkpoint",
                message=f"Checkpoint {current}/{total}",
                lane_id=lane.id,
                workload_run_id=run.id,
                run_unit_id=unit.id,
                checkpoint=current,
            )

        started = time.perf_counter()
        result = await run_drift_scenario(
            SCENARIOS[unit.scenario_id],
            seed=unit.seed,
            horizon=unit.horizon,
            condition=unit.condition,
            action_provider=provider,
            on_checkpoint=checkpoint,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        unit.result = result.model_dump(mode="json")
        unit.evaluation = EvaluationResultRecord(
            id=str(uuid4()),
            workload_run_id=run.id,
            run_unit_id=unit.id,
            passed=result.metrics.final_success,
            score=result.metrics.area_under_state_fidelity_curve,
            metrics=_numeric_metrics(result),
            formula_fingerprint=DRIFT_FORMULA_FINGERPRINT,
            deterministic=EvaluationResult(
                deterministic_score=(result.metrics.area_under_state_fidelity_curve),
                combined_score=result.metrics.area_under_state_fidelity_curve,
                passed=result.metrics.final_success,
            ),
        )
        total_tokens = telemetry.prompt_tokens + telemetry.completion_tokens
        unit.performance = PerformanceSnapshot(
            id=str(uuid4()),
            workload_run_id=run.id,
            run_unit_id=unit.id,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            total_tokens=total_tokens,
            latency_ms=wall_ms,
            tokens_per_second=(
                telemetry.completion_tokens / (telemetry.latency_ms / 1000)
                if telemetry.latency_ms > 0
                else None
            ),
            context_activation_ms=max(receipt.duration_ms, activation_ms),
        )
        if telemetry.routing:
            unit.result["routing"] = telemetry.routing
        unit.status = (
            RunUnitStatus.PASSED
            if result.metrics.final_success
            else RunUnitStatus.FAILED
        )
        unit.completed_at = utc_now()
        self.store.save_run_unit(unit)

    def _action_provider(
        self,
        *,
        lane: ExperimentLane,
        unit: RunUnit,
        profile: ExpertProfile | None,
        telemetry: _ProviderTelemetry,
        should_cancel: Callable[[], bool],
    ) -> Callable[[DriftTurnInput], Awaitable[DriftToolCall]]:
        scenario = SCENARIOS[unit.scenario_id or ""]
        plan = scenario.plan(unit.seed or 0, unit.horizon or 1)
        if self.mode == "mock":

            async def mock_provider(turn: DriftTurnInput) -> DriftToolCall:
                await asyncio.sleep(0)
                if should_cancel():
                    raise ExperimentCancellationRequested
                expected = plan[turn.checkpoint - 1].canonical_call
                if (
                    lane.role is ExperimentLaneRole.CANDIDATE
                    and turn.condition is DriftCondition.CHAINED
                    and turn.checkpoint == 2
                ):
                    return _perturb_call(expected)
                return expected

            return mock_provider

        async def model_provider(turn: DriftTurnInput) -> DriftToolCall:
            if should_cancel():
                raise ExperimentCancellationRequested
            self._event(
                self.experiments[unit.experiment_id],
                RunEventKind.MODEL_REQUEST_STARTED,
                phase="model_request",
                message=f"Model request for checkpoint {turn.checkpoint}",
                lane_id=lane.id,
                workload_run_id=unit.workload_run_id,
                run_unit_id=unit.id,
                checkpoint=turn.checkpoint,
            )
            completion = await self.runtime.complete_chat(
                _drift_messages(turn),
                request_key=(
                    f"drift:{unit.experiment_id}:{lane.id}:{unit.id}:{turn.checkpoint}"
                ),
                profile=profile,
                generation=GenerationConfig(
                    temperature=0,
                    max_tokens=256,
                    seed=(unit.seed or 0) + turn.checkpoint,
                    enable_thinking=False,
                ),
            )
            if should_cancel():
                raise ExperimentCancellationRequested
            _record_completion(
                completion,
                turn.checkpoint,
                telemetry,
                self.topology,
            )
            self._event(
                self.experiments[unit.experiment_id],
                RunEventKind.MODEL_REQUEST_COMPLETED,
                phase="model_request",
                message=f"Model response for checkpoint {turn.checkpoint}",
                lane_id=lane.id,
                workload_run_id=unit.workload_run_id,
                run_unit_id=unit.id,
                checkpoint=turn.checkpoint,
                data={
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "context_fingerprint": completion.context_fingerprint,
                },
            )
            return _parse_tool_call(completion.content)

        return model_provider

    def _aggregate_lane(
        self,
        experiment: Experiment,
        lane_id: str,
    ) -> dict[str, str | float | int | bool | None]:
        lane_units = [
            unit
            for unit in self.units.values()
            if unit.lane_id == lane_id and unit.evaluation is not None
        ]
        if experiment.workload.kind is not WorkloadKind.STATE_DRIFT:
            scored = [
                unit.evaluation.score
                for unit in lane_units
                if unit.evaluation is not None and unit.evaluation.score is not None
            ]
            passed = [
                unit.evaluation.passed
                for unit in lane_units
                if unit.evaluation is not None and unit.evaluation.passed is not None
            ]
            return {
                "mean_score": mean(scored) if scored else None,
                "pass_rate": (
                    mean(float(value) for value in passed) if passed else None
                ),
                "completed_units": len(lane_units),
                "scored_units": len(scored),
            }
        results = [
            _drift_result(unit)
            for unit in self.units.values()
            if unit.lane_id == lane_id and unit.result is not None
        ]
        by_horizon: dict[int, list[bool]] = defaultdict(list)
        for result in results:
            by_horizon[result.horizon].append(result.metrics.final_success)
        return {
            "mean_transition_accuracy": _mean_or_zero(
                [result.metrics.transition_accuracy for result in results]
            ),
            "mean_state_fidelity_auc": _mean_or_zero(
                [result.metrics.area_under_state_fidelity_curve for result in results]
            ),
            "final_success_rate": _mean_or_zero(
                [float(result.metrics.final_success) for result in results]
            ),
            "horizon_at_80_percent_reliability": reliability_horizon(by_horizon, 0.8),
            "horizon_at_50_percent_reliability": reliability_horizon(by_horizon, 0.5),
            "formula_fingerprint": DRIFT_FORMULA_FINGERPRINT,
        }

    def _comparison(self, experiment: Experiment) -> dict[str, Any] | None:
        if len(experiment.lanes) != 2:
            return None
        baseline = next(
            lane
            for lane in experiment.lanes
            if lane.role is ExperimentLaneRole.BASELINE
        )
        candidate = next(
            lane
            for lane in experiment.lanes
            if lane.role is ExperimentLaneRole.CANDIDATE
        )
        if experiment.workload.kind is not WorkloadKind.STATE_DRIFT:
            baseline_units = {
                unit.seed: unit
                for unit in self.units.values()
                if unit.lane_id == baseline.id and unit.evaluation is not None
            }
            candidate_units = {
                unit.seed: unit
                for unit in self.units.values()
                if unit.lane_id == candidate.id and unit.evaluation is not None
            }
            if baseline_units.keys() != candidate_units.keys():
                raise RuntimeError("paired experiment lanes are not aligned")
            score_deltas = []
            pass_changes = {
                "regressions": 0,
                "recoveries": 0,
                "retained_passes": 0,
                "retained_failures": 0,
            }
            for key in baseline_units:
                baseline_evaluation = baseline_units[key].evaluation
                candidate_evaluation = candidate_units[key].evaluation
                assert baseline_evaluation is not None
                assert candidate_evaluation is not None
                if (
                    baseline_evaluation.score is not None
                    and candidate_evaluation.score is not None
                ):
                    score_deltas.append(
                        candidate_evaluation.score - baseline_evaluation.score
                    )
                pair = (
                    baseline_evaluation.passed,
                    candidate_evaluation.passed,
                )
                if pair == (True, False):
                    pass_changes["regressions"] += 1
                elif pair == (False, True):
                    pass_changes["recoveries"] += 1
                elif pair == (True, True):
                    pass_changes["retained_passes"] += 1
                elif pair == (False, False):
                    pass_changes["retained_failures"] += 1
            return {
                "paired_observations": len(baseline_units),
                "mean_score_delta": (mean(score_deltas) if score_deltas else None),
                **pass_changes,
            }
        baseline_units = self._results_by_alignment(baseline.id)
        candidate_units = self._results_by_alignment(candidate.id)
        if baseline_units.keys() != candidate_units.keys():
            raise RuntimeError("paired experiment lanes are not aligned")
        baseline_local = _condition_mean(
            baseline_units.values(), DriftCondition.ORACLE_RESET
        )
        candidate_local = _condition_mean(
            candidate_units.values(), DriftCondition.ORACLE_RESET
        )
        baseline_chained = _condition_mean(
            baseline_units.values(), DriftCondition.CHAINED
        )
        candidate_chained = _condition_mean(
            candidate_units.values(), DriftCondition.CHAINED
        )
        baseline_penalty = _optional_difference(baseline_local, baseline_chained)
        candidate_penalty = _optional_difference(candidate_local, candidate_chained)
        paired_deltas = [
            candidate_units[key].metrics.area_under_state_fidelity_curve
            - baseline_units[key].metrics.area_under_state_fidelity_curve
            for key in baseline_units
        ]
        baseline_divergence = [
            result.metrics.first_divergence_checkpoint
            for result in baseline_units.values()
            if result.metrics.first_divergence_checkpoint is not None
        ]
        candidate_divergence = [
            result.metrics.first_divergence_checkpoint
            for result in candidate_units.values()
            if result.metrics.first_divergence_checkpoint is not None
        ]
        routing_pairs = self._paired_routing_metrics(
            baseline.id,
            candidate.id,
            candidate_units,
        )
        first_divergence_routing = [
            pair for pair in routing_pairs if pair["at_candidate_first_divergence"]
        ]
        return {
            "formula_fingerprint": DRIFT_FORMULA_FINGERPRINT,
            "paired_observations": len(paired_deltas),
            "baseline_local_competence": baseline_local,
            "candidate_local_competence": candidate_local,
            "local_capability_delta": _optional_difference(
                candidate_local, baseline_local
            ),
            "baseline_chained_fidelity": baseline_chained,
            "candidate_chained_fidelity": candidate_chained,
            "baseline_compounding_penalty": baseline_penalty,
            "candidate_compounding_penalty": candidate_penalty,
            "excess_compounding_penalty": _optional_difference(
                candidate_penalty, baseline_penalty
            ),
            "baseline_to_mask_paired_drift_delta": _mean_or_zero(paired_deltas),
            "baseline_median_first_divergence": (
                median(baseline_divergence) if baseline_divergence else None
            ),
            "candidate_median_first_divergence": (
                median(candidate_divergence) if candidate_divergence else None
            ),
            "mean_routing_selection_overlap": _optional_mean(
                routing_pairs, "selection_overlap"
            ),
            "mean_routing_mass_js_divergence": _optional_mean(
                routing_pairs, "routing_mass_js_divergence"
            ),
            "first_divergence_mean_routing_selection_overlap": _optional_mean(
                first_divergence_routing, "selection_overlap"
            ),
            "first_divergence_mean_routing_mass_js_divergence": _optional_mean(
                first_divergence_routing, "routing_mass_js_divergence"
            ),
            "routing_checkpoint_pairs": routing_pairs,
        }

    def _paired_routing_metrics(
        self,
        baseline_lane_id: str,
        candidate_lane_id: str,
        candidate_results: dict[tuple[str, DriftCondition, int, int], DriftRunResult],
    ) -> list[dict[str, Any]]:
        baseline_units = self._unit_records_by_alignment(baseline_lane_id)
        candidate_units = self._unit_records_by_alignment(candidate_lane_id)
        pairs: list[dict[str, Any]] = []
        for alignment in sorted(
            baseline_units.keys() & candidate_units.keys(),
            key=lambda item: (item[0], item[1].value, item[2], item[3]),
        ):
            baseline_routing = _routing_by_checkpoint(baseline_units[alignment])
            candidate_routing = _routing_by_checkpoint(candidate_units[alignment])
            first_divergence = candidate_results[
                alignment
            ].metrics.first_divergence_checkpoint
            for checkpoint in sorted(
                baseline_routing.keys() & candidate_routing.keys()
            ):
                baseline_record = baseline_routing[checkpoint]
                candidate_record = candidate_routing[checkpoint]
                selection_overlap, mass_divergence = _routing_distance(
                    baseline_record,
                    candidate_record,
                )
                pairs.append(
                    {
                        "scenario_id": alignment[0],
                        "condition": alignment[1].value,
                        "horizon": alignment[2],
                        "seed": alignment[3],
                        "checkpoint": checkpoint,
                        "at_candidate_first_divergence": (
                            checkpoint == first_divergence
                        ),
                        "selection_overlap": selection_overlap,
                        "routing_mass_js_divergence": mass_divergence,
                        "largest_selection_shifts": _largest_routing_shifts(
                            baseline_record,
                            candidate_record,
                        ),
                    }
                )
        return pairs

    def _results_by_alignment(
        self, lane_id: str
    ) -> dict[tuple[str, DriftCondition, int, int], DriftRunResult]:
        aligned = {}
        for unit in self.units.values():
            if unit.lane_id != lane_id or unit.result is None:
                continue
            result = _drift_result(unit)
            aligned[
                (
                    result.scenario_id,
                    result.condition,
                    result.horizon,
                    result.seed,
                )
            ] = result
        return aligned

    def _unit_records_by_alignment(
        self, lane_id: str
    ) -> dict[tuple[str, DriftCondition, int, int], RunUnit]:
        aligned = {}
        for unit in self.units.values():
            if (
                unit.lane_id != lane_id
                or unit.result is None
                or unit.scenario_id is None
                or unit.condition is None
                or unit.horizon is None
                or unit.seed is None
            ):
                continue
            aligned[(unit.scenario_id, unit.condition, unit.horizon, unit.seed)] = unit
        return aligned

    def _finish_cancelled(
        self, experiment: Experiment, ordered_units: list[RunUnit]
    ) -> Experiment:
        now = utc_now()
        for unit in ordered_units:
            if unit.status in {RunUnitStatus.QUEUED, RunUnitStatus.RUNNING}:
                unit.status = RunUnitStatus.CANCELLED
                unit.completed_at = now
                self.store.save_run_unit(unit)
        for lane in experiment.lanes:
            lane.status = WorkloadRunStatus.CANCELLED
            run = self.workload_runs[lane.workload_run_id]
            run.status = WorkloadRunStatus.CANCELLED
            run.completed_at = now
            self.store.save_workload_run(run)
        experiment.status = ExperimentStatus.CANCELLED
        experiment.completed_at = now
        self._save_with_event(
            experiment,
            RunEventKind.CANCELLED,
            phase="cancelled",
            message="Experiment cancelled at a unit boundary",
        )
        return experiment.model_copy(deep=True)

    def _cancelled(self, experiment_id: str, should_cancel: Callable[[], bool]) -> bool:
        return experiment_id in self._cancel_requested or should_cancel()

    def _save_with_event(
        self,
        experiment: Experiment,
        kind: RunEventKind,
        *,
        phase: str,
        message: str,
        lane_id: str | None = None,
        workload_run_id: str | None = None,
        run_unit_id: str | None = None,
        checkpoint: int | None = None,
        data: dict[str, object] | None = None,
    ) -> RunEvent:
        event = self._event(
            experiment,
            kind,
            phase=phase,
            message=message,
            lane_id=lane_id,
            workload_run_id=workload_run_id,
            run_unit_id=run_unit_id,
            checkpoint=checkpoint,
            data=data,
        )
        experiment.latest_event_sequence = event.sequence
        return event

    def _event(
        self,
        experiment: Experiment,
        kind: RunEventKind,
        *,
        phase: str,
        message: str,
        lane_id: str | None = None,
        workload_run_id: str | None = None,
        run_unit_id: str | None = None,
        checkpoint: int | None = None,
        data: dict[str, object] | None = None,
    ) -> RunEvent:
        workload_runs = [
            run
            for run in self.workload_runs.values()
            if run.experiment_id == experiment.id
        ]
        unit = self.units.get(run_unit_id) if run_unit_id is not None else None
        event = self.store.save_experiment_event(
            experiment,
            kind=kind,
            phase=phase,
            message=message,
            lane_id=lane_id,
            workload_runs=workload_runs,
            unit=unit,
            checkpoint=checkpoint,
            data=data,
        )
        experiment.latest_event_sequence = event.sequence
        return event


def _perturb_call(expected: DriftToolCall) -> DriftToolCall:
    arguments = copy.deepcopy(expected.arguments)
    if type(arguments.get("amount")) is int:
        arguments["amount"] += 1
    elif type(arguments.get("quantity")) is int:
        arguments["quantity"] += 1
    elif "name" in arguments:
        arguments["name"] = f"{arguments['name']}-drifted"
    elif "order_id" in arguments:
        arguments["order_id"] = f"{arguments['order_id']}-drifted"
    else:
        return DriftToolCall(tool="invalid_tool", arguments={})
    return DriftToolCall(tool=expected.tool, arguments=arguments)


def _drift_messages(turn: DriftTurnInput) -> list[dict[str, str]]:
    system = (
        "You operate a deterministic state store. Reply with exactly one JSON "
        "object containing keys tool and arguments. Do not include prose."
    )
    payload: dict[str, object] = {
        "instruction": turn.instruction,
        "observed_state": turn.observed_state,
    }
    if turn.canonical_state_summary is not None:
        payload["canonical_state_anchor"] = turn.canonical_state_summary
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def _parse_tool_call(content: str) -> DriftToolCall:
    try:
        payload = json.loads(content)
        return DriftToolCall.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError):
        return DriftToolCall(
            tool="invalid_model_output",
            arguments={
                "content_fingerprint": canonical_fingerprint(content),
                "content_length": len(content),
            },
        )


def _record_completion(
    completion: CompletionResult,
    checkpoint: int,
    telemetry: _ProviderTelemetry,
    topology: ModelTopology,
) -> None:
    telemetry.prompt_tokens += completion.prompt_tokens
    telemetry.completion_tokens += completion.completion_tokens
    performance = completion.performance
    telemetry.latency_ms += (
        performance.generation_time_ms or performance.time_to_first_token_ms or 0
    )
    routing = aggregate_routing(completion.routing, topology)
    layer_totals = [sum(layer) for layer in routing.selection_counts]
    telemetry.routing.append(
        {
            "checkpoint": checkpoint,
            "context_id": completion.context_id,
            "context_fingerprint": completion.context_fingerprint,
            "topology_fingerprint": completion.topology_fingerprint,
            "total_routed_slots": routing.total_routed_slots,
            "selection_count_by_layer": layer_totals,
            "selection_counts": routing.selection_counts,
            "routing_mass": routing.routing_mass,
            "routing_fingerprint": canonical_fingerprint(
                {
                    "selection_counts": routing.selection_counts,
                    "routing_mass": routing.routing_mass,
                }
            ),
        }
    )


def _numeric_metrics(result: DriftRunResult) -> dict[str, float | int | bool | None]:
    metrics = result.metrics
    return {
        "final_success": metrics.final_success,
        "transition_accuracy": metrics.transition_accuracy,
        "first_divergence_checkpoint": metrics.first_divergence_checkpoint,
        "state_fidelity_auc": metrics.area_under_state_fidelity_curve,
        "recovery_rate": metrics.recovery_rate,
        "invariant_violations": metrics.invariant_violations,
        "invalid_tool_calls": metrics.invalid_tool_calls,
        "collateral_mutations": metrics.collateral_mutations,
        "rollback_correctness": metrics.rollback_correctness,
        "divergence_growth_slope": metrics.divergence_growth_slope,
    }


def _routing_by_checkpoint(unit: RunUnit) -> dict[int, dict[str, Any]]:
    if unit.result is None:
        return {}
    records = unit.result.get("routing", [])
    if not isinstance(records, list):
        return {}
    return {
        int(record["checkpoint"]): record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("checkpoint"), int)
        and isinstance(record.get("selection_counts"), list)
        and isinstance(record.get("routing_mass"), list)
    }


def _routing_distance(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float | None, float | None]:
    baseline_counts = _flatten_numeric(baseline.get("selection_counts"))
    candidate_counts = _flatten_numeric(candidate.get("selection_counts"))
    baseline_mass = _flatten_numeric(baseline.get("routing_mass"))
    candidate_mass = _flatten_numeric(candidate.get("routing_mass"))
    return (
        _histogram_overlap(baseline_counts, candidate_counts),
        _jensen_shannon_divergence(baseline_mass, candidate_mass),
    )


def _largest_routing_shifts(
    baseline: dict[str, Any], candidate: dict[str, Any], limit: int = 10
) -> list[dict[str, float | int]]:
    baseline_layers = baseline.get("selection_counts")
    candidate_layers = candidate.get("selection_counts")
    if not isinstance(baseline_layers, list) or not isinstance(candidate_layers, list):
        return []
    baseline_values = _flatten_numeric(baseline_layers)
    candidate_values = _flatten_numeric(candidate_layers)
    if len(baseline_values) != len(candidate_values):
        return []
    baseline_total = sum(baseline_values)
    candidate_total = sum(candidate_values)
    if baseline_total <= 0 or candidate_total <= 0:
        return []
    shifts: list[dict[str, float | int]] = []
    offset = 0
    for layer_index, (baseline_layer, candidate_layer) in enumerate(
        zip(baseline_layers, candidate_layers, strict=True)
    ):
        if not isinstance(baseline_layer, list) or not isinstance(
            candidate_layer, list
        ):
            return []
        if len(baseline_layer) != len(candidate_layer):
            return []
        for expert_index in range(len(baseline_layer)):
            baseline_share = baseline_values[offset] / baseline_total
            candidate_share = candidate_values[offset] / candidate_total
            shifts.append(
                {
                    "layer": layer_index,
                    "expert": expert_index,
                    "baseline_share": baseline_share,
                    "candidate_share": candidate_share,
                    "delta": candidate_share - baseline_share,
                }
            )
            offset += 1
    return sorted(shifts, key=lambda item: abs(float(item["delta"])), reverse=True)[
        :limit
    ]


def _flatten_numeric(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    flattened: list[float] = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(_flatten_numeric(item))
        elif isinstance(item, int | float) and not isinstance(item, bool):
            flattened.append(float(item))
        else:
            return []
    return flattened


def _histogram_overlap(baseline: list[float], candidate: list[float]) -> float | None:
    if not baseline or len(baseline) != len(candidate):
        return None
    baseline_total = sum(baseline)
    candidate_total = sum(candidate)
    if baseline_total <= 0 or candidate_total <= 0:
        return None
    return sum(
        min(left / baseline_total, right / candidate_total)
        for left, right in zip(baseline, candidate, strict=True)
    )


def _jensen_shannon_divergence(
    baseline: list[float], candidate: list[float]
) -> float | None:
    if not baseline or len(baseline) != len(candidate):
        return None
    baseline_total = sum(baseline)
    candidate_total = sum(candidate)
    if baseline_total <= 0 or candidate_total <= 0:
        return None
    left = [value / baseline_total for value in baseline]
    right = [value / candidate_total for value in candidate]
    midpoint = [(a + b) / 2 for a, b in zip(left, right, strict=True)]

    def divergence(values: list[float]) -> float:
        return sum(
            value * math.log2(value / average)
            for value, average in zip(values, midpoint, strict=True)
            if value > 0 and average > 0
        )

    return (divergence(left) + divergence(right)) / 2


def _optional_mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), int | float)
        and not isinstance(record.get(key), bool)
    ]
    return mean(values) if values else None


def _drift_result(unit: RunUnit) -> DriftRunResult:
    return DriftRunResult.model_validate(unit.result)


def _condition_mean(results: Any, condition: DriftCondition) -> float | None:
    values = [
        result.metrics.area_under_state_fidelity_curve
        for result in results
        if result.condition is condition
    ]
    return mean(values) if values else None


def _optional_difference(
    minuend: float | None, subtrahend: float | None
) -> float | None:
    if minuend is None or subtrahend is None:
        return None
    return minuend - subtrahend


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0
