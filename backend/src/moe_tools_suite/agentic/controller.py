from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Protocol
from uuid import uuid4

import numpy as np

from ..domain import (
    CriterionResult,
    CriterionVisibility,
    DeterministicScorerKind,
    EvaluationContract,
    ExecutionPolicy,
    GenerationConfig,
    JudgeEvaluationRequest,
    JudgeMode,
    LLMJudgeResult,
    ModelSession,
    ModelState,
    ModelTopology,
)
from ..evaluation import evaluate_output, validate_cost_policy_pricing
from ..judges import (
    JudgeBudgetExceeded,
    JudgeProviderError,
    judge_request_cost_upper_bound,
)
from ..persistence import SqliteStore
from ..runtime import ModelRuntime
from ..settings import Settings
from .artifacts import AgentArtifactStore
from .atif import (
    BASH_JSON_SYSTEM_PROMPT,
    AtifMetrics,
    AtifSource,
    AtifStep,
    AtifTrajectory,
    append_agent_step,
    build_initial_trajectory,
    finalize_trajectory,
    make_command_observation,
    make_shell_tool_call,
)
from .catalog import AgentTaskCatalog, agent_run_contract_fingerprint
from .domain import (
    AgentBudgets,
    AgentEvaluationContractView,
    AgentRun,
    AgentRunExport,
    AgentRunPublicDetail,
    AgentRunStatus,
    AgentRunView,
    AgentTask,
    AgentTrajectoryView,
    AgentTrial,
    AgentTrialArtifactsView,
    AgentTrialStatus,
    AgentTrialSummary,
    ArtifactExport,
    CommandResult,
    CreateAgentRunRequest,
    InferenceCall,
    InferenceCallSummary,
    PatchStats,
    ProviderPreflight,
    ReasoningMode,
    SandboxFile,
    SandboxOwnership,
    SandboxProviderInfo,
    SandboxSession,
    SandboxSessionState,
    TerminationCause,
    TrajectoryStepType,
    TrajectoryStepView,
    TrialArtifact,
    TrialArtifactManifest,
    TrialPerformance,
    TrialRoutingSummary,
    VerifierResult,
    VerifierStatus,
    VerifierView,
    default_agent_evaluation_contract,
    utc_now,
)
from .gateway import InstrumentedAgentGateway, parse_agent_action
from .providers import (
    DaytonaProviderConfig,
    DaytonaSandboxProvider,
    FakeSandboxProvider,
    SandboxProvider,
    SandboxProviderError,
)

logger = logging.getLogger(__name__)
_ACTIVE_TRIAL_STATES = {
    AgentTrialStatus.QUEUED,
    AgentTrialStatus.PROVISIONING,
    AgentTrialStatus.RUNNING,
    AgentTrialStatus.VERIFYING,
    AgentTrialStatus.CLEANING,
}


@dataclass(frozen=True, slots=True)
class _SubmissionSnapshot:
    files: tuple[SandboxFile, ...]
    deleted_paths: frozenset[str]


class AgentJudge(Protocol):
    async def evaluate(
        self,
        request: JudgeEvaluationRequest,
        *,
        max_incurred_cost_usd: float | None = None,
    ) -> LLMJudgeResult: ...


class AgenticController:
    """Durable coding-trial orchestration behind a replaceable sandbox seam."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: SqliteStore,
        runtime: ModelRuntime,
        topology: ModelTopology,
        model_id: str,
        judge_service: AgentJudge,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime = runtime
        self.topology = topology
        self.model_id = model_id
        self.judge_service = judge_service
        self.catalog = AgentTaskCatalog()
        self.artifacts = AgentArtifactStore(settings.data_dir)
        fake = FakeSandboxProvider(max_output_bytes=settings.agent_max_output_bytes)
        daytona = DaytonaSandboxProvider(
            DaytonaProviderConfig(
                api_key=settings.daytona_api_key,
                api_url=settings.daytona_api_url,
                target=settings.daytona_target,
                create_timeout_seconds=settings.daytona_create_timeout_seconds,
                max_output_bytes=settings.agent_max_output_bytes,
            )
        )
        self.providers: dict[str, SandboxProvider] = {
            fake.provider_id: fake,
            daytona.provider_id: daytona,
        }
        self._cleanup_lock = asyncio.Lock()
        self.runs = store.load_agent_runs()
        self.trials = store.load_agent_trials()
        self.inferences = store.load_agent_inferences()
        self.sandboxes = store.load_sandbox_sessions()
        self.trajectories: dict[str, AtifTrajectory] = {}
        self.routings: dict[str, TrialRoutingSummary] = {}
        self.gateway = InstrumentedAgentGateway(
            runtime=runtime,
            topology=topology,
            store=store,
            artifacts=self.artifacts,
        )
        self._load_artifacts()
        self._reconcile_interrupted()

    def list_agents(self):
        return self.catalog.list_agents()

    def list_task_packs(self):
        return self.catalog.list_packs()

    def list_tasks(self, pack_id: str):
        return self.catalog.list_tasks(pack_id)

    def list_providers(self) -> list[SandboxProviderInfo]:
        return [self.providers[key].info() for key in sorted(self.providers)]

    async def preflight_provider(self, provider_id: str) -> ProviderPreflight:
        return await self._provider(provider_id).preflight()

    def validate_run_request(
        self,
        request: CreateAgentRunRequest,
        model_session: ModelSession | None,
    ) -> tuple[object, list[AgentTask], object]:
        if model_session is None or model_session.id != request.model_session_id:
            raise ValueError("agent run must reference the current model session")
        if model_session.state is not ModelState.READY:
            raise ValueError("load a ready model before starting an agent run")
        pack, tasks, agent = self.catalog.validate_run_request(request)
        provider = self._provider(request.sandbox_provider_id)
        info = provider.info()
        if not info.configured:
            raise ValueError(f"sandbox provider {info.label} is not configured")
        if info.status.value not in {"ready", "unchecked"}:
            raise ValueError(f"sandbox provider {info.label} is not ready")
        if request.sandbox_provider_id != "fake" and info.status.value != "ready":
            raise ValueError("test the Daytona connection before starting a run")
        return pack, tasks, agent

    def normalize_run_request(
        self, request: CreateAgentRunRequest
    ) -> CreateAgentRunRequest:
        contract = request.evaluation_contract or default_agent_evaluation_contract()
        unsupported = {
            criterion.kind
            for criterion in contract.criteria
            if criterion.kind
            not in {
                DeterministicScorerKind.BENCHMARK_DEFAULT,
                DeterministicScorerKind.UNGRADED,
                DeterministicScorerKind.VERIFIER,
            }
        }
        if unsupported:
            values = ", ".join(sorted(kind.value for kind in unsupported))
            raise ValueError(
                "agentic evaluation supports trusted or sandbox verifier criteria "
                f"and ungraded criteria, not {values}"
            )
        if contract.judge is not None and contract.judge.mode is not JudgeMode.SINGLE:
            raise ValueError("agentic LLM judging currently supports single mode only")

        if request.execution_policy is None:
            policy = ExecutionPolicy(
                attempts=request.attempts,
                concurrency=1,
                timeout_seconds=min(
                    86_400,
                    request.budgets.timeout_seconds
                    * len(request.task_ids)
                    * request.attempts,
                ),
                per_item_timeout_seconds=request.budgets.timeout_seconds,
                max_turns=request.budgets.max_turns,
                max_commands=request.budgets.max_commands,
                max_tokens=None,
            )
        else:
            policy = request.execution_policy
            if policy.concurrency != 1:
                raise ValueError("agentic execution currently requires concurrency=1")
            if policy.attempts > 5:
                raise ValueError("agentic execution supports at most five attempts")
        validate_cost_policy_pricing(contract, policy)
        budget_values = request.budgets.model_dump(mode="python")
        budget_values.update(
            {
                "max_turns": policy.max_turns or request.budgets.max_turns,
                "max_commands": policy.max_commands or request.budgets.max_commands,
                "max_tokens": policy.max_tokens or request.budgets.max_tokens,
                "timeout_seconds": policy.per_item_timeout_seconds,
            }
        )
        budgets = AgentBudgets.model_validate(budget_values)
        return request.model_copy(
            update={
                "attempts": policy.attempts,
                "budgets": budgets,
                "evaluation_contract": contract,
                "execution_policy": policy,
            }
        )

    def create_run(
        self,
        *,
        run_id: str,
        job_id: str,
        request: CreateAgentRunRequest,
        model_session: ModelSession,
    ) -> AgentRun:
        request = self.normalize_run_request(request)
        pack, tasks, agent = self.validate_run_request(request, model_session)
        fingerprint = agent_run_contract_fingerprint(
            request=request,
            pack=pack,
            tasks=tasks,
            agent=agent,
        )
        profile_fingerprint = _profile_fingerprint(model_session)
        run = AgentRun(
            id=run_id,
            job_id=job_id,
            task_pack_id=pack.id,
            task_pack_name=pack.name,
            task_pack_revision=pack.revision,
            task_pack_content_hash=pack.content_hash,
            task_ids=request.task_ids,
            model_session_id=model_session.id,
            profile_id=model_session.profile_id,
            profile_fingerprint=profile_fingerprint,
            agent_id=agent.id,
            agent_revision=agent.revision,
            sandbox_provider_id=request.sandbox_provider_id,
            attempts=request.attempts,
            generation=request.generation,
            budgets=request.budgets,
            reasoning_mode=request.reasoning_mode,
            evaluation_contract=request.evaluation_contract,
            execution_policy=request.execution_policy,
            contract_provenance_status="known",
            contract_fingerprint=fingerprint,
            total_trials=len(tasks) * request.attempts,
        )
        self.runs[run.id] = run
        self.store.save_agent_run(run)
        for task in tasks:
            for attempt in range(1, request.attempts + 1):
                trial = AgentTrial(
                    id=str(uuid4()),
                    run_id=run.id,
                    task_id=task.id,
                    task_title_snapshot=task.title,
                    attempt=attempt,
                    seed=request.seed + attempt - 1,
                    model_session_id=model_session.id,
                )
                self.trials[trial.id] = trial
                self.store.save_agent_trial(trial)
        return run

    async def execute_run(
        self,
        run_id: str,
        *,
        model_session: ModelSession,
        on_progress: Callable[[int, int], None],
        should_cancel: Callable[[], bool],
    ) -> AgentRun:
        run = self.runs[run_id]
        run.status = AgentRunStatus.RUNNING
        run.started_at = utc_now()
        self.store.save_agent_run(run)
        trials = self._run_trials(run_id)
        run_started = time.monotonic()
        try:
            await self.cleanup_interrupted_sandboxes(run.sandbox_provider_id)
            if any(
                sandbox.provider_id == run.sandbox_provider_id
                and sandbox.state is SandboxSessionState.CLEANUP_PENDING
                for sandbox in self.sandboxes.values()
            ):
                raise RuntimeError(
                    "persisted sandbox cleanup must succeed before a new run"
                )
            for trial in trials:
                if should_cancel():
                    self._cancel_unstarted_trials(run_id)
                    break
                stop_cause = _agent_policy_stop_cause(
                    run,
                    trials,
                    elapsed_seconds=time.monotonic() - run_started,
                )
                if stop_cause is not None:
                    self._cancel_unstarted_trials(run_id, cause=stop_cause)
                    break
                task = self.catalog.get_task(run.task_pack_id, trial.task_id)
                elapsed_seconds = time.monotonic() - run_started
                used_tokens = sum(
                    item.prompt_tokens + item.completion_tokens for item in trials
                )
                remaining_tokens = (
                    run.execution_policy.max_tokens - used_tokens
                    if run.execution_policy.max_tokens is not None
                    else run.budgets.max_tokens
                )
                remaining_seconds = max(
                    0.001,
                    run.execution_policy.timeout_seconds - elapsed_seconds,
                )
                effective_budgets = run.budgets.model_copy(
                    update={
                        "max_tokens": min(
                            run.budgets.max_tokens,
                            max(1, remaining_tokens),
                        ),
                        "timeout_seconds": min(
                            run.budgets.timeout_seconds,
                            remaining_seconds,
                        ),
                    }
                )
                await self._execute_trial(
                    run=run,
                    trial=trial,
                    task=task,
                    model_session=model_session,
                    budgets=effective_budgets,
                    should_cancel=should_cancel,
                )
                self._update_run_metrics(run)
                on_progress(run.completed_trials, run.total_trials)
                if should_cancel():
                    self._cancel_unstarted_trials(run_id)
                    break
        except Exception as error:
            logger.exception("agent run %s failed", run.id)
            run.status = AgentRunStatus.FAILED
            run.error = _error_text(error)
        else:
            self._update_run_metrics(run)
            if should_cancel() or any(
                trial.status is AgentTrialStatus.CANCELLED for trial in trials
            ):
                run.status = AgentRunStatus.CANCELLED
            else:
                run.status = AgentRunStatus.COMPLETED
        run.completed_at = utc_now()
        self.store.save_agent_run(run)
        return run

    async def cleanup_interrupted_sandboxes(
        self,
        provider_id: str | None = None,
    ) -> int:
        """Delete durable non-fake sandboxes using provider-verified labels."""

        async with self._cleanup_lock:
            return await self._cleanup_interrupted_sandboxes(provider_id)

    async def _cleanup_interrupted_sandboxes(
        self,
        provider_id: str | None = None,
    ) -> int:

        deleted = 0
        pending = [
            sandbox
            for sandbox in self.sandboxes.values()
            if sandbox.provider_id != "fake"
            and sandbox.state is SandboxSessionState.CLEANUP_PENDING
            and (provider_id is None or sandbox.provider_id == provider_id)
        ]
        for sandbox in pending:
            try:
                provider = self._provider(sandbox.provider_id)
            except KeyError:
                sandbox.cleanup_error = "sandbox provider is unavailable for cleanup"
                sandbox.updated_at = utc_now()
                self.store.save_sandbox_session(sandbox)
                continue
            sandbox.state = SandboxSessionState.DELETING
            sandbox.cleanup_error = None
            sandbox.updated_at = utc_now()
            self.store.save_sandbox_session(sandbox)
            try:
                await provider.cleanup_owned(
                    sandbox.external_id,
                    sandbox.ownership,
                )
            except SandboxProviderError as error:
                sandbox.state = (
                    SandboxSessionState.ERROR
                    if error.code == "sandbox_ownership_mismatch"
                    else SandboxSessionState.CLEANUP_PENDING
                )
                sandbox.cleanup_error = error.message
            except Exception as error:
                sandbox.state = SandboxSessionState.CLEANUP_PENDING
                sandbox.cleanup_error = (
                    f"sandbox cleanup failed ({error.__class__.__name__[:64]})"
                )
            else:
                sandbox.state = SandboxSessionState.DELETED
                sandbox.deleted_at = utc_now()
                sandbox.cleanup_error = None
                deleted += 1
            sandbox.updated_at = utc_now()
            self.store.save_sandbox_session(sandbox)
        return deleted

    def request_cancel(self, run_id: str) -> AgentRunView:
        run = self.runs[run_id]
        if run.status in {
            AgentRunStatus.QUEUED,
            AgentRunStatus.RUNNING,
        }:
            run.status = AgentRunStatus.CANCELLING
            self.store.save_agent_run(run)
        return self.run_view(run_id)

    def cancel_queued_run(self, run_id: str) -> AgentRunView:
        run = self.runs[run_id]
        self._cancel_unstarted_trials(run_id)
        self._update_run_metrics(run)
        run.status = AgentRunStatus.CANCELLED
        run.completed_at = utc_now()
        self.store.save_agent_run(run)
        return self.run_view(run_id)

    def list_run_views(self) -> list[AgentRunView]:
        return [
            self.run_view(run.id)
            for run in sorted(
                self.runs.values(), key=lambda item: item.created_at, reverse=True
            )
        ]

    def run_view(self, run_id: str) -> AgentRunView:
        run = self.runs[run_id]
        summaries = [self._trial_summary(trial) for trial in self._run_trials(run_id)]
        active = next(
            (item.id for item in summaries if item.status in _ACTIVE_TRIAL_STATES),
            None,
        )
        return AgentRunView(
            id=run.id,
            job_id=run.job_id,
            task_pack_id=run.task_pack_id,
            task_pack_name=run.task_pack_name,
            task_pack_revision=run.task_pack_revision,
            task_pack_content_hash=run.task_pack_content_hash,
            model_session_id=run.model_session_id,
            profile_id=run.profile_id,
            profile_fingerprint=run.profile_fingerprint,
            agent_id=run.agent_id,
            agent_revision=run.agent_revision,
            sandbox_provider_id=run.sandbox_provider_id,
            generation=run.generation,
            budgets=run.budgets,
            contract_provenance_status=run.contract_provenance_status,
            reasoning_mode=(
                run.reasoning_mode
                if run.contract_provenance_status == "known"
                else None
            ),
            evaluation_contract=(
                _public_evaluation_contract(run.evaluation_contract)
                if run.contract_provenance_status == "known"
                else None
            ),
            evaluation_contract_fingerprint=(
                run.evaluation_contract.fingerprint
                if run.contract_provenance_status == "known"
                else None
            ),
            hidden_evaluation_criteria_count=(
                sum(
                    criterion.visibility is not CriterionVisibility.PUBLIC
                    for criterion in run.evaluation_contract.criteria
                )
                if run.contract_provenance_status == "known"
                else None
            ),
            execution_policy=(
                run.execution_policy
                if run.contract_provenance_status == "known"
                else None
            ),
            compatibility_fingerprint=run.contract_fingerprint,
            status=run.status,
            task_ids=run.task_ids,
            total_trials=run.total_trials,
            completed_trials=run.completed_trials,
            passed_trials=run.passed_trials,
            mean_reward=run.mean_reward,
            active_trial_id=active,
            trials=summaries,
            error=run.error,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    def get_trial(self, trial_id: str) -> AgentTrialSummary:
        return self._trial_summary(self.trials[trial_id])

    def get_trajectory(self, trial_id: str) -> AgentTrajectoryView:
        trajectory = self.trajectories.get(trial_id)
        if trajectory is None:
            raise KeyError(trial_id)
        inference_by_step = {
            inference.trajectory_step_id: inference
            for inference in self.inferences.values()
            if inference.trial_id == trial_id
        }
        events: list[TrajectoryStepView] = []
        agent_turn = 0
        for step in trajectory.steps:
            timestamp = step.timestamp or self.trials[trial_id].created_at
            if step.source is not AtifSource.AGENT:
                verifier = step.message.startswith("[verifier]")
                step_type = (
                    TrajectoryStepType.VERIFIER
                    if verifier
                    else (
                        TrajectoryStepType.USER
                        if step.source is AtifSource.USER
                        else TrajectoryStepType.SYSTEM
                    )
                )
                events.append(
                    TrajectoryStepView(
                        id=f"atif-{step.step_id}",
                        sequence=len(events) + 1,
                        timestamp=timestamp,
                        type=step_type,
                        title=(
                            "Verifier"
                            if verifier
                            else (
                                "Task instruction"
                                if step_type is TrajectoryStepType.USER
                                else "Agent instructions"
                            )
                        ),
                        content=step.message.removeprefix("[verifier]\n"),
                        phase=(
                            "verifier"
                            if verifier
                            else (
                                "task"
                                if step_type is TrajectoryStepType.USER
                                else "system"
                            )
                        ),
                    )
                )
                continue
            agent_turn += 1
            inference = inference_by_step.get(str(step.step_id))
            inference_view = (
                _inference_summary(inference, self.model_id, self.topology.num_layers)
                if inference is not None
                else None
            )
            reasoning = step.reasoning_content or (
                inference.reasoning_content if inference is not None else None
            )
            if reasoning:
                events.append(
                    TrajectoryStepView(
                        id=f"atif-{step.step_id}-reasoning",
                        sequence=len(events) + 1,
                        timestamp=timestamp,
                        type=TrajectoryStepType.REASONING,
                        title="Explicit model reasoning",
                        content=reasoning,
                        phase="reasoning",
                        turn=agent_turn,
                        reasoning_visibility="explicit",
                    )
                )
            events.append(
                TrajectoryStepView(
                    id=f"atif-{step.step_id}-model",
                    sequence=len(events) + 1,
                    timestamp=timestamp,
                    type=TrajectoryStepType.ASSISTANT,
                    title="Model response",
                    content=step.message,
                    phase="response",
                    turn=agent_turn,
                    reasoning_visibility="explicit" if reasoning else "none",
                    inference=inference_view,
                )
            )
            for call in step.tool_calls or []:
                command = str(call.arguments.get("command", ""))
                events.append(
                    TrajectoryStepView(
                        id=f"atif-{step.step_id}-tool-{call.tool_call_id}",
                        sequence=len(events) + 1,
                        timestamp=timestamp,
                        type=TrajectoryStepType.TOOL,
                        title=call.function_name,
                        content="",
                        tool_name=call.function_name,
                        command=command,
                        phase="command",
                        turn=agent_turn,
                    )
                )
            for observation in step.observation.results if step.observation else []:
                extra = observation.extra or {}
                events.append(
                    TrajectoryStepView(
                        id=(
                            f"atif-{step.step_id}-observation-"
                            f"{observation.source_call_id or len(events)}"
                        ),
                        sequence=len(events) + 1,
                        timestamp=observation.timestamp or timestamp,
                        type=TrajectoryStepType.OBSERVATION,
                        title="Command output",
                        content=observation.content,
                        phase="observation",
                        turn=agent_turn,
                        stream=(
                            extra.get("stream")
                            if extra.get("stream") in {"stdout", "stderr"}
                            else None
                        ),
                        exit_code=_optional_int(extra.get("exit_code")),
                        duration_ms=_optional_float(extra.get("duration_ms")),
                        truncated=bool(extra.get("truncated", False)),
                    )
                )
        return AgentTrajectoryView(
            trial_id=trial_id,
            schema_version=trajectory.schema_version,
            reasoning_mode=(
                self.runs[self.trials[trial_id].run_id].reasoning_mode
                if self.runs[self.trials[trial_id].run_id].contract_provenance_status
                == "known"
                else None
            ),
            steps=events,
            updated_at=_trial_updated_at(self.trials[trial_id]),
        )

    def get_routing(self, trial_id: str) -> TrialRoutingSummary:
        try:
            return self.routings[trial_id]
        except KeyError as error:
            raise KeyError(trial_id) from error

    def get_artifacts(self, trial_id: str) -> AgentTrialArtifactsView:
        trial = self.trials[trial_id]
        patch = self.artifacts.read_text(trial_id, "patch.diff")
        patch_sha = hashlib.sha256(patch.encode()).hexdigest() if patch else None
        verifier = trial.verifier
        verifier_view = None
        if verifier is not None:
            output = verifier.stdout
            if verifier.stderr:
                output = f"{output}\n{verifier.stderr}".strip()
            verifier_view = VerifierView(
                status=(
                    VerifierStatus.PASSED if verifier.passed else VerifierStatus.FAILED
                ),
                reward=1.0 if verifier.passed else 0.0,
                summary=("All checks passed" if verifier.passed else "Checks failed"),
                output=output,
                exit_code=verifier.exit_code,
                duration_ms=verifier.duration_ms,
            )
        stats = trial.patch_stats or PatchStats()
        return AgentTrialArtifactsView(
            trial_id=trial_id,
            patch=patch or None,
            patch_sha256=patch_sha,
            files_changed=stats.files_changed,
            additions=stats.insertions,
            deletions=stats.deletions,
            verifier=verifier_view,
            evaluation=trial.evaluation,
            exports=[
                ArtifactExport(
                    name="ATIF trajectory",
                    media_type="application/json",
                    download_url=f"/api/trials/{trial_id}/export/atif",
                )
            ],
        )

    def export_run(self, run_id: str) -> AgentRunExport:
        trials = self._run_trials(run_id)
        return AgentRunExport(
            detail=AgentRunPublicDetail(
                run=self.run_view(run_id),
                trials=[self._trial_summary(trial) for trial in trials],
            ),
            trajectories={
                trial.id: self.trajectories[trial.id].model_dump(
                    mode="json", exclude_none=True
                )
                for trial in trials
                if trial.id in self.trajectories
            },
            routing={
                trial.id: self.routings[trial.id]
                for trial in trials
                if trial.id in self.routings
            },
            manifests={
                trial.id: trial.artifact_manifest
                for trial in trials
                if trial.artifact_manifest is not None
            },
        )

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()

    async def _execute_trial(
        self,
        *,
        run: AgentRun,
        trial: AgentTrial,
        task: AgentTask,
        model_session: ModelSession,
        budgets: AgentBudgets,
        should_cancel: Callable[[], bool],
    ) -> None:
        wall_started = time.perf_counter()
        trial_deadline = time.monotonic() + budgets.timeout_seconds
        provisioning_started = wall_started
        provisioning_time_ms = 0.0
        sandbox_time_ms = 0.0
        provider = self._provider(run.sandbox_provider_id)
        agent = self.catalog.get_agent(run.agent_id)
        trajectory = build_initial_trajectory(
            trial_id=trial.id,
            agent=agent,
            model_name=self.model_id,
            instruction=task.instruction,
            extra={
                "run_id": run.id,
                "task_pack_id": run.task_pack_id,
                "task_id": task.id,
                "model_session_id": model_session.id,
                "profile_id": model_session.profile_id,
            },
        )
        self.trajectories[trial.id] = trajectory
        self._save_trajectory(trial.id)
        trial.status = AgentTrialStatus.PROVISIONING
        trial.started_at = utc_now()
        self.store.save_agent_trial(trial)
        ownership = SandboxOwnership(
            controller_id=self.settings.agent_controller_id,
            run_id=run.id,
            trial_id=trial.id,
        )
        sandbox_session = SandboxSession(
            id=str(uuid4()),
            trial_id=trial.id,
            provider_id=provider.provider_id,
            state=SandboxSessionState.CREATING,
            ownership=ownership,
            spec=task.sandbox_spec(),
        )
        self.sandboxes[sandbox_session.id] = sandbox_session
        trial.sandbox_session_id = sandbox_session.id
        self.store.save_sandbox_session(sandbox_session)
        self.store.save_agent_trial(trial)
        handle = None
        cancelled = False
        error: Exception | None = None
        aggregated_counts = np.zeros(
            (self.topology.num_layers, self.topology.num_experts), dtype=np.int64
        )
        aggregated_mass = np.zeros(
            (self.topology.num_layers, self.topology.num_experts), dtype=np.float64
        )
        routing_artifacts = []
        patch = ""
        verifier_criteria: dict[str, CriterionResult] = {}
        try:
            handle = await provider.create(task.sandbox_spec(), ownership)
            sandbox_session.external_id = handle.id
            sandbox_session.state = SandboxSessionState.READY
            sandbox_session.effective_network_policy = task.network_policy
            sandbox_session.updated_at = utc_now()
            self.store.save_sandbox_session(sandbox_session)
            protected_verifier_paths = set(task.verifier_file_paths) | set(
                task.oracle_file_paths
            )
            for file in task.files:
                if file.path not in protected_verifier_paths:
                    await provider.upload(handle, file)
            self._configure_fake_scripts(provider, task)
            provisioning_time_ms = (time.perf_counter() - provisioning_started) * 1000
            trial.status = AgentTrialStatus.RUNNING
            self.store.save_agent_trial(trial)
            messages = [
                {"role": "system", "content": BASH_JSON_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ]
            scripted = self._scripted_actions(provider, task)
            started = time.monotonic()
            for turn_index in range(budgets.max_turns):
                if should_cancel():
                    cancelled = True
                    trial.termination_cause = TerminationCause.CANCELLED
                    break
                elapsed = time.monotonic() - started
                if elapsed >= budgets.timeout_seconds:
                    trial.termination_cause = TerminationCause.TIME_LIMIT
                    break
                used_tokens = trial.prompt_tokens + trial.completion_tokens
                if used_tokens >= budgets.max_tokens:
                    trial.termination_cause = TerminationCause.TOKEN_LIMIT
                    break
                request_messages = _bounded_messages(messages)
                prompt_token_reserve = _prompt_token_upper_bound(request_messages)
                output_token_budget = (
                    budgets.max_tokens - used_tokens - prompt_token_reserve
                )
                if output_token_budget < 1:
                    trial.termination_cause = TerminationCause.TOKEN_LIMIT
                    break
                step_id = len(trajectory.steps) + 1
                scripted_content = (
                    scripted[turn_index] if turn_index < len(scripted) else None
                )
                generation = _bounded_generation(
                    run.generation,
                    output_token_budget,
                    seed=trial.seed,
                ).model_copy(
                    update={
                        "enable_thinking": run.reasoning_mode is not ReasoningMode.OFF
                    }
                )
                try:
                    async with asyncio.timeout(
                        max(0.001, budgets.timeout_seconds - elapsed)
                    ):
                        gateway_result = await self.gateway.infer(
                            trial_id=trial.id,
                            trajectory_step_id=str(step_id),
                            model_session=model_session,
                            messages=request_messages,
                            generation=generation,
                            request_key=f"agent:{trial.id}:{turn_index}",
                            scripted_content=scripted_content,
                        )
                except TimeoutError:
                    trial.termination_cause = TerminationCause.TIME_LIMIT
                    break
                self.inferences[gateway_result.inference.id] = gateway_result.inference
                routing_artifacts.append(gateway_result.inference.routing_artifact)
                aggregated_counts += gateway_result.aggregated.selection_counts
                aggregated_mass += gateway_result.aggregated.routing_mass
                trial.turns += 1
                trial.prompt_tokens += gateway_result.inference.prompt_tokens
                trial.completion_tokens += gateway_result.inference.completion_tokens
                try:
                    action = parse_agent_action(gateway_result.content)
                except ValueError as parse_error:
                    trajectory = append_agent_step(
                        trajectory,
                        message=gateway_result.content,
                        model_name=self.model_id,
                        metrics=_atif_metrics(gateway_result.inference),
                        reasoning_content=gateway_result.reasoning,
                    )
                    messages.extend(
                        [
                            {"role": "assistant", "content": gateway_result.content},
                            {
                                "role": "user",
                                "content": (
                                    "Invalid action: "
                                    f"{parse_error}. Return exactly one JSON object."
                                ),
                            },
                        ]
                    )
                    self.trajectories[trial.id] = trajectory
                    self._save_live_trial(trial)
                    continue
                if action.action == "finish":
                    trajectory = append_agent_step(
                        trajectory,
                        message=gateway_result.content,
                        model_name=self.model_id,
                        metrics=_atif_metrics(gateway_result.inference),
                        reasoning_content=gateway_result.reasoning,
                    )
                    trial.termination_cause = TerminationCause.AGENT_FINISHED
                    self.trajectories[trial.id] = trajectory
                    self._save_live_trial(trial)
                    break
                if trial.commands >= budgets.max_commands:
                    trial.termination_cause = TerminationCause.TURN_LIMIT
                    break
                command = action.command or ""
                tool_call_id = str(uuid4())
                command_result = await provider.exec(
                    handle,
                    command,
                    cwd=task.working_directory,
                    timeout_seconds=max(1, min(120, int(task.timeout_seconds))),
                )
                sandbox_time_ms += command_result.duration_ms
                trial.commands += 1
                trajectory = append_agent_step(
                    trajectory,
                    message=gateway_result.content,
                    model_name=self.model_id,
                    metrics=_atif_metrics(gateway_result.inference),
                    reasoning_content=gateway_result.reasoning,
                    tool_calls=[
                        make_shell_tool_call(
                            tool_call_id=tool_call_id,
                            command=command,
                        )
                    ],
                    observation=make_command_observation(
                        tool_call_id=tool_call_id,
                        result=command_result,
                    ),
                )
                output = _command_output(command_result)
                messages.extend(
                    [
                        {"role": "assistant", "content": gateway_result.content},
                        {
                            "role": "user",
                            "content": (
                                "Command exited "
                                f"{command_result.exit_code}.\n"
                                f"{_bounded_prompt_output(output)}"
                            ),
                        },
                    ]
                )
                self.trajectories[trial.id] = trajectory
                self._save_live_trial(trial)
            else:
                trial.termination_cause = TerminationCause.TURN_LIMIT

            if not cancelled and should_cancel():
                cancelled = True
                trial.termination_cause = TerminationCause.CANCELLED
            if (
                not cancelled
                and trial.termination_cause is not TerminationCause.TIME_LIMIT
                and time.monotonic() >= trial_deadline
            ):
                trial.termination_cause = TerminationCause.TIME_LIMIT
            if (
                not cancelled
                and trial.termination_cause is not TerminationCause.TIME_LIMIT
            ):
                trial.status = AgentTrialStatus.VERIFYING
                self.store.save_agent_trial(trial)
                submission = await self._capture_submission(provider, handle, task)
                patch = self._submission_patch(task, submission)
                trial.patch_stats = _patch_stats(patch)
                self.artifacts.save_text(trial.id, "patch.diff", patch)
                verifier_command = task.verifier_command
                verifier_timeout = _remaining_verifier_timeout(
                    trial_deadline,
                    min(300, task.timeout_seconds),
                )
                if should_cancel():
                    cancelled = True
                    trial.termination_cause = TerminationCause.CANCELLED
                elif verifier_timeout is None:
                    trial.termination_cause = TerminationCause.TIME_LIMIT
                else:
                    try:
                        async with asyncio.timeout(
                            max(0.001, trial_deadline - time.monotonic())
                        ):
                            verifier_result = await self._run_clean_room_command(
                                provider=provider,
                                run=run,
                                trial=trial,
                                task=task,
                                submission=submission,
                                command=verifier_command,
                                slot="primary",
                                timeout_seconds=verifier_timeout,
                                include_verifier_assets=True,
                            )
                    except TimeoutError:
                        trial.termination_cause = TerminationCause.TIME_LIMIT
                    else:
                        trial.verifier = VerifierResult(
                            command=verifier_command,
                            passed=verifier_result.exit_code == 0,
                            exit_code=verifier_result.exit_code,
                            stdout=verifier_result.stdout,
                            stderr=verifier_result.stderr,
                            duration_ms=verifier_result.duration_ms,
                            timed_out=verifier_result.timed_out,
                            truncated=verifier_result.truncated,
                        )
                        trial.reward = 1.0 if trial.verifier.passed else 0.0
                        trajectory = _append_verifier_step(
                            trajectory,
                            trial.verifier,
                        )
                        self.artifacts.save_json(
                            trial.id,
                            "verifier.json",
                            trial.verifier.model_dump(mode="json"),
                        )
                if should_cancel():
                    cancelled = True
                    trial.termination_cause = TerminationCause.CANCELLED
                elif (
                    trial.termination_cause is not TerminationCause.TIME_LIMIT
                    and trial.verifier is not None
                ):
                    verifier_criteria = await self._execute_contract_verifiers(
                        provider=provider,
                        run=run,
                        trial=trial,
                        task=task,
                        submission=submission,
                        contract=run.evaluation_contract,
                        should_cancel=should_cancel,
                        trial_deadline=trial_deadline,
                    )
                    if trial.termination_cause is TerminationCause.CANCELLED:
                        cancelled = True
        except Exception as caught:
            error = caught
            trial.error = _error_text(caught)
            if trial.termination_cause is None:
                trial.termination_cause = _termination_for_error(caught)
        finally:
            trial.status = AgentTrialStatus.CLEANING
            self.store.save_agent_trial(trial)
            if handle is not None:
                sandbox_session.state = SandboxSessionState.DELETING
                sandbox_session.updated_at = utc_now()
                self.store.save_sandbox_session(sandbox_session)
                try:
                    await provider.delete(handle)
                except Exception as cleanup_error:
                    sandbox_session.state = SandboxSessionState.CLEANUP_PENDING
                    sandbox_session.cleanup_error = _error_text(cleanup_error)
                    sandbox_session.updated_at = utc_now()
                    trial.error = (
                        f"{trial.error}; cleanup: {sandbox_session.cleanup_error}"
                        if trial.error
                        else sandbox_session.cleanup_error
                    )
                    trial.termination_cause = TerminationCause.CLEANUP_ERROR
                    error = cleanup_error
                else:
                    sandbox_session.state = SandboxSessionState.DELETED
                    sandbox_session.deleted_at = utc_now()
                    sandbox_session.updated_at = sandbox_session.deleted_at
                self.store.save_sandbox_session(sandbox_session)
            else:
                if provider.provider_id == "fake":
                    sandbox_session.state = SandboxSessionState.ERROR
                    sandbox_session.cleanup_error = trial.error
                    sandbox_session.updated_at = utc_now()
                else:
                    sandbox_session.state = SandboxSessionState.DELETING
                    sandbox_session.updated_at = utc_now()
                    self.store.save_sandbox_session(sandbox_session)
                    try:
                        await provider.cleanup_owned(None, ownership)
                    except Exception as cleanup_error:
                        sandbox_session.state = SandboxSessionState.CLEANUP_PENDING
                        sandbox_session.cleanup_error = _error_text(cleanup_error)
                        trial.error = (
                            f"{trial.error}; cleanup: {sandbox_session.cleanup_error}"
                            if trial.error
                            else sandbox_session.cleanup_error
                        )
                        trial.termination_cause = TerminationCause.CLEANUP_ERROR
                        error = cleanup_error
                        sandbox_session.updated_at = utc_now()
                    else:
                        sandbox_session.state = SandboxSessionState.DELETED
                        sandbox_session.deleted_at = utc_now()
                        sandbox_session.updated_at = sandbox_session.deleted_at
                        sandbox_session.cleanup_error = None
                self.store.save_sandbox_session(sandbox_session)

        if not cancelled and error is None:
            if should_cancel():
                cancelled = True
                trial.termination_cause = TerminationCause.CANCELLED
            elif (
                trial.termination_cause is TerminationCause.TIME_LIMIT
                or time.monotonic() >= trial_deadline
            ):
                trial.termination_cause = TerminationCause.TIME_LIMIT

        if (
            not cancelled
            and error is None
            and trial.verifier is not None
            and trial.termination_cause is not TerminationCause.TIME_LIMIT
        ):
            try:
                await self._evaluate_trial_contract(
                    run=run,
                    trial=trial,
                    task=task,
                    patch=patch,
                    trajectory=trajectory,
                    verifier_criteria=verifier_criteria,
                )
            except Exception as evaluation_error:
                error = evaluation_error
                trial.error = _error_text(evaluation_error)
                trial.termination_cause = (
                    TerminationCause.COST_LIMIT
                    if isinstance(evaluation_error, JudgeBudgetExceeded)
                    or (
                        isinstance(evaluation_error, JudgeProviderError)
                        and run.execution_policy.max_cost_usd is not None
                    )
                    else TerminationCause.VERIFIER_ERROR
                )

        trajectory = finalize_trajectory(
            trajectory,
            extra_metrics={
                "reward": trial.reward,
                "evaluation_contract_fingerprint": (
                    run.evaluation_contract.fingerprint
                ),
                "termination_cause": (
                    trial.termination_cause.value
                    if trial.termination_cause is not None
                    else None
                ),
            },
        )
        self.trajectories[trial.id] = trajectory
        self._save_trajectory(trial.id)
        valid_routing_artifacts = [
            item for item in routing_artifacts if item is not None
        ]
        if valid_routing_artifacts:
            summary = TrialRoutingSummary(
                trial_id=trial.id,
                run_id=run.id,
                layer_ids=self.topology.routed_layer_ids,
                selection_counts=aggregated_counts.tolist(),
                routing_mass=aggregated_mass.tolist(),
                total_routed_slots=sum(
                    item.total_routed_slots for item in valid_routing_artifacts
                ),
                inference_count=trial.turns,
                inference_calls=trial.turns,
                captured_inference_calls=len(valid_routing_artifacts),
                served_tokens=trial.prompt_tokens + trial.completion_tokens,
                model_id=self.model_id,
                model_session_id=model_session.id,
                profile_id=model_session.profile_id,
                profile_fingerprint=run.profile_fingerprint,
                artifacts=valid_routing_artifacts,
            )
            self.routings[trial.id] = summary
            self.artifacts.save_routing_summary(summary)
        trial.artifact_manifest = self._build_manifest(trial.id)
        self.artifacts.save_json(
            trial.id,
            "manifest.json",
            trial.artifact_manifest.model_dump(mode="json"),
        )
        trial.completed_at = utc_now()
        if provisioning_time_ms == 0:
            provisioning_time_ms = (time.perf_counter() - provisioning_started) * 1000
        trial.performance = _trial_performance(
            trial,
            [
                inference
                for inference in self.inferences.values()
                if inference.trial_id == trial.id
            ],
            verifier_criteria=list(verifier_criteria.values()),
            wall_time_ms=(time.perf_counter() - wall_started) * 1000,
            provisioning_time_ms=provisioning_time_ms,
            sandbox_time_ms=sandbox_time_ms,
        )
        if cancelled:
            trial.status = AgentTrialStatus.CANCELLED
        elif error is not None:
            trial.status = AgentTrialStatus.ERROR
        elif trial.evaluation is not None and trial.evaluation.passed is True:
            trial.status = AgentTrialStatus.PASSED
        else:
            trial.status = AgentTrialStatus.FAILED
        self.store.save_agent_trial(trial)

    async def _evaluate_trial_contract(
        self,
        *,
        run: AgentRun,
        trial: AgentTrial,
        task: AgentTask,
        patch: str,
        trajectory: AtifTrajectory,
        verifier_criteria: dict[str, CriterionResult],
    ) -> None:
        contract = run.evaluation_contract
        candidate = _agent_evaluation_candidate(
            patch=patch,
            trajectory=trajectory,
            verifier=trial.verifier,
        )
        judge_result = None
        if contract.judge is not None:
            remaining_cost = self._remaining_policy_cost_usd(run)
            judge_request = JudgeEvaluationRequest(
                config=contract.judge,
                task=task.instruction,
                candidate=candidate,
                criteria=[
                    criterion.description or criterion.label
                    for criterion in contract.criteria
                    if criterion.visibility is CriterionVisibility.PUBLIC
                ],
            )
            try:
                judge_result = await self.judge_service.evaluate(
                    judge_request,
                    max_incurred_cost_usd=remaining_cost,
                )
            except JudgeBudgetExceeded:
                raise
            except JudgeProviderError:
                trial.judge_cost_debit_usd = (
                    remaining_cost
                    if run.execution_policy.max_cost_usd is not None
                    else judge_request_cost_upper_bound(judge_request) or 0
                )
                trial.judge_cost_uncertain = True
                self.store.save_agent_trial(trial)
                raise
            assert judge_result is not None
            trial.judge_cost_debit_usd = _judge_cost_debit(judge_result)
            trial.judge_cost_uncertain = judge_result.usage.incurred_cost_usd is None
            self.store.save_agent_trial(trial)
        deterministic_results = dict(verifier_criteria)
        for criterion in contract.criteria:
            if criterion.kind is not DeterministicScorerKind.BENCHMARK_DEFAULT:
                continue
            passed = bool(trial.verifier is not None and trial.verifier.passed)
            deterministic_results[criterion.id] = CriterionResult(
                criterion_id=criterion.id,
                kind=criterion.kind,
                required=criterion.required,
                weight=criterion.weight,
                score=float(passed),
                passed=passed,
                explanation=(
                    "Trusted task verifier passed."
                    if passed
                    else "Trusted task verifier failed."
                ),
                execution_provenance="trusted_task_verifier",
            )
        evaluation = evaluate_output(
            contract,
            output=candidate,
            expected="",
            benchmark_scorer=lambda _: bool(
                trial.verifier is not None and trial.verifier.passed
            ),
            judge=judge_result,
            precomputed_criteria=deterministic_results,
        )
        trial.evaluation = evaluation
        trial.reward = evaluation.combined_score
        if trial.reward is None and evaluation.passed is not None:
            trial.reward = float(evaluation.passed)
        self.artifacts.save_json(
            trial.id,
            "evaluation.json",
            evaluation.model_dump(mode="json", exclude_none=True),
        )

    def _remaining_policy_cost_usd(self, run: AgentRun) -> float | None:
        maximum = run.execution_policy.max_cost_usd
        if maximum is None:
            return None
        inference_cost = sum(
            inference.estimated_cost_usd or 0
            for inference in self.inferences.values()
            if self.trials.get(inference.trial_id) is not None
            and self.trials[inference.trial_id].run_id == run.id
        )
        judge_cost = sum(
            (
                trial.judge_cost_debit_usd
                if trial.judge_cost_debit_usd > 0 or trial.judge_cost_uncertain
                else (
                    _judge_cost_debit(trial.evaluation.judge)
                    if trial.evaluation is not None
                    and trial.evaluation.judge is not None
                    else 0
                )
            )
            for trial in self._run_trials(run.id)
        )
        return max(0.0, maximum - inference_cost - judge_cost)

    async def _execute_contract_verifiers(
        self,
        *,
        provider: SandboxProvider,
        run: AgentRun,
        trial: AgentTrial,
        task: AgentTask,
        submission: _SubmissionSnapshot,
        contract,
        should_cancel: Callable[[], bool],
        trial_deadline: float,
    ) -> dict[str, CriterionResult]:
        results: dict[str, CriterionResult] = {}
        verifier_index = 0
        for criterion in contract.criteria:
            if criterion.kind is not DeterministicScorerKind.VERIFIER:
                continue
            if should_cancel():
                trial.termination_cause = TerminationCause.CANCELLED
                break
            timeout_seconds = _remaining_verifier_timeout(
                trial_deadline,
                min(
                    criterion.verifier_timeout_seconds,
                    task.timeout_seconds,
                ),
            )
            if timeout_seconds is None:
                trial.termination_cause = TerminationCause.TIME_LIMIT
                break
            verifier_index += 1
            command = criterion.verifier_command or ""
            started = time.perf_counter()
            try:
                async with asyncio.timeout(
                    max(0.001, trial_deadline - time.monotonic())
                ):
                    result = await self._run_clean_room_command(
                        provider=provider,
                        run=run,
                        trial=trial,
                        task=task,
                        submission=submission,
                        command=command,
                        slot=f"criterion-{verifier_index}",
                        timeout_seconds=timeout_seconds,
                        include_verifier_assets=False,
                    )
            except TimeoutError:
                trial.termination_cause = TerminationCause.TIME_LIMIT
                break
            except SandboxProviderError as error:
                if error.code == "sandbox_cleanup_failed":
                    raise
                results[criterion.id] = CriterionResult(
                    criterion_id=criterion.id,
                    kind=criterion.kind,
                    required=criterion.required,
                    weight=criterion.weight,
                    score=0,
                    passed=False,
                    explanation="Sandbox verifier execution failed.",
                    error=_error_text(error),
                    execution_provenance="user_authored_sandbox",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                continue
            except Exception as error:
                results[criterion.id] = CriterionResult(
                    criterion_id=criterion.id,
                    kind=criterion.kind,
                    required=criterion.required,
                    weight=criterion.weight,
                    score=0,
                    passed=False,
                    explanation="Sandbox verifier execution failed.",
                    error=_error_text(error),
                    execution_provenance="user_authored_sandbox",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                continue
            output = f"{result.stdout}\n{result.stderr}".encode()
            passed = result.exit_code == 0 and not result.timed_out
            results[criterion.id] = CriterionResult(
                criterion_id=criterion.id,
                kind=criterion.kind,
                required=criterion.required,
                weight=criterion.weight,
                score=float(passed),
                passed=passed,
                explanation=(
                    "User-authored sandbox verifier passed."
                    if passed
                    else "User-authored sandbox verifier failed."
                ),
                execution_provenance="user_authored_sandbox",
                exit_code=result.exit_code,
                duration_ms=result.duration_ms,
                timed_out=result.timed_out,
                output_sha256=hashlib.sha256(output).hexdigest(),
            )
            if should_cancel():
                trial.termination_cause = TerminationCause.CANCELLED
                break
            if time.monotonic() >= trial_deadline:
                trial.termination_cause = TerminationCause.TIME_LIMIT
                break
        return results

    async def _run_clean_room_command(
        self,
        *,
        provider: SandboxProvider,
        run: AgentRun,
        trial: AgentTrial,
        task: AgentTask,
        submission: _SubmissionSnapshot,
        command: str,
        slot: str,
        timeout_seconds: int,
        include_verifier_assets: bool,
    ) -> CommandResult:
        ownership = SandboxOwnership(
            controller_id=self.settings.agent_controller_id,
            run_id=run.id,
            trial_id=f"{trial.id}-verify-{slot}",
        )
        session = SandboxSession(
            id=str(uuid4()),
            trial_id=trial.id,
            provider_id=provider.provider_id,
            state=SandboxSessionState.CREATING,
            ownership=ownership,
            spec=task.sandbox_spec(),
        )
        self.sandboxes[session.id] = session
        self.store.save_sandbox_session(session)
        handle = None
        try:
            handle = await provider.create(task.sandbox_spec(), ownership)
            session.external_id = handle.id
            session.state = SandboxSessionState.READY
            session.effective_network_policy = task.network_policy
            session.updated_at = utc_now()
            self.store.save_sandbox_session(session)
            clean_room_files = self._clean_room_files(
                task,
                submission,
                include_verifier_assets=include_verifier_assets,
            )
            for file in clean_room_files:
                await provider.upload(handle, file)
            hardening_command = _clean_room_hardening_command(
                task,
                clean_room_files,
            )
            if isinstance(provider, FakeSandboxProvider):
                provider.script(
                    hardening_command,
                    _command_result(
                        hardening_command,
                        stdout="Hardened verifier workspace.\n",
                    ),
                )
            hardening = await provider.exec(
                handle,
                hardening_command,
                cwd=task.working_directory,
                timeout_seconds=min(30, timeout_seconds),
            )
            if hardening.exit_code != 0 or hardening.timed_out:
                raise SandboxProviderError(
                    "verification_hardening_failed",
                    "The verifier workspace permission boundary could not be applied.",
                )
            return await provider.exec(
                handle,
                command,
                cwd=task.working_directory,
                timeout_seconds=timeout_seconds,
            )
        finally:
            session.state = SandboxSessionState.DELETING
            session.updated_at = utc_now()
            self.store.save_sandbox_session(session)
            try:
                if handle is None:
                    await provider.cleanup_owned(None, ownership)
                else:
                    await provider.delete(handle)
            except Exception as cleanup_error:
                session.state = SandboxSessionState.CLEANUP_PENDING
                session.cleanup_error = _error_text(cleanup_error)
                session.updated_at = utc_now()
                self.store.save_sandbox_session(session)
                raise SandboxProviderError(
                    "sandbox_cleanup_failed",
                    "The isolated verifier sandbox could not be deleted.",
                    retryable=True,
                ) from cleanup_error
            session.state = SandboxSessionState.DELETED
            session.deleted_at = utc_now()
            session.updated_at = session.deleted_at
            session.cleanup_error = None
            self.store.save_sandbox_session(session)

    async def _capture_submission(
        self,
        provider: SandboxProvider,
        handle,
        task: AgentTask,
    ) -> _SubmissionSnapshot:
        canonical = {file.path: file for file in task.files}
        captured: list[SandboxFile] = []
        deleted: set[str] = set()
        total_bytes = 0
        for path in task.submission_file_paths:
            remaining = self.settings.agent_max_patch_bytes - total_bytes
            if remaining <= 0:
                raise SandboxProviderError(
                    "submission_too_large",
                    "The submitted files exceed the configured byte limit.",
                )
            try:
                content = await provider.read(handle, path, max_bytes=remaining)
            except SandboxProviderError as error:
                if error.code != "file_not_found":
                    raise
                deleted.add(path)
                continue
            total_bytes += len(content)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise SandboxProviderError(
                    "invalid_submission_encoding",
                    "Submitted files must be valid UTF-8 text.",
                ) from error
            captured.append(
                SandboxFile(
                    path=path,
                    content=text,
                    executable=(
                        canonical[path].executable if path in canonical else False
                    ),
                )
            )
        return _SubmissionSnapshot(
            files=tuple(captured),
            deleted_paths=frozenset(deleted),
        )

    @staticmethod
    def _clean_room_files(
        task: AgentTask,
        submission: _SubmissionSnapshot,
        *,
        include_verifier_assets: bool,
    ) -> tuple[SandboxFile, ...]:
        submission_paths = set(task.submission_file_paths)
        oracle_paths = set(task.oracle_file_paths)
        verifier_paths = set(task.verifier_file_paths)
        canonical = [
            file.model_copy(deep=True)
            for file in task.files
            if file.path not in submission_paths
            and file.path not in oracle_paths
            and (include_verifier_assets or file.path not in verifier_paths)
        ]
        return (*canonical, *submission.files)

    def _configure_fake_scripts(
        self, provider: SandboxProvider, task: AgentTask
    ) -> None:
        if not isinstance(provider, FakeSandboxProvider):
            return
        for command in task.oracle_commands:
            provider.script(
                command,
                _command_result(command, stdout="Applied reference edit.\n"),
            )
        provider.script(
            task.verifier_command,
            _command_result(
                task.verifier_command,
                stdout=(
                    "Ran bundled unittest verifier in the simulated sandbox.\n"
                    "All tests passed.\n"
                ),
            ),
        )

    @staticmethod
    def _scripted_actions(provider: SandboxProvider, task: AgentTask) -> list[str]:
        if not isinstance(provider, FakeSandboxProvider):
            return []
        commands = list(task.oracle_commands)
        return [
            json.dumps({"action": "shell", "command": command}) for command in commands
        ] + [
            json.dumps(
                {
                    "action": "finish",
                    "summary": "Implemented the focused fix and verified it.",
                }
            )
        ]

    def _submission_patch(
        self,
        task: AgentTask,
        submission: _SubmissionSnapshot,
    ) -> str:
        chunks: list[str] = []
        size = 0
        canonical = {file.path: file.content for file in task.files}
        captured = {file.path: file.content for file in submission.files}
        for path in task.submission_file_paths:
            original = canonical.get(path, "")
            if path in submission.deleted_paths:
                current = ""
            elif path in captured:
                current = captured[path]
            else:
                continue
            if current == original:
                continue
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
            size += len(diff.encode())
            if size > self.settings.agent_max_patch_bytes:
                chunks.append("\n[patch truncated]\n")
                break
            chunks.append(diff)
        return "".join(chunks)

    def _save_live_trial(self, trial: AgentTrial) -> None:
        self.store.save_agent_trial(trial)
        self._save_trajectory(trial.id)

    def _save_trajectory(self, trial_id: str) -> None:
        self.artifacts.save_json(
            trial_id,
            "trajectory.json",
            self.trajectories[trial_id].model_dump(mode="json", exclude_none=True),
        )

    def _build_manifest(self, trial_id: str) -> TrialArtifactManifest:
        artifacts: list[TrialArtifact] = []
        for name, media_type in (
            ("trajectory.json", "application/json"),
            ("patch.diff", "text/x-diff"),
            ("verifier.json", "application/json"),
            ("evaluation.json", "application/json"),
        ):
            path = self.artifacts.path(trial_id, name)
            if not path.is_file():
                continue
            payload = path.read_bytes()
            artifacts.append(
                TrialArtifact(
                    name=name,
                    relative_path=path.relative_to(self.artifacts.root).as_posix(),
                    media_type=media_type,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
            )
        return TrialArtifactManifest(trial_id=trial_id, artifacts=artifacts)

    def _trial_summary(self, trial: AgentTrial) -> AgentTrialSummary:
        inferences = [
            item for item in self.inferences.values() if item.trial_id == trial.id
        ]
        sandbox = self.sandboxes.get(trial.sandbox_session_id or "")
        return AgentTrialSummary(
            id=trial.id,
            agent_run_id=trial.run_id,
            task_id=trial.task_id,
            title=(
                trial.task_title_snapshot or f"Unknown archived task ({trial.task_id})"
            ),
            task_provenance_status=(
                "known" if trial.task_title_snapshot is not None else "legacy_unknown"
            ),
            attempt=trial.attempt,
            seed=trial.seed,
            status=trial.status,
            reward=trial.reward,
            turns=trial.turns,
            commands=trial.commands,
            prompt_tokens=trial.prompt_tokens,
            completion_tokens=trial.completion_tokens,
            inference_calls=len(inferences),
            routed_inference_calls=sum(
                item.routing_artifact is not None for item in inferences
            ),
            termination_reason=(
                trial.termination_cause.value
                if trial.termination_cause is not None
                else None
            ),
            sandbox_status=(sandbox.state.value if sandbox else "not_created"),
            evaluation=trial.evaluation,
            performance=trial.performance,
            updated_at=_trial_updated_at(trial),
        )

    def _update_run_metrics(self, run: AgentRun) -> None:
        trials = self._run_trials(run.id)
        terminal = {
            AgentTrialStatus.PASSED,
            AgentTrialStatus.FAILED,
            AgentTrialStatus.ERROR,
            AgentTrialStatus.CANCELLED,
        }
        run.completed_trials = sum(trial.status in terminal for trial in trials)
        run.passed_trials = sum(
            trial.status is AgentTrialStatus.PASSED for trial in trials
        )
        rewards = [trial.reward for trial in trials if trial.reward is not None]
        run.mean_reward = sum(rewards) / len(rewards) if rewards else None
        self.store.save_agent_run(run)

    def _cancel_unstarted_trials(
        self,
        run_id: str,
        *,
        cause: TerminationCause = TerminationCause.CANCELLED,
    ) -> None:
        for trial in self._run_trials(run_id):
            if trial.status is AgentTrialStatus.QUEUED:
                trial.status = AgentTrialStatus.CANCELLED
                trial.termination_cause = cause
                trial.completed_at = utc_now()
                self.store.save_agent_trial(trial)

    def _run_trials(self, run_id: str) -> list[AgentTrial]:
        return sorted(
            (trial for trial in self.trials.values() if trial.run_id == run_id),
            key=lambda item: item.created_at,
        )

    def _provider(self, provider_id: str) -> SandboxProvider:
        try:
            return self.providers[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown sandbox provider {provider_id!r}") from error

    def _load_artifacts(self) -> None:
        for trial_id in self.trials:
            trajectory = self.artifacts.load_json(trial_id, "trajectory.json")
            if trajectory is not None:
                try:
                    self.trajectories[trial_id] = AtifTrajectory.model_validate(
                        trajectory
                    )
                except ValueError:
                    logger.exception("invalid saved ATIF trajectory for %s", trial_id)
            try:
                routing = self.artifacts.load_routing_summary(trial_id)
            except (OSError, ValueError, KeyError):
                logger.exception("invalid saved routing summary for %s", trial_id)
            else:
                if routing is not None:
                    self.routings[trial_id] = routing

    def _reconcile_interrupted(self) -> None:
        affected_run_ids: set[str] = set()
        active_runs = {
            AgentRunStatus.QUEUED,
            AgentRunStatus.RUNNING,
            AgentRunStatus.CANCELLING,
        }
        for run in self.runs.values():
            if run.status in active_runs:
                affected_run_ids.add(run.id)
                run.status = AgentRunStatus.FAILED
                run.error = "application restarted before the agent run completed"
                run.completed_at = utc_now()
                self.store.save_agent_run(run)
        for trial in self.trials.values():
            if trial.status in _ACTIVE_TRIAL_STATES:
                affected_run_ids.add(trial.run_id)
                trial.status = AgentTrialStatus.ERROR
                trial.termination_cause = TerminationCause.INTERRUPTED
                trial.error = "application restarted before the trial completed"
                trial.completed_at = utc_now()
                self.store.save_agent_trial(trial)
        for sandbox in self.sandboxes.values():
            if sandbox.state not in {
                SandboxSessionState.DELETED,
                SandboxSessionState.ERROR,
            }:
                sandbox.state = (
                    SandboxSessionState.DELETED
                    if sandbox.provider_id == "fake"
                    else SandboxSessionState.CLEANUP_PENDING
                )
                sandbox.cleanup_error = (
                    None
                    if sandbox.provider_id == "fake"
                    else "application restarted before deletion was confirmed"
                )
                sandbox.updated_at = utc_now()
                self.store.save_sandbox_session(sandbox)
        for run_id in affected_run_ids:
            if run_id in self.runs:
                self._update_run_metrics(self.runs[run_id])


def _profile_fingerprint(model_session: ModelSession) -> str | None:
    if model_session.profile is None:
        return None
    return hashlib.sha256(
        json.dumps(
            model_session.profile.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _clean_room_hardening_command(
    task: AgentTask,
    files: tuple[SandboxFile, ...],
) -> str:
    root = PurePosixPath(task.working_directory)
    directories = {root}
    absolute_files: dict[str, SandboxFile] = {}
    for file in files:
        absolute = root / file.path
        absolute_files[file.path] = file
        parent = absolute.parent
        while parent != root:
            directories.add(parent)
            parent = parent.parent
    verifier_paths = set(task.verifier_file_paths)
    hidden = [
        root / path for path in task.verifier_file_paths if path in absolute_files
    ]
    executable = [
        root / path
        for path, file in absolute_files.items()
        if path not in verifier_paths and file.executable
    ]
    readable = [
        root / path
        for path, file in absolute_files.items()
        if path not in verifier_paths and not file.executable
    ]

    def chmod(mode: str, paths: list[PurePosixPath]) -> str | None:
        if not paths:
            return None
        quoted = " ".join(shlex.quote(path.as_posix()) for path in sorted(paths))
        return f"chmod {mode} -- {quoted}"

    commands = [
        "set -eu",
        "umask 077",
        chmod("0755", list(directories)),
        chmod("0444", readable),
        chmod("0555", executable),
        chmod("0600", hidden),
    ]
    return "\n".join(command for command in commands if command is not None)


def _bounded_generation(
    generation: GenerationConfig, remaining_tokens: int, *, seed: int
) -> GenerationConfig:
    return generation.model_copy(
        update={
            "max_tokens": max(1, min(generation.max_tokens, remaining_tokens)),
            "seed": seed,
        }
    )


def _atif_metrics(inference) -> AtifMetrics:
    return AtifMetrics(
        prompt_tokens=inference.prompt_tokens,
        completion_tokens=inference.completion_tokens,
        cost_usd=inference.estimated_cost_usd,
        latency_ms=inference.latency_ms,
        extra={
            "inference_id": inference.id,
            "routing_artifact": (
                inference.routing_artifact.relative_path
                if inference.routing_artifact is not None
                else None
            ),
            "routing_artifact_id": (
                inference.routing_artifact.relative_path
                if inference.routing_artifact is not None
                else None
            ),
            "total_routed_slots": (
                inference.routing_artifact.total_routed_slots
                if inference.routing_artifact is not None
                else 0
            ),
            "reasoning_tokens": inference.reasoning_tokens,
            "time_to_first_token_ms": inference.time_to_first_token_ms,
            "generation_time_ms": inference.generation_time_ms,
            "queue_time_ms": inference.queue_time_ms,
            "mean_inter_token_latency_ms": (inference.mean_inter_token_latency_ms),
            "tokens_per_second": inference.tokens_per_second,
            "finish_reason": inference.finish_reason,
            "started_at": (
                inference.started_at.isoformat()
                if inference.started_at is not None
                else None
            ),
            "completed_at": (
                inference.completed_at.isoformat()
                if inference.completed_at is not None
                else None
            ),
        },
    )


def _inference_summary(
    inference: InferenceCall,
    model_id: str,
    routed_layers: int,
) -> InferenceCallSummary:
    artifact = inference.routing_artifact
    return InferenceCallSummary(
        id=inference.id,
        prompt_tokens=inference.prompt_tokens,
        reasoning_tokens=inference.reasoning_tokens,
        completion_tokens=inference.completion_tokens,
        total_tokens=inference.prompt_tokens + inference.completion_tokens,
        latency_ms=inference.latency_ms,
        ttft_ms=inference.time_to_first_token_ms,
        prefill_ms=None,
        decode_ms=inference.generation_time_ms,
        tokens_per_second=inference.tokens_per_second,
        estimated_cost_usd=inference.estimated_cost_usd,
        model_id=model_id,
        finish_reason=inference.finish_reason,
        started_at=inference.started_at,
        completed_at=inference.completed_at,
        routing_artifact_id=(artifact.relative_path if artifact is not None else None),
        routed_layers=routed_layers if artifact is not None else 0,
        total_routed_slots=(artifact.total_routed_slots if artifact is not None else 0),
    )


def _trial_performance(
    trial: AgentTrial,
    inferences: list[InferenceCall],
    *,
    verifier_criteria: list[CriterionResult],
    wall_time_ms: float,
    provisioning_time_ms: float,
    sandbox_time_ms: float,
) -> TrialPerformance:
    throughput = [
        inference.tokens_per_second
        for inference in inferences
        if inference.tokens_per_second is not None
    ]
    reasoning_values = [
        inference.reasoning_tokens
        for inference in inferences
        if inference.reasoning_tokens is not None
    ]
    inference_costs = [
        inference.estimated_cost_usd
        for inference in inferences
        if inference.estimated_cost_usd is not None
    ]
    inference_cost = sum(inference_costs) if inference_costs else None
    judge_result = (
        trial.evaluation.judge
        if trial.evaluation is not None and trial.evaluation.judge is not None
        else None
    )
    judge_cost = (
        judge_result.usage.incurred_cost_usd if judge_result is not None else None
    )
    judge_cost_debit = (
        trial.judge_cost_debit_usd
        if trial.judge_cost_debit_usd > 0 or trial.judge_cost_uncertain
        else (_judge_cost_debit(judge_result) if judge_result is not None else 0)
    )
    known_costs = [cost for cost in (inference_cost, judge_cost) if cost is not None]
    queue_time_ms = (
        max(0.0, (trial.started_at - trial.created_at).total_seconds() * 1000)
        if trial.started_at is not None
        else 0.0
    )
    return TrialPerformance(
        wall_time_ms=max(0.0, wall_time_ms),
        queue_time_ms=queue_time_ms,
        provisioning_time_ms=max(0.0, provisioning_time_ms),
        model_time_ms=sum(inference.latency_ms for inference in inferences),
        sandbox_time_ms=max(0.0, sandbox_time_ms),
        verifier_time_ms=(trial.verifier.duration_ms if trial.verifier else 0.0)
        + sum(result.duration_ms or 0.0 for result in verifier_criteria),
        prompt_tokens=trial.prompt_tokens,
        reasoning_tokens=(sum(reasoning_values) if reasoning_values else None),
        completion_tokens=trial.completion_tokens,
        total_tokens=trial.prompt_tokens + trial.completion_tokens,
        mean_tps=(float(np.mean(throughput)) if throughput else None),
        p50_tps=(float(np.percentile(throughput, 50)) if throughput else None),
        p95_tps=(float(np.percentile(throughput, 95)) if throughput else None),
        inference_cost_usd=inference_cost,
        judge_cost_usd=judge_cost,
        judge_cost_debit_usd=judge_cost_debit,
        judge_cost_uncertain=trial.judge_cost_uncertain,
        estimated_cost_usd=(sum(known_costs) if known_costs else None),
    )


def _append_verifier_step(
    trajectory: AtifTrajectory, verifier: VerifierResult
) -> AtifTrajectory:
    output = verifier.stdout
    if verifier.stderr:
        output = f"{output}\n{verifier.stderr}".strip()
    payload = trajectory.model_dump(mode="python")
    payload["steps"] = [
        *trajectory.steps,
        AtifStep(
            step_id=len(trajectory.steps) + 1,
            timestamp=utc_now(),
            source=AtifSource.SYSTEM,
            message=f"[verifier]\n{output}",
        ),
    ]
    return AtifTrajectory.model_validate(payload)


def _patch_stats(patch: str) -> PatchStats:
    lines = patch.splitlines()
    return PatchStats(
        files_changed=sum(line.startswith("--- a/") for line in lines),
        insertions=sum(
            line.startswith("+") and not line.startswith("+++") for line in lines
        ),
        deletions=sum(
            line.startswith("-") and not line.startswith("---") for line in lines
        ),
        bytes=len(patch.encode()),
    )


def _command_output(result) -> str:
    output = result.stdout
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr}" if output else result.stderr
    return output


def _agent_evaluation_candidate(
    *,
    patch: str,
    trajectory: AtifTrajectory,
    verifier: VerifierResult | None,
) -> str:
    final_response = next(
        (
            step.message
            for step in reversed(trajectory.steps)
            if step.source is AtifSource.AGENT
        ),
        "",
    )
    verifier_summary = "[not run]"
    if verifier is not None:
        verifier_summary = (
            f"passed={verifier.passed}; exit_code={verifier.exit_code}; "
            f"duration_ms={verifier.duration_ms:.3f}; "
            f"timed_out={verifier.timed_out}"
        )
    sections = [
        f"FINAL AGENT RESPONSE\n{final_response}",
        f"PATCH\n{patch or '[no patch captured]'}",
        f"TRUSTED VERIFIER RESULT\n{verifier_summary}",
    ]
    return "\n\n".join(sections)[:450_000]


def _agent_policy_stop_cause(
    run: AgentRun,
    trials: list[AgentTrial],
    *,
    elapsed_seconds: float,
) -> TerminationCause | None:
    policy = run.execution_policy
    if elapsed_seconds >= policy.timeout_seconds:
        return TerminationCause.TIME_LIMIT
    if (
        policy.max_tokens is not None
        and sum(trial.prompt_tokens + trial.completion_tokens for trial in trials)
        >= policy.max_tokens
    ):
        return TerminationCause.TOKEN_LIMIT
    if (
        policy.max_cost_usd is not None
        and sum(
            (
                (
                    trial.performance.inference_cost_usd
                    if trial.performance is not None
                    and trial.performance.inference_cost_usd is not None
                    else 0
                )
                + trial.judge_cost_debit_usd
            )
            for trial in trials
        )
        >= policy.max_cost_usd
    ):
        return TerminationCause.COST_LIMIT
    if any(trial.termination_cause is TerminationCause.COST_LIMIT for trial in trials):
        return TerminationCause.COST_LIMIT
    if any(trial.termination_cause is TerminationCause.TOKEN_LIMIT for trial in trials):
        return TerminationCause.TOKEN_LIMIT
    if policy.fail_fast and any(
        trial.status in {AgentTrialStatus.FAILED, AgentTrialStatus.ERROR}
        for trial in trials
    ):
        return TerminationCause.CANCELLED
    return None


def _judge_cost_debit(result: LLMJudgeResult) -> float:
    incurred = result.usage.incurred_cost_usd
    if incurred is not None:
        return incurred
    return result.usage.cost_upper_bound_usd or 0


def _remaining_verifier_timeout(
    trial_deadline: float,
    requested_timeout_seconds: float,
) -> int | None:
    remaining_seconds = trial_deadline - time.monotonic()
    if remaining_seconds < 1:
        return None
    return max(1, min(int(requested_timeout_seconds), int(remaining_seconds)))


def _public_evaluation_contract(
    contract: EvaluationContract,
) -> AgentEvaluationContractView:
    payload = contract.model_dump(mode="python")
    payload["criteria"] = [
        criterion
        for criterion in contract.criteria
        if criterion.visibility is CriterionVisibility.PUBLIC
    ]
    return AgentEvaluationContractView.model_validate(payload)


def _bounded_prompt_output(output: str, max_characters: int = 6_000) -> str:
    if len(output) <= max_characters:
        return output
    return f"{output[:max_characters]}\n[output truncated for model context]"


def _prompt_token_upper_bound(messages: list[dict[str, str]]) -> int:
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded) + 512


def _bounded_messages(
    messages: list[dict[str, str]], max_characters: int = 14_000
) -> list[dict[str, str]]:
    if sum(len(item.get("content", "")) for item in messages) <= max_characters:
        return messages
    fixed = messages[:2]
    remaining = max_characters - sum(len(item.get("content", "")) for item in fixed)
    tail: list[dict[str, str]] = []
    for message in reversed(messages[2:]):
        content = message.get("content", "")
        if tail and len(content) > remaining:
            break
        clipped = content[-max(0, remaining) :]
        tail.append({**message, "content": clipped})
        remaining -= len(clipped)
        if remaining <= 0:
            break
    return [*fixed, *reversed(tail)]


def _command_result(command: str, *, stdout: str):
    from .domain import CommandResult

    return CommandResult(
        command=command,
        exit_code=0,
        stdout=stdout,
        duration_ms=5,
    )


def _termination_for_error(error: Exception) -> TerminationCause:
    if (
        isinstance(error, SandboxProviderError)
        and error.code == "sandbox_cleanup_failed"
    ):
        return TerminationCause.CLEANUP_ERROR
    if error.__class__.__module__.startswith("moe_tools_suite.agentic.providers"):
        return TerminationCause.SANDBOX_ERROR
    return TerminationCause.MODEL_ERROR


def _error_text(error: Exception) -> str:
    return (str(error).strip() or error.__class__.__name__)[:1000]


def _trial_updated_at(trial: AgentTrial) -> datetime:
    return trial.completed_at or trial.started_at or trial.created_at


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
