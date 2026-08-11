from __future__ import annotations

import difflib
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePosixPath
from uuid import uuid4

import numpy as np

from ..domain import GenerationConfig, ModelSession, ModelState, ModelTopology
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
    AgentRun,
    AgentRunDetail,
    AgentRunExport,
    AgentRunStatus,
    AgentRunView,
    AgentTask,
    AgentTrajectoryView,
    AgentTrial,
    AgentTrialArtifactsView,
    AgentTrialStatus,
    AgentTrialSummary,
    ArtifactExport,
    CreateAgentRunRequest,
    InferenceCallSummary,
    PatchStats,
    ProviderPreflight,
    SandboxOwnership,
    SandboxProviderInfo,
    SandboxSession,
    SandboxSessionState,
    TerminationCause,
    TrajectoryStepType,
    TrajectoryStepView,
    TrialArtifact,
    TrialArtifactManifest,
    TrialRoutingSummary,
    VerifierResult,
    VerifierStatus,
    VerifierView,
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
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime = runtime
        self.topology = topology
        self.model_id = model_id
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

    def create_run(
        self,
        *,
        run_id: str,
        job_id: str,
        request: CreateAgentRunRequest,
        model_session: ModelSession,
    ) -> AgentRun:
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
                task = self.catalog.get_task(run.task_pack_id, trial.task_id)
                await self._execute_trial(
                    run=run,
                    trial=trial,
                    task=task,
                    model_session=model_session,
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
            model_session_id=run.model_session_id,
            profile_id=run.profile_id,
            profile_fingerprint=run.profile_fingerprint,
            agent_id=run.agent_id,
            sandbox_provider_id=run.sandbox_provider_id,
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
                    )
                )
                continue
            inference = inference_by_step.get(str(step.step_id))
            inference_view = (
                InferenceCallSummary(
                    id=inference.id,
                    prompt_tokens=inference.prompt_tokens,
                    completion_tokens=inference.completion_tokens,
                    latency_ms=inference.latency_ms,
                    routing_artifact_id=(
                        inference.routing_artifact.relative_path
                        if inference.routing_artifact is not None
                        else None
                    ),
                    routed_layers=(
                        self.topology.num_layers
                        if inference.routing_artifact is not None
                        else 0
                    ),
                    total_routed_slots=(
                        inference.routing_artifact.total_routed_slots
                        if inference.routing_artifact is not None
                        else 0
                    ),
                )
                if inference is not None
                else None
            )
            events.append(
                TrajectoryStepView(
                    id=f"atif-{step.step_id}-model",
                    sequence=len(events) + 1,
                    timestamp=timestamp,
                    type=TrajectoryStepType.ASSISTANT,
                    title="Model response",
                    content=step.message,
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
                        exit_code=_optional_int(extra.get("exit_code")),
                        duration_ms=_optional_float(extra.get("duration_ms")),
                        truncated=bool(extra.get("truncated", False)),
                    )
                )
        return AgentTrajectoryView(
            trial_id=trial_id,
            schema_version=trajectory.schema_version,
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
            detail=AgentRunDetail(run=self.runs[run_id], trials=trials),
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
        should_cancel: Callable[[], bool],
    ) -> None:
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
        try:
            handle = await provider.create(task.sandbox_spec(), ownership)
            sandbox_session.external_id = handle.id
            sandbox_session.state = SandboxSessionState.READY
            sandbox_session.effective_network_policy = task.network_policy
            sandbox_session.updated_at = utc_now()
            self.store.save_sandbox_session(sandbox_session)
            for file in task.files:
                await provider.upload(handle, file)
            self._configure_fake_scripts(provider, task)
            trial.status = AgentTrialStatus.RUNNING
            self.store.save_agent_trial(trial)
            messages = [
                {"role": "system", "content": BASH_JSON_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ]
            scripted = self._scripted_actions(provider, task)
            started = time.monotonic()
            for turn_index in range(run.budgets.max_turns):
                if should_cancel():
                    cancelled = True
                    trial.termination_cause = TerminationCause.CANCELLED
                    break
                if time.monotonic() - started > run.budgets.timeout_seconds:
                    trial.termination_cause = TerminationCause.TIME_LIMIT
                    break
                used_tokens = trial.prompt_tokens + trial.completion_tokens
                if used_tokens >= run.budgets.max_tokens:
                    trial.termination_cause = TerminationCause.TOKEN_LIMIT
                    break
                step_id = len(trajectory.steps) + 1
                scripted_content = (
                    scripted[turn_index] if turn_index < len(scripted) else None
                )
                generation = _bounded_generation(
                    run.generation,
                    run.budgets.max_tokens - used_tokens,
                    seed=trial.seed,
                )
                gateway_result = await self.gateway.infer(
                    trial_id=trial.id,
                    trajectory_step_id=str(step_id),
                    model_session=model_session,
                    messages=_bounded_messages(messages),
                    generation=generation,
                    request_key=f"agent:{trial.id}:{turn_index}",
                    scripted_content=scripted_content,
                )
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
                    )
                    trial.termination_cause = TerminationCause.AGENT_FINISHED
                    self.trajectories[trial.id] = trajectory
                    self._save_live_trial(trial)
                    break
                if trial.commands >= run.budgets.max_commands:
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
                trial.commands += 1
                trajectory = append_agent_step(
                    trajectory,
                    message=gateway_result.content,
                    model_name=self.model_id,
                    metrics=_atif_metrics(gateway_result.inference),
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
            if not cancelled:
                trial.status = AgentTrialStatus.VERIFYING
                self.store.save_agent_trial(trial)
                await self._restore_verifier_files(provider, handle, task)
                verifier_command = task.verifier_command
                verifier_result = await provider.exec(
                    handle,
                    verifier_command,
                    cwd=task.working_directory,
                    timeout_seconds=max(1, min(300, int(task.timeout_seconds))),
                )
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
                trajectory = _append_verifier_step(trajectory, trial.verifier)
                patch = await self._collect_patch(provider, handle, task)
                trial.patch_stats = _patch_stats(patch)
                self.artifacts.save_text(trial.id, "patch.diff", patch)
                self.artifacts.save_json(
                    trial.id,
                    "verifier.json",
                    trial.verifier.model_dump(mode="json"),
                )
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

        trajectory = finalize_trajectory(
            trajectory,
            extra_metrics={
                "reward": trial.reward,
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
        if cancelled:
            trial.status = AgentTrialStatus.CANCELLED
        elif error is not None:
            trial.status = AgentTrialStatus.ERROR
        elif trial.verifier is not None and trial.verifier.passed:
            trial.status = AgentTrialStatus.PASSED
        else:
            trial.status = AgentTrialStatus.FAILED
        self.store.save_agent_trial(trial)

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
        commands = [*task.oracle_commands, task.verifier_command]
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

    async def _collect_patch(
        self,
        provider: SandboxProvider,
        handle,
        task: AgentTask,
    ) -> str:
        chunks: list[str] = []
        size = 0
        for original in task.files:
            path = str(PurePosixPath(task.working_directory) / original.path)
            try:
                current = (
                    await provider.read(
                        handle,
                        path,
                        max_bytes=self.settings.agent_max_patch_bytes,
                    )
                ).decode(errors="replace")
            except Exception:
                continue
            if current == original.content:
                continue
            diff = "".join(
                difflib.unified_diff(
                    original.content.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{original.path}",
                    tofile=f"b/{original.path}",
                )
            )
            size += len(diff.encode())
            if size > self.settings.agent_max_patch_bytes:
                chunks.append("\n[patch truncated]\n")
                break
            chunks.append(diff)
        return "".join(chunks)

    @staticmethod
    async def _restore_verifier_files(
        provider: SandboxProvider,
        handle,
        task: AgentTask,
    ) -> None:
        protected = set(task.verifier_file_paths)
        for file in task.files:
            if file.path in protected:
                await provider.upload(handle, file)

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
        task = self.catalog.get_task(
            self.runs[trial.run_id].task_pack_id, trial.task_id
        )
        inferences = [
            item for item in self.inferences.values() if item.trial_id == trial.id
        ]
        sandbox = self.sandboxes.get(trial.sandbox_session_id or "")
        return AgentTrialSummary(
            id=trial.id,
            agent_run_id=trial.run_id,
            task_id=trial.task_id,
            title=task.title,
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

    def _cancel_unstarted_trials(self, run_id: str) -> None:
        for trial in self._run_trials(run_id):
            if trial.status is AgentTrialStatus.QUEUED:
                trial.status = AgentTrialStatus.CANCELLED
                trial.termination_cause = TerminationCause.CANCELLED
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
        latency_ms=inference.latency_ms,
        extra={
            "inference_id": inference.id,
            "routing_artifact": (
                inference.routing_artifact.relative_path
                if inference.routing_artifact is not None
                else None
            ),
        },
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


def _bounded_prompt_output(output: str, max_characters: int = 6_000) -> str:
    if len(output) <= max_characters:
        return output
    return f"{output[:max_characters]}\n[output truncated for model context]"


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
