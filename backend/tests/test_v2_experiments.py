import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.agentic.providers.fake import FakeSandboxProvider
from moe_tools_suite.domain import (
    CreateExpertProfileRequest,
    CreateModelSessionRequest,
    ExpertProfile,
    JobPhaseRecord,
    JobStatus,
    ModelLoadPhase,
    ModelRegistryEntry,
    ModelRuntimeRecipe,
    ModelState,
    ModelTopology,
    ProfileLayer,
    ProfileSource,
)
from moe_tools_suite.driftbench import (
    DRIFT_FORMULA_FINGERPRINT,
    SCENARIOS,
    DriftToolCall,
    exact_state_distance,
    parameterize_scenario,
    reliability_horizon,
    run_drift_scenario,
)
from moe_tools_suite.experiments import (
    ExperimentAdapterResult,
    _largest_routing_shifts,
    _parse_tool_call,
    _ProviderTelemetry,
    _record_completion,
    _routing_distance,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.model_registry import model_registry
from moe_tools_suite.profiles import (
    expert_profile_fingerprint,
    legacy_profile_fingerprint,
)
from moe_tools_suite.runtime import CompletionProgress, MockModelRuntime, VllmRuntime
from moe_tools_suite.settings import Settings
from moe_tools_suite.telemetry import encode_npy
from moe_tools_suite.v2_domain import (
    ContextActivationResult,
    CreateExperimentRequest,
    DriftCondition,
    DriftParameters,
    EvaluationResultRecord,
    Experiment,
    ExperimentStatus,
    InterventionContextRef,
    InterventionKind,
    PerformanceSnapshot,
    RunEventKind,
    RunUnitStatus,
    UpdateProfileMetadataRequest,
    WorkloadRunStatus,
    canonical_fingerprint,
)
from sqlalchemy import event as sqlalchemy_event


def _client(tmp_path: Path) -> tuple[TestClient, ResearchLab]:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    return TestClient(app), app.state.lab


def _load_model(client: TestClient) -> dict[str, object]:
    response = client.post("/api/model-sessions", json={"model_id": MODEL_ID})
    assert response.status_code == 202
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    return client.get("/api/model-sessions/current").json()


def _profile(lab: ResearchLab, *, name: str = "Candidate"):
    return lab.create_expert_profile(
        CreateExpertProfileRequest(
            name=name,
            model_id=MODEL_ID,
            profile=ExpertProfile(
                layers={
                    str(layer_id): ProfileLayer(keep=list(range(16)))
                    for layer_id in lab.topology.routed_layer_ids
                }
            ),
            source=ProfileSource.MANUAL,
        )
    )


def test_model_manifest_is_honest_about_unqualified_entries() -> None:
    registry = model_registry()

    assert [entry.id for entry in registry] == [
        MODEL_ID,
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "openai/gpt-oss-120b",
        "zai-org/GLM-4.5-Air",
    ]
    assert registry[0].qualification_status == "qualified"
    assert registry[0].revision == "95a723d08a9490559dae23d0cff1d9466213d989"
    assert registry[0].topology is not None
    for entry in registry[1:]:
        assert entry.enabled is False
        assert entry.qualification_status == "manifest_only"
        assert entry.failure_reason
    assert all(entry.revision is not None for entry in registry[1:4])
    assert all(entry.topology is not None for entry in registry[1:4])
    assert registry[4].revision is None
    assert registry[4].topology is None


def test_model_load_defaults_to_enabled_manifest_and_rejects_disabled(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)

    selected = client.post("/api/model-sessions", json={})
    blocked = client.post(
        "/api/model-sessions",
        json={"model_id": "openai/gpt-oss-120b"},
    )

    assert selected.status_code == 202
    session = client.get("/api/model-sessions/current").json()
    manifest = model_registry()[0]
    assert session["model_id"] == manifest.id
    assert session["model_revision"] == manifest.revision
    assert session["topology"] == manifest.topology.model_dump(mode="json")
    assert session["runtime_recipe"] == manifest.runtime_recipe.model_dump(mode="json")
    assert blocked.status_code == 422
    assert "manifest only" in blocked.json()["detail"].lower()
    assert len(lab.list_model_sessions()) == 1


@pytest.mark.asyncio
async def test_enabled_manifest_selects_its_runtime_topology_and_recipe(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    topology = ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=1,
        routed_layer_ids=[0, 1],
    )
    revision = "2" * 40
    recipe = ModelRuntimeRecipe(
        revision=revision,
        tensor_parallel_size=1,
        max_model_len=2048,
    )
    alternate = ModelRegistryEntry(
        id="test/enabled-moe",
        display_name="Enabled test MoE",
        enabled=True,
        revision=revision,
        topology=topology,
        notes="GPU-free manifest selection fixture.",
        tensor_parallel_size=1,
        qualification_status="experimental",
        runtime_recipe=recipe,
    )
    lab.models.append(alternate)
    try:
        session = await lab.create_model_session(
            CreateModelSessionRequest(model_id=alternate.id)
        )

        assert session.model_id == alternate.id
        assert session.model_revision == revision
        assert session.topology == topology
        assert session.runtime_recipe == recipe
        assert lab.topology == topology
        assert lab.agentic.model_id == alternate.id
        assert lab.agentic.topology == topology
        assert lab.experiments.model_id == alternate.id
        context = lab.current_intervention_context()
        assert context is not None
        assert context.model_id == alternate.id
        restored = lab.store.load_model_sessions()[session.id]
        assert restored.model_revision == revision
        assert restored.topology == topology
        assert restored.runtime_recipe == recipe
    finally:
        await lab.shutdown()


def test_profile_metadata_and_hot_context_change_without_model_reload(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    model_session = _load_model(client)
    saved = _profile(lab)
    original_mask = saved.profile.model_dump(mode="json")

    renamed = client.patch(
        f"/api/profiles/{saved.id}",
        json={"name": "Renamed candidate", "description": "Human metadata"},
    )
    activated = client.post(
        "/api/expert-contexts/activate", json={"profile_id": saved.id}
    )

    assert renamed.status_code == 200
    assert renamed.json()["profile_fingerprint"] == saved.profile_fingerprint
    assert renamed.json()["profile"] == original_mask
    assert activated.status_code == 200
    assert activated.json()["weights_reloaded"] is False
    assert activated.json()["process_id"] is None
    assert (
        client.get("/api/model-sessions/current").json()["id"] == (model_session["id"])
    )
    current = client.get("/api/expert-contexts/current").json()
    assert current["profile_id"] == saved.id
    assert current["profile_fingerprint"] == saved.profile_fingerprint
    reset = client.post(
        "/api/expert-contexts/activate", json={"profile_id": None}
    ).json()
    assert reset["weights_reloaded"] is False
    assert client.get("/api/expert-contexts/current").json()["kind"] == "baseline"


@pytest.mark.asyncio
async def test_legacy_profile_identity_is_preserved_but_runtime_uses_canonical_mask(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
    )
    lab = ResearchLab(settings)
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        saved = _profile(lab)
        legacy_fingerprint = legacy_profile_fingerprint(saved.profile)
        legacy = saved.model_copy(
            update={
                "profile_fingerprint": legacy_fingerprint,
                "profile_fingerprint_version": None,
            }
        )
        lab.profiles[legacy.id] = legacy
        lab.store.save_expert_profile(legacy)
    finally:
        await lab.shutdown()

    restored = ResearchLab(settings)
    try:
        loaded = restored.profiles[legacy.id]
        assert loaded.profile_fingerprint == legacy_fingerprint
        assert loaded.profile_fingerprint_version is None
        renamed = restored.update_profile_metadata(
            loaded.id,
            UpdateProfileMetadataRequest(name="Legacy renamed"),
        )
        assert renamed.profile_fingerprint == legacy_fingerprint

        await restored.create_model_session(CreateModelSessionRequest())
        await restored.activate_expert_context(loaded.id)
        context = restored.current_intervention_context()
        assert context is not None
        assert context.profile_fingerprint == expert_profile_fingerprint(
            loaded.profile,
            restored.topology,
        )
        assert context.profile_fingerprint != legacy_fingerprint
        assert context.metadata["saved_profile_fingerprint"] == legacy_fingerprint
    finally:
        await restored.shutdown()


def test_model_load_exposes_durable_app_observed_phases(tmp_path: Path) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
    )
    client = TestClient(create_app(settings))

    response = client.post("/api/model-sessions", json={"model_id": MODEL_ID})

    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "completed"
    observed = [
        record
        for record in job["phase_history"]
        if record["observability"] == "observed"
    ]
    assert [record["phase"] for record in observed] == [
        "queued",
        "resolving_model",
        "configuring_runtime",
        "ready",
    ]
    assert all(record["status"] == "completed" for record in observed)
    unavailable = [
        record
        for record in job["phase_history"]
        if record["observability"] == "unavailable"
    ]
    assert [record["phase"] for record in unavailable] == [
        "checking_cache",
        "downloading",
        "loading_weights",
        "initializing_distributed_workers",
        "compiling",
        "capturing_graphs",
        "warming",
    ]
    assert all(record["status"] == "unavailable" for record in unavailable)
    assert all(record["bytes_current"] is None for record in unavailable)
    assert all(record["files_current"] is None for record in unavailable)
    restarted = TestClient(create_app(settings))
    restored = restarted.get(f"/api/jobs/{job['id']}").json()
    assert restored["phase_history"] == job["phase_history"]


@pytest.mark.asyncio
async def test_model_load_failure_closes_active_phase(tmp_path: Path) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )

    class FailingStartServer:
        def __init__(self) -> None:
            self.stopped = False

        async def start(self, **kwargs):
            del kwargs
            raise RuntimeError("runtime did not start")

        async def stop(self) -> None:
            self.stopped = True

    server = FailingStartServer()
    lab.server = server  # type: ignore[assignment]
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))

    await lab.execute_job(job.id)

    failed = lab.get_job(job.id)
    assert failed.status == "failed"
    assert [record.phase.value for record in failed.phase_history] == [
        "queued",
        "resolving_model",
        "configuring_runtime",
        "failed",
    ]
    assert [record.status for record in failed.phase_history] == [
        "completed",
        "completed",
        "failed",
        "failed",
    ]
    assert lab.current_model_session() is not None
    assert lab.current_model_session().state == "failed"
    assert server.stopped is True


@pytest.mark.asyncio
async def test_context_sync_failure_stops_managed_process_before_failing(
    tmp_path: Path,
) -> None:
    class ContextFailureServer:
        def __init__(self) -> None:
            self.running = False
            self.stopped = asyncio.Event()

        @property
        def pid(self) -> int | None:
            return 4242 if self.running else None

        async def start(self, **kwargs) -> None:
            kwargs["on_phase"](
                ModelLoadPhase.LAUNCHING_PROCESS,
                "Launching context failure fixture",
            )
            kwargs["on_phase"](
                ModelLoadPhase.WAITING_FOR_READINESS,
                "Readiness succeeded",
            )
            self.running = True

        async def current_context(self) -> dict[str, object]:
            raise RuntimeError("context verification failed")

        async def stop(self) -> None:
            self.running = False
            self.stopped.set()

        async def aclose(self) -> None:
            await self.stop()

    lab = ResearchLab(
        Settings(
            mode="vllm",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    server = ContextFailureServer()
    lab.server = server  # type: ignore[assignment]
    try:
        job = lab.submit_model_session(CreateModelSessionRequest())

        await lab.execute_job(job.id)

        failed = lab.get_job(job.id)
        assert server.stopped.is_set()
        assert server.running is False
        assert failed.status is JobStatus.FAILED
        assert failed.phase_history[-1].phase is ModelLoadPhase.FAILED
        assert job.result_id is not None
        assert lab.store.load_model_sessions()[job.result_id].state is ModelState.FAILED
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_masked_cold_start_fingerprint_mismatch_stops_before_ready(
    tmp_path: Path,
) -> None:
    class MismatchedProfileServer:
        def __init__(self) -> None:
            self.running = False
            self.stopped = asyncio.Event()

        @property
        def pid(self) -> int | None:
            return 4242 if self.running else None

        async def start(self, **kwargs) -> None:
            kwargs["on_phase"](
                ModelLoadPhase.LAUNCHING_PROCESS,
                "Launching profile mismatch fixture",
            )
            kwargs["on_phase"](
                ModelLoadPhase.WAITING_FOR_READINESS,
                "Readiness succeeded",
            )
            self.running = True

        async def current_context(self) -> dict[str, object]:
            return {
                "active_context_id": "startup-profile",
                "active_context_fingerprint": "a" * 64,
                "profile_fingerprint": "b" * 64,
                "topology_fingerprint": "c" * 64,
                "process_id": 4242,
            }

        async def stop(self) -> None:
            self.running = False
            self.stopped.set()

        async def aclose(self) -> None:
            await self.stop()

    lab = ResearchLab(
        Settings(
            mode="vllm",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    server = MismatchedProfileServer()
    lab.server = server  # type: ignore[assignment]
    saved_profile = _profile(lab)
    try:
        submitted = lab.submit_model_session(
            CreateModelSessionRequest(profile_id=saved_profile.id)
        )

        await lab.execute_job(submitted.id)

        failed = lab.get_job(submitted.id)
        assert server.stopped.is_set()
        assert server.running is False
        assert failed.status is JobStatus.FAILED
        assert "different canonical expert mask" in (failed.error or "")
        assert submitted.result_id is not None
        assert (
            lab.store.load_model_sessions()[submitted.result_id].state
            is ModelState.FAILED
        )
        assert lab.active_context is None
        assert lab.store.load_intervention_contexts() == {}
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_cancellation_during_context_sync_stops_managed_process(
    tmp_path: Path,
) -> None:
    class BlockingContextServer:
        def __init__(self) -> None:
            self.running = False
            self.context_entered = asyncio.Event()
            self.stopped = asyncio.Event()

        @property
        def pid(self) -> int | None:
            return 4242 if self.running else None

        async def start(self, **kwargs) -> None:
            kwargs["on_phase"](
                ModelLoadPhase.LAUNCHING_PROCESS,
                "Launching context cancellation fixture",
            )
            kwargs["on_phase"](
                ModelLoadPhase.WAITING_FOR_READINESS,
                "Readiness succeeded",
            )
            self.running = True

        async def current_context(self) -> dict[str, object]:
            self.context_entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def stop(self) -> None:
            self.running = False
            self.stopped.set()

        async def aclose(self) -> None:
            await self.stop()

    lab = ResearchLab(
        Settings(
            mode="vllm",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    server = BlockingContextServer()
    lab.server = server  # type: ignore[assignment]
    try:
        submitted = lab.submit_model_session(CreateModelSessionRequest())
        execution = lab.start_job(submitted.id)
        await asyncio.wait_for(server.context_entered.wait(), timeout=1)

        lab.cancel_job(submitted.id)
        await asyncio.gather(execution, return_exceptions=True)

        cancelled = lab.get_job(submitted.id)
        assert server.stopped.is_set()
        assert server.running is False
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.phase_history[-1].phase is ModelLoadPhase.CANCELLED
        assert lab.current_model_session() is None
        assert submitted.result_id is not None
        assert (
            lab.store.load_model_sessions()[submitted.result_id].state
            is ModelState.STOPPED
        )
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_cancellation_while_stopping_previous_session_is_consistent(
    tmp_path: Path,
) -> None:
    class BlockingStopServer:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.first_stop_entered = asyncio.Event()
            self.stopped = asyncio.Event()

        async def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                self.first_stop_entered.set()
                await asyncio.Event().wait()
            self.stopped.set()

        async def aclose(self) -> None:
            await self.stop()

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    previous = await lab.create_model_session(CreateModelSessionRequest())
    server = BlockingStopServer()
    lab.server = server  # type: ignore[assignment]
    try:
        submitted = lab.submit_model_session(CreateModelSessionRequest())
        execution = lab.start_job(submitted.id)
        await asyncio.wait_for(server.first_stop_entered.wait(), timeout=1)

        lab.cancel_job(submitted.id)
        await asyncio.gather(execution, return_exceptions=True)

        assert server.stopped.is_set()
        assert previous.state is ModelState.STOPPED
        assert lab.store.load_model_sessions()[previous.id].state is ModelState.STOPPED
        assert submitted.result_id is not None
        assert (
            lab.store.load_model_sessions()[submitted.result_id].state
            is ModelState.STOPPED
        )
        assert lab.current_model_session() is None
        assert lab.get_job(submitted.id).status is JobStatus.CANCELLED
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_running_model_load_cancellation_stops_owned_runtime_and_is_durable(
    tmp_path: Path,
) -> None:
    class CancellableServer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()

        async def start(self, **kwargs) -> None:
            kwargs["on_phase"](
                ModelLoadPhase.LAUNCHING_PROCESS,
                "Launching cancellation fixture",
            )
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.shield(self.stop())
                raise

        async def stop(self) -> None:
            self.stopped.set()

        async def aclose(self) -> None:
            await self.stop()

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    server = CancellableServer()
    lab.server = server  # type: ignore[assignment]
    try:
        submitted = lab.submit_model_session(CreateModelSessionRequest())
        task = lab.start_job(submitted.id)
        await asyncio.wait_for(server.started.wait(), timeout=1)

        cancelling = lab.cancel_job(submitted.id)
        await asyncio.gather(task, return_exceptions=True)

        assert cancelling.status is JobStatus.CANCELLING
        assert server.stopped.is_set()
        cancelled = lab.get_job(submitted.id)
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.phase_history[-1].phase is ModelLoadPhase.CANCELLED
        assert cancelled.phase_history[-1].status == "cancelled"
        assert lab.current_model_session() is None
        assert submitted.result_id is not None
        persisted_session = lab.store.load_model_sessions()[submitted.result_id]
        persisted_job = lab.store.load_jobs()[submitted.id][0]
        assert persisted_session.state == "stopped"
        assert persisted_job.status is JobStatus.CANCELLED
        assert persisted_job.phase_history[-1].phase is ModelLoadPhase.CANCELLED
    finally:
        await lab.shutdown()


def test_unowned_running_model_load_cancellation_does_not_mutate_job(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    submitted = lab.submit_model_session(CreateModelSessionRequest())
    artifacts = lab.jobs[submitted.id]
    artifacts.record.status = JobStatus.RUNNING
    lab.store.save_job(artifacts.record, artifacts.payload)

    with pytest.raises(ValueError, match="not owned"):
        lab.cancel_job(submitted.id)

    assert lab.get_job(submitted.id).status is JobStatus.RUNNING
    persisted, _payload = lab.store.load_jobs()[submitted.id]
    assert persisted.status is JobStatus.RUNNING


def test_queued_model_load_cancellation_records_prelaunch_detail(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    submitted = lab.submit_model_session(CreateModelSessionRequest())

    cancelled = lab.cancel_job(submitted.id)

    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.phase_history[-1].phase is ModelLoadPhase.CANCELLED
    assert cancelled.phase_history[-1].detail == (
        "Model-load request cancelled before runtime launch"
    )
    persisted, _payload = lab.store.load_jobs()[submitted.id]
    assert persisted.phase_history[-1].detail == cancelled.phase_history[-1].detail


@pytest.mark.asyncio
async def test_failed_activation_preserves_prior_ready_context(tmp_path: Path) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    saved = _profile(lab)
    previous = lab.current_intervention_context()

    class FailingServer:
        pid = 4242

        async def context_capabilities(self):
            raise RuntimeError("activation controller unavailable")

    lab.server = FailingServer()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="controller unavailable"):
        await lab.activate_expert_context(saved.id)

    assert lab.current_intervention_context() == previous
    assert lab.session is not None
    assert lab.session.state == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("profile", "different canonical expert mask"),
        ("topology", "different topology"),
        ("layers", "different canonical expert-mask layers"),
    ],
)
async def test_registration_rejects_mismatched_runtime_mask_identity(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    saved = _profile(lab)
    previous = lab.current_intervention_context()
    fork_topology = "c" * 64

    class MismatchedRegistrationServer:
        pid = 4242
        activated = False

        async def context_capabilities(self):
            return {
                "supported": True,
                "topology_fingerprint": fork_topology,
            }

        async def register_context(self, **kwargs):
            result = {
                "context_id": kwargs["context_id"],
                "context_fingerprint": "d" * 64,
                "profile_fingerprint": kwargs["metadata"]["profile_fingerprint"],
                "topology_fingerprint": fork_topology,
                "layers": kwargs["layers"],
            }
            if mismatch == "profile":
                result["profile_fingerprint"] = "e" * 64
            elif mismatch == "topology":
                result["topology_fingerprint"] = "e" * 64
            else:
                result["layers"] = {}
            return result

        async def activate_context(self, context_id):
            del context_id
            self.activated = True
            raise AssertionError("mismatched registration must not activate")

    server = MismatchedRegistrationServer()
    lab.server = server  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match=message):
        await lab.activate_expert_context(saved.id)

    assert server.activated is False
    assert lab.current_intervention_context() == previous


@pytest.mark.asyncio
async def test_ambiguous_activation_reconciles_previous_context_and_raises(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    saved = _profile(lab)
    previous = lab.current_intervention_context()
    assert previous is not None

    class PreviousContextServer:
        pid = 4242

        async def context_capabilities(self):
            return {
                "supported": True,
                "topology_fingerprint": previous.topology_fingerprint,
            }

        async def register_context(self, **kwargs):
            return {
                "context_id": kwargs["context_id"],
                "context_fingerprint": "d" * 64,
                "profile_fingerprint": kwargs["metadata"]["profile_fingerprint"],
                "topology_fingerprint": previous.topology_fingerprint,
                "layers": kwargs["layers"],
            }

        async def activate_context(self, context_id):
            del context_id
            raise RuntimeError("activation response timed out")

        async def current_context(self):
            return {
                "active_context_id": previous.context_id,
                "active_context_fingerprint": previous.context_fingerprint,
                "profile_fingerprint": previous.profile_fingerprint,
                "topology_fingerprint": previous.topology_fingerprint,
                "process_id": 4242,
                "weights_reloaded": False,
                "worker_process_ids": [4242],
                "serving_safe": True,
                "recovery_required": False,
            }

    lab.server = PreviousContextServer()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="timed out"):
        await lab.activate_expert_context(saved.id)

    assert lab.current_intervention_context() == previous
    assert lab.session is not None
    assert lab.session.state == "ready"


@pytest.mark.asyncio
async def test_ambiguous_activation_reconciles_committed_target(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    saved = _profile(lab)
    fork_context_fingerprint = "d" * 64
    fork_topology_fingerprint = "c" * 64
    current_calls = 0
    activation_calls = 0

    class AmbiguousServer:
        pid = 4242

        async def context_capabilities(self):
            return {
                "supported": True,
                "topology_fingerprint": fork_topology_fingerprint,
            }

        async def register_context(self, **kwargs):
            return {
                "context_id": kwargs["context_id"],
                "context_fingerprint": fork_context_fingerprint,
                "profile_fingerprint": kwargs["metadata"]["profile_fingerprint"],
                "topology_fingerprint": fork_topology_fingerprint,
                "layers": kwargs["layers"],
            }

        async def activate_context(self, context_id):
            nonlocal activation_calls
            activation_calls += 1
            del context_id
            if activation_calls == 1:
                raise RuntimeError("activation response timed out")
            return ContextActivationResult(
                old_context_id="fork-mask-a",
                old_context_fingerprint=fork_context_fingerprint,
                new_context_id="fork-mask-a",
                new_context_fingerprint=fork_context_fingerprint,
                topology_fingerprint=fork_topology_fingerprint,
                duration_ms=1,
                process_id=4242,
                weights_reloaded=False,
            )

        async def current_context(self):
            nonlocal current_calls
            current_calls += 1
            return {
                "active_context_id": "fork-mask-a",
                "active_context_fingerprint": fork_context_fingerprint,
                "profile_fingerprint": saved.profile_fingerprint,
                "topology_fingerprint": fork_topology_fingerprint,
                "process_id": 4242,
                "weights_reloaded": False,
                "worker_process_ids": [4242],
                "serving_safe": False,
                "recovery_required": True,
            }

    lab.server = AmbiguousServer()  # type: ignore[assignment]

    receipt = await lab.activate_expert_context(saved.id)

    assert current_calls == 1
    assert activation_calls == 2
    assert receipt.new_context_fingerprint == fork_context_fingerprint
    assert lab.current_intervention_context() is not None
    assert (
        lab.current_intervention_context().context_fingerprint
        == fork_context_fingerprint
    )
    assert lab.session is not None
    assert lab.session.state == "ready"


def test_workload_library_is_task_level_and_fail_closed(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    workloads = client.get("/api/workloads")

    assert workloads.status_code == 200
    records = workloads.json()
    coding = [record for record in records if record["kind"] == "coding"]
    coding_tasks = [record for record in coding if record["task_id"] is not None]
    coding_cohorts = [record for record in coding if record["task_id"] is None]
    drift = next(record for record in records if record["id"] == "state-drift-v1")
    assert coding
    assert coding_tasks
    assert coding_cohorts
    assert all(record["task_id"] for record in coding_tasks)
    assert all(record["unit_ids"] for record in coding_cohorts)
    assert any(len(record["unit_ids"]) > 1 for record in coding_cohorts)
    assert all("tools" in record and "runtime" in record for record in coding)
    unsupported = next(
        record
        for record in coding
        if record["parent_id"] == "terminal-bench-2.1-engineering-15"
    )
    assert unsupported["ready"] is False
    assert unsupported["preparation_status"] == "source_pinned"
    assert unsupported["blocked_reason"]
    assert drift["ready"] is True
    assert drift["metadata"]["formula_fingerprint"] == (DRIFT_FORMULA_FINGERPRINT)


def test_answer_workload_uses_common_paired_experiment_layer(tmp_path: Path) -> None:
    client, lab = _client(tmp_path)
    model_session = _load_model(client)
    saved = _profile(lab)
    workload_id = next(
        record["id"]
        for record in client.get("/api/workloads").json()
        if record["id"] == "answer:fixture-arithmetic:arith-03"
    )

    created = client.post(
        "/api/experiments",
        json={
            "name": "Paired answer",
            "model_id": MODEL_ID,
            "workload_id": workload_id,
            "candidate_profile_id": saved.id,
            "seeds": [7],
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    assert detail["experiment"]["status"] == "completed"
    assert detail["experiment"]["total_units"] == 2
    assert detail["experiment"]["comparison"] == {
        "paired_observations": 1,
        "aligned_pairs": 1,
        "unscored_pairs": 0,
        "mean_score_delta": 0.0,
        "regressions": 0,
        "recoveries": 0,
        "retained_passes": 1,
        "retained_failures": 0,
    }
    assert len(detail["workload_runs"]) == 2
    assert len(detail["units"]) == 2
    for unit in detail["units"]:
        assert unit["status"] == "passed"
        assert unit["evaluation"]["score"] == 1
        assert len(unit["evaluation"]["formula_fingerprint"]) == 64
        assert unit["result"]["model_session_id"] == model_session["id"]
        assert unit["result"]["item"]["output"] == "19"
        assert unit["result"]["prompt"]
        assert unit["result"]["output"] == "19"
        assert unit["result"]["routing"]["total_routed_slots"] > 0
        assert unit["result"]["context"]["context_fingerprint"] in {
            run["context"]["context_fingerprint"] for run in detail["workload_runs"]
        }
    events = client.get(f"/api/experiments/{created.json()['id']}/events").json()[
        "events"
    ]
    assert sum(event["kind"] == "context_switched" for event in events) == 2
    assert sum(event["kind"] == "model_request_completed" for event in events) == 2


@pytest.mark.asyncio
async def test_answer_inference_progress_is_durable_before_completion(
    tmp_path: Path,
) -> None:
    class BlockingProgressRuntime:
        def __init__(self, delegate: MockModelRuntime) -> None:
            self.delegate = delegate
            self.progressed = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, *args, on_progress=None, **kwargs):
            assert on_progress is not None
            on_progress(
                CompletionProgress(
                    prompt_tokens=8,
                    completion_tokens=2,
                    total_tokens=10,
                    elapsed_ms=125,
                    current_tps=16,
                )
            )
            self.progressed.set()
            await self.release.wait()
            return await self.delegate.complete(*args, **kwargs)

        async def aclose(self) -> None:
            return None

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    runtime: BlockingProgressRuntime | None = None
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        assert isinstance(lab.runtime, MockModelRuntime)
        runtime = BlockingProgressRuntime(lab.runtime)
        lab.runtime = runtime  # type: ignore[assignment]
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="answer:fixture-arithmetic:arith-03",
                seeds=[0],
            )
        )
        execution = asyncio.create_task(lab.execute_job(experiment.job_id))
        await asyncio.wait_for(runtime.progressed.wait(), timeout=1)

        detail = lab.get_experiment(experiment.id)
        running = detail.units[0]
        assert running.status is RunUnitStatus.RUNNING
        assert running.performance is not None
        assert running.performance.prompt_tokens == 8
        assert running.performance.completion_tokens == 2
        assert running.performance.tokens_per_second == 16
        persisted = lab.store.load_run_units()[running.id]
        assert persisted.performance is not None
        assert persisted.performance.tokens_per_second == 16
        progress_events = [
            event
            for event in lab.get_experiment_events(experiment.id).events
            if event.kind is RunEventKind.INFERENCE_PROGRESS
        ]
        assert len(progress_events) == 1
        assert progress_events[0].data["current_tps"] == 16
        assert progress_events[0].data["completion_tokens"] == 2
        replay = lab.get_experiment_events(
            experiment.id, after=progress_events[0].sequence - 1
        )
        assert replay.events[0].kind is RunEventKind.INFERENCE_PROGRESS
        assert replay.events[0].data["current_tps"] == 16

        runtime.release.set()
        await execution
        completed = lab.get_experiment(experiment.id).units[0]
        assert completed.status is RunUnitStatus.PASSED
        assert completed.result is not None
        assert completed.result["output"] == "19"
        assert completed.result["routing"]["total_routed_slots"] > 0
    finally:
        if runtime is not None:
            runtime.release.set()
        await lab.shutdown()


def test_answer_cohort_persists_selected_units_and_contracts(tmp_path: Path) -> None:
    client, _lab = _client(tmp_path)
    _load_model(client)

    created = client.post(
        "/api/experiments",
        json={
            "name": "Selected answer cohort",
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic",
            "workload_unit_ids": ["arith-03", "arith-04"],
            "seeds": [5],
            "generation": {
                "temperature": 0,
                "max_tokens": 32,
                "seed": 5,
                "enable_thinking": False,
            },
            "execution_policy": {
                "attempts": 1,
                "concurrency": 1,
                "timeout_seconds": 60,
                "per_item_timeout_seconds": 10,
                "max_tokens": 10000,
            },
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    experiment = detail["experiment"]
    assert experiment["status"] == "completed"
    assert experiment["total_units"] == 2
    assert len(experiment["cohort_fingerprint"]) == 64
    assert experiment["generation"]["max_tokens"] == 32
    assert experiment["execution_policy"]["max_tokens"] == 10000
    assert len(experiment["evaluation_contract"]["fingerprint"]) == 64
    assert [unit["workload_unit_id"] for unit in detail["units"]] == [
        "arith-03",
        "arith-04",
    ]
    assert [unit["result"]["output"] for unit in detail["units"]] == ["19", "23"]
    assert all(
        unit["result"]["evaluation_contract_fingerprint"]
        == experiment["evaluation_contract"]["fingerprint"]
        for unit in detail["units"]
    )
    events = client.get(f"/api/experiments/{experiment['id']}/events").json()["events"]
    current = [event for event in events if event["kind"] == "current_unit"]
    assert [event["data"]["workload_unit_id"] for event in current] == [
        "arith-03",
        "arith-04",
    ]
    assert sum(event["kind"] == "evaluating" for event in events) == 2


def test_answer_cohort_rejects_unknown_or_nondeterministic_contracts(
    tmp_path: Path,
) -> None:
    client, _lab = _client(tmp_path)
    _load_model(client)
    base = {
        "model_id": MODEL_ID,
        "workload_id": "answer:fixture-arithmetic",
        "seeds": [0],
    }

    unknown = client.post(
        "/api/experiments",
        json={**base, "workload_unit_ids": ["not-an-item"]},
    )
    stochastic = client.post(
        "/api/experiments",
        json={
            **base,
            "generation": {
                "temperature": 0.5,
                "max_tokens": 32,
                "seed": 0,
                "enable_thinking": False,
            },
        },
    )

    assert unknown.status_code == 422
    assert "unknown workload units" in unknown.text
    assert stochastic.status_code == 422
    assert "temperature=0" in stochastic.text


def test_answer_cohort_rejects_oversized_descriptor_default_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    descriptor = lab.get_workload_descriptor("answer:fixture-arithmetic").model_copy(
        update={
            "id": "answer:oversized-fixture",
            "unit_ids": [f"oversized-{index}" for index in range(10_001)],
        }
    )
    monkeypatch.setattr(lab, "get_workload_descriptor", lambda _workload_id: descriptor)
    jobs_before = set(lab.jobs)

    response = client.post(
        "/api/experiments",
        json={
            "model_id": MODEL_ID,
            "workload_id": descriptor.id,
            "seeds": [0],
        },
    )

    assert response.status_code == 422
    assert "maximum of 10,000 units" in response.text
    assert set(lab.jobs) == jobs_before


def test_expanded_answer_cohort_enforces_run_unit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    profile = _profile(lab)
    unit_ids = [f"expanded-{index}" for index in range(251)]
    descriptor = lab.get_workload_descriptor("answer:fixture-arithmetic").model_copy(
        update={"id": "answer:expanded-fixture", "unit_ids": unit_ids}
    )
    monkeypatch.setattr(lab, "get_workload_descriptor", lambda _workload_id: descriptor)
    boundary = CreateExperimentRequest(
        model_id=MODEL_ID,
        workload_id=descriptor.id,
        workload_unit_ids=unit_ids[:250],
        candidate_profile_id=profile.id,
        seeds=list(range(20)),
    )

    assert lab.experiments.validate_request(boundary, descriptor) == 10_000

    jobs_before = set(lab.jobs)
    over_limit = boundary.model_copy(update={"workload_unit_ids": unit_ids})
    with pytest.raises(ValueError, match="10,040 requested"):
        lab.submit_experiment(over_limit)
    assert set(lab.jobs) == jobs_before


def test_expanded_drift_dimensions_fail_before_job_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    profile = _profile(lab)
    descriptor = lab.get_workload_descriptor("state-drift-v1")
    scenario = next(iter(SCENARIOS.values()))
    scenario_ids = [f"synthetic-drift-{index}" for index in range(20)]
    for scenario_id in scenario_ids:
        monkeypatch.setitem(SCENARIOS, scenario_id, scenario)
    single_lane = CreateExperimentRequest(
        model_id=MODEL_ID,
        workload_id=descriptor.id,
        scenario_ids=scenario_ids,
        conditions=list(DriftCondition),
        horizons=[2, 4, 8, 12, 16, 24],
        seeds=list(range(20)),
    )

    assert lab.experiments.validate_request(single_lane, descriptor) == 7_200

    jobs_before = set(lab.jobs)
    over_limit = single_lane.model_copy(update={"candidate_profile_id": profile.id})
    with pytest.raises(ValueError, match="14,400 requested"):
        lab.submit_experiment(over_limit)
    assert set(lab.jobs) == jobs_before


def test_v2_judge_requires_and_honors_an_explicit_priced_cost_cap(
    tmp_path: Path,
) -> None:
    client, _lab = _client(tmp_path)
    _load_model(client)
    contract = {
        "name": "Capped qualitative judge",
        "criteria": [{"id": "exact", "label": "Exact"}],
        "judge": {
            "provider": "fake",
            "model": "deterministic-fake-judge-v1",
            "rubric": "Judge correctness.",
            "input_cost_per_million_usd": 1,
            "output_cost_per_million_usd": 1,
        },
        "judge_weight": 0.5,
    }
    payload = {
        "model_id": MODEL_ID,
        "workload_id": "answer:fixture-arithmetic",
        "workload_unit_ids": ["arith-03", "arith-04"],
        "evaluation_contract": contract,
        "seeds": [0],
    }

    uncapped = client.post("/api/experiments", json=payload)
    capped = client.post(
        "/api/experiments",
        json={
            **payload,
            "execution_policy": {
                "attempts": 1,
                "concurrency": 1,
                "timeout_seconds": 60,
                "per_item_timeout_seconds": 10,
                "max_cost_usd": 0.000000001,
            },
        },
    )

    assert uncapped.status_code == 422
    assert "explicit max_cost_usd" in uncapped.text
    assert capped.status_code == 202
    detail = client.get(f"/api/experiments/{capped.json()['id']}").json()
    assert {unit["status"] for unit in detail["units"]} == {"unscored"}
    assert all(
        unit["evaluation"]["metrics"]["cost_budget_exhausted"] is True
        for unit in detail["units"]
    )


def test_record_only_contract_makes_answer_cohort_unscored(tmp_path: Path) -> None:
    client, _lab = _client(tmp_path)
    _load_model(client)
    created = client.post(
        "/api/experiments",
        json={
            "name": "Record-only answer",
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic",
            "workload_unit_ids": ["arith-03", "arith-04"],
            "evaluation_contract": {
                "name": "Record only",
                "criteria": [
                    {
                        "id": "recorded-output",
                        "label": "Recorded output",
                        "kind": "ungraded",
                        "required": False,
                    }
                ],
            },
            "seeds": [0],
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    assert detail["experiment"]["completed_units"] == 2
    assert detail["experiment"]["passed_units"] == 0
    assert {unit["status"] for unit in detail["units"]} == {"unscored"}
    assert all(unit["evaluation"]["passed"] is None for unit in detail["units"])


def test_answer_cohort_token_cap_stops_future_units_without_false_failure(
    tmp_path: Path,
) -> None:
    client, _lab = _client(tmp_path)
    _load_model(client)
    created = client.post(
        "/api/experiments",
        json={
            "name": "Token-capped answer",
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic",
            "workload_unit_ids": ["arith-03", "arith-04"],
            "execution_policy": {
                "attempts": 1,
                "concurrency": 1,
                "timeout_seconds": 60,
                "per_item_timeout_seconds": 10,
                "max_tokens": 1,
            },
            "seeds": [0],
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    assert detail["experiment"]["status"] == "completed"
    assert detail["experiment"]["passed_units"] == 0
    assert {unit["status"] for unit in detail["units"]} == {"unscored"}
    assert all(
        unit["evaluation"]["metrics"]["budget_exhausted"] is True
        for unit in detail["units"]
    )


def test_paired_unscored_units_are_not_reported_as_observations(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    profile = _profile(lab)
    created = client.post(
        "/api/experiments",
        json={
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic",
            "workload_unit_ids": ["arith-03", "arith-04"],
            "candidate_profile_id": profile.id,
            "execution_policy": {
                "attempts": 1,
                "concurrency": 1,
                "timeout_seconds": 60,
                "per_item_timeout_seconds": 10,
                "max_tokens": 1,
            },
            "seeds": [0],
        },
    )

    assert created.status_code == 202
    comparison = client.get(f"/api/experiments/{created.json()['id']}").json()[
        "experiment"
    ]["comparison"]
    assert comparison["aligned_pairs"] == 2
    assert comparison["paired_observations"] == 0
    assert comparison["unscored_pairs"] == 2


@pytest.mark.asyncio
async def test_coding_prompt_starvation_is_unscored_and_skips_verifier(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="coding:smoke-python-v1",
                workload_unit_ids=["fix-subtract", "implement-slugify"],
                sandbox_provider_id="fake",
                execution_policy={
                    "attempts": 1,
                    "concurrency": 1,
                    "timeout_seconds": 60,
                    "per_item_timeout_seconds": 10,
                    "max_tokens": 1,
                    "max_turns": 2,
                    "max_commands": 2,
                },
                seeds=[0],
            )
        )
        await lab.execute_job(experiment.job_id)

        detail = lab.get_experiment(experiment.id)
        assert {unit.status for unit in detail.units} == {RunUnitStatus.UNSCORED}
        first_trial = next(iter(lab.agentic.trials.values()))
        assert first_trial.termination_cause == "token_limit"
        assert first_trial.verifier is None
        assert len(lab.agentic.trials) == 1
    finally:
        await lab.shutdown()


def test_legacy_experiment_contract_and_phase_provenance_stay_unknown(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    created = client.post(
        "/api/experiments",
        json={
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic:arith-03",
            "seeds": [0],
        },
    ).json()
    current = lab.experiments.get(created["id"])
    legacy_payload = current.model_dump(
        mode="json",
        exclude={
            "contract_provenance",
            "generation",
            "evaluation_contract",
            "execution_policy",
        },
    )

    restored = Experiment.model_validate(legacy_payload)
    legacy_phase = JobPhaseRecord.model_validate({"phase": "queued"})

    assert restored.contract_provenance == "legacy_unknown"
    assert restored.generation is None
    assert restored.evaluation_contract is None
    assert restored.execution_policy is None
    assert legacy_phase.source == "legacy_unknown"


def test_ungraded_answer_completes_without_counting_a_pass(tmp_path: Path) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)

    async def ungraded(request):
        return ExperimentAdapterResult(
            passed=None,
            score=1,
            result={
                "kind": "answer",
                "prompt": "Record a qualitative answer.",
                "output": "A recorded response",
                "scoring": "ungraded",
            },
            evaluation=EvaluationResultRecord(
                id="evaluation-ungraded",
                workload_run_id=request.workload_run.id,
                run_unit_id=request.unit.id,
                passed=None,
                score=1,
            ),
            performance=PerformanceSnapshot(
                id="performance-ungraded",
                workload_run_id=request.workload_run.id,
                run_unit_id=request.unit.id,
            ),
        )

    lab.experiments._answer_executor = ungraded
    created = client.post(
        "/api/experiments",
        json={
            "name": "Ungraded answer",
            "model_id": MODEL_ID,
            "workload_id": "answer:fixture-arithmetic:arith-03",
            "seeds": [9],
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    assert detail["experiment"]["completed_units"] == 1
    assert detail["experiment"]["passed_units"] == 0
    assert detail["units"][0]["status"] == "unscored"
    assert detail["units"][0]["evaluation"]["score"] is None
    assert detail["workload_runs"][0]["aggregate_metrics"] == {
        "mean_score": None,
        "pass_rate": None,
        "completed_units": 1,
        "scored_units": 0,
    }
    events = client.get(f"/api/experiments/{created.json()['id']}/events").json()
    assert any(event["kind"] == "unit_unscored" for event in events["events"])


@pytest.mark.asyncio
async def test_adapter_error_and_failed_event_are_one_terminal_transition(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))

        async def fail_adapter(_request):
            raise RuntimeError("adapter exploded")

        lab.experiments._answer_executor = fail_adapter
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="answer:fixture-arithmetic:arith-03",
                seeds=[0],
            )
        )

        await lab.execute_job(experiment.job_id)

        detail = lab.get_experiment(experiment.id)
        events = lab.get_experiment_events(experiment.id).events
        failed = events[-1]
        assert detail.experiment.status is ExperimentStatus.FAILED
        assert detail.experiment.error == "adapter exploded"
        assert detail.units[0].status is RunUnitStatus.ERROR
        assert detail.units[0].result == {"error": "adapter exploded"}
        assert detail.workload_runs[0].status is WorkloadRunStatus.FAILED
        assert failed.kind is RunEventKind.FAILED
        assert failed.run_unit_id == detail.units[0].id
        assert failed.workload_run_id == detail.workload_runs[0].id
        assert failed.sequence == detail.experiment.latest_event_sequence

        persisted_experiment = lab.store.load_experiments()[experiment.id]
        persisted_unit = lab.store.load_run_units()[detail.units[0].id]
        persisted_events, latest, _has_more = lab.store.load_run_events(experiment.id)
        assert persisted_experiment.status is ExperimentStatus.FAILED
        assert persisted_unit.status is RunUnitStatus.ERROR
        assert persisted_events[-1].kind is RunEventKind.FAILED
        assert persisted_events[-1].run_unit_id == persisted_unit.id
        assert latest == persisted_experiment.latest_event_sequence
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_save_experiment_event_rolls_back_state_when_event_insert_fails(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                name="Persisted name",
                model_id=MODEL_ID,
                workload_id="answer:fixture-arithmetic:arith-03",
                seeds=[0],
            )
        )
        run = next(
            item
            for item in lab.experiments.workload_runs.values()
            if item.experiment_id == experiment.id
        )
        unit = next(
            item
            for item in lab.experiments.units.values()
            if item.experiment_id == experiment.id
        )
        experiment.name = "Must roll back"
        experiment.status = ExperimentStatus.RUNNING
        run.status = WorkloadRunStatus.RUNNING
        unit.status = RunUnitStatus.RUNNING

        def fail_event_insert(
            _connection,
            _cursor,
            statement,
            _parameters,
            _context,
            _executemany,
        ) -> None:
            if "INSERT INTO run_events" in statement:
                raise RuntimeError("injected event insert failure")

        sqlalchemy_event.listen(
            lab.store.engine,
            "before_cursor_execute",
            fail_event_insert,
        )
        try:
            with pytest.raises(RuntimeError, match="injected event insert failure"):
                lab.store.save_experiment_event(
                    experiment,
                    kind=RunEventKind.PREPARING,
                    phase="preparing",
                    message="must not become observable",
                    workload_runs=[run],
                    unit=unit,
                )
        finally:
            sqlalchemy_event.remove(
                lab.store.engine,
                "before_cursor_execute",
                fail_event_insert,
            )

        assert experiment.latest_event_sequence == 1
        persisted_experiment = lab.store.load_experiments()[experiment.id]
        persisted_run = lab.store.load_workload_runs()[run.id]
        persisted_unit = lab.store.load_run_units()[unit.id]
        persisted_events, latest, _has_more = lab.store.load_run_events(experiment.id)
        assert persisted_experiment.name == "Persisted name"
        assert persisted_experiment.status is ExperimentStatus.QUEUED
        assert persisted_run.status is WorkloadRunStatus.QUEUED
        assert persisted_unit.status is RunUnitStatus.QUEUED
        assert [event.kind for event in persisted_events] == [RunEventKind.QUEUED]
        assert latest == 1
    finally:
        await lab.shutdown()


def test_coding_workload_reuses_clean_room_controller_without_nested_job(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    model_session = _load_model(client)
    saved = _profile(lab)
    workload_id = "coding:smoke-python-v1:fix-subtract"
    descriptor = next(
        record
        for record in client.get("/api/workloads").json()
        if record["id"] == workload_id
    )
    assert descriptor["ready"] is True

    created = client.post(
        "/api/experiments",
        json={
            "name": "Paired coding",
            "model_id": MODEL_ID,
            "workload_id": workload_id,
            "candidate_profile_id": saved.id,
            "agent_id": "bash-json-v1",
            "sandbox_provider_id": "fake",
            "seeds": [3],
        },
    )

    assert created.status_code == 202
    detail = client.get(f"/api/experiments/{created.json()['id']}").json()
    assert detail["experiment"]["status"] == "completed"
    assert detail["experiment"]["total_units"] == 2
    assert detail["experiment"]["comparison"]["retained_passes"] == 1
    assert len(lab.list_jobs()) == 2
    assert {job.kind.value for job in lab.list_jobs()} == {
        "model_load",
        "experiment_run",
    }
    for unit in detail["units"]:
        assert unit["status"] == "passed"
        result = unit["result"]
        assert result["kind"] == "coding"
        assert result["model_session_id"] == model_session["id"]
        assert (
            result["routing"]["context_fingerprint"]
            == (result["context"]["context_fingerprint"])
        )
        assert all(
            artifact["context_fingerprint"] == result["context"]["context_fingerprint"]
            for artifact in result["routing"]["artifacts"]
        )
        assert result["artifact_endpoints"]["trajectory"].startswith("/api/trials/")
        assert {step["type"] for step in result["trajectory"]} >= {
            "assistant",
            "tool",
            "observation",
            "verifier",
        }
    events = client.get(f"/api/experiments/{created.json()['id']}/events").json()[
        "events"
    ]
    trajectory_events = [
        event for event in events if event["kind"] == "trajectory_step"
    ]
    assert trajectory_events
    assert {
        event["data"]["trajectory_step"]["type"] for event in trajectory_events
    } >= {"assistant", "tool", "observation", "verifier"}
    assert lab.agentic.providers["fake"]._sandboxes == {}


@pytest.mark.asyncio
async def test_coding_experiment_cancellation_cleans_active_sandbox(
    tmp_path: Path,
) -> None:
    class PausingFakeProvider(FakeSandboxProvider):
        def __init__(self) -> None:
            super().__init__()
            self.created = asyncio.Event()
            self.release = asyncio.Event()

        async def create(self, spec, ownership):
            handle = await super().create(spec, ownership)
            self.created.set()
            await self.release.wait()
            return handle

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        provider = PausingFakeProvider()
        lab.agentic.providers["fake"] = provider
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="coding:smoke-python-v1:fix-subtract",
                sandbox_provider_id="fake",
                seeds=[0],
            )
        )

        execution = asyncio.create_task(lab.execute_job(experiment.job_id))
        await provider.created.wait()
        async with asyncio.timeout(1):
            while not any(
                event.kind is RunEventKind.TRAJECTORY_STEP
                for event in lab.get_experiment_events(experiment.id).events
            ):
                await asyncio.sleep(0.01)
        lab.cancel_experiment(experiment.id)
        provider.release.set()
        await execution

        assert lab.get_job(experiment.job_id).status == "cancelled"
        detail = lab.get_experiment(experiment.id)
        assert detail.experiment.status == "cancelled"
        assert detail.units[0].status == "cancelled"
        assert provider._sandboxes == {}
        agent_run = next(iter(lab.agentic.runs.values()))
        assert agent_run.job_id == experiment.job_id
        assert agent_run.status == "cancelled"
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_coding_tool_events_are_ordered_correlated_and_truthfully_typed(
    tmp_path: Path,
) -> None:
    class BlockingExecProvider(FakeSandboxProvider):
        def __init__(self) -> None:
            super().__init__()
            self.command_started = asyncio.Event()
            self.release_command = asyncio.Event()

        async def exec(self, *args, **kwargs):
            self.command_started.set()
            await self.release_command.wait()
            result = await super().exec(*args, **kwargs)
            return result.model_copy(update={"stderr": "correlated stderr"})

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        provider = BlockingExecProvider()
        lab.agentic.providers["fake"] = provider
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="coding:smoke-python-v1:fix-subtract",
                sandbox_provider_id="fake",
                seeds=[0],
            )
        )

        execution = asyncio.create_task(lab.execute_job(experiment.job_id))
        await asyncio.wait_for(provider.command_started.wait(), timeout=1)
        events_while_running = lab.get_experiment_events(experiment.id).events
        starts_while_running = [
            event
            for event in events_while_running
            if event.kind is RunEventKind.TOOL_COMMAND_STARTED
        ]
        assert len(starts_while_running) == 1
        tool_call_id = starts_while_running[0].data["tool_call_id"]
        assert isinstance(tool_call_id, str) and tool_call_id
        assert not any(
            event.kind is RunEventKind.TOOL_COMMAND_COMPLETED
            for event in events_while_running
        )

        provider.release_command.set()
        await execution
        events = lab.get_experiment_events(experiment.id).events
        starts = [
            event for event in events if event.kind is RunEventKind.TOOL_COMMAND_STARTED
        ]
        completions = [
            event
            for event in events
            if event.kind is RunEventKind.TOOL_COMMAND_COMPLETED
        ]
        verifier_evaluations = [
            event
            for event in events
            if event.kind is RunEventKind.EVALUATING
            and event.data.get("trajectory_step_type") == "verifier"
        ]
        assert len(starts) == len(completions) == len(verifier_evaluations) == 1
        assert starts[0].data == {
            "trajectory_step_type": "tool_start",
            "tool_call_id": tool_call_id,
        }
        assert completions[0].data == {
            "trajectory_step_type": "observation",
            "tool_call_id": tool_call_id,
            "source_call_id": tool_call_id,
        }
        assert starts[0].sequence < completions[0].sequence
        assert completions[0].sequence < verifier_evaluations[0].sequence
        assert not any(
            event.kind is RunEventKind.TOOL_COMMAND_COMPLETED
            and event.data.get("trajectory_step_type") == "verifier"
            for event in events
        )

        trajectory_steps = [
            event.data["trajectory_step"]
            for event in events
            if event.kind is RunEventKind.TRAJECTORY_STEP
        ]
        tool_steps = [step for step in trajectory_steps if step["type"] == "tool"]
        observations = [
            step for step in trajectory_steps if step["type"] == "observation"
        ]
        assert {step["tool_call_id"] for step in tool_steps} == {tool_call_id}
        assert {step["source_call_id"] for step in observations} == {tool_call_id}
        assert len(observations) == 2
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_coding_cost_cap_debits_only_conservative_judge_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        original_infer = lab.agentic.gateway.infer

        async def priced_infer(**kwargs):
            result = await original_infer(**kwargs)
            return type(result)(
                content=result.content,
                reasoning=result.reasoning,
                inference=result.inference.model_copy(
                    update={"estimated_cost_usd": 10.0}
                ),
                routing=result.routing,
                aggregated=result.aggregated,
            )

        monkeypatch.setattr(lab.agentic.gateway, "infer", priced_infer)
        original_evaluate = lab.judges.evaluate
        remaining_caps: list[float | None] = []
        upper_bounds: list[float] = []

        async def uncertain_judge(request, **kwargs):
            remaining_caps.append(kwargs.get("max_incurred_cost_usd"))
            result = await original_evaluate(request, **kwargs)
            result = result.model_copy(
                update={
                    "usage": result.usage.model_copy(
                        update={
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "estimated_cost_usd": None,
                            "incurred_cost_usd": None,
                        }
                    )
                }
            )
            assert result.usage.cost_upper_bound_usd is not None
            upper_bounds.append(result.usage.cost_upper_bound_usd)
            return result

        monkeypatch.setattr(lab.judges, "evaluate", uncertain_judge)
        maximum_cost = 1.0
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="coding:smoke-python-v1",
                workload_unit_ids=["fix-subtract", "implement-slugify"],
                sandbox_provider_id="fake",
                evaluation_contract={
                    "name": "Priced coding judge",
                    "criteria": [
                        {
                            "id": "trusted_verifier",
                            "label": "Trusted task verifier",
                        }
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Judge whether the requested fix is correct.",
                        "input_cost_per_million_usd": 1,
                        "output_cost_per_million_usd": 1,
                    },
                    "judge_weight": 0.5,
                },
                execution_policy={"max_cost_usd": maximum_cost},
                seeds=[0],
            )
        )

        await lab.execute_job(experiment.job_id)

        detail = lab.get_experiment(experiment.id)
        assert len(remaining_caps) == len(upper_bounds) == len(detail.units) == 2
        assert remaining_caps[0] == pytest.approx(maximum_cost)
        assert remaining_caps[1] == pytest.approx(maximum_cost - upper_bounds[0])
        assert [
            unit.evaluation.metrics["budget_debit_usd"] for unit in detail.units
        ] == pytest.approx(upper_bounds)
        assert all(
            trial.performance is not None
            and trial.performance.inference_cost_usd is not None
            and trial.performance.inference_cost_usd > maximum_cost
            for trial in lab.agentic.trials.values()
        )
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_coding_cancellation_before_trajectory_is_terminal(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
        entered = asyncio.Event()
        original_execute_run = lab.agentic.execute_run

        async def delayed_execute_run(run_id, **kwargs):
            entered.set()
            while not kwargs["should_cancel"]():
                await asyncio.sleep(0.01)
            return await original_execute_run(run_id, **kwargs)

        lab.agentic.execute_run = delayed_execute_run
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="coding:smoke-python-v1:fix-subtract",
                sandbox_provider_id="fake",
                seeds=[0],
            )
        )

        execution = asyncio.create_task(lab.execute_job(experiment.job_id))
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert lab.agentic.trajectories == {}
        lab.cancel_experiment(experiment.id)
        await execution

        detail = lab.get_experiment(experiment.id)
        assert detail.experiment.status == "cancelled"
        assert detail.units[0].status == "cancelled"
        assert not any(
            event.kind is RunEventKind.TRAJECTORY_STEP
            for event in lab.get_experiment_events(experiment.id).events
        )
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_drift_cancellation_during_model_call_emits_no_completion(
    tmp_path: Path,
) -> None:
    class BlockingRuntime:
        def __init__(self, delegate: MockModelRuntime) -> None:
            self.delegate = delegate
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def complete_chat(self, *args, **kwargs):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return await self.delegate.complete_chat(*args, **kwargs)

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        assert isinstance(lab.runtime, MockModelRuntime)
        runtime = BlockingRuntime(lab.runtime)
        lab.experiments.runtime = runtime  # type: ignore[assignment]
        lab.experiments.mode = "vllm"
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="state-drift-v1",
                scenario_ids=["ledger-reconciliation-v1"],
                conditions=[DriftCondition.CHAINED],
                horizons=[2],
                seeds=[0],
            )
        )
        execution = asyncio.create_task(lab.execute_job(experiment.job_id))
        await asyncio.wait_for(runtime.started.wait(), timeout=1)

        lab.cancel_experiment(experiment.id)
        runtime.release.set()
        await execution

        events = lab.get_experiment_events(experiment.id).events
        assert runtime.calls == 1
        assert (
            sum(event.kind is RunEventKind.MODEL_REQUEST_STARTED for event in events)
            == 1
        )
        assert not any(
            event.kind is RunEventKind.MODEL_REQUEST_COMPLETED for event in events
        )
        assert lab.get_experiment(experiment.id).experiment.status is (
            ExperimentStatus.CANCELLED
        )
        assert lab.get_experiment(experiment.id).units[0].status is (
            RunUnitStatus.CANCELLED
        )
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_drift_failure_wins_concurrent_cancellation(tmp_path: Path) -> None:
    class FailingRuntime:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete_chat(self, *args, **kwargs):
            del args, kwargs
            self.started.set()
            await self.release.wait()
            raise RuntimeError("inference failed after cancellation request")

    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        runtime = FailingRuntime()
        lab.experiments.runtime = runtime  # type: ignore[assignment]
        lab.experiments.mode = "vllm"
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="state-drift-v1",
                scenario_ids=["ledger-reconciliation-v1"],
                conditions=[DriftCondition.CHAINED],
                horizons=[2],
                seeds=[0],
            )
        )
        execution = lab.start_job(experiment.job_id)
        await asyncio.wait_for(runtime.started.wait(), timeout=1)

        lab.cancel_experiment(experiment.id)
        runtime.release.set()
        await execution

        detail = lab.get_experiment(experiment.id)
        events = lab.get_experiment_events(experiment.id).events
        assert detail.experiment.status is ExperimentStatus.FAILED
        assert lab.get_job(experiment.job_id).status is JobStatus.FAILED
        assert events[-1].kind is RunEventKind.FAILED
        assert not any(event.kind is RunEventKind.CANCELLED for event in events)
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_completed_experiment_wins_late_job_cancellation(tmp_path: Path) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="state-drift-v1",
                scenario_ids=["ledger-reconciliation-v1"],
                conditions=[DriftCondition.CHAINED],
                horizons=[2],
                seeds=[0],
            )
        )
        artifacts = lab.jobs[experiment.job_id]
        artifacts.record.status = JobStatus.RUNNING
        artifacts.record.started_at = artifacts.record.created_at
        lab.store.save_job(artifacts.record, artifacts.payload)

        result = await lab.experiments.execute(
            experiment.id,
            on_progress=lambda _current, _total: None,
            should_cancel=lambda: False,
        )
        assert result.status is ExperimentStatus.COMPLETED

        terminal = lab.cancel_job(experiment.job_id)

        assert terminal.status is JobStatus.COMPLETED
        assert lab.get_experiment(experiment.id).experiment.status is (
            ExperimentStatus.COMPLETED
        )
        assert lab.get_job(experiment.job_id).status is JobStatus.COMPLETED
        persisted, _payload = lab.store.load_jobs()[experiment.job_id]
        assert persisted.status is JobStatus.COMPLETED
        assert not any(
            event.kind in {RunEventKind.CANCELLING, RunEventKind.CANCELLED}
            for event in lab.get_experiment_events(experiment.id).events
        )
    finally:
        await lab.shutdown()


def test_paired_drift_experiment_events_and_restart_are_durable(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
    )
    app = create_app(settings)
    lab = app.state.lab
    asyncio.run(lab.create_model_session(CreateModelSessionRequest()))
    saved = _profile(lab)
    experiment = lab.submit_experiment(
        CreateExperimentRequest(
            name="Paired smoke",
            model_id=MODEL_ID,
            workload_id="state-drift-v1",
            candidate_profile_id=saved.id,
            scenario_ids=["ledger-reconciliation-v1"],
            conditions=[DriftCondition.ORACLE_RESET, DriftCondition.CHAINED],
            horizons=[4],
            seeds=[0],
        )
    )
    asyncio.run(lab.execute_job(experiment.job_id))
    experiment_id = experiment.id
    client = TestClient(app)
    detail = lab.get_experiment(experiment_id).model_dump(mode="json")
    assert detail["experiment"]["status"] == "completed"
    assert detail["experiment"]["total_units"] == 4
    assert detail["experiment"]["completed_units"] == 4
    assert len(detail["workload_runs"]) == 2
    assert len(detail["units"]) == 4
    comparison = detail["experiment"]["comparison"]
    assert comparison["paired_observations"] == 2
    assert comparison["excess_compounding_penalty"] > 0
    assert all(
        unit["evaluation"]["formula_fingerprint"] == DRIFT_FORMULA_FINGERPRINT
        for unit in detail["units"]
    )
    page = client.get(f"/api/experiments/{experiment_id}/events").json()
    sequences = [event["sequence"] for event in page["events"]]
    assert sequences == list(range(1, page["latest_sequence"] + 1))
    tail = client.get(
        f"/api/experiments/{experiment_id}/events",
        params={"after": sequences[-2]},
    ).json()
    assert [event["sequence"] for event in tail["events"]] == [sequences[-1]]
    stream = client.get(
        f"/api/experiments/{experiment_id}/events/stream",
        headers={"Last-Event-ID": str(sequences[-2])},
    )
    assert stream.status_code == 200
    assert f"id: {sequences[-1]}" in stream.text
    assert "event: completed" in stream.text

    restarted = TestClient(create_app(settings))
    restored = restarted.get(f"/api/experiments/{experiment_id}").json()
    assert restored["experiment"]["status"] == "completed"
    assert len(restored["units"]) == 4
    restored_events = restarted.get(f"/api/experiments/{experiment_id}/events").json()
    assert restored_events["latest_sequence"] == page["latest_sequence"]


def test_paired_drift_subset_does_not_fabricate_missing_condition_metrics(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    saved = _profile(lab)

    created = client.post(
        "/api/experiments",
        json={
            "name": "Anchored only",
            "model_id": MODEL_ID,
            "workload_id": "state-drift-v1",
            "candidate_profile_id": saved.id,
            "scenario_ids": ["ledger-reconciliation-v1"],
            "conditions": ["state_anchored"],
            "horizons": [2],
            "seeds": [0],
        },
    )

    assert created.status_code == 202
    comparison = client.get(f"/api/experiments/{created.json()['id']}").json()[
        "experiment"
    ]["comparison"]
    assert comparison["baseline_local_competence"] is None
    assert comparison["candidate_local_competence"] is None
    assert comparison["baseline_chained_fidelity"] is None
    assert comparison["candidate_chained_fidelity"] is None
    assert comparison["local_capability_delta"] is None
    assert comparison["baseline_compounding_penalty"] is None
    assert comparison["candidate_compounding_penalty"] is None
    assert comparison["excess_compounding_penalty"] is None


def test_queued_experiment_cancellation_is_terminal_and_durable(
    tmp_path: Path,
) -> None:
    client, lab = _client(tmp_path)
    _load_model(client)
    saved = _profile(lab)
    experiment = lab.submit_experiment(
        CreateExperimentRequest(
            model_id=MODEL_ID,
            workload_id="state-drift-v1",
            candidate_profile_id=saved.id,
            scenario_ids=["ledger-reconciliation-v1"],
            conditions=[DriftCondition.CHAINED],
            horizons=[2],
            seeds=[0],
        )
    )

    cancelled = client.post(f"/api/experiments/{experiment.id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert lab.get_job(experiment.job_id).status == "cancelled"
    events = client.get(f"/api/experiments/{experiment.id}/events").json()
    assert events["events"][-1]["kind"] == "cancelled"


@pytest.mark.asyncio
async def test_interrupted_experiment_restart_appends_terminal_event(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
    )
    first = ResearchLab(settings)
    await first.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    queued = first.submit_experiment(
        CreateExperimentRequest(
            model_id=MODEL_ID,
            workload_id="answer:fixture-arithmetic:arith-03",
            seeds=[0],
        )
    )

    restarted = ResearchLab(settings)
    detail = restarted.get_experiment(queued.id)
    events = restarted.get_experiment_events(queued.id)

    assert detail.experiment.status == "failed"
    assert detail.experiment.latest_event_sequence == 2
    assert all(lane.status == "failed" for lane in detail.experiment.lanes)
    assert all(run.status == "failed" for run in detail.workload_runs)
    assert all(unit.status == "error" for unit in detail.units)
    assert [event.sequence for event in events.events] == [1, 2]
    assert events.events[-1].kind == "failed"
    assert events.events[-1].data == {"reconciled_after_restart": True}


@pytest.mark.asyncio
async def test_drift_formulas_detect_divergence_and_recovery() -> None:
    scenario = SCENARIOS["ledger-reconciliation-v1"]
    plan = scenario.plan(seed=0, horizon=4)

    async def provider(turn):
        expected = plan[turn.checkpoint - 1].canonical_call
        if turn.checkpoint == 1:
            arguments = dict(expected.arguments)
            arguments["amount"] += 1
            return DriftToolCall(tool=expected.tool, arguments=arguments)
        return expected

    result = await run_drift_scenario(
        scenario,
        seed=0,
        horizon=4,
        condition=DriftCondition.CHAINED,
        action_provider=provider,
    )

    assert result.metrics.first_divergence_checkpoint == 1
    assert result.metrics.recovery_rate == 0
    assert result.metrics.rollback_correctness == 0
    assert result.metrics.formula_fingerprint == DRIFT_FORMULA_FINGERPRINT
    assert exact_state_distance({"a": 1}, {"a": 1}) == 0
    assert exact_state_distance({"a": 1}, {"a": 2}) == 1
    assert reliability_horizon({2: [True], 4: [True, False], 8: [False]}, 0.5) == 4


@pytest.mark.asyncio
async def test_drift_dimensions_are_applied_and_fingerprinted() -> None:
    parameters = DriftParameters(
        dependency_span=3,
        branch_count=4,
        rollback_depth=2,
        distractor_ratio=0.25,
        tool_error_rate=0.9,
        state_size=12,
    )
    scenario = parameterize_scenario(SCENARIOS["ledger-reconciliation-v1"], parameters)
    plan = scenario.plan(seed=7, horizon=8)

    async def provider(turn):
        return plan[turn.checkpoint - 1].canonical_call

    result = await run_drift_scenario(
        scenario,
        seed=7,
        horizon=8,
        condition=DriftCondition.CHAINED,
        action_provider=provider,
    )

    context = result.initial_state["parameter_context"]
    assert result.parameters == parameters
    assert result.parameter_fingerprint == parameters.fingerprint
    assert len(context["records"]) == 12
    assert {record["branch"] for record in context["records"].values()} == {
        "branch-1",
        "branch-2",
        "branch-3",
        "branch-4",
    }
    assert (
        max(len(record["dependencies"]) for record in context["records"].values()) == 3
    )
    assert (
        sum(record["role"] == "distractor" for record in context["records"].values())
        == 3
    )
    assert (
        sum(record["note"] is not None for record in context["records"].values()) == 3
    )
    assert any(transition.rollback for transition in result.transitions)
    assert any(transition.tool_error_injected for transition in result.transitions)
    assert result.metrics.final_success is True


def test_drift_request_rejects_dimensions_that_do_not_fit_horizon() -> None:
    with pytest.raises(ValueError, match="snapshot, rollback span, and rollback"):
        CreateExperimentRequest(
            model_id=MODEL_ID,
            workload_id="state-drift-v1",
            horizons=[4],
            drift_parameters=DriftParameters(rollback_depth=3),
        )


@pytest.mark.asyncio
async def test_drift_aggregate_reports_budget_partial_without_parsing_it(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(Settings(mode="mock", data_dir=tmp_path / "data"))
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="state-drift-v1",
                scenario_ids=["ledger-reconciliation-v1"],
                conditions=[DriftCondition.CHAINED],
                horizons=[2],
                seeds=[0],
            )
        )
        unit = next(
            item
            for item in lab.experiments.units.values()
            if item.experiment_id == experiment.id
        )
        unit.result = {
            "kind": "state_drift",
            "termination_reason": "execution_policy_token_limit",
            "error": "token cap reached",
            "routing": [],
        }
        unit.evaluation = EvaluationResultRecord(
            id="partial-evaluation",
            workload_run_id=unit.workload_run_id,
            run_unit_id=unit.id,
            passed=None,
            score=None,
            metrics={"budget_exhausted": True},
        )

        aggregate = lab.experiments._aggregate_lane(
            lab.experiments.experiments[experiment.id], unit.lane_id
        )

        assert aggregate["completed_units"] == 1
        assert aggregate["scored_units"] == 0
        assert aggregate["incomplete_units"] == 1
        assert aggregate["mean_survival_curve"] == []
    finally:
        await lab.shutdown()


def test_routing_comparison_quantifies_overlap_divergence_and_expert_shifts() -> None:
    baseline = {
        "selection_counts": [[3, 1]],
        "routing_mass": [[0.75, 0.25]],
    }
    identical = {
        "selection_counts": [[3, 1]],
        "routing_mass": [[0.75, 0.25]],
    }
    shifted = {
        "selection_counts": [[0, 4]],
        "routing_mass": [[0.0, 1.0]],
    }

    assert _routing_distance(baseline, identical) == (1.0, 0.0)
    overlap, divergence = _routing_distance(baseline, shifted)
    assert overlap == pytest.approx(0.25)
    assert divergence is not None and 0 < divergence <= 1
    shifts = _largest_routing_shifts(baseline, shifted)
    assert shifts[0]["layer"] == 0
    assert shifts[0]["expert"] in {0, 1}
    assert abs(float(shifts[0]["delta"])) == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_drift_routing_telemetry_is_json_native_and_persistable(
    tmp_path: Path,
) -> None:
    topology = ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[0, 1],
    )
    runtime = MockModelRuntime(topology)
    completion = await runtime.complete_chat(
        [{"role": "user", "content": "Return 4."}],
        request_key="drift:test:baseline:unit:1",
        profile=None,
    )
    telemetry = _ProviderTelemetry()

    _record_completion(completion, 1, telemetry, topology)

    serialized = json.dumps(telemetry.routing)
    assert json.loads(serialized) == telemetry.routing

    lab = ResearchLab(Settings(mode="mock", data_dir=tmp_path / "data"))
    try:
        await lab.create_model_session(CreateModelSessionRequest())
        experiment = lab.submit_experiment(
            CreateExperimentRequest(
                model_id=MODEL_ID,
                workload_id="state-drift-v1",
                scenario_ids=["ledger-reconciliation-v1"],
                conditions=[DriftCondition.CHAINED],
                horizons=[2],
                seeds=[0],
            )
        )
        unit = next(
            item
            for item in lab.experiments.units.values()
            if item.experiment_id == experiment.id
        )
        unit.result = {"routing": telemetry.routing}

        lab.store.save_run_unit(unit)

        persisted = lab.store.load_run_units()[unit.id]
        assert persisted.result == unit.result
    finally:
        await runtime.aclose()
        await lab.shutdown()


@pytest.mark.asyncio
async def test_malformed_model_output_is_scored_as_an_invalid_transition() -> None:
    scenario = SCENARIOS["ledger-reconciliation-v1"]

    async def provider(_turn):
        return _parse_tool_call("prose instead of JSON")

    result = await run_drift_scenario(
        scenario,
        seed=0,
        horizon=2,
        condition=DriftCondition.CHAINED,
        action_provider=provider,
    )

    assert result.metrics.invalid_tool_calls == 2
    assert result.metrics.final_success is False
    assert result.transitions[0].actual_call.tool == "invalid_model_output"
    assert "prose instead" not in result.transitions[0].actual_call.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_fingerprint", [None, "f" * 64])
async def test_vllm_runtime_fails_closed_on_context_provenance_mismatch(
    response_fingerprint: str | None,
) -> None:
    topology = ModelTopology(
        num_layers=1,
        num_experts=2,
        top_k=1,
        routed_layer_ids=[0],
    )
    ids = np.array([[[0]]], dtype=np.uint8)
    weights = np.ones(ids.shape, dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "routed_experts": encode_npy(ids),
                    "routed_expert_weights": encode_npy(weights),
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        if response_fingerprint is not None:
            payload["expert_context_fingerprint"] = response_fingerprint
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(
        base_url="http://vllm", transport=httpx.MockTransport(handler)
    )
    runtime = VllmRuntime("http://vllm", "model", topology, client=client)
    context = InterventionContextRef(
        context_id="baseline",
        kind=InterventionKind.BASELINE,
        model_id="model",
        topology_fingerprint=canonical_fingerprint(topology),
        context_fingerprint="e" * 64,
        creation_source="test",
    )
    runtime.set_active_context(context)

    with pytest.raises(ValueError, match="expert-context fingerprint|omitted"):
        await runtime.complete("hello", request_key="test", profile=None)
    await client.aclose()


@pytest.mark.asyncio
async def test_mock_runtime_emits_exact_active_context_provenance() -> None:
    topology = ModelTopology(
        num_layers=1,
        num_experts=2,
        top_k=1,
        routed_layer_ids=[0],
    )
    runtime = MockModelRuntime(topology)
    context = InterventionContextRef(
        context_id="baseline",
        kind=InterventionKind.BASELINE,
        model_id="model",
        topology_fingerprint=canonical_fingerprint(topology),
        context_fingerprint="e" * 64,
        creation_source="test",
    )
    runtime.set_active_context(context)

    result = await runtime.complete("1 + 1", request_key="mock", profile=None)

    assert result.context_id == context.context_id
    assert result.context_fingerprint == context.context_fingerprint
    assert result.topology_fingerprint == context.topology_fingerprint
