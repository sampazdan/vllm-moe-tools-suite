from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.agentic.catalog import (
    AgentTaskCatalog,
    agent_run_contract_fingerprint,
)
from moe_tools_suite.agentic.controller import _clean_room_hardening_command
from moe_tools_suite.agentic.domain import (
    AgentBudgets,
    AgentRunStatus,
    CommandResult,
    CreateAgentRunRequest,
    NetworkPolicy,
    ReasoningMode,
    SandboxFile,
    SandboxOwnership,
    SandboxSession,
    SandboxSessionState,
    SandboxSpec,
    TerminationCause,
)
from moe_tools_suite.agentic.providers.base import (
    SandboxProviderError,
    ownership_labels,
)
from moe_tools_suite.agentic.providers.daytona import (
    DaytonaProviderConfig,
    DaytonaSandboxProvider,
)
from moe_tools_suite.agentic.providers.fake import FakeSandboxProvider
from moe_tools_suite.domain import (
    CreateModelSessionRequest,
    CriterionVisibility,
    DeterministicScorerKind,
    EvaluationContract,
    EvaluationCriterion,
    JudgeEvaluationRequest,
    JudgeProvider,
    LLMJudgeConfig,
)
from moe_tools_suite.judges import (
    JudgeProviderError,
    judge_request_cost_upper_bound,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.persistence import AgentRunRow, AgentTrialRow
from moe_tools_suite.settings import Settings

PACK_ID = "smoke-python-v1"
TASK_IDS = ["fix-subtract", "implement-slugify", "repair-json-cli"]
PUBLIC_TASK_FIELDS = {
    "id",
    "title",
    "instruction",
    "language",
    "tags",
    "success_criteria",
    "verifier_fingerprint",
    "hidden_verifier_file_count",
    "timeout_seconds",
}


def test_agent_contract_distinguishes_thinking_but_not_display_density() -> None:
    catalog = AgentTaskCatalog()
    compact = CreateAgentRunRequest(
        task_pack_id=PACK_ID,
        task_ids=[TASK_IDS[0]],
        model_session_id="session",
        reasoning_mode=ReasoningMode.COMPACT,
    )
    pack, tasks, agent = catalog.validate_run_request(compact)

    compact_fingerprint = agent_run_contract_fingerprint(
        request=compact,
        pack=pack,
        tasks=tasks,
        agent=agent,
    )
    full_fingerprint = agent_run_contract_fingerprint(
        request=compact.model_copy(update={"reasoning_mode": ReasoningMode.FULL}),
        pack=pack,
        tasks=tasks,
        agent=agent,
    )
    off_fingerprint = agent_run_contract_fingerprint(
        request=compact.model_copy(update={"reasoning_mode": ReasoningMode.OFF}),
        pack=pack,
        tasks=tasks,
        agent=agent,
    )

    assert compact_fingerprint == full_fingerprint
    assert compact_fingerprint != off_fingerprint


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
        **overrides,
    )


def _load_model(client: TestClient) -> dict[str, object]:
    response = client.post("/api/model-sessions", json={"model_id": MODEL_ID})
    assert response.status_code == 202
    job = response.json()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "completed"
    session = client.get("/api/model-sessions/current")
    assert session.status_code == 200
    assert session.json()["state"] == "ready"
    return session.json()


def _deny_local_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_process(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("the fake sandbox attempted a local subprocess")

    async def reject_async_process(*args: object, **kwargs: object) -> NoReturn:
        reject_process(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    monkeypatch.setattr(subprocess, "run", reject_process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_async_process)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", reject_async_process)


@pytest.mark.asyncio
async def test_idle_lab_retries_pending_sandbox_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = ResearchLab(_settings(tmp_path, agent_cleanup_retry_seconds=0.01))
    retried = asyncio.Event()
    calls = 0

    async def record_cleanup(provider_id: str | None = None) -> int:
        nonlocal calls
        del provider_id
        calls += 1
        if calls >= 2:
            retried.set()
        return 0

    monkeypatch.setattr(
        lab.agentic,
        "cleanup_interrupted_sandboxes",
        record_cleanup,
    )
    await lab.startup()
    await asyncio.wait_for(retried.wait(), timeout=0.5)
    await lab.shutdown()

    assert calls >= 3


def test_agent_max_cost_policy_counts_known_judge_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    maximum_cost = 0.000001
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": {
                    "name": "Priced fake judge",
                    "criteria": [
                        {
                            "id": "trusted_verifier",
                            "label": "Trusted task-pack verifier",
                        }
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Assess whether the requested change was made.",
                        "input_cost_per_million_usd": 1,
                        "output_cost_per_million_usd": 1,
                    },
                    "judge_weight": 0.5,
                },
                "execution_policy": {
                    "attempts": 2,
                    "max_cost_usd": maximum_cost,
                },
            },
        )

        assert submitted.status_code == 202
        run = client.get(f"/api/agent-runs/{submitted.json()['result_id']}").json()
        first, stopped = run["trials"]
        assert first["status"] == "error"
        assert first["termination_reason"] == "cost_limit"
        assert first["performance"]["inference_cost_usd"] is None
        assert first["performance"]["judge_cost_usd"] is None
        assert first["performance"]["estimated_cost_usd"] is None
        assert stopped["status"] == "cancelled"
        assert stopped["termination_reason"] == "cost_limit"


def test_judge_provider_failure_pessimistically_debits_cap_and_stops_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    maximum_cost = 0.02
    app = create_app(_settings(tmp_path))
    calls = 0

    async def fail_judge(*args: object, **kwargs: object):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise JudgeProviderError("sanitized provider failure")

    monkeypatch.setattr(app.state.lab.judges, "evaluate", fail_judge)
    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": {
                    "name": "Priced failing judge",
                    "criteria": [
                        {"id": "trusted_verifier", "label": "Trusted verifier"}
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Judge correctness.",
                        "input_cost_per_million_usd": 1,
                        "output_cost_per_million_usd": 1,
                    },
                    "judge_weight": 0.5,
                },
                "execution_policy": {
                    "attempts": 2,
                    "max_cost_usd": maximum_cost,
                },
            },
        )
        run = client.get(f"/api/agent-runs/{submitted.json()['result_id']}").json()

    first, stopped = run["trials"]
    assert calls == 1
    assert first["status"] == "error"
    assert first["termination_reason"] == "cost_limit"
    assert first["performance"]["judge_cost_usd"] is None
    assert first["performance"]["judge_cost_debit_usd"] == maximum_cost
    assert first["performance"]["judge_cost_uncertain"] is True
    assert stopped["status"] == "cancelled"
    assert stopped["termination_reason"] == "cost_limit"


def test_agent_judge_missing_usage_persists_conservative_debit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    settings = _settings(tmp_path)
    app = create_app(settings)
    original_evaluate = app.state.lab.judges.evaluate
    captured: list[JudgeEvaluationRequest] = []

    async def omit_provider_usage(
        request: JudgeEvaluationRequest,
        **kwargs: object,
    ):
        captured.append(request)
        result = await original_evaluate(request, **kwargs)
        return result.model_copy(
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

    monkeypatch.setattr(app.state.lab.judges, "evaluate", omit_provider_usage)
    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": {
                    "name": "Unknown-usage judge",
                    "criteria": [
                        {"id": "trusted_verifier", "label": "Trusted verifier"}
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Judge correctness.",
                        "repetitions": 3,
                        "input_cost_per_million_usd": 1,
                        "output_cost_per_million_usd": 2,
                    },
                    "judge_weight": 0.5,
                },
            },
        )
        run_id = submitted.json()["result_id"]
        trial = client.get(f"/api/agent-runs/{run_id}").json()["trials"][0]

    assert len(captured) == 1
    expected_debit = judge_request_cost_upper_bound(captured[0])
    assert expected_debit is not None
    assert trial["status"] == "passed"
    assert trial["performance"]["judge_cost_usd"] is None
    assert trial["performance"]["judge_cost_debit_usd"] == pytest.approx(expected_debit)
    assert trial["performance"]["judge_cost_uncertain"] is True
    assert trial["performance"]["estimated_cost_usd"] is None

    with TestClient(create_app(settings)) as restarted:
        restored = restarted.get(f"/api/agent-runs/{run_id}").json()["trials"][0]
    assert restored["performance"]["judge_cost_usd"] is None
    assert restored["performance"]["judge_cost_debit_usd"] == pytest.approx(
        expected_debit
    )
    assert restored["performance"]["judge_cost_uncertain"] is True


def test_uncapped_later_judge_repetition_failure_persists_request_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    settings = _settings(tmp_path)
    app = create_app(settings)
    original_call = app.state.lab.judges._call_fake
    calls = 0
    captured_request: JudgeEvaluationRequest | None = None

    async def fail_second_repetition(
        request: JudgeEvaluationRequest,
        candidate_label: str | None,
    ):
        nonlocal calls, captured_request
        calls += 1
        captured_request = request
        if calls == 2:
            raise JudgeProviderError("sanitized second-repetition failure")
        return await original_call(request, candidate_label)

    monkeypatch.setattr(
        app.state.lab.judges,
        "_call_fake",
        fail_second_repetition,
    )
    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": {
                    "name": "Partially billed judge",
                    "criteria": [
                        {"id": "trusted_verifier", "label": "Trusted verifier"}
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Judge correctness.",
                        "repetitions": 2,
                        "input_cost_per_million_usd": 1,
                        "output_cost_per_million_usd": 2,
                    },
                    "judge_weight": 0.5,
                },
            },
        )
        run_id = submitted.json()["result_id"]
        trial = client.get(f"/api/agent-runs/{run_id}").json()["trials"][0]

    assert calls == 2
    assert captured_request is not None
    expected_debit = judge_request_cost_upper_bound(captured_request)
    assert expected_debit is not None
    assert trial["status"] == "error"
    assert trial["termination_reason"] == "verifier_error"
    assert trial["performance"]["judge_cost_usd"] is None
    assert trial["performance"]["judge_cost_debit_usd"] == pytest.approx(expected_debit)
    assert trial["performance"]["judge_cost_uncertain"] is True
    assert trial["performance"]["estimated_cost_usd"] is None

    with TestClient(create_app(settings)) as restarted:
        restored = restarted.get(f"/api/agent-runs/{run_id}").json()["trials"][0]
    assert restored["performance"]["judge_cost_usd"] is None
    assert restored["performance"]["judge_cost_debit_usd"] == pytest.approx(
        expected_debit
    )
    assert restored["performance"]["judge_cost_uncertain"] is True


def test_public_task_catalog_never_exposes_execution_material(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        packs_response = client.get("/api/agent-task-packs")
        assert packs_response.status_code == 200
        pack = next(item for item in packs_response.json() if item["id"] == PACK_ID)
        assert pack["task_count"] == 3
        assert pack["ready"] is True
        assert pack["oracle_passed"] is True
        assert pack["noop_failed"] is True
        assert len(pack["fingerprint"]) == 64
        assert "tasks" not in pack

        tasks_response = client.get(f"/api/agent-task-packs/{PACK_ID}/tasks")
        assert tasks_response.status_code == 200
        tasks = tasks_response.json()
        assert [task["id"] for task in tasks] == TASK_IDS
        assert all(set(task) == PUBLIC_TASK_FIELDS for task in tasks)
        assert all(task["success_criteria"] for task in tasks)
        assert all(len(task["verifier_fingerprint"]) == 64 for task in tasks)
        assert all(task["hidden_verifier_file_count"] == 2 for task in tasks)

        private_fields = {
            "files",
            "verifier_file_paths",
            "oracle_commands",
            "verifier_command",
            "image_ref",
            "image_digest",
            "working_directory",
            "network_policy",
            "allowed_hosts",
        }
        assert all(set(task).isdisjoint(private_fields) for task in tasks)
        assert "return sum((left, right))" not in tasks_response.text
        assert "python -m unittest" not in tasks_response.text


def test_custom_agent_success_verifier_executes_only_in_the_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    command = (
        "python -I -c 'import calculator; "
        "raise SystemExit(calculator.subtract(5, 3) != 2)'"
    )
    contract = EvaluationContract(
        name="Hidden user-authored verifier",
        criteria=[
            EvaluationCriterion(
                id="custom_sandbox_check",
                label="Custom sandbox check",
                kind=DeterministicScorerKind.VERIFIER,
                visibility=CriterionVisibility.HIDDEN,
                verifier_command=command,
                verifier_timeout_seconds=30,
            )
        ],
    )
    app = create_app(_settings(tmp_path))
    provider = FakeSandboxProvider(
        scripted_results={
            command: CommandResult(
                command=command,
                exit_code=0,
                stdout="User-authored check passed.\n",
            )
        }
    )
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [task.id],
                "model_session_id": session["id"],
                "evaluation_contract": contract.model_dump(mode="json"),
            },
        )

        assert response.status_code == 202
        run = client.get(f"/api/agent-runs/{response.json()['result_id']}").json()
        assert run["status"] == "completed"
        assert run["passed_trials"] == 1
        assert run["evaluation_contract"]["criteria"] == []
        assert run["evaluation_contract_fingerprint"] == contract.fingerprint
        assert run["hidden_evaluation_criteria_count"] == 1
        trial = run["trials"][0]
        criterion = trial["evaluation"]["criteria"][0]
        assert criterion["criterion_id"] == "custom_sandbox_check"
        assert criterion["passed"] is True
        assert criterion["execution_provenance"] == "user_authored_sandbox"
        assert criterion["exit_code"] == 0
        assert criterion["duration_ms"] >= 0
        assert len(criterion["output_sha256"]) == 64
        assert "verifier_command" not in json.dumps(trial["evaluation"])
        assert trial["performance"]["verifier_time_ms"] >= criterion["duration_ms"]

        exported = client.get(f"/api/agent-runs/{run['id']}/export")
        assert exported.status_code == 200
        public_detail = exported.json()["detail"]
        assert task.verifier_command not in exported.text
        assert task.verifier_command not in json.dumps(public_detail)
        assert "verifier_command" not in json.dumps(public_detail)
        assert public_detail["run"]["evaluation_contract"]["criteria"] == []


class _CustomVerifierBoundaryProvider(FakeSandboxProvider):
    def __init__(self, task, command: str) -> None:
        super().__init__(
            scripted_results={
                command: CommandResult(
                    command=command,
                    exit_code=0,
                    stdout="x" * 1_000,
                )
            },
            max_output_bytes=64,
        )
        self.task = task
        self.custom_command = command
        self.primary_handle_id: str | None = None
        self.custom_handle_id: str | None = None
        self.primary_source: bytes | None = None
        self.custom_source_after: bytes | None = None
        self.custom_output_hash: str | None = None
        self.boundaries: list[dict[str, bool]] = []
        self.specs: dict[str, SandboxSpec] = {}

    async def create(self, spec: SandboxSpec, ownership: SandboxOwnership):
        self.specs[ownership.trial_id] = spec
        return await super().create(spec, ownership)

    async def exec(
        self,
        handle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        result = await super().exec(
            handle,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        sandbox = self._owned_sandbox(handle)
        source = f"/workspace/task/{self.task.submission_file_paths[0]}"
        if command == self.task.verifier_command:
            self.primary_handle_id = handle.id
            self.primary_source = bytes(sandbox.files[source])
        if command == self.custom_command:
            self.custom_handle_id = handle.id
            paths = set(sandbox.files)
            public_paths = {
                f"/workspace/task/{file.path}"
                for file in self.task.files
                if file.path
                not in set(self.task.submission_file_paths)
                | set(self.task.verifier_file_paths)
                | set(self.task.oracle_file_paths)
            }
            protected_paths = {
                f"/workspace/task/{path}"
                for path in (
                    *self.task.verifier_file_paths,
                    *self.task.oracle_file_paths,
                )
            }
            command_fingerprint = (
                hashlib.sha256(self.task.verifier_command.encode()).hexdigest().encode()
            )
            boundary = {
                "submission_present": source in paths,
                "public_files_present": public_paths <= paths,
                "protected_files_absent": protected_paths.isdisjoint(paths),
                "canonical_fingerprint_absent": all(
                    command_fingerprint not in content
                    for content in sandbox.files.values()
                ),
            }
            self.boundaries.append(boundary)
            sandbox.files[source] = b"raise RuntimeError('custom verifier mutation')\n"
            self.custom_source_after = bytes(sandbox.files[source])
            assert len(result.stdout) <= 64
            self.custom_output_hash = hashlib.sha256(
                f"{result.stdout}\n{result.stderr}".encode()
            ).hexdigest()
            return result.model_copy(
                update={"exit_code": 0 if all(boundary.values()) else 1}
            )
        return result


def test_user_authored_verifier_has_public_only_isolated_clean_room(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task("repo-engineering-v1", "repair-event-ledger")
    command = "python -I -c 'import ledger; raise SystemExit(0)'"
    contract = EvaluationContract(
        name="User-authored isolated verifier",
        criteria=[
            EvaluationCriterion(
                id="custom_sandbox_check",
                label="Custom sandbox check",
                kind=DeterministicScorerKind.VERIFIER,
                visibility=CriterionVisibility.HIDDEN,
                verifier_command=command,
                verifier_timeout_seconds=30,
            )
        ],
    )
    app = create_app(_settings(tmp_path))
    provider = _CustomVerifierBoundaryProvider(task, command)
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": task.pack_id,
                "task_ids": [task.id],
                "model_session_id": session["id"],
                "evaluation_contract": contract.model_dump(mode="json"),
            },
        )
        run = client.get(f"/api/agent-runs/{submitted.json()['result_id']}").json()

    criterion = run["trials"][0]["evaluation"]["criteria"][0]
    assert run["trials"][0]["status"] == "passed"
    assert run["trials"][0]["evaluation"]["passed"] is True
    assert criterion["passed"] is True
    assert criterion["execution_provenance"] == "user_authored_sandbox"
    assert criterion["output_sha256"] == provider.custom_output_hash
    assert provider.boundaries == [
        {
            "submission_present": True,
            "public_files_present": True,
            "protected_files_absent": True,
            "canonical_fingerprint_absent": True,
        }
    ]
    assert provider.primary_handle_id is not None
    assert provider.custom_handle_id is not None
    assert provider.primary_handle_id != provider.custom_handle_id
    assert provider.primary_source is not None
    assert provider.custom_source_after != provider.primary_source
    assert all(
        task.verifier_command != recorded
        for handle_id, recorded in provider.commands
        if handle_id == provider.custom_handle_id
    )
    assert all(
        spec.network_policy is task.network_policy for spec in provider.specs.values()
    )
    assert provider._sandboxes == {}


def test_agent_llm_judge_receives_no_protected_verifier_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    contract = EvaluationContract(
        name="Verifier plus judge",
        criteria=[
            EvaluationCriterion(
                id="trusted_verifier",
                label="Trusted task-pack verifier",
            )
        ],
        judge=LLMJudgeConfig(
            provider=JudgeProvider.FAKE,
            model="deterministic-fake-judge-v1",
            rubric="Assess whether the candidate completed the requested change.",
            input_cost_per_million_usd=1,
            output_cost_per_million_usd=1,
        ),
        judge_weight=0.5,
        pass_threshold=0.7,
    )
    app = create_app(_settings(tmp_path))
    captured: list[JudgeEvaluationRequest] = []
    original_evaluate = app.state.lab.judges.evaluate

    async def capture_judge(
        request: JudgeEvaluationRequest,
        **kwargs: object,
    ):
        captured.append(request)
        return await original_evaluate(request, **kwargs)

    monkeypatch.setattr(app.state.lab.judges, "evaluate", capture_judge)

    with TestClient(app) as client:
        session = _load_model(client)
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": contract.model_dump(mode="json"),
            },
        )
        run = client.get(f"/api/agent-runs/{response.json()['result_id']}").json()

    assert run["status"] == "completed"
    assert run["trials"][0]["evaluation"]["judge"]["provider"] == "fake"
    assert len(captured) == 1
    assert "All tests passed" not in captured[0].candidate
    assert "Ran bundled unittest" not in captured[0].candidate
    assert "passed=True; exit_code=0" in captured[0].candidate
    performance = run["trials"][0]["performance"]
    assert performance["judge_cost_usd"] > 0
    assert performance["estimated_cost_usd"] == performance["judge_cost_usd"]


def test_agent_execution_policy_controls_attempts_and_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    app = create_app(_settings(tmp_path))

    with TestClient(app) as client:
        session = _load_model(client)
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "attempts": 1,
                "execution_policy": {
                    "attempts": 2,
                    "concurrency": 1,
                    "timeout_seconds": 120,
                    "per_item_timeout_seconds": 45,
                    "max_turns": 5,
                    "max_commands": 4,
                    "max_tokens": 4096,
                    "fail_fast": True,
                },
            },
        )

        assert response.status_code == 202
        assert response.json()["progress_total"] == 2
        run = client.get(f"/api/agent-runs/{response.json()['result_id']}").json()
        assert run["total_trials"] == 2
        assert [trial["attempt"] for trial in run["trials"]] == [1, 2]
        assert run["execution_policy"]["attempts"] == 2
        assert run["execution_policy"]["per_item_timeout_seconds"] == 45
        assert run["execution_policy"]["max_turns"] == 5
        assert run["execution_policy"]["max_commands"] == 4
        assert run["execution_policy"]["max_tokens"] == 4096
        assert run["execution_policy"]["fail_fast"] is True

        unsupported = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "execution_policy": {"concurrency": 2},
            },
        )
        assert unsupported.status_code == 422
        assert "requires concurrency=1" in unsupported.json()["detail"]

        unpriced = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "evaluation_contract": {
                    "name": "Unpriced judge",
                    "criteria": [
                        {"id": "trusted_verifier", "label": "Trusted verifier"}
                    ],
                    "judge": {
                        "provider": "fake",
                        "model": "deterministic-fake-judge-v1",
                        "rubric": "Judge correctness.",
                    },
                    "judge_weight": 0.5,
                },
                "execution_policy": {"max_cost_usd": 0.01},
            },
        )
        assert unpriced.status_code == 422
        assert "requires explicit judge" in unpriced.json()["detail"]

        token_limited = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
                "execution_policy": {"attempts": 2, "max_tokens": 1},
            },
        )
        assert token_limited.status_code == 202
        limited_run = client.get(
            f"/api/agent-runs/{token_limited.json()['result_id']}"
        ).json()
        first, stopped = limited_run["trials"]
        assert first["completion_tokens"] == 0
        assert first["termination_reason"] == "token_limit"
        assert stopped["status"] == "cancelled"
        assert stopped["termination_reason"] == "token_limit"


def test_bundled_task_pack_oracles_pass_and_noop_repositories_fail(
    tmp_path: Path,
) -> None:
    pack = AgentTaskCatalog().get_pack(PACK_ID)
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MOE_TOOLS_VERIFIER_DEV_MODE": "1",
    }

    for task in pack.tasks:
        workspace = tmp_path / task.id
        workspace.mkdir()
        for file in task.files:
            destination = workspace / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content)

        noop = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert noop.returncode != 0, task.id

        for command in task.oracle_commands:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                check=True,
            )
        oracle = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert oracle.returncode == 0, oracle.stdout + oracle.stderr


def test_unittest_monkeypatch_cannot_forge_trusted_verifier_success(
    tmp_path: Path,
) -> None:
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    workspace = tmp_path / "unittest-monkeypatch"
    workspace.mkdir()
    malicious = (
        "import unittest\n"
        "unittest.TestCase.assertEqual = lambda *args, **kwargs: None\n"
        "def subtract(left, right):\n"
        "    return 0\n"
    )
    for file in task.files:
        if file.path in task.oracle_file_paths:
            continue
        destination = workspace / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            malicious if file.path == task.submission_file_paths[0] else file.content
        )
    completed = subprocess.run(
        ["/bin/sh", "-c", task.verifier_command],
        cwd=workspace,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "MOE_TOOLS_VERIFIER_DEV_MODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=task.timeout_seconds,
        check=False,
    )

    assert completed.returncode != 0
    assert "returned 0" in completed.stderr


@pytest.mark.skipif(os.geteuid() != 0, reason="requires a real root-to-nobody drop")
def test_hidden_verifier_is_unreadable_and_workspace_is_immutable_to_submission(
    tmp_path: Path,
) -> None:
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    workspace = tmp_path / "root-boundary"
    workspace.mkdir()
    bounded_task = task.model_copy(update={"working_directory": str(workspace)})
    malicious = (
        "from pathlib import Path\n"
        "try:\n"
        "    Path('tests/test_calculator.py').read_text()\n"
        "    Path('calculator.py').write_text('forged')\n"
        "    escaped = True\n"
        "except OSError:\n"
        "    escaped = False\n"
        "def subtract(left, right):\n"
        "    return left - right if escaped else 0\n"
    )
    materialized: list[SandboxFile] = []
    for file in bounded_task.files:
        if file.path in bounded_task.oracle_file_paths:
            continue
        content = (
            malicious
            if file.path == bounded_task.submission_file_paths[0]
            else file.content
        )
        materialized_file = file.model_copy(update={"content": content})
        materialized.append(materialized_file)
        destination = workspace / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    hardening = _clean_room_hardening_command(
        bounded_task,
        tuple(materialized),
    )
    subprocess.run(
        ["/bin/sh", "-c", hardening],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    completed = subprocess.run(
        ["/bin/sh", "-c", bounded_task.verifier_command],
        cwd=workspace,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        timeout=bounded_task.timeout_seconds,
        check=False,
    )

    assert completed.returncode != 0
    assert (workspace / bounded_task.submission_file_paths[0]).read_text() == malicious


class _TamperingFakeProvider(FakeSandboxProvider):
    def __init__(self, verifier_command: str, verifier_path: str) -> None:
        super().__init__()
        self.verifier_command = verifier_command
        self.verifier_path = f"/workspace/task/{verifier_path}"
        self.verifier_snapshots: list[bytes] = []
        self.verifier_handle_ids: list[str] = []
        self.tampered_handle_ids: list[str] = []
        self.agent_handle_id: str | None = None
        self.events: list[tuple[str, str]] = []

    async def create(self, spec: SandboxSpec, ownership: SandboxOwnership):
        handle = await super().create(spec, ownership)
        if self.agent_handle_id is None:
            self.agent_handle_id = handle.id
        return handle

    async def upload(self, handle, file: SandboxFile) -> None:
        self.events.append(("upload", file.path))
        await super().upload(handle, file)

    async def exec(
        self,
        handle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        self.events.append(("exec", command))
        result = await super().exec(
            handle,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        sandbox = self._owned_sandbox(handle)
        if command == self.verifier_command:
            snapshot = sandbox.files[self.verifier_path]
            self.verifier_snapshots.append(snapshot)
            self.verifier_handle_ids.append(handle.id)
            canonical = b"isolated subtraction probes" in snapshot
            return result.model_copy(update={"exit_code": 0 if canonical else 1})
        if handle.id == self.agent_handle_id:
            self.tampered_handle_ids.append(handle.id)
            sandbox.files[self.verifier_path] = b"# agent replaced trusted tests\n"
        return result


def test_clean_room_verifier_ignores_agent_workspace_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    provider = _TamperingFakeProvider(
        task.verifier_command,
        task.verifier_file_paths[0],
    )
    app = create_app(_settings(tmp_path))
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [task.id],
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "fake",
                "model_session_id": session["id"],
            },
        )
        assert response.status_code == 202
        run = client.get(f"/api/agent-runs/{response.json()['result_id']}").json()
        assert run["status"] == "completed"
        assert run["passed_trials"] == 1
        assert run["trials"][0]["status"] == "passed"
        assert (
            run["trials"][0]["evaluation"]["criteria"][0]["execution_provenance"]
            == "trusted_task_verifier"
        )
        artifacts = client.get(f"/api/trials/{run['trials'][0]['id']}/artifacts").json()
        assert artifacts["verifier"]["status"] == "passed"

    canonical = next(
        file.content.encode()
        for file in task.files
        if file.path == task.verifier_file_paths[0]
    )
    assert provider.verifier_snapshots == [canonical]
    assert set(provider.verifier_handle_ids).isdisjoint(provider.tampered_handle_ids)
    protected_uploads = [
        index
        for index, event in enumerate(provider.events)
        if event == ("upload", task.verifier_file_paths[0])
    ]
    assert len(protected_uploads) == 1
    assert protected_uploads[0] > provider.events.index(
        ("exec", task.oracle_commands[0])
    )


class _SymlinkOracleAttackProvider(FakeSandboxProvider):
    def __init__(self, task) -> None:
        super().__init__()
        self.task = task
        self.agent_handle_id: str | None = None
        self.symlinked = False
        self.uploads: list[tuple[str, str]] = []
        self.ownerships: list[SandboxOwnership] = []
        self.deleted: list[str] = []
        self.verifier_boundaries: list[dict[str, bool]] = []

    async def create(self, spec: SandboxSpec, ownership: SandboxOwnership):
        handle = await super().create(spec, ownership)
        self.ownerships.append(ownership)
        if self.agent_handle_id is None:
            self.agent_handle_id = handle.id
        return handle

    async def upload(self, handle, file: SandboxFile) -> None:
        self.uploads.append((handle.id, file.path))
        await super().upload(handle, file)

    async def read(
        self,
        handle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        if (
            handle.id == self.agent_handle_id
            and self.symlinked
            and path.endswith(self.task.submission_file_paths[0])
        ):
            sandbox = self._owned_sandbox(handle)
            oracle = f"/workspace/task/{self.task.oracle_file_paths[0]}"
            if oracle not in sandbox.files:
                raise SandboxProviderError(
                    "file_not_found",
                    "The simulated dangling submission symlink has no target.",
                )
            return sandbox.files[oracle]
        return await super().read(handle, path, max_bytes=max_bytes)

    async def exec(
        self,
        handle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        result = await super().exec(
            handle,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        sandbox = self._owned_sandbox(handle)
        if handle.id == self.agent_handle_id and command in self.task.oracle_commands:
            source = f"/workspace/task/{self.task.submission_file_paths[0]}"
            sandbox.files.pop(source, None)
            sandbox.files["/workspace/task/tests/__init__.py"] = (
                b"def load_tests(*args):\n    raise SystemExit(0)\n"
            )
            self.symlinked = True
        if command == self.task.verifier_command:
            source = f"/workspace/task/{self.task.submission_file_paths[0]}"
            hidden = f"/workspace/task/{self.task.verifier_file_paths[0]}"
            oracle = f"/workspace/task/{self.task.oracle_file_paths[0]}"
            boundary = {
                "source_present": source in sandbox.files,
                "hidden_canonical": b"isolated ledger" in sandbox.files[hidden],
                "oracle_absent": oracle not in sandbox.files,
                "load_tests_absent": b"load_tests"
                not in sandbox.files["/workspace/task/tests/__init__.py"],
            }
            self.verifier_boundaries.append(boundary)
            return result.model_copy(update={"exit_code": 1})
        return result

    async def delete(self, handle) -> None:
        self.deleted.append(handle.id)
        await super().delete(handle)


def test_dangling_oracle_symlink_and_load_tests_attack_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task("repo-engineering-v1", "repair-event-ledger")
    provider = _SymlinkOracleAttackProvider(task)
    app = create_app(_settings(tmp_path))
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": task.pack_id,
                "task_ids": [task.id],
                "model_session_id": session["id"],
            },
        )
        run = client.get(f"/api/agent-runs/{submitted.json()['result_id']}").json()

    assert run["trials"][0]["status"] == "failed"
    assert provider.verifier_boundaries == [
        {
            "source_present": False,
            "hidden_canonical": True,
            "oracle_absent": True,
            "load_tests_absent": True,
        }
    ]
    assert all(path not in task.oracle_file_paths for _, path in provider.uploads)
    assert len(provider.ownerships) == len(provider.deleted) == 2
    assert len({ownership.trial_id for ownership in provider.ownerships}) == 2
    assert provider._sandboxes == {}
    assert all(
        session.state is SandboxSessionState.DELETED
        for session in app.state.lab.agentic.sandboxes.values()
    )


class _SnapshotRaceProvider(FakeSandboxProvider):
    def __init__(self, task, good_source: bytes) -> None:
        super().__init__()
        self.task = task
        self.good_source = good_source
        self.agent_handle_id: str | None = None
        self.agent_after_snapshot: bytes | None = None
        self.verifier_source: bytes | None = None

    async def create(self, spec: SandboxSpec, ownership: SandboxOwnership):
        handle = await super().create(spec, ownership)
        if self.agent_handle_id is None:
            self.agent_handle_id = handle.id
        return handle

    async def exec(
        self,
        handle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        result = await super().exec(
            handle,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        sandbox = self._owned_sandbox(handle)
        source = f"/workspace/task/{self.task.submission_file_paths[0]}"
        if handle.id == self.agent_handle_id and command in self.task.oracle_commands:
            sandbox.files[source] = self.good_source
        if command == self.task.verifier_command:
            self.verifier_source = sandbox.files[source]
            return result.model_copy(
                update={
                    "exit_code": (0 if self.verifier_source == self.good_source else 1)
                }
            )
        return result

    async def read(
        self,
        handle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        content = await super().read(handle, path, max_bytes=max_bytes)
        if handle.id == self.agent_handle_id and path.endswith(
            self.task.submission_file_paths[0]
        ):
            sandbox = self._owned_sandbox(handle)
            source = f"/workspace/task/{self.task.submission_file_paths[0]}"
            sandbox.files[source] = b"raise RuntimeError('background tamper')\n"
            self.agent_after_snapshot = sandbox.files[source]
        return content


def test_submission_snapshot_is_immutable_after_background_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    original = next(file.content for file in task.files if file.path == "calculator.py")
    good = original.replace("return sum((left, right))", "return left - right").encode()
    provider = _SnapshotRaceProvider(task, good)
    app = create_app(_settings(tmp_path))
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": task.pack_id,
                "task_ids": [task.id],
                "model_session_id": session["id"],
            },
        )
        run = client.get(f"/api/agent-runs/{submitted.json()['result_id']}").json()

    assert run["trials"][0]["status"] == "passed"
    assert provider.verifier_source == good
    assert provider.agent_after_snapshot == b"raise RuntimeError('background tamper')\n"


def test_fake_three_task_vertical_slice_persists_all_research_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    settings = _settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        session = _load_model(client)
        session_id = str(session["id"])
        create_response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": TASK_IDS,
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "fake",
                "model_session_id": session_id,
                "budgets": {
                    "max_turns": 8,
                    "max_commands": 8,
                    "max_tokens": 32_768,
                    "timeout_seconds": 60,
                },
            },
        )
        assert create_response.status_code == 202
        queued_job = create_response.json()
        assert queued_job["result_id"]
        assert queued_job["progress_total"] == 3

        job = client.get(f"/api/jobs/{queued_job['id']}").json()
        assert job["status"] == "completed"
        assert job["progress_current"] == job["progress_total"] == 3
        run_id = job["result_id"]

        run_response = client.get(f"/api/agent-runs/{run_id}")
        assert run_response.status_code == 200
        run = run_response.json()
        assert run["status"] == "completed"
        assert run["task_pack_id"] == PACK_ID
        assert run["model_session_id"] == session_id
        assert run["sandbox_provider_id"] == "fake"
        assert run["reasoning_mode"] == "compact"
        assert run["task_ids"] == TASK_IDS
        assert run["total_trials"] == run["completed_trials"] == 3
        assert run["passed_trials"] == 3
        assert run["mean_reward"] == 1.0
        assert run["active_trial_id"] is None
        assert len(run["compatibility_fingerprint"]) == 64

        summaries = run["trials"]
        assert [trial["task_id"] for trial in summaries] == TASK_IDS
        assert all(trial["status"] == "passed" for trial in summaries)
        assert all(trial["reward"] == 1.0 for trial in summaries)
        assert all(trial["turns"] == 2 for trial in summaries)
        assert all(trial["commands"] == 1 for trial in summaries)
        assert all(trial["inference_calls"] == 2 for trial in summaries)
        assert all(trial["routed_inference_calls"] == 2 for trial in summaries)
        assert all(
            trial["termination_reason"] == "agent_finished" for trial in summaries
        )
        assert all(trial["sandbox_status"] == "deleted" for trial in summaries)
        assert all(trial["performance"] is not None for trial in summaries)
        assert all(trial["performance"]["wall_time_ms"] > 0 for trial in summaries)
        assert all(
            trial["performance"]["total_tokens"]
            == trial["performance"]["prompt_tokens"]
            + trial["performance"]["completion_tokens"]
            for trial in summaries
        )
        assert all(trial["performance"]["mean_tps"] is None for trial in summaries)

        fake_provider = app.state.lab.agentic.providers["fake"]
        assert fake_provider._sandboxes == {}
        assert len(fake_provider.commands) == 9

        artifact_root = settings.data_dir / "artifacts" / "agent-trials"
        trial_ids = {trial["id"] for trial in summaries}
        for trial in summaries:
            trial_id = trial["id"]
            detail = client.get(f"/api/trials/{trial_id}")
            assert detail.status_code == 200
            assert detail.json() == trial

            trajectory_response = client.get(f"/api/trials/{trial_id}/trajectory")
            assert trajectory_response.status_code == 200
            trajectory = trajectory_response.json()
            assert trajectory["format"] == "ATIF"
            assert trajectory["schema_version"] == "ATIF-v1.7"
            assert trajectory["reasoning_mode"] == "compact"
            assert [step["sequence"] for step in trajectory["steps"]] == list(
                range(1, len(trajectory["steps"]) + 1)
            )
            step_types = [step["type"] for step in trajectory["steps"]]
            assert step_types.count("system") == 1
            assert step_types.count("user") == 1
            assert step_types.count("assistant") == 2
            assert step_types.count("tool") == 1
            assert step_types.count("observation") == 1
            assert step_types[-1] == "verifier"
            assert [
                step["turn"]
                for step in trajectory["steps"]
                if step["phase"] == "response"
            ] == [1, 2]
            assert all(
                step["stream"] == "stdout"
                for step in trajectory["steps"]
                if step["phase"] == "observation"
            )
            inference_views = [
                step["inference"]
                for step in trajectory["steps"]
                if step["type"] == "assistant"
            ]
            assert all(inference is not None for inference in inference_views)
            assert all(
                inference["routed_layers"] == 40
                for inference in inference_views
                if inference is not None
            )
            assert all(
                inference["total_routed_slots"] > 0
                for inference in inference_views
                if inference is not None
            )

            atif_response = client.get(f"/api/trials/{trial_id}/export/atif")
            assert atif_response.status_code == 200
            assert atif_response.headers["content-type"].startswith("application/json")
            assert "attachment" in atif_response.headers["content-disposition"]
            atif = atif_response.json()
            assert atif["schema_version"] == "ATIF-v1.7"
            assert atif["trajectory_id"] == trial_id
            agent_steps = [step for step in atif["steps"] if step["source"] == "agent"]
            assert len(agent_steps) == trial["inference_calls"]
            assert all(step["llm_call_count"] == 1 for step in agent_steps)
            assert all(step["metrics"]["extra"]["inference_id"] for step in agent_steps)
            assert all(
                step["metrics"]["extra"]["routing_artifact"] for step in agent_steps
            )
            assert atif["final_metrics"]["extra"] == {
                "reward": 1.0,
                "termination_cause": "agent_finished",
                "evaluation_contract_fingerprint": run[
                    "evaluation_contract_fingerprint"
                ],
            }

            routing_response = client.get(f"/api/trials/{trial_id}/routing")
            assert routing_response.status_code == 200
            routing = routing_response.json()
            assert routing["trial_id"] == trial_id
            assert routing["run_id"] == run_id
            assert routing["model_id"] == MODEL_ID
            assert routing["model_session_id"] == session_id
            assert routing["profile_id"] is None
            assert routing["profile_fingerprint"] is None
            assert routing["layer_ids"] == list(range(40))
            assert len(routing["selection_counts"]) == 40
            assert all(len(row) == 256 for row in routing["selection_counts"])
            assert len(routing["routing_mass"]) == 40
            assert all(len(row) == 256 for row in routing["routing_mass"])
            assert routing["inference_count"] == 2
            assert routing["inference_calls"] == 2
            assert routing["captured_inference_calls"] == 2
            assert routing["served_tokens"] == (
                trial["prompt_tokens"] + trial["completion_tokens"]
            )
            assert routing["total_routed_slots"] == sum(
                artifact["total_routed_slots"] for artifact in routing["artifacts"]
            )
            assert len(routing["artifacts"]) == 2
            for artifact in routing["artifacts"]:
                artifact_path = artifact_root / artifact["relative_path"]
                assert artifact_path.is_file()
                assert (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == (artifact["sha256"])
                )

            artifacts_response = client.get(f"/api/trials/{trial_id}/artifacts")
            assert artifacts_response.status_code == 200
            artifacts = artifacts_response.json()
            assert artifacts["trial_id"] == trial_id
            assert artifacts["patch"] is None
            assert artifacts["patch_sha256"] is None
            assert artifacts["files_changed"] == 0
            assert artifacts["additions"] == 0
            assert artifacts["deletions"] == 0
            assert artifacts["verifier"]["status"] == "passed"
            assert artifacts["verifier"]["reward"] == 1.0
            assert artifacts["verifier"]["exit_code"] == 0
            assert artifacts["exports"] == [
                {
                    "name": "ATIF trajectory",
                    "media_type": "application/json",
                    "download_url": f"/api/trials/{trial_id}/export/atif",
                }
            ]

        export_response = client.get(f"/api/agent-runs/{run_id}/export")
        assert export_response.status_code == 200
        assert "attachment" in export_response.headers["content-disposition"]
        exported = export_response.json()
        assert exported["schema_version"] == 1
        assert exported["detail"]["run"]["id"] == run_id
        assert set(exported["trajectories"]) == trial_ids
        assert set(exported["routing"]) == trial_ids
        assert set(exported["manifests"]) == trial_ids
        for manifest in exported["manifests"].values():
            assert {item["name"] for item in manifest["artifacts"]} == {
                "trajectory.json",
                "patch.diff",
                "verifier.json",
                "evaluation.json",
            }
            for artifact in manifest["artifacts"]:
                artifact_path = artifact_root / artifact["relative_path"]
                assert artifact_path.is_file()
                assert artifact_path.stat().st_size == artifact["size_bytes"]
                assert (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == (artifact["sha256"])
                )

        first_trial_id = summaries[0]["id"]
        source_refs = [
            {"kind": "agent_trial", "id": trial["id"], "weight": 1}
            for trial in summaries
        ]
        explorer_response = client.post(
            "/api/routing/explore",
            json={
                "sources": source_refs,
                "metric": "routing_mass",
                "filters": {"passed": True},
            },
        )
        assert explorer_response.status_code == 200
        explorer = explorer_response.json()
        assert explorer["layer_ids"] == list(range(40))
        assert len(explorer["selection_counts"]) == 40
        assert all(len(row) == 256 for row in explorer["selection_counts"])
        assert explorer["captured_inference_calls"] == 6
        assert explorer["total_inference_calls"] == 6
        assert explorer["filter_capabilities"] == {
            "item": False,
            "trial": True,
            "step_type": False,
            "outcome": True,
        }
        assert {source["id"] for source in explorer["sources"]} == trial_ids

        custom_profile_payload = {
            "version": 1,
            "layers": {
                str(layer_id): {"keep": list(range(8))} for layer_id in range(40)
            },
        }
        custom_profile_response = client.post(
            "/api/profiles",
            json={
                "name": "Three-trial manual eight",
                "description": "A hand-edited profile from the complete smoke run.",
                "model_id": MODEL_ID,
                "profile": custom_profile_payload,
                "source": "multi_source",
                "source_refs": source_refs,
                "metric": "routing_mass",
                "selection_strategy": "manual",
                "selection_config": {"editor": "expert-explorer-v1"},
            },
        )
        assert custom_profile_response.status_code == 201
        custom_profile = custom_profile_response.json()
        assert custom_profile["validation"]["retained_fraction"] == 0.03125
        assert custom_profile["source_refs"] == source_refs
        assert custom_profile["selection_strategy"] == "manual"
        assert custom_profile["selection_config"] == {"editor": "expert-explorer-v1"}
        custom_profile_id = custom_profile["id"]

        proposal_response = client.post(
            f"/api/trials/{first_trial_id}/profile-proposal",
            params={"keep_per_layer": 64, "metric": "routing_mass"},
        )
        assert proposal_response.status_code == 200
        proposal = proposal_response.json()
        assert proposal["validation"]["valid"] is True
        assert proposal["validation"]["retained_fraction"] == 0.25
        assert len(proposal["profile"]["layers"]) == 40
        assert all(
            len(layer["keep"]) == 64 for layer in proposal["profile"]["layers"].values()
        )

        profile_response = client.post(
            "/api/profiles",
            json={
                "name": "Agentic smoke routing mass 64",
                "description": "Generated from a persisted coding trial.",
                "model_id": MODEL_ID,
                "profile": proposal["profile"],
                "source": "agentic",
                "source_trial_id": first_trial_id,
                "metric": "routing_mass",
            },
        )
        assert profile_response.status_code == 201
        profile = profile_response.json()
        assert profile["source"] == "agentic"
        assert profile["source_run_id"] is None
        assert profile["source_trial_id"] == first_trial_id
        assert profile["metric"] == "routing_mass"
        assert profile["observed_mass_retained"] == pytest.approx(
            proposal["observed_mass_retained"]
        )
        assert len(profile["profile_fingerprint"]) == 64
        profile_id = profile["id"]

    with TestClient(create_app(settings)) as restarted:
        restored_run_response = restarted.get(f"/api/agent-runs/{run_id}")
        assert restored_run_response.status_code == 200
        restored_run = restored_run_response.json()
        assert restored_run["status"] == "completed"
        assert restored_run["passed_trials"] == 3
        assert {trial["id"] for trial in restored_run["trials"]} == trial_ids

        assert (
            restarted.get(f"/api/trials/{first_trial_id}/trajectory").status_code == 200
        )
        assert (
            restarted.get(f"/api/trials/{first_trial_id}/routing").json()[
                "captured_inference_calls"
            ]
            == 2
        )
        assert (
            restarted.get(f"/api/trials/{first_trial_id}/artifacts").json()["verifier"][
                "status"
            ]
            == "passed"
        )

        restored_profile = next(
            item
            for item in restarted.get("/api/profiles").json()
            if item["id"] == profile_id
        )
        assert restored_profile["source"] == "agentic"
        assert restored_profile["source_trial_id"] == first_trial_id
        assert restored_profile["profile_fingerprint"] == profile["profile_fingerprint"]
        restored_custom_profile = restarted.get(
            f"/api/profiles/{custom_profile_id}"
        ).json()
        assert restored_custom_profile["source"] == "multi_source"
        assert restored_custom_profile["source_refs"] == source_refs


def test_legacy_agent_run_restart_exposes_unknown_contract_without_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        session = _load_model(client)
        submitted = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "model_session_id": session["id"],
            },
        )
        run_id = submitted.json()["result_id"]
        trial_id = client.get(f"/api/agent-runs/{run_id}").json()["trials"][0]["id"]

    with app.state.lab.store.sessions.begin() as database:
        run_row = database.get(AgentRunRow, run_id)
        assert run_row is not None
        run_payload = json.loads(run_row.record_json)
        for field in (
            "contract_provenance_status",
            "reasoning_mode",
            "evaluation_contract",
            "execution_policy",
        ):
            run_payload.pop(field, None)
        run_payload["task_pack_name"] = "Archived task pack"
        run_payload["task_pack_revision"] = "legacy-revision"
        run_payload["task_pack_content_hash"] = "c" * 64
        run_payload["agent_revision"] = "legacy-agent-revision"
        run_row.record_json = json.dumps(run_payload)

        trial_row = database.get(AgentTrialRow, trial_id)
        assert trial_row is not None
        trial_payload = json.loads(trial_row.record_json)
        trial_payload.pop("task_title_snapshot", None)
        trial_row.record_json = json.dumps(trial_payload)

    with TestClient(create_app(settings)) as restarted:
        response = restarted.get(f"/api/agent-runs/{run_id}")
        exported = restarted.get(f"/api/agent-runs/{run_id}/export")
        explored = restarted.post(
            "/api/routing/explore",
            json={
                "sources": [{"kind": "agent_trial", "id": trial_id}],
                "metric": "routing_mass",
            },
        )

    assert response.status_code == 200
    archived = response.json()
    assert archived["task_pack_name"] == "Archived task pack"
    assert archived["task_pack_revision"] == "legacy-revision"
    assert archived["task_pack_content_hash"] == "c" * 64
    assert archived["agent_revision"] == "legacy-agent-revision"
    assert archived["generation"] == run_payload["generation"]
    assert archived["budgets"] == run_payload["budgets"]
    assert archived["contract_provenance_status"] == "legacy_unknown"
    assert archived["reasoning_mode"] is None
    assert archived["evaluation_contract"] is None
    assert archived["evaluation_contract_fingerprint"] is None
    assert archived["execution_policy"] is None
    assert archived["trials"][0]["task_provenance_status"] == "legacy_unknown"
    assert archived["trials"][0]["title"] == (f"Unknown archived task ({TASK_IDS[0]})")
    assert "Fix subtraction" not in response.text
    assert exported.status_code == 200
    assert explored.status_code == 200
    assert explored.json()["sources"][0]["label"] == (
        f"Unknown archived task ({TASK_IDS[0]}) · attempt 1"
    )
    exported_run = exported.json()["detail"]["run"]
    for field in (
        "task_pack_name",
        "task_pack_revision",
        "task_pack_content_hash",
        "agent_revision",
        "generation",
        "budgets",
        "contract_provenance_status",
    ):
        assert exported_run[field] == archived[field]
    assert exported_run["trials"][0]["title"] == archived["trials"][0]["title"]
    assert "reasoning_mode" not in exported_run
    assert "evaluation_contract" not in exported_run
    assert "execution_policy" not in exported_run


class _EmptyAsyncIterator:
    def __aiter__(self) -> _EmptyAsyncIterator:
        return self

    async def __anext__(self) -> NoReturn:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


class _ReadOnlyDaytonaClient:
    def __init__(self) -> None:
        self.list_calls = 0
        self.create_calls = 0
        self.closed = False

    def list(self) -> _EmptyAsyncIterator:
        self.list_calls += 1
        return _EmptyAsyncIterator()

    async def create(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.create_calls += 1
        raise AssertionError("preflight must never create a billed sandbox")

    async def close(self) -> None:
        self.closed = True


class _DaytonaNotFound(Exception):
    pass


class _DaytonaFileSystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def upload_file(self, content: bytes, path: str, timeout: int) -> None:
        assert timeout > 0
        self.files[path] = content

    async def get_file_info(self, path: str) -> SimpleNamespace:
        content = self.files[path]
        return SimpleNamespace(is_dir=False, size=len(content))

    async def download_file(self, path: str, timeout: int) -> bytes:
        assert timeout > 0
        return self.files[path]


class _DaytonaProcess:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str | None, dict[str, str] | None]] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int,
    ) -> SimpleNamespace:
        assert timeout > 0
        self.commands.append((command, cwd, env))
        if command == "sleep forever":
            raise TimeoutError("provider detail must be sanitized")
        result = "command output\n" if command == "python hello.py" else ""
        return SimpleNamespace(exit_code=0, result=result)


class _LifecycleDaytonaSandbox:
    def __init__(self, sandbox_id: str, labels: dict[str, str]) -> None:
        self.id = sandbox_id
        self.labels = labels
        self.state = "started"
        self.process = _DaytonaProcess()
        self.fs = _DaytonaFileSystem()


class _LifecycleDaytonaClient:
    def __init__(self) -> None:
        self.create_params: dict[str, object] | None = None
        self.create_timeout: float | None = None
        self.deleted = False
        self.closed = False
        self.sandbox: _LifecycleDaytonaSandbox | None = None

    async def create(
        self, params: dict[str, object], *, timeout: float
    ) -> _LifecycleDaytonaSandbox:
        self.create_params = params
        self.create_timeout = timeout
        self.sandbox = _LifecycleDaytonaSandbox(
            "daytona-test-sandbox", dict(params["labels"])
        )
        return self.sandbox

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        if self.deleted or self.sandbox is None or sandbox_id != self.sandbox.id:
            raise _DaytonaNotFound
        return self.sandbox

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert self.sandbox is sandbox
        assert timeout > 0
        self.deleted = True

    async def close(self) -> None:
        self.closed = True


class _FailedWorkspaceDaytonaProcess(_DaytonaProcess):
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int,
    ) -> SimpleNamespace:
        assert timeout > 0
        self.commands.append((command, cwd, env))
        return SimpleNamespace(exit_code=1, result="workspace setup failed")


class _CreateCleanupDaytonaClient:
    def __init__(self, *, recovery_succeeds: bool) -> None:
        self.recovery_succeeds = recovery_succeeds
        self.sandbox: _LifecycleDaytonaSandbox | None = None
        self.deleted = False
        self.delete_calls = 0
        self.list_queries: list[dict[str, object] | None] = []
        self.closed = False

    async def create(
        self, params: dict[str, object], *, timeout: float
    ) -> _LifecycleDaytonaSandbox:
        assert timeout > 0
        self.sandbox = _LifecycleDaytonaSandbox(
            "daytona-create-failure", dict(params["labels"])
        )
        self.sandbox.process = _FailedWorkspaceDaytonaProcess()
        return self.sandbox

    def list(
        self, query: dict[str, object] | None = None
    ) -> AsyncIterator[_LifecycleDaytonaSandbox]:
        self.list_queries.append(query)

        async def iterate() -> AsyncIterator[_LifecycleDaytonaSandbox]:
            if query is not None and self.sandbox is not None and not self.deleted:
                yield self.sandbox

        return iterate()

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        if self.deleted or self.sandbox is None or sandbox_id != self.sandbox.id:
            raise _DaytonaNotFound
        return self.sandbox

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert self.sandbox is sandbox
        assert timeout > 0
        self.delete_calls += 1
        if self.delete_calls == 1 or not self.recovery_succeeds:
            raise ConnectionError("simulated delete failure")
        self.deleted = True

    async def close(self) -> None:
        self.closed = True


class _RecoveryDaytonaClient:
    def __init__(self, sandboxes: list[_LifecycleDaytonaSandbox]) -> None:
        self.sandboxes = {sandbox.id: sandbox for sandbox in sandboxes}
        self.deleted_ids: list[str] = []
        self.closed = False

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        try:
            return self.sandboxes[sandbox_id]
        except KeyError as error:
            raise _DaytonaNotFound from error

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert timeout > 0
        self.deleted_ids.append(sandbox.id)
        del self.sandboxes[sandbox.id]

    async def close(self) -> None:
        self.closed = True


def test_daytona_preflight_is_read_only_and_api_responses_redact_secret(
    tmp_path: Path,
) -> None:
    secret = "daytona-test-secret-never-return-this"
    sdk_client = _ReadOnlyDaytonaClient()
    seen_configs: list[DaytonaProviderConfig] = []

    def client_factory(config: DaytonaProviderConfig) -> _ReadOnlyDaytonaClient:
        seen_configs.append(config)
        return sdk_client

    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(
            api_key=secret,
            api_url="https://app.daytona.io/api",
            target="us",
        ),
        client_factory=client_factory,
    )
    app = create_app(_settings(tmp_path, daytona_api_key=secret))
    app.state.lab.agentic.providers["daytona"] = provider

    with TestClient(app) as client:
        providers_response = client.get("/api/sandbox-providers")
        assert providers_response.status_code == 200
        daytona = next(
            item for item in providers_response.json() if item["id"] == "daytona"
        )
        assert daytona["configured"] is True
        assert daytona["credential_mode"] == "api_key"
        assert daytona["status"] == "unchecked"
        assert secret not in providers_response.text

        preflight_response = client.post("/api/sandbox-providers/daytona/preflight")
        assert preflight_response.status_code == 200
        preflight = preflight_response.json()
        assert preflight["provider_id"] == "daytona"
        assert preflight["configured"] is True
        assert preflight["reachable"] is True
        assert preflight["authenticated"] is True
        assert preflight["status"] == "ready"
        assert preflight["api_url_host"] == "app.daytona.io"
        assert preflight["region"] == "us"
        assert preflight["error"] is None
        assert secret not in preflight_response.text
        assert sdk_client.list_calls == 1
        assert sdk_client.create_calls == 0
        assert len(seen_configs) == 1
        assert seen_configs[0].api_key is not None
        assert seen_configs[0].api_key.get_secret_value() == secret

    assert sdk_client.closed is True


@pytest.mark.asyncio
async def test_daytona_provider_remote_lifecycle_is_private_bounded_and_deleted() -> (
    None
):
    sdk_client = _LifecycleDaytonaClient()
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key="configured-test-key", target="us"),
        client_factory=lambda config: sdk_client,
    )
    ownership = SandboxOwnership(
        controller_id="test-controller",
        run_id="run-1",
        trial_id="trial-1",
    )
    spec = SandboxSpec(
        image_ref="python:3.12-slim",
        working_directory="/workspace/task",
        network_policy=NetworkPolicy.NONE,
        cpu=1,
        memory_mb=1024,
        disk_mb=2048,
    )

    try:
        handle = await provider.create(spec, ownership)
        assert handle.id == "daytona-test-sandbox"
        assert sdk_client.create_timeout == 180
        params = sdk_client.create_params
        assert params is not None
        assert params["public"] is False
        assert params["ephemeral"] is True
        assert params["network_block_all"] is True
        assert params["domain_allow_list"] is None
        assert params["auto_stop_interval"] == 15
        assert "auto_delete_interval" not in params
        assert params["env_vars"] == {}
        assert params["secrets"] == {}
        assert params["resources"] == {"cpu": 1, "memory": 1, "disk": 3}
        assert params["labels"] == {
            "moe-tools-controller": "test-controller",
            "moe-tools-run": "run-1",
            "moe-tools-trial": "trial-1",
            "moe-tools-managed": "true",
            "moe-tools-provider": "daytona",
        }

        await provider.upload(
            handle,
            SandboxFile(
                path="hello.py",
                content='print("hello")\n',
                executable=True,
            ),
        )
        assert sdk_client.sandbox is not None
        assert sdk_client.sandbox.fs.files["/workspace/task/hello.py"] == (
            b'print("hello")\n'
        )
        assert await provider.read(handle, "hello.py") == b'print("hello")\n'

        result = await provider.exec(
            handle,
            "python hello.py",
            cwd="/workspace/task",
            timeout_seconds=20,
        )
        assert result.exit_code == 0
        assert result.stdout == "command output\n"
        assert result.stderr == ""
        assert sdk_client.sandbox.process.commands[-1] == (
            "python hello.py",
            "/workspace/task",
            {},
        )

        timed_out = await provider.exec(
            handle,
            "sleep forever",
            cwd="/workspace/task",
            timeout_seconds=1,
        )
        assert timed_out.exit_code is None
        assert timed_out.timed_out is True
        assert timed_out.stderr == "The Daytona request timed out."
        assert "provider detail" not in timed_out.stderr

        await provider.delete(handle)
        assert sdk_client.deleted is True
        await provider.delete(handle)
    finally:
        await provider.aclose()

    assert sdk_client.closed is True


@pytest.mark.parametrize(
    ("recovery_succeeds", "expected_state", "expected_termination"),
    [
        (True, SandboxSessionState.DELETED, TerminationCause.SANDBOX_ERROR),
        (False, SandboxSessionState.CLEANUP_PENDING, TerminationCause.CLEANUP_ERROR),
    ],
)
def test_failed_daytona_create_uses_label_scoped_cleanup(
    tmp_path: Path,
    recovery_succeeds: bool,
    expected_state: SandboxSessionState,
    expected_termination: TerminationCause,
) -> None:
    sdk_client = _CreateCleanupDaytonaClient(recovery_succeeds=recovery_succeeds)
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key="configured-test-key", target="us"),
        client_factory=lambda config: sdk_client,
    )
    app = create_app(_settings(tmp_path, daytona_api_key="configured-test-key"))
    app.state.lab.agentic.providers["daytona"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        preflight = client.post("/api/sandbox-providers/daytona/preflight")
        assert preflight.status_code == 200
        assert preflight.json()["status"] == "ready"
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "daytona",
                "model_session_id": session["id"],
            },
        )
        assert response.status_code == 202
        run_id = response.json()["result_id"]
        trial = next(
            item
            for item in app.state.lab.agentic.trials.values()
            if item.run_id == run_id
        )
        assert trial.sandbox_session_id is not None
        sandbox = app.state.lab.agentic.sandboxes[trial.sandbox_session_id]

        assert sandbox.external_id is None
        assert sandbox.state is expected_state
        assert trial.termination_cause is expected_termination
        persisted = app.state.lab.store.load_sandbox_sessions()[sandbox.id]
        assert persisted.state is expected_state
        assert persisted.cleanup_error == sandbox.cleanup_error

        recovery_query = next(
            query for query in reversed(sdk_client.list_queries) if query is not None
        )
        assert recovery_query == {
            "labels": ownership_labels(sandbox.ownership, "daytona"),
            "limit": 10,
        }
        assert sdk_client.delete_calls == 2
        assert sdk_client.deleted is recovery_succeeds
        if recovery_succeeds:
            assert sandbox.deleted_at is not None
            assert sandbox.cleanup_error is None
        else:
            assert sandbox.deleted_at is None
            assert sandbox.cleanup_error == "The Daytona API could not be reached."

    assert sdk_client.closed is True


@pytest.mark.asyncio
async def test_startup_recovers_only_label_matched_daytona_sandboxes_and_metrics(
    tmp_path: Path,
) -> None:
    secret = "startup-recovery-secret"
    settings = _settings(tmp_path, daytona_api_key=secret)
    initial = ResearchLab(settings)
    session = await initial.create_model_session(
        CreateModelSessionRequest(model_id=MODEL_ID)
    )
    request = CreateAgentRunRequest(
        task_pack_id=PACK_ID,
        task_ids=TASK_IDS,
        agent_id="bash-json-v1",
        sandbox_provider_id="fake",
        model_session_id=session.id,
    )
    run = initial.agentic.create_run(
        run_id="interrupted-run",
        job_id="interrupted-job",
        request=request,
        model_session=session,
    )
    run.status = AgentRunStatus.RUNNING
    initial.store.save_agent_run(run)
    trials = [
        trial for trial in initial.agentic.trials.values() if trial.run_id == run.id
    ]
    owned = SandboxOwnership(
        controller_id=settings.agent_controller_id,
        run_id=run.id,
        trial_id=trials[0].id,
    )
    mismatched = SandboxOwnership(
        controller_id=settings.agent_controller_id,
        run_id=run.id,
        trial_id=trials[1].id,
    )
    task_spec = initial.agentic.catalog.get_task(PACK_ID, TASK_IDS[0]).sandbox_spec()
    sessions = [
        SandboxSession(
            id="owned-session",
            trial_id=owned.trial_id,
            provider_id="daytona",
            external_id="owned-sandbox",
            state=SandboxSessionState.READY,
            ownership=owned,
            spec=task_spec,
        ),
        SandboxSession(
            id="mismatched-session",
            trial_id=mismatched.trial_id,
            provider_id="daytona",
            external_id="mismatched-sandbox",
            state=SandboxSessionState.READY,
            ownership=mismatched,
            spec=task_spec,
        ),
    ]
    for sandbox_session in sessions:
        initial.agentic.sandboxes[sandbox_session.id] = sandbox_session
        initial.store.save_sandbox_session(sandbox_session)
    await initial.shutdown()

    correct_remote = _LifecycleDaytonaSandbox(
        "owned-sandbox",
        ownership_labels(owned, "daytona"),
    )
    wrong_labels = ownership_labels(mismatched, "daytona")
    wrong_labels["moe-tools-trial"] = "some-other-trial"
    mismatched_remote = _LifecycleDaytonaSandbox(
        "mismatched-sandbox",
        wrong_labels,
    )
    sdk_client = _RecoveryDaytonaClient([correct_remote, mismatched_remote])
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key=secret, target="us"),
        client_factory=lambda config: sdk_client,
    )
    app = create_app(settings)
    app.state.lab.agentic.providers["daytona"] = provider

    reconciled = app.state.lab.agentic.runs[run.id]
    assert reconciled.status == "failed"
    assert reconciled.completed_trials == reconciled.total_trials == 3
    assert all(
        trial.status == "error"
        for trial in app.state.lab.agentic.trials.values()
        if trial.run_id == run.id
    )

    with TestClient(app):
        owned_session = app.state.lab.agentic.sandboxes["owned-session"]
        mismatched_session = app.state.lab.agentic.sandboxes["mismatched-session"]
        assert owned_session.state == "deleted"
        assert owned_session.deleted_at is not None
        assert owned_session.cleanup_error is None
        assert mismatched_session.state == "error"
        assert mismatched_session.deleted_at is None
        assert mismatched_session.cleanup_error == (
            "The sandbox is not owned by this controller run."
        )
        assert secret not in (mismatched_session.cleanup_error or "")
        assert sdk_client.deleted_ids == ["owned-sandbox"]
        assert "mismatched-sandbox" in sdk_client.sandboxes

        persisted = app.state.lab.store.load_sandbox_sessions()
        assert persisted["owned-session"].state == "deleted"
        assert persisted["mismatched-session"].state == "error"

    assert sdk_client.closed is True


def _multi_verifier_judge_contract() -> EvaluationContract:
    return EvaluationContract(
        name="Two custom verifiers and judge",
        criteria=[
            EvaluationCriterion(
                id="custom_one",
                label="First custom verifier",
                kind=DeterministicScorerKind.VERIFIER,
                visibility=CriterionVisibility.HIDDEN,
                verifier_command="custom-verifier-one",
                verifier_timeout_seconds=60,
            ),
            EvaluationCriterion(
                id="custom_two",
                label="Second custom verifier",
                kind=DeterministicScorerKind.VERIFIER,
                visibility=CriterionVisibility.HIDDEN,
                verifier_command="custom-verifier-two",
                verifier_timeout_seconds=60,
            ),
        ],
        judge=LLMJudgeConfig(
            provider=JudgeProvider.FAKE,
            model="deterministic-fake-judge-v1",
            rubric="Judge correctness.",
        ),
        judge_weight=0.5,
    )


@pytest.mark.asyncio
async def test_cancel_during_custom_verifier_stops_fanout_and_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    lab = ResearchLab(_settings(tmp_path))
    try:
        session = await lab.create_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        lab.agentic.create_run(
            run_id="cancel-during-verifier",
            job_id="cancel-during-verifier-job",
            request=CreateAgentRunRequest(
                task_pack_id=PACK_ID,
                task_ids=[TASK_IDS[0]],
                model_session_id=session.id,
                evaluation_contract=_multi_verifier_judge_contract(),
            ),
            model_session=session,
        )
        original_clean_room = lab.agentic._run_clean_room_command
        verifier_slots: list[str] = []
        cancelled = False

        async def cancel_after_first_custom(**kwargs: object):
            nonlocal cancelled
            verifier_slots.append(str(kwargs["slot"]))
            result = await original_clean_room(**kwargs)
            if kwargs["slot"] == "criterion-1":
                cancelled = True
            return result

        monkeypatch.setattr(
            lab.agentic,
            "_run_clean_room_command",
            cancel_after_first_custom,
        )
        judge_calls = 0
        original_evaluate = lab.judges.evaluate

        async def count_judge_calls(*args: object, **kwargs: object):
            nonlocal judge_calls
            judge_calls += 1
            return await original_evaluate(*args, **kwargs)

        monkeypatch.setattr(lab.judges, "evaluate", count_judge_calls)
        run = await lab.agentic.execute_run(
            "cancel-during-verifier",
            model_session=session,
            on_progress=lambda current, total: None,
            should_cancel=lambda: cancelled,
        )

        trial = next(
            item for item in lab.agentic.trials.values() if item.run_id == run.id
        )
        assert verifier_slots == ["primary", "criterion-1"]
        assert judge_calls == 0
        assert run.status == "cancelled"
        assert trial.status == "cancelled"
        assert trial.termination_cause == "cancelled"
        assert trial.evaluation is None
        assert all(
            sandbox.state == "deleted"
            for sandbox in lab.agentic.sandboxes.values()
            if sandbox.trial_id == trial.id
        )
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_trial_wall_time_stops_multi_verifier_fanout_before_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(
        "moe_tools_suite.agentic.controller.time",
        SimpleNamespace(
            monotonic=lambda: clock.value,
            perf_counter=lambda: clock.value,
        ),
    )
    lab = ResearchLab(_settings(tmp_path))
    try:
        session = await lab.create_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        lab.agentic.create_run(
            run_id="verifier-wall-time",
            job_id="verifier-wall-time-job",
            request=CreateAgentRunRequest(
                task_pack_id=PACK_ID,
                task_ids=[TASK_IDS[0]],
                model_session_id=session.id,
                budgets=AgentBudgets(timeout_seconds=1),
                evaluation_contract=_multi_verifier_judge_contract(),
            ),
            model_session=session,
        )
        original_clean_room = lab.agentic._run_clean_room_command
        verifier_calls: list[tuple[str, int]] = []

        async def exhaust_time_after_first_custom(**kwargs: object):
            verifier_calls.append((str(kwargs["slot"]), int(kwargs["timeout_seconds"])))
            result = await original_clean_room(**kwargs)
            if kwargs["slot"] == "criterion-1":
                clock.value = 1.1
            return result

        monkeypatch.setattr(
            lab.agentic,
            "_run_clean_room_command",
            exhaust_time_after_first_custom,
        )
        judge_calls = 0
        original_evaluate = lab.judges.evaluate

        async def count_judge_calls(*args: object, **kwargs: object):
            nonlocal judge_calls
            judge_calls += 1
            return await original_evaluate(*args, **kwargs)

        monkeypatch.setattr(lab.judges, "evaluate", count_judge_calls)
        run = await lab.agentic.execute_run(
            "verifier-wall-time",
            model_session=session,
            on_progress=lambda current, total: None,
            should_cancel=lambda: False,
        )

        trial = next(
            item for item in lab.agentic.trials.values() if item.run_id == run.id
        )
        assert verifier_calls == [("primary", 1), ("criterion-1", 1)]
        assert judge_calls == 0
        assert run.status == "completed"
        assert trial.status == "failed"
        assert trial.termination_cause == "time_limit"
        assert trial.evaluation is None
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_controller_cancellation_cleans_an_active_fake_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    lab = ResearchLab(_settings(tmp_path))
    try:
        session = await lab.create_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        request = CreateAgentRunRequest(
            task_pack_id=PACK_ID,
            task_ids=[TASK_IDS[0]],
            agent_id="bash-json-v1",
            sandbox_provider_id="fake",
            model_session_id=session.id,
        )
        lab.agentic.create_run(
            run_id="cancel-run",
            job_id="cancel-job",
            request=request,
            model_session=session,
        )
        cancellation_checks = 0

        def should_cancel() -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        progress: list[tuple[int, int]] = []
        run = await lab.agentic.execute_run(
            "cancel-run",
            model_session=session,
            on_progress=lambda current, total: progress.append((current, total)),
            should_cancel=should_cancel,
        )

        assert run.status == "cancelled"
        assert run.completed_trials == 1
        assert run.passed_trials == 0
        assert progress == [(1, 1)]
        trial = next(
            item for item in lab.agentic.trials.values() if item.run_id == run.id
        )
        assert trial.status == "cancelled"
        assert trial.termination_cause == "cancelled"
        assert trial.reward is None
        assert trial.turns == 0
        assert trial.completed_at is not None

        assert trial.sandbox_session_id is not None
        sandbox = lab.agentic.sandboxes[trial.sandbox_session_id]
        assert sandbox.state == "deleted"
        assert sandbox.external_id is not None
        assert sandbox.deleted_at is not None
        assert sandbox.cleanup_error is None
        fake_provider = lab.agentic.providers["fake"]
        assert fake_provider._sandboxes == {}
        assert fake_provider.commands == []
    finally:
        await lab.shutdown()
