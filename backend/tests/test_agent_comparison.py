from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.agentic.comparison import (
    AgentComparisonTransition,
    CreateAgentComparisonRequest,
    compare_agent_runs,
)
from moe_tools_suite.agentic.domain import (
    AgentBudgets,
    AgentRun,
    AgentRunStatus,
    AgentTrial,
    AgentTrialStatus,
    TrialPerformance,
)
from moe_tools_suite.domain import GenerationConfig
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings

FINGERPRINT = "f" * 64


def _run(run_id: str, *, profile_id: str | None = None) -> AgentRun:
    return AgentRun(
        id=run_id,
        job_id=f"job-{run_id}",
        status=AgentRunStatus.COMPLETED,
        task_pack_id="pack",
        task_pack_name="Repository pack",
        task_pack_revision="1",
        task_pack_content_hash="a" * 64,
        task_ids=["task-a", "task-b"],
        model_session_id=f"session-{run_id}",
        profile_id=profile_id,
        profile_fingerprint="b" * 64 if profile_id else None,
        agent_id="agent",
        agent_revision="1",
        sandbox_provider_id="daytona",
        generation=GenerationConfig(),
        budgets=AgentBudgets(),
        contract_fingerprint=FINGERPRINT,
        completed_trials=2,
        total_trials=2,
        passed_trials=1,
        mean_reward=0.5,
    )


def _trial(
    run_id: str,
    task_id: str,
    status: AgentTrialStatus,
    reward: float | None,
    *,
    tokens: int,
) -> AgentTrial:
    return AgentTrial(
        id=f"{run_id}-{task_id}",
        run_id=run_id,
        task_id=task_id,
        model_session_id=f"session-{run_id}",
        status=status,
        reward=reward,
        performance=TrialPerformance(
            wall_time_ms=100,
            model_time_ms=60,
            sandbox_time_ms=20,
            verifier_time_ms=20,
            prompt_tokens=tokens - 2,
            reasoning_tokens=1,
            completion_tokens=2,
            total_tokens=tokens,
            mean_tps=10,
            estimated_cost_usd=0,
        ),
    )


def test_agent_comparison_pairs_quality_and_reported_performance() -> None:
    baseline = _run("baseline")
    candidate = _run("candidate", profile_id="profile")
    baseline_trials = [
        _trial("baseline", "task-a", AgentTrialStatus.PASSED, 1, tokens=12),
        _trial("baseline", "task-b", AgentTrialStatus.FAILED, 0, tokens=8),
    ]
    candidate_trials = [
        _trial("candidate", "task-a", AgentTrialStatus.FAILED, 0, tokens=10),
        _trial("candidate", "task-b", AgentTrialStatus.PASSED, 1, tokens=10),
    ]

    comparison = compare_agent_runs(
        CreateAgentComparisonRequest(
            baseline_run_id=baseline.id,
            candidate_run_id=candidate.id,
        ),
        baseline,
        candidate,
        baseline_trials,
        candidate_trials,
    )

    assert comparison.regressions == 1
    assert comparison.recoveries == 1
    assert comparison.retained_passes == comparison.retained_failures == 0
    assert [pair.transition for pair in comparison.pairs] == [
        AgentComparisonTransition.REGRESSION,
        AgentComparisonTransition.RECOVERY,
    ]
    assert comparison.baseline_performance.total_tokens == 20
    assert comparison.candidate_performance.total_tokens == 20
    assert comparison.baseline_performance.reported_mean_tps == 10
    assert comparison.candidate_profile_id == "profile"
    assert len(comparison.id) == 64


def test_agent_comparison_rejects_contract_or_cohort_drift() -> None:
    baseline = _run("baseline")
    candidate = _run("candidate").model_copy(update={"contract_fingerprint": "c" * 64})
    trials = [
        _trial("baseline", "task-a", AgentTrialStatus.PASSED, 1, tokens=4),
        _trial("baseline", "task-b", AgentTrialStatus.PASSED, 1, tokens=4),
    ]

    with pytest.raises(ValueError, match="different task, agent, provider, or budgets"):
        compare_agent_runs(
            CreateAgentComparisonRequest(
                baseline_run_id="baseline",
                candidate_run_id="candidate",
            ),
            baseline,
            candidate,
            trials,
            [
                trial.model_copy(
                    update={
                        "id": trial.id.replace("baseline", "candidate"),
                        "run_id": "candidate",
                    }
                )
                for trial in trials
            ],
        )


def test_agent_comparison_api_pairs_persisted_run_state(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    baseline = _run("baseline")
    candidate = _run("candidate", profile_id="profile")
    trials = [
        _trial("baseline", "task-a", AgentTrialStatus.PASSED, 1, tokens=4),
        _trial("baseline", "task-b", AgentTrialStatus.FAILED, 0, tokens=4),
        _trial("candidate", "task-a", AgentTrialStatus.PASSED, 1, tokens=4),
        _trial("candidate", "task-b", AgentTrialStatus.PASSED, 1, tokens=4),
    ]
    app.state.lab.agentic.runs = {
        baseline.id: baseline,
        candidate.id: candidate,
    }
    app.state.lab.agentic.trials = {trial.id: trial for trial in trials}

    with TestClient(app) as client:
        response = client.post(
            "/api/agent-comparisons",
            json={
                "baseline_run_id": baseline.id,
                "candidate_run_id": candidate.id,
                "name": "Masked coding check",
            },
        )

    assert response.status_code == 200
    comparison = response.json()
    assert comparison["name"] == "Masked coding check"
    assert comparison["recoveries"] == 1
    assert comparison["regressions"] == 0
    assert comparison["candidate_profile_id"] == "profile"
