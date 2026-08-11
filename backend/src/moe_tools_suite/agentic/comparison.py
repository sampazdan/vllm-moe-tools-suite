from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from .domain import (
    AgenticModel,
    AgentRun,
    AgentRunStatus,
    AgentTrial,
    AgentTrialStatus,
    TrialPerformance,
)


class AgentComparisonTransition(StrEnum):
    REGRESSION = "regression"
    RECOVERY = "recovery"
    RETAINED_PASS = "retained_pass"
    RETAINED_FAILURE = "retained_failure"
    UNSCORED = "unscored"


class CreateAgentComparisonRequest(AgenticModel):
    baseline_run_id: Annotated[str, Field(min_length=1, max_length=200)]
    candidate_run_id: Annotated[str, Field(min_length=1, max_length=200)]
    name: Annotated[str | None, Field(max_length=120)] = None

    @model_validator(mode="after")
    def require_distinct_runs(self) -> CreateAgentComparisonRequest:
        if self.baseline_run_id == self.candidate_run_id:
            raise ValueError("agent comparison requires two different runs")
        return self


class AgentTrialPair(AgenticModel):
    task_id: str
    attempt: Annotated[int, Field(ge=1)]
    baseline_trial_id: str
    candidate_trial_id: str
    baseline_status: AgentTrialStatus
    candidate_status: AgentTrialStatus
    baseline_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    candidate_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    reward_delta: Annotated[float, Field(ge=-1, le=1)] | None = None
    transition: AgentComparisonTransition
    baseline_performance: TrialPerformance | None = None
    candidate_performance: TrialPerformance | None = None


class AgentRunPerformanceSummary(AgenticModel):
    wall_time_ms: Annotated[float, Field(ge=0)] = 0
    model_time_ms: Annotated[float, Field(ge=0)] = 0
    sandbox_time_ms: Annotated[float, Field(ge=0)] = 0
    verifier_time_ms: Annotated[float, Field(ge=0)] = 0
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    reported_mean_tps: Annotated[float, Field(ge=0)] | None = None
    reported_tps_trials: Annotated[int, Field(ge=0)] = 0
    inference_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_cost_usd: Annotated[float, Field(ge=0)] | None = None
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None


class AgentRunComparison(AgenticModel):
    id: Annotated[str, Field(min_length=64, max_length=64)]
    name: str
    baseline_run_id: str
    candidate_run_id: str
    task_pack_id: str
    compatibility_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    baseline_profile_id: str | None = None
    candidate_profile_id: str | None = None
    baseline_profile_fingerprint: str | None = None
    candidate_profile_fingerprint: str | None = None
    trial_count: Annotated[int, Field(ge=1)]
    baseline_passed_trials: Annotated[int, Field(ge=0)]
    candidate_passed_trials: Annotated[int, Field(ge=0)]
    baseline_mean_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    candidate_mean_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    mean_reward_delta: Annotated[float, Field(ge=-1, le=1)] | None = None
    regressions: Annotated[int, Field(ge=0)] = 0
    recoveries: Annotated[int, Field(ge=0)] = 0
    retained_passes: Annotated[int, Field(ge=0)] = 0
    retained_failures: Annotated[int, Field(ge=0)] = 0
    unscored: Annotated[int, Field(ge=0)] = 0
    baseline_performance: AgentRunPerformanceSummary
    candidate_performance: AgentRunPerformanceSummary
    pairs: list[AgentTrialPair]


def compare_agent_runs(
    request: CreateAgentComparisonRequest,
    baseline: AgentRun,
    candidate: AgentRun,
    baseline_trials: list[AgentTrial],
    candidate_trials: list[AgentTrial],
) -> AgentRunComparison:
    """Create a reproducible paired comparison from two durable agent runs."""

    if baseline.id != request.baseline_run_id:
        raise ValueError("baseline run does not match comparison request")
    if candidate.id != request.candidate_run_id:
        raise ValueError("candidate run does not match comparison request")
    if baseline.status is not AgentRunStatus.COMPLETED:
        raise ValueError("baseline agent run is not completed")
    if candidate.status is not AgentRunStatus.COMPLETED:
        raise ValueError("candidate agent run is not completed")
    if baseline.contract_fingerprint != candidate.contract_fingerprint:
        raise ValueError("agent runs use different task, agent, provider, or budgets")
    if baseline.task_pack_id != candidate.task_pack_id:
        raise ValueError("agent runs use different task packs")

    baseline_by_key = _trials_by_key(baseline_trials, baseline.id)
    candidate_by_key = _trials_by_key(candidate_trials, candidate.id)
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("agent runs do not contain the same task attempts")

    pairs = [
        _pair_trials(baseline_by_key[key], candidate_by_key[key])
        for key in sorted(baseline_by_key, key=lambda value: (value[0], value[1]))
    ]
    transitions = [pair.transition for pair in pairs]
    fingerprint_payload = {
        "baseline_run_id": baseline.id,
        "candidate_run_id": candidate.id,
        "compatibility_fingerprint": baseline.contract_fingerprint,
        "pairs": [
            {
                "task_id": pair.task_id,
                "attempt": pair.attempt,
                "baseline_trial_id": pair.baseline_trial_id,
                "candidate_trial_id": pair.candidate_trial_id,
                "baseline_reward": pair.baseline_reward,
                "candidate_reward": pair.candidate_reward,
                "transition": pair.transition.value,
            }
            for pair in pairs
        ],
    }
    comparison_id = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    baseline_mean = _mean_reward(baseline_trials)
    candidate_mean = _mean_reward(candidate_trials)
    return AgentRunComparison(
        id=comparison_id,
        name=request.name
        or f"{baseline.task_pack_name} · {baseline.id[:8]} → {candidate.id[:8]}",
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        task_pack_id=baseline.task_pack_id,
        compatibility_fingerprint=baseline.contract_fingerprint,
        baseline_profile_id=baseline.profile_id,
        candidate_profile_id=candidate.profile_id,
        baseline_profile_fingerprint=baseline.profile_fingerprint,
        candidate_profile_fingerprint=candidate.profile_fingerprint,
        trial_count=len(pairs),
        baseline_passed_trials=sum(
            trial.status is AgentTrialStatus.PASSED for trial in baseline_trials
        ),
        candidate_passed_trials=sum(
            trial.status is AgentTrialStatus.PASSED for trial in candidate_trials
        ),
        baseline_mean_reward=baseline_mean,
        candidate_mean_reward=candidate_mean,
        mean_reward_delta=(
            candidate_mean - baseline_mean
            if baseline_mean is not None and candidate_mean is not None
            else None
        ),
        regressions=transitions.count(AgentComparisonTransition.REGRESSION),
        recoveries=transitions.count(AgentComparisonTransition.RECOVERY),
        retained_passes=transitions.count(AgentComparisonTransition.RETAINED_PASS),
        retained_failures=transitions.count(AgentComparisonTransition.RETAINED_FAILURE),
        unscored=transitions.count(AgentComparisonTransition.UNSCORED),
        baseline_performance=_performance_summary(baseline_trials),
        candidate_performance=_performance_summary(candidate_trials),
        pairs=pairs,
    )


def _trials_by_key(
    trials: list[AgentTrial], run_id: str
) -> dict[tuple[str, int], AgentTrial]:
    if any(trial.run_id != run_id for trial in trials):
        raise ValueError("comparison includes a trial from the wrong run")
    indexed = {(trial.task_id, trial.attempt): trial for trial in trials}
    if len(indexed) != len(trials):
        raise ValueError("agent run contains duplicate task attempts")
    return indexed


def _pair_trials(baseline: AgentTrial, candidate: AgentTrial) -> AgentTrialPair:
    baseline_passed = _passed(baseline)
    candidate_passed = _passed(candidate)
    if baseline_passed is None or candidate_passed is None:
        transition = AgentComparisonTransition.UNSCORED
    elif baseline_passed and not candidate_passed:
        transition = AgentComparisonTransition.REGRESSION
    elif not baseline_passed and candidate_passed:
        transition = AgentComparisonTransition.RECOVERY
    elif baseline_passed:
        transition = AgentComparisonTransition.RETAINED_PASS
    else:
        transition = AgentComparisonTransition.RETAINED_FAILURE
    return AgentTrialPair(
        task_id=baseline.task_id,
        attempt=baseline.attempt,
        baseline_trial_id=baseline.id,
        candidate_trial_id=candidate.id,
        baseline_status=baseline.status,
        candidate_status=candidate.status,
        baseline_reward=baseline.reward,
        candidate_reward=candidate.reward,
        reward_delta=(
            candidate.reward - baseline.reward
            if baseline.reward is not None and candidate.reward is not None
            else None
        ),
        transition=transition,
        baseline_performance=baseline.performance,
        candidate_performance=candidate.performance,
    )


def _passed(trial: AgentTrial) -> bool | None:
    if trial.status is AgentTrialStatus.PASSED:
        return True
    if trial.status in {AgentTrialStatus.FAILED, AgentTrialStatus.ERROR}:
        return False
    return None


def _mean_reward(trials: list[AgentTrial]) -> float | None:
    rewards = [trial.reward for trial in trials if trial.reward is not None]
    return sum(rewards) / len(rewards) if rewards else None


def _performance_summary(
    trials: list[AgentTrial],
) -> AgentRunPerformanceSummary:
    performance = [trial.performance for trial in trials if trial.performance]
    reasoning = [
        item.reasoning_tokens
        for item in performance
        if item.reasoning_tokens is not None
    ]
    tps = [item.mean_tps for item in performance if item.mean_tps is not None]
    costs = [
        item.estimated_cost_usd
        for item in performance
        if item.estimated_cost_usd is not None
    ]
    inference_costs = [
        item.inference_cost_usd
        for item in performance
        if item.inference_cost_usd is not None
    ]
    judge_costs = [
        item.judge_cost_usd for item in performance if item.judge_cost_usd is not None
    ]
    return AgentRunPerformanceSummary(
        wall_time_ms=sum(item.wall_time_ms for item in performance),
        model_time_ms=sum(item.model_time_ms for item in performance),
        sandbox_time_ms=sum(item.sandbox_time_ms for item in performance),
        verifier_time_ms=sum(item.verifier_time_ms for item in performance),
        prompt_tokens=sum(item.prompt_tokens for item in performance),
        reasoning_tokens=(
            sum(reasoning)
            if performance and len(reasoning) == len(performance)
            else None
        ),
        completion_tokens=sum(item.completion_tokens for item in performance),
        total_tokens=sum(item.total_tokens for item in performance),
        reported_mean_tps=sum(tps) / len(tps) if tps else None,
        reported_tps_trials=len(tps),
        inference_cost_usd=(sum(inference_costs) if inference_costs else None),
        judge_cost_usd=(sum(judge_costs) if judge_costs else None),
        estimated_cost_usd=(
            sum(costs) if performance and len(costs) == len(performance) else None
        ),
    )
