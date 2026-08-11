from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from scripts.acceptance_canary import (
    AcceptanceCanary,
    ApiClient,
    CanaryConfig,
    Reporter,
    main,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
ITEM_IDS = ("arith-03", "arith-04")


class FakeCanaryApi:
    def __init__(self, *, daytona: bool = False, anthropic_judge: bool = False) -> None:
        self.daytona = daytona
        self.anthropic_judge = anthropic_judge
        self.paths: list[tuple[str, str]] = []
        self.login_token: str | None = None
        self.current_session_id = "baseline-session"
        self.run_count = 0
        self.agent_run_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        path = request.url.path
        self.paths.append((method, path))
        payload = _payload(request)
        if method == "GET" and path == "/api/session":
            return self._json(
                request,
                {
                    "auth_required": True,
                    "authenticated": False,
                    "csrf_token": None,
                },
            )
        if method == "POST" and path == "/api/session/login":
            assert isinstance(payload, Mapping)
            self.login_token = payload.get("token")
            return self._json(
                request,
                {
                    "auth_required": True,
                    "authenticated": True,
                    "csrf_token": "csrf-canary",
                },
            )
        if method == "GET" and path == "/api/jobs/active":
            return self._json(request, None)
        if method == "GET" and path == "/api/evaluation/capabilities":
            assert self.anthropic_judge
            return self._json(
                request,
                {
                    "judge_providers": [
                        {
                            "provider": "anthropic",
                            "configured": True,
                            "default_model": "claude-sonnet-5",
                        }
                    ]
                },
            )
        if method == "POST" and path == "/api/evaluation/judge":
            assert self.anthropic_judge
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            config = payload.get("config")
            assert isinstance(config, Mapping)
            assert config.get("provider") == "anthropic"
            assert config.get("model") == "claude-sonnet-5"
            return self._json(
                request,
                {
                    "id": "judge-canary",
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "score": 1,
                    "request_hash": "a" * 64,
                },
            )

        if method == "GET" and path == "/api/models":
            return self._json(
                request,
                [{"id": MODEL_ID, "enabled": True}],
            )
        if method == "POST" and path == "/api/model-sessions":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            if payload.get("profile_id") is None:
                assert payload == {"model_id": MODEL_ID}
                session_id = "baseline-session"
            else:
                assert payload == {
                    "model_id": MODEL_ID,
                    "profile_id": "profile-canary",
                }
                session_id = "masked-session"
            self.current_session_id = session_id
            return self._json(
                request,
                _job("model-load", "model_load", result_id=session_id),
            )
        if method == "GET" and path.startswith("/api/model-sessions/"):
            session_id = path.rsplit("/", maxsplit=1)[-1]
            return self._json(
                request,
                {
                    "id": session_id,
                    "model_id": MODEL_ID,
                    "state": "ready",
                    "profile_id": (
                        "profile-canary" if session_id == "masked-session" else None
                    ),
                },
            )

        if method == "GET" and path == "/api/benchmarks/fixture-arithmetic/items":
            return self._json(
                request,
                {"items": [{"id": item_id} for item_id in ITEM_IDS]},
            )
        if method == "POST" and path == "/api/runs":
            self._assert_csrf(request)
            assert payload == {
                "benchmark_id": "fixture-arithmetic",
                "item_ids": list(ITEM_IDS),
            }
            self.run_count += 1
            run_id = "baseline-run" if self.run_count == 1 else "masked-run"
            return self._json(
                request,
                _job(f"benchmark-{self.run_count}", "benchmark_run", result_id=run_id),
            )
        if method == "GET" and path in {
            "/api/runs/baseline-run",
            "/api/runs/masked-run",
        }:
            run_id = path.rsplit("/", maxsplit=1)[-1]
            session_id = (
                "baseline-session" if run_id == "baseline-run" else "masked-session"
            )
            return self._json(
                request,
                {
                    "id": run_id,
                    "status": "completed",
                    "model_session_id": session_id,
                    "score": 1.0,
                },
            )
        if method == "GET" and path in {
            "/api/runs/baseline-run/items",
            "/api/runs/masked-run/items",
        }:
            return self._json(
                request,
                {
                    "run_id": path.split("/")[-2],
                    "items": [{"item_id": item_id} for item_id in ITEM_IDS],
                    "total": len(ITEM_IDS),
                    "offset": 0,
                    "limit": 250,
                },
            )
        if method == "GET" and path == "/api/runs/baseline-run/routing":
            return self._json(
                request,
                {"run_id": "baseline-run", "total_routed_slots": 128},
            )
        if method == "POST" and path == "/api/routing/explore":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            sources = payload.get("sources")
            assert isinstance(sources, list)
            source = sources[0]
            assert isinstance(source, Mapping)
            if source.get("kind") == "benchmark_run":
                assert payload == {
                    "sources": [
                        {
                            "kind": "benchmark_run",
                            "id": "baseline-run",
                            "weight": 1,
                        }
                    ],
                    "metric": "routing_mass",
                }
                comparison_mass = None
            else:
                assert self.daytona
                assert payload == {
                    "sources": [
                        {
                            "kind": "agent_trial",
                            "id": "trial-baseline",
                            "weight": 1,
                        }
                    ],
                    "comparison_sources": [
                        {
                            "kind": "agent_trial",
                            "id": "trial-candidate",
                            "weight": 1,
                        }
                    ],
                    "metric": "routing_mass",
                }
                comparison_mass = [
                    [0.7, 0.8, 0.5, 0.6, 0.3, 0.4, 0.1, 0.2],
                    [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
                ]
            return self._json(
                request,
                {
                    "fingerprint": "e" * 64,
                    "layer_ids": [0, 1],
                    "num_experts": 8,
                    "top_k": 2,
                    "routing_mass": [
                        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    ],
                    "comparison_routing_mass": comparison_mass,
                    "total_routed_slots": 128,
                },
            )
        if method == "POST" and path == "/api/profiles/validate":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            layers = payload.get("layers")
            assert isinstance(layers, Mapping)
            assert len(layers["0"]["keep"]) == 7
            assert len(layers["1"]["keep"]) == 8
            return self._json(request, {"valid": True})
        if method == "POST" and path == "/api/profiles":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            assert payload.get("selection_strategy") == ("canary_non_uniform_top_mass")
            is_agentic = payload.get("source") == "agentic"
            if is_agentic:
                assert self.daytona
                assert payload.get("source_trial_id") == "trial-baseline"
                profile_id = "profile-agent-canary"
            else:
                assert payload.get("source_run_id") == "baseline-run"
                profile_id = "profile-canary"
            return self._json(
                request,
                {
                    "id": profile_id,
                    "profile": payload["profile"],
                    "source_trial_id": ("trial-baseline" if is_agentic else None),
                    "validation": {"valid": True},
                },
            )
        if method == "POST" and path == "/api/comparisons":
            self._assert_csrf(request)
            return self._json(
                request,
                {
                    "id": "comparison-canary",
                    "profile_id": "profile-canary",
                    "cohort_item_ids": list(ITEM_IDS),
                },
            )

        if self.daytona:
            response = self._daytona(request, payload)
            if response is not None:
                return response
        raise AssertionError(f"unexpected canary request: {method} {path}")

    def _daytona(
        self, request: httpx.Request, payload: object
    ) -> httpx.Response | None:
        method = request.method
        path = request.url.path
        if method == "GET" and path == "/api/sandbox-providers":
            return self._json(
                request,
                [{"id": "daytona", "configured": True}],
            )
        if method == "POST" and path == "/api/sandbox-providers/daytona/preflight":
            self._assert_csrf(request)
            return self._json(
                request,
                {
                    "status": "ready",
                    "reachable": True,
                    "authenticated": True,
                },
            )
        if method == "GET" and path == "/api/agent-task-packs":
            return self._json(
                request,
                [
                    {
                        "id": "smoke-python-v1",
                        "ready": True,
                        "oracle_passed": True,
                        "noop_failed": True,
                    }
                ],
            )
        if method == "GET" and path == "/api/agent-task-packs/smoke-python-v1/tasks":
            return self._json(request, [{"id": "fix-subtract"}])
        if method == "GET" and path == "/api/agents":
            return self._json(
                request,
                [{"id": "bash-json-v1", "is_default": True, "available": True}],
            )
        if method == "POST" and path == "/api/agent-runs":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            assert payload.get("sandbox_provider_id") == "daytona"
            assert payload.get("task_ids") == ["fix-subtract"]
            assert payload.get("budgets") == {
                "max_turns": 10,
                "max_commands": 12,
                "max_tokens": 16384,
                "timeout_seconds": 300,
            }
            self.agent_run_count += 1
            suffix = "baseline" if self.agent_run_count == 1 else "candidate"
            expected_session = (
                "baseline-session" if suffix == "baseline" else "masked-session"
            )
            assert payload.get("model_session_id") == expected_session
            return self._json(
                request,
                _job(
                    f"agent-job-{suffix}",
                    "agent_run",
                    result_id=f"agent-run-{suffix}",
                ),
            )
        if method == "GET" and path in {
            "/api/agent-runs/agent-run-baseline",
            "/api/agent-runs/agent-run-candidate",
        }:
            suffix = path.rsplit("-", maxsplit=1)[-1]
            session_id = (
                "baseline-session" if suffix == "baseline" else "masked-session"
            )
            return self._json(
                request,
                {
                    "id": f"agent-run-{suffix}",
                    "status": "completed",
                    "model_session_id": session_id,
                    "sandbox_provider_id": "daytona",
                    "task_ids": ["fix-subtract"],
                    "total_trials": 1,
                    "passed_trials": 1,
                    "trials": [
                        {
                            "id": f"trial-{suffix}",
                            "status": "passed",
                            "sandbox_status": "deleted",
                            "routed_inference_calls": 2,
                        }
                    ],
                },
            )
        if method == "POST" and path == "/api/agent-comparisons":
            self._assert_csrf(request)
            assert payload == {
                "baseline_run_id": "agent-run-baseline",
                "candidate_run_id": "agent-run-candidate",
                "name": "Acceptance canary agent comparison",
            }
            return self._json(
                request,
                {
                    "id": "c" * 64,
                    "trial_count": 1,
                    "baseline_profile_id": None,
                    "candidate_profile_id": "profile-canary",
                },
            )
        return None

    @staticmethod
    def _assert_csrf(request: httpx.Request) -> None:
        assert request.headers["X-CSRF-Token"] == "csrf-canary"

    @staticmethod
    def _json(
        request: httpx.Request, payload: object, status_code: int = 200
    ) -> httpx.Response:
        return httpx.Response(status_code, json=payload, request=request)


@pytest.mark.parametrize(
    ("daytona", "anthropic_judge"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_public_canary_flow_is_explicit_and_secret_safe(
    daytona: bool,
    anthropic_judge: bool,
) -> None:
    fake = FakeCanaryApi(daytona=daytona, anthropic_judge=anthropic_judge)
    output: list[str] = []
    config = CanaryConfig(
        base_url="https://canary.test",
        token="top-secret-token",
        poll_interval_seconds=0,
        daytona=daytona,
        anthropic_judge=anthropic_judge,
    )
    reporter = Reporter(
        total_steps=9 + (3 if daytona else 0) + (1 if anthropic_judge else 0),
        secrets=(config.token or "",),
        write=output.append,
    )
    with ApiClient(
        config,
        transport=httpx.MockTransport(fake),
        sleep=lambda _: None,
    ) as api:
        result = AcceptanceCanary(config, api, reporter).run()

    assert result["comparison_id"] == "comparison-canary"
    assert ("agent_comparison_id" in result) is daytona
    assert ("agent_profile_id" in result) is daytona
    assert ("judge_result_id" in result) is anthropic_judge
    assert fake.login_token == "top-secret-token"
    assert "top-secret-token" not in "\n".join(output)
    touched_daytona = any("sandbox-providers" in path for _, path in fake.paths)
    assert touched_daytona is daytona
    touched_judge = any("evaluation/judge" in path for _, path in fake.paths)
    assert touched_judge is anthropic_judge


def test_job_conflict_is_adopted_until_terminal_then_submission_retries() -> None:
    posts = 0
    active = _job("blocking-job", "dataset_prepare", status="queued")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST" and request.url.path == "/api/runs":
            posts += 1
            if posts == 1:
                return httpx.Response(
                    409,
                    json={
                        "detail": {
                            "code": "active_job_conflict",
                            "message": "another job is active",
                            "active_job": active,
                            "recovery_url": "/api/jobs/blocking-job",
                        }
                    },
                    request=request,
                )
            return httpx.Response(
                202,
                json=_job("benchmark-job", "benchmark_run", status="queued"),
                request=request,
            )
        if request.url.path == "/api/jobs/blocking-job":
            return httpx.Response(
                200,
                json=_job("blocking-job", "dataset_prepare"),
                request=request,
            )
        raise AssertionError(request.url)

    config = CanaryConfig(
        base_url="https://canary.test",
        poll_interval_seconds=0,
    )
    with ApiClient(
        config,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    ) as api:
        canary = AcceptanceCanary(config, api, Reporter(total_steps=1))
        submitted = canary.submit_job(
            "/api/runs",
            {"benchmark_id": "fixture-arithmetic"},
            expected_kind="benchmark_run",
        )

    assert submitted["id"] == "benchmark-job"
    assert posts == 2


def test_unknown_submission_response_adopts_matching_active_job() -> None:
    active = _job("recoverable-load", "model_load", status="running")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            raise httpx.ReadError("response disconnected", request=request)
        if request.url.path == "/api/jobs/active":
            return httpx.Response(200, json=active, request=request)
        raise AssertionError(request.url)

    config = CanaryConfig(base_url="https://canary.test")
    with ApiClient(
        config,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    ) as api:
        canary = AcceptanceCanary(config, api, Reporter(total_steps=1))
        submitted = canary.submit_job(
            "/api/model-sessions",
            {"model_id": MODEL_ID},
            expected_kind="model_load",
        )

    assert submitted == active


def test_cli_fails_closed_without_an_explicit_or_environment_base_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("MOE_TOOLS_CANARY_BASE_URL", raising=False)

    exit_code = main([])

    assert exit_code == 2
    assert "supply --base-url" in capsys.readouterr().err


def _payload(request: httpx.Request) -> object:
    if not request.content:
        return None
    return json.loads(request.content)


def _job(
    job_id: str,
    kind: str,
    *,
    status: str = "completed",
    result_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": kind,
        "status": status,
        "progress_current": 1 if status == "completed" else 0,
        "progress_total": 1,
        "result_id": result_id,
        "error": None,
    }
