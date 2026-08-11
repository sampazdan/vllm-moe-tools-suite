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
    def __init__(self, *, daytona: bool = False) -> None:
        self.daytona = daytona
        self.paths: list[tuple[str, str]] = []
        self.login_token: str | None = None
        self.current_session_id = "baseline-session"
        self.run_count = 0

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
                    "items": [{"item_id": item_id} for item_id in ITEM_IDS],
                },
            )
        if method == "GET" and path == "/api/runs/baseline-run/routing":
            return self._json(
                request,
                {"run_id": "baseline-run", "total_routed_slots": 128},
            )
        if method == "POST" and path == "/api/profiles/propose":
            self._assert_csrf(request)
            return self._json(
                request,
                {
                    "profile": {
                        "version": 1,
                        "layers": {"0": {"keep": list(range(8))}},
                    },
                    "validation": {"valid": True},
                    "observed_mass_retained": 0.9,
                },
            )
        if method == "POST" and path == "/api/profiles":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            assert payload.get("source_run_id") == "baseline-run"
            return self._json(request, {"id": "profile-canary"})
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
                "max_turns": 4,
                "max_commands": 6,
                "max_tokens": 4096,
                "timeout_seconds": 180,
            }
            return self._json(
                request,
                _job("agent-job", "agent_run", result_id="agent-run"),
            )
        if method == "GET" and path == "/api/agent-runs/agent-run":
            return self._json(
                request,
                {
                    "id": "agent-run",
                    "status": "completed",
                    "model_session_id": "masked-session",
                    "sandbox_provider_id": "daytona",
                    "task_ids": ["fix-subtract"],
                    "total_trials": 1,
                    "passed_trials": 1,
                    "trials": [
                        {
                            "status": "passed",
                            "sandbox_status": "deleted",
                            "routed_inference_calls": 2,
                        }
                    ],
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


@pytest.mark.parametrize("daytona", [False, True])
def test_public_canary_flow_is_explicit_and_secret_safe(daytona: bool) -> None:
    fake = FakeCanaryApi(daytona=daytona)
    output: list[str] = []
    config = CanaryConfig(
        base_url="https://canary.test",
        token="top-secret-token",
        poll_interval_seconds=0,
        daytona=daytona,
    )
    reporter = Reporter(
        total_steps=10 if daytona else 9,
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
    assert ("agent_run_id" in result) is daytona
    assert fake.login_token == "top-secret-token"
    assert "top-secret-token" not in "\n".join(output)
    touched_daytona = any("sandbox-providers" in path for _, path in fake.paths)
    assert touched_daytona is daytona


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
