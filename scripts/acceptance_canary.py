"""Run the public-API release canary against a deployed MoE Tools appliance.

The default path performs deterministic model, benchmark, expert-explorer,
non-uniform profile, and comparison checks. Daytona and Anthropic checks are
explicit opt-ins because they can create paid external work.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx

ACTIVE_JOB_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_ITEM_IDS = ("arith-03", "arith-04")


class CanaryFailure(RuntimeError):
    """A release-gating assertion or API operation failed."""


class ApiFailure(CanaryFailure):
    """The appliance returned an unsuccessful API response."""

    def __init__(self, status_code: int, detail: object) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(_error_message(status_code, detail))

    def active_job(self) -> dict[str, Any] | None:
        if self.status_code != 409 or not isinstance(self.detail, Mapping):
            return None
        if self.detail.get("code") != "active_job_conflict":
            return None
        active = self.detail.get("active_job")
        return dict(active) if isinstance(active, Mapping) else None


@dataclass(frozen=True)
class CanaryConfig:
    base_url: str
    token: str | None = field(default=None, repr=False)
    item_ids: tuple[str, ...] = DEFAULT_ITEM_IDS
    model_id: str | None = None
    keep_per_layer: int = 64
    job_timeout_seconds: float = 2400
    poll_interval_seconds: float = 2
    request_timeout_seconds: float = 30
    anthropic_judge: bool = False
    anthropic_judge_model: str = "claude-sonnet-5"
    daytona: bool = False
    daytona_provider_id: str = "daytona"
    daytona_pack_id: str = "smoke-python-v1"
    daytona_task_id: str = "fix-subtract"


@dataclass(frozen=True)
class AgentCanaryRun:
    run_id: str
    trial_id: str


class Reporter:
    """Emit concise progress while redacting configured secrets."""

    def __init__(
        self,
        *,
        total_steps: int,
        secrets: Sequence[str] = (),
        write: Callable[[str], None] = print,
    ) -> None:
        self.total_steps = total_steps
        self._secrets = tuple(secret for secret in secrets if secret)
        self._write = write
        self._current = 0

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        self._current += 1
        prefix = f"[{self._current:02d}/{self.total_steps:02d}]"
        self._emit(f"{prefix} {label} ...")
        try:
            yield
        except BaseException as error:
            self._emit(f"{prefix} FAIL: {error}")
            raise
        self._emit(f"{prefix} OK")

    def info(self, message: str) -> None:
        self._emit(f"         {message}")

    def _emit(self, message: str) -> None:
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        self._write(redacted)


class ApiClient:
    """Authenticated JSON client with bounded transient GET retries."""

    def __init__(
        self,
        config: CanaryConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self.monotonic = monotonic
        self.csrf_token: str | None = None
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            follow_redirects=False,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, path: str) -> Any:
        return self._request("GET", path, retry_transient=True)

    def post(self, path: str, payload: object | None = None) -> Any:
        return self._request("POST", path, payload=payload)

    def patch(self, path: str, payload: object) -> Any:
        return self._request("PATCH", path, payload=payload)

    def login(self) -> dict[str, Any]:
        session = _mapping(self.get("/api/session"), "session status")
        if not _boolean(session, "auth_required"):
            return session
        if not self.config.token:
            raise CanaryFailure(
                "authentication is required; supply --token or MOE_TOOLS_CANARY_TOKEN"
            )
        authenticated = _mapping(
            self._request(
                "POST",
                "/api/session/login",
                payload={"token": self.config.token},
                include_csrf=False,
            ),
            "login response",
        )
        if not _boolean(authenticated, "authenticated"):
            raise CanaryFailure("login response did not confirm authentication")
        csrf_token = authenticated.get("csrf_token")
        if not isinstance(csrf_token, str) or not csrf_token:
            raise CanaryFailure("login response omitted the CSRF token")
        self.csrf_token = csrf_token
        return authenticated

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: object | None = None,
        include_csrf: bool = True,
        retry_transient: bool = False,
    ) -> Any:
        attempts = 4 if retry_transient else 1
        for attempt in range(attempts):
            headers: dict[str, str] = {}
            if method != "GET":
                headers["Content-Type"] = "application/json"
                if include_csrf and self.csrf_token:
                    headers["X-CSRF-Token"] = self.csrf_token
            try:
                response = self._client.request(
                    method,
                    path,
                    json=payload if method != "GET" else None,
                    headers=headers,
                )
            except httpx.RequestError:
                if attempt + 1 == attempts:
                    raise
                self._sleep(min(2**attempt, 5))
                continue
            body = _response_body(response)
            if response.is_success:
                return body
            if (
                retry_transient
                and _transient_status(response.status_code)
                and attempt + 1 < attempts
            ):
                self._sleep(min(2**attempt, 5))
                continue
            detail = body.get("detail") if isinstance(body, Mapping) else body
            raise ApiFailure(response.status_code, detail)
        raise AssertionError("request retry loop exhausted")


class AcceptanceCanary:
    """Drive the public release-gate workflow and validate every artifact."""

    def __init__(
        self,
        config: CanaryConfig,
        api: ApiClient,
        reporter: Reporter,
    ) -> None:
        self.config = config
        self.api = api
        self.reporter = reporter

    def run(self) -> dict[str, str]:
        baseline_agent_run: AgentCanaryRun | None = None
        judge_result_id: str | None = None
        with self.reporter.step("Authenticate public API session"):
            session = self.api.login()
            auth_mode = "required" if session.get("auth_required") else "disabled"
            self.reporter.info(f"Authentication mode: {auth_mode}")

        with self.reporter.step("Reconcile any observable active job"):
            active = self._active_job()
            if active is None:
                self.reporter.info("No active job found")
            else:
                self.reporter.info(
                    f"Adopting active {_job_kind(active)} job {_short_id(active)}"
                )
                self.wait_for_job(active)

        if self.config.anthropic_judge:
            with self.reporter.step("Exercise the opt-in Anthropic judge protocol"):
                judge_result_id = self._run_anthropic_judge()

        with self.reporter.step("Load an unprofiled baseline model"):
            model_id = self._select_model()
            baseline_load = self.submit_job(
                "/api/model-sessions",
                {"model_id": model_id},
                expected_kind="model_load",
            )
            baseline_load = self.wait_for_job(baseline_load)
            baseline_session = self._model_session_from_job(baseline_load)
            if baseline_session.get("model_id") != model_id:
                raise CanaryFailure("baseline session loaded an unexpected model")
            if baseline_session.get("profile_id") is not None:
                raise CanaryFailure(
                    "baseline model session unexpectedly uses a profile"
                )
            baseline_session_id = _string(baseline_session, "id")
            self.reporter.info(f"Baseline session {baseline_session_id[:8]} is ready")

        with self.reporter.step("Run the explicit fixture cohort"):
            self._validate_fixture_selection()
            baseline_run = self._run_fixture(baseline_session_id)
            baseline_run_id = _string(baseline_run, "id")
            self.reporter.info(
                f"Baseline run {baseline_run_id[:8]} scored {baseline_run.get('score')}"
            )

        with self.reporter.step("Explore routing and build a non-uniform profile"):
            routing = _mapping(
                self.api.get(f"/api/runs/{baseline_run_id}/routing"),
                "routing summary",
            )
            if _integer(routing, "total_routed_slots") <= 0:
                raise CanaryFailure("baseline run captured no routed expert slots")
            explorer = _mapping(
                self.api.post(
                    "/api/routing/explore",
                    {
                        "sources": [
                            {
                                "kind": "benchmark_run",
                                "id": baseline_run_id,
                                "weight": 1,
                            }
                        ],
                        "metric": "routing_mass",
                    },
                ),
                "routing explorer",
            )
            if _integer(explorer, "total_routed_slots") <= 0:
                raise CanaryFailure("routing explorer captured no routed expert slots")
            custom_profile, selection_config = _custom_profile_from_explorer(
                explorer,
                keep_per_layer=self.config.keep_per_layer,
            )
            validation = _mapping(
                self.api.post("/api/profiles/validate", custom_profile),
                "profile validation",
            )
            if not _boolean(validation, "valid"):
                raise CanaryFailure("custom profile failed topology validation")

        with self.reporter.step("Save the immutable custom profile"):
            saved_profile = _mapping(
                self.api.post(
                    "/api/profiles",
                    {
                        "name": _profile_name(),
                        "description": (
                            "Automated non-uniform acceptance profile from the "
                            "explicitly selected fixture cohort."
                        ),
                        "model_id": model_id,
                        "profile": custom_profile,
                        "source": "manual",
                        "source_run_id": baseline_run_id,
                        "source_refs": [
                            {
                                "kind": "benchmark_run",
                                "id": baseline_run_id,
                                "weight": 1,
                            }
                        ],
                        "metric": "routing_mass",
                        "selection_strategy": "canary_non_uniform_top_mass",
                        "selection_config": selection_config,
                    },
                ),
                "saved profile",
            )
            profile_id = _string(saved_profile, "id")
            saved_validation = _mapping(
                saved_profile.get("validation"), "saved profile validation"
            )
            if not _boolean(saved_validation, "valid"):
                raise CanaryFailure("saved custom profile is invalid")
            saved_layer_sizes = {
                len(_list(layer.get("keep"), "saved expert selection"))
                for layer in _mapping(
                    _mapping(saved_profile.get("profile"), "saved profile").get(
                        "layers"
                    ),
                    "saved profile layers",
                ).values()
                if isinstance(layer, Mapping)
            }
            if len(saved_layer_sizes) < 2:
                raise CanaryFailure("saved profile is not non-uniform across layers")
            self.reporter.info(f"Saved profile {profile_id[:8]}")

        if self.config.daytona:
            with self.reporter.step("Run the baseline Daytona coding canary"):
                baseline_agent_run = self._run_daytona(baseline_session_id)

        with self.reporter.step("Reload using profile_id only"):
            masked_load = self.submit_job(
                "/api/model-sessions",
                {"model_id": model_id, "profile_id": profile_id},
                expected_kind="model_load",
            )
            masked_session = self._model_session_from_job(
                self.wait_for_job(masked_load)
            )
            if masked_session.get("model_id") != model_id:
                raise CanaryFailure("masked session loaded an unexpected model")
            if masked_session.get("profile_id") != profile_id:
                raise CanaryFailure(
                    "masked session did not retain saved profile identity"
                )
            masked_session_id = _string(masked_session, "id")

        with self.reporter.step("Rerun the identical paired cohort"):
            masked_run = self._run_fixture(masked_session_id)
            masked_run_id = _string(masked_run, "id")
            self.reporter.info(
                f"Masked run {masked_run_id[:8]} scored {masked_run.get('score')}"
            )

        with self.reporter.step("Create and validate the paired comparison"):
            comparison = _mapping(
                self.api.post(
                    "/api/comparisons",
                    {
                        "name": "Acceptance canary paired comparison",
                        "baseline_run_id": baseline_run_id,
                        "candidate_run_id": masked_run_id,
                    },
                ),
                "comparison",
            )
            if comparison.get("profile_id") != profile_id:
                raise CanaryFailure("comparison lost its candidate profile provenance")
            if comparison.get("cohort_item_ids") != list(self.config.item_ids):
                raise CanaryFailure("comparison did not preserve the paired cohort")
            comparison_id = _string(comparison, "id")
            self.reporter.info(f"Comparison {comparison_id[:8]} is reproducible")

        result = {
            "model_id": model_id,
            "baseline_run_id": baseline_run_id,
            "profile_id": profile_id,
            "masked_run_id": masked_run_id,
            "comparison_id": comparison_id,
        }
        if judge_result_id is not None:
            result["judge_result_id"] = judge_result_id
        if self.config.daytona:
            if baseline_agent_run is None:
                raise AssertionError("baseline Daytona run was not recorded")
            with self.reporter.step("Run the masked Daytona coding canary"):
                candidate_agent_run = self._run_daytona(masked_session_id)
            result["baseline_agent_run_id"] = baseline_agent_run.run_id
            result["candidate_agent_run_id"] = candidate_agent_run.run_id
            with self.reporter.step(
                "Compare agent runs and save an agent-derived custom profile"
            ):
                result.update(
                    self._analyze_daytona_pair(
                        baseline=baseline_agent_run,
                        candidate=candidate_agent_run,
                        model_id=model_id,
                        candidate_profile_id=profile_id,
                    )
                )
        return result

    def _run_anthropic_judge(self) -> str:
        capabilities = _mapping(
            self.api.get("/api/evaluation/capabilities"),
            "evaluation capabilities",
        )
        providers = _list(
            capabilities.get("judge_providers"), "judge provider capabilities"
        )
        anthropic = next(
            (
                provider
                for provider in providers
                if isinstance(provider, Mapping)
                and provider.get("provider") == "anthropic"
            ),
            None,
        )
        if anthropic is None or anthropic.get("configured") is not True:
            raise CanaryFailure("Anthropic judge is not configured")
        response = _mapping(
            self.api.post(
                "/api/evaluation/judge",
                {
                    "config": {
                        "provider": "anthropic",
                        "model": self.config.anthropic_judge_model,
                        "mode": "single",
                        "rubric": (
                            "Score 1 only when the candidate states that two plus two "
                            "equals four; otherwise score 0. Return the required JSON."
                        ),
                        "pass_threshold": 0.9,
                        "repetitions": 1,
                        "max_output_tokens": 256,
                    },
                    "task": "State the result of adding two and two.",
                    "candidate": "Two plus two equals four.",
                    "criteria": ["The arithmetic statement is correct."],
                },
            ),
            "Anthropic judge result",
        )
        if response.get("provider") != "anthropic":
            raise CanaryFailure("judge response used an unexpected provider")
        if response.get("model") != self.config.anthropic_judge_model:
            raise CanaryFailure("judge response used an unexpected model")
        score = response.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise CanaryFailure("judge response omitted a numeric score")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise CanaryFailure("judge response score is outside [0, 1]")
        request_hash = response.get("request_hash")
        if not isinstance(request_hash, str) or len(request_hash) != 64:
            raise CanaryFailure("judge response omitted a reproducible request hash")
        result_id = _string(response, "id")
        self.reporter.info(f"Anthropic judge result {result_id[:8]} is protocol-valid")
        return result_id

    def submit_job(
        self,
        path: str,
        payload: object,
        *,
        expected_kind: str,
    ) -> dict[str, Any]:
        for _ in range(4):
            try:
                submitted = _mapping(self.api.post(path, payload), "submitted job")
            except ApiFailure as error:
                active = error.active_job()
                if active is None:
                    raise
                self.reporter.info(
                    f"Conflict with {_job_kind(active)} job {_short_id(active)}; "
                    "adopting it until terminal before retry"
                )
                self.wait_for_job(active)
                continue
            except httpx.RequestError as error:
                active = self._active_job_after_unknown_submission(error)
                if active is not None and _job_kind(active) == expected_kind:
                    self.reporter.info(
                        f"Submission response was interrupted; adopted active "
                        f"{expected_kind} job {_short_id(active)}"
                    )
                    return active
                raise CanaryFailure(
                    "submission outcome is unknown and no matching active job "
                    "could be recovered"
                ) from error
            if _job_kind(submitted) != expected_kind:
                raise CanaryFailure(
                    f"submission returned {_job_kind(submitted)!r}, expected "
                    f"{expected_kind!r}"
                )
            return submitted
        raise CanaryFailure("active job conflicts did not clear after four attempts")

    def wait_for_job(self, submitted: Mapping[str, Any]) -> dict[str, Any]:
        job = dict(submitted)
        job_id = _string(job, "id")
        deadline = self.api.monotonic() + self.config.job_timeout_seconds
        last_progress: tuple[object, object, object] | None = None
        while _job_status(job) in ACTIVE_JOB_STATUSES:
            progress = (
                _job_status(job),
                job.get("progress_current"),
                job.get("progress_total"),
            )
            if progress != last_progress:
                self.reporter.info(
                    f"Job {job_id[:8]}: {progress[0]} {progress[1]}/{progress[2]}"
                )
                last_progress = progress
            if self.api.monotonic() >= deadline:
                raise CanaryFailure(
                    f"job {job_id} exceeded {self.config.job_timeout_seconds:g}s"
                )
            self.api._sleep(self.config.poll_interval_seconds)
            job = _mapping(self.api.get(f"/api/jobs/{job_id}"), "job status")
        status = _job_status(job)
        if status not in TERMINAL_JOB_STATUSES:
            raise CanaryFailure(f"job {job_id} returned unknown status {status!r}")
        if status != "completed":
            error = job.get("error")
            suffix = f": {error}" if isinstance(error, str) and error else ""
            raise CanaryFailure(f"job {job_id} ended {status}{suffix}")
        return job

    def _active_job(self) -> dict[str, Any] | None:
        active = self.api.get("/api/jobs/active")
        if active is None:
            return None
        return _mapping(active, "active job")

    def _active_job_after_unknown_submission(
        self, error: httpx.RequestError
    ) -> dict[str, Any] | None:
        del error
        try:
            return self._active_job()
        except (CanaryFailure, httpx.RequestError):
            return None

    def _select_model(self) -> str:
        models = _list(self.api.get("/api/models"), "model registry")
        if self.config.model_id:
            selected = next(
                (
                    model
                    for model in models
                    if isinstance(model, Mapping)
                    and model.get("id") == self.config.model_id
                    and model.get("enabled") is True
                ),
                None,
            )
            if selected is None:
                raise CanaryFailure(
                    f"requested model {self.config.model_id!r} is not enabled"
                )
            return self.config.model_id
        enabled = [
            model
            for model in models
            if isinstance(model, Mapping) and model.get("enabled") is True
        ]
        if len(enabled) != 1:
            raise CanaryFailure(
                "select --model-id when the registry does not expose exactly one "
                "enabled model"
            )
        return _string(enabled[0], "id")

    def _model_session_from_job(
        self, completed_job: Mapping[str, Any]
    ) -> dict[str, Any]:
        session_id = completed_job.get("result_id")
        if not isinstance(session_id, str) or not session_id:
            raise CanaryFailure("completed model-load job omitted result_id")
        session = _mapping(
            self.api.get(f"/api/model-sessions/{session_id}"),
            "model session",
        )
        if session.get("state") != "ready":
            raise CanaryFailure("model session is not ready after completed load")
        return session

    def _validate_fixture_selection(self) -> None:
        page = _mapping(
            self.api.get("/api/benchmarks/fixture-arithmetic/items"),
            "fixture item page",
        )
        items = _list(page.get("items"), "fixture items")
        available = {
            item.get("id")
            for item in items
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        missing = set(self.config.item_ids) - available
        if missing:
            raise CanaryFailure(f"fixture item IDs are unavailable: {sorted(missing)}")

    def _run_fixture(self, model_session_id: str) -> dict[str, Any]:
        job = self.submit_job(
            "/api/runs",
            {
                "benchmark_id": "fixture-arithmetic",
                "item_ids": list(self.config.item_ids),
            },
            expected_kind="benchmark_run",
        )
        completed = self.wait_for_job(job)
        run_id = completed.get("result_id")
        if not isinstance(run_id, str) or not run_id:
            raise CanaryFailure("completed benchmark job omitted result_id")
        run = _mapping(self.api.get(f"/api/runs/{run_id}"), "benchmark run")
        if run.get("status") != "completed":
            raise CanaryFailure("benchmark run did not complete")
        if run.get("model_session_id") != model_session_id:
            raise CanaryFailure("benchmark run used an unexpected model session")
        result_page = _mapping(
            self.api.get(f"/api/runs/{run_id}/items?limit=250"),
            "benchmark result page",
        )
        item_ids = [
            item.get("item_id")
            for item in _list(result_page.get("items"), "benchmark results")
            if isinstance(item, Mapping)
        ]
        if item_ids != list(self.config.item_ids):
            raise CanaryFailure("benchmark results changed the explicit cohort order")
        if run.get("score") != 1.0:
            raise CanaryFailure(
                f"fixture cohort score was {run.get('score')!r}, expected 1.0"
            )
        return run

    def _run_daytona(self, model_session_id: str) -> AgentCanaryRun:
        providers = _list(self.api.get("/api/sandbox-providers"), "sandbox providers")
        provider = next(
            (
                item
                for item in providers
                if isinstance(item, Mapping)
                and item.get("id") == self.config.daytona_provider_id
            ),
            None,
        )
        if provider is None or provider.get("configured") is not True:
            raise CanaryFailure("Daytona provider is not configured")
        preflight = _mapping(
            self.api.post(
                f"/api/sandbox-providers/{self.config.daytona_provider_id}/preflight"
            ),
            "Daytona preflight",
        )
        if not (
            preflight.get("status") == "ready"
            and preflight.get("reachable") is True
            and preflight.get("authenticated") is True
        ):
            raise CanaryFailure("Daytona preflight did not reach authenticated/ready")

        packs = _list(self.api.get("/api/agent-task-packs"), "agent task packs")
        pack = next(
            (
                item
                for item in packs
                if isinstance(item, Mapping)
                and item.get("id") == self.config.daytona_pack_id
            ),
            None,
        )
        if pack is None or not all(
            pack.get(field_name) is True
            for field_name in ("ready", "oracle_passed", "noop_failed")
        ):
            raise CanaryFailure("Daytona canary task pack is not release-eligible")
        tasks = _list(
            self.api.get(f"/api/agent-task-packs/{self.config.daytona_pack_id}/tasks"),
            "agent tasks",
        )
        if not any(
            isinstance(task, Mapping) and task.get("id") == self.config.daytona_task_id
            for task in tasks
        ):
            raise CanaryFailure("configured Daytona canary task is unavailable")
        agents = _list(self.api.get("/api/agents"), "agents")
        agent = next(
            (
                item
                for item in agents
                if isinstance(item, Mapping)
                and item.get("is_default") is True
                and item.get("available") is True
            ),
            None,
        )
        if agent is None:
            raise CanaryFailure("no available default coding agent is configured")

        self.reporter.info(
            "Daytona enabled explicitly: creating one small remote coding sandbox"
        )
        job = self.submit_job(
            "/api/agent-runs",
            {
                "task_pack_id": self.config.daytona_pack_id,
                "task_ids": [self.config.daytona_task_id],
                "agent_id": _string(agent, "id"),
                "sandbox_provider_id": self.config.daytona_provider_id,
                "model_session_id": model_session_id,
                "attempts": 1,
                "seed": 0,
                "generation": {
                    "temperature": 0,
                    "max_tokens": 1024,
                    "seed": 0,
                    "enable_thinking": False,
                },
                "budgets": {
                    "max_turns": 10,
                    "max_commands": 12,
                    "max_tokens": 16384,
                    "timeout_seconds": 300,
                },
            },
            expected_kind="agent_run",
        )
        completed = self.wait_for_job(job)
        run_id = completed.get("result_id")
        if not isinstance(run_id, str) or not run_id:
            raise CanaryFailure("completed agent job omitted result_id")
        run = _mapping(self.api.get(f"/api/agent-runs/{run_id}"), "agent run")
        if run.get("status") != "completed":
            raise CanaryFailure(f"agent run ended {run.get('status')!r}")
        if run.get("model_session_id") != model_session_id:
            raise CanaryFailure("Daytona canary used an unexpected model session")
        if run.get("sandbox_provider_id") != self.config.daytona_provider_id:
            raise CanaryFailure("Daytona canary used an unexpected provider")
        if run.get("task_ids") != [self.config.daytona_task_id]:
            raise CanaryFailure("Daytona canary used an unexpected task cohort")
        if run.get("total_trials") != 1 or run.get("passed_trials") != 1:
            raise CanaryFailure("Daytona coding task did not pass its verifier")
        trials = _list(run.get("trials"), "Daytona trial summaries")
        if len(trials) != 1 or not isinstance(trials[0], Mapping):
            raise CanaryFailure("Daytona canary did not return exactly one trial")
        trial = trials[0]
        if trial.get("status") != "passed":
            raise CanaryFailure("Daytona trial did not reach passed state")
        if trial.get("sandbox_status") != "deleted":
            raise CanaryFailure(
                "Daytona sandbox deletion was not confirmed after the trial"
            )
        routed_calls = trial.get("routed_inference_calls")
        if type(routed_calls) is not int or routed_calls <= 0:
            raise CanaryFailure(
                "Daytona trial captured no routed-expert inference telemetry"
            )
        trial_id = trial.get("id")
        if not isinstance(trial_id, str) or not trial_id:
            raise CanaryFailure("Daytona trial summary omitted its trial ID")
        self.reporter.info(f"Daytona agent run {run_id[:8]} passed")
        return AgentCanaryRun(run_id=run_id, trial_id=trial_id)

    def _analyze_daytona_pair(
        self,
        *,
        baseline: AgentCanaryRun,
        candidate: AgentCanaryRun,
        model_id: str,
        candidate_profile_id: str,
    ) -> dict[str, str]:
        comparison = _mapping(
            self.api.post(
                "/api/agent-comparisons",
                {
                    "baseline_run_id": baseline.run_id,
                    "candidate_run_id": candidate.run_id,
                    "name": "Acceptance canary agent comparison",
                },
            ),
            "agent comparison",
        )
        if _integer(comparison, "trial_count") != 1:
            raise CanaryFailure("agent comparison did not preserve the one-task cohort")
        if comparison.get("baseline_profile_id") is not None:
            raise CanaryFailure(
                "baseline agent run unexpectedly used an expert profile"
            )
        if comparison.get("candidate_profile_id") != candidate_profile_id:
            raise CanaryFailure("agent comparison lost candidate profile provenance")
        comparison_id = _string(comparison, "id")

        explorer = _mapping(
            self.api.post(
                "/api/routing/explore",
                {
                    "sources": [
                        {
                            "kind": "agent_trial",
                            "id": baseline.trial_id,
                            "weight": 1,
                        }
                    ],
                    "comparison_sources": [
                        {
                            "kind": "agent_trial",
                            "id": candidate.trial_id,
                            "weight": 1,
                        }
                    ],
                    "metric": "routing_mass",
                },
            ),
            "agent routing explorer",
        )
        comparison_matrix = _list(
            explorer.get("comparison_routing_mass"),
            "agent comparison routing matrix",
        )
        if not comparison_matrix:
            raise CanaryFailure("agent routing comparison returned an empty matrix")
        profile, selection_config = _custom_profile_from_explorer(
            explorer,
            keep_per_layer=self.config.keep_per_layer,
        )
        validation = _mapping(
            self.api.post("/api/profiles/validate", profile),
            "agent-derived profile validation",
        )
        if not _boolean(validation, "valid"):
            raise CanaryFailure("agent-derived custom profile is invalid")
        saved = _mapping(
            self.api.post(
                "/api/profiles",
                {
                    "name": f"{_profile_name()} · agent-derived",
                    "description": (
                        "Automated non-uniform profile derived from baseline agent "
                        "routing telemetry."
                    ),
                    "model_id": model_id,
                    "profile": profile,
                    "source": "agentic",
                    "source_trial_id": baseline.trial_id,
                    "source_refs": [
                        {
                            "kind": "agent_trial",
                            "id": baseline.trial_id,
                            "weight": 1,
                        }
                    ],
                    "metric": "routing_mass",
                    "selection_strategy": "canary_non_uniform_top_mass",
                    "selection_config": selection_config,
                },
            ),
            "agent-derived saved profile",
        )
        if saved.get("source_trial_id") != baseline.trial_id:
            raise CanaryFailure("agent-derived profile lost trial provenance")
        saved_validation = _mapping(
            saved.get("validation"), "agent-derived saved profile validation"
        )
        if not _boolean(saved_validation, "valid"):
            raise CanaryFailure("saved agent-derived profile is invalid")
        profile_id = _string(saved, "id")
        self.reporter.info(
            f"Agent comparison {comparison_id[:8]} and profile "
            f"{profile_id[:8]} are valid"
        )
        return {
            "agent_comparison_id": comparison_id,
            "agent_profile_id": profile_id,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MoE Tools release canary. Daytona is disabled unless "
            "--daytona is supplied explicitly."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MOE_TOOLS_CANARY_BASE_URL"),
        help="Appliance origin (or MOE_TOOLS_CANARY_BASE_URL).",
    )
    parser.add_argument(
        "--token",
        default=(
            os.getenv("MOE_TOOLS_CANARY_TOKEN") or os.getenv("MOE_TOOLS_AUTH_TOKEN")
        ),
        help=(
            "Access token (or MOE_TOOLS_CANARY_TOKEN/MOE_TOOLS_AUTH_TOKEN). "
            "The value is never printed."
        ),
    )
    parser.add_argument("--model-id")
    parser.add_argument(
        "--item-id",
        action="append",
        dest="item_ids",
        help=(
            "Explicit fixture item ID; repeat to select a cohort. Defaults to "
            f"{', '.join(DEFAULT_ITEM_IDS)}."
        ),
    )
    parser.add_argument("--keep-per-layer", type=int, default=64)
    parser.add_argument("--job-timeout", type=float, default=2400)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument(
        "--anthropic-judge",
        action="store_true",
        help="Opt in to one small paid Anthropic judge protocol canary.",
    )
    parser.add_argument("--anthropic-judge-model", default="claude-sonnet-5")
    parser.add_argument(
        "--daytona",
        action="store_true",
        help="Opt in to one small paid remote coding-sandbox canary.",
    )
    parser.add_argument("--daytona-provider-id", default="daytona")
    parser.add_argument("--daytona-pack-id", default="smoke-python-v1")
    parser.add_argument("--daytona-task-id", default="fix-subtract")
    return parser


def config_from_args(arguments: argparse.Namespace) -> CanaryConfig:
    if not arguments.base_url:
        raise CanaryFailure("supply --base-url or set MOE_TOOLS_CANARY_BASE_URL")
    base_url = _normalize_base_url(arguments.base_url)
    item_ids = tuple(arguments.item_ids or DEFAULT_ITEM_IDS)
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise CanaryFailure("fixture item IDs must be non-empty and unique")
    if arguments.keep_per_layer <= 0:
        raise CanaryFailure("--keep-per-layer must be positive")
    if arguments.job_timeout <= 0:
        raise CanaryFailure("--job-timeout must be positive")
    if arguments.poll_interval < 0:
        raise CanaryFailure("--poll-interval cannot be negative")
    if arguments.request_timeout <= 0:
        raise CanaryFailure("--request-timeout must be positive")
    if not arguments.anthropic_judge_model.strip():
        raise CanaryFailure("--anthropic-judge-model cannot be blank")
    return CanaryConfig(
        base_url=base_url,
        token=arguments.token,
        item_ids=item_ids,
        model_id=arguments.model_id,
        keep_per_layer=arguments.keep_per_layer,
        job_timeout_seconds=arguments.job_timeout,
        poll_interval_seconds=arguments.poll_interval,
        request_timeout_seconds=arguments.request_timeout,
        anthropic_judge=arguments.anthropic_judge,
        anthropic_judge_model=arguments.anthropic_judge_model,
        daytona=arguments.daytona,
        daytona_provider_id=arguments.daytona_provider_id,
        daytona_pack_id=arguments.daytona_pack_id,
        daytona_task_id=arguments.daytona_task_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        config = config_from_args(arguments)
    except CanaryFailure as error:
        print(f"Acceptance canary configuration failed: {error}", file=sys.stderr)
        return 2
    reporter = Reporter(
        total_steps=(
            9 + (3 if config.daytona else 0) + (1 if config.anthropic_judge else 0)
        ),
        secrets=(config.token or "",),
    )
    try:
        with ApiClient(config) as api:
            result = AcceptanceCanary(config, api, reporter).run()
    except (CanaryFailure, httpx.HTTPError, ValueError, TypeError) as error:
        reporter.info(f"Acceptance canary FAILED: {error}")
        return 1
    reporter.info(f"Acceptance canary PASSED: comparison {result['comparison_id'][:8]}")
    return 0


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CanaryFailure("base URL must be an http(s) origin")
    if parsed.username is not None or parsed.password is not None:
        raise CanaryFailure("base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise CanaryFailure("base URL must not contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise CanaryFailure("base URL must be an origin without an API path")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _profile_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Acceptance canary · {timestamp} · {uuid4().hex[:8]}"


def _custom_profile_from_explorer(
    explorer: Mapping[str, Any],
    *,
    keep_per_layer: int,
) -> tuple[dict[str, object], dict[str, object]]:
    layer_ids = _list(explorer.get("layer_ids"), "routing explorer layer IDs")
    matrix = _list(explorer.get("routing_mass"), "routing explorer mass matrix")
    num_experts = _integer(explorer, "num_experts")
    top_k = _integer(explorer, "top_k")
    if len(layer_ids) != len(matrix) or not layer_ids:
        raise CanaryFailure("routing explorer returned an inconsistent layer matrix")
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
        raise CanaryFailure("routing explorer returned invalid expert topology")
    default_keep = min(keep_per_layer, num_experts)
    if default_keep < top_k:
        raise CanaryFailure(
            f"--keep-per-layer must retain at least the model top-k ({top_k})"
        )
    if default_keep > top_k:
        custom_keep = default_keep - 1
    elif default_keep < num_experts:
        custom_keep = default_keep + 1
    else:
        raise CanaryFailure(
            "model topology cannot express a non-uniform canary profile"
        )

    profile_layers: dict[str, dict[str, list[int]]] = {}
    layer_rows = zip(layer_ids, matrix, strict=True)
    for index, (raw_layer_id, raw_values) in enumerate(layer_rows):
        if type(raw_layer_id) is not int:
            raise CanaryFailure("routing explorer layer IDs must be integers")
        values = _list(raw_values, f"routing mass for layer {raw_layer_id}")
        if len(values) != num_experts:
            raise CanaryFailure(
                f"routing explorer layer {raw_layer_id} has the wrong expert count"
            )
        numeric: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise CanaryFailure("routing mass values must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise CanaryFailure(
                    "routing mass values must be finite and non-negative"
                )
            numeric.append(normalized)
        layer_keep = custom_keep if index == 0 else default_keep
        ranked = sorted(
            range(num_experts),
            key=lambda expert: (-numeric[expert], expert),
        )
        profile_layers[str(raw_layer_id)] = {"keep": sorted(ranked[:layer_keep])}

    fingerprint = explorer.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise CanaryFailure("routing explorer omitted its source fingerprint")
    selection_config: dict[str, object] = {
        "explorer_fingerprint": fingerprint,
        "metric": "routing_mass",
        "default_keep_per_layer": default_keep,
        "custom_layer_id": layer_ids[0],
        "custom_layer_keep": custom_keep,
    }
    return {"version": 1, "layers": profile_layers}, selection_config


def _response_body(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError as error:
        raise CanaryFailure(
            f"API returned non-JSON content with status {response.status_code}"
        ) from error


def _transient_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


def _error_message(status_code: int, detail: object) -> str:
    if isinstance(detail, str) and detail:
        return f"API request failed ({status_code}): {detail}"
    if isinstance(detail, Mapping):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return f"API request failed ({status_code}): {message}"
    return f"API request failed ({status_code})"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryFailure(f"{label} is not a JSON object")
    return dict(value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CanaryFailure(f"{label} is not a JSON array")
    return value


def _string(value: Mapping[str, Any], field_name: str) -> str:
    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value:
        raise CanaryFailure(f"response omitted non-empty {field_name}")
    return field_value


def _integer(value: Mapping[str, Any], field_name: str) -> int:
    field_value = value.get(field_name)
    if type(field_value) is not int:
        raise CanaryFailure(f"response omitted integer {field_name}")
    return field_value


def _boolean(value: Mapping[str, Any], field_name: str) -> bool:
    field_value = value.get(field_name)
    if type(field_value) is not bool:
        raise CanaryFailure(f"response omitted boolean {field_name}")
    return field_value


def _job_kind(job: Mapping[str, Any]) -> str:
    return _string(job, "kind")


def _job_status(job: Mapping[str, Any]) -> str:
    return _string(job, "status")


def _short_id(job: Mapping[str, Any]) -> str:
    return _string(job, "id")[:8]


if __name__ == "__main__":
    raise SystemExit(main())
