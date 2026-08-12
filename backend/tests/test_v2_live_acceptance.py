from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from moe_tools_suite.domain import EvaluationContract, ExecutionPolicy
from moe_tools_suite.v2_domain import canonical_fingerprint

from scripts.acceptance_canary import ApiClient, CanaryFailure, Reporter
from scripts.v2_live_acceptance import (
    V2_ACCEPTANCE_STEP_COUNT,
    V2AcceptanceConfig,
    V2LiveAcceptance,
    _activation_statistics,
    _answer_evaluation_contract,
    _answer_execution_policy,
    _answer_generation,
    _answer_seed_repetitions,
    _canonical_fingerprint,
    _contract_fingerprint,
    _drift_insight,
    _profile_payloads,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
MODEL_REVISION = "1" * 40
TOPOLOGY_FINGERPRINT = "a" * 64
BASELINE_FINGERPRINT = "b" * 64
PROFILE_FINGERPRINTS = ("1" * 64, "2" * 64, "3" * 64)
PROFILE_CONTEXT_FINGERPRINTS = ("c" * 64, "d" * 64, "e" * 64)
ANSWER_CONTENT_FINGERPRINT = "8" * 64
ANSWER_UNIT_IDS = tuple(f"arith-{index:02d}" for index in range(3, 15))


class FakeV2Deployment:
    def __init__(self) -> None:
        self.loaded = False
        self.model_load_posts = 0
        self.profiles: dict[str, dict[str, Any]] = {}
        self.current = self._context(None)
        self.progress_reads = 0
        self.experiments: dict[str, str] = {}
        self.progress_payload: dict[str, Any] | None = None
        self.daytona_preflight_posts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        path = request.url.path
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
            assert payload == {"token": "live-secret"}
            return self._json(
                request,
                {
                    "auth_required": True,
                    "authenticated": True,
                    "csrf_token": "csrf-v2",
                },
            )
        if method == "GET" and path == "/api/jobs/active":
            return self._json(request, None)
        if method == "GET" and path == "/api/models":
            return self._json(request, [self._model()])
        if method == "GET" and path == "/api/jobs":
            jobs = [self._load_job()] if self.loaded else []
            return self._json(request, jobs)
        if method == "POST" and path == "/api/model-sessions":
            self._assert_csrf(request)
            assert payload == {"model_id": MODEL_ID}
            self.model_load_posts += 1
            if self.model_load_posts > 1:
                raise AssertionError("V2 harness attempted a second cold model load")
            self.loaded = True
            self.current = self._context(None)
            return self._json(request, self._load_job(), status_code=202)
        if method == "GET" and path == "/api/model-sessions/session-v2":
            return self._json(request, self._session())
        if method == "GET" and path == "/api/model-sessions/current":
            return self._json(request, self._session())
        if method == "GET" and path == "/api/runtime/status":
            return self._json(
                request,
                {
                    "managed": True,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "pid": 4242,
                    "session_id": "session-v2",
                    "log_tail": "",
                },
            )
        if method == "POST" and path == "/api/profiles/validate":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            assert len(payload["layers"]) == 4
            return self._json(
                request,
                {
                    "valid": True,
                    "errors": [],
                    "eligible_experts": 20,
                    "total_experts": 32,
                    "retained_fraction": 0.625,
                },
            )
        if method == "POST" and path == "/api/profiles":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            index = len(self.profiles)
            profile_id = f"profile-{index + 1}"
            saved = {
                **payload,
                "id": profile_id,
                "profile_fingerprint": PROFILE_FINGERPRINTS[index],
                "profile_fingerprint_version": 1,
                "validation": {"valid": True},
            }
            self.profiles[profile_id] = saved
            return self._json(request, saved, status_code=201)
        if method == "PATCH" and path.startswith("/api/profiles/"):
            self._assert_csrf(request)
            profile_id = path.rsplit("/", maxsplit=1)[-1]
            assert isinstance(payload, Mapping)
            self.profiles[profile_id]["name"] = payload["name"]
            return self._json(request, self.profiles[profile_id])
        if method == "GET" and path.startswith("/api/profiles/"):
            profile_id = path.rsplit("/", maxsplit=1)[-1]
            return self._json(request, self.profiles[profile_id])
        if method == "POST" and path == "/api/expert-contexts/activate":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            profile_id = payload.get("profile_id")
            if isinstance(profile_id, str) and profile_id.startswith(
                "v2-acceptance-missing-"
            ):
                return self._json(request, {"detail": "not found"}, status_code=404)
            old = self.current
            self.current = self._context(profile_id)
            return self._json(
                request,
                {
                    "old_context_id": old["context_id"],
                    "old_context_fingerprint": old["context_fingerprint"],
                    "new_context_id": self.current["context_id"],
                    "new_context_fingerprint": self.current["context_fingerprint"],
                    "topology_fingerprint": TOPOLOGY_FINGERPRINT,
                    "duration_ms": 7.5,
                    "process_id": 4242,
                    "weights_reloaded": False,
                },
            )
        if method == "GET" and path == "/api/expert-contexts/current":
            return self._json(request, self.current)
        if method == "GET" and path == "/api/workloads":
            return self._json(request, self._workloads())
        if method == "GET" and path == "/api/sandbox-providers":
            return self._json(
                request,
                [{"id": "daytona", "configured": True, "status": "unknown"}],
            )
        if method == "POST" and path == "/api/sandbox-providers/daytona/preflight":
            self._assert_csrf(request)
            self.daytona_preflight_posts += 1
            return self._json(
                request,
                {
                    "provider_id": "daytona",
                    "status": "ready",
                    "configured": True,
                    "reachable": True,
                    "authenticated": True,
                    "api_url_host": "app.daytona.io",
                    "region": "us",
                    "error": None,
                },
            )
        if method == "POST" and path == "/api/experiments":
            self._assert_csrf(request)
            assert isinstance(payload, Mapping)
            if payload["workload_id"].startswith("answer:"):
                experiment_id = "experiment-progress"
                self.progress_payload = dict(payload)
                assert payload["workload_id"] == "answer:fixture-arithmetic"
                assert payload["workload_unit_ids"] == list(ANSWER_UNIT_IDS)
                assert payload["seeds"] == [0, 1]
                assert payload["generation"] == _answer_generation()
                assert payload["evaluation_contract"] == (_answer_evaluation_contract())
                assert payload["execution_policy"] == _answer_execution_policy()
            elif payload["workload_id"].startswith("coding:"):
                experiment_id = "experiment-coding"
            else:
                experiment_id = "experiment-drift"
            self.experiments[experiment_id] = str(payload.get("candidate_profile_id"))
            return self._json(
                request,
                {"id": experiment_id, "job_id": f"job-{experiment_id}"},
                status_code=202,
            )
        if method == "GET" and path == "/api/experiments/experiment-progress":
            self.progress_reads += 1
            if self.progress_reads == 1:
                return self._json(request, self._progress_detail(0, "running"))
            if self.progress_reads == 2:
                return self._json(request, self._progress_detail(1, "running"))
            if self.progress_reads == 3:
                return self._json(request, self._progress_detail(5, "running"))
            return self._json(request, self._progress_detail(20, "completed"))
        if method == "GET" and path == "/api/experiments/experiment-progress/events":
            after = int(request.url.params.get("after", "0"))
            if self.progress_reads >= 4:
                all_events = self._progress_events()
                events = [event for event in all_events if event["sequence"] > after]
                latest = all_events[-1]["sequence"]
            elif after == 0:
                events = [self._queued_event(), self._current_event(2, 0)]
                latest = 2
            else:
                events = [
                    self._inference_event(after + 1, 0),
                    self._current_event(after + 2, 1),
                ]
                latest = after + 2
            return self._json(
                request,
                {
                    "experiment_id": "experiment-progress",
                    "events": events,
                    "after": after,
                    "latest_sequence": latest,
                    "has_more": False,
                },
            )
        if method == "GET" and path == "/api/experiments/experiment-coding":
            return self._json(request, self._paired_detail("coding"))
        if method == "GET" and path == "/api/experiments/experiment-drift":
            return self._json(request, self._paired_detail("drift"))
        if method == "GET" and path in {
            "/api/agent-runs/agent-baseline",
            "/api/agent-runs/agent-candidate",
        }:
            suffix = path.rsplit("-", maxsplit=1)[-1]
            return self._json(request, self._agent_run(suffix))
        if method == "GET" and path == "/api/agent-runs":
            return self._json(
                request,
                [self._agent_run("baseline"), self._agent_run("candidate")],
            )
        raise AssertionError(f"unexpected request: {method} {request.url}")

    @staticmethod
    def _json(
        request: httpx.Request, payload: object, status_code: int = 200
    ) -> httpx.Response:
        return httpx.Response(status_code, json=payload, request=request)

    @staticmethod
    def _assert_csrf(request: httpx.Request) -> None:
        assert request.headers["X-CSRF-Token"] == "csrf-v2"

    @staticmethod
    def _model() -> dict[str, Any]:
        return {
            "id": MODEL_ID,
            "enabled": True,
            "revision": MODEL_REVISION,
            "hot_switch_status": "qualified",
            "qualification_status": "qualified",
            "topology": {
                "num_layers": 4,
                "num_experts": 32,
                "top_k": 4,
                "routed_layer_ids": [0, 1, 2, 3],
            },
        }

    @staticmethod
    def _load_job() -> dict[str, Any]:
        phases = [
            "queued",
            "resolving_model",
            "checking_cache",
            "downloading",
            "configuring_runtime",
            "launching_process",
            "loading_weights",
            "initializing_distributed_workers",
            "compiling",
            "capturing_graphs",
            "warming",
            "waiting_for_readiness",
            "verifying_ready_context",
            "ready",
        ]
        return {
            "id": "load-v2",
            "kind": "model_load",
            "status": "completed",
            "progress_current": 1,
            "progress_total": 1,
            "result_id": "session-v2",
            "error": None,
            "phase_history": [
                {
                    "phase": phase,
                    "status": "completed",
                    "started_at": "2026-08-12T12:00:00+00:00",
                    "completed_at": "2026-08-12T12:00:01+00:00",
                    "detail": phase,
                }
                for phase in phases
            ],
        }

    @staticmethod
    def _session() -> dict[str, Any]:
        return {
            "id": "session-v2",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "state": "ready",
            "profile_id": None,
        }

    def _context(self, profile_id: object) -> dict[str, Any]:
        if profile_id is None:
            return {
                "context_id": "baseline",
                "kind": "baseline",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "topology_fingerprint": TOPOLOGY_FINGERPRINT,
                "profile_id": None,
                "profile_fingerprint": None,
                "context_fingerprint": BASELINE_FINGERPRINT,
                "creation_source": "baseline",
                "metadata": {},
            }
        assert isinstance(profile_id, str)
        index = int(profile_id.rsplit("-", maxsplit=1)[-1]) - 1
        return {
            "context_id": f"expert-mask:{index + 1}",
            "kind": "expert_mask",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "topology_fingerprint": TOPOLOGY_FINGERPRINT,
            "profile_id": profile_id,
            "profile_fingerprint": PROFILE_FINGERPRINTS[index],
            "context_fingerprint": PROFILE_CONTEXT_FINGERPRINTS[index],
            "creation_source": "manual",
            "metadata": {},
        }

    @staticmethod
    def _workloads() -> list[dict[str, Any]]:
        return [
            {
                "id": "answer:fixture-arithmetic",
                "kind": "answer",
                "ready": True,
                "content_fingerprint": ANSWER_CONTENT_FINGERPRINT,
                "unit_ids": list(ANSWER_UNIT_IDS),
            },
            {
                "id": "coding:smoke-python-v1:fix-subtract",
                "kind": "coding",
                "ready": True,
            },
            {"id": "state-drift-v1", "kind": "state_drift", "ready": True},
        ]

    def _progress_detail(self, completed: int, status: str) -> dict[str, Any]:
        assert self.progress_payload is not None
        contract = {
            **self.progress_payload["evaluation_contract"],
            "fingerprint": _contract_fingerprint(
                self.progress_payload["evaluation_contract"]
            ),
        }
        policy = {
            **self.progress_payload["execution_policy"],
            "fingerprint": _contract_fingerprint(
                self.progress_payload["execution_policy"]
            ),
        }
        cohort_fingerprint = _canonical_fingerprint(
            {
                "version": 1,
                "workload_fingerprint": ANSWER_CONTENT_FINGERPRINT,
                "workload_unit_ids": list(ANSWER_UNIT_IDS),
                "scenario_ids": [],
                "conditions": [],
                "horizons": [],
                "seeds": [0, 1],
                "generation": self.progress_payload["generation"],
                "evaluation_contract_fingerprint": contract["fingerprint"],
                "execution_policy_fingerprint": policy["fingerprint"],
            }
        )
        return {
            "experiment": {
                "id": "experiment-progress",
                "status": status,
                "completed_units": completed,
                "total_units": 24,
                "latest_event_sequence": completed + 2,
                "cohort_fingerprint": cohort_fingerprint,
                "generation": self.progress_payload["generation"],
                "evaluation_contract": contract,
                "execution_policy": policy,
            },
            "workload_runs": [],
            "units": self._answer_units() if status == "completed" else [],
        }

    def _answer_units(self) -> list[dict[str, Any]]:
        assert self.progress_payload is not None
        contract_fingerprint = _contract_fingerprint(
            self.progress_payload["evaluation_contract"]
        )
        policy_fingerprint = _contract_fingerprint(
            self.progress_payload["execution_policy"]
        )
        units = []
        for workload_unit_id in ANSWER_UNIT_IDS:
            for seed in (0, 1):
                units.append(
                    {
                        "id": f"answer-unit-{workload_unit_id}-{seed}",
                        "workload_unit_id": workload_unit_id,
                        "seed": seed,
                        "status": "passed",
                        "result": {
                            "workload_unit_id": workload_unit_id,
                            "evaluation_contract_fingerprint": contract_fingerprint,
                            "execution_policy_fingerprint": policy_fingerprint,
                            "generation": {
                                **self.progress_payload["generation"],
                                "seed": seed,
                            },
                        },
                    }
                )
        return units

    def _queued_event(self) -> dict[str, Any]:
        assert self.progress_payload is not None
        cohort_fingerprint = self._progress_detail(0, "running")["experiment"][
            "cohort_fingerprint"
        ]
        return {
            "experiment_id": "experiment-progress",
            "sequence": 1,
            "kind": "queued",
            "phase": "queued",
            "message": "queued",
            "data": {
                "cohort_fingerprint": cohort_fingerprint,
                "workload_unit_count": len(ANSWER_UNIT_IDS),
            },
        }

    @staticmethod
    def _current_event(sequence: int, ordinal: int) -> dict[str, Any]:
        workload_unit_id = ANSWER_UNIT_IDS[ordinal % len(ANSWER_UNIT_IDS)]
        seed = ordinal // len(ANSWER_UNIT_IDS)
        return {
            "experiment_id": "experiment-progress",
            "sequence": sequence,
            "kind": "current_unit",
            "phase": "running",
            "message": "progress",
            "run_unit_id": f"answer-unit-{workload_unit_id}-{seed}",
            "data": {
                "workload_unit_id": workload_unit_id,
                "seed": seed,
            },
        }

    @staticmethod
    def _inference_event(sequence: int, ordinal: int) -> dict[str, Any]:
        workload_unit_id = ANSWER_UNIT_IDS[ordinal % len(ANSWER_UNIT_IDS)]
        seed = ordinal // len(ANSWER_UNIT_IDS)
        return {
            "experiment_id": "experiment-progress",
            "sequence": sequence,
            "kind": "inference_progress",
            "phase": "inference",
            "message": "streaming",
            "run_unit_id": f"answer-unit-{workload_unit_id}-{seed}",
            "data": {
                "workload_unit_id": workload_unit_id,
                "seed": seed,
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "elapsed_ms": 125,
                "current_tps": 16,
            },
        }

    def _progress_events(self) -> list[dict[str, Any]]:
        events = [self._queued_event()]
        for ordinal in range(24):
            events.extend(
                [
                    self._current_event(2 + ordinal * 2, ordinal),
                    self._inference_event(3 + ordinal * 2, ordinal),
                ]
            )
        events.append(
            {
                "experiment_id": "experiment-progress",
                "sequence": 50,
                "kind": "completed",
                "phase": "completed",
                "message": "completed",
                "data": {},
            }
        )
        return events

    def _paired_detail(self, kind: str) -> dict[str, Any]:
        candidate_profile_id = "profile-1" if kind == "coding" else "profile-2"
        baseline_context = self._context(None)
        candidate_context = self._context(candidate_profile_id)
        experiment_id = f"experiment-{kind}"
        lanes = [
            {
                "id": f"lane-{kind}-baseline",
                "role": "baseline",
                "context": baseline_context,
            },
            {
                "id": f"lane-{kind}-candidate",
                "role": "candidate",
                "context": candidate_context,
            },
        ]
        if kind == "coding":
            units = [
                self._coding_unit("baseline", baseline_context),
                self._coding_unit("candidate", candidate_context),
            ]
            comparison: dict[str, Any] = {
                "paired_observations": 1,
                "retained_passes": 1,
            }
        else:
            units = []
            comparison = {
                "formula_fingerprint": "f" * 64,
                "paired_observations": 12,
                "baseline_to_mask_paired_drift_delta": -0.04,
                "excess_compounding_penalty": 0.08,
                "mean_routing_selection_overlap": 0.72,
                "mean_routing_mass_js_divergence": 0.13,
                "routing_checkpoint_pairs": [
                    {
                        "selection_overlap": 0.72,
                        "routing_mass_js_divergence": 0.13,
                    }
                ],
            }
        return {
            "experiment": {
                "id": experiment_id,
                "status": "completed",
                "completed_units": len(units) or 24,
                "total_units": len(units) or 24,
                "latest_event_sequence": 10,
                "lanes": lanes,
                "comparison": comparison,
            },
            "workload_runs": [],
            "units": units,
        }

    @staticmethod
    def _coding_unit(role: str, context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": f"unit-{role}",
            "status": "passed",
            "result": {
                "agent_run_id": f"agent-{role}",
                "trial_id": f"trial-{role}",
                "context": context,
                "trajectory": [
                    {"type": "assistant"},
                    {"type": "tool"},
                    {"type": "observation"},
                    {"type": "verifier"},
                ],
                "routing": {"total_routed_slots": 128},
            },
        }

    @staticmethod
    def _agent_run(suffix: str) -> dict[str, Any]:
        return {
            "id": f"agent-{suffix}",
            "sandbox_provider_id": "fake",
            "trials": [
                {
                    "id": f"trial-{suffix}",
                    "sandbox_status": "deleted",
                }
            ],
        }


def test_full_v2_harness_loads_once_and_emits_machine_readable_evidence() -> None:
    deployment = FakeV2Deployment()
    messages: list[str] = []
    config = V2AcceptanceConfig(
        base_url="https://v2.test",
        token="live-secret",
        switch_rounds=2,
        poll_interval_seconds=0.001,
        coding_provider_id="daytona",
    )
    canary_config = config.canary_config()

    def client() -> ApiClient:
        return ApiClient(
            canary_config,
            transport=httpx.MockTransport(deployment),
            sleep=lambda _: None,
        )

    with client() as api:
        runner = V2LiveAcceptance(
            config,
            api,
            Reporter(
                total_steps=V2_ACCEPTANCE_STEP_COUNT,
                secrets=(config.token or "",),
                write=messages.append,
            ),
            reconnect_client=client,
        )
        report = runner.run()

    assert deployment.model_load_posts == 1
    assert deployment.daytona_preflight_posts == 1
    assert report["app_acceptance_passed"] is True
    assert report["release_acceptance_complete"] is False
    assert report["paired_coding"]["release_qualifying"] is True
    assert report["paired_coding"]["provider_preflight"] == {
        "provider_id": "daytona",
        "status": "ready",
        "configured": True,
        "reachable": True,
        "authenticated": True,
        "api_url_host": "app.daytona.io",
        "region": "us",
    }
    assert report["model"]["process_id"] == 4242
    benchmark = report["activation_benchmark"]
    assert len(benchmark["samples"]) == 8
    assert benchmark["statistics"]["wall_duration_ms"]["p95"] < 2000
    assert {sample["process_id"] for sample in benchmark["samples"]} == {4242}
    assert {sample["weights_reloaded"] for sample in benchmark["samples"]} == {False}
    assert report["ordinary_live_progress"]["reconnect"]["adopted_while_active"]
    ordinary = report["ordinary_live_progress"]
    assert ordinary["workload_id"] == "answer:fixture-arithmetic"
    assert ordinary["workload_unit_ids"] == list(ANSWER_UNIT_IDS)
    assert ordinary["distinct_workload_unit_count"] == 12
    assert ordinary["seed_repetitions"] == 2
    assert ordinary["lane_units"] == 24
    assert ordinary["persisted_cohort_evidence"]["persisted_run_unit_count"] == 24
    assert ordinary["persisted_cohort_evidence"]["unique_run_unit_ids"] == 24
    assert ordinary["event_item_evidence"]["current_unit_event_count"] == 24
    assert ordinary["event_item_evidence"]["distinct_workload_unit_ids"] == list(
        ANSWER_UNIT_IDS
    )
    assert ordinary["event_item_evidence"]["current_tps_sample_count"] == 24
    assert ordinary["reconnect"]["replayed_workload_unit_ids"] == [
        "arith-03",
        "arith-04",
    ]
    assert ordinary["reconnect"]["inference_progress_visible_after_reconnect"]
    assert ordinary["reconnect"]["replayed_inference_progress"][0]["current_tps"] == 16
    assert report["in_appliance_gates"]["ordinary_inference_progress_and_current_tps"]
    assert report["paired_coding"]["lanes"][0]["sandbox_status"] == "deleted"
    assert report["driftbench"]["insight"][
        "mean_routing_mass_js_divergence"
    ] == pytest.approx(0.13)
    assert (
        report["activation_failure_safety"]["transactional_commit_failure_injected"]
        is False
    )
    step_starts = [message for message in messages if message.endswith(" ...")]
    assert len(step_starts) == V2_ACCEPTANCE_STEP_COUNT
    assert [message.split(maxsplit=1)[0] for message in step_starts] == [
        f"[{index:02d}/{V2_ACCEPTANCE_STEP_COUNT:02d}]"
        for index in range(1, V2_ACCEPTANCE_STEP_COUNT + 1)
    ]


def test_profile_generator_is_full_non_uniform_and_distinct() -> None:
    profiles = _profile_payloads(
        {
            "num_experts": 16,
            "top_k": 2,
            "routed_layer_ids": [2, 4, 7, 9],
        },
        run_id="0123456789abcdef",
    )

    assert len(profiles) == 3
    assert len({json.dumps(profile, sort_keys=True) for profile in profiles}) == 3
    for profile in profiles:
        assert set(profile["layers"]) == {"2", "4", "7", "9"}
        keep_counts = {len(layer["keep"]) for layer in profile["layers"].values()}
        assert len(keep_counts) == 2
        assert min(keep_counts) >= 2
        assert max(keep_counts) < 16


def test_activation_statistics_use_nearest_rank_and_preserve_targets() -> None:
    samples = [
        {
            "target": "baseline" if index % 2 == 0 else "profile_1",
            "receipt_duration_ms": float(index),
            "wall_duration_ms": float(index * 2),
        }
        for index in range(1, 21)
    ]

    statistics = _activation_statistics(samples)

    assert statistics["receipt_duration_ms"]["p50"] == 10
    assert statistics["receipt_duration_ms"]["p95"] == 19
    assert statistics["wall_duration_ms"]["p95"] == 38
    assert statistics["by_target"]["baseline"]["count"] == 10
    assert statistics["by_target"]["profile_1"]["count"] == 10


def test_answer_seed_repetitions_choose_minimum_twenty_unit_cohort() -> None:
    assert _answer_seed_repetitions(12, configured=None) == 2
    assert _answer_seed_repetitions(20, configured=None) == 1
    assert _answer_seed_repetitions(3, configured=None) == 7
    assert _answer_seed_repetitions(12, configured=3) == 3

    with pytest.raises(CanaryFailure, match="use at least 2"):
        _answer_seed_repetitions(12, configured=1)


def test_harness_contract_and_cohort_fingerprints_match_server_contract() -> None:
    evaluation = _answer_evaluation_contract()
    policy = _answer_execution_policy()
    cohort_identity = {
        "version": 1,
        "workload_fingerprint": ANSWER_CONTENT_FINGERPRINT,
        "workload_unit_ids": list(ANSWER_UNIT_IDS),
        "scenario_ids": [],
        "conditions": [],
        "horizons": [],
        "seeds": [0, 1],
        "generation": _answer_generation(),
        "evaluation_contract_fingerprint": _contract_fingerprint(evaluation),
        "execution_policy_fingerprint": _contract_fingerprint(policy),
    }

    assert (
        _contract_fingerprint(evaluation)
        == EvaluationContract.model_validate(evaluation).fingerprint
    )
    assert (
        _contract_fingerprint(policy)
        == ExecutionPolicy.model_validate(policy).fingerprint
    )
    assert _canonical_fingerprint(cohort_identity) == canonical_fingerprint(
        cohort_identity
    )


def test_drift_insight_fails_closed_without_real_routing_metrics() -> None:
    comparison = {
        "paired_observations": 2,
        "baseline_to_mask_paired_drift_delta": -0.1,
        "excess_compounding_penalty": 0.2,
        "mean_routing_selection_overlap": None,
        "mean_routing_mass_js_divergence": None,
    }

    with pytest.raises(CanaryFailure, match="routing selection overlap"):
        _drift_insight(comparison, [])


def _payload(request: httpx.Request) -> object:
    if not request.content:
        return None
    return json.loads(request.content)
