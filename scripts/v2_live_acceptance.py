"""Collect release-gating evidence from a deployed MoE Atelier V2 appliance.

This driver intentionally loads the model exactly once. Every subsequent
baseline/profile change goes through the expert-context activation API; a cold
model-session request after the initial load is a release-gating failure.

The output is a machine-readable evidence record. It distinguishes assertions
the appliance API can prove from provider-wide inventory and injected
transaction-failure checks that require an external control plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

if __package__:
    from .acceptance_canary import (
        ACTIVE_JOB_STATUSES,
        AcceptanceCanary,
        ApiClient,
        ApiFailure,
        CanaryConfig,
        CanaryFailure,
        Reporter,
        _boolean,
        _integer,
        _list,
        _mapping,
        _normalize_base_url,
        _string,
    )
else:
    from acceptance_canary import (  # type: ignore[no-redef]
        ACTIVE_JOB_STATUSES,
        AcceptanceCanary,
        ApiClient,
        ApiFailure,
        CanaryConfig,
        CanaryFailure,
        Reporter,
        _boolean,
        _integer,
        _list,
        _mapping,
        _normalize_base_url,
        _string,
    )

SCHEMA_VERSION = "moe-atelier-v2-live-acceptance/v1"
TERMINAL_EXPERIMENT_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_ANSWER_WORKLOAD = "answer:fixture-arithmetic"
DEFAULT_CODING_WORKLOAD = "coding:smoke-python-v1:fix-subtract"
DEFAULT_DRIFT_SCENARIOS = (
    "ledger-reconciliation-v1",
    "order-lifecycle-v1",
)
DEFAULT_DRIFT_CONDITIONS = ("oracle_reset", "chained", "state_anchored")
DEFAULT_DRIFT_HORIZONS = (4, 8)
MINIMUM_LOAD_PHASES = {
    "queued",
    "resolving_model",
    "configuring_runtime",
    "launching_process",
    "waiting_for_readiness",
    "verifying_ready_context",
    "ready",
}
MINIMUM_LOAD_PHASE_ORDER = (
    "queued",
    "resolving_model",
    "configuring_runtime",
    "launching_process",
    "waiting_for_readiness",
    "verifying_ready_context",
    "ready",
)
REQUESTED_DETAILED_LOAD_PHASES = {
    "checking_cache",
    "downloading",
    "loading_weights",
    "initializing_distributed_workers",
    "compiling",
    "capturing_graphs",
    "warming",
}
MINIMUM_ORDINARY_LANE_UNITS = 20


@dataclass(frozen=True)
class V2AcceptanceConfig:
    base_url: str
    token: str | None = field(default=None, repr=False)
    output_path: Path = Path("v2-live-acceptance.json")
    model_id: str | None = None
    switch_rounds: int = 4
    activation_p95_limit_ms: float = 2000
    answer_workload_id: str = DEFAULT_ANSWER_WORKLOAD
    answer_seed_repetitions: int | None = None
    coding_workload_id: str = DEFAULT_CODING_WORKLOAD
    coding_provider_id: str = "fake"
    drift_scenario_ids: tuple[str, ...] = DEFAULT_DRIFT_SCENARIOS
    drift_conditions: tuple[str, ...] = DEFAULT_DRIFT_CONDITIONS
    drift_horizons: tuple[int, ...] = DEFAULT_DRIFT_HORIZONS
    job_timeout_seconds: float = 3600
    poll_interval_seconds: float = 0.25
    request_timeout_seconds: float = 30
    require_routing_evidence: bool = True

    def canary_config(self) -> CanaryConfig:
        return CanaryConfig(
            base_url=self.base_url,
            token=self.token,
            model_id=self.model_id,
            job_timeout_seconds=self.job_timeout_seconds,
            poll_interval_seconds=self.poll_interval_seconds,
            request_timeout_seconds=self.request_timeout_seconds,
        )


class V2LiveAcceptance:
    """Drive the in-appliance V2 live gates without reloading warm weights."""

    def __init__(
        self,
        config: V2AcceptanceConfig,
        api: ApiClient,
        reporter: Reporter,
        reconnect_client: Callable[[], ApiClient] | None = None,
    ) -> None:
        self.config = config
        self.api = api
        self.reporter = reporter
        self.canary = AcceptanceCanary(config.canary_config(), api, reporter)
        self._reconnect_client = reconnect_client or (
            lambda: ApiClient(config.canary_config())
        )
        self.run_id = uuid4().hex
        self.report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "base_url": config.base_url,
            "started_at": _utc_now(),
            "app_acceptance_passed": False,
            "release_acceptance_complete": False,
            "external_gates": {
                "transactional_commit_failure_injected": False,
                "provider_wide_daytona_inventory_verified": False,
                "runpod_inventory_verified": False,
                "paid_compute_stopped": False,
                "additional_model_qualified": False,
                "browser_hard_refresh_verified": False,
                "stable_weight_memory_verified": False,
                "cold_hot_output_equivalence_verified": False,
                "tp_cache_cuda_equivalence_verified": False,
            },
        }
        self.model: dict[str, Any] = {}
        self.session: dict[str, Any] = {}
        self.initial_pid: int | None = None
        self.initial_model_load_job_id: str | None = None
        self.expected_model_load_job_ids: set[str] = set()
        self.profiles: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        with self.reporter.step("Authenticate and reconcile active work"):
            session_status = self.api.login()
            active = self.canary._active_job()
            if active is not None:
                self.reporter.info(
                    f"Adopting pre-existing job {_string(active, 'id')[:8]}"
                )
                self.canary.wait_for_job(active)
            self.report["authentication"] = {
                "required": bool(session_status.get("auth_required")),
                "authenticated": bool(session_status.get("authenticated")),
            }

        with self.reporter.step("Load the immutable model exactly once"):
            self._load_model_once()

        with self.reporter.step("Create three topology-valid named profiles"):
            self._create_profiles()

        with self.reporter.step("Benchmark baseline and three hot contexts"):
            self._benchmark_activations()

        with self.reporter.step("Rename without changing mask identity"):
            self._exercise_profile_rename()

        with self.reporter.step("Reject a failed activation without state loss"):
            self._exercise_rejection_safety()

        with self.reporter.step("Observe and reconnect to ordinary live progress"):
            self._run_progress_acceptance()

        with self.reporter.step("Run paired coding lanes"):
            self._run_coding_acceptance()

        with self.reporter.step("Run paired DriftBench and derive an insight"):
            self._run_drift_acceptance()

        with self.reporter.step("Verify app-visible cleanup and final invariants"):
            self._verify_cleanup_and_final_state()

        detailed_phases_complete = not self.report["initial_model_load"][
            "missing_requested_detailed_phases"
        ]
        self.report["in_appliance_gates"] = {
            "single_model_load": True,
            "hot_activation": True,
            "stable_process_identity_and_no_reload_receipts": True,
            "profile_rename_identity": True,
            "rejected_activation_safety": True,
            "ordinary_progress_and_reconnect": True,
            "ordinary_inference_progress_and_current_tps": bool(
                self.report["ordinary_live_progress"]["event_item_evidence"][
                    "current_tps_sample_count"
                ]
                and self.report["ordinary_live_progress"]["reconnect"][
                    "inference_progress_visible_after_reconnect"
                ]
            ),
            "model_authored_paired_coding": (self.config.coding_provider_id != "fake"),
            "paired_drift_with_routing_insight": True,
            "app_visible_cleanup": True,
            "requested_detailed_load_phases": detailed_phases_complete,
        }
        self.report["app_acceptance_passed"] = all(
            self.report["in_appliance_gates"].values()
        )
        self.report["finished_at"] = _utc_now()
        self.report["release_acceptance_complete"] = all(
            self.report["external_gates"].values()
        )
        return self.report

    def _load_model_once(self) -> None:
        registry = _list(self.api.get("/api/models"), "model registry")
        enabled = [
            _mapping(item, "model registry entry")
            for item in registry
            if isinstance(item, Mapping) and item.get("enabled") is True
        ]
        if self.config.model_id is None:
            if len(enabled) != 1:
                raise CanaryFailure(
                    "select --model-id unless exactly one model is enabled"
                )
            self.model = enabled[0]
        else:
            self.model = next(
                (item for item in enabled if item.get("id") == self.config.model_id),
                {},
            )
            if not self.model:
                raise CanaryFailure(
                    f"requested model {self.config.model_id!r} is not enabled"
                )
        if self.model.get("hot_switch_status") == "unsupported":
            raise CanaryFailure("selected model explicitly lacks hot-switch support")
        model_id = _string(self.model, "id")
        model_load_ids_before = self._model_load_job_ids()
        started = time.perf_counter()
        submitted = self.canary.submit_job(
            "/api/model-sessions",
            {"model_id": model_id},
            expected_kind="model_load",
        )
        self.initial_model_load_job_id = _string(submitted, "id")
        completed = self.canary.wait_for_job(submitted)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.session = self.canary._model_session_from_job(completed)
        if self.session.get("model_id") != model_id:
            raise CanaryFailure("ready session uses a different model")
        if self.session.get("profile_id") is not None:
            raise CanaryFailure("initial model load unexpectedly selected a profile")
        runtime = _mapping(self.api.get("/api/runtime/status"), "runtime status")
        self.initial_pid = _positive_integer(runtime, "pid")
        if runtime.get("session_id") != self.session.get("id"):
            raise CanaryFailure("managed runtime and ready session identities differ")
        if runtime.get("model_id") != model_id:
            raise CanaryFailure("managed runtime loaded a different model")
        phases = _load_phase_evidence(completed)
        available_phases = {
            record["phase"]
            for record in phases
            if record["observability"] == "observed"
            and record["status"] != "unavailable"
        }
        missing_minimum = sorted(MINIMUM_LOAD_PHASES - available_phases)
        if missing_minimum:
            raise CanaryFailure(
                f"model load omitted required durable phases: {missing_minimum}"
            )
        phase_positions = {
            record["phase"]: index for index, record in enumerate(phases)
        }
        if [phase_positions[phase] for phase in MINIMUM_LOAD_PHASE_ORDER] != sorted(
            phase_positions[phase] for phase in MINIMUM_LOAD_PHASE_ORDER
        ):
            raise CanaryFailure("model-load phases are not in lifecycle order")
        model_load_ids_after = self._model_load_job_ids()
        new_job_ids = model_load_ids_after - model_load_ids_before
        if new_job_ids != {self.initial_model_load_job_id}:
            raise CanaryFailure(
                "initial load did not create exactly one observable model-load job"
            )
        self.expected_model_load_job_ids = model_load_ids_after
        self.report["model"] = {
            "id": model_id,
            "revision": self.session.get("model_revision"),
            "registry_qualification": self.model.get("qualification_status"),
            "registry_hot_switch": self.model.get("hot_switch_status"),
            "session_id": self.session.get("id"),
            "process_id": self.initial_pid,
        }
        self.report["initial_model_load"] = {
            "job_id": self.initial_model_load_job_id,
            "elapsed_ms": elapsed_ms,
            "phases": phases,
            "minimum_phase_contract_complete": True,
            "missing_requested_detailed_phases": sorted(
                REQUESTED_DETAILED_LOAD_PHASES - available_phases
            ),
            "requested_detailed_phase_contract_complete": (
                available_phases >= REQUESTED_DETAILED_LOAD_PHASES
            ),
        }

    def _create_profiles(self) -> None:
        topology = _mapping(self.model.get("topology"), "model topology")
        payloads = _profile_payloads(topology, run_id=self.run_id)
        created = []
        for index, profile in enumerate(payloads, start=1):
            validation = _mapping(
                self.api.post("/api/profiles/validate", profile),
                "profile validation",
            )
            if not _boolean(validation, "valid"):
                raise CanaryFailure(f"acceptance profile {index} failed validation")
            saved = _mapping(
                self.api.post(
                    "/api/profiles",
                    {
                        "name": f"V2 live acceptance {self.run_id[:8]} · mask {index}",
                        "description": (
                            "Generated by the V2 live-acceptance harness from the "
                            "deployed immutable model topology."
                        ),
                        "model_id": _string(self.model, "id"),
                        "profile": profile,
                        "source": "manual",
                        "selection_strategy": "v2_live_acceptance_rotating_mask",
                        "selection_config": {
                            "acceptance_run_id": self.run_id,
                            "variant": index,
                        },
                    },
                ),
                "saved profile",
            )
            if saved.get("profile_fingerprint_version") != 1:
                raise CanaryFailure("new profile omitted canonical fingerprint version")
            _sha256(saved.get("profile_fingerprint"), "profile fingerprint")
            if saved.get("profile") != profile:
                raise CanaryFailure("saved profile changed the submitted expert mask")
            created.append(saved)
        fingerprints = {profile["profile_fingerprint"] for profile in created}
        if len(fingerprints) != 3:
            raise CanaryFailure("acceptance profiles do not have distinct masks")
        self.profiles = created
        self.report["profiles"] = [
            {
                "id": profile["id"],
                "name": profile["name"],
                "profile_fingerprint": profile["profile_fingerprint"],
                "profile_fingerprint_version": profile["profile_fingerprint_version"],
                "layer_keep_counts": sorted(
                    {
                        len(_mapping(layer, "profile layer")["keep"])
                        for layer in _mapping(
                            _mapping(profile["profile"], "profile")["layers"],
                            "profile layers",
                        ).values()
                    }
                ),
            }
            for profile in created
        ]

    def _benchmark_activations(self) -> None:
        initial_job_ids = self._model_load_job_ids()
        samples: list[dict[str, Any]] = []
        targets = [("baseline", None, None)] + [
            (
                f"profile_{index}",
                _string(profile, "id"),
                _sha256(profile.get("profile_fingerprint"), "profile fingerprint"),
            )
            for index, profile in enumerate(self.profiles, start=1)
        ]
        for round_index in range(1, self.config.switch_rounds + 1):
            for label, profile_id, profile_fingerprint in targets:
                wall_started = time.perf_counter()
                receipt = _mapping(
                    self.api.post(
                        "/api/expert-contexts/activate",
                        {"profile_id": profile_id},
                    ),
                    "context activation receipt",
                )
                wall_ms = (time.perf_counter() - wall_started) * 1000
                current = _mapping(
                    self.api.get("/api/expert-contexts/current"),
                    "current expert context",
                )
                runtime = _mapping(
                    self.api.get("/api/runtime/status"), "runtime status"
                )
                _validate_activation(
                    receipt,
                    current,
                    expected_model_id=_string(self.model, "id"),
                    expected_profile_id=profile_id,
                    expected_profile_fingerprint=profile_fingerprint,
                    expected_pid=self.initial_pid,
                )
                if runtime.get("pid") != self.initial_pid:
                    raise CanaryFailure("vLLM PID changed during context activation")
                if runtime.get("session_id") != self.session.get("id"):
                    raise CanaryFailure(
                        "model session changed during context activation"
                    )
                samples.append(
                    {
                        "round": round_index,
                        "target": label,
                        "profile_id": profile_id,
                        "receipt_duration_ms": _nonnegative_number(
                            receipt.get("duration_ms"), "activation duration"
                        ),
                        "wall_duration_ms": wall_ms,
                        "process_id": receipt.get("process_id"),
                        "weights_reloaded": receipt.get("weights_reloaded"),
                        "context_id": receipt.get("new_context_id"),
                        "context_fingerprint": receipt.get("new_context_fingerprint"),
                        "topology_fingerprint": receipt.get("topology_fingerprint"),
                    }
                )
        if self._model_load_job_ids() != initial_job_ids:
            raise CanaryFailure("context activation created a model-load job")
        stats = _activation_statistics(samples)
        if stats["wall_duration_ms"]["p95"] > self.config.activation_p95_limit_ms:
            raise CanaryFailure(
                "hot-context wall-time P95 exceeded "
                f"{self.config.activation_p95_limit_ms:g} ms"
            )
        if stats["receipt_duration_ms"]["p95"] > (self.config.activation_p95_limit_ms):
            raise CanaryFailure(
                "server-reported hot-context P95 exceeded "
                f"{self.config.activation_p95_limit_ms:g} ms"
            )
        self.report["activation_benchmark"] = {
            "rounds": self.config.switch_rounds,
            "targets": [target[0] for target in targets],
            "samples": samples,
            "statistics": stats,
            "p95_limit_ms": self.config.activation_p95_limit_ms,
            "stable_process_id": self.initial_pid,
            "stable_model_session_id": self.session.get("id"),
            "weights_reloaded": False,
            "new_model_load_jobs": [],
        }

    def _exercise_profile_rename(self) -> None:
        profile = self.profiles[0]
        profile_id = _string(profile, "id")
        before_context = _mapping(
            self.api.get("/api/expert-contexts/current"), "current context"
        )
        before_jobs = self._model_load_job_ids()
        renamed = _mapping(
            self.api.patch(
                f"/api/profiles/{profile_id}",
                {"name": f"Renamed V2 live mask · {self.run_id[:8]}"},
            ),
            "renamed profile",
        )
        fetched = _mapping(
            self.api.get(f"/api/profiles/{profile_id}"), "renamed profile"
        )
        after_context = _mapping(
            self.api.get("/api/expert-contexts/current"), "current context"
        )
        if renamed != fetched:
            raise CanaryFailure("renamed profile was not durably readable")
        for immutable_field in ("id", "profile", "profile_fingerprint"):
            if renamed.get(immutable_field) != profile.get(immutable_field):
                raise CanaryFailure(
                    f"profile rename changed immutable field {immutable_field}"
                )
        if before_context != after_context:
            raise CanaryFailure("profile rename activated a different context")
        if self._model_load_job_ids() != before_jobs:
            raise CanaryFailure("profile rename created a model-load job")
        self.profiles[0] = fetched
        self.report["profile_rename"] = {
            "profile_id": profile_id,
            "old_name": profile.get("name"),
            "new_name": fetched.get("name"),
            "profile_fingerprint": fetched.get("profile_fingerprint"),
            "context_unchanged": True,
            "model_load_jobs_created": 0,
        }

    def _exercise_rejection_safety(self) -> None:
        before_context = _mapping(
            self.api.get("/api/expert-contexts/current"), "current context"
        )
        before_runtime = _mapping(self.api.get("/api/runtime/status"), "runtime status")
        missing_profile_id = f"v2-acceptance-missing-{self.run_id}"
        try:
            self.api.post(
                "/api/expert-contexts/activate",
                {"profile_id": missing_profile_id},
            )
        except ApiFailure as error:
            if error.status_code != 404:
                raise CanaryFailure(
                    "invalid activation failed at an unexpected API boundary"
                ) from error
            failure = {"status_code": error.status_code, "detail": error.detail}
        else:
            raise CanaryFailure(
                "activation of a missing profile unexpectedly succeeded"
            )
        after_context = _mapping(
            self.api.get("/api/expert-contexts/current"), "current context"
        )
        after_runtime = _mapping(self.api.get("/api/runtime/status"), "runtime status")
        ready_session = _mapping(
            self.api.get("/api/model-sessions/current"), "current model session"
        )
        if before_context != after_context:
            raise CanaryFailure("rejected activation changed the ready context")
        if before_runtime.get("pid") != after_runtime.get("pid"):
            raise CanaryFailure("rejected activation changed the vLLM PID")
        if ready_session.get("state") != "ready":
            raise CanaryFailure("rejected activation left the model unusable")
        self.report["activation_failure_safety"] = {
            "failure": failure,
            "failure_stage": "application_profile_resolution",
            "prior_context_retained": True,
            "model_remained_ready": True,
            "process_id_unchanged": True,
            "transactional_commit_failure_injected": False,
            "limitation": (
                "The public appliance API exposes no production-safe commit-fault "
                "injection. Fork integration tests or an externally controlled "
                "fault-injection run must prove distributed commit rollback."
            ),
        }

    def _run_progress_acceptance(self) -> None:
        workload = self._require_workload(
            self.config.answer_workload_id, expected_kind="answer"
        )
        workload_unit_ids = _published_workload_unit_ids(workload)
        seed_repetitions = _answer_seed_repetitions(
            len(workload_unit_ids),
            configured=self.config.answer_seed_repetitions,
        )
        seeds = list(range(seed_repetitions))
        generation = _answer_generation()
        evaluation_contract = _answer_evaluation_contract()
        execution_policy = _answer_execution_policy()
        expected_contract_fingerprint = _contract_fingerprint(evaluation_contract)
        expected_policy_fingerprint = _contract_fingerprint(execution_policy)
        expected_cohort_fingerprint = _canonical_fingerprint(
            {
                "version": 1,
                "workload_fingerprint": _sha256(
                    workload.get("content_fingerprint"),
                    "answer workload content fingerprint",
                ),
                "workload_unit_ids": workload_unit_ids,
                "scenario_ids": [],
                "conditions": [],
                "horizons": [],
                "seeds": seeds,
                "generation": generation,
                "evaluation_contract_fingerprint": (expected_contract_fingerprint),
                "execution_policy_fingerprint": expected_policy_fingerprint,
            }
        )
        experiment = self._submit_experiment(
            {
                "name": f"V2 live progress · {self.run_id[:8]}",
                "model_id": _string(self.model, "id"),
                "workload_id": workload["id"],
                "workload_unit_ids": workload_unit_ids,
                "seeds": seeds,
                "generation": generation,
                "evaluation_contract": evaluation_contract,
                "execution_policy": execution_policy,
            }
        )
        adoption = self._reconnect_mid_experiment(_string(experiment, "id"))
        detail, observations = self._wait_for_experiment(
            _string(experiment, "id"),
            initial_observations=adoption["progress_observations"],
        )
        if not any(
            0 < item["completed_units"] < item["total_units"] for item in observations
        ):
            raise CanaryFailure(
                "ordinary experiment completed without observable item-level progress"
            )
        evidence = _validate_answer_cohort_evidence(
            detail,
            workload_unit_ids=workload_unit_ids,
            seeds=seeds,
            generation=generation,
            evaluation_contract=evaluation_contract,
            execution_policy=execution_policy,
            expected_contract_fingerprint=expected_contract_fingerprint,
            expected_policy_fingerprint=expected_policy_fingerprint,
            expected_cohort_fingerprint=expected_cohort_fingerprint,
        )
        events = self._all_experiment_events(_string(experiment, "id"))
        event_evidence = _validate_answer_progress_events(
            events,
            workload_unit_ids=workload_unit_ids,
            seeds=seeds,
            expected_cohort_fingerprint=expected_cohort_fingerprint,
        )
        replayed_item_ids = adoption["replayed_workload_unit_ids"]
        if not replayed_item_ids:
            raise CanaryFailure(
                "reconnected event replay did not expose a workload item identity"
            )
        if not set(replayed_item_ids) <= set(workload_unit_ids):
            raise CanaryFailure("reconnected progress referenced an unknown item")
        self.report["ordinary_live_progress"] = {
            "experiment_id": _string(
                _mapping(detail.get("experiment"), "experiment"), "id"
            ),
            "workload_id": workload["id"],
            "workload_unit_ids": workload_unit_ids,
            "distinct_workload_unit_count": len(workload_unit_ids),
            "seed_repetitions": seed_repetitions,
            "seeds": seeds,
            "lane_units": len(workload_unit_ids) * seed_repetitions,
            "cohort_fingerprint": expected_cohort_fingerprint,
            "generation": generation,
            "evaluation_contract_fingerprint": expected_contract_fingerprint,
            "execution_policy_fingerprint": expected_policy_fingerprint,
            "persisted_cohort_evidence": evidence,
            "event_item_evidence": event_evidence,
            "progress_observations": observations,
            "reconnect": adoption,
            "terminal_status": _mapping(detail.get("experiment"), "experiment").get(
                "status"
            ),
        }

    def _run_coding_acceptance(self) -> None:
        workload = self._require_workload(
            self.config.coding_workload_id, expected_kind="coding"
        )
        experiment = self._submit_experiment(
            {
                "name": f"V2 paired coding · {self.run_id[:8]}",
                "model_id": _string(self.model, "id"),
                "workload_id": workload["id"],
                "candidate_profile_id": _string(self.profiles[0], "id"),
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": self.config.coding_provider_id,
                "seeds": [0],
            }
        )
        detail, _ = self._wait_for_experiment(_string(experiment, "id"))
        self._validate_experiment_contexts(detail, self.profiles[0])
        experiment_record = _mapping(detail.get("experiment"), "experiment")
        units = [
            _mapping(unit, "coding unit")
            for unit in _list(detail.get("units"), "coding units")
        ]
        if len(units) != 2:
            raise CanaryFailure("paired coding did not produce exactly two lane units")
        required_steps = {"assistant", "tool", "observation", "verifier"}
        lane_evidence = []
        for unit in units:
            result = _mapping(unit.get("result"), "coding unit result")
            trajectory = [
                _mapping(step, "trajectory step")
                for step in _list(result.get("trajectory"), "coding trajectory")
            ]
            observed_steps = {step.get("type") for step in trajectory}
            if not required_steps <= observed_steps:
                raise CanaryFailure("coding trajectory omitted required step types")
            routing = result.get("routing")
            if self.config.require_routing_evidence:
                routing_map = _mapping(routing, "coding routing evidence")
                if _integer(routing_map, "total_routed_slots") <= 0:
                    raise CanaryFailure("coding lane captured no real routing slots")
            lane_evidence.append(
                {
                    "unit_id": unit.get("id"),
                    "status": unit.get("status"),
                    "agent_run_id": result.get("agent_run_id"),
                    "trial_id": result.get("trial_id"),
                    "trajectory_step_types": sorted(observed_steps),
                    "sandbox_status": self._trial_sandbox_status(result),
                    "routing_total_slots": (
                        routing.get("total_routed_slots")
                        if isinstance(routing, Mapping)
                        else None
                    ),
                }
            )
        self.report["paired_coding"] = {
            "experiment_id": experiment_record.get("id"),
            "workload_id": workload["id"],
            "sandbox_provider_id": self.config.coding_provider_id,
            "release_qualifying": self.config.coding_provider_id != "fake",
            "diagnostic_reason": (
                "FakeSandboxProvider substitutes pinned oracle actions and cannot "
                "qualify model-authored coding behavior."
                if self.config.coding_provider_id == "fake"
                else None
            ),
            "comparison": experiment_record.get("comparison"),
            "lanes": lane_evidence,
        }

    def _run_drift_acceptance(self) -> None:
        self._require_workload("state-drift-v1", expected_kind="state_drift")
        experiment = self._submit_experiment(
            {
                "name": f"V2 paired DriftBench · {self.run_id[:8]}",
                "model_id": _string(self.model, "id"),
                "workload_id": "state-drift-v1",
                "candidate_profile_id": _string(self.profiles[1], "id"),
                "scenario_ids": list(self.config.drift_scenario_ids),
                "conditions": list(self.config.drift_conditions),
                "horizons": list(self.config.drift_horizons),
                "seeds": [0],
            }
        )
        detail, _ = self._wait_for_experiment(_string(experiment, "id"))
        self._validate_experiment_contexts(detail, self.profiles[1])
        experiment_record = _mapping(detail.get("experiment"), "experiment")
        comparison = _mapping(
            experiment_record.get("comparison"), "DriftBench comparison"
        )
        paired_observations = _integer(comparison, "paired_observations")
        expected_pairs = (
            len(self.config.drift_scenario_ids)
            * len(self.config.drift_conditions)
            * len(self.config.drift_horizons)
        )
        if paired_observations != expected_pairs:
            raise CanaryFailure(
                "DriftBench comparison did not preserve every paired dimension"
            )
        _sha256(comparison.get("formula_fingerprint"), "drift formula fingerprint")
        routing_pairs = [
            _mapping(pair, "routing checkpoint pair")
            for pair in _list(
                comparison.get("routing_checkpoint_pairs"),
                "routing checkpoint pairs",
            )
        ]
        routed_pairs = [
            pair
            for pair in routing_pairs
            if _is_number(pair.get("selection_overlap"))
            and _is_number(pair.get("routing_mass_js_divergence"))
        ]
        if self.config.require_routing_evidence and not routed_pairs:
            raise CanaryFailure(
                "DriftBench insight cannot be generated without paired real-routing "
                "checkpoint evidence"
            )
        insight = _drift_insight(comparison, routed_pairs)
        self.report["driftbench"] = {
            "experiment_id": experiment_record.get("id"),
            "scenario_ids": list(self.config.drift_scenario_ids),
            "conditions": list(self.config.drift_conditions),
            "horizons": list(self.config.drift_horizons),
            "comparison": comparison,
            "routing_checkpoint_pairs_with_evidence": len(routed_pairs),
            "insight": insight,
        }

    def _verify_cleanup_and_final_state(self) -> None:
        active_job = self.api.get("/api/jobs/active")
        if active_job is not None:
            raise CanaryFailure("acceptance left an app job active")
        runs = [
            _mapping(run, "agent run")
            for run in _list(self.api.get("/api/agent-runs"), "agent runs")
        ]
        daytona_trials = []
        for run in runs:
            if run.get("sandbox_provider_id") != "daytona":
                continue
            for trial in _list(run.get("trials"), "agent trials"):
                trial_map = _mapping(trial, "agent trial")
                daytona_trials.append(
                    {
                        "run_id": run.get("id"),
                        "trial_id": trial_map.get("id"),
                        "sandbox_status": trial_map.get("sandbox_status"),
                    }
                )
        uncleared = [
            trial
            for trial in daytona_trials
            if trial["sandbox_status"] not in {"deleted", "not_created"}
        ]
        if uncleared:
            raise CanaryFailure(
                "application-visible Daytona trials retain undeleted sandboxes"
            )
        runtime = _mapping(self.api.get("/api/runtime/status"), "runtime status")
        current_session = _mapping(
            self.api.get("/api/model-sessions/current"), "current model session"
        )
        if runtime.get("pid") != self.initial_pid:
            raise CanaryFailure("vLLM PID changed by the acceptance workloads")
        if current_session.get("id") != self.session.get("id"):
            raise CanaryFailure("acceptance workloads replaced the warm model session")
        if self._model_load_job_ids() != self.expected_model_load_job_ids:
            raise CanaryFailure("more than one model-load job exists for this run")
        self.report["cleanup"] = {
            "active_app_job": None,
            "application_visible_daytona_trials": daytona_trials,
            "application_visible_daytona_uncleared": [],
            "scope": (
                "The appliance API proves cleanup for archived app-owned trial "
                "records only. Provider-wide Daytona and Runpod inventories require "
                "their control-plane APIs."
            ),
            "final_process_id": runtime.get("pid"),
            "final_model_session_id": current_session.get("id"),
        }

    def _submit_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        for _ in range(4):
            try:
                return _mapping(
                    self.api.post("/api/experiments", payload),
                    "submitted experiment",
                )
            except ApiFailure as error:
                active = error.active_job()
                if active is None:
                    raise
                self.canary.wait_for_job(active)
        raise CanaryFailure("active experiment conflicts did not clear")

    def _wait_for_experiment(
        self,
        experiment_id: str,
        *,
        initial_observations: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        deadline = self.api.monotonic() + self.config.job_timeout_seconds
        observations = [dict(item) for item in initial_observations]
        seen = {
            (item.get("status"), item.get("completed_units"), item.get("total_units"))
            for item in observations
        }
        while True:
            detail = _mapping(
                self.api.get(f"/api/experiments/{experiment_id}"),
                "experiment detail",
            )
            experiment = _mapping(detail.get("experiment"), "experiment")
            observation = {
                "status": experiment.get("status"),
                "completed_units": experiment.get("completed_units"),
                "total_units": experiment.get("total_units"),
                "latest_event_sequence": experiment.get("latest_event_sequence"),
                "observed_at": _utc_now(),
            }
            key = (
                observation["status"],
                observation["completed_units"],
                observation["total_units"],
            )
            if key not in seen:
                observations.append(observation)
                seen.add(key)
            status = experiment.get("status")
            if status in TERMINAL_EXPERIMENT_STATUSES:
                if status != "completed":
                    raise CanaryFailure(
                        f"experiment {experiment_id} ended {status}: "
                        f"{experiment.get('error')}"
                    )
                return detail, observations
            if status not in ACTIVE_JOB_STATUSES:
                raise CanaryFailure(
                    f"experiment {experiment_id} returned unknown status {status!r}"
                )
            if self.api.monotonic() >= deadline:
                raise CanaryFailure(f"experiment {experiment_id} timed out")
            self.api._sleep(self.config.poll_interval_seconds)

    def _reconnect_mid_experiment(self, experiment_id: str) -> dict[str, Any]:
        deadline = self.api.monotonic() + self.config.job_timeout_seconds
        observations = []
        cursor = 0
        while self.api.monotonic() < deadline:
            detail = _mapping(
                self.api.get(f"/api/experiments/{experiment_id}"),
                "experiment detail",
            )
            experiment = _mapping(detail.get("experiment"), "experiment")
            observations.append(
                {
                    "status": experiment.get("status"),
                    "completed_units": experiment.get("completed_units"),
                    "total_units": experiment.get("total_units"),
                    "latest_event_sequence": experiment.get("latest_event_sequence"),
                    "observed_at": _utc_now(),
                }
            )
            if experiment.get("status") in ACTIVE_JOB_STATUSES:
                events = _mapping(
                    self.api.get(
                        f"/api/experiments/{experiment_id}/events?after=0&limit=1000"
                    ),
                    "experiment events",
                )
                cursor = _integer(events, "latest_sequence")
                if cursor > 0:
                    break
            if experiment.get("status") in TERMINAL_EXPERIMENT_STATUSES:
                raise CanaryFailure(
                    "ordinary experiment became terminal before reconnect adoption"
                )
            self.api._sleep(self.config.poll_interval_seconds)
        else:
            raise CanaryFailure("ordinary experiment was not observable while active")

        replay_events: list[dict[str, Any]] = []
        disconnect_cursor = cursor
        replay_cursor = cursor
        with self._reconnect_client() as reconnect:
            reconnect.login()
            adopted = _mapping(
                reconnect.get(f"/api/experiments/{experiment_id}"),
                "reconnected experiment detail",
            )
            adopted_experiment = _mapping(adopted.get("experiment"), "experiment")
            if adopted_experiment.get("id") != experiment_id:
                raise CanaryFailure("new client adopted a different experiment")
            if adopted_experiment.get("status") not in ACTIVE_JOB_STATUSES:
                raise CanaryFailure("new client did not adopt the experiment mid-run")
            while reconnect.monotonic() < deadline:
                page = _mapping(
                    reconnect.get(
                        f"/api/experiments/{experiment_id}/events"
                        f"?after={replay_cursor}&limit=1000"
                    ),
                    "replayed experiment events",
                )
                batch = [
                    _mapping(event, "replayed event")
                    for event in _list(page.get("events"), "replayed events")
                ]
                if batch:
                    batch_sequences = [_integer(event, "sequence") for event in batch]
                    if batch_sequences[0] != replay_cursor + 1:
                        raise CanaryFailure(
                            "reconnected event replay skipped an unseen sequence"
                        )
                    replay_events.extend(batch)
                    replay_cursor = batch_sequences[-1]
                if any(
                    event.get("kind") == "inference_progress"
                    and isinstance(event.get("data"), Mapping)
                    and _is_number(event["data"].get("current_tps"))
                    and float(event["data"]["current_tps"]) > 0
                    for event in replay_events
                ):
                    break
                reconnect._sleep(self.config.poll_interval_seconds)
            if not replay_events:
                raise CanaryFailure("event replay returned no post-disconnect events")
        sequences = [_integer(event, "sequence") for event in replay_events]
        if sequences[0] != disconnect_cursor + 1:
            raise CanaryFailure("event replay skipped the first unseen sequence")
        if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
            raise CanaryFailure("event replay sequence is not contiguous")
        return {
            "new_authenticated_client": True,
            "adopted_while_active": True,
            "disconnect_cursor": disconnect_cursor,
            "replayed_sequences": sequences,
            "replayed_workload_unit_ids": [
                data["workload_unit_id"]
                for event in replay_events
                if event.get("kind") in {"current_unit", "inference_progress"}
                and isinstance((data := event.get("data")), Mapping)
                and isinstance(data.get("workload_unit_id"), str)
            ],
            "replayed_inference_progress": [
                {
                    "run_unit_id": event.get("run_unit_id"),
                    "completion_tokens": data.get("completion_tokens"),
                    "elapsed_ms": data.get("elapsed_ms"),
                    "current_tps": data.get("current_tps"),
                }
                for event in replay_events
                if event.get("kind") == "inference_progress"
                and isinstance((data := event.get("data")), Mapping)
                and _is_number(data.get("current_tps"))
                and float(data["current_tps"]) > 0
            ],
            "inference_progress_visible_after_reconnect": any(
                event.get("kind") == "inference_progress"
                and isinstance(event.get("data"), Mapping)
                and _is_number(event["data"].get("current_tps"))
                and float(event["data"]["current_tps"]) > 0
                for event in replay_events
            ),
            "progress_observations": observations,
            "scope": (
                "A fresh authenticated HTTP client proved durable adoption and "
                "sequence replay. Browser-level hard-refresh proof remains external."
            ),
        }

    def _all_experiment_events(self, experiment_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = _mapping(
                self.api.get(
                    f"/api/experiments/{experiment_id}/events?after={cursor}&limit=1000"
                ),
                "experiment event archive",
            )
            batch = [
                _mapping(event, "experiment event")
                for event in _list(page.get("events"), "experiment events")
            ]
            if batch:
                sequences = [_integer(event, "sequence") for event in batch]
                if sequences[0] != cursor + 1:
                    raise CanaryFailure("experiment event archive has a sequence gap")
                if sequences != list(range(sequences[0], sequences[0] + len(batch))):
                    raise CanaryFailure("experiment event archive is not contiguous")
                events.extend(batch)
                cursor = sequences[-1]
            latest = _integer(page, "latest_sequence")
            if cursor >= latest:
                return events
            if not batch:
                raise CanaryFailure("experiment event archive stopped before latest")

    def _require_workload(
        self, workload_id: str, *, expected_kind: str
    ) -> dict[str, Any]:
        workloads = _list(self.api.get("/api/workloads"), "workloads")
        workload = next(
            (
                _mapping(item, "workload")
                for item in workloads
                if isinstance(item, Mapping) and item.get("id") == workload_id
            ),
            None,
        )
        if workload is None:
            raise CanaryFailure(f"required workload {workload_id!r} is unavailable")
        if workload.get("kind") != expected_kind:
            raise CanaryFailure(f"workload {workload_id!r} has an unexpected kind")
        if workload.get("ready") is not True:
            raise CanaryFailure(
                f"workload {workload_id!r} is blocked: {workload.get('blocked_reason')}"
            )
        return workload

    def _validate_experiment_contexts(
        self,
        detail: Mapping[str, Any],
        candidate_profile: Mapping[str, Any],
    ) -> None:
        experiment = _mapping(detail.get("experiment"), "experiment")
        lanes = [
            _mapping(lane, "experiment lane")
            for lane in _list(experiment.get("lanes"), "experiment lanes")
        ]
        if {lane.get("role") for lane in lanes} != {"baseline", "candidate"}:
            raise CanaryFailure(
                "paired experiment omitted a baseline or candidate lane"
            )
        topology_fingerprints = set()
        for lane in lanes:
            context = _mapping(lane.get("context"), "lane context")
            topology_fingerprints.add(
                _sha256(
                    context.get("topology_fingerprint"), "lane topology fingerprint"
                )
            )
            _sha256(context.get("context_fingerprint"), "lane context fingerprint")
            if context.get("model_id") != self.model.get("id"):
                raise CanaryFailure("paired experiment lane used a different model")
            if lane.get("role") == "baseline":
                if (
                    context.get("profile_id") is not None
                    or context.get("profile_fingerprint") is not None
                ):
                    raise CanaryFailure("baseline lane retained profile provenance")
            elif context.get("profile_id") != candidate_profile.get(
                "id"
            ) or context.get("profile_fingerprint") != candidate_profile.get(
                "profile_fingerprint"
            ):
                raise CanaryFailure("candidate lane lost canonical profile provenance")
        if len(topology_fingerprints) != 1:
            raise CanaryFailure("paired lanes used different topology fingerprints")

    def _trial_sandbox_status(self, result: Mapping[str, Any]) -> str:
        run_id = _string(result, "agent_run_id")
        trial_id = _string(result, "trial_id")
        run = _mapping(self.api.get(f"/api/agent-runs/{run_id}"), "agent run")
        trial = next(
            (
                _mapping(item, "agent trial")
                for item in _list(run.get("trials"), "agent trials")
                if isinstance(item, Mapping) and item.get("id") == trial_id
            ),
            None,
        )
        if trial is None:
            raise CanaryFailure("coding result trial is absent from its agent run")
        status = trial.get("sandbox_status")
        if status != "deleted":
            raise CanaryFailure(f"coding sandbox cleanup ended in {status!r}")
        return status

    def _model_load_job_ids(self) -> set[str]:
        jobs = _list(self.api.get("/api/jobs"), "job archive")
        return {
            _string(_mapping(job, "job"), "id")
            for job in jobs
            if isinstance(job, Mapping) and job.get("kind") == "model_load"
        }


def _profile_payloads(
    topology: Mapping[str, Any], *, run_id: str
) -> list[dict[str, Any]]:
    num_experts = _integer(topology, "num_experts")
    top_k = _integer(topology, "top_k")
    layer_ids = _list(topology.get("routed_layer_ids"), "routed layer IDs")
    if num_experts <= top_k:
        raise CanaryFailure("model topology cannot express a masked profile")
    if not layer_ids or any(type(layer_id) is not int for layer_id in layer_ids):
        raise CanaryFailure("model topology has invalid routed layer IDs")
    profiles = []
    for variant in range(3):
        preferred = round(num_experts * (0.3 + variant * 0.2))
        keep_count = max(top_k, min(num_experts - 1, preferred))
        layers: dict[str, dict[str, list[int]]] = {}
        for layer_index, layer_id in enumerate(layer_ids):
            layer_keep = keep_count
            if layer_index % 3 == 0:
                if keep_count > top_k:
                    layer_keep -= 1
                elif keep_count < num_experts - 1:
                    layer_keep += 1
            seed = int(run_id[:8], 16)
            offset = (seed + variant * 31 + layer_index * 7) % num_experts
            selected = sorted(
                {(offset + expert) % num_experts for expert in range(layer_keep)}
            )
            if len(selected) != layer_keep:
                raise AssertionError("rotating profile unexpectedly duplicated experts")
            layers[str(layer_id)] = {"keep": selected}
        profiles.append({"version": 1, "layers": layers})
    encoded = {
        json.dumps(profile, sort_keys=True, separators=(",", ":"))
        for profile in profiles
    }
    if len(encoded) != 3:
        raise CanaryFailure("model topology cannot express three distinct profiles")
    return profiles


def _published_workload_unit_ids(workload: Mapping[str, Any]) -> list[str]:
    raw_unit_ids = _list(workload.get("unit_ids"), "answer workload unit IDs")
    unit_ids = []
    for value in raw_unit_ids:
        if not isinstance(value, str) or not value:
            raise CanaryFailure("answer workload published an invalid unit ID")
        unit_ids.append(value)
    if not unit_ids:
        raise CanaryFailure("answer cohort publishes no executable unit IDs")
    if len(unit_ids) != len(set(unit_ids)):
        raise CanaryFailure("answer cohort publishes duplicate unit IDs")
    if len(unit_ids) < 2:
        raise CanaryFailure(
            "ordinary live progress requires a genuine multi-item answer cohort"
        )
    return unit_ids


def _answer_seed_repetitions(
    workload_unit_count: int,
    *,
    configured: int | None,
) -> int:
    if workload_unit_count <= 0:
        raise CanaryFailure("answer workload unit count must be positive")
    minimum = math.ceil(MINIMUM_ORDINARY_LANE_UNITS / workload_unit_count)
    if minimum > 20:
        raise CanaryFailure(
            "answer cohort cannot reach 20 lane units within the 20-seed API limit"
        )
    if configured is None:
        return minimum
    if configured < minimum:
        raise CanaryFailure(
            "configured answer seed repetitions would produce fewer than "
            f"{MINIMUM_ORDINARY_LANE_UNITS} lane units; use at least {minimum}"
        )
    if configured > 20:
        raise CanaryFailure("answer seed repetitions cannot exceed 20")
    return configured


def _answer_generation() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_tokens": 64,
        "seed": 0,
        "enable_thinking": False,
    }


def _answer_evaluation_contract() -> dict[str, Any]:
    description = (
        "Score every selected pinned fixture item with its immutable benchmark "
        "adapter and reference data."
    )
    return {
        "version": 1,
        "name": "V2 live acceptance pinned dataset scorer",
        "description": description,
        "criteria": [
            {
                "id": "workload-correctness",
                "label": "Dataset correctness",
                "description": description,
                "kind": "benchmark_default",
                "visibility": "public",
                "required": True,
                "weight": 1.0,
                "case_sensitive": True,
                "strip_whitespace": True,
                "expected": None,
                "pattern": None,
                "numeric_tolerance": 0.0,
                "json_schema": None,
                "verifier_command": None,
                "verifier_timeout_seconds": 300.0,
            }
        ],
        "aggregation": "all_required",
        "pass_threshold": 1.0,
        "judge": None,
        "judge_weight": 0.0,
        "judge_can_override_deterministic_failure": False,
    }


def _answer_execution_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "attempts": 1,
        "concurrency": 1,
        "timeout_seconds": 3600.0,
        "per_item_timeout_seconds": 120.0,
        "max_turns": None,
        "max_commands": None,
        "max_tokens": 1_000_000,
        "max_cost_usd": None,
        "fail_fast": False,
    }


def _validate_answer_cohort_evidence(
    detail: Mapping[str, Any],
    *,
    workload_unit_ids: Sequence[str],
    seeds: Sequence[int],
    generation: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    expected_contract_fingerprint: str,
    expected_policy_fingerprint: str,
    expected_cohort_fingerprint: str,
) -> dict[str, Any]:
    experiment = _mapping(detail.get("experiment"), "answer experiment")
    cohort_fingerprint = _sha256(
        experiment.get("cohort_fingerprint"), "answer cohort fingerprint"
    )
    if cohort_fingerprint != expected_cohort_fingerprint:
        raise CanaryFailure("answer cohort fingerprint does not match its request")
    persisted_generation = _mapping(
        experiment.get("generation"), "persisted answer generation"
    )
    if persisted_generation != generation:
        raise CanaryFailure("answer experiment changed the generation contract")
    persisted_contract = _mapping(
        experiment.get("evaluation_contract"), "persisted evaluation contract"
    )
    contract_fingerprint = _sha256(
        persisted_contract.get("fingerprint"), "evaluation contract fingerprint"
    )
    if contract_fingerprint != expected_contract_fingerprint:
        raise CanaryFailure("persisted evaluation contract fingerprint changed")
    if _without_fingerprint(persisted_contract) != evaluation_contract:
        raise CanaryFailure("answer experiment changed the evaluation contract")
    persisted_policy = _mapping(
        experiment.get("execution_policy"), "persisted execution policy"
    )
    policy_fingerprint = _sha256(
        persisted_policy.get("fingerprint"), "execution policy fingerprint"
    )
    if policy_fingerprint != expected_policy_fingerprint:
        raise CanaryFailure("persisted execution policy fingerprint changed")
    if _without_fingerprint(persisted_policy) != execution_policy:
        raise CanaryFailure("answer experiment changed the execution policy")

    units = [
        _mapping(unit, "answer run unit")
        for unit in _list(detail.get("units"), "answer run units")
    ]
    expected_dimensions = {
        (workload_unit_id, seed)
        for workload_unit_id in workload_unit_ids
        for seed in seeds
    }
    observed_dimensions = {
        (unit.get("workload_unit_id"), unit.get("seed")) for unit in units
    }
    if observed_dimensions != expected_dimensions:
        raise CanaryFailure("persisted answer units changed the selected cohort")
    run_unit_ids = [_string(unit, "id") for unit in units]
    if len(run_unit_ids) != len(set(run_unit_ids)):
        raise CanaryFailure("persisted answer run-unit IDs are not unique")
    if len(units) != len(expected_dimensions):
        raise CanaryFailure("answer cohort persisted duplicate item/seed dimensions")
    if _integer(experiment, "total_units") != len(expected_dimensions):
        raise CanaryFailure("answer experiment total does not match its lane cohort")
    for unit in units:
        result = _mapping(unit.get("result"), "answer unit result")
        workload_unit_id = _string(unit, "workload_unit_id")
        if result.get("workload_unit_id") != workload_unit_id:
            raise CanaryFailure("answer result changed its workload item identity")
        if result.get("evaluation_contract_fingerprint") != (
            expected_contract_fingerprint
        ):
            raise CanaryFailure("answer result lost evaluation contract provenance")
        if result.get("execution_policy_fingerprint") != expected_policy_fingerprint:
            raise CanaryFailure("answer result lost execution policy provenance")
        unit_generation = _mapping(result.get("generation"), "answer unit generation")
        expected_generation = {**generation, "seed": unit.get("seed")}
        if unit_generation != expected_generation:
            raise CanaryFailure("answer unit generation does not match its seed")
    return {
        "cohort_fingerprint_verified": True,
        "generation_verified": True,
        "evaluation_contract_fingerprint": contract_fingerprint,
        "execution_policy_fingerprint": policy_fingerprint,
        "persisted_run_unit_count": len(units),
        "unique_run_unit_ids": len(set(run_unit_ids)),
        "distinct_workload_unit_ids": len(
            {unit.get("workload_unit_id") for unit in units}
        ),
        "distinct_seeds": len({unit.get("seed") for unit in units}),
    }


def _validate_answer_progress_events(
    events: Sequence[Mapping[str, Any]],
    *,
    workload_unit_ids: Sequence[str],
    seeds: Sequence[int],
    expected_cohort_fingerprint: str,
) -> dict[str, Any]:
    queued = next((event for event in events if event.get("kind") == "queued"), None)
    if queued is None:
        raise CanaryFailure("answer event archive omitted its queued event")
    queued_data = _mapping(queued.get("data"), "answer queued event data")
    if queued_data.get("cohort_fingerprint") != expected_cohort_fingerprint:
        raise CanaryFailure("queued event lost the answer cohort fingerprint")
    if queued_data.get("workload_unit_count") != len(workload_unit_ids):
        raise CanaryFailure("queued event changed the answer workload-unit count")
    current = [event for event in events if event.get("kind") == "current_unit"]
    dimensions = []
    run_unit_ids = []
    for event in current:
        data = _mapping(event.get("data"), "answer progress event data")
        workload_unit_id = _string(data, "workload_unit_id")
        seed = _integer(data, "seed")
        dimensions.append((workload_unit_id, seed))
        run_unit_ids.append(_string(event, "run_unit_id"))
    expected_dimensions = {
        (workload_unit_id, seed)
        for workload_unit_id in workload_unit_ids
        for seed in seeds
    }
    if set(dimensions) != expected_dimensions or len(dimensions) != len(
        expected_dimensions
    ):
        raise CanaryFailure("answer progress events changed the item/seed cohort")
    if len(run_unit_ids) != len(set(run_unit_ids)):
        raise CanaryFailure("answer progress events reused a run-unit ID")
    inference_samples = []
    for event in events:
        if event.get("kind") != "inference_progress":
            continue
        data = _mapping(event.get("data"), "answer inference progress data")
        current_tps = data.get("current_tps")
        completion_tokens = data.get("completion_tokens")
        elapsed_ms = data.get("elapsed_ms")
        if (
            _is_number(current_tps)
            and float(current_tps) > 0
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens > 0
            and _is_number(elapsed_ms)
            and float(elapsed_ms) >= 0
        ):
            inference_samples.append(
                {
                    "run_unit_id": _string(event, "run_unit_id"),
                    "completion_tokens": completion_tokens,
                    "elapsed_ms": float(elapsed_ms),
                    "current_tps": float(current_tps),
                }
            )
    if not inference_samples:
        raise CanaryFailure(
            "answer event archive omitted durable in-flight current TPS"
        )
    return {
        "current_unit_event_count": len(current),
        "distinct_workload_unit_ids": sorted(
            {workload_unit_id for workload_unit_id, _seed in dimensions}
        ),
        "seeds": sorted({seed for _workload_unit_id, seed in dimensions}),
        "unique_event_run_unit_ids": len(set(run_unit_ids)),
        "cohort_fingerprint_in_queued_event": True,
        "current_tps_sample_count": len(inference_samples),
        "first_current_tps_sample": inference_samples[0],
    }


def _without_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "fingerprint"}


def _contract_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                key: item
                for key, item in value.items()
                if key not in {"fingerprint", "request_hash"}
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _canonical_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _validate_activation(
    receipt: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_profile_id: str | None,
    expected_profile_fingerprint: str | None,
    expected_pid: int | None,
) -> None:
    if receipt.get("weights_reloaded") is not False:
        raise CanaryFailure("context activation reported a weight reload")
    if receipt.get("process_id") != expected_pid:
        raise CanaryFailure("context activation receipt changed process identity")
    if receipt.get("new_context_id") != current.get("context_id"):
        raise CanaryFailure("activation receipt and current context IDs differ")
    if receipt.get("new_context_fingerprint") != current.get("context_fingerprint"):
        raise CanaryFailure("activation receipt and context fingerprints differ")
    if receipt.get("topology_fingerprint") != current.get("topology_fingerprint"):
        raise CanaryFailure("activation receipt and topology fingerprints differ")
    if current.get("model_id") != expected_model_id:
        raise CanaryFailure("activated context uses a different model")
    if current.get("profile_id") != expected_profile_id:
        raise CanaryFailure("activated context uses a different profile")
    if current.get("profile_fingerprint") != expected_profile_fingerprint:
        raise CanaryFailure("activated context uses a different profile fingerprint")
    _sha256(current.get("context_fingerprint"), "context fingerprint")
    _sha256(current.get("topology_fingerprint"), "topology fingerprint")


def _activation_statistics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise CanaryFailure("activation benchmark produced no samples")
    metrics = {}
    for key in ("receipt_duration_ms", "wall_duration_ms"):
        values = [
            _nonnegative_number(sample.get(key), f"activation {key}")
            for sample in samples
        ]
        metrics[key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
        }
    by_target = {}
    for target in sorted({_string(sample, "target") for sample in samples}):
        target_samples = [
            sample for sample in samples if sample.get("target") == target
        ]
        values = [
            _nonnegative_number(sample.get("wall_duration_ms"), "wall duration")
            for sample in target_samples
        ]
        by_target[target] = {
            "count": len(values),
            "p50_wall_ms": _nearest_rank(values, 0.50),
            "p95_wall_ms": _nearest_rank(values, 0.95),
        }
    metrics["by_target"] = by_target
    return metrics


def _load_phase_evidence(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        _mapping(record, "model-load phase")
        for record in _list(job.get("phase_history"), "model-load phase history")
    ]
    if not records:
        raise CanaryFailure("model-load job omitted durable phase history")
    if job.get("status") == "completed" and any(
        record.get("status") == "active" for record in records
    ):
        raise CanaryFailure("completed model-load job retains an active phase")
    evidence = []
    for record in records:
        phase = _string(record, "phase")
        status = _string(record, "status")
        if status not in {
            "completed",
            "failed",
            "cancelled",
            "active",
            "unavailable",
        }:
            raise CanaryFailure(f"model-load phase {phase!r} has invalid status")
        observability = record.get("observability", "observed")
        if observability not in {"observed", "unavailable"}:
            raise CanaryFailure(f"model-load phase {phase!r} has invalid observability")
        evidence.append(
            {
                "phase": phase,
                "status": status,
                "observability": observability,
                "source": record.get("source", "application"),
                "started_at": record.get("started_at"),
                "completed_at": record.get("completed_at"),
                "duration_ms": _timestamp_delta_ms(
                    record.get("started_at"), record.get("completed_at")
                ),
                "detail": record.get("detail"),
                "bytes_current": record.get("bytes_current"),
                "bytes_total": record.get("bytes_total"),
                "files_current": record.get("files_current"),
                "files_total": record.get("files_total"),
                "failure_code": record.get("failure_code"),
                "recovery_action": record.get("recovery_action"),
                "diagnostics": record.get("diagnostics"),
            }
        )
    return evidence


def _drift_insight(
    comparison: Mapping[str, Any], routed_pairs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    paired_delta = _required_number(
        comparison.get("baseline_to_mask_paired_drift_delta"),
        "paired drift delta",
    )
    overlap = _required_number(
        comparison.get("mean_routing_selection_overlap"),
        "mean routing selection overlap",
    )
    js_divergence = _required_number(
        comparison.get("mean_routing_mass_js_divergence"),
        "mean routing mass JS divergence",
    )
    excess_penalty = _required_number(
        comparison.get("excess_compounding_penalty"),
        "excess compounding penalty",
    )
    if abs(paired_delta) < 1e-9:
        behavior = "no aggregate state-fidelity change"
    elif paired_delta < 0:
        behavior = "lower masked state fidelity"
    else:
        behavior = "higher masked state fidelity"
    direction = (
        "more compounding drift"
        if excess_penalty > 1e-9
        else (
            "less compounding drift"
            if excess_penalty < -1e-9
            else "no excess compounding drift"
        )
    )
    summary = (
        f"Across {_integer(comparison, 'paired_observations')} aligned runs, the "
        f"masked lane showed {behavior} (paired AUC delta {paired_delta:.6f}) and "
        f"{direction} ({excess_penalty:.6f}). Routing evidence covered "
        f"{len(routed_pairs)} paired checkpoints with mean selection overlap "
        f"{overlap:.6f} and mean mass JS divergence {js_divergence:.6f}."
    )
    return {
        "summary": summary,
        "baseline_to_mask_paired_drift_delta": paired_delta,
        "excess_compounding_penalty": excess_penalty,
        "mean_routing_selection_overlap": overlap,
        "mean_routing_mass_js_divergence": js_divergence,
        "routing_checkpoint_pairs": len(routed_pairs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run V2 live acceptance against one authenticated deployed appliance. "
            "The default fake coding provider is diagnostic-only; pass "
            "--coding-provider daytona for a release-qualifying model-authored run."
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
        help="Access token; never written to output.",
    )
    parser.add_argument("--output", type=Path, default=Path("v2-live-acceptance.json"))
    parser.add_argument("--model-id")
    parser.add_argument("--switch-rounds", type=int, default=4)
    parser.add_argument("--activation-p95-limit-ms", type=float, default=2000)
    parser.add_argument("--answer-workload-id", default=DEFAULT_ANSWER_WORKLOAD)
    parser.add_argument(
        "--answer-seed-repetitions",
        "--answer-seeds",
        type=int,
        default=None,
        help=(
            "Deterministic repetitions of every published answer-cohort item. "
            "Defaults to the minimum producing at least 20 lane units (2 for "
            "the 12-item fixture cohort)."
        ),
    )
    parser.add_argument("--coding-workload-id", default=DEFAULT_CODING_WORKLOAD)
    parser.add_argument(
        "--coding-provider",
        choices=("fake", "daytona"),
        default="fake",
        help=(
            "Fake is diagnostic-only and makes acceptance fail; use Daytona only "
            "for an explicitly authorized paid release run."
        ),
    )
    parser.add_argument(
        "--drift-scenario",
        action="append",
        dest="drift_scenarios",
        help="Repeat to select DriftBench scenarios.",
    )
    parser.add_argument("--job-timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument(
        "--allow-missing-routing",
        action="store_true",
        help=(
            "Diagnostic-only: do not gate on real routing artifacts. The resulting "
            "record cannot qualify a live release insight."
        ),
    )
    return parser


def config_from_args(arguments: argparse.Namespace) -> V2AcceptanceConfig:
    if not arguments.base_url:
        raise CanaryFailure("supply --base-url or set MOE_TOOLS_CANARY_BASE_URL")
    if arguments.switch_rounds < 2:
        raise CanaryFailure("--switch-rounds must be at least 2")
    if arguments.activation_p95_limit_ms <= 0:
        raise CanaryFailure("--activation-p95-limit-ms must be positive")
    if arguments.answer_seed_repetitions is not None and not (
        1 <= arguments.answer_seed_repetitions <= 20
    ):
        raise CanaryFailure("--answer-seed-repetitions must be between 1 and 20")
    if arguments.job_timeout <= 0:
        raise CanaryFailure("--job-timeout must be positive")
    if arguments.poll_interval <= 0:
        raise CanaryFailure("--poll-interval must be positive")
    if arguments.request_timeout <= 0:
        raise CanaryFailure("--request-timeout must be positive")
    scenarios = tuple(arguments.drift_scenarios or DEFAULT_DRIFT_SCENARIOS)
    if len(scenarios) != len(set(scenarios)):
        raise CanaryFailure("--drift-scenario values must be unique")
    return V2AcceptanceConfig(
        base_url=_normalize_base_url(arguments.base_url),
        token=arguments.token,
        output_path=arguments.output,
        model_id=arguments.model_id,
        switch_rounds=arguments.switch_rounds,
        activation_p95_limit_ms=arguments.activation_p95_limit_ms,
        answer_workload_id=arguments.answer_workload_id,
        answer_seed_repetitions=arguments.answer_seed_repetitions,
        coding_workload_id=arguments.coding_workload_id,
        coding_provider_id=arguments.coding_provider,
        drift_scenario_ids=scenarios,
        job_timeout_seconds=arguments.job_timeout,
        poll_interval_seconds=arguments.poll_interval,
        request_timeout_seconds=arguments.request_timeout,
        require_routing_evidence=not arguments.allow_missing_routing,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = config_from_args(build_parser().parse_args(argv))
    except CanaryFailure as error:
        print(f"V2 acceptance configuration failed: {error}", file=sys.stderr)
        return 2
    reporter = Reporter(
        total_steps=9,
        secrets=(config.token or "",),
    )
    runner: V2LiveAcceptance | None = None
    try:
        with ApiClient(config.canary_config()) as api:
            runner = V2LiveAcceptance(config, api, reporter)
            report = runner.run()
    except (CanaryFailure, httpx.HTTPError, ValueError, TypeError) as error:
        report = (
            runner.report
            if runner is not None
            else {
                "schema_version": SCHEMA_VERSION,
                "started_at": _utc_now(),
                "app_acceptance_passed": False,
            }
        )
        report["failure"] = {
            "type": error.__class__.__name__,
            "message": _redact(str(error), (config.token or "",)),
        }
        report["finished_at"] = _utc_now()
        _write_report(config.output_path, report)
        reporter.info(f"V2 live acceptance FAILED; evidence: {config.output_path}")
        return 1
    _write_report(config.output_path, report)
    if report["app_acceptance_passed"] is not True:
        reporter.info(
            "V2 appliance workflow completed but release gates are INCOMPLETE; "
            f"evidence: {config.output_path}"
        )
        return 1
    reporter.info(
        "V2 in-appliance acceptance PASSED; external release gates remain "
        f"explicit in {config.output_path}"
    )
    return 0


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank inputs are invalid")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _timestamp_delta_ms(started_at: object, completed_at: object) -> float | None:
    if not isinstance(started_at, str) or not isinstance(completed_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (completed - started).total_seconds() * 1000)


def _positive_integer(value: Mapping[str, Any], key: str) -> int:
    result = _integer(value, key)
    if result <= 0:
        raise CanaryFailure(f"{key} must be positive")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CanaryFailure(f"{label} must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise CanaryFailure(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _required_number(value: object, label: str) -> float:
    if not _is_number(value):
        raise CanaryFailure(f"{label} is missing from the insight evidence")
    return float(value)


def _nonnegative_number(value: object, label: str) -> float:
    result = _required_number(value, label)
    if result < 0:
        raise CanaryFailure(f"{label} cannot be negative")
    return result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _redact(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


if __name__ == "__main__":
    raise SystemExit(main())
