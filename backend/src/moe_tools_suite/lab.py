from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkRun,
    CreateModelSessionRequest,
    ModelRegistryEntry,
    ModelSession,
    ModelState,
    ModelTopology,
    ProfileProposal,
    ProfileProposalRequest,
    ProfileValidation,
    RoutingSummary,
    RunItemResult,
    RunRequest,
)
from .profiles import propose_fixed_budget_profile, validate_profile
from .runtime import MockModelRuntime, ModelRuntime, VllmRuntime
from .settings import Settings
from .telemetry import aggregate_routing

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
FIXTURE_BENCHMARK_ID = "fixture-arithmetic"


@dataclass
class RunArtifacts:
    run: BenchmarkRun
    routing: RoutingSummary


class ResearchLab:
    """Application service for the first model-to-profile vertical slice."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.topology = ModelTopology(
            num_layers=40,
            num_experts=256,
            top_k=8,
            routed_layer_ids=list(range(40)),
        )
        self.models = [
            ModelRegistryEntry(
                id=MODEL_ID,
                display_name="Qwen 3.6 35B A3B FP8",
                enabled=True,
                topology=self.topology,
                notes=(
                    "Validated target for one RTX PRO 6000 Blackwell. "
                    "Mock mode synthesizes its 40 × 256 routing topology."
                ),
            )
        ]
        self.items = _fixture_items()
        self.session: ModelSession | None = None
        self.runs: dict[str, RunArtifacts] = {}
        self.runtime: ModelRuntime = self._create_runtime()

    def _create_runtime(self) -> ModelRuntime:
        if self.settings.mode == "vllm":
            return VllmRuntime(
                base_url=self.settings.vllm_base_url,
                model_id=MODEL_ID,
                topology=self.topology,
            )
        return MockModelRuntime(self.topology)

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        return [
            BenchmarkInfo(
                id=FIXTURE_BENCHMARK_ID,
                name="Arithmetic routing fixture",
                description=(
                    "Deterministic GPU-free prompts used to exercise scoring, "
                    "routing aggregation, and expert-profile creation."
                ),
                item_count=len(self.items),
                categories=sorted({item.category for item in self.items}),
            )
        ]

    def create_model_session(
        self, request: CreateModelSessionRequest
    ) -> ModelSession:
        if request.model_id != MODEL_ID:
            raise ValueError(f"unsupported model {request.model_id!r}")
        if request.profile is not None:
            validation = validate_profile(request.profile, self.topology)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
        self.session = ModelSession(
            id=str(uuid4()),
            model_id=request.model_id,
            state=ModelState.READY,
            mode=self.settings.mode,
            profile=request.profile,
        )
        return self.session

    async def run_benchmark(self, request: RunRequest) -> BenchmarkRun:
        if self.session is None or self.session.state is not ModelState.READY:
            raise RuntimeError("load a model before starting a benchmark")
        if request.benchmark_id != FIXTURE_BENCHMARK_ID:
            raise ValueError(f"unknown benchmark {request.benchmark_id!r}")
        requested_ids = set(request.item_ids or [item.id for item in self.items])
        selected = [item for item in self.items if item.id in requested_ids]
        missing = requested_ids - {item.id for item in selected}
        if missing:
            raise ValueError(f"unknown benchmark items: {sorted(missing)}")
        if not selected:
            raise ValueError("select at least one benchmark item")

        run_id = str(uuid4())
        counts = np.zeros(
            (self.topology.num_layers, self.topology.num_experts), dtype=np.int64
        )
        mass = np.zeros(counts.shape, dtype=np.float64)
        total_slots = 0
        results: list[RunItemResult] = []
        for item in selected:
            started = time.perf_counter()
            completion = await self.runtime.complete(
                item.prompt,
                request_key=item.id,
                profile=self.session.profile,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            aggregate = aggregate_routing(completion.routing, self.topology)
            counts += aggregate.selection_counts
            mass += aggregate.routing_mass
            total_slots += aggregate.total_routed_slots
            output = completion.content.strip()
            results.append(
                RunItemResult(
                    item_id=item.id,
                    prompt=item.prompt,
                    expected=item.expected,
                    output=output,
                    passed=output == item.expected,
                    latency_ms=latency_ms,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                )
            )

        passed = sum(item.passed for item in results)
        run = BenchmarkRun(
            id=run_id,
            benchmark_id=request.benchmark_id,
            model_session_id=self.session.id,
            status="completed",
            score=passed / len(results),
            completed_items=len(results),
            total_items=len(results),
            items=results,
        )
        routing = RoutingSummary(
            run_id=run_id,
            layer_ids=self.topology.routed_layer_ids,
            selection_counts=counts.tolist(),
            routing_mass=mass.tolist(),
            total_routed_slots=total_slots,
        )
        self.runs[run_id] = RunArtifacts(run=run, routing=routing)
        return run

    def validate_profile(self, profile) -> ProfileValidation:
        return validate_profile(profile, self.topology)

    def propose_profile(
        self, request: ProfileProposalRequest
    ) -> ProfileProposal:
        artifacts = self.runs.get(request.run_id)
        if artifacts is None:
            raise KeyError(request.run_id)
        return propose_fixed_budget_profile(
            artifacts.routing,
            self.topology,
            request.keep_per_layer,
            request.metric,
        )


def _fixture_items() -> list[BenchmarkItem]:
    expressions = [
        ("03", "12 + 7", "19", "addition"),
        ("04", "42 - 19", "23", "subtraction"),
        ("05", "9 * 8", "72", "multiplication"),
        ("06", "31 + 46", "77", "addition"),
        ("07", "100 - 37", "63", "subtraction"),
        ("08", "13 * 6", "78", "multiplication"),
        ("09", "-4 + 15", "11", "addition"),
        ("10", "81 - 99", "-18", "subtraction"),
        ("11", "17 * 5", "85", "multiplication"),
        ("12", "128 + 64", "192", "addition"),
        ("13", "72 - 18", "54", "subtraction"),
        ("14", "21 * 4", "84", "multiplication"),
    ]
    return [
        BenchmarkItem(
            id=f"arith-{item_id}",
            prompt=f"Return only the integer result of {expression}.",
            expected=expected,
            category=category,
        )
        for item_id, expression, expected, category in expressions
    ]
