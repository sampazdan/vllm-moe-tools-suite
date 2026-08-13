from __future__ import annotations

from .domain import ModelRegistryEntry, ModelRuntimeRecipe, ModelTopology

QUALIFIED_MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"

_QWEN_A3B_TOPOLOGY = ModelTopology(
    num_layers=40,
    num_experts=256,
    top_k=8,
    routed_layer_ids=list(range(40)),
)
_QWEN_A10B_TOPOLOGY = ModelTopology(
    num_layers=48,
    num_experts=256,
    top_k=8,
    routed_layer_ids=list(range(48)),
)
_GPT_OSS_TOPOLOGY = ModelTopology(
    num_layers=36,
    num_experts=128,
    top_k=4,
    routed_layer_ids=list(range(36)),
)


def model_registry() -> list[ModelRegistryEntry]:
    """Return the immutable model manifest exposed by the workbench."""

    common_unqualified = {
        "enabled": False,
        "masking_status": "experimental",
        "routing_telemetry_status": "experimental",
        "hot_switch_status": "unsupported",
        "qualification_status": "manifest_only",
        "failure_reason": (
            "Manifest only: checkpoint metadata is pinned where listed, but live "
            "router discovery, telemetry, profile activation, and masked inference "
            "have not completed qualification."
        ),
    }
    return [
        ModelRegistryEntry(
            id=QUALIFIED_MODEL_ID,
            display_name="Qwen 3.6 35B A3B FP8",
            enabled=True,
            revision="95a723d08a9490559dae23d0cff1d9466213d989",
            topology=_QWEN_A3B_TOPOLOGY,
            architecture="Qwen3_5Moe",
            dtype="float8",
            tensor_parallel_size=1,
            context_defaults={"max_model_len": 4096, "enable_thinking": False},
            reasoning_parser="qwen3",
            recommended_hardware="1 × RTX PRO 6000 Blackwell 96 GB",
            minimum_memory_gib=90,
            runtime_recipe=ModelRuntimeRecipe(
                revision="95a723d08a9490559dae23d0cff1d9466213d989",
                tensor_parallel_size=1,
                max_model_len=4096,
                reasoning_parser="qwen3",
            ),
            masking_status="qualified",
            routing_telemetry_status="qualified",
            hot_switch_status="experimental",
            qualification_status="qualified",
            last_live_evidence="V1 evidence · 2026-08-11 · 0.3.0-rc.1",
            notes=(
                "V1 live-qualified for masking and routing telemetry. Hot switching "
                "is exposed as experimental until V2 live acceptance completes."
            ),
        ),
        ModelRegistryEntry(
            id="Qwen/Qwen3.5-122B-A10B",
            display_name="Qwen 3.5 122B A10B",
            revision="dc4d348443bc740c68e2d77492492c11606384d5",
            topology=_QWEN_A10B_TOPOLOGY,
            architecture="Qwen3_5Moe",
            dtype="bfloat16",
            tensor_parallel_size=8,
            recommended_hardware="Multi-GPU; qualify against the official recipe",
            minimum_memory_gib=240,
            notes="Primary large-model qualification target; not yet runnable.",
            **common_unqualified,
        ),
        ModelRegistryEntry(
            id="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
            display_name="Qwen 3.5 122B A10B GPTQ Int4",
            revision="30cd92cba9707a9aba09d1e490ed4b66b78e9606",
            topology=_QWEN_A10B_TOPOLOGY,
            architecture="Qwen3_5Moe",
            dtype="int4",
            quantization="gptq-moe_wna16",
            tensor_parallel_size=2,
            recommended_hardware="Qualification candidate on high-memory GPUs",
            minimum_memory_gib=80,
            notes="The moe_wna16 router path must prove masking compatibility.",
            **common_unqualified,
        ),
        ModelRegistryEntry(
            id="openai/gpt-oss-120b",
            display_name="gpt-oss 120B",
            revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            topology=_GPT_OSS_TOPOLOGY,
            architecture="GptOssForCausalLM",
            dtype="mxfp4",
            quantization="mxfp4",
            tensor_parallel_size=1,
            recommended_hardware="1 × 80 GB GPU or larger",
            minimum_memory_gib=80,
            notes="Single-80GB qualification candidate; not yet runnable.",
            **common_unqualified,
        ),
        ModelRegistryEntry(
            id="zai-org/GLM-4.5-Air",
            display_name="GLM 4.5 Air",
            revision=None,
            topology=None,
            architecture="Glm4MoeForCausalLM",
            dtype="bfloat16",
            tensor_parallel_size=4,
            recommended_hardware="Multi-GPU experimental target",
            minimum_memory_gib=160,
            notes="Secondary architecture experiment; not yet runnable.",
            **common_unqualified,
        ),
    ]


def qualified_topology() -> ModelTopology:
    return _QWEN_A3B_TOPOLOGY.model_copy(deep=True)
