from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from moe_tools_suite import __version__

ROOT = Path(__file__).parents[2]
EXPECTED_IMAGE_TAG = "ghcr.io/sampazdan/vllm-moe-tools-suite:0.3.0-rc.1"
EXPECTED_VLLM_ENV = {
    "RUNPOD_ENABLE_REQUEST_METRICS": "1",
    "RUNPOD_REASONING_PARSER": "qwen3",
    "RUNPOD_VLLM_GPU_MEMORY_UTILIZATION": "0.90",
    "RUNPOD_VLLM_HOST": "127.0.0.1",
    "RUNPOD_VLLM_MAX_MODEL_LEN": "4096",
    "RUNPOD_VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
    "RUNPOD_VLLM_MAX_NUM_SEQS": "8",
    "RUNPOD_VLLM_MODEL": "Qwen/Qwen3.6-35B-A3B-FP8",
    "RUNPOD_VLLM_PORT": "8000",
}


@pytest.mark.parametrize(
    ("filename", "mode"),
    [
        ("runpod-template.example.json", "mock"),
        ("runpod-agentic-template.example.json", "vllm"),
    ],
)
def test_runpod_templates_preserve_the_secure_pod_contract(
    filename: str,
    mode: str,
) -> None:
    template = json.loads((ROOT / "deploy" / filename).read_text())
    environment = template["env"]

    image = template["imageName"]
    assert image == EXPECTED_IMAGE_TAG or re.fullmatch(
        rf"{re.escape(EXPECTED_IMAGE_TAG)}@sha256:[0-9a-f]{{64}}",
        image,
    )
    assert template["dockerEntrypoint"] == []
    assert template["dockerStartCmd"] == []
    assert template["isPublic"] is False
    assert template["isServerless"] is False
    assert template["ports"] == ["8080/http", "22/tcp"]
    assert template["volumeMountPath"] == "/workspace"
    assert environment["MOE_TOOLS_MODE"] == mode
    assert environment["MOE_TOOLS_REQUIRE_AUTH"] == "1"
    assert environment["MOE_TOOLS_COOKIE_SECURE"] == "1"
    assert environment["MOE_TOOLS_AUTH_TOKEN"] == (
        "{{ RUNPOD_SECRET_moe_tools_auth_token }}"
    )
    assert environment["MOE_TOOLS_APP_STARTUP_TIMEOUT_SECONDS"] == "300"
    for key, value in EXPECTED_VLLM_ENV.items():
        assert environment[key] == value


def test_agentic_template_wires_only_secret_references_for_paid_providers() -> None:
    template = json.loads(
        (ROOT / "deploy" / "runpod-agentic-template.example.json").read_text()
    )

    assert template["env"]["MOE_TOOLS_DAYTONA_API_KEY"] == (
        "{{ RUNPOD_SECRET_daytona_api_key }}"
    )
    assert template["env"]["MOE_TOOLS_ANTHROPIC_API_KEY"] == (
        "{{ RUNPOD_SECRET_MOE_TOOLS_ANTHROPIC_API_KEY }}"
    )


def test_runpod_dockerfile_requires_a_pin_and_inherits_the_base_startup() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.runpod").read_text()
    instructions = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "ARG VLLM_SOURCE_REF" in instructions
    assert not any(line.startswith("ARG VLLM_SOURCE_REF=") for line in instructions)
    assert "'^[0-9a-f]{40}$'" in dockerfile
    assert (
        "node:24.6.0-bookworm-slim@sha256:"
        "527e3f1e925c5727a98b17575e68d0bb0b577965bf3b60330790bfcf8a0b3035"
    ) in dockerfile
    assert any(
        line.startswith("FROM --platform=linux/amd64 runpod/pytorch:")
        and "@sha256:" in line
        for line in instructions
    )
    assert not any(
        line.startswith(("CMD ", "ENTRYPOINT ")) for line in dockerfile.splitlines()
    )
    assert any(line.startswith("HEALTHCHECK ") for line in instructions)
    assert any("/readyz" in line for line in instructions)


def test_release_versions_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    frontend = json.loads((ROOT / "frontend" / "package.json").read_text())

    assert project["project"]["version"] == __version__ == "0.3.0rc1"
    assert frontend["version"] == "0.3.0-rc.1"
