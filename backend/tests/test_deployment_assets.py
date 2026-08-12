from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import tomllib
from pathlib import Path

import pytest
from moe_tools_suite import __version__

ROOT = Path(__file__).parents[2]
EXPECTED_IMAGE_TAG = "ghcr.io/sampazdan/vllm-moe-tools-suite:0.3.0-rc.1"
EXPECTED_V2_IMAGE_TAG = "ghcr.io/sampazdan/vllm-moe-tools-suite:0.4.0-rc.1"
V2_IMAGE_PLACEHOLDER = (
    "ghcr.io/sampazdan/vllm-moe-tools-suite@sha256:REPLACE_WITH_PUBLISHED_V2_DIGEST"
)
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
        ("runpod-v2-template.example.json", "vllm"),
    ],
)
def test_runpod_templates_preserve_the_secure_pod_contract(
    filename: str,
    mode: str,
) -> None:
    template = json.loads((ROOT / "deploy" / filename).read_text())
    environment = template["env"]

    image = template["imageName"]
    if filename == "runpod-v2-template.example.json":
        assert image == V2_IMAGE_PLACEHOLDER or re.fullmatch(
            r"ghcr\.io/sampazdan/vllm-moe-tools-suite@sha256:[0-9a-f]{64}",
            image,
        )
    else:
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


def test_v2_template_is_isolated_from_v1_data_and_uses_secret_references() -> None:
    template = json.loads(
        (ROOT / "deploy" / "runpod-v2-template.example.json").read_text()
    )
    environment = template["env"]

    assert template["imageName"] == V2_IMAGE_PLACEHOLDER or re.fullmatch(
        r"ghcr\.io/sampazdan/vllm-moe-tools-suite@sha256:[0-9a-f]{64}",
        template["imageName"],
    )
    assert environment["MOE_TOOLS_DATA_DIR"] == "/workspace/moe-tools-v2"
    assert environment["RUNPOD_VLLM_REVISION"] == (
        "95a723d08a9490559dae23d0cff1d9466213d989"
    )
    assert environment["HF_TOKEN"] == "{{ RUNPOD_SECRET_HF_TOKEN }}"
    assert environment["MOE_TOOLS_DAYTONA_API_KEY"] == (
        "{{ RUNPOD_SECRET_daytona_api_key }}"
    )
    assert environment["MOE_TOOLS_ANTHROPIC_API_KEY"] == (
        "{{ RUNPOD_SECRET_MOE_TOOLS_ANTHROPIC_API_KEY }}"
    )


def test_v2_template_renderer_rejects_mutable_image_tags() -> None:
    environment = os.environ | {
        "MOE_TOOLS_IMAGE": EXPECTED_V2_IMAGE_TAG,
        "MOE_TOOLS_TEMPLATE_DRY_RUN": "1",
        "MOE_TOOLS_TEMPLATE_MODE": "v2",
    }

    result = subprocess.run(
        [str(ROOT / "scripts" / "create_runpod_template.sh")],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "immutable sha256 digest" in result.stderr


def test_runpod_dockerfile_requires_a_pin_and_inherits_the_base_startup() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.runpod").read_text()
    instructions = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "ARG APP_SOURCE_REF" in instructions
    assert "ARG VLLM_SOURCE_REF" in instructions
    assert not any(line.startswith("ARG APP_SOURCE_REF=") for line in instructions)
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
    assert 'org.opencontainers.image.revision="${APP_SOURCE_REF}"' in dockerfile
    assert 'io.moe-atelier.vllm-source-revision="${VLLM_SOURCE_REF}"' in dockerfile


def test_runpod_image_contains_fail_closed_opt_in_v2_acceptance() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.runpod").read_text()
    pre_start = (ROOT / "docker" / "runpod" / "pre_start.sh").read_text()
    runner_path = ROOT / "docker" / "runpod" / "run_v2_live_acceptance_on_start.sh"
    runner = runner_path.read_text()

    assert "scripts/v2_live_acceptance.py" in dockerfile
    assert "scripts/daytona_inventory.py" in dockerfile
    assert "/usr/local/bin/run-v2-live-acceptance-on-start" in dockerfile
    assert "MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START" in pre_start
    assert pre_start.count("launch_startup_acceptance") == 3
    assert "--token" not in runner
    assert 'MOE_TOOLS_CANARY_TOKEN="${MOE_TOOLS_AUTH_TOKEN}"' in runner
    assert "flock -n" in runner
    assert "provider_wide_empty == true" in runner
    assert "daytona-before.json" in runner
    assert "daytona-after.json" in runner

    result = subprocess.run(
        ["bash", "-n", str(runner_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_startup_acceptance_is_disabled_by_default_and_requires_credentials(
    tmp_path: Path,
) -> None:
    runner_path = ROOT / "docker" / "runpod" / "run_v2_live_acceptance_on_start.sh"
    environment = os.environ | {"MOE_TOOLS_DATA_DIR": str(tmp_path)}
    environment.pop("MOE_TOOLS_AUTH_TOKEN", None)
    environment.pop("MOE_TOOLS_DAYTONA_API_KEY", None)

    disabled = subprocess.run(
        ["bash", str(runner_path)],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert disabled.returncode == 0
    assert not (tmp_path / "acceptance" / "v2-live-acceptance.status").exists()

    enabled = subprocess.run(
        ["bash", str(runner_path)],
        env=environment
        | {
            "MOE_TOOLS_MODE": "vllm",
            "MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START": "1",
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert enabled.returncode == 2
    assert "requires a resolved MOE_TOOLS_AUTH_TOKEN" in enabled.stderr
    assert not (tmp_path / "acceptance" / "v2-live-acceptance.status").exists()

    mock_mode = subprocess.run(
        ["bash", str(runner_path)],
        env=environment
        | {
            "MOE_TOOLS_AUTH_TOKEN": "a" * 32,
            "MOE_TOOLS_DAYTONA_API_KEY": "d" * 32,
            "MOE_TOOLS_MODE": "mock",
            "MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START": "1",
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert mock_mode.returncode == 2
    assert "requires MOE_TOOLS_MODE=vllm" in mock_mode.stderr
    assert not (tmp_path / "acceptance" / "v2-live-acceptance.status").exists()


def test_daytona_inventory_fails_secret_free_without_inherited_credential(
    tmp_path: Path,
) -> None:
    output = tmp_path / "inventory.json"
    environment = os.environ.copy()
    environment.pop("MOE_TOOLS_DAYTONA_API_KEY", None)

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "daytona_inventory.py"),
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    report = json.loads(output.read_text())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == ""
    assert report["success"] is False
    assert report["error"] == {"code": "daytona_credential_unavailable"}
    assert output.stat().st_mode & 0o777 == 0o600


def test_interrupted_startup_acceptance_runs_post_inventory(
    tmp_path: Path,
) -> None:
    runner_path = ROOT / "docker" / "runpod" / "run_v2_live_acceptance_on_start.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    harness_marker = tmp_path / "harness-started"
    after_marker = tmp_path / "after-inventory"
    python_stub = bin_dir / "python-stub"
    python_stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    *daytona_inventory.py)
        output="${3}"
        printf '%s\n' \
            '{"success":true,"provider_wide_empty":true,"sandbox_count":0}' \
            >"${output}"
        if [[ "${output}" == *daytona-after.json ]]; then
            : >"${AFTER_MARKER}"
        fi
        ;;
    *v2_live_acceptance.py)
        : >"${HARNESS_MARKER}"
        sleep 30 &
        child=$!
        terminate_child() {
            kill -TERM "${child}" 2>/dev/null || true
            wait "${child}" 2>/dev/null || true
            exit 143
        }
        trap terminate_child TERM
        wait "${child}"
        ;;
    -c)
        exit 0
        ;;
    *)
        exit 2
        ;;
esac
"""
    )
    python_stub.chmod(0o700)
    for name, body in {
        "flock": "#!/usr/bin/env bash\nexit 0\n",
        "setsid": '#!/usr/bin/env bash\nexec "$@"\n',
    }.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o700)

    environment = os.environ | {
        "AFTER_MARKER": str(after_marker),
        "HARNESS_MARKER": str(harness_marker),
        "MOE_TOOLS_ACCEPTANCE_PYTHON_PATH": str(python_stub),
        "MOE_TOOLS_ACCEPTANCE_RUNNER_PATH": str(runner_path),
        "MOE_TOOLS_APP_ROOT": str(tmp_path / "app"),
        "MOE_TOOLS_AUTH_TOKEN": "a" * 32,
        "MOE_TOOLS_DATA_DIR": str(tmp_path / "data"),
        "MOE_TOOLS_DAYTONA_API_KEY": "d" * 32,
        "MOE_TOOLS_MODE": "vllm",
        "MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START": "1",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    launched = subprocess.run(
        ["bash", str(runner_path)],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert launched.returncode == 0, launched.stderr

    acceptance_dir = tmp_path / "data" / "acceptance"
    pid_file = acceptance_dir / "v2-live-acceptance.pid"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not harness_marker.exists():
        time.sleep(0.02)
    assert harness_marker.exists()
    os.kill(int(pid_file.read_text()), signal.SIGTERM)

    status_file = acceptance_dir / "v2-live-acceptance.status"
    report: dict[str, object] = {}
    while time.monotonic() < deadline:
        report = json.loads(status_file.read_text())
        if report.get("state") == "interrupted":
            break
        time.sleep(0.02)

    assert report["state"] == "interrupted"
    assert report["cancellation_exit_code"] == 0
    assert report["daytona_after_exit_code"] == 0
    assert after_marker.exists()


def test_release_workflows_pin_actions_and_require_both_source_revisions() -> None:
    workflows = [
        (ROOT / ".github" / "workflows" / filename).read_text()
        for filename in ("ci.yml", "publish-container.yml")
    ]
    action_pattern = re.compile(
        r"^\s*uses:\s*[^\s]+@([^\s#]+)",
        re.MULTILINE,
    )

    action_refs = [
        match for workflow in workflows for match in action_pattern.findall(workflow)
    ]
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)

    publish = workflows[1]
    assert "inputs.vllm_source_ref ||" not in publish
    assert "APP_SOURCE_REF=${{ github.sha }}" in publish
    assert "VLLM_SOURCE_REF: ${{ inputs.vllm_source_ref }}" in publish
    assert "IMAGE_TAG: ${{ inputs.image_tag }}" in publish
    assert 'echo "- Tag: `${{ inputs.image_tag }}`"' not in publish
    assert '"${IMAGE_TAG}" =~ ^[A-Za-z0-9_]' in publish


def test_release_versions_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    frontend = json.loads((ROOT / "frontend" / "package.json").read_text())

    assert project["project"]["version"] == __version__ == "0.4.0rc1"
    assert frontend["version"] == "0.4.0-rc.1"
