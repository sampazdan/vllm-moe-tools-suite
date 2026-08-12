from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest
from moe_tools_suite.agentic.catalog import AgentTaskCatalog
from moe_tools_suite.agentic.domain import AgentTaskInfo, SandboxFile
from moe_tools_suite.agentic.task_packs.external import (
    ExternalTaskPackManifest,
    PreparationState,
    load_external_task_pack_registry,
)
from moe_tools_suite.agentic.task_packs.importer import (
    ExternalImportAttestation,
    ExternalPreparationError,
    PreparedExternalPack,
    PreparedExternalTask,
    external_task_pack_importers,
)
from pydantic import ValidationError

REPO_ENGINEERING_PACK_ID = "repo-engineering-v1"
AIDER_CANARY_PACK_ID = "aider-polyglot-python-canary-3"
AIDER_EXPANSION_PACK_ID = "aider-polyglot-python-expansion-5"


def test_external_registry_is_pinned_inspectable_and_fail_closed() -> None:
    registry = load_external_task_pack_registry()

    assert registry.harness.id == "harbor"
    assert len(registry.harness.revision) == 40
    assert len(registry.content_hash) == 64
    assert {pack.id for pack in registry.task_packs} == {
        "terminal-bench-2.1-engineering-15",
        "aider-polyglot-balanced-12",
        AIDER_CANARY_PACK_ID,
        "featurebench-fast-100-reference",
    }
    canary = registry.get(AIDER_CANARY_PACK_ID)
    assert canary.launchable
    assert canary.assets_prepared and canary.oracle_passed and canary.noop_failed
    assert all(
        not pack.launchable
        for pack in registry.task_packs
        if pack.id != AIDER_CANARY_PACK_ID
    )


def test_external_cohorts_use_real_counts_and_balanced_task_ids() -> None:
    registry = load_external_task_pack_registry()
    terminal_bench = registry.get("terminal-bench-2.1-engineering-15")
    aider = registry.get("aider-polyglot-balanced-12")
    featurebench = registry.get("featurebench-fast-100-reference")

    assert terminal_bench.upstream_task_count == 89
    assert len(terminal_bench.curated_task_ids) == 15
    assert "fix-code-vulnerability" in terminal_bench.curated_task_ids
    assert aider.upstream_task_count == 225
    assert len(aider.curated_task_ids) == 12
    assert {
        task_id.split("_", maxsplit=2)[1] for task_id in aider.curated_task_ids
    } == {"cpp", "go", "java", "javascript", "python", "rust"}
    assert featurebench.upstream_task_count == 100
    assert featurebench.curated_task_ids == []
    assert featurebench.preparation_state is PreparationState.REFERENCE_ONLY
    assert any("HEAD" in limitation for limitation in featurebench.limitations)


def test_external_manifest_resource_contains_no_private_execution_assets() -> None:
    resource = resources.files("moe_tools_suite.agentic.task_packs").joinpath(
        "external_manifests.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True)

    for forbidden_key in (
        '"files"',
        '"oracle_commands"',
        '"solution"',
        '"verifier_command"',
        '"verifier_file_paths"',
    ):
        assert forbidden_key not in encoded


def test_validated_external_pack_requires_all_eligibility_checks() -> None:
    source = load_external_task_pack_registry().task_packs[0]
    payload = source.model_dump(mode="json")
    payload["preparation_state"] = "validated"

    with pytest.raises(ValidationError, match="passing oracle"):
        ExternalTaskPackManifest.model_validate(payload)


def test_external_manifest_rejects_moving_or_duplicate_provenance() -> None:
    source = load_external_task_pack_registry().task_packs[0]
    moving = source.model_dump(mode="json")
    moving["source_revision"] = "main"
    duplicate = source.model_dump(mode="json")
    duplicate["curated_task_ids"] = ["fix-git", "fix-git"]

    with pytest.raises(ValidationError, match="40 characters"):
        ExternalTaskPackManifest.model_validate(moving)
    with pytest.raises(ValidationError, match="must be unique"):
        ExternalTaskPackManifest.model_validate(duplicate)


def test_repo_engineering_pack_is_runnable_and_publicly_inspectable() -> None:
    catalog = AgentTaskCatalog()
    info_by_id = {pack.id: pack for pack in catalog.list_packs()}
    pack = info_by_id[REPO_ENGINEERING_PACK_ID]

    assert pack.task_count == 3
    assert pack.ready is True
    assert pack.oracle_passed is True
    assert pack.noop_failed is True
    assert "smoke" not in pack.tags
    tasks = catalog.list_tasks(REPO_ENGINEERING_PACK_ID)
    assert {task.id for task in tasks} == {
        "repair-event-ledger",
        "build-release-plan",
        "bound-async-worker-pool",
    }
    assert all(len(task.success_criteria) >= 3 for task in tasks)
    public = [
        AgentTaskInfo(
            id=task.id,
            title=task.title,
            instruction=task.instruction,
            language=task.language,
            tags=task.tags,
            success_criteria=task.success_criteria,
            verifier_fingerprint=task.verifier_fingerprint(),
            hidden_verifier_file_count=len(task.verifier_file_paths),
            timeout_seconds=task.timeout_seconds,
        ).model_dump(mode="json")
        for task in tasks
    ]
    encoded = json.dumps(public)
    assert "oracle_commands" not in encoded
    assert "verifier_command" not in encoded
    assert "oracle/" not in encoded


def test_repo_engineering_pack_noop_fails_and_oracle_passes(
    tmp_path: Path,
) -> None:
    pack = AgentTaskCatalog().get_pack(REPO_ENGINEERING_PACK_ID)
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MOE_TOOLS_VERIFIER_DEV_MODE": "1",
    }

    for task in pack.tasks:
        workspace = tmp_path / task.id
        workspace.mkdir()
        for file in task.files:
            destination = workspace / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content)

        noop = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert noop.returncode != 0, f"{task.id} no-op unexpectedly passed"

        for command in task.oracle_commands:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                check=True,
            )
        oracle = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert oracle.returncode == 0, oracle.stdout + oracle.stderr


def test_aider_python_canary_is_launchable_and_honestly_scoped() -> None:
    catalog = AgentTaskCatalog()
    pack = catalog.get_pack(AIDER_CANARY_PACK_ID)

    assert pack.ready and pack.oracle_passed and pack.noop_failed
    assert len(pack.tasks) == 3
    assert {task.id for task in pack.tasks} == {
        "polyglot_python_zipper",
        "polyglot_python_wordy",
        "polyglot_python_zebra-puzzle",
    }
    assert "not an official Aider" in pack.description
    assert "f30b14415dd733c83627204bad0af69a89ceb46f" in pack.source
    assert "488af1b12b3b728b9364ab5e1bb663bd3e0ae643" in pack.source
    assert "373c1a204c4ec13e778200683784faba" in pack.source
    for task in pack.tasks:
        assert task.image_digest == (
            "sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
        )
        assert task.network_policy.value == "none"
        assert len(task.submission_file_paths) == 1
        assert task.verifier_file_paths == ["tests/verifier.py"]
        assert len(task.oracle_file_paths) == 1
        assert not (
            set(task.submission_file_paths)
            & (set(task.verifier_file_paths) | set(task.oracle_file_paths))
        )
    public = [
        AgentTaskInfo(
            id=task.id,
            title=task.title,
            instruction=task.instruction,
            language=task.language,
            tags=task.tags,
            success_criteria=task.success_criteria,
            verifier_fingerprint=task.verifier_fingerprint(),
            hidden_verifier_file_count=len(task.verifier_file_paths),
            timeout_seconds=task.timeout_seconds,
        ).model_dump(mode="json")
        for task in pack.tasks
    ]
    encoded = json.dumps(public)
    assert "MOE_TOOLS_VERIFIER_DEV_MODE" not in encoded
    assert ".moe_tools/candidate_runner.py" not in encoded
    assert "oracle/" not in encoded


def test_aider_python_canary_matches_pinned_source_attestation() -> None:
    package = resources.files(
        "moe_tools_suite.agentic.task_packs.aider_polyglot_python_canary_v1"
    )
    attestation = json.loads(
        package.joinpath("source_attestation.json").read_text(encoding="utf-8")
    )
    pack = AgentTaskCatalog().get_pack(AIDER_CANARY_PACK_ID)

    assert attestation["source"]["revision"] == (
        "f30b14415dd733c83627204bad0af69a89ceb46f"
    )
    assert attestation["harbor_adapter"]["revision"] == (
        "488af1b12b3b728b9364ab5e1bb663bd3e0ae643"
    )
    for task in pack.tasks:
        evidence = attestation["tasks"][task.id]
        starter = next(
            file for file in task.files if file.path in set(task.submission_file_paths)
        )
        oracle = next(
            file for file in task.files if file.path in set(task.oracle_file_paths)
        )
        assert (
            hashlib.sha256(task.instruction.encode()).hexdigest()
            == evidence["instruction_sha256"]
        )
        assert (
            hashlib.sha256(starter.content.encode()).hexdigest()
            == evidence["starter_sha256"]
        )
        assert (
            hashlib.sha256(oracle.content.encode()).hexdigest()
            == evidence["oracle_sha256"]
        )
        assert evidence["noop_failed"] and evidence["oracle_passed"]


def test_aider_python_canary_noop_fails_and_exact_oracle_passes(
    tmp_path: Path,
) -> None:
    pack = AgentTaskCatalog().get_pack(AIDER_CANARY_PACK_ID)
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MOE_TOOLS_VERIFIER_DEV_MODE": "1",
    }

    for task in pack.tasks:
        workspace = tmp_path / task.id
        workspace.mkdir()
        for file in task.files:
            destination = workspace / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content)

        noop = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert noop.returncode != 0, f"{task.id} no-op unexpectedly passed"

        for command in task.oracle_commands:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                check=True,
            )
        oracle = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert oracle.returncode == 0, oracle.stdout + oracle.stderr


def test_aider_python_expansion_is_launchable_and_honestly_scoped() -> None:
    catalog = AgentTaskCatalog()
    pack = catalog.get_pack(AIDER_EXPANSION_PACK_ID)

    assert pack.ready and pack.oracle_passed and pack.noop_failed
    assert len(pack.tasks) == 5
    assert {task.id for task in pack.tasks} == {
        "polyglot_python_affine-cipher",
        "polyglot_python_dominoes",
        "polyglot_python_proverb",
        "polyglot_python_transpose",
        "polyglot_python_variable-length-quantity",
    }
    assert "not an official Aider" in pack.description
    assert "f30b14415dd733c83627204bad0af69a89ceb46f" in pack.source
    assert "488af1b12b3b728b9364ab5e1bb663bd3e0ae643" in pack.source
    assert "cdafc517554ca0494a0b164f7df561ea" in pack.source
    for task in pack.tasks:
        assert task.image_digest == (
            "sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
        )
        assert task.network_policy.value == "none"
        assert len(task.submission_file_paths) == 1
        assert set(task.verifier_file_paths) == {
            "tests/verifier.py",
            "tests/verifier_support.py",
        }
        assert len(task.oracle_file_paths) == 1
        assert not (
            set(task.submission_file_paths)
            & (set(task.verifier_file_paths) | set(task.oracle_file_paths))
        )


def test_aider_python_expansion_matches_pinned_source_attestation() -> None:
    package = resources.files(
        "moe_tools_suite.agentic.task_packs.aider_polyglot_python_expansion_v2"
    )
    attestation = json.loads(
        package.joinpath("source_attestation.json").read_text(encoding="utf-8")
    )
    pack = AgentTaskCatalog().get_pack(AIDER_EXPANSION_PACK_ID)

    assert attestation["source"]["revision"] == (
        "f30b14415dd733c83627204bad0af69a89ceb46f"
    )
    assert attestation["harbor_adapter"]["revision"] == (
        "488af1b12b3b728b9364ab5e1bb663bd3e0ae643"
    )
    for task in pack.tasks:
        evidence = attestation["tasks"][task.id]
        file_by_path = {file.path: file for file in task.files}
        starter = next(
            file for file in task.files if file.path in set(task.submission_file_paths)
        )
        oracle = next(
            file for file in task.files if file.path in set(task.oracle_file_paths)
        )
        assert (
            hashlib.sha256(task.instruction.encode()).hexdigest()
            == evidence["instruction_sha256"]
        )
        assert (
            hashlib.sha256(starter.content.encode()).hexdigest()
            == evidence["starter_sha256"]
        )
        assert (
            hashlib.sha256(oracle.content.encode()).hexdigest()
            == evidence["oracle_sha256"]
        )
        assert (
            hashlib.sha256(
                file_by_path["tests/verifier.py"].content.encode()
            ).hexdigest()
            == evidence["verifier_sha256"]
        )
        assert (
            hashlib.sha256(
                file_by_path["tests/verifier_support.py"].content.encode()
            ).hexdigest()
            == attestation["isolation_adapter"]["verifier_support_sha256"]
        )
        assert (
            hashlib.sha256(
                file_by_path[".moe_tools/candidate_runner.py"].content.encode()
            ).hexdigest()
            == attestation["isolation_adapter"]["candidate_runner_sha256"]
        )
        assert evidence["noop_failed"] and evidence["oracle_passed"]


def test_aider_python_expansion_noop_fails_and_exact_oracle_passes(
    tmp_path: Path,
) -> None:
    pack = AgentTaskCatalog().get_pack(AIDER_EXPANSION_PACK_ID)
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MOE_TOOLS_VERIFIER_DEV_MODE": "1",
    }

    for task in pack.tasks:
        workspace = tmp_path / task.id
        workspace.mkdir()
        for file in task.files:
            destination = workspace / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content)

        noop = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert noop.returncode != 0, f"{task.id} no-op unexpectedly passed"

        for command in task.oracle_commands:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                check=True,
            )
        oracle = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert oracle.returncode == 0, oracle.stdout + oracle.stderr


@pytest.mark.asyncio
async def test_external_importers_advertise_requirements_and_fail_closed(
    tmp_path: Path,
) -> None:
    importers = external_task_pack_importers()

    assert {importer.manifest.id for importer in importers} == {
        "terminal-bench-2.1-engineering-15",
        "aider-polyglot-balanced-12",
    }
    for importer in importers:
        assert importer.requirements
        assert importer.manifest.launchable is False
        with pytest.raises(ExternalPreparationError, match="provenance-only"):
            await importer.prepare(tmp_path)


def test_external_import_contract_requires_exact_cohort_and_attestations() -> None:
    registry = load_external_task_pack_registry()
    manifest = registry.get("terminal-bench-2.1-engineering-15")
    tasks = [_prepared_external_task(task_id) for task_id in manifest.curated_task_ids]
    incomplete = PreparedExternalPack(
        manifest_id=manifest.id,
        tasks=tasks,
        attestation=ExternalImportAttestation(
            source_revision=manifest.source_revision,
            harness_revision=registry.harness.revision,
            source_tree_sha256="1" * 64,
            prepared_task_ids=list(manifest.curated_task_ids),
            oracle_passed_task_ids=[],
            noop_failed_task_ids=[],
            license_reviewed=True,
        ),
    )

    with pytest.raises(ExternalPreparationError, match="eligibility checks"):
        incomplete.to_agent_task_pack(registry)

    complete = incomplete.model_copy(
        update={
            "attestation": incomplete.attestation.model_copy(
                update={
                    "oracle_passed_task_ids": list(manifest.curated_task_ids),
                    "noop_failed_task_ids": list(manifest.curated_task_ids),
                }
            )
        }
    )
    pack = complete.to_agent_task_pack(registry)

    assert pack.ready and pack.oracle_passed and pack.noop_failed
    assert len(pack.tasks) == 15
    assert all(task.image_digest == f"sha256:{'2' * 64}" for task in pack.tasks)
    assert all(
        task.verifier_file_paths == ["tests/hidden_test.py"] for task in pack.tasks
    )
    assert all(
        "tests/hidden_test.py" not in [file.path for file in task.files[:1]]
        for task in pack.tasks
    )


def _prepared_external_task(task_id: str) -> PreparedExternalTask:
    return PreparedExternalTask(
        id=task_id,
        title=f"Imported {task_id}",
        instruction="Inspect the repository, implement the requested fix, and test it.",
        category="terminal-bench",
        language="python",
        tags=["external", "terminal-bench"],
        success_criteria=["The protected upstream verifier passes."],
        image_ref="example/task-image:immutable",
        image_digest=f"sha256:{'2' * 64}",
        public_files=[SandboxFile(path="main.py", content="VALUE = 0\n")],
        submission_file_paths=["main.py"],
        protected_files=[
            SandboxFile(
                path="tests/hidden_test.py",
                content="assert True\n",
            )
        ],
        verifier_command="python -I tests/hidden_test.py",
        oracle_commands=["sed -i 's/0/1/' main.py"],
    )
