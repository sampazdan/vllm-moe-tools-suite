from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from importlib import resources
from pathlib import PurePosixPath

from .atif import BASH_JSON_SYSTEM_PROMPT, BASH_TOOL_DEFINITION
from .domain import (
    AgentBudgets,
    AgentDefinition,
    AgentDescriptor,
    AgentTask,
    AgentTaskPack,
    AgentTaskPackInfo,
    CreateAgentRunRequest,
    agent_definition,
)

DEFAULT_AGENT_ID = "bash-json-v1"
DEFAULT_TASK_PACK_ID = "smoke-python-v1"

BUNDLED_TASK_PACK_PACKAGES = (
    "moe_tools_suite.agentic.task_packs.smoke_python_v1",
    "moe_tools_suite.agentic.task_packs.repo_engineering_v1",
    "moe_tools_suite.agentic.task_packs.aider_polyglot_python_canary_v1",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_bundled_text_resource(package: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid bundled task-pack resource {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"invalid bundled task-pack resource {value!r}")
    resource = resources.files(package)
    for part in path.parts:
        resource = resource.joinpath(part)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid bundled task file resource {value!r}") from error


DEFAULT_AGENT = AgentDescriptor(
    id=DEFAULT_AGENT_ID,
    name="bash-json",
    label="Minimal bash agent",
    revision="1",
    description=(
        "A small model-centered coding loop with one JSON shell action per turn."
    ),
    available=True,
    is_default=True,
    tool_names=["bash"],
    default_budgets=AgentBudgets(),
    system_prompt_hash=hashlib.sha256(BASH_JSON_SYSTEM_PROMPT.encode()).hexdigest(),
    tool_schema_hash=_canonical_hash(BASH_TOOL_DEFINITION),
)


def load_bundled_task_pack(package: str) -> AgentTaskPack:
    """Load and fingerprint an immutable task pack using package resources."""

    resource = resources.files(package).joinpath("pack.json")
    try:
        raw = resource.read_bytes()
        payload = json.loads(raw)
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(f"invalid bundled task pack resource {package!r}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"bundled task pack {package!r} must contain a JSON object")
    resolved_resources = False
    for task in payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        if "instruction_resource" in task:
            if "instruction" in task:
                raise ValueError(
                    "bundled task cannot define both instruction and "
                    "instruction_resource"
                )
            task["instruction"] = _read_bundled_text_resource(
                package, task.pop("instruction_resource")
            )
            resolved_resources = True
        for file in task.get("files", []):
            if not isinstance(file, dict) or "resource" not in file:
                continue
            resource_path = file.pop("resource")
            if "content" in file:
                raise ValueError(
                    "bundled task file cannot define both content and resource"
                )
            file["content"] = _read_bundled_text_resource(package, resource_path)
            resolved_resources = True
    fingerprint = (
        _canonical_hash(payload)
        if resolved_resources
        else hashlib.sha256(raw).hexdigest()
    )
    payload["content_hash"] = fingerprint
    payload["fingerprint"] = fingerprint
    try:
        return AgentTaskPack.model_validate(payload)
    except ValueError as error:
        raise ValueError(f"invalid bundled task pack {package!r}: {error}") from error


class AgentTaskCatalog:
    """Registry for pinned agents and immutable bundled coding task packs."""

    def __init__(
        self,
        *,
        packs: Iterable[AgentTaskPack] | None = None,
        agents: Iterable[AgentDescriptor] | None = None,
    ) -> None:
        loaded_packs = (
            list(packs)
            if packs is not None
            else [
                load_bundled_task_pack(package)
                for package in BUNDLED_TASK_PACK_PACKAGES
            ]
        )
        loaded_agents = list(agents) if agents is not None else [DEFAULT_AGENT]
        self._packs = {pack.id: pack for pack in loaded_packs}
        self._agents = {agent.id: agent for agent in loaded_agents}
        if len(self._packs) != len(loaded_packs):
            raise ValueError("agent task-pack IDs must be unique")
        if len(self._agents) != len(loaded_agents):
            raise ValueError("agent IDs must be unique")
        if sum(agent.is_default for agent in loaded_agents) != 1:
            raise ValueError("exactly one agent must be the default")

    def list_agents(self) -> list[AgentDefinition]:
        return sorted(
            (agent_definition(agent) for agent in self._agents.values()),
            key=lambda agent: (not agent.is_default, agent.label.casefold()),
        )

    def list_agent_descriptors(self) -> list[AgentDescriptor]:
        return sorted(
            (agent.model_copy(deep=True) for agent in self._agents.values()),
            key=lambda agent: (not agent.is_default, agent.label.casefold()),
        )

    def get_agent(self, agent_id: str) -> AgentDescriptor:
        try:
            return self._agents[agent_id].model_copy(deep=True)
        except KeyError as error:
            raise KeyError(f"unknown agent {agent_id!r}") from error

    def list_packs(self) -> list[AgentTaskPackInfo]:
        return sorted(
            (pack.info() for pack in self._packs.values()),
            key=lambda pack: pack.name.casefold(),
        )

    def get_pack(self, pack_id: str) -> AgentTaskPack:
        try:
            return self._packs[pack_id].model_copy(deep=True)
        except KeyError as error:
            raise KeyError(f"unknown agent task pack {pack_id!r}") from error

    def list_tasks(self, pack_id: str) -> list[AgentTask]:
        pack = self.get_pack(pack_id)
        return [task.model_copy(deep=True) for task in pack.tasks]

    def get_task(self, pack_id: str, task_id: str) -> AgentTask:
        pack = self.get_pack(pack_id)
        task = next((task for task in pack.tasks if task.id == task_id), None)
        if task is None:
            raise KeyError(f"unknown task {task_id!r} in pack {pack_id!r}")
        return task.model_copy(deep=True)

    def validate_run_request(
        self,
        request: CreateAgentRunRequest,
    ) -> tuple[AgentTaskPack, list[AgentTask], AgentDescriptor]:
        pack = self.get_pack(request.task_pack_id)
        if not pack.ready or not pack.oracle_passed or not pack.noop_failed:
            raise ValueError("agent task pack has not passed eligibility checks")
        agent = self.get_agent(request.agent_id)
        if not agent.available:
            raise ValueError(f"agent {request.agent_id!r} is unavailable")
        task_by_id = {task.id: task for task in pack.tasks}
        missing = set(request.task_ids) - task_by_id.keys()
        if missing:
            raise ValueError(f"unknown agent tasks: {sorted(missing)}")
        selected = [task_by_id[task_id] for task_id in request.task_ids]
        return pack, selected, agent


def agent_run_contract_fingerprint(
    *,
    request: CreateAgentRunRequest,
    pack: AgentTaskPack,
    tasks: list[AgentTask],
    agent: AgentDescriptor,
) -> str:
    """Fingerprint the comparable task/agent/sandbox contract, not its profile."""

    task_contracts = [
        {
            "id": task.id,
            "image_ref": task.image_ref,
            "image_digest": task.image_digest,
            "verifier_hash": hashlib.sha256(task.verifier_command.encode()).hexdigest(),
            "submission_file_paths": task.submission_file_paths,
            "verifier_file_paths": task.verifier_file_paths,
            "oracle_file_paths": task.oracle_file_paths,
            "network_policy": task.network_policy.value,
            "allowed_hosts": task.allowed_hosts,
            "cpu": task.cpu,
            "memory_mb": task.memory_mb,
            "disk_mb": task.disk_mb,
            "timeout_seconds": task.timeout_seconds,
        }
        for task in tasks
    ]
    return _canonical_hash(
        {
            "task_pack_id": pack.id,
            "task_pack_revision": pack.revision,
            "task_pack_fingerprint": pack.fingerprint,
            "tasks": task_contracts,
            "agent_id": agent.id,
            "agent_revision": agent.revision,
            "system_prompt_hash": agent.system_prompt_hash,
            "tool_schema_hash": agent.tool_schema_hash,
            "sandbox_provider_id": request.sandbox_provider_id,
            "attempts": request.attempts,
            "seed": request.seed,
            "generation": request.generation.model_dump(mode="json"),
            "thinking_enabled": request.reasoning_mode.value != "off",
            "budgets": request.budgets.model_dump(mode="json"),
            "evaluation_contract_fingerprint": (
                request.evaluation_contract.fingerprint
                if request.evaluation_contract is not None
                else None
            ),
            "execution_policy_fingerprint": (
                request.execution_policy.fingerprint
                if request.execution_policy is not None
                else None
            ),
        }
    )


def oracle_commands(pack_id: str, task_id: str) -> list[str]:
    """Return a copy of the bundled reference actions for scripted fake runs."""

    task = AgentTaskCatalog().get_task(pack_id, task_id)
    return list(task.oracle_commands)
