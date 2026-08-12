from __future__ import annotations

import copy
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .v2_domain import DriftCondition, canonical_fingerprint

DRIFT_FORMULAS = {
    "version": "state-drift-v1",
    "state_distance": (
        "unequal_leaf_paths / max(union_of_leaf_paths, 1); missing paths differ"
    ),
    "state_fidelity": "1 - state_distance",
    "auc": "arithmetic mean of post-transition state fidelity",
    "recovery_rate": (
        "divergent checkpoints followed immediately by an exact checkpoint / "
        "divergent checkpoints with a successor"
    ),
    "divergence_growth_slope": (
        "ordinary-least-squares slope of (1 - fidelity) over checkpoint index"
    ),
    "horizon_reliability": (
        "largest tested horizon whose paired final-success mean meets threshold"
    ),
    "routing_selection_overlap": (
        "histogram intersection of normalized baseline and candidate per-expert "
        "selection counts at an aligned checkpoint"
    ),
    "routing_mass_js_divergence": (
        "base-2 Jensen-Shannon divergence between normalized baseline and "
        "candidate per-expert routing-mass distributions"
    ),
}
DRIFT_FORMULA_FINGERPRINT = canonical_fingerprint(DRIFT_FORMULAS)


class DriftToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DriftPlannedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str
    canonical_call: DriftToolCall


class DriftTurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    checkpoint: int
    horizon: int
    condition: DriftCondition
    instruction: str
    observed_state: dict[str, Any]
    canonical_state_summary: str | None = None


class DriftTransition(BaseModel):
    checkpoint: int
    instruction: str
    expected_call: DriftToolCall
    actual_call: DriftToolCall
    expected_state: dict[str, Any]
    actual_state: dict[str, Any]
    expected_state_fingerprint: str
    actual_state_fingerprint: str
    state_distance: float
    state_fidelity: float
    exact: bool
    valid_tool_call: bool
    invariant_violations: list[str] = Field(default_factory=list)
    collateral_mutations: list[str] = Field(default_factory=list)
    rollback: bool = False
    rollback_correct: bool | None = None


class DriftMetrics(BaseModel):
    formula_fingerprint: str = DRIFT_FORMULA_FINGERPRINT
    final_success: bool
    transition_accuracy: float
    per_transition_correctness: list[bool]
    state_fidelity: list[float]
    first_divergence_checkpoint: int | None = None
    survival_curve: list[float]
    area_under_state_fidelity_curve: float
    recovery_rate: float
    invariant_violations: int
    invalid_tool_calls: int
    collateral_mutations: int
    rollback_correctness: float | None = None
    divergence_growth_slope: float


class DriftRunResult(BaseModel):
    scenario_id: str
    scenario_revision: str
    seed: int
    condition: DriftCondition
    horizon: int
    initial_state: dict[str, Any]
    final_expected_state: dict[str, Any]
    final_actual_state: dict[str, Any]
    transitions: list[DriftTransition]
    metrics: DriftMetrics


class DriftScenario(Protocol):
    id: str
    revision: str
    name: str
    description: str

    def initial_state(self, seed: int) -> dict[str, Any]: ...

    def plan(self, seed: int, horizon: int) -> list[DriftPlannedTurn]: ...

    def apply(
        self, state: dict[str, Any], call: DriftToolCall
    ) -> tuple[dict[str, Any], bool]: ...

    def invariant_violations(self, state: dict[str, Any]) -> list[str]: ...

    def compact_summary(self, state: dict[str, Any]) -> str: ...

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]: ...


ActionProvider = Callable[[DriftTurnInput], Awaitable[DriftToolCall]]


@dataclass(frozen=True)
class LedgerReconciliationScenario:
    id: str = "ledger-reconciliation-v1"
    revision: str = "1.0.0"
    name: str = "Ledger and account reconciliation"
    description: str = (
        "Transfers, balanced corrections, snapshots, and rollbacks over a fixed "
        "three-account ledger."
    )

    def initial_state(self, seed: int) -> dict[str, Any]:
        offset = seed % 17
        return {
            "accounts": {
                "checking": 1000 + offset,
                "savings": 500,
                "reserve": 250,
            },
            "ledger": [],
            "snapshots": {},
            "total_assets": 1750 + offset,
        }

    def plan(self, seed: int, horizon: int) -> list[DriftPlannedTurn]:
        amount = 20 + seed % 9
        cycle = [
            DriftPlannedTurn(
                instruction=(
                    f"Move {amount} units from checking to savings and record tx-a."
                ),
                canonical_call=DriftToolCall(
                    tool="transfer",
                    arguments={
                        "source": "checking",
                        "target": "savings",
                        "amount": amount,
                        "reference": "tx-a",
                    },
                ),
            ),
            DriftPlannedTurn(
                instruction="Snapshot the reconciled ledger as close-1.",
                canonical_call=DriftToolCall(
                    tool="snapshot", arguments={"name": "close-1"}
                ),
            ),
            DriftPlannedTurn(
                instruction=(
                    "Apply a balanced correction of 7 units from savings to reserve."
                ),
                canonical_call=DriftToolCall(
                    tool="transfer",
                    arguments={
                        "source": "savings",
                        "target": "reserve",
                        "amount": 7,
                        "reference": "correction-1",
                    },
                ),
            ),
            DriftPlannedTurn(
                instruction="Roll the ledger back to snapshot close-1.",
                canonical_call=DriftToolCall(
                    tool="rollback", arguments={"name": "close-1"}
                ),
            ),
            DriftPlannedTurn(
                instruction="Move 11 units from reserve to checking and record tx-b.",
                canonical_call=DriftToolCall(
                    tool="transfer",
                    arguments={
                        "source": "reserve",
                        "target": "checking",
                        "amount": 11,
                        "reference": "tx-b",
                    },
                ),
            ),
            DriftPlannedTurn(
                instruction="Snapshot the current ledger as close-2.",
                canonical_call=DriftToolCall(
                    tool="snapshot", arguments={"name": "close-2"}
                ),
            ),
            DriftPlannedTurn(
                instruction="Move 5 units from checking to reserve and record tx-c.",
                canonical_call=DriftToolCall(
                    tool="transfer",
                    arguments={
                        "source": "checking",
                        "target": "reserve",
                        "amount": 5,
                        "reference": "tx-c",
                    },
                ),
            ),
            DriftPlannedTurn(
                instruction="Roll the ledger back to snapshot close-2.",
                canonical_call=DriftToolCall(
                    tool="rollback", arguments={"name": "close-2"}
                ),
            ),
        ]
        return [cycle[index % len(cycle)] for index in range(horizon)]

    def apply(
        self, state: dict[str, Any], call: DriftToolCall
    ) -> tuple[dict[str, Any], bool]:
        updated = copy.deepcopy(state)
        args = call.arguments
        try:
            if call.tool == "transfer":
                source = str(args["source"])
                target = str(args["target"])
                amount = _positive_int(args["amount"])
                reference = str(args["reference"])
                accounts = updated["accounts"]
                if source == target or source not in accounts or target not in accounts:
                    return state, False
                if accounts[source] < amount:
                    return state, False
                accounts[source] -= amount
                accounts[target] += amount
                updated["ledger"].append(
                    {
                        "reference": reference,
                        "source": source,
                        "target": target,
                        "amount": amount,
                    }
                )
            elif call.tool == "snapshot":
                name = str(args["name"])
                if not name:
                    return state, False
                snapshot = copy.deepcopy(updated)
                snapshot["snapshots"] = {}
                updated["snapshots"][name] = snapshot
            elif call.tool == "rollback":
                name = str(args["name"])
                if name not in updated["snapshots"]:
                    return state, False
                snapshots = updated["snapshots"]
                updated = copy.deepcopy(snapshots[name])
                updated["snapshots"] = snapshots
            else:
                return state, False
        except (KeyError, TypeError, ValueError):
            return state, False
        return updated, True

    def invariant_violations(self, state: dict[str, Any]) -> list[str]:
        violations = []
        accounts = state.get("accounts")
        if not isinstance(accounts, dict) or any(
            not isinstance(value, int) or value < 0 for value in accounts.values()
        ):
            violations.append("account balances must be non-negative integers")
        elif sum(accounts.values()) != state.get("total_assets"):
            violations.append("account balances must conserve total assets")
        return violations

    def compact_summary(self, state: dict[str, Any]) -> str:
        public = self.public_state(state)
        return json.dumps(public, sort_keys=True, separators=(",", ":"))

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "accounts": copy.deepcopy(state["accounts"]),
            "ledger": copy.deepcopy(state["ledger"]),
            "total_assets": state["total_assets"],
            "snapshot_names": sorted(state["snapshots"]),
        }


@dataclass(frozen=True)
class OrderLifecycleScenario:
    id: str = "order-lifecycle-v1"
    revision: str = "1.0.0"
    name: str = "Order lifecycle and inventory correction"
    description: str = (
        "Reservation, shipment, inventory correction, cancellation, and rollback "
        "over a small deterministic order store."
    )

    def initial_state(self, seed: int) -> dict[str, Any]:
        return {
            "inventory": {"widget": 12 + seed % 3, "cable": 8},
            "orders": {},
            "snapshots": {},
        }

    def plan(self, seed: int, horizon: int) -> list[DriftPlannedTurn]:
        quantity = 2 + seed % 2
        turns: list[DriftPlannedTurn] = []
        cycle_index = 1
        while len(turns) < horizon:
            order_id = f"o-{cycle_index}"
            snapshot = f"pre-ship-{cycle_index}"
            turns.extend(
                [
                    DriftPlannedTurn(
                        instruction=(
                            f"Create order {order_id} for {quantity} widgets."
                        ),
                        canonical_call=DriftToolCall(
                            tool="create_order",
                            arguments={
                                "order_id": order_id,
                                "sku": "widget",
                                "quantity": quantity,
                            },
                        ),
                    ),
                    DriftPlannedTurn(
                        instruction=f"Reserve inventory for order {order_id}.",
                        canonical_call=DriftToolCall(
                            tool="reserve", arguments={"order_id": order_id}
                        ),
                    ),
                    DriftPlannedTurn(
                        instruction=f"Snapshot the order store as {snapshot}.",
                        canonical_call=DriftToolCall(
                            tool="snapshot", arguments={"name": snapshot}
                        ),
                    ),
                    DriftPlannedTurn(
                        instruction=f"Ship order {order_id}.",
                        canonical_call=DriftToolCall(
                            tool="ship", arguments={"order_id": order_id}
                        ),
                    ),
                    DriftPlannedTurn(
                        instruction=(
                            f"Roll back to {snapshot} so {order_id} is reserved."
                        ),
                        canonical_call=DriftToolCall(
                            tool="rollback", arguments={"name": snapshot}
                        ),
                    ),
                    DriftPlannedTurn(
                        instruction=(
                            f"Cancel {order_id} and release its reserved inventory."
                        ),
                        canonical_call=DriftToolCall(
                            tool="cancel", arguments={"order_id": order_id}
                        ),
                    ),
                ]
            )
            cycle_index += 1
        return turns[:horizon]

    def apply(
        self, state: dict[str, Any], call: DriftToolCall
    ) -> tuple[dict[str, Any], bool]:
        updated = copy.deepcopy(state)
        args = call.arguments
        try:
            if call.tool == "create_order":
                order_id = str(args["order_id"])
                sku = str(args["sku"])
                quantity = _positive_int(args["quantity"])
                if order_id in updated["orders"] or sku not in updated["inventory"]:
                    return state, False
                updated["orders"][order_id] = {
                    "sku": sku,
                    "quantity": quantity,
                    "status": "created",
                }
            elif call.tool == "reserve":
                order = updated["orders"][str(args["order_id"])]
                if order["status"] != "created":
                    return state, False
                if updated["inventory"][order["sku"]] < order["quantity"]:
                    return state, False
                updated["inventory"][order["sku"]] -= order["quantity"]
                order["status"] = "reserved"
            elif call.tool == "ship":
                order = updated["orders"][str(args["order_id"])]
                if order["status"] != "reserved":
                    return state, False
                order["status"] = "shipped"
            elif call.tool == "cancel":
                order = updated["orders"][str(args["order_id"])]
                if order["status"] not in {"created", "reserved"}:
                    return state, False
                if order["status"] == "reserved":
                    updated["inventory"][order["sku"]] += order["quantity"]
                order["status"] = "cancelled"
            elif call.tool == "snapshot":
                name = str(args["name"])
                snapshot = copy.deepcopy(updated)
                snapshot["snapshots"] = {}
                updated["snapshots"][name] = snapshot
            elif call.tool == "rollback":
                name = str(args["name"])
                if name not in updated["snapshots"]:
                    return state, False
                snapshots = updated["snapshots"]
                updated = copy.deepcopy(snapshots[name])
                updated["snapshots"] = snapshots
            else:
                return state, False
        except (KeyError, TypeError, ValueError):
            return state, False
        return updated, True

    def invariant_violations(self, state: dict[str, Any]) -> list[str]:
        violations = []
        if any(
            not isinstance(value, int) or value < 0
            for value in state.get("inventory", {}).values()
        ):
            violations.append("inventory must be non-negative integers")
        statuses = {"created", "reserved", "shipped", "cancelled"}
        if any(
            order.get("status") not in statuses
            for order in state.get("orders", {}).values()
        ):
            violations.append("order status is invalid")
        return violations

    def compact_summary(self, state: dict[str, Any]) -> str:
        return json.dumps(
            self.public_state(state), sort_keys=True, separators=(",", ":")
        )

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "inventory": copy.deepcopy(state["inventory"]),
            "orders": copy.deepcopy(state["orders"]),
            "snapshot_names": sorted(state["snapshots"]),
        }


@dataclass(frozen=True)
class RecordWorkflowScenario:
    """Deterministic typed-record workflow used by several first-party domains."""

    id: str
    name: str
    description: str
    record_ids: tuple[str, str, str]
    revision: str = "1.0.0"

    def initial_state(self, seed: int) -> dict[str, Any]:
        return {
            "records": {
                record_id: {
                    "value": 10 + seed % 5 + index,
                    "status": "draft",
                    "links": [],
                }
                for index, record_id in enumerate(self.record_ids)
            },
            "snapshots": {},
            "revision": 0,
        }

    def plan(self, seed: int, horizon: int) -> list[DriftPlannedTurn]:
        first, second, third = self.record_ids
        delta = 1 + seed % 3
        cycle = [
            DriftPlannedTurn(
                instruction=f"Increase {first}'s value by {delta}.",
                canonical_call=DriftToolCall(
                    tool="adjust", arguments={"record_id": first, "delta": delta}
                ),
            ),
            DriftPlannedTurn(
                instruction=f"Link {second} to its prerequisite {first}.",
                canonical_call=DriftToolCall(
                    tool="link",
                    arguments={"record_id": second, "target_id": first},
                ),
            ),
            DriftPlannedTurn(
                instruction=f"Mark {first} as reviewed.",
                canonical_call=DriftToolCall(
                    tool="set_status",
                    arguments={"record_id": first, "status": "reviewed"},
                ),
            ),
            DriftPlannedTurn(
                instruction="Snapshot the workflow as stable-1.",
                canonical_call=DriftToolCall(
                    tool="snapshot", arguments={"name": "stable-1"}
                ),
            ),
            DriftPlannedTurn(
                instruction=f"Increase {third}'s value by 2.",
                canonical_call=DriftToolCall(
                    tool="adjust", arguments={"record_id": third, "delta": 2}
                ),
            ),
            DriftPlannedTurn(
                instruction=f"Mark {second} as approved.",
                canonical_call=DriftToolCall(
                    tool="set_status",
                    arguments={"record_id": second, "status": "approved"},
                ),
            ),
            DriftPlannedTurn(
                instruction="Roll back to stable-1.",
                canonical_call=DriftToolCall(
                    tool="rollback", arguments={"name": "stable-1"}
                ),
            ),
            DriftPlannedTurn(
                instruction=f"Mark {third} as released.",
                canonical_call=DriftToolCall(
                    tool="set_status",
                    arguments={"record_id": third, "status": "released"},
                ),
            ),
        ]
        return [cycle[index % len(cycle)] for index in range(horizon)]

    def apply(
        self, state: dict[str, Any], call: DriftToolCall
    ) -> tuple[dict[str, Any], bool]:
        updated = copy.deepcopy(state)
        arguments = call.arguments
        try:
            if call.tool == "adjust":
                record = updated["records"][str(arguments["record_id"])]
                delta = arguments["delta"]
                if type(delta) is not int or delta == 0:
                    return state, False
                record["value"] += delta
                if record["value"] < 0:
                    return state, False
                updated["revision"] += 1
            elif call.tool == "link":
                record_id = str(arguments["record_id"])
                target_id = str(arguments["target_id"])
                records = updated["records"]
                if (
                    record_id not in records
                    or target_id not in records
                    or record_id == target_id
                ):
                    return state, False
                if target_id not in records[record_id]["links"]:
                    records[record_id]["links"].append(target_id)
                    records[record_id]["links"].sort()
                    updated["revision"] += 1
            elif call.tool == "set_status":
                record = updated["records"][str(arguments["record_id"])]
                status = str(arguments["status"])
                if status not in {"draft", "reviewed", "approved", "released"}:
                    return state, False
                if record["status"] != status:
                    record["status"] = status
                    updated["revision"] += 1
            elif call.tool == "snapshot":
                name = str(arguments["name"])
                if not name:
                    return state, False
                snapshot = copy.deepcopy(updated)
                snapshot["snapshots"] = {}
                updated["snapshots"][name] = snapshot
            elif call.tool == "rollback":
                name = str(arguments["name"])
                if name not in updated["snapshots"]:
                    return state, False
                snapshots = updated["snapshots"]
                updated = copy.deepcopy(snapshots[name])
                updated["snapshots"] = snapshots
            else:
                return state, False
        except (KeyError, TypeError, ValueError):
            return state, False
        return updated, True

    def invariant_violations(self, state: dict[str, Any]) -> list[str]:
        records = state.get("records")
        if not isinstance(records, dict) or set(records) != set(self.record_ids):
            return ["workflow must retain its complete typed record set"]
        violations = []
        statuses = {"draft", "reviewed", "approved", "released"}
        for record_id, record in records.items():
            if not isinstance(record.get("value"), int) or record["value"] < 0:
                violations.append(f"{record_id} value must be a non-negative integer")
            if record.get("status") not in statuses:
                violations.append(f"{record_id} status is invalid")
            links = record.get("links")
            if (
                not isinstance(links, list)
                or record_id in links
                or any(link not in records for link in links)
                or links != sorted(set(links))
            ):
                violations.append(f"{record_id} links are invalid")
        if not isinstance(state.get("revision"), int) or state["revision"] < 0:
            violations.append("workflow revision must be a non-negative integer")
        return violations

    def compact_summary(self, state: dict[str, Any]) -> str:
        return json.dumps(
            self.public_state(state), sort_keys=True, separators=(",", ":")
        )

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "records": copy.deepcopy(state["records"]),
            "revision": state["revision"],
            "snapshot_names": sorted(state["snapshots"]),
        }


SCENARIOS: dict[str, DriftScenario] = {
    scenario.id: scenario
    for scenario in (
        LedgerReconciliationScenario(),
        OrderLifecycleScenario(),
        RecordWorkflowScenario(
            id="dependency-release-planning-v1",
            name="Dependency graph and release planning",
            description=(
                "Dependency edits, approval gates, snapshots, and rollback over "
                "a compact release graph."
            ),
            record_ids=("api", "worker", "release"),
        ),
        RecordWorkflowScenario(
            id="structured-data-transformation-v1",
            name="Structured data transformation",
            description=(
                "Sequential source, normalized, and export record mutations with "
                "exact rollback."
            ),
            record_ids=("source", "normalized", "export"),
        ),
        RecordWorkflowScenario(
            id="ticket-crm-workflow-v1",
            name="Ticket and CRM workflow",
            description=(
                "Linked ticket, account, and follow-up state with durable status "
                "transitions."
            ),
            record_ids=("ticket", "account", "followup"),
        ),
        RecordWorkflowScenario(
            id="repository-evolution-v1",
            name="Repository evolution",
            description=(
                "Sequential branch and release metadata changes under explicit "
                "constraints."
            ),
            record_ids=("main", "feature", "release"),
        ),
        RecordWorkflowScenario(
            id="branch-counterfactual-rollback-v1",
            name="Branch, counterfactual, and rollback",
            description=(
                "Counterfactual state edits followed by exact snapshot recovery."
            ),
            record_ids=("base", "counterfactual", "restored"),
        ),
    )
}


async def run_drift_scenario(
    scenario: DriftScenario,
    *,
    seed: int,
    horizon: int,
    condition: DriftCondition,
    action_provider: ActionProvider,
    on_checkpoint: Callable[[int, int], Awaitable[None]] | None = None,
) -> DriftRunResult:
    if horizon < 1:
        raise ValueError("drift horizon must be positive")
    plan = scenario.plan(seed, horizon)
    if len(plan) != horizon:
        raise ValueError("scenario plan length does not match the requested horizon")
    initial = scenario.initial_state(seed)
    canonical_state = copy.deepcopy(initial)
    actual_state = copy.deepcopy(initial)
    transitions: list[DriftTransition] = []

    for checkpoint, planned in enumerate(plan, start=1):
        canonical_before = copy.deepcopy(canonical_state)
        canonical_state, canonical_valid = scenario.apply(
            canonical_state, planned.canonical_call
        )
        if not canonical_valid or scenario.invariant_violations(canonical_state):
            raise RuntimeError(
                f"scenario {scenario.id} contains an invalid canonical transition"
            )
        if condition is DriftCondition.ORACLE_RESET:
            actual_state = canonical_before
        turn = DriftTurnInput(
            scenario_id=scenario.id,
            checkpoint=checkpoint,
            horizon=horizon,
            condition=condition,
            instruction=planned.instruction,
            observed_state=scenario.public_state(actual_state),
            canonical_state_summary=(
                scenario.compact_summary(canonical_before)
                if condition is DriftCondition.STATE_ANCHORED
                else None
            ),
        )
        actual_before = copy.deepcopy(actual_state)
        actual_call = await action_provider(turn)
        actual_state, actual_valid = scenario.apply(actual_state, actual_call)
        violations = scenario.invariant_violations(actual_state)
        expected_public = scenario.public_state(canonical_state)
        actual_public = scenario.public_state(actual_state)
        distance = exact_state_distance(expected_public, actual_public)
        expected_changed = _changed_paths(
            scenario.public_state(canonical_before), expected_public
        )
        actual_changed = _changed_paths(
            scenario.public_state(actual_before), actual_public
        )
        rollback = planned.canonical_call.tool == "rollback"
        transition = DriftTransition(
            checkpoint=checkpoint,
            instruction=planned.instruction,
            expected_call=planned.canonical_call,
            actual_call=actual_call,
            expected_state=expected_public,
            actual_state=actual_public,
            expected_state_fingerprint=canonical_fingerprint(expected_public),
            actual_state_fingerprint=canonical_fingerprint(actual_public),
            state_distance=distance,
            state_fidelity=1.0 - distance,
            exact=distance == 0,
            valid_tool_call=actual_valid,
            invariant_violations=violations,
            collateral_mutations=sorted(actual_changed - expected_changed),
            rollback=rollback,
            rollback_correct=(distance == 0 if rollback else None),
        )
        transitions.append(transition)
        if on_checkpoint is not None:
            await on_checkpoint(checkpoint, horizon)

    metrics = score_transitions(transitions)
    return DriftRunResult(
        scenario_id=scenario.id,
        scenario_revision=scenario.revision,
        seed=seed,
        condition=condition,
        horizon=horizon,
        initial_state=scenario.public_state(initial),
        final_expected_state=scenario.public_state(canonical_state),
        final_actual_state=scenario.public_state(actual_state),
        transitions=transitions,
        metrics=metrics,
    )


def score_transitions(transitions: list[DriftTransition]) -> DriftMetrics:
    if not transitions:
        raise ValueError("at least one transition is required")
    correctness = [transition.exact for transition in transitions]
    fidelity = [transition.state_fidelity for transition in transitions]
    first_divergence = next(
        (transition.checkpoint for transition in transitions if not transition.exact),
        None,
    )
    survived = True
    survival_curve = []
    for exact in correctness:
        survived = survived and exact
        survival_curve.append(float(survived))
    recovery_opportunities = sum(
        not transitions[index].exact for index in range(len(transitions) - 1)
    )
    recoveries = sum(
        not transitions[index].exact and transitions[index + 1].exact
        for index in range(len(transitions) - 1)
    )
    rollbacks = [
        transition.rollback_correct
        for transition in transitions
        if transition.rollback_correct is not None
    ]
    return DriftMetrics(
        final_success=correctness[-1],
        transition_accuracy=sum(correctness) / len(correctness),
        per_transition_correctness=correctness,
        state_fidelity=fidelity,
        first_divergence_checkpoint=first_divergence,
        survival_curve=survival_curve,
        area_under_state_fidelity_curve=sum(fidelity) / len(fidelity),
        recovery_rate=(
            recoveries / recovery_opportunities if recovery_opportunities else 0.0
        ),
        invariant_violations=sum(
            len(transition.invariant_violations) for transition in transitions
        ),
        invalid_tool_calls=sum(
            not transition.valid_tool_call for transition in transitions
        ),
        collateral_mutations=sum(
            len(transition.collateral_mutations) for transition in transitions
        ),
        rollback_correctness=(
            sum(bool(value) for value in rollbacks) / len(rollbacks)
            if rollbacks
            else None
        ),
        divergence_growth_slope=_linear_slope([1.0 - value for value in fidelity]),
    )


def exact_state_distance(expected: object, actual: object) -> float:
    expected_leaves = _leaf_paths(expected)
    actual_leaves = _leaf_paths(actual)
    paths = set(expected_leaves) | set(actual_leaves)
    if not paths:
        return 0.0
    differing = sum(
        expected_leaves.get(path) != actual_leaves.get(path) for path in paths
    )
    return differing / len(paths)


def reliability_horizon(
    final_success_by_horizon: dict[int, list[bool]], threshold: float
) -> int | None:
    if not 0 <= threshold <= 1:
        raise ValueError("reliability threshold must be between zero and one")
    eligible = [
        horizon
        for horizon, values in final_success_by_horizon.items()
        if values and sum(values) / len(values) >= threshold
    ]
    return max(eligible, default=None)


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("value must be a positive integer")
    return value


def _leaf_paths(value: object, prefix: str = "$") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        leaves: dict[str, object] = {}
        for key in sorted(value):
            leaves.update(_leaf_paths(value[key], f"{prefix}.{key}"))
        return leaves
    if isinstance(value, list):
        if not value:
            return {prefix: []}
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_leaf_paths(item, f"{prefix}[{index}]"))
        return leaves
    return {prefix: value}


def _changed_paths(before: object, after: object) -> set[str]:
    before_leaves = _leaf_paths(before)
    after_leaves = _leaf_paths(after)
    paths = set(before_leaves) | set(after_leaves)
    return {path for path in paths if before_leaves.get(path) != after_leaves.get(path)}


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) + 1) / 2
    mean_y = sum(values) / len(values)
    numerator = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values, start=1)
    )
    denominator = sum((index - mean_x) ** 2 for index in range(1, len(values) + 1))
    return numerator / denominator if denominator else 0.0
