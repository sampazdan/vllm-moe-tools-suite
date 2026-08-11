from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .bounded_regex import bounded_regex_search
from .domain import (
    CriterionResult,
    DeterministicScorerKind,
    EvaluationAggregation,
    EvaluationContract,
    EvaluationResult,
    ExecutionPolicy,
    LLMJudgeResult,
)

BenchmarkScorer = Callable[[str], bool | None]
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def validate_cost_policy_pricing(
    contract: EvaluationContract,
    policy: ExecutionPolicy,
) -> None:
    """Reject an external-judge spend cap that cannot be measured."""

    judge = contract.judge
    if policy.max_cost_usd is None or judge is None:
        return
    if (
        judge.input_cost_per_million_usd is None
        or judge.output_cost_per_million_usd is None
    ):
        raise ValueError("max_cost_usd requires explicit judge input and output prices")


def evaluate_output(
    contract: EvaluationContract,
    *,
    output: str,
    expected: str,
    benchmark_scorer: BenchmarkScorer,
    judge: LLMJudgeResult | None = None,
    precomputed_criteria: Mapping[str, CriterionResult] | None = None,
) -> EvaluationResult:
    """Evaluate one output without executing host commands.

    Verifier criteria are intentionally returned as errors here. They must be
    consumed by a sandbox executor so user-authored commands never run on the
    application or model-serving host.
    """
    supplied = precomputed_criteria or {}
    known_ids = {criterion.id for criterion in contract.criteria}
    unknown_ids = set(supplied) - known_ids
    if unknown_ids:
        raise ValueError(
            f"precomputed criterion IDs are not in the contract: {sorted(unknown_ids)}"
        )
    results = []
    for criterion in contract.criteria:
        precomputed = supplied.get(criterion.id)
        if precomputed is not None:
            if precomputed.kind is not criterion.kind:
                raise ValueError(
                    f"precomputed criterion {criterion.id!r} has the wrong kind"
                )
            results.append(
                precomputed.model_copy(
                    update={
                        "criterion_id": criterion.id,
                        "required": criterion.required,
                        "weight": criterion.weight,
                    }
                )
            )
            continue
        results.append(
            _evaluate_criterion(
                criterion,
                output=output,
                expected=expected,
                benchmark_scorer=benchmark_scorer,
            )
        )
    scored = [result for result in results if result.score is not None]
    total_weight = sum(result.weight for result in scored)
    deterministic_score = (
        sum((result.score or 0) * result.weight for result in scored) / total_weight
        if total_weight
        else None
    )
    deterministic_passed = _deterministic_passed(
        contract,
        results,
        deterministic_score,
    )

    judge_score = judge.score if judge is not None else None
    combined_score = deterministic_score
    if judge_score is not None:
        combined_score = (
            judge_score
            if deterministic_score is None
            else (
                deterministic_score * (1 - contract.judge_weight)
                + judge_score * contract.judge_weight
            )
        )

    passed = deterministic_passed
    if judge is not None:
        deterministic_failure_is_final = (
            deterministic_passed is False
            and not contract.judge_can_override_deterministic_failure
        )
        if deterministic_failure_is_final:
            passed = False
        elif combined_score is not None:
            passed = combined_score >= contract.pass_threshold
        else:
            passed = judge.passed

    return EvaluationResult(
        deterministic_score=deterministic_score,
        judge_score=judge_score,
        combined_score=combined_score,
        passed=passed,
        criteria=results,
        judge=judge,
    )


def _evaluate_criterion(
    criterion,
    *,
    output: str,
    expected: str,
    benchmark_scorer: BenchmarkScorer,
) -> CriterionResult:
    actual = output.strip() if criterion.strip_whitespace else output
    target = criterion.expected if criterion.expected is not None else expected
    if criterion.strip_whitespace:
        target = target.strip()
    comparable_actual = actual if criterion.case_sensitive else actual.casefold()
    comparable_target = target if criterion.case_sensitive else target.casefold()
    try:
        if criterion.kind is DeterministicScorerKind.BENCHMARK_DEFAULT:
            passed = benchmark_scorer(output)
        elif criterion.kind is DeterministicScorerKind.EXACT:
            passed = comparable_actual == comparable_target
        elif criterion.kind is DeterministicScorerKind.CONTAINS:
            passed = comparable_target in comparable_actual
        elif criterion.kind is DeterministicScorerKind.REGEX:
            pattern = criterion.pattern if criterion.pattern is not None else target
            passed = bounded_regex_search(
                pattern,
                actual,
                case_sensitive=criterion.case_sensitive,
            )
        elif criterion.kind is DeterministicScorerKind.NUMERIC:
            actual_number = _last_number(actual)
            expected_number = _last_number(target)
            passed = (
                actual_number is not None
                and expected_number is not None
                and abs(actual_number - expected_number)
                <= Decimal(str(criterion.numeric_tolerance))
            )
        elif criterion.kind is DeterministicScorerKind.MULTIPLE_CHOICE:
            passed = _multiple_choice_answer(actual) == _multiple_choice_answer(target)
        elif criterion.kind is DeterministicScorerKind.JSON:
            parsed = json.loads(actual)
            _validate_json_schema(parsed, criterion.json_schema or {})
            passed = True
        elif criterion.kind is DeterministicScorerKind.UNGRADED:
            passed = None
        else:
            return CriterionResult(
                criterion_id=criterion.id,
                kind=criterion.kind,
                required=criterion.required,
                weight=criterion.weight,
                error="verifier criteria require an isolated sandbox executor",
            )
    except (
        AttributeError,
        InvalidOperation,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        re.error,
    ) as error:
        return CriterionResult(
            criterion_id=criterion.id,
            kind=criterion.kind,
            required=criterion.required,
            weight=criterion.weight,
            score=0,
            passed=False,
            explanation="Criterion evaluation failed.",
            error=(str(error) or error.__class__.__name__)[:1000],
        )
    return CriterionResult(
        criterion_id=criterion.id,
        kind=criterion.kind,
        required=criterion.required,
        weight=criterion.weight,
        score=None if passed is None else float(passed),
        passed=passed,
        explanation=(
            "Not scored."
            if passed is None
            else "Criterion passed."
            if passed
            else "Criterion failed."
        ),
    )


def _deterministic_passed(
    contract: EvaluationContract,
    results: list[CriterionResult],
    score: float | None,
) -> bool | None:
    failed_required = any(
        result.required and result.passed is False for result in results
    )
    errors_in_required = any(
        result.required and result.error is not None for result in results
    )
    if failed_required or errors_in_required:
        return False
    if score is None:
        return None
    if contract.aggregation is EvaluationAggregation.ALL_REQUIRED:
        required = [result for result in results if result.required]
        if required and any(result.passed is not True for result in required):
            return False
    return score >= contract.pass_threshold


def _last_number(value: str) -> Decimal | None:
    matches = _NUMBER_PATTERN.findall(value)
    if not matches:
        return None
    return Decimal(matches[-1].replace(",", ""))


def _multiple_choice_answer(value: str) -> str:
    normalized = value.strip().upper()
    match = re.search(r"(?:^|\b)([A-Z])(?:\b|$)", normalized)
    return match.group(1) if match else normalized


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the deterministic JSON Schema subset exposed by the UI."""
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not equal the required constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not one of the allowed values")
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: (
            isinstance(item, int | float) and not isinstance(item, bool)
        ),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_type is not None:
        checker = type_checks.get(expected_type)
        if checker is None:
            raise ValueError(f"unsupported JSON schema type {expected_type!r}")
        if not checker(value):
            raise ValueError(f"{path} must be {expected_type}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required keys {missing}")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                _validate_json_schema(value[key], child_schema, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValueError(f"{path} has unexpected keys {sorted(extras)}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is longer than maxLength")
        if "pattern" in schema and not bounded_regex_search(
            schema["pattern"],
            value,
        ):
            raise ValueError(f"{path} does not match pattern")
