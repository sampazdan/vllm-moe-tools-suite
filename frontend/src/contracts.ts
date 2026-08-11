import type {
  EvaluationContract,
  ExecutionPolicy,
  GenerationConfig,
  RunItemResult,
} from "./types";

export function runItemKey(
  item: Pick<RunItemResult, "item_id" | "attempt">,
) {
  return `${item.item_id}\u0000${item.attempt}`;
}

export function executionPolicyPayload(policy: ExecutionPolicy): ExecutionPolicy {
  return {
    version: 1,
    attempts: policy.attempts ?? 1,
    concurrency: policy.concurrency ?? 1,
    timeout_seconds: policy.timeout_seconds,
    per_item_timeout_seconds:
      policy.per_item_timeout_seconds ?? policy.timeout_seconds,
    max_turns: policy.max_turns ?? null,
    max_commands: policy.max_commands ?? null,
    max_tokens: policy.max_tokens,
    max_cost_usd: policy.max_cost_usd ?? null,
    fail_fast: policy.fail_fast ?? false,
  };
}

export function withGenerationConfig(
  policy: ExecutionPolicy,
  generation: GenerationConfig,
): ExecutionPolicy {
  return {
    ...policy,
    generation_max_tokens: generation.max_tokens,
    temperature: generation.temperature,
    seed: generation.seed,
    enable_thinking: generation.enable_thinking,
    reasoning_visibility: generation.enable_thinking ? "compact" : "off",
  };
}

export function generationConfigPayload(
  policy: ExecutionPolicy,
  fallback: GenerationConfig,
): GenerationConfig {
  return {
    temperature: policy.temperature ?? fallback.temperature,
    max_tokens: policy.generation_max_tokens ?? fallback.max_tokens,
    seed: policy.seed ?? fallback.seed,
    enable_thinking: policy.enable_thinking ?? fallback.enable_thinking,
  };
}

export function evaluationContractPayload(
  contract: EvaluationContract,
): EvaluationContract {
  const judge = contract.judge
    ? {
        provider: contract.judge.provider,
        model: contract.judge.model,
        mode: contract.judge.mode,
        rubric: contract.judge.rubric,
        pass_threshold: contract.judge.pass_threshold,
        repetitions: contract.judge.repetitions,
        temperature: contract.judge.temperature ?? null,
        max_output_tokens: contract.judge.max_output_tokens,
        input_cost_per_million_usd:
          contract.judge.input_cost_per_million_usd ?? null,
        output_cost_per_million_usd:
          contract.judge.output_cost_per_million_usd ?? null,
      }
    : null;
  return {
    version: 1,
    name: contract.name,
    description: contract.description ?? "",
    criteria: contract.criteria.map((criterion) => ({
      id: criterion.id,
      label: criterion.label,
      description: criterion.description ?? "",
      kind: criterion.kind,
      visibility: criterion.visibility,
      required: criterion.required,
      weight: criterion.weight,
      case_sensitive: criterion.case_sensitive,
      strip_whitespace: criterion.strip_whitespace,
      expected: criterion.expected ?? null,
      pattern: criterion.pattern ?? null,
      numeric_tolerance: criterion.numeric_tolerance ?? 0,
      json_schema: criterion.json_schema ?? null,
      verifier_command: criterion.verifier_command ?? null,
      verifier_timeout_seconds: criterion.verifier_timeout_seconds ?? 300,
    })),
    aggregation: contract.aggregation,
    pass_threshold: contract.pass_threshold,
    judge,
    judge_weight: judge ? contract.judge_weight : 0,
    judge_can_override_deterministic_failure:
      contract.judge_can_override_deterministic_failure,
  };
}
