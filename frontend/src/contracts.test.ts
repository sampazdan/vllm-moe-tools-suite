import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluationContractPayload,
  executionPolicyPayload,
  generationConfigPayload,
  runItemKey,
  withGenerationConfig,
} from "./contracts.ts";

test("run item identity preserves repeated attempts", () => {
  assert.notEqual(
    runItemKey({ item_id: "problem-1", attempt: 1 }),
    runItemKey({ item_id: "problem-1", attempt: 2 }),
  );
});

test("execution payload strips presentation-only generation fields", () => {
  const payload = executionPolicyPayload({
    max_tokens: 512,
    generation_max_tokens: 2048,
    timeout_seconds: 120,
    temperature: 0.2,
    seed: 42,
    enable_thinking: true,
    reasoning_visibility: "compact",
  });
  assert.equal("temperature" in payload, false);
  assert.equal("reasoning_visibility" in payload, false);
  assert.equal("generation_max_tokens" in payload, false);
  assert.equal(payload.max_tokens, 512);
  assert.equal(payload.per_item_timeout_seconds, 120);
});

test("generation max and whole-run metered-token cap remain independent", () => {
  const benchmarkGeneration = {
    temperature: 0,
    max_tokens: 2048,
    seed: 7,
    enable_thinking: true,
  };
  const policy = withGenerationConfig(
    {
      max_tokens: null,
      timeout_seconds: 120,
    },
    benchmarkGeneration,
  );

  assert.equal(generationConfigPayload(policy, benchmarkGeneration).max_tokens, 2048);
  assert.equal(executionPolicyPayload(policy).max_tokens, null);

  const capped = { ...policy, max_tokens: 10_000, generation_max_tokens: 1024 };
  assert.equal(generationConfigPayload(capped, benchmarkGeneration).max_tokens, 1024);
  assert.equal(executionPolicyPayload(capped).max_tokens, 10_000);
});

test("evaluation payload removes stale response fingerprints", () => {
  const payload = evaluationContractPayload({
    name: "Judge assisted",
    criteria: [
      {
        id: "correctness",
        kind: "exact",
        label: "Correct",
        visibility: "public",
        required: true,
        weight: 1,
        case_sensitive: true,
        strip_whitespace: true,
      },
    ],
    aggregation: "all_required",
    pass_threshold: 1,
    judge: {
      provider: "anthropic",
      model: "frontier-model",
      mode: "single",
      rubric: "Assess correctness.",
      pass_threshold: 0.7,
      repetitions: 1,
      max_output_tokens: 512,
      input_cost_per_million_usd: 3,
      output_cost_per_million_usd: 15,
      fingerprint: "stale",
      rubric_hash: "stale",
    },
    judge_weight: 0.25,
    judge_can_override_deterministic_failure: false,
    fingerprint: "stale",
  });
  assert.equal("fingerprint" in payload, false);
  assert.equal("fingerprint" in (payload.judge ?? {}), false);
  assert.equal(payload.judge?.input_cost_per_million_usd, 3);
  assert.equal(payload.judge?.output_cost_per_million_usd, 15);
  assert.equal(payload.judge_weight, 0.25);
  assert.equal(payload.criteria[0].verifier_command, null);
});
