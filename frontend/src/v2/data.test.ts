import assert from "node:assert/strict";
import test from "node:test";

import {
  buildExperimentCreateRequest,
  normalizeEvent,
  normalizeExperiment,
  normalizeModel,
  normalizeWorkload,
} from "./data.ts";

test("V2 create payload enables answer and coding without drift placeholders", () => {
  const base = {
    agentId: "bash-json-v1",
    candidateProfileId: null,
    conditions: ["chained"],
    horizon: 8,
    modelId: "model-1",
    name: "Ready task",
    sandboxProviderId: "fake",
    seed: 3,
    workloadUnitIds: ["task-1"],
  };
  const answer = buildExperimentCreateRequest({
    ...base,
    kind: "answer",
    workloadId: "answer:fixture:item-1",
  });
  const coding = buildExperimentCreateRequest({
    ...base,
    kind: "coding",
    workloadId: "coding:pack:task-1",
  });

  assert.equal(answer.scenario_ids, undefined);
  assert.equal(answer.agent_id, undefined);
  assert.equal(coding.scenario_ids, undefined);
  assert.equal(coding.agent_id, "bash-json-v1");
  assert.equal(coding.sandbox_provider_id, "fake");
  assert.deepEqual(coding.seeds, [3]);
});

test("model normalization preserves an undiscovered topology as null", () => {
  const model = normalizeModel({
    id: "unverified-model",
    display_name: "Unverified model",
    enabled: false,
    revision: null,
    topology: null,
    notes: "Topology discovery pending.",
  });

  assert.equal(model.topology, null);
});

test("task-level workload DTO preserves readiness evidence", () => {
  const workload = normalizeWorkload({
    id: "coding:aider-expansion:task-8",
    kind: "coding",
    name: "Repair the parser",
    description: "Fix a parser regression without viewing hidden tests.",
    source: "https://example.test/aider",
    revision: "pinned-revision",
    content_fingerprint: "a".repeat(64),
    ready: false,
    blocked_reason: "oracle/no-op qualification is incomplete",
    unit_ids: ["task-8"],
    parent_id: "aider-expansion",
    task_id: "task-8",
    language: "python",
    difficulty: "medium",
    expected_horizon: 8,
    tools: ["bash"],
    runtime: `python@sha256:${"b".repeat(64)}; 300s timeout`,
    preparation_status: "not_eligible",
    public_problem_statement: "Repair the parser.",
    public_success_criteria: ["Protected verifier passes."],
    metadata: { tags: ["parser", "canary"] },
  });

  assert.equal(workload.title, "Repair the parser");
  assert.equal(workload.pack_id, "aider-expansion");
  assert.equal(workload.source_revision, "pinned-revision");
  assert.equal(workload.unit_count, 1);
  assert.deepEqual(workload.tags, ["parser", "canary"]);
  assert.equal(workload.runtime?.timeout_seconds, 300);
  assert.equal(workload.runtime?.image_digest, `sha256:${"b".repeat(64)}`);
  assert.deepEqual(workload.blocked_reasons, ["oracle/no-op qualification is incomplete"]);
});

test("ExperimentDetail DTO aligns per-lane run units and drift checkpoints", () => {
  const workload = {
    id: "state-drift-v1",
    kind: "state_drift",
    name: "State Drift Bench",
    source: "first-party",
    revision: "1.0.0",
    content_fingerprint: "c".repeat(64),
    ready: true,
    unit_ids: ["ledger-reconciliation-v1"],
  };
  const context = (profile: boolean) => ({
    context_id: profile ? "profile-context" : "baseline-context",
    kind: profile ? "expert_mask" : "baseline",
    topology_fingerprint: "d".repeat(64),
    profile_id: profile ? "profile-1" : null,
    profile_fingerprint: profile ? "e".repeat(64) : null,
    context_fingerprint: profile ? "f".repeat(64) : "0".repeat(64),
  });
  const transition = (exact: boolean) => ({
    checkpoint: 1,
    instruction: "Post the adjustment.",
    expected_state: { ledger: { balance: 10 } },
    actual_state: { ledger: { balance: exact ? 10 : 8 } },
    state_distance: exact ? 0 : .5,
    state_fidelity: exact ? 1 : .5,
    exact,
    valid_tool_call: true,
    invariant_violations: [],
  });
  const detail = normalizeExperiment({
    experiment: {
      id: "experiment-1",
      job_id: "job-1",
      name: "Ledger paired run",
      model_id: "Qwen/Qwen3.6-35B-A3B-FP8",
      workload,
      status: "completed",
      completed_units: 2,
      passed_units: 1,
      total_units: 2,
      lanes: [
        { id: "lane-b", role: "baseline", label: "Baseline", context: context(false), status: "completed" },
        { id: "lane-c", role: "candidate", label: "Named mask", context: context(true), status: "completed" },
      ],
      comparison: { excess_compounding_penalty: .2 },
      created_at: "2026-08-12T00:00:00Z",
    },
    workload_runs: [
      { lane_id: "lane-b", completed_units: 1, passed_units: 1, total_units: 1, aggregate_metrics: { mean_state_fidelity_auc: 1 } },
      { lane_id: "lane-c", completed_units: 1, passed_units: 0, total_units: 1, aggregate_metrics: { mean_state_fidelity_auc: .5 } },
    ],
    units: [
      {
        id: "unit-b",
        lane_id: "lane-b",
        ordinal: 0,
        unit_key: "ledger-reconciliation-v1:chained:h8:s0:baseline",
        scenario_id: "ledger-reconciliation-v1",
        condition: "chained",
        horizon: 8,
        seed: 0,
        status: "passed",
        result: { transitions: [transition(true)], routing: { context: "baseline" } },
        evaluation: { passed: true, score: 1, metrics: { state_fidelity_auc: 1 } },
        performance: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14, latency_ms: 200, tokens_per_second: 20 },
      },
      {
        id: "unit-c",
        lane_id: "lane-c",
        ordinal: 1,
        unit_key: "ledger-reconciliation-v1:chained:h8:s0:candidate",
        scenario_id: "ledger-reconciliation-v1",
        condition: "chained",
        horizon: 8,
        seed: 0,
        status: "failed",
        result: { transitions: [transition(false)], routing: { context: "profile" } },
        evaluation: { passed: false, score: .5, metrics: { state_fidelity_auc: .5 } },
        performance: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16, latency_ms: 250, tokens_per_second: 20 },
      },
    ],
  });

  assert.equal(detail.progress_current, 2);
  assert.equal(detail.progress_total, 2);
  assert.equal(detail.model_name, "Qwen3.6 35B A3B FP8");
  assert.equal(detail.units.length, 1);
  assert.equal(detail.units[0].baseline?.checkpoints[0].status, "exact");
  assert.equal(detail.units[0].candidate?.checkpoints[0].status, "diverged");
  assert.equal(detail.units[0].candidate?.checkpoints[0].actual_state.ledger.balance, 8);
  assert.equal(detail.lanes[1].progress_current, 1);
  assert.equal(detail.drift_metrics.candidate?.area_under_fidelity_curve, .5);
  assert.equal(detail.drift_metrics.candidate?.excess_mask_drift, .2);
});

test("answer ExperimentDetail exposes nested prompt, output, and scorer truth", () => {
  const detail = normalizeExperiment({
    experiment: {
      id: "answer-experiment",
      job_id: "answer-job",
      name: "Answer run",
      model_id: "Qwen/Qwen3.6-35B-A3B-FP8",
      workload: {
        id: "answer:fixture:item-1",
        kind: "answer",
        name: "Fixture item",
        source: "fixture",
        revision: "1",
        content_fingerprint: "a".repeat(64),
        ready: true,
        unit_ids: ["item-1"],
      },
      status: "completed",
      completed_units: 1,
      passed_units: 0,
      total_units: 1,
      lanes: [{ id: "lane-b", role: "baseline", label: "Baseline", context: {}, status: "completed" }],
      created_at: "2026-08-12T00:00:00Z",
    },
    workload_runs: [{ lane_id: "lane-b", completed_units: 1, passed_units: 0, total_units: 1 }],
    units: [{
      id: "unit-b",
      lane_id: "lane-b",
      ordinal: 0,
      unit_key: "item-1:s0:baseline",
      seed: 0,
      status: "unscored",
      result: {
        kind: "answer",
        item: { prompt: "What is two plus two?", output: "Four", scoring: "ungraded" },
      },
      evaluation: { passed: null, score: null },
    }],
  });

  assert.equal(detail.units[0].prompt, "What is two plus two?");
  assert.equal(detail.units[0].baseline?.output, "Four");
  assert.equal(detail.units[0].baseline?.passed, null);
  assert.equal(detail.units[0].baseline?.criteria[0].label, "Evaluation contract");
});

test("answer result normalization accepts flattened truth and preserves UNSCORED criteria", () => {
  const detail = normalizeExperiment({
    experiment: {
      id: "answer-flat",
      name: "Qualitative answer",
      model_id: "model-1",
      workload: {
        id: "answer:fixture:item-2",
        kind: "answer",
        name: "Qualitative item",
        ready: true,
        public_problem_statement: "Explain the tradeoff.",
      },
      execution_config: {},
      status: "completed",
      completed_units: 1,
      passed_units: 0,
      total_units: 1,
      lanes: [{ id: "lane-b", role: "baseline", label: "Baseline", context: {}, status: "completed" }],
      created_at: "2026-08-12T00:00:00Z",
    },
    units: [{
      id: "answer-unit",
      lane_id: "lane-b",
      unit_key: "item-2:s0:baseline",
      seed: 0,
      status: "unscored",
      result: {
        kind: "answer",
        prompt: "Explain the tradeoff.",
        output: { summary: "It depends on latency." },
      },
      evaluation: {
        passed: null,
        score: null,
        deterministic: {
          criteria: [{
            criterion_id: "qualitative",
            passed: null,
            explanation: "Human review is required.",
          }],
        },
      },
    }],
  });

  assert.equal(detail.units[0].run_unit_ids.baseline, "answer-unit");
  assert.equal(detail.units[0].prompt, "Explain the tradeoff.");
  assert.equal(detail.units[0].baseline?.output, '{\n  "summary": "It depends on latency."\n}');
  assert.equal(detail.units[0].baseline?.status, "unscored");
  assert.equal(detail.units[0].baseline?.criteria[0].label, "Qualitative");
  assert.equal(detail.units[0].baseline?.criteria[0].passed, null);
  assert.equal(detail.units[0].baseline?.criteria[0].detail, "Human review is required.");
});

test("drift detail preserves routing arrays and aligns checkpoint comparison evidence", () => {
  const comparison = {
    routing_checkpoint_pairs: [{
      scenario_id: "ledger-reconciliation-v1",
      condition: "chained",
      horizon: 8,
      seed: 0,
      checkpoint: 2,
      selection_overlap: .72,
      routing_mass_js_divergence: .18,
      at_candidate_first_divergence: true,
      largest_selection_shifts: [{
        layer: 4,
        expert: 17,
        baseline_share: .08,
        candidate_share: .23,
        delta: .15,
      }],
    }],
  };
  const detail = normalizeExperiment({
    experiment: {
      id: "drift-routing",
      name: "Routing drift",
      model_id: "model-1",
      workload: { id: "state-drift-v1", kind: "state_drift", name: "State Drift Bench", ready: true },
      status: "completed",
      completed_units: 2,
      passed_units: 1,
      total_units: 2,
      lanes: [
        { id: "lane-b", role: "baseline", label: "Baseline", context: {}, status: "completed" },
        { id: "lane-c", role: "candidate", label: "Candidate", context: {}, status: "completed" },
      ],
      comparison,
      created_at: "2026-08-12T00:00:00Z",
    },
    units: ["lane-b", "lane-c"].map((laneId, index) => ({
      id: `unit-${index}`,
      lane_id: laneId,
      unit_key: `ledger-reconciliation-v1:chained:h8:s0:${index ? "candidate" : "baseline"}`,
      scenario_id: "ledger-reconciliation-v1",
      condition: "chained",
      horizon: 8,
      seed: 0,
      status: index ? "failed" : "passed",
      result: {
        transitions: [{ checkpoint: 2, instruction: "Adjust", exact: !index }],
        routing: [{ checkpoint: 2, selection_counts: [[1, 2]], routing_mass: [[.3, .7]] }],
      },
      evaluation: { passed: !index, score: index ? .5 : 1 },
    })),
  });

  assert.ok(Array.isArray(detail.units[0].candidate?.routing));
  assert.deepEqual(
    detail.units[0].candidate?.checkpoints[0].routing_evidence?.selection_counts,
    [[1, 2]],
  );
  const aligned = detail.units[0].candidate?.checkpoints[0].routing_comparison;
  assert.equal(aligned?.selection_overlap, .72);
  assert.equal(aligned?.at_candidate_first_divergence, true);
  assert.equal(aligned?.largest_selection_shifts[0].expert, 17);
});

test("run event DTO maps backend cursor fields without losing payload", () => {
  const event = normalizeEvent({
    experiment_id: "experiment-1",
    sequence: 7,
    kind: "current_checkpoint",
    phase: "checkpoint",
    message: "Checkpoint 2/8",
    lane_id: "lane-c",
    run_unit_id: "unit-c",
    checkpoint: 2,
    data: { context_fingerprint: "abc" },
    created_at: "2026-08-12T00:00:07Z",
  }, "experiment-1", 1);

  assert.equal(event.type, "current_checkpoint");
  assert.equal(event.unit_id, "unit-c");
  assert.equal(event.payload.context_fingerprint, "abc");
});
