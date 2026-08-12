import assert from "node:assert/strict";
import test from "node:test";

import {
  codingTimeline,
  driftRoutingSummaries,
  liveProgress,
  mergeRunEvents,
  stateDifferences,
} from "./progress.ts";
import type { ExperimentRecord, ExperimentUnit, LaneUnitResult, RunEvent } from "./types.ts";

function event(sequence: number, message = `event ${sequence}`): RunEvent {
  return {
    experiment_id: "experiment-1",
    sequence,
    type: "unit_completed",
    phase: "running",
    message,
    lane_id: "baseline",
    lane_role: "baseline",
    unit_id: `unit-${sequence}`,
    checkpoint: null,
    created_at: "2026-08-12T00:00:00Z",
    payload: {},
  };
}

test("durable event replay is ordered and idempotent", () => {
  const merged = mergeRunEvents(
    [event(2), event(1)],
    [event(2, "authoritative replay"), event(3)],
  );
  assert.deepEqual(merged.map((item) => item.sequence), [1, 2, 3]);
  assert.equal(merged[1].message, "authoritative replay");
});

test("state diff reports stable nested paths", () => {
  assert.deepEqual(
    stateDifferences(
      { account: { balance: 10, tags: ["open", "paid"] }, untouched: true },
      { account: { balance: 8, tags: ["open", "late"] }, untouched: true },
    ),
    [
      { path: "$.account.balance", expected: 10, actual: 8 },
      { path: "$.account.tags[1]", expected: "paid", actual: "late" },
    ],
  );
});

test("coding timeline combines inline steps with durable lane-scoped trajectory events", () => {
  const result = {
    kind: "coding",
    trajectory: [
      { id: "step-1", sequence: 1, type: "assistant", title: "Plan", content: "Inspect files" },
      { id: "step-2", sequence: 2, type: "tool", title: "Run tests", content: "npm test" },
    ],
  } as LaneUnitResult;
  const events: RunEvent[] = [
    {
      ...event(7),
      type: "trajectory_step",
      lane_id: "lane-b",
      lane_role: null,
      unit_id: "run-unit-b",
      payload: {
        trajectory_step: { id: "step-1", sequence: 1, type: "assistant", title: "Plan", content: "Inspect files" },
      },
    },
    {
      ...event(8),
      type: "trajectory_step",
      lane_id: "lane-c",
      lane_role: null,
      unit_id: "run-unit-c",
      payload: {
        trajectory_step: { id: "candidate-step", sequence: 1, type: "tool", title: "Wrong lane" },
      },
    },
  ];

  const timeline = codingTimeline(result, events, "run-unit-b", "lane-b");
  assert.deepEqual(timeline.map((step) => step.id), ["step-1", "step-2"]);
  assert.equal(timeline[0].event_sequence, 7);
  assert.equal(timeline.some((step) => step.id === "candidate-step"), false);
});

test("routing summaries prioritize candidate first divergence and preserve shifts", () => {
  const unit = {
    candidate: {
      checkpoints: [
        { index: 1, routing_comparison: { selection_overlap: .95, routing_mass_js_divergence: .01, at_candidate_first_divergence: false, largest_selection_shifts: [] } },
        { index: 2, routing_comparison: { selection_overlap: .6, routing_mass_js_divergence: .2, at_candidate_first_divergence: true, largest_selection_shifts: [{ layer: 5, expert: 9, baseline_share: .1, candidate_share: .3, delta: .2 }] } },
      ],
    },
  } as ExperimentUnit;

  const summaries = driftRoutingSummaries(unit);
  assert.equal(summaries[0].checkpoint, 2);
  assert.equal(summaries[0].firstDivergence, true);
  assert.deepEqual(summaries[0].shifts[0], {
    layer: 5,
    expert: 9,
    baselineShare: .1,
    candidateShare: .3,
    delta: .2,
  });
});

test("paired live progress aggregates both lanes instead of borrowing the first ETA", () => {
  const performance = (elapsed_ms: number, mean_tps: number) => ({
    elapsed_ms,
    eta_ms: 1,
    prompt_tokens: 10,
    reasoning_tokens: null,
    completion_tokens: 10,
    total_tokens: 20,
    current_tps: null,
    mean_tps,
    estimated_cost_usd: null,
  });
  const experiment = {
    phase: "running",
    active_unit_id: "candidate-unit",
    progress_current: 4,
    progress_total: 8,
    passed: 3,
    failed: 1,
    unscored: 0,
    lanes: [
      {
        id: "baseline",
        role: "baseline",
        progress_current: 3,
        progress_total: 4,
        passed: 3,
        failed: 0,
        unscored: 0,
        performance: performance(10_000, 20),
      },
      {
        id: "candidate",
        role: "candidate",
        progress_current: 1,
        progress_total: 4,
        passed: 0,
        failed: 1,
        unscored: 0,
        performance: performance(4_000, 10),
      },
    ],
  } as unknown as ExperimentRecord;
  experiment.lanes[1].performance.current_tps = 12;

  const progress = liveProgress(experiment, []);

  assert.equal(progress.completed, 4);
  assert.equal(progress.total, 8);
  assert.equal(progress.elapsedMs, 10_000);
  assert.equal(progress.etaMs, 10_000);
  assert.equal(progress.meanTps, 17.5);
  assert.equal(progress.currentTps, 12);
});
