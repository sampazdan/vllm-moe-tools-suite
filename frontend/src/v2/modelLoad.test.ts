import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { JobPhaseRecord, JobRecord } from "../types.ts";
import {
  canRetryModelLoad,
  currentModelLoadPhase,
  failedModelLoadPhase,
  failedModelLoadStage,
  formatDuration,
  modelLoadElapsedMs,
  modelLoadPhaseElapsedMs,
  phaseCounters,
} from "./modelLoad.ts";

const observed: JobPhaseRecord = {
  phase: "waiting_for_readiness",
  status: "active",
  observability: "observed",
  source: "managed_runtime",
  started_at: "2026-08-12T00:00:03Z",
  completed_at: null,
  detail: "Waiting for readiness",
  bytes_current: null,
  bytes_total: null,
  files_current: null,
  files_total: null,
  failure_code: null,
  recovery_action: null,
  diagnostics: null,
};

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "load-1",
    kind: "model_load",
    status: "running",
    progress_current: 0,
    progress_total: 1,
    result_id: "session-1",
    error: null,
    phase_history: [observed],
    created_at: "2026-08-12T00:00:00Z",
    started_at: "2026-08-12T00:00:01Z",
    completed_at: null,
    ...overrides,
  };
}

describe("model-load lifecycle presentation", () => {
  it("uses the active observed phase and computes durable elapsed time", () => {
    const running = job();

    assert.equal(currentModelLoadPhase(running), observed);
    assert.equal(modelLoadElapsedMs(running, Date.parse("2026-08-12T00:00:11Z")), 10_000);
    assert.equal(
      modelLoadPhaseElapsedMs(observed, Date.parse("2026-08-12T00:00:11Z")),
      8_000,
    );
    assert.equal(formatDuration(68_000), "1m 8s");
  });

  it("keeps opaque child phases explicitly unavailable without invented counters", () => {
    const unavailable: JobPhaseRecord = {
      ...observed,
      phase: "downloading",
      status: "unavailable",
      observability: "unavailable",
      completed_at: observed.started_at,
      detail: "The managed runtime does not expose download progress.",
    };

    assert.equal(modelLoadPhaseElapsedMs(unavailable), null);
    assert.equal(phaseCounters(unavailable), null);
    assert.equal(formatDuration(null), "Unavailable");
  });

  it("exposes structured recovery and blocks unsafe cleanup-failure retries", () => {
    const failedPhase: JobPhaseRecord = {
      ...observed,
      phase: "failed",
      status: "failed",
      completed_at: "2026-08-12T00:00:12Z",
      failure_code: "readiness_failed",
      recovery_action: "Inspect diagnostics, then retry.",
      diagnostics: "readiness failed\ntrace detail",
    };
    const failed = job({
      status: "failed",
      error: "readiness failed",
      phase_history: [{ ...observed, status: "failed" }, failedPhase],
      completed_at: "2026-08-12T00:00:12Z",
    });

    assert.equal(failedModelLoadPhase(failed), failedPhase);
    assert.equal(failedModelLoadStage(failed)?.phase, "waiting_for_readiness");
    assert.equal(canRetryModelLoad(failed), true);
    assert.equal(
      canRetryModelLoad(job({
        status: "failed",
        phase_history: [{ ...failedPhase, failure_code: "cleanup_failed" }],
      })),
      false,
    );
  });

  it("formats structured byte and file counters only when the runtime provides them", () => {
    assert.equal(
      phaseCounters({
        ...observed,
        phase: "downloading",
        bytes_current: 1_048_576,
        bytes_total: 2_097_152,
        files_current: 2,
        files_total: 4,
      }),
      "1.0 MiB / 2.0 MiB · 2 / 4 files",
    );
  });
});
