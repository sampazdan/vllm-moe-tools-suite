import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  ApiError,
  JobTerminalError,
  activeJobFromConflict,
  waitForJob,
} from "./api.ts";
import type { JobRecord, JobStatus } from "./types.ts";

function job(status: JobStatus, overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job-1",
    kind: "benchmark_run",
    status,
    progress_current: status === "completed" ? 3 : 0,
    progress_total: 3,
    result_id: status === "completed" ? "run-1" : null,
    error: null,
    phase_history: [],
    created_at: "2026-08-11T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

describe("waitForJob", () => {
  it("keeps polling through a transient fetch failure and clears recovery state", async () => {
    const waits: number[] = [];
    const progress: JobStatus[] = [];
    const connectionEvents: string[] = [];
    let poll = 0;

    const result = await waitForJob(job("queued"), {
      pollIntervalMs: 100,
      maxRetryIntervalMs: 1_000,
      wait: async (milliseconds) => {
        waits.push(milliseconds);
      },
      getJob: async () => {
        poll += 1;
        if (poll === 1) throw new TypeError("Failed to fetch");
        return poll === 2
          ? job("running", { progress_current: 1 })
          : job("completed");
      },
      onProgress: (nextJob) => progress.push(nextJob.status),
      onTransientError: (error, attempt) => {
        connectionEvents.push(error ? `${error.message}:${attempt}` : "reconnected");
      },
    });

    assert.equal(result.status, "completed");
    assert.deepEqual(waits, [100, 200, 100]);
    assert.deepEqual(progress, ["queued", "running", "completed"]);
    assert.deepEqual(connectionEvents, ["Failed to fetch:1", "reconnected"]);
  });

  it("preserves the terminal job when the server reports failure", async () => {
    const failed = job("failed", { error: "GPU process exited" });

    await assert.rejects(
      waitForJob(job("running"), {
        wait: async () => undefined,
        getJob: async () => failed,
      }),
      (error) => {
        assert.ok(error instanceof JobTerminalError);
        assert.equal(error.job, failed);
        assert.equal(error.message, "GPU process exited");
        return true;
      },
    );
  });

  it("adopts a structured active-job conflict and resumes it to completion", async () => {
    const active = job("running", {
      id: "server-job",
      kind: "model_load",
      result_id: "session-1",
    });
    const conflict = new ApiError("A job is already running", 409, {
      code: "active_job_conflict",
      message: "A job is already running",
      active_job: active,
      recovery_url: "/api/jobs/server-job",
    });

    const adopted = activeJobFromConflict(conflict);
    assert.equal(adopted, active);
    const result = await waitForJob(adopted, {
      wait: async () => undefined,
      getJob: async () => job("completed", {
        id: active.id,
        kind: active.kind,
        result_id: active.result_id,
      }),
    });

    assert.equal(result.status, "completed");
    assert.equal(result.result_id, "session-1");
  });
});
