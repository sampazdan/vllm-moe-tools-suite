import assert from "node:assert/strict";
import test from "node:test";

import { parseRoute, routeHref } from "./router.ts";

test("parses every command-center route", () => {
  assert.deepEqual(parseRoute("/"), { name: "dashboard", path: "/" });
  assert.equal(parseRoute("/benchmarks").name, "benchmarks");
  assert.deepEqual(parseRoute("/benchmarks/mmlu%2Fpro"), {
    name: "benchmark",
    path: "/benchmarks/mmlu%2Fpro",
    benchmarkId: "mmlu/pro",
  });
  assert.deepEqual(parseRoute("/runs/run-1/"), {
    name: "run",
    path: "/runs/run-1",
    runId: "run-1",
  });
  assert.deepEqual(parseRoute("/agent-runs/agent-1"), {
    name: "agentRun",
    path: "/agent-runs/agent-1",
    runId: "agent-1",
  });
  assert.deepEqual(parseRoute("/agent-comparisons/base-1/candidate-1"), {
    name: "agentComparison",
    path: "/agent-comparisons/base-1/candidate-1",
    baselineRunId: "base-1",
    candidateRunId: "candidate-1",
  });
  assert.deepEqual(
    parseRoute("/profiles/new", "?trial=trial-1&trial=trial-2&run=run-1&profile=profile-1"),
    {
      name: "profileStudio",
      path: "/profiles/new",
      runId: "run-1",
      trialIds: ["trial-1", "trial-2"],
      profileId: "profile-1",
    },
  );
  assert.equal(parseRoute("/profiles/profile-1").name, "profile");
  assert.equal(parseRoute("/comparisons/comparison-1").name, "comparison");
  assert.equal(parseRoute("/missing").name, "notFound");
});

test("encodes dynamic route IDs", () => {
  assert.equal(routeHref("benchmark", "mmlu/pro"), "/benchmarks/mmlu%2Fpro");
  assert.equal(routeHref("agentRun", "run one"), "/agent-runs/run%20one");
});
