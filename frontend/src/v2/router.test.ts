import assert from "node:assert/strict";
import test from "node:test";

import { parseV2Route, v2Href } from "./router.ts";

test("V2 routes unify the primary research surfaces", () => {
  assert.equal(parseV2Route("/").name, "experiments");
  assert.equal(parseV2Route("/experiments").name, "experiments");
  assert.deepEqual(parseV2Route("/experiments/new", "?workload=drift%2Fledger"), {
    name: "experimentNew",
    path: "/experiments/new",
    workloadId: "drift/ledger",
  });
  assert.deepEqual(parseV2Route("/experiments/paired%20one"), {
    name: "experiment",
    path: "/experiments/paired%20one",
    experimentId: "paired one",
  });
  assert.equal(parseV2Route("/workload-library").name, "workloads");
  assert.equal(parseV2Route("/profiles/profile-1").name, "profile");
  assert.equal(parseV2Route("/models").name, "models");
  assert.equal(parseV2Route("/settings").name, "settings");
});

test("legacy research URLs remain delegated while profile detail is V2", () => {
  assert.equal(parseV2Route("/benchmarks/mmlu-pro").name, "legacy");
  assert.equal(parseV2Route("/agent-runs/run-1").name, "legacy");
  assert.equal(parseV2Route("/profiles/new", "?run=run-1").name, "legacy");
  assert.equal(parseV2Route("/profiles/profile-1").name, "profile");
});

test("V2 hrefs encode opaque identifiers", () => {
  assert.equal(v2Href("experiment", "run one"), "/experiments/run%20one");
  assert.equal(
    v2Href("newExperiment", "drift/ledger"),
    "/experiments/new?workload=drift%2Fledger",
  );
});
