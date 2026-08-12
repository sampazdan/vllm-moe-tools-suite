import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_SELECTED_WORKLOAD_UNITS,
  defaultWorkloadUnitSelection,
  MAX_SELECTED_WORKLOAD_UNITS,
  orderedWorkloadUnitSelection,
  plannedExperimentUnits,
  selectAllWorkloadUnits,
} from "./cohortSelection.ts";

test("large cohorts start bounded and never select every MMLU-Pro test item", () => {
  const mmluPro = Array.from(
    { length: 12_032 },
    (_, index) => `mmlu-pro-test-${index}`,
  );

  const initial = defaultWorkloadUnitSelection(mmluPro);
  const all = selectAllWorkloadUnits(mmluPro);

  assert.equal(initial.length, DEFAULT_SELECTED_WORKLOAD_UNITS);
  assert.equal(all.length, MAX_SELECTED_WORKLOAD_UNITS);
  assert.ok(all.length < mmluPro.length);
});

test("selected cohort IDs follow immutable workload order and are deduplicated", () => {
  const selected = new Set(["item-c", "item-a", "unknown"]);

  assert.deepEqual(
    orderedWorkloadUnitSelection(
      ["item-a", "item-b", "item-c", "item-a"],
      selected,
    ),
    ["item-a", "item-c"],
  );
});

test("planned units use selected scenarios, conditions, and lanes", () => {
  assert.equal(plannedExperimentUnits({
    conditionCount: 3,
    kind: "state_drift",
    paired: true,
    selectedUnitCount: 4,
  }), 24);
  assert.equal(plannedExperimentUnits({
    conditionCount: 3,
    kind: "answer",
    paired: false,
    selectedUnitCount: 4,
  }), 4);
});
