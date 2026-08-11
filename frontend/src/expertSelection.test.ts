import assert from "node:assert/strict";
import test from "node:test";

import {
  createFullProfile,
  normalizeEngagementMatrix,
  observedMassRetained,
  profileSelectionCount,
  selectByCumulativeMass,
  selectGlobalTopExperts,
  selectTopExperts,
  toggleProfileExpert,
} from "./expertSelection.ts";

test("engagement normalization preserves rank and removes run volume", () => {
  const normalized = normalizeEngagementMatrix([[10, 30], [20, 40]]);
  assert.ok(Math.abs(normalized.flat().reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.deepEqual(
    normalized.flat().map((value) => Math.round(value * 100)),
    [10, 30, 20, 40],
  );
});

test("top expert selection is independently ranked per layer", () => {
  const full = createFullProfile([1, 3], 4);
  const selected = selectTopExperts(
    full,
    [1, 3],
    [[0.1, 0.8, 0.2, 0.3], [8, 2, 7, 1]],
    2,
  );
  assert.deepEqual(selected.layers["1"].keep, [1, 3]);
  assert.deepEqual(selected.layers["3"].keep, [0, 2]);
  assert.equal(profileSelectionCount(selected, [1, 3], 4), 4);
});

test("global budget preserves the minimum in every layer", () => {
  const profile = selectGlobalTopExperts(
    createFullProfile([1, 2], 4),
    [1, 2],
    [[100, 2, 1, 0], [10, 9, 8, 7]],
    5,
    2,
  );
  assert.equal(profileSelectionCount(profile, [1, 2], 4), 5);
  assert.deepEqual(profile.layers["1"].keep, [0, 1]);
  assert.deepEqual(profile.layers["2"].keep, [0, 1, 2]);
});

test("cumulative mass selection crosses the global threshold safely", () => {
  const profile = selectByCumulativeMass(
    createFullProfile([1, 2], 3),
    [1, 2],
    [[6, 2, 2], [4, 3, 3]],
    0.75,
    1,
  );
  assert.deepEqual(profile.layers["1"].keep, [0]);
  assert.deepEqual(profile.layers["2"].keep, [0, 1, 2]);
  assert.ok(observedMassRetained(profile, [1, 2], 3, [[6, 2, 2], [4, 3, 3]]) >= 0.75);
});

test("manual toggles preserve the top-k floor", () => {
  let profile = selectTopExperts(
    createFullProfile([2], 4),
    [2],
    [[4, 3, 2, 1]],
    2,
  );
  profile = toggleProfileExpert(profile, 2, 0, 4, 2);
  assert.deepEqual(profile.layers["2"].keep, [0, 1]);
  profile = toggleProfileExpert(profile, 2, 3, 4, 2);
  assert.deepEqual(profile.layers["2"].keep, [0, 1, 3]);
  profile = toggleProfileExpert(profile, 2, 0, 4, 2);
  assert.deepEqual(profile.layers["2"].keep, [1, 3]);
});

test("retained mass is calculated over selected layer experts", () => {
  const profile = selectTopExperts(
    createFullProfile([1, 2], 3),
    [1, 2],
    [[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]],
    1,
  );
  assert.equal(observedMassRetained(profile, [1, 2], 3, [[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]]), 0.7);
});
