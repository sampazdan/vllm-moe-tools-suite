import assert from "node:assert/strict";
import test from "node:test";

import { benchmarkSelectionInitialization } from "./selectionInitialization.ts";

test("benchmark selection initializes once after its async item page arrives", () => {
  assert.equal(
    benchmarkSelectionInitialization(null, "fixture", undefined),
    null,
  );

  const firstPage = benchmarkSelectionInitialization(null, "fixture", [
    { id: "one" },
    { id: "two" },
  ]);
  assert.deepEqual(firstPage, {
    benchmarkId: "fixture",
    selectedIds: ["one", "two"],
    focusedItemId: "one",
  });

  assert.equal(
    benchmarkSelectionInitialization("fixture", "fixture", [
      { id: "one" },
      { id: "two" },
    ]),
    null,
    "an intentional empty selection must not be treated as uninitialized",
  );

  assert.deepEqual(
    benchmarkSelectionInitialization("fixture", "next-benchmark", [
      { id: "three" },
    ]),
    {
      benchmarkId: "next-benchmark",
      selectedIds: ["three"],
      focusedItemId: "three",
    },
  );
});
