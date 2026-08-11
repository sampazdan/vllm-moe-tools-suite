import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  expertsForLayer,
  materializeProfileForEditor,
  profileMeetsTopKFloor,
} from "./profileEditor.ts";
import type { ExpertProfile } from "./types.ts";

describe("partial profile editing", () => {
  it("materializes omitted routed layers as all experts eligible", () => {
    const partial: ExpertProfile = {
      version: 1,
      layers: { "1": { keep: [1, 3] } },
    };

    const draft = materializeProfileForEditor(partial, [1, 2, 3], 4);

    assert.deepEqual(draft.layers["1"].keep, [1, 3]);
    assert.deepEqual(draft.layers["2"].keep, [0, 1, 2, 3]);
    assert.deepEqual(draft.layers["3"].keep, [0, 1, 2, 3]);
    assert.deepEqual(expertsForLayer(partial, 2, 4), [0, 1, 2, 3]);
    assert.equal(profileMeetsTopKFloor(partial, [1, 2, 3], 4, 2), true);
    assert.deepEqual(partial.layers, { "1": { keep: [1, 3] } });
  });

  it("preserves an explicit empty layer as invalid", () => {
    const partial: ExpertProfile = {
      version: 1,
      layers: {
        "1": { keep: [1, 3] },
        "2": { keep: [] },
      },
    };
    const draft = materializeProfileForEditor(partial, [1, 2, 3], 4);

    assert.deepEqual(expertsForLayer(partial, 2, 4), []);
    assert.deepEqual(draft.layers["2"].keep, []);
    assert.deepEqual(draft.layers["3"].keep, [0, 1, 2, 3]);
    assert.equal(profileMeetsTopKFloor(draft, [1, 2, 3], 4, 2), false);
  });
});
