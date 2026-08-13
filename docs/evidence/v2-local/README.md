# MoE Atelier V2 local acceptance evidence

Captured on 2026-08-12 against the local mock runtime at desktop widths of
1440 px and 1280 px and a phone width of 390 px.

This evidence validates application behavior, durable persistence, API/UI
contracts, responsive layout, and the mock hot-context control seam. It is not
real-GPU evidence and must not be used to claim live vLLM latency, routing
telemetry, PID stability, rollback safety, or model qualification.

It is also not publication evidence. No V2 branch push, pull request, OCI digest,
attestation, Runpod template, Pod, Daytona sandbox, additional-model attempt,
provider spend, or V2 teardown is represented here. Historical V1 coordinates in
the root documentation remain separate.

## Local implementation state

The current local candidate extends the captured screens with these tested
contracts:

- Model loads preserve an ordered durable phase history. Application phases are
  observed directly; cache/download/weights/workers/compile/graphs/warm phases are
  observed only when the managed runtime emits complete authenticated loopback
  signals. Otherwise they are explicitly `unavailable`, never zero-duration or
  fabricated progress. Failures retain redacted diagnostics, a failure code, and a
  recovery action; unsafe cleanup blocks retry.
- Answer and coding descriptors expose stable unit IDs. Experiment creation uses
  an explicit ordered subset, defaults large workloads to a bounded selection, and
  caps one request at 10,000 units. Seeds and baseline/candidate lanes expand that
  real cohort; Drift separately expands scenarios, conditions, horizons, seeds,
  and lanes.
- Cohort identity binds workload content, exact unit/scenario selection,
  generation, evaluation contract, execution policy, and Drift parameters. V2
  supports workload-owned deterministic evaluation or record-only collection.
  Time, per-unit time, tokens, turns, commands, and priced-judge cost are execution
  limits, not success criteria; exhausted work remains unscored. Legacy missing
  contracts stay unknown.
- Drift scenarios persist dependency span, branch count, rollback depth,
  distractor ratio, deterministic tool-error rate, and state size. The UI exposes
  exact transition/fidelity/survival/recovery/rollback/reliability metrics,
  local-versus-chained compounding, paired deltas, and checkpoint routing. Absolute
  selection slots and routing mass remain alongside normalized overlap and
  divergence when real telemetry exists.

The original screenshots below predate the last contract-hardening pass. The five
files marked **current pass** were recaptured afterward and cover structured
model-load presentation, a selected ordinary cohort, inline rename provenance,
the expanded Drift metric panel, and the 390 px command center. They are still
local mock evidence, not substitutes for immutable-image GPU screenshots.

## Durable experiments

- Paired answer: `f14fe633-be5a-4f72-9400-37c27a69a45b` — 2/2 lane units
  passed, with the prompt, output, and deterministic scorer rendered in the
  common command center.
- Paired coding: `d92dd162-fa49-44d7-9bac-fd20816d737b` — 2/2 lane units
  passed, with commands, output, verifier result, and lane-scoped trajectory
  rendered in the common command center. This used `FakeSandboxProvider`, which
  substitutes pinned oracle actions after model calls. It validates wiring only;
  it is not model-authored coding or a release-qualifying coding result.
- Paired DriftBench: `8bdf6dd7-c751-4e45-b54b-bff712afc078` — 72 lane
  units, 36 aligned observations, and 62 passes. The selected comparison first
  diverges at checkpoint 2 and exposes the exact changed state path.
- Live-progress acceptance: `010cde09-83a0-40ef-9abf-6ff814768c03` — a
  5,040-unit matrix. `live-progress-1440.png` was captured while 417/5,040
  units were durable, then the page was hard-refreshed and adopted the same run
  at 667/5,040 with checkpoint progress intact. The oversized mock run was
  cancelled after this acceptance check.

## Screenshots

- `model-lifecycle-1440.png` and `model-lifecycle-phases-1440.png` — **current
  pass** model selection, hot-context separation, and durable phase history. The
  mock runtime honestly marks child-only signals unavailable.
- `paired-answer-cohort-1440.png` — **current pass** two-item paired answer cohort
  with deterministic scoring, immutable context fingerprints, and an inline
  profile rename.
- `paired-answer-mobile-390.png` — **current pass** phone command center after a
  hard refresh, with zero horizontal overflow.
- `state-drift-parameters-metrics-1280.png` — **current pass** bounded paired
  Drift run with parameter provenance and the expanded decomposition panel.
- `experiment-list-1440.png` — unified experiment archive.
- `live-progress-1440.png` — genuinely in-flight durable progress.
- `paired-answer-1440.png` — answer lane comparison.
- `paired-coding-1440.png` — coding trajectory and verifier evidence.
- `state-drift-1440.png` — DriftBench command center.
- `state-drift-divergence-1440.png` and
  `state-drift-divergence-checkpoint-1440.png` — aligned divergence and exact
  state comparison.
- `state-drift-routing-1440.png` — routing inspector shape with honest unknown
  values because the mock runtime emits no real expert telemetry.
- `profile-library-1440.png` and `profile-renamed-1440.png` — named profile and
  inline rename. The saved mask fingerprint was unchanged.
- `workload-library-1280.png` — task-level readiness and blocker metadata.
- `state-drift-mobile-390.png` and
  `state-drift-mobile-inspect-390.png` — phone navigation and inspector layout.

Browser console error logs were empty after desktop and phone acceptance. The
production frontend build reports only the known large-chunk warning.

## Local hot-context observation

The mock runtime switched baseline to the named half-expert profile and back
without a weight reload. Mock receipts report zero duration and no process ID,
so they validate only the application contract; real switch P50/P95 and
unchanged-process proof remain Runpod release gates.

## V2 live evidence driver

The repository contains a machine-readable driver, but it has not been run here
against a V2 GPU image:

```bash
MOE_TOOLS_CANARY_BASE_URL=https://POD_ID-8080.proxy.runpod.net \
MOE_TOOLS_CANARY_TOKEN="..." \
uv run python scripts/v2_live_acceptance.py \
  --coding-provider daytona \
  --output artifacts/v2-live-acceptance.json
```

The default fake coding mode and `--allow-missing-routing` are diagnostic-only and
leave acceptance incomplete. The generated JSON distinguishes in-appliance proofs
from external checks; no manually authored substitute is committed in this local
evidence directory.

## Outstanding V2 gates

- Push reviewed application and fork commits, open reviewable PRs, publish an
  attested `linux/amd64` image, and create a digest-pinned V2 template.
- Capture real Qwen load phases, profile switch distribution, unchanged PID and
  memory, no weight reread, rejected and injected commit-failure rollback, real
  routing, cold/hot equivalence, tensor-parallel/cache, eager, and CUDA evidence.
- Run a genuine multi-item answer cohort, model-authored Daytona coding, and a
  meaningful paired Drift experiment with routing around first divergence.
- Attempt an additional MoE model and record qualification or its precise blocker.
- Perform public desktop/phone refresh QA against the exact image.
- Record Runpod and Daytona inventories, spend, teardown, and zero remaining paid
  compute/sandboxes. No current V2 cleanup claim is made because no V2 external
  resource was created during this local pass.
