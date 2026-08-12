# MoE Atelier V2 local acceptance evidence

Captured on 2026-08-12 against the local mock runtime at desktop widths of
1440 px and 1280 px and a phone width of 390 px.

This evidence validates application behavior, durable persistence, API/UI
contracts, responsive layout, and the mock hot-context control seam. It is not
real-GPU evidence and must not be used to claim live vLLM latency, routing
telemetry, PID stability, rollback safety, or model qualification.

## Durable experiments

- Paired answer: `f14fe633-be5a-4f72-9400-37c27a69a45b` — 2/2 lane units
  passed, with the prompt, output, and deterministic scorer rendered in the
  common command center.
- Paired coding: `d92dd162-fa49-44d7-9bac-fd20816d737b` — 2/2 lane units
  passed, with commands, output, verifier result, and lane-scoped trajectory
  rendered in the common command center.
- Paired DriftBench: `8bdf6dd7-c751-4e45-b54b-bff712afc078` — 72 lane
  units, 36 aligned observations, and 62 passes. The selected comparison first
  diverges at checkpoint 2 and exposes the exact changed state path.
- Live-progress acceptance: `010cde09-83a0-40ef-9abf-6ff814768c03` — a
  5,040-unit matrix. `live-progress-1440.png` was captured while 417/5,040
  units were durable, then the page was hard-refreshed and adopted the same run
  at 667/5,040 with checkpoint progress intact. The oversized mock run was
  cancelled after this acceptance check.

## Screenshots

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
