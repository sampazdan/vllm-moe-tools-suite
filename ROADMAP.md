# MoE Tools Test Suite Roadmap

Status: `0.3.0-rc.1` finish-line candidate, updated 2026-08-11

The product is now a complete local research loop rather than a collection of MVP
cards. The remaining release work is integration verification, immutable image
publication, and real Runpod/Daytona acceptance. This roadmap distinguishes those
states deliberately:

- **Implemented locally** means the code, UI, persistence contract, and automated
  coverage are present on this branch.
- **Live-accepted** means a specific application SHA, fork SHA, image digest, and
  external-resource run were recorded together.
- **Fail-closed** means metadata or a future integration may be visible, but the
  application refuses to launch it because admission evidence is incomplete.

The exact `0.3.0-rc.1` build is not yet live-accepted. An older internal RC3 build
has historical evidence, recorded below, but it does not certify the new candidate.

## North star

The application helps a researcher answer one practical question: _can a large MoE
model be specialized for a target workflow by retaining only the experts that
matter, without losing the behavior that matters?_

The first causal experiment is an expert-eligibility mask:

1. load a pinned model and record its topology/runtime;
2. run a pinned benchmark or agentic workflow;
3. capture routed expert IDs and probabilities for every instrumented inference;
4. inspect aggregate and per-layer engagement;
5. create an immutable custom profile from one or more workload traces;
6. reload vLLM with that profile;
7. rerun the identical evaluation and execution contracts; and
8. compare quality, tool behavior, routing, performance, and cost.

The persistence model treats the mask as one intervention. Later interventions can
add selective quantization, offload, or physically materialized checkpoints without
changing the benchmark and comparison records.

## Finish-line capability map

No percentage here stands in for evidence. Every row names what exists locally and
what must still happen before release acceptance.

| Area | Implemented locally in `0.3.0-rc.1` | Remaining release gate |
| --- | --- | --- |
| Runpod appliance | Authenticated same-origin FastAPI/React appliance; persistent data root; immutable-image/template tooling; managed vLLM on loopback; phone/desktop responsive shell. | Publish the final image, pin its digest, and repeat proxy/login/storage checks. |
| Model lifecycle | Registry-based Qwen A3B load/stop/reload; topology and runtime facts; startup log; chat; persisted model sessions; job adoption; process-group cleanup and startup reconciliation. | Real-load, hard-reload, stop, and restart acceptance on the final image. |
| Benchmark command center | Full-screen catalog/problem/run views; editable criteria and budgets; item/attempt progress; performance; routing; paginated archive/items; streamed JSON/CSV export. | Run bounded samples from every bundled adapter on the real model. |
| Benchmark breadth | Fixture, GSM8K, MMLU-Pro validation 70/test 12,032, IFEval strict 387/full reference 541, LiveBench Zebra 50, and custom JSONL. | Verify lazy downloads, pins, scorers, cancellation, and representative real outputs. |
| Profile Studio | Regular and agentic sources; explicit source weights; aggregate/per-layer References/Selection/Delta; mass/count metrics; fixed/global/cumulative/manual selection; immutable revisions. | Create, reload, and ablate a non-uniform multi-source profile on the real fork. |
| Comparisons | Strict paired benchmark and agent-run resources/views; outcome transitions; score/reward and performance deltas; profile/provenance compatibility. | Save and reopen one baseline/masked comparison of each kind on the final image. |
| Agentic foundation | Native bounded controller; Daytona/fake providers; reasoning/tool/command/stdout/stderr/verifier timeline; ATIF; per-inference routing and performance; durable trials/artifacts. | Run complete baseline and masked cohorts on Daytona with zero retained sandboxes. |
| Agent profiles | Profile Studio accepts any selected agentic trials, including all trials in a run; sources can mix agentic and regular workloads. | Derive a custom profile from a real agent run and use it in a paired rerun. |
| Evaluation | Fingerprinted deterministic criteria, trusted and user-authored verifier provenance, weighted aggregation, Anthropic/OpenAI/fake judges, cache/cost accounting. | Make one tightly capped real judge call and exercise a controlled failure/cache path. |
| Isolation | Digest-pinned task images; public/submission/verifier/oracle separation; fresh verifier sandboxes; root-only hidden assets; unprivileged candidate subprocesses; exact-label cleanup. | Run clean-room exploit regressions and observe ownership/deletion on Daytona. |
| Recovery | Durable jobs, typed conflicts, reconnect adoption, cooperative cancellation, restart reconciliation, periodic Daytona cleanup, bounded external errors. | Live cancellation, proxy disconnect, app restart, and orphan-recovery injection. |
| External coding suites | Launchable Aider Python canary 3/225; pinned Terminal-Bench engineering 15, Aider balanced 12, and FeatureBench Fast manifests. | The three larger manifests remain fail-closed pending full admission. |

## Current application shape

The frontend uses a small route-based command shell rather than nesting every
workflow in one dashboard:

```text
/
/benchmarks
/benchmarks/{benchmark_id}
/runs
/runs/{run_id}
/agent-runs
/agent-runs/{run_id}
/agent-comparisons/{baseline_id}/{candidate_id}
/profiles
/profiles/new
/profiles/{profile_id}
/comparisons
/comparisons/{comparison_id}
```

Home owns model lifecycle and compact chat. Benchmarks and agents each have a
dedicated builder, live progress surface, and immutable run view. Profile Studio is
shared across one-request and multi-turn traces. Comparison pages make provenance
mismatches explicit. The responsive layout uses full matrices and horizontal
controls on desktop, then switches to touch-friendly tabs, drawers, and one-column
problem/trajectory views on a phone.

The visual direction remains warm off-white paper, ink-like text, restrained earth
accents, serif narrative typography, and clean sans/monospace data surfaces. Color
is not the only status or selection signal.

## What the custom vLLM fork provides

The fork exposes version 1 expert-selection profiles, routed expert IDs, and the
paired float32 weights through the OpenAI-compatible API. The launcher flags are:

```text
--enable-return-routed-experts
--enable-return-routed-expert-weights
--moe-expert-selection-profile /path/to/profile.json
```

IDs and weights arrive as base64-encoded NumPy arrays with shape
`[tokens, routed_layers, top_k]`. The application validates, aggregates, and stores
those arrays outside the vLLM scheduler.

The fork also accepts `routed_experts_prompt_start` on an instrumented chat request.
For an agentic trial, the application sends that offset only when locally returned
token IDs prove that a new prompt extends the previous prompt exactly. The returned
routing then begins at the novel prompt suffix and continues through the new
generation. Missing tokens, clipping, unrelated prompts, or ambiguous alignment use
offset zero. This yields de-duplicated served-token engagement, not semantic or
causal attribution.

A profile is intentionally simple:

```json
{
  "version": 1,
  "layers": {
    "0": { "keep": [0, 1, 2, 3, 4, 5, 6, 7] }
  }
}
```

The fork validates layer/expert IDs, duplicates, the `top_k` floor, router
compatibility, and supported execution paths. It applies eligibility before top-k
selection and fails closed on unsupported routing implementations.

### Masking limitation

Profiles are bound during model load. Changing one restarts the model process.
Ineligible expert weights remain resident, so this is behavioral masking—not
checkpoint compaction, offload, or evidence of lower VRAM. Routed capture also has
memory, transport, and latency cost. Quality analysis should compare runs with the
same capture mode; throughput claims need a separate capture-disabled run.

## Architecture decisions

### Persistent Runpod Pod

A Pod fits an interactive, stateful, warm-model appliance better than Serverless.
The target remains one RTX PRO 6000 Blackwell Server Edition with 96 GB-class VRAM.
The application serves `8080/http`; managed vLLM stays on `127.0.0.1:8000`.
Persistent state and model caches live below `/workspace`, ideally on a network
volume when they must outlive the Pod.

External access uses Runpod's built-in HTTPS proxy:

```text
https://<pod-id>-8080.proxy.runpod.net
```

The app supplies authentication because the proxy supplies transport, not identity.
Long operations return a durable job immediately and use polling, so core workflows
do not depend on a long-lived proxy connection. This avoids an ngrok account or
sidecar without coupling the app to Runpod; a different HTTPS reverse proxy can
forward the same single port later.

### One application, one managed vLLM child

FastAPI owns the SQLite metadata store, filesystem artifacts, durable job runner,
frontend, model gateway, and controller. Only one GPU operation is active at a time.
The managed vLLM process runs in its own validated process group. Model/profile or
telemetry-mode changes create a new model session and restart that group.

SQLite uses one application writer and keeps large arrays/artifacts as files. Run
summaries are bounded; item pages are fetched separately. JSON and CSV exports are
generated row-by-row.

### Reproducible build

The `linux/amd64` Docker build requires a full vLLM fork commit. Its Node 24.6.0
builder and Runpod CUDA 13/PyTorch 2.9 base are pinned by OCI digest; Python 3.12
dependencies are locked by `uv.lock`; the build verifies the CUDA PyTorch shared
library. The publish workflow uses pinned GitHub Actions, creates an OCI provenance
attestation, and emits version and full-application-commit tags. Deployment should
use the published digest.

Task packs also require effective image digests. The current runnable Python packs
pin the official `python:3.12-slim` digest. The final application SHA, fork SHA, and
release image digest will be inserted only after they are real.

## Benchmark and evaluation design

### Adapter contract

Each benchmark adapter owns:

- dataset identity, source revision, split, license, artifact digest, and size;
- stable item IDs, categories, searchable problem fields, and exact selection;
- prompt-template version and generation defaults;
- deterministic scorer/evaluator version; and
- content hash and immutable cohort fingerprint.

The application currently runs one request per selected item/attempt. This keeps
progress, cancellation, scoring, routing, retry, and provenance observable.
Concurrency remains an explicit execution-policy setting rather than hidden runner
behavior.

### Runnable one-request catalog

| Benchmark | Cohort | Contract note |
| --- | ---: | --- |
| Arithmetic fixture | 12 | Bundled exact-match application smoke. |
| [GSM8K](https://github.com/openai/grade-school-math) | 1,319 | Pinned official main test, exact final number. |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) validation | 70 | Complete validation calibration set, not the leaderboard test aggregate. |
| MMLU-Pro test | 12,032 | Complete pinned test split, final answer-letter accuracy. |
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) strict | 387 | Application subset of 20 dependency-free strict evaluator families; not an official aggregate. |
| IFEval full reference | 541 | Full corpus, intentionally ungraded reference/routing run. |
| [LiveBench](https://github.com/LiveBench/LiveBench) Zebra 2024-06 | 50 | Archived release cohort/scorer, not current leaderboard coverage. |
| Custom JSONL | Variable | Content-addressed user tasks and editable criteria. |

The benchmark command center exposes problem text and public success criteria before
launch. Protected criteria are represented only by count/fingerprint. Dataset
generation defaults are preserved unless the user edits them; the execution
policy's total-token cap is not silently reused as the per-response generation cap.

### EvaluationContract

Success is an immutable, fingerprinted contract. Criteria support dataset default,
exact, contains, bounded regex, numeric tolerance, multiple choice, JSON schema,
ungraded, trusted task verifier, and user-authored sandbox verifier modes as
appropriate. Each criterion records label, description, public/protected visibility,
required flag, weight, configuration, result, duration, output hash, error, and
execution provenance. Aggregation is either all-required or weighted threshold.

A structured LLM judge can augment that deterministic result. Anthropic, OpenAI,
and deterministic fake providers share one schema for single, reference, and
explicit pairwise judging. Rubric, presentation randomization, repetitions, prices,
model, threshold, and protocol are fingerprinted. Cached results record equivalent
historical cost but zero newly incurred spend.

### ExecutionPolicy

Budgets stop work; they do not decide success. The fingerprinted policy covers:

- attempts and concurrency;
- whole-run and per-item wall time;
- agent turns and commands;
- whole-run prompt plus generated tokens;
- fail-fast behavior; and
- incremental frontier-judge spend.

Per-response output tokens, temperature, seed, and thinking/reasoning presentation
belong to generation configuration. Before inference, the runner reserves a
conservative prompt-token bound against the whole-run token cap. Before an uncached
paid judge call, it reserves the conservative maximum for every repetition; capped
judging requires explicit prices. Possible provider spend after a partial failure is
recorded as uncertain and debited conservatively.

Legacy records do not acquire modern defaults retroactively. Their archive/detail
views show unknown contract/provenance fields explicitly.

## Expert analytics and profile design

The explorer consumes benchmark runs and agentic trials through one strict source
contract. It records selection counts, routing mass, captured/total inference calls,
served tokens, source/model/profile lineage, compatible filters, and a server-derived
fingerprint.

### `weighted_source_normalized`

For multiple sources, each source matrix is normalized to unit total before its
explicit relative weight is applied. Equal weights therefore give equal influence
to two workloads even if one emitted far more tokens. The operation is identified as
`weighted_source_normalized:v1` in the fingerprint. Persisted observed-mass metrics
are calculated against that same combined matrix.

The explorer defaults to the complete layer × expert heatmap. Layer selection never
merges expert 7 in one routed layer with expert 7 in another. Its modes are:

- **References:** probability mass or reference/share counts;
- **Selection:** eligibility, selected count, and observed mass retained; and
- **Delta:** signed change against compatible comparison sources.

### Selection paths

Implemented strategies are:

1. fixed top-K per layer, globally or for one selected layer;
2. global expert budget with a `top_k` minimum in every layer;
3. cumulative global routing-mass target with the same floor; and
4. manual expert toggles and layer operations.

Users can combine ordinary runs and agentic trials, give every source a weight, and
edit the assisted result. A full agent workload becomes a source by selecting all of
its trials; a researcher can also use only successful/failed/selected trials or mix
them with regular runs. Every save is immutable and records its source refs, source
fingerprint, metric, strategy/configuration, parent revision, topology validation,
observed mass retained, and exact fork-compatible profile JSON.

The server recomputes the routing-source fingerprint rather than trusting the
browser. Profiles that omit a routed layer materialize it as fully eligible; an
explicit empty layer is invalid.

## Comparison design

A strict benchmark comparison requires compatible model, dataset/cohort item order,
attempt identity, prompts/scorers, generation, evaluator, and execution policy. It
reports aggregate score delta, per-item pass/fail transitions, output/evaluation
details, profile eligibility, routing views, and performance/cost deltas.

A strict agent comparison additionally requires the same task-pack revision/content
hash, task attempt order, agent/controller revision, provider/policy, generation,
budgets, and seed. It reports mean reward and pass-rate delta, task transitions,
turns, commands, tokens, terminations, timing, judge cost, and routing. A masked run
is never called comparable merely because it used the same task names.

## Agentic coding foundation

### Native controller and telemetry

`bash-json-v1` revision 1 is a small model-centered loop. One model response yields
one JSON shell action or final response. Every turn flows through the local
instrumented gateway and stores model/reasoning text, the parsed command, stdout or
stderr observation, performance, and routing artifact. The command center renders
reasoning, response, tool, command, observation, verifier, and termination as
visually distinct events.

Trajectories export as `ATIF-v1.7`, following Harbor's
[trajectory RFC](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md).
The task/provider/domain boundaries map cleanly to Harbor, but current runs are
owned by this native controller rather than a hidden Harbor or Aider process.

Performance reports include true runtime fields when supplied: prompt, reasoning,
and completion tokens; request/model latency; time to first token; prefill, decode,
queue, and inter-token time; tokens per second; total run wall time; commands/turns;
and judge/API cost. Unreported fields stay null.

### Runnable packs

- `smoke-python-v1` revision 2 has three tiny Python tasks for plumbing and failure
  checks.
- `repo-engineering-v1` revision 2 has three moderate first-party tasks for JSONL
  data-pipeline repair, dependency-graph release planning, and asynchronous worker
  concurrency.
- `aider-polyglot-python-canary-3` has three source-attested Python tasks selected
  from the 225-task [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark)
  conversion: zipper, wordy, and zebra puzzle. It is a 3/225 application canary,
  not an Aider leaderboard cohort or score.

Every launchable pack has a content hash, exact image digest, declared public,
submission, verifier, and oracle file sets, an oracle-pass record, and a no-op-fail
record.

### External pack admission

The application also records immutable manifests for:

- Terminal-Bench 2.1 engineering 15;
- Aider Polyglot balanced 12 across six languages; and
- FeatureBench Fast 100 reference.

They remain fail-closed. Terminal-Bench still needs runner/image compatibility and
per-task oracle/no-op validation. Aider balanced 12 still needs imported assets,
license review, and per-task admission. FeatureBench still needs the official CLI,
images, and result schema; its recorded Harbor source did not provide an acceptable
immutable converted-task revision. None may be presented as a runnable or official
score merely because its source manifest exists.

### Sandbox and clean-room boundary

The fake provider is an in-memory contract double and cannot execute a local
subprocess. Daytona is the real remote provider through the pinned 0.192.0 SDK. A
read-only preflight proves reachability/authentication without creating a sandbox.

For every real trial, the controller creates a private, labelled, digest-pinned
sandbox with no controller secrets and the task's network/resource policy. It
captures only declared submission paths. Trusted evaluation then creates a fresh,
network-disabled verifier sandbox from canonical starter files, the captured
submission, and root-only hidden verifier files. Oracle solutions do not enter that
runtime. Candidate imports/probes execute as an unprivileged uid/gid, so model code
cannot read or mutate protected tests. Each UI-authored verifier command runs in a
separate fresh sandbox and is labelled `user_authored_sandbox`, not trusted.

Cleanup validates controller/run/trial ownership labels before every destructive
operation, polls to confirmed deletion, retries after partial create failures,
reconciles at startup, and periodically reaps only owned stale sessions. A retryable
cleanup error stays durable and blocks conflicting Daytona work rather than being
reported as success.

## Recovery and operational behavior

- Model session and job are persisted atomically before a load response.
- A dropped HTTP waiter cannot cancel server-owned work.
- Identical model-load submissions adopt the same job; incompatible work receives a
  typed conflict and recovery link.
- Browsers back off through transient network/429/5xx errors, wake on reconnect or
  tab visibility, and rediscover the active job after reload.
- Running cancellation transitions to `cancelling`; only queued work may become
  cancelled immediately. The active-work lock remains held until worker exit.
- Managed vLLM has an owned process group, validated identity, bounded TERM/KILL
  shutdown, and startup reconciliation that does not trust a PID file alone.
- Interrupted benchmark and agent records retain explicit provenance and terminal
  causes. Archived task titles/content hashes do not depend on the current catalog.
- Daytona cleanup handles ordinary deletion, handle-less create failure, app
  restart, and periodic exact-label reconciliation.

## Security guardrails

- Require a long `MOE_TOOLS_AUTH_TOKEN` on Runpod and use secure same-site cookies.
- Keep port 8000 private; expose only 8080 and optional maintenance SSH.
- Strip application auth, Runpod, Hugging Face, Daytona, Anthropic, and OpenAI
  credentials before launching managed vLLM or task/verifier sandboxes.
- Treat prompts, model output, imports, judge rationales, and artifact text as
  untrusted browser content.
- Bound upload size, regex execution, model output, retained artifacts, concurrency,
  wall time, turns, commands, tokens, and paid judging.
- Never run model-generated code or a benchmark verifier in the application/model
  container.
- Require digest-pinned task images and immutable application images for accepted
  experiments.
- Preserve exact ownership checks for every sandbox delete; never broad-delete
  resources by provider or account.
- Show that network volumes continue billing after Pods stop and require explicit
  teardown confirmation.

## Release gates

### Gate A — integrated local validation

- locked Python and Node installs;
- backend lint and complete pytest suite;
- frontend contract tests, type check, and production build;
- JSON, shell, template dry-run, and lock consistency checks;
- task-pack oracle/no-op and clean-room exploit tests;
- job cancellation/recovery and judge budget/cache/failure tests; and
- repository secret scan and review of the complete diff.

Status: implementation present; final integrated result pending while this candidate
is assembled.

### Gate B — immutable container

- commit and push the exact vLLM fork;
- pass that full 40-character SHA to the publish workflow;
- build the exact application commit for `linux/amd64`;
- record OCI digest and provenance attestation;
- update the private Runpod template to the digest; and
- verify the image startup hook retains Runpod `/start.sh` services while the app
  reaches `/readyz` within its bounded startup timeout.

Status: pending. No final fork SHA or application image digest is invented in this
roadmap.

### Gate C — model and benchmark loop

1. Authenticate through the public Runpod proxy on desktop and phone.
2. Load `Qwen/Qwen3.6-35B-A3B-FP8`; hard-reload mid-load and require adoption of the
   same job/process.
3. Inspect topology, chat, reasoning presentation, and request metrics.
4. Run bounded GSM8K, MMLU-Pro, IFEval, and LiveBench samples.
5. Open problem/success criteria, live progress, item attempts, routing, and
   performance from their command-center pages.
6. Combine at least two sources into a non-uniform custom profile, reload it, rerun
   the identical cohort, save the strict comparison, and reopen it after a hard
   refresh.

Status: pending for the final image.

### Gate D — agentic/profile loop

1. Run read-only Daytona preflight.
2. Run one complete unmasked smoke or repository-engineering cohort.
3. Require distinct reasoning/response/command/stdout/stderr/verifier events, ATIF,
   per-inference routing, performance, captured submissions, trusted outcomes, and
   confirmed task/verifier sandbox deletion.
4. Create a custom profile from all or selected trial traces; inspect aggregate and
   individual layers, then make a deliberate manual edit.
5. Reload the profile and rerun the identical agent contract.
6. Save and inspect the strict agent comparison.
7. Exercise cancellation and one interruption/orphan recovery path.

Status: pending for the final image.

### Gate E — frontier judge and cost

- run one small Anthropic judge under an explicit low spend cap and configured token
  prices;
- verify structured result, fingerprint, rationale, usage, latency, incurred cost,
  and budget debit;
- repeat to prove a cache hit incurs zero new cost; and
- exercise a preflight rejection or controlled failure without exceeding the cap.

Status: pending for the final image.

### Gate F — teardown

- list and confirm zero application-owned Daytona task/verifier sandboxes;
- stop every Pod created or resumed for acceptance;
- confirm no paid Runpod compute remains active;
- record actual scoped spend; and
- retain only network volumes/storage chosen intentionally.

Status: pending after Gates C–E. Existing unrelated resources are not deletion
targets.

## RC1 acceptance record — intentionally pending

Fill this section only from observed final-image evidence:

- application commit: pending;
- vLLM fork commit: pending;
- GHCR image digest/attestation: pending;
- Runpod template and Pod/GPU: pending;
- local gate results: pending;
- model load/topology/runtime: pending;
- benchmark/profile/comparison result: pending;
- agent/profile/comparison result: pending;
- judge model/request/cost result: pending;
- cancellation/recovery result: pending; and
- final Runpod and Daytona teardown: pending.

## Historical RC3 evidence — not RC1 acceptance

On 2026-08-11, an older internal candidate used application commit
`6d999f6288592f9414df1745a6371916cacc9fba`, fork commit
`729d4f23ae9cc27f797b8c13a9276565976637ed`, and image digest
`sha256:bb5e4addc10db7d2f8166d8eef05be3c032ee83533495c7d4f8d3b2dc8026362`.
It ran on one RTX PRO 6000 Blackwell through the authenticated Runpod proxy.

That candidate loaded `Qwen/Qwen3.6-35B-A3B-FP8`; a browser hard reload adopted the
same load job and PID. A two-item arithmetic baseline → 64-experts-per-layer profile
→ masked rerun → saved comparison completed with both scores at `1.0`.

The unmasked Daytona `fix-subtract` trial passed with reward `1.0`, eight
instrumented inference/routing turns, seven commands, a trusted verifier pass, and
confirmed sandbox deletion. Under the arithmetic-derived 25% profile, the same task
returned reward `0`, emitted no command, reached `token_limit`, failed verification,
and still deleted its sandbox. That was negative transfer from that profile, not a
provider lifecycle failure or a cohort-level coding conclusion.

All Pods used for that historical acceptance were stopped and no acceptance-owned
Daytona sandbox remained. The old evidence does not cover RC1's standard benchmark
breadth, editable contracts, paid judges, multi-source/custom agent profiles,
clean-room verifier design, full-screen views, or strict paired agent comparison.

## Beyond `0.3.0-rc.1`

These are real future stages, not hidden release blockers:

### Admit broader public agentic suites

- import and admission-test Aider Polyglot balanced 12 across six languages;
- integrate the pinned Terminal-Bench 2.1 engineering 15 runner/images;
- integrate FeatureBench Fast's official CLI, images, and partial metrics;
- require oracle pass, no-op fail, digest pin, license/provenance, public/protected
  asset separation, and clean-room compatibility for every task;
- add repeated attempts, `pass@k`, confidence intervals, and held-out cohort design;
  and
- add optional pinned external agents (Aider, Mini-SWE-Agent, Terminus-2) as separate
  scaffold dimensions that are never pooled with the native controller.

### Improve research selection

- target-versus-reference routing lift;
- successful-versus-failed/recovery-step weighting;
- held-out profile-validation workflows;
- brush selection and richer undo/redo history; and
- automated bounded search over a small candidate family with explicit validation
  cost, never an opaque claim of causal expert importance.

### Materialize smaller artifacts

Once masks remain stable across held-out workflows, add a separate pipeline for:

- checkpoint compaction;
- expert weight offload or lazy loading;
- per-expert/selective quantization; and
- combined mask/quantization variants.

Those experiments must measure actual GPU memory, host memory, disk size, load time,
latency, throughput, and quality. Theoretical eligible-expert fraction is not a
memory result.

### Broader appliance work

- optional self-hosted sandbox provider for private repositories;
- backup/import and retention controls for raw routing artifacts;
- schema migration tooling for long-lived volumes;
- capture-disabled serving performance mode integrated with vLLM benchmarks;
- additional model registry entries after topology/backend acceptance; and
- optional multi-user identity only if the appliance stops being single-researcher.

## Definition of release complete

`0.3.0-rc.1` is accepted when a new user can launch the immutable Pod template,
authenticate from a phone or laptop, load the supported Qwen model, inspect topology
and reasoning/performance telemetry, run a selected benchmark, create a custom
profile from regular and/or agentic traces, reload the masked model, rerun identical
contracts, and inspect/export strict paired benchmark and agent comparisons without
opening a shell after launch.

Completion also requires that trusted verifier assets never enter the model
sandbox, user-authored verifiers are labelled accurately, paid judge work remains
inside its cap, cancellation/recovery remain observable, and all acceptance-owned
paid compute and sandboxes are stopped or deleted at handoff.

## Open product-policy questions

No question blocks the current release gate. Before broader usage, decide:

1. whether custom tasks may contain private repositories and whether Daytona is an
   approved data boundary for them;
2. whether raw per-inference routing arrays are retained by default or only on
   explicit opt-in;
3. how long failed sandbox metadata and large submission artifacts should remain;
4. when a self-hosted sandbox provider becomes necessary; and
5. which held-out task packs and repetition counts are required before calling a
   profile “coding-specialized.”

## Research references

- [Runpod Pod templates](https://docs.runpod.io/pods/templates/overview)
- [Runpod HTTP proxy and exposed ports](https://docs.runpod.io/pods/configuration/expose-ports)
- [Runpod storage types](https://docs.runpod.io/pods/storage/types)
- [Runpod network volumes](https://docs.runpod.io/storage/network-volumes)
- [Qwen3.6-35B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)
- [GSM8K](https://github.com/openai/grade-school-math)
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- [Google IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [LiveBench](https://github.com/LiveBench/LiveBench)
- [Daytona API keys](https://www.daytona.io/docs/api-keys/)
- [Daytona async Python client](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/)
- [Harbor](https://github.com/harbor-framework/harbor)
- [Harbor tasks](https://www.harborframework.com/docs/tasks)
- [Harbor agent integrations](https://www.harborframework.com/docs/agents)
- [Harbor ATIF format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
- [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark)
- [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
- [FeatureBench](https://github.com/LiberCoders/FeatureBench)
