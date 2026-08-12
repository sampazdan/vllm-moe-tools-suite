# MoE Tools Test Suite Roadmap

Status: `0.4.0-rc.1` V2 candidate, updated 2026-08-12

## V2 candidate

V2 changes the primary experiment primitive from “reload the model with another
profile” to “alternate immutable intervention contexts on one warm engine.” The
fork now has a bounded, loopback-only expert-context controller with canonical
topology/profile/context fingerprints, stable in-place router buffers, request
admission and cache drain, tensor-parallel prepare/commit/rollback, and output
provenance. Unsupported router, data-parallel, pipeline-parallel, and multi-API
server configurations fail closed.

The application adds migration-safe experiments, lanes, workload runs, run units,
evaluations, performance snapshots, intervention refs, and append-only sequenced
events. Answer, coding, and deterministic state-drift adapters use the same command
center and paired-lane model. Polling and reconnectable SSE expose durable progress;
refresh never turns an active experiment back into an anonymous job.

State Drift Bench measures exact checkpoint state rather than only final success.
Seven first-party scenario families cover ledger and order lifecycles, dependency
release planning, repository evolution, structured-data transformation, ticket/CRM
workflow, and branch/counterfactual rollback under oracle-reset, chained, and
state-anchored conditions. Formula and scenario configuration are fingerprinted,
and paired comparisons separate local competence from compounding penalty.

The model registry is manifest-backed and pins verified Hugging Face revisions.
Only `Qwen/Qwen3.6-35B-A3B-FP8` inherits qualified V1 evidence; V2 hot activation
remains experimental until immutable-image GPU acceptance records PID, timing,
memory, output/routing equivalence, and rollback evidence. Qwen 122B, GPTQ, gpt-oss
120B, and GLM entries remain disabled/fail-closed until their real router paths are
live-qualified.

The V1 sections and coordinates below remain historical evidence. A V2 manifest,
mock run, or local unit test is never substituted for live model acceptance.

The product is now a complete research loop rather than a collection of MVP cards.
Its exact application and fork commits were published as an attested image and
exercised on real Runpod, Daytona, and Anthropic services. All GPU compute and owned
Daytona sandboxes were stopped or deleted at handoff. This roadmap distinguishes
those states deliberately:

- **Implemented locally** means the code, UI, persistence contract, and automated
  coverage are present on this branch.
- **Live-accepted** means a specific application SHA, fork SHA, image digest, and
  external-resource run were recorded together.
- **Fail-closed** means metadata or a future integration may be visible, but the
  application refuses to launch it because admission evidence is incomplete.

The exact `0.3.0-rc.1` build has bounded live evidence and a completed compute
teardown recorded below. The stopped acceptance Pod remains undeleted only because
deletion would irreversibly erase its fresh 150 GB Pod volume and requires explicit
user approval. An older internal RC3 build remains historical evidence only.

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
| Runpod appliance | Authenticated same-origin FastAPI/React appliance; persistent data root; attested image/template tooling; managed vLLM on loopback; responsive shell. Exact-build desktop/390 px QA passed locally; public-proxy auth passed by API. | All eight account Pods were `EXITED` and no endpoints remained. Optional permanent deletion of the stopped acceptance Pod requires user approval; a second public-proxy visual pass was not claimed. |
| Model lifecycle | Registry-based Qwen A3B load/reload; topology/runtime facts; startup log; chat; persisted sessions; job adoption; process-group cleanup and startup reconciliation. The application has no public model-stop action. | Pod stop is the paid-compute control. A separately authenticated live client adopted the same load; public-browser hard reload remains unclaimed. |
| Benchmark command center | Full-screen catalog/problem/run views; editable criteria and budgets; item/attempt progress; performance; routing; paginated archive/items; streamed JSON/CSV export. | Fixture and tuned visible-answer MMLU-Pro/IFEval/LiveBench pairs passed the application flow live. |
| Benchmark breadth | Fixture, GSM8K, MMLU-Pro validation 70/test 12,032, IFEval strict 387/full reference 541, LiveBench Zebra 50, and custom JSONL. | MMLU-Pro/IFEval/LiveBench were prepared, problem-inspected, and paired in a non-thinking 4/3/1 cohort. This is acceptance evidence, not a leaderboard aggregate. |
| Profile Studio | Regular and agentic sources; explicit source weights; aggregate/per-layer References/Selection/Delta; mass/count metrics; fixed/global/cumulative/manual selection; immutable revisions. | Three non-uniform profiles were created live. The fixture and Aider-derived profiles were loaded; the smoke-derived 60/62/64 profile was created and inspected but not used for a reload. |
| Comparisons | Strict paired benchmark and agent-run views; outcome transitions; score/reward and performance deltas; profile/provenance compatibility. Benchmark comparisons persist; agent comparisons are deterministic on-demand views over durable runs. | Fixture comparison persisted; smoke and Aider agent comparisons recomputed with stable IDs and one regression each. |
| Agentic foundation | Native bounded controller; Daytona/fake providers; distinct reasoning/tool/command/stdout/stderr/verifier timeline; ATIF; per-inference routing/performance; durable trials/artifacts. | Baseline/masked smoke and Aider zebra contracts ran on Daytona; all acceptance-owned sandboxes were deleted. |
| Agent profiles | Profile Studio accepts selected agentic trials or whole runs and can mix agentic and regular sources. | Smoke- and Aider-derived custom profiles were created; the Aider profile was used in a paired rerun. |
| Evaluation | Fingerprinted deterministic criteria, trusted/user-authored verifier provenance, weighted aggregation, Anthropic/OpenAI/fake judges, cache/cost accounting. | One real Sonnet 5 request and zero-cost cache replay passed. Live standalone judging used a client guard because that endpoint has no server-side spend-cap field. |
| Isolation | Digest-pinned task images; public/submission/verifier/oracle separation; fresh verifier sandboxes; root-only hidden assets; unprivileged candidate subprocesses; exact-label cleanup. | Local exploit coverage passed; real trusted Aider clean-room verification passed and provider cleanup ended empty. |
| Recovery | Durable jobs, typed conflicts, reconnect adoption, cooperative cancellation, restart reconciliation, periodic Daytona cleanup, bounded external errors. | Live adoption and cancellation after 1/20 passed. Public-browser reload and a destructive orphan injection were not claimed. |
| External coding suites | Launchable Aider Python canary 3/225; pinned Terminal-Bench engineering 15, Aider balanced 12, and FeatureBench Fast manifests. | Aider zebra passed unmasked live and regressed under its custom profile. The three larger manifests remain fail-closed pending full admission. |

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
The target is one compatible 96 GB RTX PRO 6000 Blackwell. The Runpod template
describes the container, ports, storage, and environment; GPU type is selected
separately when the Pod is created. RC1 acceptance used the Workstation Edition.
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
use the published digest where Runpod accepts that coordinate; otherwise use the
full-application-SHA tag and record the verified digest it resolved to.

Task packs also require effective image digests. The current runnable Python packs
pin the official `python:3.12-slim` digest. RC1's immutable coordinates are:

- application `0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345`;
- fork `28a44cf4291c05af070c7d8398c462459b9992d1`;
- OCI index `sha256:b31b5b376c637fbd7c9c995deeae32e4773a8d005cc9f1c0d051c624c6c68694`;
- `linux/amd64` child
  `sha256:c1525fe1edfded45c8c5355e09a053a7c8efc77a98bc564a2039233e089fe9e8`; and
- [provenance attestation](https://github.com/sampazdan/vllm-moe-tools-suite/attestations/40091479),
  verified by `gh attestation verify` and recorded at Rekor index `2424156563`.

The accepted Pod used the full-application-SHA tag and resolved to that verified
child image. This was the proven Runpod launch coordinate; experiment records keep
both the tag and its resolved digest.

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

Benchmark and agent execution policies enforce `max_cost_usd` before paid judge
work by reserving a conservative upper bound. The standalone
`/api/evaluation/judge` diagnostic does not carry its own server-side spend cap; it
reports priced bounds and usage, while the caller must apply a guard or use the
run-integrated path when enforced admission is required.

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
turns, commands, tokens, terminations, timing, judge cost, and routing. Agent runs
are durable, but their comparison is a deterministic on-demand view identified by
the run pair; it is not persisted as a separate comparison record. A masked run is
never called comparable merely because it used the same task names.

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
- Managed shutdown is currently an internal reload/application-lifecycle operation;
  there is no public model-stop endpoint or Home-screen stop control.
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

Status: passed on application commit `0c8c8ffc…`: 189 backend tests passed with one
expected macOS root/UID skip; 18/18 frontend tests and the production build passed;
Ruff, lock, shell, JSON, Docker, secret, and final-diff checks passed. Exact-build
desktop and 390 px responsive QA also passed locally.

### Gate B — immutable container

- commit and push the exact vLLM fork;
- pass that full 40-character SHA to the publish workflow;
- build the exact application commit for `linux/amd64`;
- record OCI digest and provenance attestation;
- update the private Runpod template to the immutable launch coordinate; and
- verify the image startup hook retains Runpod `/start.sh` services while the app
  reaches `/readyz` within its bounded startup timeout.

Status: passed. Application `0c8c8ffc…` and fork `28a44cf…` produced attested OCI
index `sha256:b31b5b…` and verified `linux/amd64` child `sha256:c1525f…`.
The accepted Pod used the full-application-SHA tag, which resolved to that child;
this was the proven Runpod coordinate rather than a floating tag.

### Gate C — model and benchmark loop

1. Authenticate through the public Runpod proxy; separately verify the exact-build
   desktop and phone-width UI.
2. Load `Qwen/Qwen3.6-35B-A3B-FP8`; reconnect during load and require adoption of
   the same job/session.
3. Inspect topology, chat, reasoning presentation, and request metrics.
4. Run bounded GSM8K, MMLU-Pro, IFEval, and LiveBench samples.
5. Open problem/success criteria, live progress, item attempts, routing, and
   performance from their command-center pages.
6. Create a non-uniform custom profile, reload it, rerun the identical cohort, and
   persist the strict benchmark comparison.

Status: bounded core accepted. Public-proxy auth/CSRF and independent-client job
adoption passed; the 40 × 256, top-k 8 model loaded; and a 2/2 fixture baseline and
masked rerun both scored 1.0 with a persisted zero-delta comparison. Local
desktop/390 px visual QA passed, but a live-proxy visual browser pass is not claimed.
The initial MMLU-Pro/IFEval/LiveBench 3/2/2 cohorts exercised preparation,
inspection, execution, routing, and scoring but exhausted their small output
budgets in hidden reasoning. The tuned non-thinking 4/3/1 rerun then produced
visible answers and strict persisted pairs: MMLU-Pro 0.75 → 0.25 (two regressions),
IFEval 1.0 → 0.0 (three regressions), and LiveBench 1.0 → 0.0 (one regression).
All items reported zero reasoning tokens. This eight-item canary is not a
leaderboard result.

### Gate D — agentic/profile loop

1. Run read-only Daytona preflight.
2. Run one complete unmasked smoke or repository-engineering cohort.
3. Require distinct reasoning/response/command/stdout/stderr/verifier events, ATIF,
   per-inference routing, performance, captured submissions, trusted outcomes, and
   confirmed task/verifier sandbox deletion.
4. Create a custom profile from all or selected trial traces; inspect aggregate and
   individual layers, then make a deliberate manual edit.
5. Reload the profile and rerun the identical agent contract.
6. Open and inspect the deterministic strict agent comparison derived from both
   durable runs.
7. Exercise cancellation and one interruption/orphan recovery path.

Status: accepted for the bounded smoke and Aider zebra canaries. Baseline and
masked runs, distinct full-reasoning trajectories, per-call routing/performance,
trusted clean-room verification, two 60/62/64 agent-derived profiles, and two
deterministic regression comparisons were observed. Every acceptance-owned Daytona
session was deleted; provider listing ended empty. Live cancellation passed after
1/20 items. A destructive orphan injection was not required or claimed.

### Gate E — frontier judge and cost

- run one small Anthropic judge with configured token prices and a bounded request;
- verify structured result, fingerprint, rationale, usage, latency, incurred cost,
  and budget debit;
- repeat to prove a cache hit incurs zero new cost; and
- exercise cache and local preflight/failure coverage without exceeding the guard.

Status: live request/cache path passed. Sonnet 5 returned score 1.0/pass for
341 input and 68 output tokens at $0.001362 incurred; the identical replay used the
same ID/hash and incurred $0. The standalone endpoint has no server-side
`max_cost_usd`; acceptance used a $0.025 driver guard and observed a $0.01115
server-computed request bound. Server-enforced run-level caps remain covered by
local tests.

### Gate F — teardown

- list and confirm zero application-owned Daytona task/verifier sandboxes;
- stop every Pod created or resumed for acceptance;
- confirm no paid Runpod compute remains active;
- record the current provider ledger and conservative budget ceiling, then recheck
  after billing catches up; and
- retain only network volumes/storage chosen intentionally.

Status: complete for paid compute and owned sandboxes; spend accounting remains a
provisional provider snapshot. Acceptance Pod
`7te5po3smp0227` reached `EXITED` at approximately 2026-08-11 19:23 UTC; all eight
account Pods were `EXITED`, no Runpod endpoint remained, all eight durable Daytona
rows were `deleted`, and direct provider inventory was empty. Permanent deletion of
the stopped acceptance Pod awaits explicit user approval because it would
irreversibly erase its fresh 150 GB volume; storage may continue billing. Existing
network volumes `w7jl3o07i1`, `hhsfk4sdrc`, and `gbichawf78` were retained
untouched. Earlier stalled fresh attempt Pod `o204fumgnxb1dj` was permanently
deleted, and its Pod-volume data is unrecoverable.

## RC1 acceptance record — observed 2026-08-11

- **Immutable build:** application
  `0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345`, fork
  `28a44cf4291c05af070c7d8398c462459b9992d1`, OCI index
  `sha256:b31b5b376c637fbd7c9c995deeae32e4773a8d005cc9f1c0d051c624c6c68694`,
  `linux/amd64` child
  `sha256:c1525fe1edfded45c8c5355e09a053a7c8efc77a98bc564a2039233e089fe9e8`,
  and [attestation 40091479](https://github.com/sampazdan/vllm-moe-tools-suite/attestations/40091479)
  at Rekor index `2424156563`.
- **Runpod:** template `w2wkhbveg2` (`moe-tools-agentic-a3b-rc1`) and Pod
  `7te5po3smp0227` used one 96 GB RTX PRO 6000 Blackwell Workstation Edition.
  The template itself did not choose the GPU. The full-SHA image tag resolved to
  the verified child digest.
- **Auth/reconnect/UI:** public-proxy API checks observed anonymous 401, login 200,
  a `Secure`/`HttpOnly`/`SameSite=Strict` cookie, missing-CSRF 403, valid-CSRF 200,
  and independent-client adoption of the same load job/session. Exact-build visual
  QA passed locally at desktop and 390 px. The live in-app browser was unavailable,
  so no public-proxy visual or browser-hard-reload claim is made.
- **Model:** the first fresh load downloaded 34.89 GiB in 74.46 seconds, loaded
  34.23 GiB of weights in 89.07 seconds, and compiled in 94.1 seconds. Runtime was
  vLLM `0.1.dev2+moe-tools`, `Qwen3_5Moe`, FP8, 4,096 maximum length, 0.90 GPU
  utilization target, 2,048 maximum batched tokens, 8 maximum sequences, routed
  IDs/weights, per-request metrics, and the `qwen3` parser. Topology was 40 layers ×
  256 experts, top-k 8; memory was 86.3/97.9 GiB. Model replacement exercised
  internal stop/reload; there is no public stop-model endpoint.
- **Fixture pair:** baseline `78aea4ce…` and masked `7f233d71…` each scored 1.0
  with 2/2 passes. Profile `92d9f816…` kept 63 experts in layer 0 and 64 elsewhere
  (2,559/10,240, 24.9902%) while retaining 96.1701% observed mass. Persisted
  comparison `febeed7f…` had delta 0 and two retained passes.
- **Smoke pair/profile:** baseline `4afc6304…`/`06dd84fe…` passed at reward 1.0
  with 9 turns, 7 commands, 6,284 tokens, and 186.53 TPS. Masked
  `f35a42fe…`/`b9aacaf9…` reached `turn_limit`, reward 0, with 10 turns, 0
  commands, 12,930 tokens, and 216.70 TPS. Deterministic on-demand comparison
  `c7bd0e…` reported one regression. Smoke-derived custom profile `73100ed9…`
  cycled 60/62/64 eligible experts per layer, kept 24.1992% of coordinates, and
  retained 77.8729% observed mass.
- **Aider pair/profile:** baseline `4d2c8222…`/`55050d5c…` passed its trusted
  clean-room verifier at reward 1.0 with full reasoning, 19 turns, 3 commands,
  52,429 tokens, and 218.39 TPS. Aider-derived custom profile `5a509409…` cycled
  60/62/64, kept 24.1992% of coordinates, and retained 74.0881% observed mass.
  Masked `c2b5be96…`/`0391b5e5…` reached `token_limit`, reward 0, with 23
  turns, 5 commands, 59,817 tokens, and 208.58 TPS; on-demand comparison
  `758bb58…` reported one regression. Their task/verifier sandboxes were deleted.
  These are negative-transfer results for the profiles, not provider failures.
- **Higher-level cohorts:** MMLU-Pro validation, IFEval strict, and LiveBench Zebra
  were prepared and problem-inspected. The initial bounded 3/2/2 thinking-enabled
  run spent its response budgets in hidden reasoning and left empty answer text, so
  it is not quality evidence. The tuned 4/3/1 pairs used `enable_thinking=false`,
  reported zero reasoning tokens, and applied profile
  `5a509409-7e93-40bf-8153-14a32b1ce604`:
  - MMLU-Pro baseline `d3aa4a4f-f463-4390-8508-e1a69aa70f54` scored 0.75
    (3/4, 214.30 TPS), masked `c584aa35-93af-480a-98e0-c29460150b52`
    scored 0.25 (1/4, 205.80 TPS), and comparison
    `e69c95c3-f88d-4dab-8c2a-f0366fad86d7` recorded delta -0.50 with two
    regressions;
  - IFEval baseline `f60fc5c4-f459-4f4e-97cd-9a85d886ff32` scored 1.0
    (3/3, 211.53 TPS), masked `6ab614ef-0edf-4591-870d-0ce257d6602a`
    scored 0.0 (0/3, 176.37 TPS), and comparison
    `2cb149fd-dc0a-49fc-9377-1ead6f7df206` recorded delta -1.0 with three
    regressions; and
  - LiveBench baseline `2986ae98-a81f-43f5-9599-a1c04b636925` scored 1.0
    (1/1, 221.20 TPS), masked `d2777294-1a6d-47d1-ab18-a2a97337e3ce`
    scored 0.0 (0/1, 219.23 TPS), and comparison
    `4d750ba7-b6c5-49e3-852e-35dba4ff1e8a` recorded delta -1.0 with one
    regression.
  Baseline completion-token/routed-slot counts were 3,257/1,307,200,
  1,495/550,720, and 1,678/639,680 respectively. These eight items are paired
  acceptance evidence, not benchmark aggregates.
- **Judge:** standalone Sonnet 5 result `fc09361a…`, request `533adbbb…`, scored
  1.0/pass with 341 input/68 output tokens and 3,219.16 ms latency. Equivalent and
  incurred cost were $0.001362 at $2/M input and $10/M output; computed bound was
  $0.01115. The identical cached replay retained ID/hash with $0 incurred and
  $0 bound. This was client-guarded at $0.025 because standalone judging has no
  server-enforced cap.
- **Recovery/cleanup:** job `000bd930…` cancelled at 1/20 and released the active
  slot. All eight Daytona rows are `deleted`, no cleanup error remains, and direct
  provider inventory found zero controller-owned sandboxes.
- **Final teardown/spend:** acceptance Pod `7te5po3smp0227` stopped/`EXITED` at
  approximately 2026-08-11 19:23 UTC, ending GPU billing. All eight account Pods
  were `EXITED`; no Runpod endpoint remained. The Pod was not permanently deleted
  because that would irreversibly erase its fresh 150 GB volume and requires
  explicit user approval. Existing network volumes `w7jl3o07i1`, `hhsfk4sdrc`,
  and `gbichawf78` were untouched. Earlier stalled fresh attempt Pod
  `o204fumgnxb1dj` was permanently deleted, and its Pod-volume data is
  unrecoverable. As of the final audit, Runpod's lagging billing API reported for
  `7te5po3smp0227` $0.0493180654 at 17:00–18:00 UTC plus $0.9599268073 at
  18:00–19:00 UTC: $1.0092448727 posted so far. It had no 19:00–19:23 bucket yet,
  so this is not a finalized total. A conservative advertised-rate ceiling for that
  Pod is approximately 1 hour 23 minutes × $1.89/hour ≈ $2.62, plus minor disk
  charges, well below the $50 budget. The ledger exposed no separate row for
  deleted attempt `o204fumgnxb1dj`; neither figure is a finalized all-attempt
  acceptance total. The ledger must be checked after billing catches up; no exact
  tail estimate is claimed. Daytona's eight durable rows were deleted and direct
  owned inventory was empty; the tuned benchmark rerun used no Daytona. Its
  separate monetary total was not exposed by this audit and is not invented.
- **Post-acceptance configuration:** template `w2wkhbveg2` was verified on immutable
  tag `sha-0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345` instead of the slow OCI-index
  coordinate. Stopped original Pod `kw3whlid6esyhs` had its temporary auth value
  restored to the Runpod secret reference, with its other 23 environment entries
  preserved.

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
contracts, inspect strict paired benchmark and agent views, and export their durable
source runs/profiles without opening a shell after launch.

Completion also requires that trusted verifier assets never enter the model
sandbox, user-authored verifiers are labelled accurately, paid judge work remains
inside its enforced run cap or explicit standalone caller guard,
cancellation/recovery remain observable, and all acceptance-owned paid compute and
sandboxes are stopped or deleted at handoff.

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
