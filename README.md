# MoE Tools Test Suite

MoE Atelier is a single-user research workbench for understanding expert routing,
building expert-eligibility profiles, and measuring what those profiles do to a
Mixture-of-Experts model. It keeps one-request benchmarks and multi-turn coding
workloads in the same reproducible loop:

1. load a model through the custom vLLM fork;
2. run a pinned cohort and capture routed expert IDs and probabilities;
3. explore the full layer-by-expert topology;
4. create an immutable custom profile from one or more workloads;
5. reload the model with that profile; and
6. rerun the same contract and inspect a paired quality, behavior, routing, and
   performance comparison.

The enabled model is `Qwen/Qwen3.6-35B-A3B-FP8`. The registry, topology contract,
and model-session records are designed for more models later.

See [ROADMAP.md](ROADMAP.md) for architecture, scope, release gates, and the next
research stages.

## Release status

`0.3.0-rc.1` is the current finish-line candidate. The exact application and fork
commits have been published as an attested `linux/amd64` image and have completed a
bounded live Runpod/Daytona/Anthropic acceptance pass. The immutable coordinates
and observed results are recorded below. All GPU compute and owned Daytona
sandboxes were stopped or deleted at handoff. The stopped acceptance Pod remains
undeleted because permanent deletion would irreversibly erase its fresh 150 GB Pod
volume and requires an explicit user decision.

An older internal candidate called **RC3** completed a bounded Runpod/Daytona test
on 2026-08-11. Those historical results are retained near the end of this README
because they establish that the basic appliance, real-model telemetry, profile
reload, and Daytona lifecycle worked. They are not acceptance evidence for
`0.3.0-rc.1` or its newer evaluator, profile, command-center, and clean-room paths.

## What is in the application

The warm, editorial interface is organized as full-screen research command centers
that also collapse cleanly onto a phone:

- **Home** owns model load/reload state, topology, runtime facts, startup logs, a
  compact chat, and the active durable job.
- **Bench** browses datasets, opens individual problems and their public success
  criteria, creates immutable cohorts, edits evaluation and execution contracts,
  follows progress, and opens complete run records.
- **Agents** selects a pinned coding pack and task attempts, configures the native
  controller and remote sandbox, follows every trial, and opens its full
  trajectory, verifier, artifact, routing, and performance record.
- **Profiles** combines regular benchmark runs and agentic trials, assigns explicit
  source weights, explores routing, edits expert eligibility, and saves immutable
  profile revisions.
- **Compare** contains strict paired views for both regular benchmarks and agentic
  runs, including scores/rewards, outcome transitions, routing, budgets, and
  performance. Benchmark comparisons are persisted. Agent comparisons are
  deterministic views recomputed from their two durable runs; they are not stored
  as independent archive records.
- **Archive** persists benchmark runs, agent runs, profiles, and benchmark
  comparisons.
  Benchmark summaries and item histories are paginated; JSON/CSV exports stream so
  large cohorts do not need a second full in-memory response.

Browser polling survives transient Runpod proxy failures, reconnects, and hard
reloads. Model loads, dataset preparation, benchmark runs, and agent runs are
durable server jobs. An identical model-load retry adopts the existing job; an
incompatible submission receives a typed conflict with the active work. Cooperative
cancellation uses a visible `cancelling` state and does not release the single-work
mutex until the worker has acknowledged termination. Startup reconciliation repairs
interrupted records and retries owned process or sandbox cleanup.

There is no public “stop model” action in this candidate. The managed process is
stopped internally when a model/profile reload replaces it or when the application
shuts down; stopping paid idle compute remains a Runpod Pod operation.

## Benchmark catalog

Every adapter pins source provenance, selected item order, prompt and scorer
versions, generation defaults, and a content fingerprint. Downloads are lazy and
verified before a dataset becomes runnable.

| Adapter | Runnable cohort | Purpose and scoring |
| --- | ---: | --- |
| Arithmetic routing fixture | 12 | Deterministic GPU-free exact-match smoke test. |
| [GSM8K](https://github.com/openai/grade-school-math) | 1,319 | Pinned official test split; final numeric answer exact match. |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) validation | 70 | Complete official validation split, five questions in each of 14 domains; application calibration, not a leaderboard test score. |
| MMLU-Pro test | 12,032 | Complete pinned test split with final answer-letter scoring. |
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) strict subset | 387 | Dependency-free application cohort across 20 evaluator families; all constraints must pass. It is not the official loose/instruction-level aggregate. |
| IFEval full reference | 541 | Complete pinned prompt corpus for routing and qualitative inspection; intentionally ungraded until all official evaluator families are integrated. |
| [LiveBench](https://github.com/LiveBench/LiveBench) Zebra 2024-06 | 50 | Reproducible archived zebra-puzzle cohort with its release-specific boolean extractor; not a current LiveBench leaderboard aggregate. |
| Custom JSONL | User supplied | Content-addressed imports with dataset-default, exact, contains, bounded regex, numeric tolerance, multiple-choice, JSON-schema, or ungraded criteria. |

Problem inspectors show the prompt, metadata, generation defaults, public criteria,
and a fingerprint/count for protected criteria without exposing their contents.
Repeated attempts remain distinct records rather than being silently pooled or
best-of-selected.

### Custom JSONL

Use **Import JSONL** in the benchmark command center. Each non-empty line is one
object:

```json
{"id":"sum-1","prompt":"Return only the result of 2 + 2.","expected":"4","category":"smoke","scoring":"exact"}
{"id":"workflow-1","prompt":"Inspect the repository and propose a fix.","category":"agentic","scoring":"ungraded","metadata":{"harness":"daytona"}}
```

`prompt` is required. `id` and `category` are optional. A criterion that needs a
reference also needs `expected`. Imports are validated before storage and limited
to 2 MB by default (`MOE_TOOLS_MAX_CUSTOM_DATASET_BYTES`). User regexes are compiled
and executed through bounded validation rather than Python's unbounded `re` engine.

## Success criteria are not execution budgets

This distinction is central to the workbench.

An **evaluation contract** answers _“did the result succeed?”_ It contains named,
weighted, public or protected criteria; required flags; an `all_required` or
`weighted_threshold` aggregation rule; and the final pass threshold. Regular runs
can use the dataset scorer, exact/contains/regex/numeric/multiple-choice/JSON checks,
or remain ungraded. Agentic runs always retain the trusted task-pack verifier and
may add clearly labelled user-authored sandbox verifier commands.

An **execution policy** answers _“how much work may this run do?”_ It contains
attempts, concurrency, whole-run and per-item timeouts, total turns, commands, the
whole-run **prompt plus generated token** cap, fail-fast behavior, and an optional
incremental frontier-judge API cost cap. A separate generation setting controls the
maximum output tokens for each individual model response, alongside temperature,
seed, and thinking/reasoning presentation. Reaching a token, turn, command, time, or
cost limit is a termination cause; it is never counted as success by itself. The
evaluation contract still runs after termination whenever safe cleanup permits.

Both contracts are fingerprinted and archived. A legacy record that predates them
is shown as **metadata unavailable** instead of being assigned invented defaults.

### Frontier-model judges

Evaluation contracts can add a structured single, reference, or explicit pairwise
judge using Anthropic, OpenAI, or the deterministic fake provider. The rubric,
threshold, repetitions, presentation order, model, prices, and protocol are part of
the immutable judge fingerprint. Results include rationale, token usage, latency,
cache status, and both equivalent and newly incurred cost.

Cost caps are fail-closed when paid judging is part of a benchmark or agent run.
Explicit input/output prices are required, and the runner reserves a conservative
full-request upper bound before every uncached provider request, including all
repetitions. Cached decisions incur zero new spend. If a provider fails after a
request may have been accepted, the run records cost uncertainty and debits
conservatively instead of pretending the call was free.

The standalone `/api/evaluation/judge` diagnostic endpoint does **not** accept an
independent server-enforced `max_cost_usd`. It returns priced upper-bound and usage
accounting, but callers must bound repetitions/output and apply their own guard, or
put the judge inside a run execution policy when server-enforced spend admission is
required.

Configure either provider in the controller environment; credentials are stripped
before vLLM and sandbox processes are launched:

```text
MOE_TOOLS_ANTHROPIC_API_KEY=...
MOE_TOOLS_ANTHROPIC_JUDGE_MODEL=claude-sonnet-5   # optional
MOE_TOOLS_OPENAI_API_KEY=...                       # optional alternative
MOE_TOOLS_OPENAI_JUDGE_MODEL=gpt-5                 # optional
MOE_TOOLS_JUDGE_TIMEOUT_SECONDS=180                 # optional
```

## Expert Explorer and Profile Studio

Every regular run and agentic trial can open the same explorer. It defaults to
**Aggregate · all layers**, which preserves every layer × expert coordinate; it
does not collapse equal expert IDs across layers. A layer selector opens the
individual expert grid.

The explorer provides:

- **References** for probability mass or reference/share counts;
- **Selection** for eligible/ineligible experts and observed mass retained; and
- **Delta** when a compatible comparison source is present.

Profile Studio can combine many benchmark runs and agentic trials. Its
`weighted_source_normalized` aggregation first normalizes each source matrix to unit
total, then applies the source's explicit relative weight. A long run therefore
does not dominate solely because it emitted more tokens. The aggregation version,
source IDs, weights, filters, metric, and resulting matrix are fingerprinted, and
the server recomputes that fingerprint when a profile is saved.

Users can select by routing probability mass or reference count and then:

- keep a fixed number of top experts in each layer;
- spend a global expert budget while preserving the model's `top_k` floor in every
  routed layer;
- keep the smallest global selection that reaches a cumulative observed-mass
  target;
- apply a fixed selection to one layer; or
- edit experts manually, keep a layer at its minimum, restore it fully, revert it,
  or copy the previous layer.

Every save creates an immutable revision with source lineage, strategy settings,
observed mass retained, parent profile, topology validation, and profile/source
fingerprints. A profile can be created directly from a whole agent run by selecting
all of its trial traces, or from any hand-picked mix of trials and ordinary runs.

Profiles are applied only during model load. Changing one restarts vLLM and creates
a new model session. The current fork changes expert eligibility before top-k
routing; ineligible weights remain loaded. This is **behavioral masking**, not
checkpoint pruning, expert offload, or a demonstrated reduction in VRAM.

## Native agentic coding

The model-centered controller is `bash-json-v1`, revision 1. Each turn returns one
JSON action: a sandbox shell command or a final response. The controller calls the
local model gateway, never an opaque external agent, so each inference is linked to
its routed-expert artifact.

The full-screen trajectory clearly separates:

- model reasoning, with off/compact/full presentation;
- assistant responses;
- parsed tool and shell commands;
- stdout and stderr observations;
- verifier events and termination; and
- artifacts and patch/submission files.

Per-call and aggregate performance panels report prompt, reasoning, and completion
tokens; latency; time to first token when the runtime supplies a true streaming
timestamp; prefill/decode/queue/inter-token timing when supplied; tokens per second;
wall time; command and turn counts; and judge/API cost. Missing runtime metrics stay
unknown rather than being inferred from unrelated timings.

Extended chat prompts repeat history. The gateway retains token IDs per trial and
uses the fork's `routed_experts_prompt_start` request field only after exact
common-prefix alignment. It then records routing for the novel suffix and new
generation. Ambiguous alignment fails open to a full capture, and prefix state is
isolated per trial.

Trajectories export as `ATIF-v1.7`, following Harbor's
[Agent Trajectory Interchange Format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md).
The native controller is Harbor-compatible at the artifact boundary; it is not an
embedded Harbor runtime.

### Runnable task packs

| Pack | Tasks | Intended use |
| --- | ---: | --- |
| `smoke-python-v1` revision 2 | 3 | Tiny dependency-free repair/implementation tasks for controller, routing, verifier, and cleanup acceptance. |
| `repo-engineering-v1` revision 2 | 3 | Moderate first-party repository tasks covering JSONL pipeline repair, dependency-graph release planning, and bounded asynchronous concurrency. |
| `aider-polyglot-python-canary-3` | 3 of 225 | Pinned Python `zipper`, `wordy`, and `zebra-puzzle` edits adapted from [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) through Harbor. This is an application canary, **not** an official Aider leaderboard cohort or score. |

The source-pinned external manifests for Terminal-Bench 2.1 engineering 15, Aider
Polyglot balanced 12, and FeatureBench Fast 100 remain intentionally
**non-launchable**. The application fails closed because their complete runner/image
integration, license review where needed, and per-task oracle-pass/no-op-fail checks
are not complete. A pinned manifest is provenance, not permission to claim a score.

### Daytona provider

The fake provider is deterministic and never invokes a host subprocess. Real coding
trials use private ephemeral Daytona sandboxes through the exact
[`daytona==0.192.0`](https://pypi.org/project/daytona/0.192.0/) SDK pin. Provider
listing is side-effect free; the explicit preflight is a bounded read-only list
request and creates no sandbox.

```text
MOE_TOOLS_DAYTONA_API_KEY=...
MOE_TOOLS_DAYTONA_API_URL=https://app.daytona.io/api   # optional
MOE_TOOLS_DAYTONA_TARGET=us                            # us or eu
```

Each trial records ownership labels, image digest, resources, network policy,
timeouts, artifacts, and cleanup state. The controller never forwards Runpod,
Daytona, application-auth, Hugging Face, Anthropic, or OpenAI credentials into a
task. Cleanup verifies exact ownership before deletion, waits for confirmed absence,
reconciles interrupted sessions at startup, and periodically retries owned stale
sessions. A retryable cleanup failure remains visible and blocks conflicting work.

### Trusted and user-authored verifiers

Task contracts explicitly separate starter/public files, submission files,
protected verifier files, and oracle files. The model sees only the public task
surface. After the agent finishes, the controller captures only declared submission
files and builds each verifier in a fresh, network-disabled clean room from the
canonical starter plus that captured submission.

Trusted pack verifiers receive protected test assets but never oracle solutions.
Protected files are root-only; candidate code runs through an isolated subprocess
as an unprivileged uid/gid so it cannot read or rewrite hidden tests. Oracle assets
are eligibility-time evidence only and do not enter the agent or verifier runtime.
Each additional verifier command authored in the UI runs in its own fresh sandbox
and is labelled `user_authored_sandbox`; it is useful research policy, not a trusted
benchmark verifier. Verifier sandboxes are ownership-tracked and deleted with the
same strict cleanup rules as primary task sandboxes.

## Local development

Prerequisites are Python 3.12, `uv`, Node.js 24.6.0, and npm. Dependency locks are
authoritative.

```bash
uv sync --frozen
npm --prefix frontend ci
```

Run the backend:

```bash
uv run uvicorn --app-dir backend/src moe_tools_suite.main:app \
  --host 127.0.0.1 --port 8080 --reload
```

Run the frontend in another terminal:

```bash
npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to port 8080. Authentication is
disabled by default in development. To test it locally, copy `.env.example` to
`.env`, generate a long `MOE_TOOLS_AUTH_TOKEN`, set `MOE_TOOLS_REQUIRE_AUTH=1`, and
leave `MOE_TOOLS_COOKIE_SECURE=0` while using plain localhost HTTP.

Run the release checks:

```bash
uv run ruff check backend scripts
uv run pytest -q
npm --prefix frontend test -- --run
npm --prefix frontend run build
bash -n docker/runpod/pre_start.sh scripts/create_runpod_template.sh
```

## Container and Runpod runbook

The appliance is a persistent Runpod Pod, not Serverless: it owns an interactive
session, a warm 35B-A3B model, durable research jobs, and a persistent cache. The
public surface is only the same-origin application on `8080/http`; vLLM stays on
`127.0.0.1:8000`. Phone and laptop access use Runpod's built-in HTTPS proxy:

```text
https://<pod-id>-8080.proxy.runpod.net
```

The proxy is public transport, not authentication, so the application requires a
long single-user token and issues an `HttpOnly`, secure, same-site session cookie.
Long operations return job IDs and poll rather than relying on a connection longer
than the proxy limit. See Runpod's [template](https://docs.runpod.io/pods/templates/overview),
[port](https://docs.runpod.io/pods/configuration/expose-ports), and
[network-volume](https://docs.runpod.io/storage/network-volumes) guides.

### Reproducible image

The Dockerfile pins its Node 24.6.0 builder and Runpod CUDA 13/PyTorch 2.9 base by
OCI digest, installs Python 3.12 from the locked `uv` environment, requires a full
40-character vLLM fork commit, and verifies `libtorch_cuda.so` during the build.
Agent task images are also pinned by digest; the current Python packs use an
immutable `python:3.12-slim` digest. No release uses a floating `latest` tag.

Build locally with the sibling fork as a named BuildKit context:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-context vllm_source=../vllm-moe-tools \
  --file docker/Dockerfile.runpod \
  --build-arg VLLM_SOURCE_REF="$(git -C ../vllm-moe-tools rev-parse HEAD)" \
  --tag YOUR_REGISTRY/moe-tools-test-suite:0.3.0-rc.1 \
  .
```

`.github/workflows/publish-container.yml` publishes version and application-commit
tags to `ghcr.io/sampazdan/vllm-moe-tools-suite`, creates a provenance attestation,
and requires the exact fork SHA as a manual input. RC1 was published from:

- application `0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345`;
- vLLM fork `28a44cf4291c05af070c7d8398c462459b9992d1`;
- OCI index `sha256:b31b5b376c637fbd7c9c995deeae32e4773a8d005cc9f1c0d051c624c6c68694`;
- `linux/amd64` child
  `sha256:c1525fe1edfded45c8c5355e09a053a7c8efc77a98bc564a2039233e089fe9e8`; and
- tags `0.3.0-rc.1` and
  `sha-0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345`.

The [GitHub provenance attestation](https://github.com/sampazdan/vllm-moe-tools-suite/attestations/40091479)
verified successfully with `gh attestation verify`; its transparency-log entry is
Rekor index `2424156563`. The accepted Runpod Pod used the full-application-SHA tag,
which resolved to the verified child image above. Keep the full SHA plus resolved
digest in experiment records; the full-SHA tag was the proven Runpod launch
coordinate for this build.

### Secrets and template

Create Runpod secrets before deploying `deploy/runpod-agentic-template.example.json`:

| Template reference | Application environment | Required for |
| --- | --- | --- |
| `{{ RUNPOD_SECRET_moe_tools_auth_token }}` | `MOE_TOOLS_AUTH_TOKEN` | Login; required. |
| `{{ RUNPOD_SECRET_daytona_api_key }}` | `MOE_TOOLS_DAYTONA_API_KEY` | Real coding trials. |
| `{{ RUNPOD_SECRET_MOE_TOOLS_ANTHROPIC_API_KEY }}` | `MOE_TOOLS_ANTHROPIC_API_KEY` | Anthropic judge; optional. |

The uppercase Anthropic secret name is deliberate. An unresolved placeholder is
treated as unconfigured and is never passed to vLLM or a sandbox. A Hugging Face
token may be added through the private template when the model download needs it.

Create the template with the full application-SHA tag proven on Runpod. A digest
coordinate is also valid when the selected Runpod deployment path accepts OCI
index digests:

```bash
export RUNPOD_API_KEY="..."
export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:sha-0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345"
export MOE_TOOLS_TEMPLATE_MODE="agentic"
export MOE_TOOLS_TEMPLATE_NAME="moe-tools-agentic-a3b"
./scripts/create_runpod_template.sh
```

The script rejects `latest`, malformed digests, and invalid modes. The generated
template configures port 8080, a 50 GB container disk, and `/workspace` for
application state and the model cache. Runpod templates do not select a GPU: choose
a compatible 96 GB RTX PRO 6000 Blackwell offering when the Pod is deployed. RC1
acceptance used the **Workstation Edition**, not the Server Edition. Attach a
network volume at `/workspace` if records and weights must outlive the Pod. A
volume is tied to one Runpod data center and continues billing after the Pod stops.

### RC1 live-acceptance driver

The public API canary covers the bounded baseline-to-mask path. Daytona and
Anthropic are opt-in because they create paid work:

```bash
MOE_TOOLS_CANARY_BASE_URL=https://POD_ID-8080.proxy.runpod.net \
MOE_TOOLS_CANARY_TOKEN="..." \
uv run python scripts/acceptance_canary.py --daytona --anthropic-judge
```

### RC1 acceptance record — observed 2026-08-11

The immutable application/fork/image coordinates are recorded in
[Reproducible image](#reproducible-image). Template `w2wkhbveg2`
(`moe-tools-agentic-a3b-rc1`) launched acceptance Pod `7te5po3smp0227` on one
96 GB RTX PRO 6000 Blackwell **Workstation Edition**. GPU choice happened at Pod
deployment, not in the template.

Local gates on the exact application commit completed with 189 backend tests
passing and one expected macOS root/UID isolation skip, 18/18 frontend tests,
the production frontend build, Ruff lint/format, lock consistency, shell/JSON and
Docker checks, and a clean release diff. The exact production UI was exercised
locally at desktop and 390 px widths. The in-app browser runtime was unavailable
during the live Runpod window, so this record does not claim a second visual
desktop/phone pass against the public proxy. Live public-proxy evidence is API
level: anonymous API access returned 401, login returned 200, the session cookie
was `Secure`, `HttpOnly`, and `SameSite=Strict`, missing CSRF returned 403, and a
valid CSRF request returned 200. A separately authenticated client rediscovered the
same active load job and model session, exercising reconnect/adoption without
claiming a live browser hard reload.

The first uncached `Qwen/Qwen3.6-35B-A3B-FP8` load downloaded 34.89 GiB in
74.46 seconds, loaded 34.23 GiB of weights in 89.07 seconds, and compiled in
94.1 seconds. The runtime reported vLLM `0.1.dev2+moe-tools`, `Qwen3_5Moe`, FP8,
maximum length 4,096, GPU utilization target 0.90, 2,048 maximum batched tokens,
eight maximum sequences, routed expert IDs and weights, per-request metrics, and
the `qwen3` reasoning parser. The accepted topology was 40 routed layers × 256
experts with router top-k 8. Post-load GPU memory was 86.3/97.9 GiB. Profile
changes exercised managed stop-and-reload; a public stop-model action is not part
of this candidate.

The deterministic fixture produced a complete paired benchmark record:

- baseline run `78aea4ce…`: score 1.0, 2/2 passed;
- custom profile `92d9f816…`: 63 eligible experts in layer 0 and 64 in every
  other routed layer, 2,559/10,240 coordinates (24.9902%) eligible, retaining
  96.1701% of observed routing mass;
- masked run `7f233d71…`: score 1.0, 2/2 passed; and
- persisted comparison `febeed7f…`: delta 0 with two retained passes.

Two separate agent traces also produced genuinely custom, non-uniform profiles:

- The unmasked `smoke-python-v1` run `4afc6304…`/trial `06dd84fe…` passed with
  reward 1.0, 9 turns, 7 commands, 6,284 tokens, and reported mean 186.53 TPS.
  Under the fixture profile, run `f35a42fe…`/trial `b9aacaf9…` reached its turn
  limit with reward 0 after 10 turns, 0 commands, 12,930 tokens, and 216.70 TPS.
  Deterministic comparison `c7bd0e…` recorded one regression. Its trace then
  yielded custom profile `73100ed9…`, with a cyclic 60/62/64 experts per layer,
  24.1992% of topology coordinates eligible, and 77.8729% observed mass retained.
- The unmasked Aider zebra canary run `4d2c8222…`/trial `55050d5c…` passed its
  trusted clean-room verifier with reward 1.0, full reasoning, 19 turns, 3
  commands, 52,429 tokens, and 218.39 TPS. Its custom profile `5a509409…` used
  the same deliberately non-uniform 60/62/64 layer cycle, retaining 24.1992% of
  coordinates and 74.0881% of observed mass. The identical masked contract in run
  `c2b5be96…`/trial `0391b5e5…` reached its token limit with reward 0 after 23
  turns, 5 commands, 59,817 tokens, and 208.58 TPS; deterministic comparison
  `758bb58…` recorded one regression.

Both negative-transfer outcomes are model/profile results, not Daytona lifecycle
failures. All involved task and verifier sandboxes reached `deleted`. Agent
comparison IDs above are deterministic values recomputed on demand from the two
durable run records; unlike `febeed7f…`, they are not independent persisted
comparison rows.

MMLU-Pro validation, IFEval strict, and LiveBench Zebra were downloaded, prepared,
and problem-inspected. Their first bounded 3/2/2-item thinking-enabled pass consumed
the small response budgets in model reasoning and left no answer text after
hidden-reasoning parsing. Those outputs remain generation-policy evidence, not
quality measurements.

The tuned 4/3/1-item rerun disabled thinking and recorded zero reasoning tokens for
every item. It paired the baseline against Aider-derived custom profile
`5a509409-7e93-40bf-8153-14a32b1ce604` under otherwise identical contracts:

| Cohort | Baseline | Masked | Persisted comparison |
| --- | --- | --- | --- |
| MMLU-Pro validation 4 | `d3aa4a4f-f463-4390-8508-e1a69aa70f54`: 3/4, 0.75, 214.30 TPS | `c584aa35-93af-480a-98e0-c29460150b52`: 1/4, 0.25, 205.80 TPS | `e69c95c3-f88d-4dab-8c2a-f0366fad86d7`: delta -0.50, 2 regressions |
| IFEval strict 3 | `f60fc5c4-f459-4f4e-97cd-9a85d886ff32`: 3/3, 1.0, 211.53 TPS | `6ab614ef-0edf-4591-870d-0ce257d6602a`: 0/3, 0.0, 176.37 TPS | `2cb149fd-dc0a-49fc-9377-1ead6f7df206`: delta -1.0, 3 regressions |
| LiveBench Zebra 1 | `2986ae98-a81f-43f5-9599-a1c04b636925`: 1/1, 1.0, 221.20 TPS | `d2777294-1a6d-47d1-ab18-a2a97337e3ce`: 0/1, 0.0, 219.23 TPS | `4d750ba7-b6c5-49e3-852e-35dba4ff1e8a`: delta -1.0, 1 regression |

The baseline runs recorded 3,257/1,495/1,678 completion tokens and
1,307,200/550,720/639,680 routed slots respectively. These tiny acceptance cohorts
are not leaderboard aggregates, but they provide visible-answer paired evidence:
the baseline passed 7/8 items, the masked model passed 1/8, and the comparisons
recorded six regressions. There is no basis here for a broad quality-retention
claim.

One standalone Anthropic `claude-sonnet-5` diagnostic returned judge `fc09361a…`
and request hash `533adbbb…`, score 1.0/pass, 341 input and 68 output tokens, and
3,219.16 ms latency. At the configured $2/M input and $10/M output prices, its
equivalent and incurred cost were both $0.001362; the server-computed upper bound
was $0.01115. Repeating the identical request returned the same ID/hash from cache
with zero incurred cost and zero remaining bound. Because the standalone endpoint
has no `max_cost_usd`, the driver—not the server—guarded this call at $0.025 with
one repetition, 128 maximum output tokens, and a bounded request body.

Cancellation job `000bd930…` stopped a 20-item LiveBench run after 1/20 progress,
reached `cancelled`, and released the active-job slot. Eight durable Daytona
sandbox-session rows were all `deleted`, no cleanup error remained, and a direct
provider listing found zero controller-owned sandboxes.

Acceptance Pod `7te5po3smp0227` stopped and reached `EXITED` at approximately
2026-08-11 19:23 UTC, ending its GPU-compute billing. All eight Pods in the account
were `EXITED`, and the account had no Runpod endpoints. Permanent deletion was not
performed because it would irreversibly erase the Pod's fresh 150 GB persistent
volume; that destructive action awaits explicit user approval and storage may
continue billing meanwhile. Existing network volumes `w7jl3o07i1`, `hhsfk4sdrc`,
and `gbichawf78` were unrelated to this acceptance run and were retained untouched.
An earlier stalled fresh attempt Pod, `o204fumgnxb1dj`, was permanently deleted;
its Pod-volume data is unrecoverable.

Template `w2wkhbveg2` was left on the proven immutable
`sha-0c8c8ffcbcd72f1cf9b664a1d4cf7813f701c345` image tag rather than the slow OCI
index coordinate. On stopped original Pod `kw3whlid6esyhs`, the temporary
acceptance auth value was restored to its Runpod secret reference while its other
23 environment entries were preserved.

As of the final audit, Runpod's lagging billing API reported for
`7te5po3smp0227` $0.0493180654 at 17:00–18:00 UTC and $0.9599268073 at
18:00–19:00 UTC: $1.0092448727 posted so far. It had not yet emitted the
19:00–19:23 bucket, so this is a provisional Pod-specific ledger, not a finalized
total. A deliberately conservative advertised-rate ceiling for that Pod is
approximately 1 hour 23 minutes × $1.89/hour, or about $2.62, plus minor disk
charges—still well below the $50 budget. The ledger exposed no separate row for
deleted attempt `o204fumgnxb1dj`; neither the posted subtotal nor the ceiling is a
finalized all-attempt acceptance total. The final ledger should be checked after
Runpod billing catches up; no exact tail estimate is invented here. Daytona cleanup
was reconfirmed after its last work—eight durable sessions deleted and zero owned
provider sandboxes—and the tuned benchmark rerun used no Daytona. A separate
Daytona monetary total was not exposed by the cleanup evidence and is likewise not
invented.

## Historical RC3 acceptance — not the current release

On 2026-08-11, application commit
`6d999f6288592f9414df1745a6371916cacc9fba`, fork commit
`729d4f23ae9cc27f797b8c13a9276565976637ed`, and image digest
`sha256:bb5e4addc10db7d2f8166d8eef05be3c032ee83533495c7d4f8d3b2dc8026362`
ran on one RTX PRO 6000 Blackwell through the authenticated Runpod proxy.

The model loaded, a hard reload rediscovered the same job and PID, and a two-item
arithmetic baseline → routing → 64-experts-per-layer profile → masked rerun → saved
comparison passed with both cohorts scoring `1.0`. An unmasked Daytona
`fix-subtract` trial passed with reward `1.0`, eight routed inference turns, seven
real commands, a trusted-verifier pass, and confirmed sandbox deletion. Repeating
that task under the arithmetic-derived 25% profile produced reward `0`, no command,
`token_limit`, verifier failure, and confirmed deletion. That is a useful historical
negative-transfer observation, not a Daytona failure and not a broad coding-quality
claim.

All acceptance Pods from that historical run were stopped and no acceptance-owned
Daytona sandbox remained. The old run did not test the current RC1 command centers,
multi-source profiles, benchmark breadth, frontier-judge accounting, clean-room
verifier isolation, or paired agent comparison.

## Important limitations

- Mask changes require a model restart; they are not hot-swappable.
- Ineligible experts stay resident. Eligibility masking does not shrink the
  checkpoint, VRAM, host memory, or disk footprint.
- Routed telemetry adds memory, transport, and latency overhead. Quality analysis
  should compare like with like; clean throughput claims need capture disabled.
- The application is a single-user, one-model, one-GPU research appliance, not a
  multi-tenant service.
- Agent results depend on the exact task, scaffold, sandbox, model session, profile,
  seed, generation, evaluator, and budget contract. A three-task canary is not a
  leaderboard score.
- Terminal-Bench engineering 15, Aider balanced 12, and FeatureBench Fast remain
  fail-closed manifests until their real assets and harnesses pass admission.
- Selective quantization, physical checkpoint materialization, expert offload, and
  multi-node serving are future interventions; the persistence model leaves room
  for them without presenting them as implemented.

## Primary references

- [Qwen3.6-35B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [Runpod Pod templates](https://docs.runpod.io/pods/templates/overview)
- [Runpod exposed ports](https://docs.runpod.io/pods/configuration/expose-ports)
- [Runpod network volumes](https://docs.runpod.io/storage/network-volumes)
- [Daytona API keys](https://www.daytona.io/docs/api-keys/)
- [Daytona async Python SDK](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/)
- [Harbor](https://github.com/harbor-framework/harbor) and
  [Harbor tasks](https://www.harborframework.com/docs/tasks)
- [Aider benchmark harness](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)
- [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
- [FeatureBench](https://github.com/LiberCoders/FeatureBench)
- [vLLM benchmarking](https://docs.vllm.ai/en/stable/benchmarking/cli/)
