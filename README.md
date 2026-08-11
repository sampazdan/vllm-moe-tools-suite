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

`0.3.0-rc.1` is the current finish-line candidate on this branch. Its complete
application flow is implemented locally, but this exact application commit, fork
commit, and container image have **not yet completed live acceptance**. The final
SHAs and image digest will be recorded only after publishing and exercising the
immutable build.

An older internal candidate called **RC3** completed a bounded Runpod/Daytona test
on 2026-08-11. Those historical results are retained near the end of this README
because they establish that the basic appliance, real-model telemetry, profile
reload, and Daytona lifecycle worked. They are not acceptance evidence for
`0.3.0-rc.1` or its newer evaluator, profile, command-center, and clean-room paths.

## What is in the application

The warm, editorial interface is organized as full-screen research command centers
that also collapse cleanly onto a phone:

- **Home** owns model load/stop state, topology, runtime facts, startup logs, a
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
  performance.
- **Archive** persists benchmark runs, agent runs, profiles, and comparisons.
  Benchmark summaries and item histories are paginated; JSON/CSV exports stream so
  large cohorts do not need a second full in-memory response.

Browser polling survives transient Runpod proxy failures, reconnects, and hard
reloads. Model loads, dataset preparation, benchmark runs, and agent runs are
durable server jobs. An identical model-load retry adopts the existing job; an
incompatible submission receives a typed conflict with the active work. Cooperative
cancellation uses a visible `cancelling` state and does not release the single-work
mutex until the worker has acknowledged termination. Startup reconciliation repairs
interrupted records and retries owned process or sandbox cleanup.

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

Cost caps are fail-closed. Explicit input/output prices are required for capped
paid judging. Before every uncached provider request, the evaluator reserves a
conservative full-request upper bound, including all repetitions. Cached decisions
incur zero new spend. If a provider fails after a request may have been accepted,
the run records cost uncertainty and debits conservatively instead of pretending
the call was free.

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

Profile Studio can combine up to many benchmark runs and agentic trials. Its
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
and requires the exact fork SHA as a manual input. After publishing, deploy the
returned immutable `@sha256:...` coordinate. The final RC1 app SHA, fork SHA, and
image digest are intentionally not written here before they exist.

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

Create the template with a versioned tag for a first build or, preferably, its
published digest:

```bash
export RUNPOD_API_KEY="..."
export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.3.0-rc.1"
export MOE_TOOLS_TEMPLATE_MODE="agentic"
export MOE_TOOLS_TEMPLATE_NAME="moe-tools-agentic-a3b"
./scripts/create_runpod_template.sh
```

The script rejects `latest`, malformed digests, and invalid modes. The agentic
template uses one RTX PRO 6000 Blackwell Server Edition, port 8080, a 50 GB container
disk, and `/workspace` for the application and model cache. Attach a network volume
at `/workspace` if records and weights must outlive the Pod. A volume is tied to one
Runpod data center and continues billing after the Pod stops.

### Planned RC1 live-acceptance ladder

The current candidate remains pending until this ladder is recorded against one
immutable image:

1. local mock checks, production frontend build, deployment-asset validation, and
   secret scan;
2. boot the published image, authenticate through the public proxy on desktop and
   phone, and verify persistence/reconnect;
3. load `Qwen/Qwen3.6-35B-A3B-FP8`, hard-reload during startup, and require adoption
   of the same durable job and process;
4. run bounded GSM8K, MMLU-Pro, IFEval, and LiveBench samples; inspect problems,
   contracts, progress, routing, and performance;
5. build a non-uniform custom multi-source profile, reload it, rerun an identical
   cohort, and save the paired benchmark comparison;
6. run an unmasked Daytona agent cohort, require trusted-verifier outcomes and zero
   retained sandboxes, then create a custom profile from its selected trials;
7. reload with that profile, repeat the identical agent contract, and save the
   paired agent comparison;
8. make one tightly capped Anthropic judge request and verify usage, cost, cache,
   and failure accounting;
9. exercise cancellation, proxy reconnect, application restart, and owned-orphan
   cleanup; and
10. confirm no owned Daytona sandbox remains, stop every paid Pod, and retain only
    storage chosen intentionally.

The public API canary covers the bounded baseline-to-mask path. Daytona and
Anthropic are opt-in because they create paid work:

```bash
MOE_TOOLS_CANARY_BASE_URL=https://POD_ID-8080.proxy.runpod.net \
MOE_TOOLS_CANARY_TOKEN="..." \
uv run python scripts/acceptance_canary.py --daytona --anthropic-judge
```

### RC1 acceptance record — pending

This bounded section is intentionally incomplete until the ladder above finishes:

- application commit: pending;
- vLLM fork commit: pending;
- published image digest and Runpod template revision: pending;
- local release-gate results: pending final integrated run;
- live model/benchmark/profile results: pending;
- live Daytona clean-room and agent-comparison results: pending;
- live frontier-judge result and cost: pending; and
- final Runpod/Daytona teardown confirmation: pending.

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
