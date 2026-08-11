# MoE Tools Test Suite

An interactive research workbench for measuring expert routing, creating vLLM
expert-selection profiles, and comparing baseline and masked MoE behavior in both
one-request benchmarks and multi-turn coding trials.

The implementation has two tested vertical slices. The first uses a deterministic
mock of `Qwen/Qwen3.6-35B-A3B-FP8`, including its 40 routed layers, 256 experts per
layer, top-8 routing, and the fork's paired NumPy telemetry contract. The second is a
native, model-centered coding controller with remote-sandbox and trajectory
boundaries. Both slices use persisted background jobs, immutable run contracts,
per-inference routing artifacts, named profile revisions, and restart-safe research
records. Browser polling survives transient proxy failures and reloads by
rediscovering the one durable server job; model-load submissions are persisted with
their planned session before execution, and identical retries rejoin the same work.

On 2026-08-11, the immutable RC3 candidate completed bounded live acceptance on an
RTX PRO 6000 Blackwell and Daytona. The model/profile canary passed end to end, and
the ordinary Daytona coding trial passed with confirmed sandbox deletion. A
two-item arithmetic-derived 25% profile then caused a useful negative-transfer
result on the same coding task; that is a masking-quality result, not a provider
failure. Exact provenance, outcomes, remaining gaps, and the repeat ladder are
recorded below.

The benchmark workspace now has an adapter boundary for dataset loading, prompting,
scoring, and generation defaults. It includes the deterministic fixture, the pinned
[official GSM8K test split](https://github.com/openai/grade-school-math)
(downloaded once and verified by size and SHA-256), and
validated custom JSONL imports with exact, contains, regex, or ungraded scoring.
Benchmark cohorts are immutable and fingerprint the dataset revision, selected item
order, prompt/scorer versions, and generation settings. Runs support cooperative
cancellation, per-item failures, provenance-rich restart recovery, and JSON or CSV
exports. Generated code is never executed inside the application/model container.

See [ROADMAP.md](ROADMAP.md) for the complete product and agentic-coding plan.

## Native agentic coding slice

The implemented controller is `bash-json-v1`, revision `1`: a deliberately small
loop in which the model returns one JSON action per turn, either a sandbox shell
command or a final summary. Its default limits are eight turns, eight commands,
32,768 tokens, and 300 seconds. Qwen thinking is disabled by default so the bounded
completion budget is spent on the machine-readable action; the generation contract
records that choice. It is native application code, not an embedded Harbor,
Mini-SWE-Agent, or Aider process.

Every model turn goes through the same instrumented local gateway. The gateway calls
the current model session, stores an inference record, and links that call's routed
expert IDs and weights to the corresponding trajectory step. The controller persists
run, trial, sandbox, verifier, trajectory, inference, routing, and artifact records;
the workbench shows trial progress, the complete trajectory, verifier output,
artifacts, and a trial routing heatmap.

For extending multi-turn chats, the gateway uses returned prompt/generated token IDs
and the local tokenizer to calculate an exact common prefix. The fork omits only that
already-recorded prompt routing on the next turn. Prefix state is isolated per trial
and falls back to capturing the full prompt whenever token metadata or template
alignment is ambiguous, so earlier turns are neither double-counted nor silently
shared across trials.

Trajectories are exported as `ATIF-v1.7`, following Harbor's
[Agent Trajectory Interchange Format RFC](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
and [trajectory documentation](https://www.harborframework.com/docs/agents/trajectory-format).
The bundled `smoke-python-v1` pack is immutable and content-addressed. It contains
three dependency-free Python repositories, each marked oracle-passing and no-op-
failing:

- `fix-subtract` — repair a small arithmetic bug;
- `implement-slugify` — implement a standard-library text utility;
- `repair-json-cli` — repair a compact JSON command-line contract.

Each task contract asks for one CPU, 1 GiB RAM, 2 GiB task disk, a 180-second
timeout, and no network egress; the Daytona adapter raises disk to its 3 GiB
minimum. The native pack is designed to map cleanly to Harbor's
[task model](https://www.harborframework.com/docs/tasks), but a full Harbor runtime,
Harbor task-pack import/validation, and Aider or Mini-SWE-Agent scaffolds remain
deferred. The compatibility review used Harbor's official
[v0.20.0 release](https://github.com/harbor-framework/harbor/releases/tag/v0.20.0)
as its pinned reference; the application does not currently install or invoke that
release. Harbor's [agent integration boundary](https://www.harborframework.com/docs/agents),
[Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent), and the
[Aider benchmark harness](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)
are the upstream references, not components silently substituted into current runs.

### Sandbox providers

| Provider | Intended use | Execution behavior |
| --- | --- | --- |
| `fake` | GPU-free contract, persistence, and UI tests | In-memory and deterministic. It returns only explicitly scripted command responses; an unscripted command exits `127`. It **never invokes a local subprocess**. |
| `daytona` | Real isolated coding trials | Remote create, upload, command execution, file read, and confirmed delete through the exact [`daytona==0.192.0`](https://pypi.org/project/daytona/0.192.0/) Python SDK pin. |

Provider listing is local and side-effect free. Daytona becomes runnable only after
`POST /api/sandbox-providers/daytona/preflight` succeeds; that preflight makes a
bounded, read-only list call and creates no sandbox. The public response separates
`configured`, `reachable`, and `authenticated`, exposes only the API host/region and
a sanitized bounded error, and never returns a credential. Configure the app with:

```text
MOE_TOOLS_DAYTONA_API_KEY=...
MOE_TOOLS_DAYTONA_API_URL=https://app.daytona.io/api  # optional
MOE_TOOLS_DAYTONA_TARGET=us                           # optional: us or eu
```

Daytona documents both [API-key creation](https://www.daytona.io/docs/api-keys/)
and the [async Python client lifecycle](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/).
For each real trial the controller creates a private ephemeral sandbox with managed
run/trial ownership labels, uploads only task files, sends an empty environment and
empty secrets map, executes commands in the task workspace, and verifies ownership
before reads, commands, and deletion. The task's deny/allowlist/public egress policy
maps to Daytona's [network controls](https://www.daytona.io/docs/en/network-limits/).
Cleanup waits until the sandbox is destroyed or absent. Daytona credentials and the
loopback model endpoint remain controller-side.

Immediately before scoring, the controller re-uploads the pack's canonical verifier
modules and package markers, then runs the verifier with Python isolated mode. On app
startup it also retries deletion of durable interrupted sessions: Daytona must return
the exact persisted controller/run/trial ownership labels before deletion proceeds.
An ownership mismatch is retained as a visible terminal cleanup error, while a
retryable provider failure remains pending and blocks another Daytona run.

### Trial-to-profile flow

A successful trial's routing summary can already drive a profile through the API:

1. Request `POST /api/trials/{trial_id}/profile-proposal` with
   `keep_per_layer=64&metric=routing_mass`.
2. Save the returned profile with `POST /api/profiles`, `source="agentic"`, and the
   originating `source_trial_id`.
3. Load that saved profile through the existing model-session flow. Profile changes
   restart vLLM and produce a new model session.
4. Run the same pack revision, task IDs, agent revision, provider, seed, generation
   settings, attempts, and budgets against that new session.

The current proposal is derived from **one trial**, not an aggregate of all trials
in a run, and the agentic workbench does not yet present this as a one-click paired
comparison. Most importantly, the resulting mask changes expert eligibility before
top-k routing. Ineligible expert weights remain loaded, so this is behavioral
masking—not checkpoint shrinking, expert offload, or demonstrated VRAM savings.

## Custom benchmark JSONL

Use **Import JSONL** in the benchmark card. Each non-empty line must be one object:

```json
{"id":"sum-1","prompt":"Return only the result of 2 + 2.","expected":"4","category":"smoke","scoring":"exact"}
{"id":"workflow-1","prompt":"Inspect the repository and propose a fix.","category":"agentic","scoring":"ungraded","metadata":{"harness":"daytona"}}
```

`id` and `category` are optional; `prompt` is required. Graded items also require
`expected`. Supported scorers are `exact`, `contains`, `regex`, and `ungraded`.
Imports are content-addressed, validated before being stored, and limited to 2 MB
by default (`MOE_TOOLS_MAX_CUSTOM_DATASET_BYTES`).

## Local development

Prerequisites: Python 3.12, `uv`, Node.js 24, and npm.

```bash
uv sync
npm --prefix frontend install
```

Run the backend:

```bash
uv run uvicorn --app-dir backend/src moe_tools_suite.main:app \
  --host 127.0.0.1 --port 8080 --reload
```

Run the frontend in a second terminal:

```bash
npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173`. Vite proxies API requests to port 8080.

Authentication is disabled by default for local development. To exercise the
single-user login locally, copy `.env.example` to `.env`, replace `REPLACE_ME`
with the output of `openssl rand -base64 32`, and leave
`MOE_TOOLS_COOKIE_SECURE=0` while using plain HTTP on localhost.

## Verification

```bash
uv run pytest
uv run ruff check backend
npm --prefix frontend test
npm --prefix frontend run build
```

The public-API release canary exercises the complete baseline-to-mask loop against a
running appliance. It uses two explicit fixture items by default and never touches
Daytona unless `--daytona` is supplied:

```bash
MOE_TOOLS_CANARY_BASE_URL=http://127.0.0.1:8080 \
uv run python scripts/acceptance_canary.py
```

For a protected deployment, set `MOE_TOOLS_CANARY_TOKEN` in the environment. The
opt-in `--daytona` extension performs a read-only provider preflight followed by one
bounded `fix-subtract` trial, and requires both routed inference telemetry and a
confirmed `deleted` sandbox state before it passes.

## Runpod image

The Runpod Dockerfile consumes the sibling custom vLLM checkout as a named
BuildKit context. Build from this repository root on any machine with Buildx:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-context vllm_source=../vllm-moe-tools \
  --file docker/Dockerfile.runpod \
  --build-arg VLLM_SOURCE_REF="$(git -C ../vllm-moe-tools rev-parse HEAD)" \
  --tag YOUR_REGISTRY/moe-tools-test-suite:0.2.0-rc.3 \
  .
```

### GitHub Container Registry

The repository workflow at `.github/workflows/publish-container.yml` builds the
same image natively for `linux/amd64` and publishes it alongside this repository
as `ghcr.io/sampazdan/vllm-moe-tools-suite`. It checks out the vLLM fork at the
pinned commit recorded in the workflow and uses GitHub's repository-scoped token;
no long-lived registry password is required.

Publish either by running **Publish Runpod container** from the GitHub Actions tab
with an explicit version such as `0.2.0-rc.3`, or by pushing a semantic tag:

```bash
git tag v0.2.0-rc.3
git push origin v0.2.0-rc.3
```

The versioned image and a commit-addressed image are both produced:

```text
ghcr.io/sampazdan/vllm-moe-tools-suite:0.2.0-rc.3
ghcr.io/sampazdan/vllm-moe-tools-suite:sha-<full-application-commit>
```

The OCI source label links the package back to this repository. Keep the versioned
tag in the Runpod template; the SHA tag is useful when an experiment must pin the
exact application build. If the package is private, add a GitHub classic token with
`read:packages` to Runpod's Container Registry credentials. Public packages require
no pull credential.

The image retains the official Runpod `/start.sh`, starts the app on port 8080,
keeps managed vLLM on loopback port 8000, and writes persistent state under
`/workspace/moe-tools`. The image defaults to mock mode. In `vllm` mode, a model
job owns the fork's `runpod-serve` process, captures its log, waits for the model
registry endpoint, and restarts it with a persisted profile when requested.

### Repeatable Runpod appliance checkpoint

This checkpoint verifies the container, proxy, login, and storage without paying to
load the model. It can run on an inexpensive Pod; reserve the RTX PRO 6000 for the
managed-vLLM checkpoint.

1. Push the image under a unique version tag. For strict repeatability, deploy the
   workflow's `sha-<full-app-commit>` tag or the published image digest.
2. In the Runpod console, create a secret named `moe_tools_auth_token`. Generate
   its value with `openssl rand -base64 32`. The template references it as
   `{{ RUNPOD_SECRET_moe_tools_auth_token }}` so the token is not stored in the
   template. See Runpod's [secrets guide](https://docs.runpod.io/pods/templates/secrets).
3. Create the template from `deploy/runpod-template.example.json` in the console,
   or install `jq` and run:

   ```bash
   export RUNPOD_API_KEY="..."
   export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.2.0-rc.3"
   ./scripts/create_runpod_template.sh
   ```

   The script uses Runpod's current
   [template REST API](https://docs.runpod.io/api-reference/templates/POST/templates)
   and leaves the image's entrypoint and start command intact.
4. Deploy from the new private template. For expendable smoke tests, its 150 GB Pod
   volume is sufficient. Before downloading model weights, create a network volume
   in the desired data center and attach it during deployment; network volumes
   outlive Pod termination and constrain the available GPU location. See Runpod's
   [network-volume guide](https://docs.runpod.io/storage/network-volumes).
5. Open `https://POD_ID-8080.proxy.runpod.net`, sign in, run the fixture, restart the
   Pod, and confirm that the run remains visible. Port 8080 uses Runpod's HTTPS proxy;
   the app returns job IDs immediately because the proxy has a 100-second connection
   ceiling. See Runpod's [port guide](https://docs.runpod.io/pods/configuration/expose-ports).

The smoke test passes when `/readyz` returns `200`, invalid tokens are rejected, a
valid token opens the workbench on phone and desktop, a fixture job progresses to
completion, and its result survives a Pod restart. The startup hook fails closed if
the configured secret is missing, unresolved, or shorter than 24 characters.

### Repeatable RTX PRO 6000 checkpoint

After the mock appliance smoke passes, create a second template from the same
versioned image with the managed runtime enabled:

```bash
export RUNPOD_API_KEY="..."
export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.2.0-rc.3"
export MOE_TOOLS_TEMPLATE_MODE="vllm"
export MOE_TOOLS_TEMPLATE_NAME="moe-tools-test-suite-a3b"
./scripts/create_runpod_template.sh
```

Deploy it on one RTX PRO 6000 Blackwell Server Edition with the network volume
attached at `/workspace`. Do the first GPU acceptance run in deliberately small
steps:

1. Click **Load model** and wait for the persisted model job to complete. The UI
   shows the live tail of `/workspace/moe-tools/logs/model-sessions/<session>.log`.
2. Run one or two arithmetic items and confirm the response includes a 40 × 256
   routing summary with nonzero routed slots.
3. Run the full 12-item baseline, create a 64-expert-per-layer profile, and click
   **Load this profile**. A new model session and process must be created.
4. Run the masked cohort. The app reuses the exact baseline item IDs and exposes
   base/masked heatmaps plus a paired item-level score comparison.
5. Restart the Pod once and use the archive to reopen the persisted paired
   comparison. Confirm runs, jobs, immutable profile lineage, model-session
   provenance, and compressed routing artifacts remain available. Then stop the
   Pod; do not leave the GPU running merely to preserve the network volume.

This target-GPU regression checkpoint validates image compatibility, the fork's real
telemetry response, process restart semantics, and profile application. It is not by
itself a meaningful pruning-quality experiment; use held-out benchmarks and repeated
runs for quality claims.

### Agentic Runpod template

`deploy/runpod-agentic-template.example.json` combines managed-vLLM mode with the
Daytona provider. It is private, exposes only app HTTP and maintenance SSH, stores
application state at `/workspace/moe-tools`, and references the app and Daytona
tokens as [Runpod secrets](https://docs.runpod.io/pods/templates/secrets). Create it
from the same versioned image used for the non-agentic checkpoint (or its immutable
SHA tag/digest):

```bash
export RUNPOD_API_KEY="..."
export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.2.0-rc.3"
export MOE_TOOLS_TEMPLATE_MODE="agentic"
export MOE_TOOLS_TEMPLATE_NAME="moe-tools-agentic-a3b"
./scripts/create_runpod_template.sh
```

Before deployment, create `moe_tools_auth_token` and `daytona_api_key` in Runpod.
The template uses one RTX PRO 6000 Pod for the model and Daytona for disposable task
sandboxes; it does not attempt Docker-in-Docker. Runpod explicitly documents that
[Docker and Compose are unavailable inside a Pod](https://docs.runpod.io/tutorials/pods/build-docker-images).
Attach a network volume only if model cache and research records should outlive the
Pod. Runpod's [template](https://docs.runpod.io/pods/templates/overview),
[port](https://docs.runpod.io/pods/configuration/expose-ports), and
[network-volume](https://docs.runpod.io/storage/network-volumes) guides describe
those lifecycle boundaries.

### 2026-08-11 live acceptance result

The accepted application commit is
`6d999f6288592f9414df1745a6371916cacc9fba`; the fork commit is
`729d4f23ae9cc27f797b8c13a9276565976637ed`; and the published image digest is
`sha256:bb5e4addc10db7d2f8166d8eef05be3c032ee83533495c7d4f8d3b2dc8026362`.
The private template `w2wkhbveg2` is named `moe-tools-agentic-a3b-rc3`, pins the
immutable image, and references the application, Hugging Face, and Daytona
credentials through Runpod secrets.

The secure Runpod appliance ran on one RTX PRO 6000 Blackwell with 94.97 GiB VRAM.
`Qwen/Qwen3.6-35B-A3B-FP8` loaded with 34.77 GiB consumed and 48.84 GiB allocated
to KV cache. Authentication through the public proxy passed. A hard browser reload
during model loading rediscovered the same durable job and PID and did not submit a
duplicate load. The core canary completed baseline generation, routing capture, a
64-expert-per-layer arithmetic profile (25% of experts), profile-ID-only reload, the
same two-item masked run, and persisted comparison. Both arithmetic cohorts scored
`1.0`.

The live Daytona results separated provider correctness from profile quality:

- The unmasked `fix-subtract` trial passed with reward `1.0`, eight turns, eight
  instrumented inference/routing calls, seven real sandbox commands, the correct
  `+1` to `-1` patch, a passing trusted verifier, `agent_finished` termination, and
  confirmed sandbox deletion.
- The same task under the arithmetic-derived 25% profile finished with reward `0`,
  three routed turns, zero commands, repetitive shell text, `token_limit`
  termination, a failed trusted verifier, and confirmed sandbox deletion. This is a
  useful negative-transfer observation, not a Daytona lifecycle failure.

The FUSE checkpoint load reported 34.89 GiB and took roughly three minutes. It also
warned that automatic prefetch was disabled; a repeat should evaluate
`--safetensors-load-strategy=prefetch`. All acceptance Pods are `EXITED`, and no
acceptance-owned paid Daytona sandbox was retained. The scoped GPU estimate is
approximately `$2.04`:
about `$1.056` in reported Pod billing plus a `$0.98` fallback runtime estimate; the
current billing bucket may post later. Retained Runpod storage continues to bill.

This acceptance proves the bounded proxy, durable-job, model, routing/profile,
Daytona execution, trusted-verifier, and normal cleanup paths. It does **not** yet
verify live cancellation or crash/orphan failure injection, or complete masked and
unmasked cohorts across all three bundled tasks. Expert masking is behavioral
eligibility control; it does not shrink the loaded weights or VRAM footprint.

For a repeat or release regression, run this concise ladder serially:

1. Pin the application commit, fork commit, image digest, template, model revision,
   and budget; authenticate through the public proxy and repeat the hard-reload
   same-job/same-PID check during load.
2. Run the read-only Daytona preflight, then the unmasked `fix-subtract` canary;
   require its verifier/artifacts and confirmed sandbox deletion.
3. Repeat the two-item arithmetic baseline → routing → 64-per-layer profile →
   profile-ID-only reload → identical masked cohort and comparison.
4. Run all three bundled tasks unmasked and masked with identical contracts. Treat
   quality deltas separately from provider/lifecycle failures.
5. Inject cancellation and crash-after-allocation cases and require exact-label
   recovery with no Daytona orphan.
6. Confirm zero owned sandboxes, stop every paid Pod, record final billing, and keep
   persistent storage only intentionally.
