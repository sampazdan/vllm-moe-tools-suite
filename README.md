# MoE Tools Test Suite

An interactive research workbench for measuring expert routing, creating vLLM
expert-selection profiles, and comparing baseline and masked MoE behavior.

The current implementation is an authenticated, GPU-free vertical slice. It uses
a deterministic mock of `Qwen/Qwen3.6-35B-A3B-FP8`, including its 40 routed
layers, 256 experts per layer, top-8 routing, and the fork's paired NumPy telemetry
contract. The UI can load the mock model, choose fixture items, run them, inspect
expert engagement, and create a fork-compatible profile. Model and benchmark work
runs as persisted background jobs, and completed runs plus compressed routing
artifacts survive application restarts. The research archive also persists named,
fingerprinted profile revisions and strict paired comparisons. Profiles can be
imported/exported in the fork's version 1 JSON format and revised with a per-layer
manual expert editor.

The benchmark workspace now has an adapter boundary for dataset loading, prompting,
scoring, and generation defaults. It includes the deterministic fixture, the pinned
[official GSM8K test split](https://github.com/openai/grade-school-math)
(downloaded once and verified by size and SHA-256), and
validated custom JSONL imports with exact, contains, regex, or ungraded scoring.
Benchmark cohorts are immutable and fingerprint the dataset revision, selected item
order, prompt/scorer versions, and generation settings. Runs support cooperative
cancellation, per-item failures, provenance-rich restart recovery, and JSON or CSV
exports. This one-request adapter layer is also the storage and reporting foundation
for the later Harbor/Daytona agentic-coding runner; it does not execute generated
code inside the application container.

See [ROADMAP.md](ROADMAP.md) for the complete product and agentic-coding plan.

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
npm --prefix frontend run build
```

## Runpod image

The Runpod Dockerfile consumes the sibling custom vLLM checkout as a named
BuildKit context. Build from this repository root on any machine with Buildx:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-context vllm_source=../vllm-moe-tools \
  --file docker/Dockerfile.runpod \
  --build-arg VLLM_SOURCE_REF="$(git -C ../vllm-moe-tools rev-parse HEAD)" \
  --tag YOUR_REGISTRY/moe-tools-test-suite:0.1.0 \
  .
```

### GitHub Container Registry

The repository workflow at `.github/workflows/publish-container.yml` builds the
same image natively for `linux/amd64` and publishes it alongside this repository
as `ghcr.io/sampazdan/vllm-moe-tools-suite`. It checks out the vLLM fork at the
pinned commit recorded in the workflow and uses GitHub's repository-scoped token;
no long-lived registry password is required.

Publish either by running **Publish Runpod container** from the GitHub Actions tab
with an explicit version such as `0.1.0`, or by pushing a semantic tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The versioned image and a commit-addressed image are both produced:

```text
ghcr.io/sampazdan/vllm-moe-tools-suite:0.1.0
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

### First Runpod appliance checkpoint

This checkpoint deliberately verifies the container, proxy, login, and storage
without paying to load the model. It can run on an inexpensive Pod; reserve the RTX
PRO 6000 for the managed-vLLM checkpoint.

1. Push the image under an immutable tag.
2. In the Runpod console, create a secret named `moe_tools_auth_token`. Generate
   its value with `openssl rand -base64 32`. The template references it as
   `{{ RUNPOD_SECRET_moe_tools_auth_token }}` so the token is not stored in the
   template. See Runpod's [secrets guide](https://docs.runpod.io/pods/templates/secrets).
3. Create the template from `deploy/runpod-template.example.json` in the console,
   or install `jq` and run:

   ```bash
   export RUNPOD_API_KEY="..."
   export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.1.0"
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

### First RTX PRO 6000 checkpoint

After the mock appliance smoke passes, create a second template from the same
immutable image with the managed runtime enabled:

```bash
export RUNPOD_API_KEY="..."
export MOE_TOOLS_IMAGE="ghcr.io/sampazdan/vllm-moe-tools-suite:0.1.0"
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

This is the first checkpoint worth paying for the target GPU. It validates image
compatibility, the fork's real telemetry response, process restart semantics, and
profile application. It is not yet a meaningful pruning-quality experiment; use
held-out benchmarks and repeated runs only after this plumbing passes.
