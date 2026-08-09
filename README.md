# MoE Tools Test Suite

An interactive research workbench for measuring expert routing, creating vLLM
expert-selection profiles, and comparing baseline and masked MoE behavior.

The current implementation is the GPU-free first vertical slice. It uses a
deterministic mock of `Qwen/Qwen3.6-35B-A3B-FP8`, including its 40 routed layers,
256 experts per layer, top-8 routing, and the fork's paired NumPy telemetry
contract. The UI can load the mock model, choose fixture items, run them, inspect
expert engagement, and create a fork-compatible profile.

See [ROADMAP.md](ROADMAP.md) for the complete product and agentic-coding plan.

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

The image retains the official Runpod `/start.sh`, starts the app on port 8080,
keeps vLLM on loopback port 8000, and writes persistent state under
`/workspace/moe-tools`. This first image defaults to mock mode; managed model
lifecycle is the next implementation milestone.
