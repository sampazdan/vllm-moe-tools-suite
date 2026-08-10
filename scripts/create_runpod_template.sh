#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY to a Runpod API key}"
: "${MOE_TOOLS_IMAGE:?Set MOE_TOOLS_IMAGE to a versioned container tag or digest}"

template_mode="${MOE_TOOLS_TEMPLATE_MODE:-mock}"
if [[ "${template_mode}" != "mock" && "${template_mode}" != "vllm" && "${template_mode}" != "agentic" ]]; then
    echo "MOE_TOOLS_TEMPLATE_MODE must be mock, vllm, or agentic." >&2
    exit 2
fi
if [[ "${template_mode}" == "agentic" ]]; then
    template_source="${repository_root}/deploy/runpod-agentic-template.example.json"
    runtime_mode="vllm"
else
    template_source="${repository_root}/deploy/runpod-template.example.json"
    runtime_mode="${template_mode}"
fi
template_name="${MOE_TOOLS_TEMPLATE_NAME:-moe-tools-test-suite-${template_mode}}"
rendered_template="$(mktemp)"
trap 'rm -f "${rendered_template}"' EXIT

jq \
    --arg image "${MOE_TOOLS_IMAGE}" \
    --arg mode "${runtime_mode}" \
    --arg name "${template_name}" \
    '.imageName = $image | .name = $name | .env.MOE_TOOLS_MODE = $mode' \
    "${template_source}" >"${rendered_template}"

curl --fail --silent --show-error \
    --request POST \
    --url https://rest.runpod.io/v1/templates \
    --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --header "Content-Type: application/json" \
    --data-binary "@${rendered_template}"
echo
