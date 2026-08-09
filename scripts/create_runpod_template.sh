#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
template_source="${repository_root}/deploy/runpod-template.example.json"

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY to a Runpod API key}"
: "${MOE_TOOLS_IMAGE:?Set MOE_TOOLS_IMAGE to an immutable container tag}"

template_mode="${MOE_TOOLS_TEMPLATE_MODE:-mock}"
if [[ "${template_mode}" != "mock" && "${template_mode}" != "vllm" ]]; then
    echo "MOE_TOOLS_TEMPLATE_MODE must be mock or vllm." >&2
    exit 2
fi
template_name="${MOE_TOOLS_TEMPLATE_NAME:-moe-tools-test-suite-${template_mode}}"
rendered_template="$(mktemp)"
trap 'rm -f "${rendered_template}"' EXIT

jq \
    --arg image "${MOE_TOOLS_IMAGE}" \
    --arg mode "${template_mode}" \
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
