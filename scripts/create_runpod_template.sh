#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
: "${MOE_TOOLS_IMAGE:?Set MOE_TOOLS_IMAGE to a versioned container tag or digest}"
dry_run="${MOE_TOOLS_TEMPLATE_DRY_RUN:-0}"

if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
    echo "MOE_TOOLS_TEMPLATE_DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ "${dry_run}" == "0" ]]; then
    : "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY to a Runpod API key}"
fi
if [[ "${MOE_TOOLS_IMAGE}" =~ [[:space:]] ]]; then
    echo "MOE_TOOLS_IMAGE cannot contain whitespace." >&2
    exit 2
fi
image_leaf="${MOE_TOOLS_IMAGE##*/}"
if [[ "${MOE_TOOLS_IMAGE}" == *@sha256:* ]]; then
    if [[ ! "${MOE_TOOLS_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "MOE_TOOLS_IMAGE contains an invalid sha256 digest." >&2
        exit 2
    fi
elif [[ "${image_leaf}" != *:* || "${image_leaf##*:}" == "latest" ]]; then
    echo "MOE_TOOLS_IMAGE must use a non-latest version tag or immutable digest." >&2
    exit 2
fi
command -v jq >/dev/null || {
    echo "jq is required to render a Runpod template." >&2
    exit 2
}

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

if [[ "${dry_run}" == "1" ]]; then
    jq . "${rendered_template}"
    exit 0
fi

curl --fail --silent --show-error \
    --request POST \
    --url https://rest.runpod.io/v1/templates \
    --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --header "Content-Type: application/json" \
    --data-binary "@${rendered_template}"
echo
