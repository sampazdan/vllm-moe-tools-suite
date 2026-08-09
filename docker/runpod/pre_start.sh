#!/usr/bin/env bash
set -euo pipefail

app_root="${MOE_TOOLS_APP_ROOT:-/opt/moe-tools-test-suite}"
data_root="${MOE_TOOLS_DATA_DIR:-/workspace/moe-tools}"
log_dir="${data_root}/logs"
pid_file="${data_root}/app.pid"

mkdir -p \
    "${data_root}" \
    "${log_dir}" \
    /workspace/cache/huggingface/hub \
    /workspace/cache/torchinductor \
    /workspace/cache/triton \
    /workspace/cache/vllm \
    /workspace/profiles \
    /workspace/results

ln -sfn /opt/vllm-src /workspace/vllm-src
ln -sfn "${app_root}" /workspace/moe-tools-test-suite

if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
    echo "MoE Tools Test Suite is already running."
    exit 0
fi

export MOE_TOOLS_DATA_DIR="${data_root}"
export MOE_TOOLS_FRONTEND_DIST="${app_root}/frontend/dist"

setsid /opt/vllm-venv/bin/python -m uvicorn \
    moe_tools_suite.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    >"${log_dir}/app.log" 2>&1 < /dev/null &
echo "$!" >"${pid_file}"

echo "MoE Tools Test Suite started on port 8080 (PID $(<"${pid_file}"))."
