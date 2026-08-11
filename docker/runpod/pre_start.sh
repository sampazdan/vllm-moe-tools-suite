#!/usr/bin/env bash
set -euo pipefail
umask 077

app_root="${MOE_TOOLS_APP_ROOT:-/opt/moe-tools-test-suite}"
data_root="${MOE_TOOLS_DATA_DIR:-/workspace/moe-tools}"
log_dir="${data_root}/logs"
pid_file="${data_root}/app.pid"

if [[ "${MOE_TOOLS_REQUIRE_AUTH:-1}" == "1" ]]; then
    auth_token="${MOE_TOOLS_AUTH_TOKEN:-}"
    if (( ${#auth_token} < 24 )) || [[ "${auth_token}" == *RUNPOD_SECRET_* ]]; then
        echo "MOE_TOOLS_AUTH_TOKEN must resolve to at least 24 characters." >&2
        exit 2
    fi
fi

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

if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        if curl --fail --silent http://127.0.0.1:8080/readyz >/dev/null; then
            echo "MoE Tools Test Suite is already running."
            exit 0
        fi

        existing_command=""
        if [[ -r "/proc/${existing_pid}/cmdline" ]]; then
            existing_command="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline")"
        fi
        if [[ "${existing_command}" == *"moe_tools_suite.main:app"* ]]; then
            echo "Stopping an unhealthy prior application process (PID ${existing_pid})."
            kill -TERM -- "-${existing_pid}" 2>/dev/null \
                || kill -TERM "${existing_pid}" 2>/dev/null \
                || true
            for _ in {1..50}; do
                if ! kill -0 "${existing_pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.1
            done
            if kill -0 "${existing_pid}" 2>/dev/null; then
                kill -KILL -- "-${existing_pid}" 2>/dev/null \
                    || kill -KILL "${existing_pid}" 2>/dev/null \
                    || true
            fi
        else
            echo "Ignoring a stale application PID file reused by another process."
        fi
    fi
    rm -f "${pid_file}"
fi

export MOE_TOOLS_DATA_DIR="${data_root}"
export MOE_TOOLS_FRONTEND_DIST="${app_root}/frontend/dist"

setsid /opt/vllm-venv/bin/python -m uvicorn \
    moe_tools_suite.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    >"${log_dir}/app.log" 2>&1 < /dev/null &
echo "$!" >"${pid_file}"

for _ in {1..30}; do
    if curl --fail --silent http://127.0.0.1:8080/readyz >/dev/null; then
        echo "MoE Tools Test Suite is ready on port 8080 (PID $(<"${pid_file}"))."
        exit 0
    fi
    if ! kill -0 "$(<"${pid_file}")" 2>/dev/null; then
        tail -n 80 "${log_dir}/app.log" >&2
        exit 1
    fi
    sleep 1
done

echo "MoE Tools Test Suite did not become ready within 30 seconds." >&2
tail -n 80 "${log_dir}/app.log" >&2
exit 1
