#!/usr/bin/env bash
set -euo pipefail
umask 077

runner_path="${MOE_TOOLS_ACCEPTANCE_RUNNER_PATH:-/usr/local/bin/run-v2-live-acceptance-on-start}"
python_path="${MOE_TOOLS_ACCEPTANCE_PYTHON_PATH:-/opt/vllm-venv/bin/python}"
app_root="${MOE_TOOLS_APP_ROOT:-/opt/moe-tools-test-suite}"
data_root="${MOE_TOOLS_DATA_DIR:-/workspace/moe-tools}"
guard_dir="${data_root}/acceptance"
private_dir="${MOE_TOOLS_ACCEPTANCE_PRIVATE_DIR:-/var/lib/moe-tools/acceptance}"
status_file="${guard_dir}/v2-live-acceptance.status"
pid_file="${guard_dir}/v2-live-acceptance.pid"
start_file="${guard_dir}/v2-live-acceptance.start"
lock_file="${guard_dir}/v2-live-acceptance.lock"
log_file="${private_dir}/v2-live-acceptance.log"
evidence_file="${private_dir}/v2-live-acceptance.json"
before_file="${private_dir}/daytona-before.json"
after_file="${private_dir}/daytona-after.json"

utc_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

read_mode() {
    local path="$1"
    local mode
    if mode="$(stat -c '%a' -- "${path}" 2>/dev/null)"; then
        :
    elif mode="$(stat -f '%Lp' -- "${path}" 2>/dev/null)"; then
        :
    else
        return 1
    fi
    [[ "${mode}" =~ ^[0-7]+$ ]] || return 1
    printf '%s\n' "${mode}"
}

secure_private_directory() {
    if [[ -L "${private_dir}" ]]; then
        echo "Private acceptance artifact directory must not be a symlink." >&2
        return 1
    fi
    mkdir -p "${private_dir}" || return 1
    if [[ ! -d "${private_dir}" || -L "${private_dir}" ]]; then
        echo "Private acceptance artifact path is not a real directory." >&2
        return 1
    fi
    chmod 0700 "${private_dir}" || return 1
    local mode
    mode="$(read_mode "${private_dir}")" || return 1
    if [[ "${mode}" != "700" ]]; then
        echo "Private acceptance artifact directory did not retain mode 0700." >&2
        return 1
    fi
}

secure_private_file() {
    local path="$1"
    if [[ ! -f "${path}" || -L "${path}" ]]; then
        echo "Private acceptance artifact is missing or is not a regular file." >&2
        return 1
    fi
    chmod 0600 "${path}" || return 1
    local mode
    mode="$(read_mode "${path}")" || return 1
    if [[ "${mode}" != "600" ]]; then
        echo "Private acceptance artifact did not retain mode 0600." >&2
        return 1
    fi
}

secure_private_file_if_present() {
    local path="$1"
    if [[ -e "${path}" || -L "${path}" ]]; then
        secure_private_file "${path}"
    fi
}

probe_private_artifact_storage() {
    secure_private_directory || return 1
    local probe
    probe="$(mktemp "${private_dir}/.permissions.XXXXXX")" || return 1
    if ! secure_private_file "${probe}"; then
        rm -f "${probe}"
        return 1
    fi
    rm -f "${probe}"
}

create_private_log() {
    if [[ -L "${log_file}" || ( -e "${log_file}" && ! -f "${log_file}" ) ]]; then
        echo "Private acceptance log path is not a regular file." >&2
        return 1
    fi
    : >"${log_file}" || return 1
    secure_private_file "${log_file}"
}

write_status() {
    local state="$1"
    local harness_exit_code="$2"
    local before_exit_code="$3"
    local after_exit_code="$4"
    local cancellation_exit_code="$5"
    local secret_scan_exit_code="$6"
    local artifact_permissions_exit_code="$7"
    local started_at="$8"
    local finished_at="$9"
    local temporary
    temporary="$(mktemp "${guard_dir}/.v2-live-acceptance.status.XXXXXX")"
    jq -n \
        --arg state "${state}" \
        --arg evidence_path "${evidence_file}" \
        --arg log_path "${log_file}" \
        --arg before_path "${before_file}" \
        --arg after_path "${after_file}" \
        --arg started_at "${started_at}" \
        --arg finished_at "${finished_at}" \
        --arg harness_exit_code "${harness_exit_code}" \
        --arg before_exit_code "${before_exit_code}" \
        --arg after_exit_code "${after_exit_code}" \
        --arg cancellation_exit_code "${cancellation_exit_code}" \
        --arg secret_scan_exit_code "${secret_scan_exit_code}" \
        --arg artifact_permissions_exit_code "${artifact_permissions_exit_code}" \
        '{
            schema_version: "moe-atelier-v2-startup-acceptance-status/v1",
            state: $state,
            evidence_path: $evidence_path,
            log_path: $log_path,
            daytona_before_path: $before_path,
            daytona_after_path: $after_path,
            started_at: (if $started_at == "" then null else $started_at end),
            finished_at: (if $finished_at == "" then null else $finished_at end),
            harness_exit_code: (
                if $harness_exit_code == "" then null
                else ($harness_exit_code | tonumber)
                end
            ),
            daytona_before_exit_code: (
                if $before_exit_code == "" then null
                else ($before_exit_code | tonumber)
                end
            ),
            daytona_after_exit_code: (
                if $after_exit_code == "" then null
                else ($after_exit_code | tonumber)
                end
            ),
            cancellation_exit_code: (
                if $cancellation_exit_code == "" then null
                else ($cancellation_exit_code | tonumber)
                end
            ),
            secret_scan_exit_code: (
                if $secret_scan_exit_code == "" then null
                else ($secret_scan_exit_code | tonumber)
                end
            ),
            artifact_permissions_exit_code: (
                if $artifact_permissions_exit_code == "" then null
                else ($artifact_permissions_exit_code | tonumber)
                end
            )
        }' >"${temporary}"
    mv -f "${temporary}" "${status_file}"
}

credential_is_resolved() {
    local value="$1"
    [[ -n "${value}" && "${value}" != *RUNPOD_SECRET_* && "${value}" != REPLACE* ]]
}

worker_is_live() {
    [[ -f "${pid_file}" ]] || return 1
    local worker_pid
    worker_pid="$(<"${pid_file}")"
    [[ "${worker_pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${worker_pid}" 2>/dev/null || return 1
    [[ -r "/proc/${worker_pid}/cmdline" ]] || return 1
    tr '\0' ' ' <"/proc/${worker_pid}/cmdline" \
        | grep -Fq "${runner_path} --worker"
}

claim_worker_handoff() {
    exec 9>"${lock_file}"
    if ! flock -w 10 9; then
        echo "V2 startup acceptance worker could not acquire its handoff lock." >&2
        return 1
    fi

    local state worker_pid
    for _ in {1..100}; do
        state="$(jq -er '.state | strings' "${status_file}" 2>/dev/null)" || {
            sleep 0.1
            continue
        }
        if [[ "${state}" != "starting" && "${state}" != "running" ]]; then
            echo "V2 startup acceptance worker handoff is not running." >&2
            return 1
        fi
        if [[ "${state}" != "running" || ! -e "${pid_file}" ]]; then
            sleep 0.1
            continue
        fi
        if [[ ! -f "${pid_file}" || -L "${pid_file}" ]]; then
            echo "V2 startup acceptance worker PID handoff is invalid." >&2
            return 1
        fi
        worker_pid="$(<"${pid_file}")"
        if [[ ! "${worker_pid}" =~ ^[0-9]+$ || "${worker_pid}" != "$$" ]]; then
            echo "V2 startup acceptance worker PID does not match its handoff." >&2
            return 1
        fi
        if [[ -L "${start_file}" ]]; then
            echo "V2 startup acceptance one-shot handoff is invalid." >&2
            return 1
        fi
        if [[ ! -f "${start_file}" ]]; then
            sleep 0.1
            continue
        fi
        rm -f "${start_file}"
        return 0
    done
    echo "V2 startup acceptance worker handoff timed out." >&2
    return 1
}

worker_interrupted=0
active_child_pid=""

handle_worker_signal() {
    worker_interrupted=1
    if [[ -n "${active_child_pid}" ]]; then
        kill -TERM "${active_child_pid}" 2>/dev/null || true
    fi
}

mark_worker_interrupted() {
    worker_interrupted=1
}

run_tracked() {
    "$@" &
    active_child_pid=$!
    wait "${active_child_pid}"
    local exit_code=$?
    active_child_pid=""
    return "${exit_code}"
}

cancel_active_app_job() {
    run_tracked env PYTHONPATH="${app_root}" "${python_path}" -c '
import os
import time

from scripts.acceptance_canary import ApiClient, CanaryConfig

config = CanaryConfig(
    base_url="http://127.0.0.1:8080",
    token=os.environ["MOE_TOOLS_AUTH_TOKEN"],
    poll_interval_seconds=0.5,
    request_timeout_seconds=10,
)
with ApiClient(config) as api:
    api.login()
    active = api.get("/api/jobs/active")
    if active is not None:
        job_id = active.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise SystemExit(1)
        api.post(f"/api/jobs/{job_id}/cancel")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = api.get(f"/api/jobs/{job_id}")
            if job.get("status") not in {"queued", "running", "cancelling"}:
                break
            time.sleep(0.5)
        else:
            raise SystemExit(1)
'
}

run_worker() {
    trap handle_worker_signal HUP INT TERM
    local started_at
    started_at="$(utc_now)"
    if ! probe_private_artifact_storage \
        || ! secure_private_file "${log_file}"; then
        write_status "failed" "" "" "" "" "" "1" \
            "${started_at}" "$(utc_now)"
        return 1
    fi

    export MOE_TOOLS_CANARY_TOKEN="${MOE_TOOLS_AUTH_TOKEN}"
    export MOE_TOOLS_CANARY_BASE_URL="http://127.0.0.1:8080"

    set +e
    local artifact_permissions_exit_code=0
    run_tracked "${python_path}" "${app_root}/scripts/daytona_inventory.py" \
        --output "${before_file}"
    local before_exit_code=$?
    if ! secure_private_file_if_present "${before_file}"; then
        artifact_permissions_exit_code=1
    fi
    if (( before_exit_code == 0 )) && ! jq -e \
        '.success == true and .provider_wide_empty == true and .sandbox_count == 0' \
        "${before_file}" >/dev/null; then
        before_exit_code=3
    fi

    local harness_exit_code=125
    if (( before_exit_code == 0 && artifact_permissions_exit_code == 0 )); then
        run_tracked "${python_path}" "${app_root}/scripts/v2_live_acceptance.py" \
            --base-url "http://127.0.0.1:8080" \
            --model-id "Qwen/Qwen3.6-35B-A3B-FP8" \
            --switch-rounds 5 \
            --coding-provider daytona \
            --output "${evidence_file}"
        harness_exit_code=$?
    else
        echo "Refusing paid acceptance because its preflight did not pass."
    fi
    if ! secure_private_file_if_present "${evidence_file}"; then
        artifact_permissions_exit_code=1
    elif (( harness_exit_code == 0 )) && [[ ! -f "${evidence_file}" ]]; then
        artifact_permissions_exit_code=1
    fi

    trap mark_worker_interrupted HUP INT TERM
    local cancellation_exit_code=0
    if (( worker_interrupted != 0 )); then
        cancel_active_app_job
        cancellation_exit_code=$?
    fi

    run_tracked "${python_path}" "${app_root}/scripts/daytona_inventory.py" \
        --output "${after_file}"
    local after_exit_code=$?
    if ! secure_private_file_if_present "${after_file}"; then
        artifact_permissions_exit_code=1
    fi
    if (( after_exit_code == 0 )) && ! jq -e \
        '.success == true and .provider_wide_empty == true and .sandbox_count == 0' \
        "${after_file}" >/dev/null; then
        after_exit_code=3
    fi

    if ! secure_private_file "${log_file}"; then
        artifact_permissions_exit_code=1
    fi

    run_tracked "${python_path}" -c '
import json
import os
from pathlib import Path
import sys

secrets = [
    os.environ.get("MOE_TOOLS_AUTH_TOKEN", ""),
    os.environ.get("MOE_TOOLS_DAYTONA_API_KEY", ""),
]
needles = []
for secret in secrets:
    if secret:
        needles.extend(
            (secret.encode(), json.dumps(secret, ensure_ascii=False)[1:-1].encode())
        )
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if path.is_file() and any(needle in path.read_bytes() for needle in needles):
        raise SystemExit(1)
' "${before_file}" "${evidence_file}" "${after_file}" "${log_file}"
    local secret_scan_exit_code=$?
    set -e

    local state="completed"
    if (( worker_interrupted != 0 )); then
        state="interrupted"
    fi
    if ((
        before_exit_code != 0
        || harness_exit_code != 0
        || after_exit_code != 0
        || secret_scan_exit_code != 0
        || cancellation_exit_code != 0
        || artifact_permissions_exit_code != 0
    )); then
        if [[ "${state}" != "interrupted" ]]; then
            state="failed"
        fi
    fi
    write_status \
        "${state}" \
        "${harness_exit_code}" \
        "${before_exit_code}" \
        "${after_exit_code}" \
        "${cancellation_exit_code}" \
        "${secret_scan_exit_code}" \
        "${artifact_permissions_exit_code}" \
        "${started_at}" \
        "$(utc_now)"
    if [[ "${state}" == "completed" ]]; then
        return 0
    fi
    return 1
}

mkdir -p "${guard_dir}"

if [[ "${1:-}" == "--worker" ]]; then
    claim_worker_handoff || exit 1
    run_worker >>"${log_file}" 2>&1
    exit $?
fi
if (( $# != 0 )); then
    echo "run-v2-live-acceptance-on-start accepts no arguments." >&2
    exit 2
fi

case "${MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START:-0}" in
    0)
        exit 0
        ;;
    1) ;;
    *)
        echo "MOE_TOOLS_RUN_V2_ACCEPTANCE_ON_START must be 0 or 1." >&2
        exit 2
        ;;
esac

if [[ "${MOE_TOOLS_MODE:-}" != "vllm" ]]; then
    echo "Startup acceptance requires MOE_TOOLS_MODE=vllm." >&2
    exit 2
fi

if ! credential_is_resolved "${MOE_TOOLS_AUTH_TOKEN:-}"; then
    echo "Startup acceptance requires a resolved MOE_TOOLS_AUTH_TOKEN." >&2
    exit 2
fi
if ! credential_is_resolved "${MOE_TOOLS_DAYTONA_API_KEY:-}"; then
    echo "Startup acceptance requires a resolved MOE_TOOLS_DAYTONA_API_KEY." >&2
    exit 2
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "V2 startup acceptance launch is already being evaluated."
    exit 0
fi

if [[ -f "${status_file}" ]]; then
    prior_state="$(jq -er '.state | strings' "${status_file}" 2>/dev/null)" || {
        echo "Refusing to replace an unreadable V2 acceptance status file." >&2
        exit 2
    }
    if [[ "${prior_state}" == "starting" || "${prior_state}" == "running" ]]; then
        if worker_is_live; then
            echo "V2 startup acceptance is already running."
            exit 0
        fi
        write_status "interrupted" "" "" "" "" "" "" "" "$(utc_now)"
        echo "Stale V2 startup acceptance was marked interrupted; not rerunning."
        exit 0
    fi
    echo "V2 startup acceptance already has terminal state ${prior_state}; not rerunning."
    exit 0
fi

permission_check_started_at="$(utc_now)"
if ! probe_private_artifact_storage || ! create_private_log; then
    write_status "failed" "" "" "" "" "" "1" \
        "${permission_check_started_at}" "$(utc_now)"
    echo "Private acceptance artifact storage cannot enforce POSIX modes." >&2
    exit 1
fi

write_status "starting" "" "" "" "" "" "0" "" ""
rm -f "${start_file}"
setsid "${runner_path}" --worker >/dev/null 2>&1 < /dev/null &
worker_pid=$!
pid_temporary="$(mktemp "${guard_dir}/.v2-live-acceptance.pid.XXXXXX")"
printf '%s\n' "${worker_pid}" >"${pid_temporary}"
mv -f "${pid_temporary}" "${pid_file}"
write_status "running" "" "" "" "" "" "0" "$(utc_now)" ""
: >"${start_file}"
echo "V2 startup acceptance launched in the background; see ${status_file}."
