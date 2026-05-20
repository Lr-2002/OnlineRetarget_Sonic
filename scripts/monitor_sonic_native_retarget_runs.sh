#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_1m_20260520T220222Z}"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1800}"
ONCE="${MONITOR_ONCE:-0}"

RUN_ROOT="${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}"
LOG_DIR="${RUN_ROOT}/_launcher"
MONITOR_DIR="${RUN_ROOT}/_monitor"
STATUS_JSONL="${MONITOR_DIR}/status.jsonl"
SUMMARY_MD="${MONITOR_DIR}/latest_status.md"

mkdir -p "${MONITOR_DIR}"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

latest_iteration() {
  local log_file="$1"
  grep -aoE 'Learning iteration[[:space:]]+[0-9]+' "${log_file}" \
    | tail -n 1 \
    | awk '{print $3}'
}

hard_error() {
  local log_file="$1"
  grep -E 'Traceback|CUDA out of memory|RuntimeError|ModuleNotFoundError|Error executing job' "${log_file}" \
    | tail -n 1 \
    || true
}

validation_file_count() {
  find "${RUN_ROOT}" -path '*online_retarget_visual_validation*' -type f 2>/dev/null \
    | wc -l \
    | tr -d ' '
}

write_snapshot() {
  local ts tmp validation_count
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmp="${MONITOR_DIR}/latest_status.tmp"
  validation_count="$(validation_file_count)"

  {
    printf '# Sonic Native Retarget Monitor\n\n'
    printf -- '- time_utc: `%s`\n' "${ts}"
    printf -- '- run_group: `%s`\n' "${RUN_GROUP}"
    printf -- '- validation_file_count: `%s`\n\n' "${validation_count}"
    printf '| Variant | Iteration | Log mtime UTC | Hard error |\n'
    printf '| --- | ---: | --- | --- |\n'
  } > "${tmp}"

  for log_file in "${LOG_DIR}"/*.log; do
    local variant iter mtime error_text error_label
    variant="${log_file##*/}"
    variant="${variant%.log}"
    iter="$(latest_iteration "${log_file}" || true)"
    iter="${iter:-0}"
    mtime="$(date -u -d "@$(stat -c %Y "${log_file}")" +%Y-%m-%dT%H:%M:%SZ)"
    error_text="$(hard_error "${log_file}")"
    if [[ -z "${error_text}" ]]; then
      error_label="none"
    else
      error_label="$(printf '%s' "${error_text}" | cut -c1-80)"
    fi

    printf '{"time_utc":"%s","variant":"%s","iteration":%s,"log_mtime_utc":"%s","hard_error":%s,"validation_file_count":%s}\n' \
      "${ts}" \
      "${variant}" \
      "${iter}" \
      "${mtime}" \
      "$(printf '%s' "${error_text}" | json_escape)" \
      "${validation_count}" \
      >> "${STATUS_JSONL}"

    printf '| `%s` | %s | `%s` | `%s` |\n' \
      "${variant}" \
      "${iter}" \
      "${mtime}" \
      "${error_label}" \
      >> "${tmp}"
  done

  mv "${tmp}" "${SUMMARY_MD}"
}

while true; do
  write_snapshot
  if [[ "${ONCE}" == "1" ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
