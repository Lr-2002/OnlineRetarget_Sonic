#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
SONIC_ROOT="${SONIC_ROOT:-/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training}"
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_1m_20260520T220222Z}"
INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-1800}"
VALIDATION_STEP="${VALIDATION_STEP:-20000}"

RUN_ROOT="${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}"
LOG_DIR="${RUN_ROOT}/_launcher"
MONITOR_DIR="${RUN_ROOT}/_monitor"
READY_MD="${MONITOR_DIR}/validation_20k_ready.md"

mkdir -p "${MONITOR_DIR}"

validation_files() {
  if [[ -d "${RUN_ROOT}" ]]; then
    find "${RUN_ROOT}" -path '*online_retarget_visual_validation*' -type f 2>/dev/null
  fi
  if [[ -d "${SONIC_ROOT}/logs_rl/OnlineRetarget" ]]; then
    find "${SONIC_ROOT}/logs_rl/OnlineRetarget" \
      -path "*${RUN_GROUP}*/online_retarget_visual_validation*" \
      -type f \
      2>/dev/null
  fi
}

latest_iteration() {
  local log_file="$1"
  grep -aoE 'Learning iteration[[:space:]]+[0-9]+' "${log_file}" \
    | tail -n 1 \
    | awk '{print $3}'
}

min_iteration() {
  local min_iter="" iter
  for log_file in "${LOG_DIR}"/*.log; do
    iter="$(latest_iteration "${log_file}" || true)"
    iter="${iter:-0}"
    if [[ -z "${min_iter}" || "${iter}" -lt "${min_iter}" ]]; then
      min_iter="${iter}"
    fi
  done
  printf '%s\n' "${min_iter:-0}"
}

write_ready_report() {
  local ts min_iter mp4_count report_count upload_count tmp
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  min_iter="$(min_iteration)"
  mp4_count="$(validation_files | grep -cE '\.mp4$' || true)"
  report_count="$(validation_files | grep -cE 'clip_[0-9]+_report\.json$|rank_report\.json$' || true)"
  upload_count="$(validation_files | grep -c 'main_upload_report.json' || true)"
  tmp="${READY_MD}.tmp"

  {
    printf '# 20k Validation Watch\n\n'
    printf -- '- time_utc: `%s`\n' "${ts}"
    printf -- '- run_group: `%s`\n' "${RUN_GROUP}"
    printf -- '- min_iteration: `%s`\n' "${min_iter}"
    printf -- '- validation_step: `%s`\n' "${VALIDATION_STEP}"
    printf -- '- mp4_count: `%s`\n' "${mp4_count}"
    printf -- '- report_count: `%s`\n' "${report_count}"
    printf -- '- upload_report_count: `%s`\n\n' "${upload_count}"
    printf '## Sample Files\n\n'
    validation_files | sort | head -n 40 | sed 's/^/- `/' | sed 's/$/`/'
  } > "${tmp}"
  mv "${tmp}" "${READY_MD}"
}

while true; do
  if [[ -x "${ROOT}/scripts/monitor_sonic_native_retarget_runs.sh" ]]; then
    MONITOR_ONCE=1 \
      ROOT="${ROOT}" \
      SONIC_ROOT="${SONIC_ROOT}" \
      RETARGET_RUN_GROUP="${RUN_GROUP}" \
      bash "${ROOT}/scripts/monitor_sonic_native_retarget_runs.sh" >/dev/null
  fi

  if [[ "$(validation_files | wc -l | tr -d ' ')" -gt 0 ]]; then
    write_ready_report
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
