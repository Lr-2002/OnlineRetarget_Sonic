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
VALIDATION_STEP="${VALIDATION_STEP:-20000}"
FINAL_STEP="${FINAL_STEP:-1000000}"

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

rate_fields() {
  local variant="$1"
  local iter="$2"
  local ts="$3"
  python3 - "${STATUS_JSONL}" "${variant}" "${iter}" "${ts}" "${VALIDATION_STEP}" "${FINAL_STEP}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
variant = sys.argv[2]
current_iter = int(sys.argv[3])
current_ts = dt.datetime.strptime(sys.argv[4], "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=dt.timezone.utc
)
validation_step = int(sys.argv[5])
final_step = int(sys.argv[6])

records: list[tuple[dt.datetime, int]] = []
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("variant") != variant:
                continue
            row_ts = dt.datetime.strptime(
                str(row["time_utc"]), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=dt.timezone.utc)
            row_iter = int(row["iteration"])
        except Exception:
            continue
        if row_ts < current_ts and row_iter <= current_iter:
            records.append((row_ts, row_iter))

rate_per_hour: float | None = None
if records:
    base_ts, base_iter = min(records, key=lambda item: item[0])
    seconds = (current_ts - base_ts).total_seconds()
    delta_iter = current_iter - base_iter
    if seconds > 0 and delta_iter > 0:
        rate_per_hour = delta_iter / seconds * 3600.0


def format_eta(target_step: int) -> str:
    if rate_per_hour is None or rate_per_hour <= 0:
        return "unknown"
    remaining = max(0, target_step - current_iter)
    seconds = remaining / rate_per_hour * 3600.0
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


rate_text = "unknown" if rate_per_hour is None else f"{rate_per_hour:.1f}"
print("\t".join((rate_text, format_eta(validation_step), format_eta(final_step))))
PY
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
    printf '| Variant | Iteration | Iter/hr | ETA 20k | ETA 1M | Log mtime UTC | Hard error |\n'
    printf '| --- | ---: | ---: | --- | --- | --- | --- |\n'
  } > "${tmp}"

  for log_file in "${LOG_DIR}"/*.log; do
    local variant iter mtime error_text error_label iter_per_hour eta_20k eta_1m iter_per_hour_json
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
    IFS=$'\t' read -r iter_per_hour eta_20k eta_1m < <(rate_fields "${variant}" "${iter}" "${ts}")
    if [[ "${iter_per_hour}" == "unknown" ]]; then
      iter_per_hour_json="null"
    else
      iter_per_hour_json="${iter_per_hour}"
    fi

    printf '{"time_utc":"%s","variant":"%s","iteration":%s,"iter_per_hour":%s,"eta_20k":%s,"eta_1m":%s,"log_mtime_utc":"%s","hard_error":%s,"validation_file_count":%s}\n' \
      "${ts}" \
      "${variant}" \
      "${iter}" \
      "${iter_per_hour_json}" \
      "$(printf '%s' "${eta_20k}" | json_escape)" \
      "$(printf '%s' "${eta_1m}" | json_escape)" \
      "${mtime}" \
      "$(printf '%s' "${error_text}" | json_escape)" \
      "${validation_count}" \
      >> "${STATUS_JSONL}"

    printf '| `%s` | %s | %s | `%s` | `%s` | `%s` | `%s` |\n' \
      "${variant}" \
      "${iter}" \
      "${iter_per_hour}" \
      "${eta_20k}" \
      "${eta_1m}" \
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
