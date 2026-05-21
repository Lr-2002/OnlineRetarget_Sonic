#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
SONIC_ROOT="${SONIC_ROOT:-/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training}"
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_1m_20260520T220222Z}"
INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-1800}"
VALIDATION_STEP="${VALIDATION_STEP:-20000}"
EXPECTED_UPLOAD_REPORTS="${EXPECTED_UPLOAD_REPORTS:-4}"
EXPECTED_MP4_COUNT="${EXPECTED_MP4_COUNT:-32}"
VALIDATION_STEP_DIR="$(printf 'step_%08d' "${VALIDATION_STEP}")"

RUN_ROOT="${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}"
LOG_DIR="${RUN_ROOT}/_launcher"
MONITOR_DIR="${RUN_ROOT}/_monitor"
READY_MD="${MONITOR_DIR}/validation_20k_ready.md"

mkdir -p "${MONITOR_DIR}"

validation_files() {
  if [[ -d "${RUN_ROOT}" ]]; then
    find "${RUN_ROOT}" \
      -path "*online_retarget_visual_validation/${VALIDATION_STEP_DIR}/*" \
      -type f \
      2>/dev/null
  fi
  if [[ -d "${SONIC_ROOT}/logs_rl/OnlineRetarget" ]]; then
    find "${SONIC_ROOT}/logs_rl/OnlineRetarget" \
      -path "*${RUN_GROUP}*/online_retarget_visual_validation/${VALIDATION_STEP_DIR}/*" \
      -type f \
      2>/dev/null
  fi
}

upload_reports() {
  validation_files | grep 'main_upload_report.json' || true
}

upload_status_summary() {
  python3 - "$@" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

counts = {"ok": 0, "failed": 0, "skipped": 0, "other": 0}
videos_uploaded = 0
for raw_path in sys.argv[1:]:
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except Exception:
        counts["other"] += 1
        continue
    status = None
    for key, value in payload.items():
        if str(key).endswith("/wandb_upload_status"):
            status = str(value)
            break
    if status in counts:
        counts[status] += 1
    else:
        counts["other"] += 1
    for key, value in payload.items():
        if str(key).endswith("/videos_uploaded"):
            try:
                videos_uploaded += int(value)
            except Exception:
                pass
            break
print(
    "\t".join(
        str(value)
        for value in (
            counts["ok"],
            counts["failed"],
            counts["skipped"],
            counts["other"],
            videos_uploaded,
        )
    )
)
PY
}

ready_condition_met() {
  local mp4_count upload_count upload_ok upload_failed upload_skipped upload_other videos_uploaded
  local upload_paths=()
  mp4_count="$(validation_files | grep -cE '\.mp4$' || true)"
  mapfile -t upload_paths < <(upload_reports | sort)
  upload_count="${#upload_paths[@]}"
  IFS=$'\t' read -r upload_ok upload_failed upload_skipped upload_other videos_uploaded \
    < <(upload_status_summary "${upload_paths[@]}")

  [[ "${mp4_count}" -ge "${EXPECTED_MP4_COUNT}" ]] \
    && [[ "${upload_count}" -ge "${EXPECTED_UPLOAD_REPORTS}" ]] \
    && [[ "${upload_ok}" -ge "${EXPECTED_UPLOAD_REPORTS}" ]] \
    && [[ "${upload_failed}" -eq 0 ]] \
    && [[ "${upload_skipped}" -eq 0 ]] \
    && [[ "${upload_other}" -eq 0 ]] \
    && [[ "${videos_uploaded}" -ge "${EXPECTED_MP4_COUNT}" ]]
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
  local upload_ok upload_failed upload_skipped upload_other videos_uploaded
  local upload_paths=()
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  min_iter="$(min_iteration)"
  mp4_count="$(validation_files | grep -cE '\.mp4$' || true)"
  report_count="$(validation_files | grep -cE 'clip_[0-9]+_report\.json$|rank_report\.json$' || true)"
  mapfile -t upload_paths < <(upload_reports | sort)
  upload_count="${#upload_paths[@]}"
  IFS=$'\t' read -r upload_ok upload_failed upload_skipped upload_other videos_uploaded \
    < <(upload_status_summary "${upload_paths[@]}")
  tmp="${READY_MD}.tmp"

  {
    printf '# 20k Validation Watch\n\n'
    printf -- '- time_utc: `%s`\n' "${ts}"
    printf -- '- run_group: `%s`\n' "${RUN_GROUP}"
    printf -- '- min_iteration: `%s`\n' "${min_iter}"
    printf -- '- validation_step: `%s`\n' "${VALIDATION_STEP}"
    printf -- '- validation_step_dir: `%s`\n' "${VALIDATION_STEP_DIR}"
    printf -- '- expected_mp4_count: `%s`\n' "${EXPECTED_MP4_COUNT}"
    printf -- '- expected_upload_report_count: `%s`\n' "${EXPECTED_UPLOAD_REPORTS}"
    printf -- '- mp4_count: `%s`\n' "${mp4_count}"
    printf -- '- report_count: `%s`\n' "${report_count}"
    printf -- '- upload_report_count: `%s`\n' "${upload_count}"
    printf -- '- wandb_upload_ok: `%s`\n' "${upload_ok}"
    printf -- '- wandb_upload_failed: `%s`\n' "${upload_failed}"
    printf -- '- wandb_upload_skipped: `%s`\n' "${upload_skipped}"
    printf -- '- wandb_upload_other: `%s`\n' "${upload_other}"
    printf -- '- wandb_videos_uploaded_total: `%s`\n\n' "${videos_uploaded}"
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

  if ready_condition_met; then
    write_ready_report
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
