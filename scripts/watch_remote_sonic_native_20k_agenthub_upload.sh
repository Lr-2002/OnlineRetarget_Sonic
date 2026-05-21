#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-5090}"
ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_1m_20260520T220222Z}"
INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-1800}"
VALIDATION_STEP="${VALIDATION_STEP:-20000}"
AGENT_HUB_REPO="${AGENT_HUB_REPO:-/home/user/agent-hub}"
AGENT_HUB_PROJECT="${AGENT_HUB_PROJECT:-online-retarget}"
AGENT_HUB_RUN="${AGENT_HUB_RUN:-20260521-080713-onlineretarget-sonic-native-1m-run-status}"
AGENT_HUB_SOURCE_DEVICE="${AGENT_HUB_SOURCE_DEVICE:-$(hostname)}"
AGENT_HUB_DEVICE_TYPE="${AGENT_HUB_DEVICE_TYPE:-server}"
AGENT_HUB_AGENT_IDENTITY="${AGENT_HUB_AGENT_IDENTITY:-codex@$(hostname)}"

VALIDATION_STEP_DIR="$(printf 'step_%08d' "${VALIDATION_STEP}")"
REMOTE_MONITOR_DIR="${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}/_monitor"
REMOTE_READY_MD="${REMOTE_MONITOR_DIR}/validation_20k_ready.md"
REMOTE_STATUS_MD="${REMOTE_MONITOR_DIR}/latest_status.md"

LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-outputs/agenthub_20k_uploads/${RUN_GROUP}}"
LOCAL_READY_MD="${LOCAL_OUTPUT_DIR}/validation_20k_ready.md"
LOCAL_STATUS_MD="${LOCAL_OUTPUT_DIR}/latest_status.md"
LOCAL_MARKER="${LOCAL_OUTPUT_DIR}/agenthub_upload_done.marker"
LOCAL_LOG="${LOCAL_OUTPUT_DIR}/watch.log"

REPO_ROOT="$(pwd)"
mkdir -p "${LOCAL_OUTPUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOCAL_LOG}"
}

remote_ready() {
  ssh "${REMOTE_HOST}" "test -s '${REMOTE_READY_MD}'"
}

copy_remote_reports() {
  scp "${REMOTE_HOST}:${REMOTE_READY_MD}" "${LOCAL_READY_MD}"
  scp "${REMOTE_HOST}:${REMOTE_STATUS_MD}" "${LOCAL_STATUS_MD}"
}

upload_to_agenthub() {
  cd "${AGENT_HUB_REPO}"
  uv run ah upload \
    "$(realpath "${REPO_ROOT}/${LOCAL_READY_MD}")" \
    "$(realpath "${REPO_ROOT}/${LOCAL_STATUS_MD}")" \
    --project "${AGENT_HUB_PROJECT}" \
    --run "${AGENT_HUB_RUN}" \
    --title "OnlineRetarget 20k validation ready" \
    --summary "20k validation gate is ready for ${RUN_GROUP}; uploaded validation_20k_ready.md and latest_status.md for review." \
    --artifact-type markdown \
    --source-device "${AGENT_HUB_SOURCE_DEVICE}" \
    --device-type "${AGENT_HUB_DEVICE_TYPE}" \
    --agent-identity "${AGENT_HUB_AGENT_IDENTITY}"
}

if [[ -s "${LOCAL_MARKER}" ]]; then
  log "already uploaded according to ${LOCAL_MARKER}"
  exit 0
fi

log "watching ${REMOTE_HOST}:${REMOTE_READY_MD} (${VALIDATION_STEP_DIR})"
while true; do
  if remote_ready; then
    log "remote 20k ready report detected"
    copy_remote_reports
    upload_to_agenthub | tee -a "${LOCAL_LOG}"
    {
      printf 'uploaded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'run_group=%s\n' "${RUN_GROUP}"
      printf 'validation_step=%s\n' "${VALIDATION_STEP}"
      printf 'agenthub_project=%s\n' "${AGENT_HUB_PROJECT}"
      printf 'agenthub_run=%s\n' "${AGENT_HUB_RUN}"
    } > "${LOCAL_MARKER}"
    log "upload complete"
    exit 0
  fi
  log "not ready; sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
