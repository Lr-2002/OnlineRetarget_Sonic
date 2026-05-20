#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/isaaclab/_isaac_sim/python.sh}"
RUN_GROUP="${KIN_RUN_GROUP:-kin_skeleton_$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${ROOT}/outputs/sonic_kin_skeleton_runs/${RUN_GROUP}/_launcher}"
GIT_FETCH_TIMEOUT_SECONDS="${GIT_FETCH_TIMEOUT_SECONDS:-60}"

cd "${ROOT}"

if [[ -n "${CONFIG:-}" ]]; then
  CONFIGS=("${CONFIG}")
else
  CONFIGS=(
    "configs/sonic_kin_skeleton_a1_concat_1gpu.json"
    "configs/sonic_kin_skeleton_a2_film_1gpu.json"
    "configs/sonic_kin_skeleton_b1_adapter_1gpu.json"
    "configs/sonic_kin_skeleton_b2_expert_1gpu.json"
  )
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#CONFIGS[@]}" -eq 1 ]]; then
  GPUS=("${CUDA_VISIBLE_DEVICES}")
else
  read -r -a GPUS <<< "${GPU_ASSIGNMENTS:-0 1 2 3}"
fi

if [[ "${#GPUS[@]}" -lt "${#CONFIGS[@]}" ]]; then
  echo "need at least ${#CONFIGS[@]} GPU assignments, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing python launcher: ${PYTHON_BIN}" >&2
  exit 1
fi

require_latest_git() {
  local repo="$1"
  local label="$2"
  local upstream remote branch head upstream_head

  upstream="$(git -C "${repo}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [[ -z "${upstream}" ]]; then
    echo "${label} has no upstream tracking branch; set upstream before remote training" >&2
    exit 1
  fi

  remote="${upstream%%/*}"
  branch="${upstream#*/}"
  if [[ -z "${remote}" || -z "${branch}" || "${remote}" == "${branch}" ]]; then
    echo "${label} has unsupported upstream '${upstream}'" >&2
    exit 1
  fi

  if ! timeout "${GIT_FETCH_TIMEOUT_SECONDS}" git -C "${repo}" fetch --quiet "${remote}" "${branch}"; then
    echo "${label} could not fetch ${upstream}; refusing to start training without a latest-code check" >&2
    exit 1
  fi

  head="$(git -C "${repo}" rev-parse HEAD)"
  upstream_head="$(git -C "${repo}" rev-parse FETCH_HEAD)"
  if [[ "${head}" != "${upstream_head}" ]]; then
    echo "${label} is not latest: HEAD=${head}, ${upstream}=${upstream_head}" >&2
    echo "pull or sync the remote checkout before training" >&2
    exit 1
  fi
}

if git diff --quiet && git diff --cached --quiet; then
  CONTROL_COMMIT="$(git rev-parse HEAD)"
else
  echo "control repo has uncommitted tracked changes; commit before training" >&2
  git status --short >&2
  exit 1
fi
require_latest_git "${ROOT}" "OnlineRetarget repo"

SONIC_ROOT="$("${PYTHON_BIN}" - "${CONFIGS[0]}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["source_repo"])
PY
)"
if git -C "${SONIC_ROOT}" diff --quiet && git -C "${SONIC_ROOT}" diff --cached --quiet; then
  SONIC_COMMIT="$(git -C "${SONIC_ROOT}" rev-parse HEAD)"
else
  echo "SONIC source repo has uncommitted tracked changes; commit before training" >&2
  git -C "${SONIC_ROOT}" status --short >&2
  exit 1
fi

mkdir -p "${LAUNCH_ROOT}"

declare -a SESSIONS=()
for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  if [[ ! -f "${cfg}" ]]; then
    echo "missing config: ${cfg}" >&2
    exit 1
  fi
  variant="$("${PYTHON_BIN}" - "${cfg}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["variant"]["name"])
PY
)"
  session="sonic_${RUN_GROUP}_${variant}"
  session="${session//[^A-Za-z0-9_]/_}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi
  log_path="${LAUNCH_ROOT}/${variant}.log"
  cmd=$(cat <<EOF
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export KIN_RUN_GROUP="${RUN_GROUP}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "variant=${variant} gpu=${gpu} control_commit=${CONTROL_COMMIT} sonic_commit=${SONIC_COMMIT}"
"${PYTHON_BIN}" scripts/train_sonic_kin_skeleton_ae.py --config "${cfg}" 2>&1 | tee -a "${log_path}"
EOF
)
  if [[ "${NO_TMUX:-0}" == "1" && "${#CONFIGS[@]}" -eq 1 ]]; then
    bash -lc "${cmd}"
  else
    tmux new-session -d -s "${session}" "bash -lc $(printf '%q' "${cmd}")"
    SESSIONS+=("${session}")
  fi
done

"${PYTHON_BIN}" - "${LAUNCH_ROOT}/launch_manifest.json" "${RUN_GROUP}" "${CONTROL_COMMIT}" "${SONIC_COMMIT}" "${CONFIGS[@]}" -- "${GPUS[@]}" "${SESSIONS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_group = sys.argv[2]
control_commit = sys.argv[3]
sonic_commit = sys.argv[4]
sep = sys.argv.index("--")
configs = sys.argv[5:sep]
tail = sys.argv[sep + 1 :]
gpu_count = len(configs)
gpus = tail[:gpu_count]
sessions = tail[gpu_count:]
manifest = {
    "run_group": run_group,
    "control_commit": control_commit,
    "sonic_commit": sonic_commit,
    "configs": configs,
    "gpus": gpus,
    "tmux_sessions": sessions,
}
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${#SESSIONS[@]}" -gt 0 ]]; then
  printf 'started run_group=%s sessions=%s\n' "${RUN_GROUP}" "${SESSIONS[*]}"
fi
