#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data_cpfs/code/wxh/OnlineRetarget}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/isaaclab/_isaac_sim/python.sh}"
# Default is the LR-273 temporal-consistency loss-on treatment.
# For the matched LR-274 loss-off baseline, run:
# CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# Final LR-280 kin/walk data-package smoke targets:
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# LR-290 A+B non-uniform MLP capacity experiments:
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_mlp_512_1024_512_1m_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_mlp_512_1024_1024_512_1m_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_mlp_512_1024_2048_1024_512_1m_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
# CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_mlp_512_1024_2048_1024_512_temporal0p1_1m_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
CONFIG="${CONFIG:-configs/sonic_kin_soma_motionlib_proportional_4gpu.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_GROUP="${KIN_RUN_GROUP:-kin_soma_motionlib_supervised_$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${ROOT}/outputs/sonic_kin_soma_motionlib_supervised_runs/${RUN_GROUP}/_launcher}"
GIT_FETCH_TIMEOUT_SECONDS="${GIT_FETCH_TIMEOUT_SECONDS:-60}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

cd "${ROOT}"

if [[ "${CONFIG}" == *" "* ]]; then
  echo "CONFIG must name exactly one supervised soma_motionlib config" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing python launcher: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "missing supervised resume checkpoint: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

TRAINING_LANE="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("training_lane",""))' "${CONFIG}")"
if [[ "${TRAINING_LANE}" != "soma_motionlib_kin_only" ]]; then
  echo "CONFIG must use training_lane=soma_motionlib_kin_only for strict supervised treatment/baseline configs, got ${TRAINING_LANE}" >&2
  exit 1
fi

REQUIRED_GPU_COUNT="$("${PYTHON_BIN}" -c 'import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg.get("runtime",{}).get("required_gpu_count", cfg.get("training",{}).get("required_gpu_count", "")))' "${CONFIG}")"
if [[ "${REQUIRED_GPU_COUNT}" != "${NPROC_PER_NODE}" ]]; then
  echo "NPROC_PER_NODE must match required_gpu_count=${REQUIRED_GPU_COUNT}, got ${NPROC_PER_NODE}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
input_data = config.get("input_data", {})
for key in ("robot_motion_dir", "soma_motion_dir"):
    path = Path(str(input_data.get(key, "")))
    if not path.exists():
        raise SystemExit(f"{key} is missing: {path}")
PY

if "${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bad_tokens = (
    "train_agent_trl.py",
    "KinematicActionUniversalTokenModule",
    "sonic_hydra",
    "num_envs",
    "reward",
    "episode_length",
)
descriptive_string_fields = {"purpose"}


def iter_forbidden_matches(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            for token in bad_tokens:
                if token in key_text:
                    yield ".".join(child_path), token, "key"
            yield from iter_forbidden_matches(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from iter_forbidden_matches(child, (*path, str(idx)))
    elif isinstance(value, str) and (not path or path[-1] not in descriptive_string_fields):
        for token in bad_tokens:
            if token in value:
                yield ".".join(path), token, "value"


matches = list(iter_forbidden_matches(config))
if matches:
    location, token, kind = matches[0]
    raise SystemExit(
        f"CONFIG contains forbidden strict-supervised token {token!r} in {kind} at {location}"
    )
PY
then
  :
else
  echo "CONFIG contains PPO/Isaac/reward/episode-length tokens and is not a strict supervised config: ${CONFIG}" >&2
  exit 1
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "need at least ${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES entries, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" - "${NPROC_PER_NODE}" <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}") from exc

required = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for strict supervised 4-GPU smoke")
if torch.cuda.device_count() < required:
    raise SystemExit(
        f"expected at least {required} visible CUDA device(s), found {torch.cuda.device_count()}"
    )
if not torch.distributed.is_available():
    raise SystemExit("torch.distributed is required for strict supervised DDP smoke")
PY

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
  echo "OnlineRetarget repo has uncommitted tracked changes; commit before training" >&2
  git status --short >&2
  exit 1
fi
require_latest_git "${ROOT}" "OnlineRetarget repo"

SOURCE_ROOT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_repo", ""))' "${CONFIG}")"
EXTERNAL_SOURCE_COMMIT="not-required-for-supervised-entrypoint"
if [[ -n "${SOURCE_ROOT}" ]] && git -C "${SOURCE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  EXTERNAL_SOURCE_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
fi

VARIANT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["variant"]["name"])' "${CONFIG}")"
SESSION="sonic_${RUN_GROUP}_${VARIANT}"
SESSION="${SESSION//[^A-Za-z0-9_]/_}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${LAUNCH_ROOT}"
LOG_PATH="${LAUNCH_ROOT}/${VARIANT}.log"
TRAIN_ARGS=(scripts/train_sonic_kin_skeleton_ae.py --config "${CONFIG}")
if [[ -n "${MAX_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
  TRAIN_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi
if [[ "${DISABLE_VISUAL_VALIDATION:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--disable-visual-validation)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
TRAIN_COMMAND="$(printf '%q ' "${PYTHON_BIN}" -m torch.distributed.run --standalone "--nproc-per-node=${NPROC_PER_NODE}" "${TRAIN_ARGS[@]}")"
LOG_PATH_QUOTED="$(printf '%q' "${LOG_PATH}")"

cmd=$(cat <<EOF
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export KIN_RUN_GROUP="${RUN_GROUP}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
echo "variant=${VARIANT} nproc=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} control_commit=${CONTROL_COMMIT} external_source_commit=${EXTERNAL_SOURCE_COMMIT}"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
${TRAIN_COMMAND} 2>&1 | tee -a ${LOG_PATH_QUOTED}
EOF
)

"${PYTHON_BIN}" - "${LAUNCH_ROOT}/launch_manifest.json" "${RUN_GROUP}" "${CONFIG}" "${VARIANT}" "${NPROC_PER_NODE}" "${CUDA_VISIBLE_DEVICES}" "${CONTROL_COMMIT}" "${EXTERNAL_SOURCE_COMMIT}" "${WANDB_MODE:-online}" "${DISABLE_VISUAL_VALIDATION:-0}" "${MAX_STEPS:-}" "${SESSION}" "${RESUME_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = {
    "run_group": sys.argv[2],
    "config": sys.argv[3],
    "variant": sys.argv[4],
    "nproc_per_node": int(sys.argv[5]),
    "cuda_visible_devices": sys.argv[6],
    "control_commit": sys.argv[7],
    "external_source_commit": sys.argv[8],
    "external_source_guard": "not_required_supervised_entrypoint_no_external_import_exec",
    "wandb_mode": sys.argv[9],
    "disable_visual_validation": sys.argv[10] == "1",
    "max_steps_override": sys.argv[11],
    "tmux_session": sys.argv[12],
    "resume_checkpoint": sys.argv[13],
    "entrypoint": "scripts/train_sonic_kin_skeleton_ae.py",
    "distributed_launcher": "python -m torch.distributed.run",
    "contract": "strict_supervised_soma_motionlib_kin_only_no_ppo_no_isaac_no_reward_episode_length",
}
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${NO_TMUX:-0}" == "1" ]]; then
  bash -lc "${cmd}"
else
  tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${cmd}")"
  printf 'started run_group=%s session=%s config=%s\n' "${RUN_GROUP}" "${SESSION}" "${CONFIG}"
fi
