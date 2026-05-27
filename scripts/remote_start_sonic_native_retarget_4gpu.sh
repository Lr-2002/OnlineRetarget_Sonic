#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-${SCRIPT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/isaaclab/_isaac_sim/python.sh}"
if [[ -z "${ACCELERATE_CMD:-}" ]]; then
  ACCELERATE_CMD="${PYTHON_BIN} -m accelerate.commands.launch"
fi
CONFIG="${CONFIG:-configs/sonic_native_retarget_a1_concat_1gpu.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GPU_ASSIGNMENTS="${GPU_ASSIGNMENTS:-0,1,2,3}"
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_4gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}/_launcher}"
GIT_FETCH_TIMEOUT_SECONDS="${GIT_FETCH_TIMEOUT_SECONDS:-60}"
EXECUTE_SONIC_NATIVE_TRAINING="${EXECUTE_SONIC_NATIVE_TRAINING:-0}"
CHECK_SONIC_PATHS="${CHECK_SONIC_PATHS:-${EXECUTE_SONIC_NATIVE_TRAINING}}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
ACCELERATE_DYNAMO_BACKEND="${ACCELERATE_DYNAMO_BACKEND:-no}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_ALGO="${NCCL_ALGO:-Ring}"
NCCL_DEBUG="${NCCL_DEBUG:-}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-}"
TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-}"

cd "${ROOT}"

if [[ "${CONFIG}" == *" "* ]]; then
  echo "CONFIG must name exactly one formal config for a single 4-GPU training job: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! "${NPROC_PER_NODE}" =~ ^[0-9]+$ || "${NPROC_PER_NODE}" -lt 2 ]]; then
  echo "NPROC_PER_NODE must be >= 2 for native multi-GPU training, got ${NPROC_PER_NODE}" >&2
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

require_latest_git_if_configured() {
  local repo="$1"
  local label="$2"
  local upstream

  upstream="$(git -C "${repo}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [[ -z "${upstream}" ]]; then
    echo "${label} has no upstream tracking branch; requiring a clean tree and recording the local commit only" >&2
    return 0
  fi
  require_latest_git "${repo}" "${label}"
}

if git diff --quiet && git diff --cached --quiet; then
  CONTROL_COMMIT="$(git rev-parse HEAD)"
else
  echo "OnlineRetarget repo has uncommitted tracked changes; commit before training" >&2
  git status --short >&2
  exit 1
fi
require_latest_git "${ROOT}" "OnlineRetarget repo"

VALIDATE_ARGS=(--require-formal)
if [[ "${CHECK_SONIC_PATHS}" == "1" ]]; then
  VALIDATE_ARGS+=(--check-paths)
fi
"${PYTHON_BIN}" scripts/validate_sonic_native_retarget_config.py "${VALIDATE_ARGS[@]}" "${CONFIG}"

SONIC_ROOT="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["source_repo"])
PY
)"

if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then
  if git -C "${SONIC_ROOT}" diff --quiet && git -C "${SONIC_ROOT}" diff --cached --quiet; then
    SONIC_COMMIT="$(git -C "${SONIC_ROOT}" rev-parse HEAD)"
  else
    echo "SONIC source repo has uncommitted tracked changes; commit before training" >&2
    git -C "${SONIC_ROOT}" status --short >&2
    exit 1
  fi
  require_latest_git_if_configured "${SONIC_ROOT}" "SONIC source repo"
else
  SONIC_COMMIT="not-checked-contract-only"
fi

"${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not config.get("sonic_hydra", {}).get("variant_wired", False):
    raise SystemExit(
        "formal config is validated, but sonic_hydra.variant_wired is not true; "
        "refusing to launch a run that would ignore the requested encoder variant"
    )
PY

mkdir -p "${LAUNCH_ROOT}"

variant="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["variant"]["name"])
PY
)"

hydra_args="$("${PYTHON_BIN}" - "${CONFIG}" "${ROOT}" "${RUN_GROUP}" "${CONTROL_COMMIT}" "${SONIC_COMMIT}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
root = Path(sys.argv[2])
run_group = sys.argv[3]
online_retarget_commit = sys.argv[4]
sonic_commit = sys.argv[5]
config = json.loads(cfg_path.read_text(encoding="utf-8"))
args = list(config.get("sonic_hydra", {}).get("args", []))
variant = config["variant"]["name"]
args.append("use_wandb=True")
args.append(f"wandb.wandb_group={run_group}")
args.append(f"wandb.wandb_dir={root / 'outputs' / 'wandb'}")
args.append(f"exp_var={variant}_{run_group}")
args.append(f"++online_retarget.config_path={root / cfg_path}")
args.append(f"++online_retarget.encoder_variant={variant}")
args.append("++online_retarget.contract=sonic_native_retarget")
args.append(f"++online_retarget.run_group={run_group}")
args.append(f"++online_retarget.git_sha={online_retarget_commit}")
args.append(f"++online_retarget.sonic_git_sha={sonic_commit}")
print(" ".join(shlex.quote(str(arg)) for arg in args))
PY
)"

session="sonic_native_4gpu_${RUN_GROUP}_${variant}"
session="${session//[^A-Za-z0-9_]/_}"
log_path="${LAUNCH_ROOT}/${variant}_4gpu.log"
cmd=$(cat <<EOF
set -euo pipefail
cd "${SONIC_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENTS}"
export ONLINE_RETARGET_ROOT="${ROOT}"
export ONLINE_RETARGET_CONFIG="${ROOT}/${CONFIG}"
export ONLINE_RETARGET_GIT_SHA="${CONTROL_COMMIT}"
export SONIC_GIT_SHA="${SONIC_COMMIT}"
export PYTHONPATH="${ROOT}/src:\${PYTHONPATH:-}"
export ACCELERATE_CMD="${ACCELERATE_CMD}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE}"
if [[ -n "${NCCL_IB_DISABLE}" ]]; then export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"; fi
if [[ -n "${NCCL_ALGO}" ]]; then export NCCL_ALGO="${NCCL_ALGO}"; fi
if [[ -n "${NCCL_DEBUG}" ]]; then export NCCL_DEBUG="${NCCL_DEBUG}"; fi
if [[ -n "${NCCL_DEBUG_SUBSYS}" ]]; then export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}"; fi
if [[ -n "${TORCH_CPP_LOG_LEVEL}" ]]; then export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL}"; fi
if [[ -n "${TORCH_DISTRIBUTED_DEBUG}" ]]; then export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}"; fi
echo "variant=${variant} gpus=${GPU_ASSIGNMENTS} nproc=${NPROC_PER_NODE} online_retarget_commit=${CONTROL_COMMIT} sonic_commit=${SONIC_COMMIT}"
read -r -a _accelerate_cmd <<< "\${ACCELERATE_CMD}"
"\${_accelerate_cmd[@]}" --num_processes="${NPROC_PER_NODE}" --num_machines=1 --mixed_precision="${ACCELERATE_MIXED_PRECISION}" --dynamo_backend="${ACCELERATE_DYNAMO_BACKEND}" gear_sonic/train_agent_trl.py ${hydra_args} \${HYDRA_EXTRA_ARGS:-} 2>&1 | tee -a "${log_path}"
EOF
)

"${PYTHON_BIN}" - "${LAUNCH_ROOT}/launch_manifest.json" "${RUN_GROUP}" "${CONTROL_COMMIT}" "${SONIC_COMMIT}" "${EXECUTE_SONIC_NATIVE_TRAINING}" "${CONFIG}" "${GPU_ASSIGNMENTS}" "${NPROC_PER_NODE}" "${session}" "${NCCL_SHM_DISABLE}" "${NCCL_IB_DISABLE}" "${NCCL_ALGO}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = {
    "run_group": sys.argv[2],
    "online_retarget_commit": sys.argv[3],
    "sonic_commit": sys.argv[4],
    "executed": sys.argv[5] == "1",
    "config": sys.argv[6],
    "gpus": sys.argv[7],
    "accelerate_num_processes": int(sys.argv[8]),
    "tmux_session": sys.argv[9],
    "nccl_shm_disable": sys.argv[10],
    "nccl_ib_disable": sys.argv[11],
    "nccl_algo": sys.argv[12],
}
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then
  if [[ "${NO_TMUX:-0}" == "1" ]]; then
    bash -lc "${cmd}"
  else
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "tmux session already exists: ${session}" >&2
      exit 1
    fi
    tmux new-session -d -s "${session}" "bash -lc $(printf '%q' "${cmd}")"
    printf 'started run_group=%s session=%s\n' "${RUN_GROUP}" "${session}"
  fi
else
  echo "validated formal config and wrote 4-GPU launch manifest: ${LAUNCH_ROOT}/launch_manifest.json"
  echo "set EXECUTE_SONIC_NATIVE_TRAINING=1 to start the single multi-GPU tmux session"
fi
