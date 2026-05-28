#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-${SCRIPT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${ACCELERATE_CMD:-}" ]]; then
  if [[ -n "${ACCELERATE_BIN:-}" ]]; then
    ACCELERATE_CMD="${ACCELERATE_BIN} launch"
  else
    ACCELERATE_CMD="${PYTHON_BIN} -m accelerate.commands.launch"
  fi
fi
RUN_GROUP="${RETARGET_RUN_GROUP:-sonic_native_retarget_$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${ROOT}/outputs/sonic_native_retarget_runs/${RUN_GROUP}/_launcher}"
GIT_FETCH_TIMEOUT_SECONDS="${GIT_FETCH_TIMEOUT_SECONDS:-60}"
EXECUTE_SONIC_NATIVE_TRAINING="${EXECUTE_SONIC_NATIVE_TRAINING:-0}"
CHECK_SONIC_PATHS="${CHECK_SONIC_PATHS:-${EXECUTE_SONIC_NATIVE_TRAINING}}"

cd "${ROOT}"

if [[ -z "${CONFIG:-}" && "${ALLOW_HISTORICAL_A_B_4X1GPU:-0}" != "1" ]]; then
  echo "A1/A2/B1/B2 4x1-GPU launching is historical for this requirement." >&2
  echo "Use scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh with CONFIG=configs/sonic_kin_soma_motionlib_uniform_4gpu.json or configs/sonic_kin_soma_motionlib_proportional_4gpu.json." >&2
  exit 1
fi

if [[ -n "${CONFIG:-}" ]]; then
  if [[ "${CONFIG}" == *"sonic_kin_only_soma_encoder_"* || "${CONFIG}" == *"sonic_kin_soma_motionlib_"*"_4gpu.json" ]]; then
    echo "active kin-only SOMA encoder baselines must run as one 4-GPU job; use scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh" >&2
    exit 1
  fi
  CONFIGS=("${CONFIG}")
else
  CONFIGS=(
    "configs/sonic_native_retarget_a1_concat_1gpu.json"
    "configs/sonic_native_retarget_a2_film_contact_1gpu.json"
    "configs/sonic_native_retarget_b1_adapter_1gpu.json"
    "configs/sonic_native_retarget_b2_expert_1gpu.json"
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
    echo "${label} could not fetch ${upstream}; refusing to train without a latest-code check" >&2
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
"${PYTHON_BIN}" scripts/validate_sonic_native_retarget_config.py "${VALIDATE_ARGS[@]}" "${CONFIGS[@]}"

SONIC_ROOT="$("${PYTHON_BIN}" - "${CONFIGS[0]}" <<'PY'
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

mkdir -p "${LAUNCH_ROOT}"

declare -a SESSIONS=()
declare -a COMMANDS=()
for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  if [[ ! -f "${cfg}" ]]; then
    echo "missing config: ${cfg}" >&2
    exit 1
  fi
  if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then
    "${PYTHON_BIN}" - "${cfg}" <<'PY'
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
  fi
  variant="$("${PYTHON_BIN}" - "${cfg}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["variant"]["name"])
PY
)"
  hydra_args="$("${PYTHON_BIN}" - "${cfg}" "${ROOT}" "${RUN_GROUP}" "${CONTROL_COMMIT}" "${SONIC_COMMIT}" <<'PY'
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
  session="sonic_native_${RUN_GROUP}_${variant}"
  session="${session//[^A-Za-z0-9_]/_}"
  log_path="${LAUNCH_ROOT}/${variant}.log"
  cmd=$(cat <<EOF
set -euo pipefail
cd "${SONIC_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export ONLINE_RETARGET_ROOT="${ROOT}"
export ONLINE_RETARGET_CONFIG="${ROOT}/${cfg}"
export ONLINE_RETARGET_GIT_SHA="${CONTROL_COMMIT}"
export SONIC_GIT_SHA="${SONIC_COMMIT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export ACCELERATE_CMD="${ACCELERATE_CMD}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "variant=${variant} gpu=${gpu} online_retarget_commit=${CONTROL_COMMIT} sonic_commit=${SONIC_COMMIT}"
read -r -a _accelerate_cmd <<< "\${ACCELERATE_CMD}"
"\${_accelerate_cmd[@]}" --num_processes=1 gear_sonic/train_agent_trl.py ${hydra_args} 2>&1 | tee -a "${log_path}"
EOF
)
  COMMANDS+=("${cmd}")

  if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "tmux session already exists: ${session}" >&2
      exit 1
    fi
    tmux new-session -d -s "${session}" "bash -lc $(printf '%q' "${cmd}")"
    SESSIONS+=("${session}")
  fi
done

"${PYTHON_BIN}" - "${LAUNCH_ROOT}/launch_manifest.json" "${RUN_GROUP}" "${CONTROL_COMMIT}" "${SONIC_COMMIT}" "${EXECUTE_SONIC_NATIVE_TRAINING}" "${CONFIGS[@]}" -- "${GPUS[@]}" "${SESSIONS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_group = sys.argv[2]
control_commit = sys.argv[3]
sonic_commit = sys.argv[4]
execute = sys.argv[5] == "1"
sep = sys.argv.index("--")
configs = sys.argv[6:sep]
tail = sys.argv[sep + 1 :]
gpu_count = len(configs)
manifest = {
    "run_group": run_group,
    "online_retarget_commit": control_commit,
    "sonic_commit": sonic_commit,
    "configs": configs,
    "gpus": tail[:gpu_count],
    "tmux_sessions": tail[gpu_count:],
    "executed": execute,
}
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then
  printf 'started run_group=%s sessions=%s\n' "${RUN_GROUP}" "${SESSIONS[*]}"
else
  echo "validated formal configs and wrote launch manifest: ${LAUNCH_ROOT}/launch_manifest.json"
  echo "set EXECUTE_SONIC_NATIVE_TRAINING=1 to start tmux sessions after committing this repo"
fi
