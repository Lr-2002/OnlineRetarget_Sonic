#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-bones-bvh-mlp-5000}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/bones_bvh_mlp_5000.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/train/bones_bvh_mlp_5000traj_4090}"
MAX_STEPS="${MAX_STEPS:-250000}"
SBATCH="${SBATCH:-4096}"
CHECKPOINT="${CHECKPOINT:-}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for long training runs" >&2
  exit 1
fi

TRAIN_CMD=(
  env
  PYTHONPATH=src:.
  "$PYTHON_BIN"
  scripts/train.py
  --config "$CONFIG_PATH"
  --allow-debug-data
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --batch-size "$SBATCH"
)
if [[ -n "$CHECKPOINT" ]]; then
  TRAIN_CMD+=(--checkpoint "$CHECKPOINT")
fi

tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && ${TRAIN_CMD[*]}"
echo "started tmux session: $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
