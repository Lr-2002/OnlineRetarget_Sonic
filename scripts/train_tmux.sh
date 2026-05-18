#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baseline_mlp.yaml}"
SESSION_NAME="${2:-online-retarget-train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for long training runs" >&2
  exit 1
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  TRAIN_CMD="PYTHONPATH=src torchrun --standalone --nproc-per-node '$NPROC_PER_NODE' scripts/train.py --config '$CONFIG_PATH' ${EXTRA_ARGS[*]}"
else
  TRAIN_CMD="PYTHONPATH=src '$PYTHON_BIN' scripts/train.py --config '$CONFIG_PATH' ${EXTRA_ARGS[*]}"
fi

tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && $TRAIN_CMD"

echo "started tmux session: $SESSION_NAME"
