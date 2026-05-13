#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/baseline_mlp.yaml}"
SESSION_NAME="${2:-online-retarget-train}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for long training runs" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd $(pwd) && PYTHONPATH=src python scripts/train.py --config '$CONFIG_PATH'"

echo "started tmux session: $SESSION_NAME"
