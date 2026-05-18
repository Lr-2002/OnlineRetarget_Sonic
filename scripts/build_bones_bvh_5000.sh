#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python}"
INDEX_CSV="${INDEX_CSV:-runs/indices/bones_sonic_index_full_v0/sonic_index.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
TRAIN_LIMIT="${TRAIN_LIMIT:-5000}"
VAL_LIMIT="${VAL_LIMIT:-1000}"
WINDOW_STRIDE="${WINDOW_STRIDE:-10}"
MAX_WINDOWS_PER_CLIP="${MAX_WINDOWS_PER_CLIP:-1}"

COMMON_ARGS=(
  -m online_retarget.cli build-sonic-windowed-jsonl
  --index-csv "$INDEX_CSV"
  --output-root "$OUTPUT_ROOT"
  --task-query ""
  --source-mode soma_bvh
  --history-frames 8
  --window-stride "$WINDOW_STRIDE"
  --max-windows-per-clip "$MAX_WINDOWS_PER_CLIP"
)

PYTHONPATH=src:. "$PYTHON_BIN" "${COMMON_ARGS[@]}" --split train --limit "$TRAIN_LIMIT"
PYTHONPATH=src:. "$PYTHON_BIN" "${COMMON_ARGS[@]}" --split val --limit "$VAL_LIMIT"
