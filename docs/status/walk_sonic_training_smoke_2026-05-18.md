# Walk SONIC Training Smoke

Date: 2026-05-18.

Purpose: start model-building and training with `walk` data from BONES/SONIC, using the current `bones_sonic` target lane.

This is a debug smoke baseline, not a formal retargeter and not a promoted M2Q quality-gated training run.

## Data

Source index:

- `runs/indices/bones_sonic_index_full_v0/sonic_index.csv`

Task filter:

- `task_query=walk`
- mirror clips excluded
- actor-heldout split by `actor_uid`, seed `17`, train/val/test ratio `0.8/0.1/0.1`

Generated samples:

- train: `runs/supervised/sonicbody_walk_train_h8_stride10_limit512/samples.jsonl`
- train manifest: `runs/supervised/sonicbody_walk_train_h8_stride10_limit512/manifest.json`
- val: `runs/supervised/sonicbody_walk_val_h8_stride10_limit128/samples.jsonl`
- val manifest: `runs/supervised/sonicbody_walk_val_h8_stride10_limit128/manifest.json`

Train manifest summary:

- candidate walk clips: 11,571
- actor split counts: train 240, val 30, test 30
- walk clip split counts: train 9,207, val 1,242, test 1,122
- selected train clips: 64
- train samples: 512
- skipped train clips: 0
- input dim: 1,547
- output dim: 29

Val summary:

- selected val clips: 16
- val samples: 128
- skipped val clips: 0
- input dim: 1,547
- output dim: 29

## Current Input/Output Contract

Fast debug mode:

- input source mode: `sonic_body_pos`
- input: SONIC NPZ `body_pos_w` relative to pelvis, 30 bodies, 8-frame history, positions plus finite differences, zero-filled robot-state side channel.
- target: SONIC NPZ `joint_pos`, 29D, same target frame.

This proves the JSONL -> MLP -> checkpoint -> offline-eval chain on walk data. It does not yet prove cross-skeleton retargeting from an external human skeleton source, because `sonic_body_pos` is already G1 target body state. The builder also supports `source_mode=soma_bvh`, but that path is slower because it reads `soma_proportional.tar`.

## Commands

Build train samples:

```bash
PYTHONPATH=src /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/inspect_bones_seed.py build-sonic-windowed-jsonl \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/bones_sonic_index_full_v0/sonic_index.csv \
  --output-root runs \
  --split train \
  --task-query walk \
  --source-mode sonic_body_pos \
  --limit 512 \
  --clip-limit 96 \
  --history-frames 8 \
  --window-stride 10 \
  --max-windows-per-clip 8 \
  --split-seed 17 \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

Build val samples:

```bash
PYTHONPATH=src /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/inspect_bones_seed.py build-sonic-windowed-jsonl \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/bones_sonic_index_full_v0/sonic_index.csv \
  --output-root runs \
  --split val \
  --task-query walk \
  --source-mode sonic_body_pos \
  --limit 128 \
  --clip-limit 48 \
  --history-frames 8 \
  --window-stride 10 \
  --max-windows-per-clip 8 \
  --split-seed 17 \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

Train smoke:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/train.py \
  --config configs/walk_sonic_mlp_debug.yaml \
  --samples-jsonl runs/supervised/sonicbody_walk_train_h8_stride10_limit512/samples.jsonl \
  --output-dir runs/train/walk_sonic_mlp_debug_smoke \
  --max-steps 20 \
  --batch-size 64 \
  --allow-debug-data
```

Val predict/eval:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/train.py \
  --config configs/walk_sonic_mlp_debug.yaml \
  --samples-jsonl runs/supervised/sonicbody_walk_val_h8_stride10_limit128/samples.jsonl \
  --output-dir runs/eval/walk_sonic_mlp_debug_val \
  --predict-only \
  --checkpoint runs/train/walk_sonic_mlp_debug_smoke/checkpoint.pt \
  --allow-debug-data
```

## Artifacts

Training:

- checkpoint: `runs/train/walk_sonic_mlp_debug_smoke/checkpoint.pt`
- report: `runs/train/walk_sonic_mlp_debug_smoke/train_report.json`
- train predictions: `runs/train/walk_sonic_mlp_debug_smoke/train_predictions.jsonl`
- train eval summary: `runs/train/walk_sonic_mlp_debug_smoke/eval/train_offline_eval/eval_summary.json`

Val:

- predictions: `runs/eval/walk_sonic_mlp_debug_val/predictions.jsonl`
- report: `runs/eval/walk_sonic_mlp_debug_val/predict_report.json`
- eval summary: `runs/eval/walk_sonic_mlp_debug_val/eval/offline_eval/eval_summary.json`

## Results

Train smoke:

- samples: 512
- model: MLP `[256, 256]`, SiLU
- steps: 20
- batch size: 64
- device: CPU
- final train MSE: `0.07438526302576065`
- train joint RMSE: `0.262895735436447`
- train action similarity: `0.8559659105862444`

Val predict/eval:

- samples: 128
- MSE: `0.06072814762592316`
- val joint RMSE: `0.24033642891431903`
- val action similarity: `0.8435393261182356`

## Limitations

- This run uses `--allow-debug-data`; it bypasses the formal M2Q curation policy gate.
- The input is `sonic_body_pos`, already G1 target body state, so it is not yet a general human-skeleton-to-G1 retargeter.
- The validation split is actor-heldout, but the task is a small smoke subset and not a formal benchmark.
- The model ran on CPU because `.venv_sim` reports `torch.cuda.is_available() == False`.
- WandB is disabled in `configs/walk_sonic_mlp_debug.yaml`.
- No tmux session was used because this was a short debug run, not a long training run.

## Next Step

Move from `source_mode=sonic_body_pos` to `source_mode=soma_bvh` or a real SMPL/human skeleton source builder, then train/evaluate with the same walk split. That will test actual retargeting rather than target-state reconstruction.
