# BONES BVH Training Scaffold

Date: 2026-05-18.

Purpose: establish the first NMR-inspired but BONES-native training path:

```text
SOMA proportional BVH FK history + velocities + actor morphology + robot-state side channel
  -> BONES-SONIC G1 joint_pos
```

This is not a reproduction of the released NMR code. NMR's useful design input is the supervised temporal mapping, artifact-first evaluation, and later kinematic-to-physics target provenance hierarchy. The current implementation starts with BONES/SONIC kinematic targets and keeps simulator-refined targets as a later lane.

## Code Added

- `src/online_retarget/models/registry.py`
  - config-driven model builder
  - aliases: `mlp`, `temporal_mlp`, `tf`, `transformer`, `temporal_transformer`, `fm`, `flow_matching`
- `src/online_retarget/models/temporal.py`
  - `TemporalTransformerRetargeter`
  - `FlowMatchingRetargeter`
- `src/online_retarget/models/mlp.py`
  - configurable activation
- `scripts/train.py`
  - DDP-aware runtime setup
  - `DistributedSampler`
  - configurable model family
  - configurable loss terms
  - configurable evaluation metric list/thresholds
  - rank0-only checkpoint, prediction, eval, and WandB output
- `scripts/train_tmux.sh`
  - supports `NPROC_PER_NODE>1` through `torchrun`
- configs:
  - `configs/bones_bvh_mlp_debug.yaml`
  - `configs/bones_bvh_transformer_debug.yaml`
  - `configs/bones_bvh_flow_debug.yaml`

## Data Smoke

Source data is read-only under `/home/user/data/motion_data`.

Generated derived samples:

- train: `runs/supervised/somabvh_walk_train_h8_stride10_limit128/samples.jsonl`
- train manifest: `runs/supervised/somabvh_walk_train_h8_stride10_limit128/manifest.json`
- val: `runs/supervised/somabvh_walk_val_h8_stride10_limit64/samples.jsonl`
- val manifest: `runs/supervised/somabvh_walk_val_h8_stride10_limit64/manifest.json`

Train sample summary:

- task: `walk`
- source mode: `soma_bvh`
- split: actor-heldout by `actor_uid`, seed `17`
- candidate walk clips: `11,571`
- selected clips: `32`
- samples: `128`
- input dim: `1547`
- output dim: `29`

Val sample summary:

- selected clips: `16`
- samples: `64`
- input dim: `1547`
- output dim: `29`

Data build bottleneck: reading `soma_proportional.tar` is slow. Before formal-scale training, add a repo-local derived FK cache or pre-extracted sample shard under `runs/`/`outputs/`; do not write into `/home/user/data/motion_data`.

## Training Smoke

Command family:

```bash
PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/train.py \
  --config configs/bones_bvh_mlp_debug.yaml \
  --samples-jsonl runs/supervised/somabvh_walk_train_h8_stride10_limit128/samples.jsonl \
  --output-dir runs/train/bones_bvh_mlp_smoke_128x30 \
  --max-steps 30 \
  --batch-size 64 \
  --allow-debug-data
```

Result:

- checkpoint: `runs/train/bones_bvh_mlp_smoke_128x30/checkpoint.pt`
- train report: `runs/train/bones_bvh_mlp_smoke_128x30/train_report.json`
- train eval summary: `runs/train/bones_bvh_mlp_smoke_128x30/eval/train_offline_eval/eval_summary.json`
- model: `temporal_mlp`, hidden dims `[512, 512, 256]`
- device: `cuda:0` on one RTX 4090
- steps: `30`
- final train MSE: `0.07579446583986282`
- train joint RMSE: `0.2641241654520922`
- train action similarity: `0.8979783491422182`
- train predicted joint jump rate: `0.0`

Val predict/eval:

- predictions: `runs/eval/bones_bvh_mlp_smoke_val64/predictions.jsonl`
- predict report: `runs/eval/bones_bvh_mlp_smoke_val64/predict_report.json`
- eval summary: `runs/eval/bones_bvh_mlp_smoke_val64/eval/offline_eval/eval_summary.json`
- val MSE: `0.04882701858878136`
- val joint RMSE: `0.21285850783511157`
- val action similarity: `0.919354258548973`
- val predicted joint jump rate: `0.0`

This is a debug run, not a formal benchmark. It uses `--allow-debug-data` and only 192 total windows.

## Model Path Checks

Both non-MLP families ran one-step training on the same samples:

- Transformer path: `runs/train/bones_bvh_transformer_pathcheck`
- Flow matching path: `runs/train/bones_bvh_flow_pathcheck`

These runs only prove config/training path viability; their metrics are not quality claims.

## DDP Check

Current machine reports one RTX 4090. Multi-GPU could not be exercised directly, but the DDP path was smoke-tested with two CPU ranks:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/torchrun \
  --standalone --nproc-per-node 2 scripts/train.py \
  --config configs/bones_bvh_mlp_debug.yaml \
  --samples-jsonl runs/supervised/somabvh_walk_train_h8_stride10_limit128/samples.jsonl \
  --output-dir runs/train/bones_bvh_mlp_ddp_cpu_pathcheck \
  --max-steps 1 \
  --batch-size 16 \
  --allow-debug-data
```

Result:

- DDP initialized with `world_size=2`, backend `gloo`.
- Rank0 wrote checkpoint, train report, predictions, and eval summary.
- This validates the code path. Real 8-GPU training should use `torchrun --nproc-per-node 8` on a machine exposing 8 CUDA devices.

## Verification

- `compileall`: passed for `scripts/train.py`, `src/online_retarget/models`, and `src/online_retarget/evaluation.py`.
- `bash -n scripts/train_tmux.sh`: passed.
- `unittest discover -s tests`: `129` tests passed.
- `ruff`: not run because `/home/user/repos/GR00T-WholeBodyControl/.venv_sim` does not have `ruff` installed.

## Next Steps

1. Add derived BVH-FK cache/shards to avoid rereading the 276GB tar for every sample build.
2. Build a larger actor-heldout train/val/test set with a named debug policy first, then promote through M2Q when quality gates are ready.
3. Add full-sequence eval artifacts, not only one-window joint RMSE, so joint jump and smoothness metrics become meaningful.
4. Add optional G1 FK/body output or FK reconstruction for MPJPE/contact metrics on predictions.
5. Once the data path is faster, start tmux training with WandB enabled and commit SHA captured.
