# BONES Sonic TXT Filtered 8-GPU Training Status

Date: 2026-05-19  
Run type: `bones_sonic/data.txt` filtered baseline, not promotable M2Q  
Remote host: `root@106.14.35.26:1050`  
Remote training directory: `/root/OnlineRetarget_train_9824b7e_20260519`

## Current Result

The first real 8-GPU baseline run completed 2000 steps with a finite loss curve.
The failed first attempt exposed seven non-finite observation rows in the
derived supervised JSONL; the training entry point now filters those rows at
runtime and records the dropped samples in the train report.

Traceability:

- Code commit: `4db2e51b2cb64f5d55efde127d8598d703f1c24c`
- Training command: `torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/bones_bvh_token_transformer_sonic_txt_filtered.yaml --wandb-mode online --output-dir runs/train/bones_bvh_token_transformer_sonic_txt_filtered_8gpu --allow-debug-data`
- WandB: project `online-retarget`, run id `vua8pqzo`, run name `bones_bvh_token_transformer_sonic_txt_filtered`
- Checkpoint: `/root/OnlineRetarget_train_9824b7e_20260519/runs/train/bones_bvh_token_transformer_sonic_txt_filtered_8gpu/checkpoint.pt`
- Train report copied locally: `runs/remote/bones_bvh_token_transformer_sonic_txt_filtered_8gpu/train_report.json`
- Eval summary copied locally: `runs/remote/bones_bvh_token_transformer_sonic_txt_filtered_8gpu/eval/train_offline_eval/eval_summary.json`

## Data

The filtered list is from:

- `/home/user/data/motion_data/bones_sonic/data.txt`

Derived artifacts are repo-local and do not modify the source data:

- Curated index: `runs/curated/bones_sonic_txt_filtered_pair_s17/curated_index.csv`
- Curated report: `runs/curated/bones_sonic_txt_filtered_pair_s17/curated_report.json`
- Train JSONL: `runs/filtered_sonic_txt/supervised/train_merged-quality-action_30b_h8_stride30_limit999999/samples.jsonl`
- Val JSONL: `runs/filtered_sonic_txt/supervised/val_merged-quality-action_30b_h8_stride30_limit999999/samples.jsonl`

Counts:

- Train JSONL before runtime finite filter: 87,393 samples
- Runtime-filtered train samples: 87,386
- Dropped train samples: 7, all due to `observation_nonfinite`
- Val JSONL: 12,758 samples
- Input dimension: 1547
- Output dimension: 29 G1 joint positions

This run uses `allow_debug_data: true` because there is no promotable policy
audit. Do not describe it as formal M2Q training.

## Model

Configured model family: `token_transformer`.

Current input layout:

- Source motion token input: BVH FK positions and velocities over 8 history frames.
- Skeleton/morphology input: actor numeric measurements from the BONES metadata.
- Previous robot state input: previous target G1 joints from the supervised sample.
- Output: one 29D G1 joint-position vector.

Current tokenized transformer structure:

- MLP motion encoder maps the flattened source-motion window to a 128D motion token.
- MLP skeleton encoder maps morphology values to a 128D skeleton token.
- MLP state encoder maps previous G1 target state to a 128D state token.
- Transformer encoder processes `[skeleton, motion]` memory tokens.
- Transformer decoder uses a learned query token plus optional previous-state token.
- MLP head decodes the query output to the 29D G1 pose.
- Auxiliary reconstruction heads keep motion, skeleton, and state tokens inspectable.

Configured loss:

- Supervised L1 on G1 joint positions: weight `1.0`
- Skeleton token reconstruction MSE: weight `0.05`
- Motion token reconstruction MSE: weight `0.01`
- State token reconstruction MSE: weight `0.05`
- Latent alignment: disabled

## Training Evidence

The full 8-GPU run was launched inside tmux session
`ort_sonic_txt_filtered_8gpu` and completed. The important log evidence:

- `world_size=8`
- `git_sha=4db2e51b2cb64f5d55efde127d8598d703f1c24c`
- `git_dirty=False`
- `sample_filter.dropped_count=7`
- Finite losses through step 2000:
  - step 1: `541.6504516601562`
  - step 80: `20.602609634399414`
  - step 900: `5.209070682525635`
  - step 1600: `2.805567741394043`
  - step 2000: `1.9887855052947998`

Offline train-set evaluation from `train_report.json`:

- `joint_mae`: `1.5858781151081855`
- `joint_mse`: `13.984658314127156`
- `joint_rmse`: `2.2011191881506864`
- `max_joint_abs_error`: `6.184344358553256`
- `action_similarity`: `0.9948072318817983`
- `predicted_joint_jump_rate`: `0.0`
- `target_joint_jump_rate`: `0.0`

Interpretation caveat: this is train-set offline evaluation over one-frame
joint predictions. It is useful for checking that the pipeline trains and the
model fits the supervised mapping, but it is not yet a held-out validation
claim and not a physics/simulator tracking claim.

## Codebase Supports Now

Data and curation:

- BONES pair index construction and split/curation utilities.
- BVH FK windowed supervised JSONL builder for `soma_proportional -> G1`.
- SONIC-specific NPZ windowed builder and review utilities.
- Quality scanners for source BVH, G1 target CSVs, pair consistency, and merged quality actions.
- Review clip export paths for 3D capsule videos.

Training:

- Config-driven `scripts/train.py`.
- DDP via `torchrun`.
- WandB online/offline/disabled modes.
- Model registry with `temporal_mlp`, `temporal_transformer`, `token_transformer`, and `flow_matching`.
- Runtime filtering for non-finite supervised samples.
- Train report with git SHA, git dirty state, quality gate, model config, loss config, sample filter, checkpoint paths, and WandB metadata.

Evaluation:

- Offline JSONL evaluation for joint MAE/MSE/RMSE, max joint error, action cosine similarity, joint velocity RMSE, joint jump rate.
- MPJPE and contact artifact metrics when predicted/target body positions are present.
- Per-sample metrics CSV, failure manifest CSV, aggregate summary by actor/category/package/quality flag.

## Known Gaps

- The current run is not promotable M2Q because `quality_policy_audit` is empty and `allow_debug_data` is true.
- Validation and test JSONL evaluation have not been run for this checkpoint yet.
- Current train/eval predictions are one-frame samples, so `joint_velocity_rmse` and jump metrics are not meaningful for long-horizon temporal behavior.
- The training loader loads the full 2.7GB JSONL independently in each rank. It works, but startup is inefficient and should become a sharded or memory-mapped dataset before larger experiments.
- The current output is direct G1 joint position, not simulator-executed data.
- No Isaac Lab tracking rollout is wired into this run.

## Next Milestones

1. Run predict/eval on the held-out val JSONL with this checkpoint.
2. Add a sharded JSONL dataset path so DDP startup does not duplicate full-file parsing on every rank.
3. Add sequence-level validation exports so temporal metrics and visual review are meaningful.
4. Promote a quality policy only after policy audit and manual review criteria are satisfied.
5. Compare `token_transformer` against compact MLP/TCN and flow matching under the same filtered split.
