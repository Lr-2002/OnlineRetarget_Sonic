# LR-177 A0 SOMA Motionlib Usage

Date: 2026-06-01.

## Scope

LR-177 adds four supervised A0 SOMA motionlib configs. They all train the same kin-only target:
G1 joint position and velocity command windows plus G1 root position/orientation diagnostics. The
comparison is only the source topology and skeleton-conditioning choice:

- `uniform` versus `proportional` SOMA motionlib source.
- frozen Skeleton Geometry AE `z_skel` conditioning versus no skeleton encoder.

Do not change objectives, loss weights, target fields, or source/target data paths when launching
these runs.

## Config Matrix

| Run family | Topology | Config | Skeleton input | Model input dim | Target dim |
|---|---|---|---|---:|---:|
| Frozen Skeleton Geometry AE | Uniform SOMA | `configs/sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json` | `x_skel=104` encoded to frozen `z_skel=64` | 904 | 670 |
| Frozen Skeleton Geometry AE | Proportional SOMA | `configs/sonic_kin_soma_motionlib_a0_frozen_ae_proportional_4gpu.json` | `x_skel=104` encoded to frozen `z_skel=64` | 904 | 670 |
| No skeleton encoder | Uniform SOMA | `configs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_uniform_4gpu.json` | zero-width skeleton feature | 840 | 670 |
| No skeleton encoder | Proportional SOMA | `configs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_proportional_4gpu.json` | zero-width skeleton feature | 840 | 670 |

All four configs use `training_lane: soma_motionlib_kin_only`, one 4-GPU DDP job, `batch_frames:
4096`, `max_steps: 1000000`, and the same `g1_kin` target decoder. Uniform configs read
`soma_uniform_filtered_v1`; proportional configs read `soma_proportional_filtered_v1`. All write
under `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs`.

## Feature Dimensions

The expected feature guard runs before model construction in both dry-run and formal paths.

- Motion token: 840 dims for 10 future frames of SOMA joint local positions plus SOMA root
  orientation.
- Frozen-AE skeleton conditioning: raw static geometry `x_skel=104`, frozen AE latent
  `z_skel=64`, model input `840 + 64 = 904`.
- No-skeleton-encoder ablation: `x_skel=0`, `z_skel=0`, model input `840`.
- Target: 670 dims, computed as 10 future frames of 29 G1 joint positions, 29 G1 joint
  velocities, 3 root-position dims, and 6 root-orientation dims.

Frozen-AE configs require the configured checkpoint, normalization file, and registry CSV before
the dry-run or formal run can start. No-skeleton-encoder configs must not create a `skeleton_ae`
manifest block.

## Dry-Run

Use dry-run to verify runtime guardrails, data paths, feature dimensions, normalization stats,
manifest writing, and validation metric logging without optimization.

```bash
export CONFIG=configs/sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json
export KIN_RUN_GROUP=lr177_a0_frozen_ae_uniform_dryrun_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=src:. /workspace/isaaclab/_isaac_sim/python.sh -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  scripts/train_sonic_kin_skeleton_ae.py \
  --config "${CONFIG}" \
  --dry-run \
  --wandb-mode disabled
```

Change only `CONFIG` and `KIN_RUN_GROUP` for the other three configs. The official 4-GPU configs
require `WORLD_SIZE=4`, so run dry-runs through `torch.distributed.run`; a single direct Python
process is only suitable for local fixture configs that lower the runtime GPU requirement.

Dry-run outputs include:

- `manifest.json`
- `dry_run_summary.json`
- `stats/normalization.pt`
- `cache/skeleton_embedding_cache.pt` for frozen-AE configs only
- `logs/a0_stage_trace/*` when stage tracing is enabled

## Formal Launch

Formal launches should use the guarded tmux launcher:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json \
KIN_RUN_GROUP=lr177_a0_frozen_ae_uniform_$(date -u +%Y%m%dT%H%M%SZ) \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

Optional smoke overrides:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_proportional_4gpu.json \
KIN_RUN_GROUP=lr177_a0_no_encoder_prop_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
MAX_STEPS=2000 \
WANDB_MODE=offline \
DISABLE_VISUAL_VALIDATION=1 \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

Before the launcher starts tmux, it checks that:

- the config is `soma_motionlib_kin_only`;
- `NPROC_PER_NODE` matches `required_gpu_count`;
- required input directories exist;
- CUDA and `torch.distributed` are available;
- the OnlineRetarget checkout is clean and equal to the latest upstream commit.

Do not stop, restart, or modify remote training sessions that are already running from
`b0af812e2cbaa2d9196b95625db6bda1228998a0`. New launches should use a fresh `KIN_RUN_GROUP`.

## Metrics And Supplemental MPJPE

The built-in A0 evaluation metric is `g1_joint_pos_rmse_rad`, logged as
`train/g1_joint_pos_rmse_rad` and `validation/g1_joint_pos_rmse_rad`. It is a G1 joint-angle
command RMSE over the 29-DoF joint position target window.

The shared metric registry lives in `src/online_retarget/metrics.py`. Online logging calls
`compute_online_metrics(...)`, and offline reports call the same registry through
`online-retarget offline-eval` / `scripts/inspect_bones_seed.py offline-eval`. For LR-225-style
prediction artifacts, run:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py offline-eval \
  --input-jsonl runs/train/<run_name>/train_predictions.jsonl \
  --output-root runs \
  --run-name lr177_a0_eval \
  --metric g1_joint_pos_rmse_rad \
  --metric joint_velocity_rmse \
  --metric root_position_rmse \
  --metric root_rot6d_rmse \
  --metric global_body_position_error \
  --metric root_relative_body_position_error \
  --metric joint_rotation_error \
  --metric joint_velocity_error \
  --metric joint_acceleration_error \
  --metric joint_jump_count \
  --metric joint_limit_proximity_count \
  --metric self_collision_count \
  --metric floating_guard \
  --metric cross_ratio_guard \
  --metric phc_failure_guard \
  --metric mpjpe \
  --metric w_mpjpe
```

The evaluator writes JSON and CSV reports under `runs/eval/lr177_a0_eval/`. Each requested metric
has metadata plus `<metric>_status` and `<metric>_reason` fields, so unavailable root/body metrics
are explicit instead of zero-filled.

The shared registry also exposes paper/NMR/PHC labels and formulas: `E_g_mpbpe`,
`E_mpbpe`, `root_relative_MPJPE`, full-joint `joint_rotation_error`, PHC-style
velocity/acceleration error terms, NMR `joint_jump_count` using `abs(delta_q) > 0.5 rad`, MuJoCo
`self_collision_count`, and the `floating_guard`, `cross_ratio_guard`, and `phc_failure_guard`
gates. A0 artifacts may still report these as `unavailable` when the JSONL rows do not include
full-joint rotations, MuJoCo contact counts/pairs, lowest-foot height or self-intersection ratio,
PHC avg body-joint distance, or joint limit arrays. The G1 FK self-collision scanner proxy is not
substituted for MuJoCo contacts.
If a row supplies `method_id`, metrics outside that metric's `method_coverage` are emitted as
`not_applicable`; missing/unavailable/blocked/not-applicable metrics are tracked in availability
counts and are not summarized as numeric 0.

`body_position_mpjpe` is not produced by these A0 training targets because the target tensor is a
joint-angle command window, not an FK/body-position target. Any report that compares body-position
MPJPE for these four run families must include a separate
`body_position_mpjpe_supplemental.json` artifact for each family:

- `A0_frozen_skeleton_ae_uniform`
- `A0_frozen_skeleton_ae_proportional`
- `A0_no_skeleton_encoder_uniform`
- `A0_no_skeleton_encoder_proportional`

The supplemental artifact should record the evaluator command, control-repo SHA, FK model or body
position source, sample split/count, units, and the resulting body-position MPJPE values. Do not
rename or treat `g1_joint_pos_rmse_rad` as MPJPE.

Current blockers for true body-position MPJPE and W-MPJPE are: paired predicted-vs-target G1
body/link position arrays, pinned FK link order, position units in meters, root-alignment semantics,
and body/link weights for W-MPJPE.

`root_relative_body_position_error` additionally needs paired root position arrays or an explicit
`root_body_index` / `root_link_index` so the metric can perform body-root subtraction rather than
reusing global positions.

## Skeleton Geometry AE Pretraining

The frozen-AE configs consume an existing AE checkpoint and normalization stats. To rebuild that
artifact family, use:

```bash
PYTHONPATH=src:. python3 scripts/train_skeleton_geometry_ae.py \
  --config configs/skeleton_geometry_ae_all_skeletons.json \
  --dry-run
```

Remove `--dry-run` for a formal AE pretraining run in a torch environment. The expected AE geometry
shape is 104, and the latent dimension is 64.
