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

## Training-Time W&B Metric And Visual Logging

New LR-177 A0 launches expose a `remote_logging` contract in `manifest.json` and
`dry_run_summary.json`. This contract is non-invasive to training:

- it does not change the loss, objective, optimizer, checkpoint selection, or DDP collectives;
- only rank 0 initializes W&B and uploads visual artifacts;
- `g1_joint_pos_rmse_rad` is logged as joint-angle RMSE in radians under the registry schema
  `humanoid_retarget_eval.v1`;
- `body_position_mpjpe` is emitted as `missing` until a separate
  `body_position_mpjpe_supplemental.json` evaluator artifact exists;
- simulator policy metrics such as `policy_success` are `not_applicable` for this kin-only
  supervised training lane.

Expected scalar keys include:

- `train/g1_joint_pos_rmse_rad`
- `validation/g1_joint_pos_rmse_rad`
- `metric_registry/train/g1_joint_pos_rmse_rad`
- `metric_registry/validation/g1_joint_pos_rmse_rad`

Visual upload is controlled by both `visual_validation` and `remote_logging`:

- `visual_validation.enabled`
- `visual_validation.every_steps` or `visual_validation.every_minutes`
- `visual_validation.num_videos`
- `visual_validation.duration_sec`
- `visual_validation.wandb_upload`
- `remote_logging.enabled`
- `remote_logging.visual_upload`
- `remote_logging.visual_every_n_steps`
- `remote_logging.num_visual_samples`
- `remote_logging.max_video_sec`
- `--wandb-mode online|offline|disabled`

When the same visual knobs are present in both blocks, `remote_logging` is the override surface and
is applied to the effective `visual_validation` runtime config before cadence checks, sample
selection, duration selection, and W&B upload gating. The manifest records this mapping under
`remote_logging.visuals.controls_apply_to` so a probe cannot report one visual cadence while the
trainer runs another.

When visual validation runs, `visual_validation/step_*/summary.json` records W&B video keys and
artifact specs. Artifact metadata includes run id, run group, variant, step, checkpoint path,
control/source commits, sequence id, local clip metadata path, and whether the render was the
acceptance backend. If SomaMesh or IsaacLab assets are absent and the capsule fallback is used, the
metadata is labeled `fallback_not_final_somamesh`; that fallback is not acceptable as a final
SomaMesh render.

## Remote Logging Probe Contract

MLOps can validate the W&B metric/visual contract without launching optimization or a heavy render
job by using the launcher dry-run probe:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_uniform_4gpu.json \
KIN_RUN_GROUP=lr177_a0_remote_logging_probe_$(date -u +%Y%m%dT%H%M%SZ) \
DRY_RUN=1 \
REMOTE_LOGGING_PROBE=1 \
WANDB_MODE=offline \
DISABLE_VISUAL_VALIDATION=0 \
STAGE_TRACE=1 \
NO_TMUX=1 \
scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```

The probe should produce `manifest.json` and `dry_run_summary.json` under the configured
`output_dir`. With `REMOTE_LOGGING_PROBE=1` and effective W&B mode not disabled, rank 0 also logs
the registry scalar probe payload and uploads `dry_run_summary.json` as an
`online_retarget_remote_logging_probe` artifact. Reviewers should check:

- `dry_run_summary.json.remote_logging.schema_version == online_retarget.remote_training_logging.v1`
- `remote_logging.non_invasive.changes_training_loss == false`
- `remote_logging.non_invasive.changes_ddp_collectives == false`
- `remote_logging.metric_registry_results` contains measured `g1_joint_pos_rmse_rad`, missing
  `body_position_mpjpe`, and not-applicable `policy_success`
- `remote_logging.visuals.wandb_upload_enabled` matches the effective W&B mode and config flags
- `remote_logging.visuals.expected_artifact_template` matches the W&B artifact naming contract

For an online W&B upload probe after code review, set `WANDB_MODE=online` and keep
`REMOTE_LOGGING_PROBE=1`. That validates scalar logging and manifest/dry-run W&B config. It still
does not prove final SomaMesh visual upload unless the SomaMesh source assets, IsaacLab renderer,
and `visual_validation` cadence are available and a real visual validation step is executed.
If `A0_DDP_PROBE_ONLY=1` is used with `REMOTE_LOGGING_PROBE=1`, W&B upload is intentionally skipped
and `dry_run_summary.json.remote_logging_probe_wandb.reason` is
`ddp_probe_only_has_no_metric_results`; that DDP-only path has no validation metrics to upload.

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
