# OnlineRetarget Project Goal

## Goal

OnlineRetarget 当前目标是：以 Sonic 代码和数据语义为基准，只训练 Kinematics 路径，不训练、不启动、不依赖 `g1_dyn` Dynamics Decoder。当前执行面是 strict supervised `soma_motionlib_kin_only`，从 paired SOMA motionlib features 监督到 G1 `g1_kin` targets。

目标路径为：

```text
SOMA / BVH source motion
+ source skeleton / morphology feature
    -> OnlineRetarget SOMA encoder baseline
    -> supervised Sonic g1_kin target fields
    -> G1 kinematic retarget motion
```

## Primary Objective

训练并比较当前 strict supervised configs，在 paired SOMA motionlib -> G1 kinematics target 上降低 kinematics loss：

- LR-280 kin/walk data-package final smoke targets:
  - `sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu`
  - `sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu`
- LR-273/LR-284 proportional treatment:
  - `sonic_kin_soma_motionlib_proportional_4gpu`
- LR-274 matched proportional loss-off baseline:
  - `sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu`

Primary target:

- Sonic `g1_kin` target fields:
  - `command_multi_future_nonflat`: 29 DoF joint position + 29 DoF joint velocity。
  - `root_pos_w_mf`: future root position target。
  - `root_rot_w_mf`: future root orientation 6D。

Explicit non-target:

- 不使用 `g1_dyn`。
- 不使用 Sonic PPO / Isaac rollout 作为本阶段训练入口。
- 不把 `body_pos_w` / `body_quat_w` 当作 deployable source input。

## Data Contract

Source input 必须来自部署时可获得的 human side 信息：

- SOMA / BVH motionlib 中的 `soma_joints`、`soma_root_quat`、skeleton morphology。
- BVH/SOMA 先按 Sonic motionlib 逻辑从 120Hz 对齐到 50Hz target timeline。

Target supervision 来自 paired G1 robot motionlib：

- `dof` -> joint position。
- finite difference `dof` -> joint velocity。
- `root_rot` -> anchor-relative 6D orientation。

Training、validation、inference 必须共用同一套 feature packing contract。

## Experiments

- `kin_walk_data_package_a_only`：pinned LR-280 paired SOMA motionlib kin/walk package，开启 temporal-consistency auxiliary loss，关闭 A/B overlap。
- `kin_walk_data_package_a_plus_b`：同一 pinned data package / split / model / eval cohort，开启 temporal-consistency 和 A/B overlap auxiliary loss。
- `proportional_loss_on`：broader proportional SOMA motionlib treatment，开启 temporal-consistency 和 A/B overlap auxiliary loss。
- `proportional_loss_off`：matched broader proportional SOMA motionlib baseline，关闭 temporal/A-B auxiliary loss。

每个 formal run 使用单个 4-GPU DDP job。远程训练必须先确认代码已 commit、已 push、远程 checkout 最新，并在 W&B / manifest 记录 OnlineRetarget git SHA、config、baseline、motionlib 路径、data-package digest（如适用）和 run group。

## Validation

训练 loop 内置 validation，不依赖训练后手动 copy。

Weekly walk-data outcome target: online retargeter walk should be non-jittery, non-drifting, avoid self-collision, and be at least similar to the source walk data.

Visual validation 约每 2k steps 触发一次，每次 8 个 4 秒视频，并上传 W&B。每个视频包含：

1. Source SOMA/BVH capsule motion。
2. Dataset G1 target FK motion。
3. Inference G1 FK motion。

三路视频必须按同一物理时间和 50Hz target timeline 对齐。

## Acceptance Criteria

- Formal configs keep `training_lane=soma_motionlib_kin_only`, target `g1_kin` only, and contain no PPO / Isaac rollout / reward / episode-length / `g1_dyn` training surface.
- SOMA motionlib 能被直接读入并生成 kin-only supervised batch；LR-280 data-package configs must pin and validate the paired kin/walk package digest before training.
- 每个 config 都能作为单个 4-GPU tmux training job 启动；不要再把当前需求拆成 A1 / A2 / B1 / B2 四个 1-GPU runs。
- Weekly walk-data outcome target is met: online retargeter walk should be non-jittery, non-drifting, avoid self-collision, and be at least similar to the source walk data.
- W&B 中能看到 loss、git SHA、config，以及定期 visual validation videos。
- 最终根据 kin loss、joint RMSE、velocity RMSE、body MPJPE、anchor orientation RMSE、visual validation sliding/jitter cases 和 inference latency 选择下一步主线。
