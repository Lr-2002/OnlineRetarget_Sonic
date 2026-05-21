# OnlineRetarget Project Goal

## Goal

OnlineRetarget 当前目标是：以 Sonic 代码和数据语义为基准，只训练 Kinematics 路径，不训练、不启动、不依赖 `g1_dyn` Dynamics Decoder。

目标路径为：

```text
SOMA / BVH source motion
+ source skeleton / morphology feature
    -> OnlineRetarget Encoder variant
    -> supervised Sonic g1_kin target fields
    -> G1 kinematic retarget motion
```

## Primary Objective

训练并比较 A1 / A2 / B1 / B2 四种 Encoder，在 paired SOMA motionlib -> G1 kinematics target 上降低 kinematics loss。

Primary target:

- Sonic `g1_kin` target fields:
  - `command_multi_future_nonflat`: 29 DoF joint position + 29 DoF joint velocity。
  - `motion_anchor_ori_b_mf_nonflat`: future root orientation 6D。

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

- A1：Concat MLP，1 GPU。
- A2：FiLM MLP，1 GPU。
- B1：Adapter MLP，1 GPU。
- B2：Expert MLP，1 GPU。

每个 variant 训练 1M steps。远程训练必须先确认代码已 commit、已 push、远程 checkout 最新，并在 W&B 记录 OnlineRetarget git SHA、Sonic git SHA、config、variant、motionlib 路径和 run group。

## Validation

训练 loop 内置 validation，不依赖训练后手动 copy。

Visual validation 约每 2k steps 触发一次，每次 8 个 4 秒视频，并上传 W&B。每个视频包含：

1. Source SOMA/BVH capsule motion。
2. Dataset G1 target FK motion。
3. Inference G1 FK motion。

三路视频必须按同一物理时间和 50Hz target timeline 对齐。

## Acceptance Criteria

- 没有任何正式训练命令包含 `g1_dyn`、`train_agent_trl.py`、PPO rollout 或 `num_envs=4096`。
- SOMA motionlib 能被直接读入并生成 kin-only supervised batch。
- A1 / A2 / B1 / B2 都能在 tmux 中启动 1M-step kin-only 训练。
- W&B 中能看到 loss、git SHA、config，以及定期 visual validation videos。
- 最终根据 joint RMSE、velocity RMSE、anchor orientation RMSE、visual validation 和 inference latency 选择下一步主线。
