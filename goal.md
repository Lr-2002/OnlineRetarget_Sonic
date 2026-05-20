# OnlineRetarget Project Goal

## 1. 项目目标

OnlineRetarget 的正式目标是：以 Sonic 原生代码路径为基准，固定 Sonic 面向 Unitree G1 的 Dynamics Decoder 作为主要训练目标，重新训练一组面向不同 skeleton / feature 条件的 Retarget Encoder。

最终系统需要实现：

```
Human / SOMA / BVH motion + skeleton feature
    -> OnlineRetarget Encoder
    -> Sonic g1_dyn Dynamics Decoder
    -> G1 retarget action / motion
```

这次从头改代码的重点不是做一个独立 AutoEncoder，也不是只做 G1 kinematics reconstruction，而是让新的 Encoder 接入 Sonic 原本可部署的 dynamics decoder 路径，用于真正的 human skeleton motion -> G1 retarget。

## 2. 基本原则

- Sonic 是基准实现：训练入口、motion timing、decoder、tokenizer、DDP、W&B 记录方式都优先复用 Sonic 原生逻辑。
- `g1_dyn` 是主目标：Dynamics Decoder 的 action / latent / token 监督是主训练目标。
- Kinematics 只做辅助：`g1_kin`、FK、joint/body loss、visualization 可以作为辅助监督和诊断，不能替代 dynamics 目标。
- Encoder 输入必须来自部署时可获得的 source-side 信息，例如 Human/SOMA/BVH motion、skeleton morphology、contact、phase、root-local feature。
- `body_pos_w` / `body_quat_w` 不能作为正式 Encoder source input；它们只能作为 target label、FK/render 诊断或 validation 对照。
- Training、validation、inference 必须共享同一套 feature packing contract，避免训练时使用部署时拿不到的 target-only signal。

## 3. Source / Target 定义

Source：

- Human / SOMA / BVH 侧 motion。
- Skeleton / morphology 信息，例如 actor id、skeleton id、bone length、height、limb proportion、foot/leg/arm/torso descriptor。
- Root-local pose / velocity / orientation feature。
- Contact / phase / foot-state feature。
- Sonic 50Hz 时间轴对齐后的 temporal window。

Target：

- Sonic / G1 侧 dynamics decoder target，优先为 `g1_dyn` action 或与 `g1_dyn` teacher path 对齐的 latent/token。
- 可选辅助 target：G1 joint position / velocity、FK body pose、contact consistency、temporal smoothness。
- Dataset G1 target motion 只作为监督和验证目标，不能泄漏到 Encoder source input。

## 4. 四种 Encoder 方案

四个方案共享同一个 Sonic-native decoder target path，只改变 source feature encoding 和 skeleton conditioning 方式。

### A1: Concat Encoder

- 将 source motion feature 与 skeleton/morphology feature 直接 concat。
- 使用 compact MLP。
- 作为最简单、最稳定、最容易 debug 的 baseline。

### A2: FiLM / Contact Encoder

- 用 skeleton/morphology feature 生成 FiLM conditioning。
- 加入 contact / phase feature。
- 目标是验证骨架条件与接触状态是否能降低 dynamics loss 和 kinematics auxiliary loss。

### B1: Adapter Encoder

- 使用共享 Encoder backbone。
- 为不同 skeleton / proposal / actor group 加 lightweight adapter。
- Adapter routing 必须写入 config，并记录到 W&B。
- 目标是验证共享表示加少量骨架专用参数是否优于纯 concat / FiLM。

### B2: Expert Encoder

- 使用 lightweight expert 或 mixture-style branches。
- Expert route 由 skeleton id、proposal id 或 morphology cluster 决定。
- Routing 必须 deterministic、可复现、可记录、可分析。
- 目标是验证不同 skeleton family 是否需要更强的参数隔离。

## 5. 训练目标和 Loss

主目标：

- 新 Encoder 输出的 representation 能驱动 Sonic `g1_dyn` Dynamics Decoder，复现 Sonic teacher path 或 dataset 中可用的 G1 target action。

Loss 优先级：

1. Dynamics action loss：对齐 `g1_dyn` action / body action / meta action。
2. Latent/token alignment：对齐 Sonic teacher encoder 的 latent/token。
3. Kinematic auxiliary loss：G1 joint RMSE、FK body MPJPE、body orientation error。
4. Temporal smoothness：约束 inferred action / joint trajectory 的跳变。
5. Contact-aware loss：在 source/target contact 与频率验证正确后加入。

核心指标：

- `g1_dyn` action MSE / cosine similarity。
- Latent/token MSE / cosine similarity。
- G1 joint RMSE / MAE。
- FK body MPJPE。
- Foot sliding / contact artifact。
- Batch size 1 inference latency，目标是满足在线部署预算。

## 6. 频率和 Pose 规则

- Sonic target timeline 以 50Hz 为准。
- BVH / SOMA source motion 必须对齐或重采样到 Sonic 50Hz target timeline。
- Validation video 中 source BVH/SOMA、dataset G1 target、inferred G1 必须按同一物理时间播放。
- 不允许用原始 BVH frame count 直接和 50Hz G1 target 硬对齐。
- 不把 world-frame absolute XY 作为 deployable source target。
- Anchor/root Z 可以作为 height 或 diagnostic signal，但必须明确标注。
- Root orientation 使用 Sonic 兼容表示，并区分 world-frame body pose 与 root-local / anchor-relative representation。

## 7. 训练计划

初始资源分配：

- A1：1 张 GPU。
- A2：1 张 GPU。
- B1：1 张 GPU。
- B2：1 张 GPU。

正式训练要求：

- 每个 variant 训练 1M steps。
- 每次远程训练前，必须确认 OnlineRetarget 仓库已 commit、已 push、远程 checkout 是最新版本。
- W&B 必须记录 OnlineRetarget git SHA、Sonic git SHA、config、encoder variant、dataset/motionlib revision、run group。
- 长训练必须放在 tmux 或等价可恢复后台会话中。
- `/home/user/data/motion_data` 只读；derived data、logs、checkpoints、renders 写入 repo-local `runs/`、`outputs/` 或显式输出目录。

## 8. Integrated Validation

Validation 必须集成在 training loop 中，不能依赖训练后手动 copy 或单独脚本人工拼接。

每 20k steps 自动执行一次 visual validation：

- 每次 8 个 validation clips。
- 每个 clip 使用 4 秒 inference window。
- 每个 rank/GPU 只写自己的输出 slice，路径不能互相覆盖。
- 自动上传完整视频到 W&B。

每个 validation video 至少包含三路同步结果：

1. Source BVH / SOMA proportional capsule motion。
2. Dataset G1 target motion。
3. Inferred G1 motion from OnlineRetarget Encoder + Sonic `g1_dyn` path。

Validation 日志必须记录：

- source FPS。
- target FPS。
- Sonic target frame count。
- source frame index range。
- physical duration。
- OnlineRetarget git SHA。
- Sonic git SHA。
- encoder variant 和 routing 信息。

## 9. Deliverables

- Sonic-native training entrypoint。
- A1/A2/B1/B2 四个正式 config。
- 共享的 train / validation / inference feature packer。
- Source feature guardrail，确保正式 retarget config 不把 `body_pos_w` / `body_quat_w` 作为 Encoder input。
- `g1_dyn` dynamics decoder target loss。
- 必要的 kinematic auxiliary loss 和 FK/visual diagnostic。
- Integrated W&B video validation callback。
- Remote launcher：启动前检查 git committed、pushed、synced、latest。
- 四个 1M-step runs 的 W&B 结果、日志、checkpoint 和对比报告。

## 10. Acceptance Criteria

该 Goal 完成的标准：

- 四个 Encoder variant 都能在 Sonic-native training path 下启动并训练。
- 主训练目标接入 Sonic `g1_dyn` Dynamics Decoder，而不是 standalone reconstruction decoder。
- Source input 全部来自 Human/SOMA/BVH/skeleton feature，不使用 G1 target-only field。
- Training、validation、inference 使用同一套 feature contract。
- 每个 variant 完成 1M steps，或给出明确失败原因、可复现日志、W&B run、git SHA。
- W&B 中能看到每个 run 的 config、git SHA、metrics 和 20k-step validation videos。
- Validation videos 中 source、dataset target、inference output 按同一物理时间对齐。
- 最终能基于 dynamics loss、kinematics auxiliary loss、visual validation 和 latency，选择下一步主线方案。

## 11. Non-Goals

- 不把 standalone OnlineRetarget AutoEncoder 作为正式目标。
- 不把 `body_pos_w` / `body_quat_w` 当作 source feature 训练。
- 不在 MLP / FiLM / Adapter / Expert baseline 跑通前引入 diffusion、flow matching 或大型 transformer。
- 不先做 simulator-heavy dynamics rollout；当前阶段以 Sonic dynamics decoder target 和 retarget training 为核心。
- 不把 MaskController 作为本项目目标。
