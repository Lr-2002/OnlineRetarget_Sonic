# OnlineRetarget Project Goal

## 目标

本项目的正式目标是：以 Sonic 原生代码为基准，保留 Sonic 面向 Unitree G1 的 Decoder / Policy 目标路径，重新训练一组面向不同人形骨架与不同 source feature 的 Encoder，使其能够把 Human / SOMA / BVH 侧运动特征映射到 Sonic 可用的 G1 dynamics decoder 输入，从而完成在线 Retarget。

这不是重新实现一个独立 AutoEncoder，也不是只做 G1 motion reconstruction。核心目标是让新的 Encoder 接入 Sonic 原本用于部署的 dynamics decoder 路径，最终服务于 Human skeleton motion -> G1 retargeted motion / action。

## 核心原则

- 以 Sonic 原生训练、数据流、tokenizer、decoder、motion timing 和 DDP 结构为基准。
- 训练目标优先对齐 Sonic dynamics decoder，也就是 `g1_dyn` 路径。
- Kinematics decoder 或 G1 kinematic target 只作为辅助监督、诊断和可视化，不作为最终项目目标。
- Encoder 的输入必须来自 human-side motion、SOMA/BVH skeleton feature、root-local motion feature、contact/phase feature 或 skeleton morphology feature。
- `body_pos_w` / `body_quat_w` 只能作为 target label、FK 检查、render target 或 validation diagnostic，不能作为正式 retarget encoder 的 source input。
- 训练、validation、inference 必须共享同一套 feature packing contract，避免训练时使用部署时不可获得的 target-only signal。

## 要解决的问题

当前项目需要从“G1 target reconstruction”转向真正的“Human/Skeleton -> Sonic G1 dynamics decoder retarget”。

需要明确区分：

- Source：Human / SOMA / BVH motion，以及对应的 skeleton/morphology 描述。
- Target：Sonic / G1 侧的 action、body action、joint state、body pose、kinematic target 和 dynamics decoder target。
- Encoder：负责把不同 skeleton、不同 motion feature 编码到 Sonic decoder 可消费的 latent/token/action-conditioned representation。
- Decoder：尽量复用 Sonic 原生 dynamics decoder，不为 OnlineRetarget 单独发明一套非 Sonic 的目标路径。

## Source Feature Contract

正式 Encoder 可以使用的 source feature 包括：

- SOMA/BVH proportional skeleton motion window。
- Root-local joint positions / rotations。
- Root orientation，优先使用 Sonic 兼容的 6D orientation 表示。
- Contact、phase 或 foot state feature。
- Skeleton / morphology descriptor，例如 actor id、skeleton id、bone length、height、arm/leg/torso proportion、foot length 等。
- Motion timing feature，例如 Sonic 50Hz 对齐后的 frame index、phase 或 temporal window。

正式 Encoder 不允许使用：

- G1 `body_pos_w` / `body_quat_w` 作为 source input。
- 任何只在 target G1 trajectory 中可获得、但真实在线 retarget 部署时不可获得的字段。
- 手动 validation 或 inference 中临时拼出来、训练路径没有使用的 feature。

## Target And Loss Contract

主训练目标：

- 新 Encoder 输出的 latent/token/action-conditioned representation 能驱动 Sonic `g1_dyn` dynamics decoder，并复现 Sonic teacher path 或 dataset 中可用的 G1 target action。

建议 loss 结构：

- Dynamics action loss：对齐 Sonic `g1_dyn` decoder 目标 action / body action / meta action。
- Latent alignment loss：对齐 Sonic canonical encoder 或 teacher path 的 latent/token。
- Kinematic auxiliary loss：用 G1 joint、body position、body orientation 或 FK metric 做辅助监督。
- Temporal smoothness loss：约束 inferred action / joint trajectory 的跳变。
- Contact-aware loss：在 source/target contact 和 timing 验证正确后加入。

关键指标：

- Dynamics action MSE / cosine similarity。
- Latent MSE / cosine similarity。
- G1 joint RMSE / MAE。
- FK body MPJPE。
- Foot sliding / contact artifact。
- Batch size 1 推理延迟，目标硬件上需要满足在线部署预算。

## 四种 Encoder 结构

本项目先实现并比较四个 Sonic-native Encoder 变体。四者共享 Sonic decoder target path，只改变 source feature encoding 和 skeleton conditioning 方式。

### A1: Concat Encoder

- 将 source motion feature 与 skeleton/morphology feature 直接 concat。
- 使用 compact MLP 作为 baseline。
- 目标是建立最简单、最稳定、最容易 debug 的 Sonic-native retarget baseline。

### A2: FiLM / Contact Encoder

- 使用 skeleton/morphology feature 生成 FiLM conditioning。
- 可加入 contact / phase feature。
- 目标是验证骨架差异和接触状态是否能显著降低 dynamics 与 kinematics loss。

### B1: Adapter Encoder

- 使用共享 Encoder backbone。
- 为不同 skeleton/proposal/actor group 加入轻量 Adapter。
- Adapter routing 必须显式记录到 config 和 W&B。
- 目标是验证共享表示加少量骨架专用参数是否优于纯 concat / FiLM。

### B2: Expert Encoder

- 使用 lightweight expert 或 mixture-style branches。
- Expert selection 可以由 skeleton id、proposal id 或 morphology cluster 决定。
- Expert routing 必须可复现、可记录、可分析。
- 目标是验证不同 skeleton family 是否需要更强的参数隔离。

## Frequency And Pose Rules

训练、validation、inference 必须统一 Sonic timing：

- Sonic target timeline 以 50Hz 为准。
- BVH/SOMA source motion 必须对齐或重采样到 Sonic 50Hz target timeline。
- Visualization 中 source BVH、dataset G1 target、inferred G1 必须按同一物理时间播放，不能按原始 frame count 强行对齐。

Pose 约束：

- 不把 world-frame absolute XY 当作 deployable source target。
- Anchor/root Z 可以作为 height 或 diagnostic signal，但需要明确标注。
- Root orientation 使用 Sonic 兼容表示，并区分 world-frame body pose 与 root-local / anchor-relative representation。

## Training Plan

初始训练资源分配：

- A1：1 张 GPU。
- A2：1 张 GPU。
- B1：1 张 GPU。
- B2：1 张 GPU。

正式比较要求：

- 每个 variant 训练 1M steps。
- 每次远程训练前必须确认 OnlineRetarget git 仓库是最新 commit。
- 每次训练前必须保证代码已 commit。
- W&B 必须记录 git SHA、config、encoder variant、dataset / motionlib revision、Sonic commit 和 run group。
- 长训练必须放在 tmux 或等价的可恢复后台会话中。
- `/home/user/data/motion_data` 只读，所有派生数据、日志、checkpoint、render 输出放到 repo-local `runs/`、`outputs/` 或显式输出路径。

## Integrated Validation

Validation 必须集成在 training loop 中，不能依赖手动 copy 或训练后人工脚本。

每 20k steps 自动执行一次 visual validation：

- 每次渲染 8 个 validation clips。
- 每个 clip 使用 4 秒推理窗口。
- 每张卡 / 每个 rank 负责自己的 inference 和 render slice，输出路径不能互相覆盖。
- 自动上传完整视频到 W&B。

每个 validation video 至少包含三路对齐结果：

1. Source BVH / SOMA proportional capsule motion。
2. Dataset G1 target motion。
3. Inferred G1 motion from new Encoder + Sonic dynamics decoder path。

Validation 日志必须记录：

- source FPS。
- target FPS。
- Sonic target frame count。
- source frame index range。
- physical duration。
- OnlineRetarget git SHA。
- Sonic git SHA。

## Deliverables

- Sonic-native training entrypoint 和四个正式 config：A1、A2、B1、B2。
- 共享的 train / validation / inference feature packer。
- Source feature guardrail：禁止正式 retarget config 使用 `body_pos_w` / `body_quat_w` 作为 encoder input。
- Dynamics decoder target loss 和必要的 kinematic auxiliary loss。
- Integrated W&B video validation。
- Remote launcher：训练前检查远程 OnlineRetarget 是否为最新 commit。
- A1/A2/B1/B2 的 1M-step 训练结果和对比报告。

## Acceptance Criteria

该 Goal 完成的标准是：

- 四个 Encoder variant 都能在 Sonic-native training path 下启动。
- 主目标接入 Sonic `g1_dyn` dynamics decoder，而不是 standalone reconstruction decoder。
- Source input 全部来自 human/SOMA/BVH/skeleton feature，不使用 G1 target-only field。
- Training、validation、inference 使用同一套 feature contract。
- 每个 variant 完成 1M steps 或给出明确失败原因与可复现日志。
- W&B 中能看到每个 run 的 config、git SHA、metrics 和 20k-step validation videos。
- Validation videos 中 source、dataset target、inference output 在同一物理时间下对齐。
- 最终能基于 dynamics loss、kinematics auxiliary loss、visual validation 和 latency 选择下一步主线方案。

## Non-Goals

- 不把 standalone OnlineRetarget AE 作为正式目标。
- 不把 `body_pos_w` / `body_quat_w` 当作 source feature 训练。
- 不在 MLP / Adapter / Expert baseline 跑通前引入 diffusion、flow matching 或大型 transformer。
- 不先做 simulator-heavy dynamics rollout；当前阶段以 Sonic dynamics decoder target 和 retarget training 为核心。
- 不把 MaskController 作为本项目目标；如果引用历史 motionlib 路径，只作为数据来源或兼容说明。
