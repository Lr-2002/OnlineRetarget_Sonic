# OnlineRetarget Project Goal

## Goal

Build a Sonic-native online retargeting training lane that learns feature-conditioned source encoders for heterogeneous human skeleton motion, while reusing Sonic's existing decoder/policy surfaces as the target path.

The immediate target is not a standalone OnlineRetarget autoencoder. The target is a Sonic-compatible encoder whose latent/token output can drive Sonic's dynamics decoder path for G1 retargeting. Kinematic reconstruction remains an auxiliary diagnostic, but the primary training and inference contract must match Sonic's deployable path.

## Problem To Fix

The previous standalone trainer used `body_pos_w` and `body_quat_w` from `bones_sonic` as model input. That is not valid for the retargeting goal:

- `body_pos_w/body_quat_w` are G1 target/reference robot states, not human BVH/SOMA source features.
- Using them as encoder input makes the task G1 reconstruction, not human-to-G1 retargeting.
- This path must be treated as legacy diagnostics only, not as the formal OnlineRetarget training lane.

## Sonic-Native Contract

All new training code must be based on Sonic's native architecture and data flow:

- Use Sonic `UniversalTokenModule`, tokenizer observations, motion library timing, encoder/decoder configs, and DDP-ready training surfaces.
- Source features must come from human-side motion and skeleton information, such as SOMA/BVH proportional skeleton windows, root orientation, contacts, and skeleton/morphology descriptors.
- Target decoding must use Sonic's G1 decoder path, especially the dynamics decoder path (`g1_dyn`) for deployable retargeting.
- Kinematic decoder outputs (`g1_kin`) may be used as auxiliary supervision, diagnostics, and visualization, but should not replace the dynamics-decoder objective.
- Training and inference must share the same feature packing contract. No target-only fields may silently enter the deployable encoder input.

## Source Feature Contract

Primary source inputs should be Sonic-compatible human/skeleton features:

- `soma_joints_multi_future_local_nonflat` or equivalent root-local proportional BVH/SOMA joints.
- `soma_root_ori_b_multi_future` or equivalent root orientation feature.
- Skeleton/morphology features: actor/skeleton ID, bone lengths, body proportions, height, foot/leg/arm/torso measurements.
- Optional contact/phase features when available from source motion.

Important constraint:

- `body_pos_w` and `body_quat_w` may be used only as G1 target-side labels, rendering targets, FK checks, or diagnostics.
- They must not be used as source encoder features in formal retargeting configs.
- Any Sonic field that is target-derived but appears in a source encoder input, such as G1 future wrist joint features, must be audited and either removed/replaced for deployable configs or marked as a teacher-forced ablation.

## Target And Loss Contract

Primary objective:

- Train each source encoder so its latent/token output drives Sonic's G1 dynamics decoder consistently with Sonic's native G1/SOMA policy path.

Required losses and signals:

- Dynamics decoder imitation loss: decoded action/body action/meta action from the new encoder should match the Sonic teacher path or dataset-compatible target action when available.
- Latent alignment loss: new encoder latent should align with Sonic's canonical G1/SOMA latent for matched motion clips.
- Kinematic auxiliary loss: decoded or derived G1 kinematics should match target G1 joint/body motion for debugging and validation.
- Smoothness and temporal consistency losses for decoded action/reference sequences.
- Optional contact-aware loss after source/target timing is verified.

Validation metrics:

- dynamics action MSE / cosine similarity
- latent MSE / cosine similarity
- G1 joint RMSE / MAE
- body MPJPE from FK or dataset `body_pos_w`
- foot sliding / contact artifacts when contact labels are available
- latency for batch size 1 on target hardware

## Frequency And Pose Rules

All train, validation, and visualization paths must use Sonic's timing rules:

- Sonic target timeline is 50Hz.
- BVH/SOMA source motions must be time-aligned or resampled to the Sonic 50Hz target timeline.
- Validation videos must play source BVH, dataset G1 target, and inferred G1 at the same physical time, not by equal raw frame count.

Pose handling rules:

- Anchor/root XY should not be used as a direct absolute-position target for the deployable retarget encoder.
- Anchor/root Z may be used as height supervision or diagnostic signal when consistent with Sonic.
- Root rotation should be represented in Sonic-compatible 6D orientation features.
- Distinguish full world-frame body pose from Sonic's root-local / anchor-relative source representation in code and config names.

## Encoder Experiments

Implement four Sonic-native encoder variants. These should be different encoder/conditioning choices feeding the same Sonic decoder target path, not separate standalone retargeters.

1. A1: concat encoder
   - Concatenate source motion features and skeleton/morphology features.
   - Compact MLP baseline.

2. A2: FiLM/contact encoder
   - Use FiLM-style conditioning from skeleton/morphology and optional contact/phase features.
   - Keep model small and deployable.

3. B1: adapter encoder
   - Shared source encoder plus skeleton-specific adapter modules.
   - Adapter routing must be explicit and logged.

4. B2: expert encoder
   - Lightweight expert or mixture-style branches for skeleton/proposal groups.
   - Expert selection must be deterministic or logged for reproducibility.

## Training Runs

Initial run allocation:

- A1: 1 GPU
- A2: 1 GPU
- B1: 1 GPU
- B2: 1 GPU

Training requirements:

- Run 1M steps for formal comparison.
- Use tmux for long training.
- Commit code before each meaningful training run.
- Log git commit SHA, config, dataset/index revision, and encoder variant to W&B.
- Before remote training starts, verify the remote git repository is on the expected latest commit.
- Keep `/home/user/data/motion_data` read-only. Derived files go under `runs/`, `outputs/`, or explicit repo-local output paths.

## Integrated Visualization Validation

Validation must be integrated into training, not manually run afterward.

Every 20k training steps:

- render 8 validation clips
- use 4 seconds per clip
- each GPU/rank should handle its own inference/render slice without colliding output paths
- upload videos to W&B

Each visual validation should show aligned panels:

1. source proportional BVH/SOMA capsule motion
2. dataset G1 target motion
3. inferred G1 motion from the current encoder + Sonic decoder path

The validation report must include source FPS, target FPS, source frame indices, target frame count, duration, and git SHA.

## Deliverables

1. Sonic-native training entrypoint/configs for the four encoder variants.
2. Shared train/inference feature packer with explicit deployable feature contract.
3. Guardrails that prevent formal retarget configs from using `body_pos_w/body_quat_w` as source inputs.
4. Integrated validation and W&B video upload.
5. Documentation of source feature semantics, target decoder semantics, timing, and pose conventions.
6. A comparison report over A1/A2/B1/B2 after 1M-step runs.

## Acceptance Criteria

The project goal is satisfied when:

- Formal training uses Sonic-native modules and decoder paths rather than the standalone G1 reconstruction trainer.
- Source encoder inputs are human/SOMA/BVH and skeleton features, not G1 `body_pos_w/body_quat_w`.
- The same feature contract is used for training, validation, and inference.
- A1/A2/B1/B2 can each launch under the assigned GPU budget.
- W&B records each run with git SHA, config, metrics, and validation videos.
- Validation videos are time-aligned at Sonic target frequency and show source, dataset target, and inference output together.
- The best variant is selected by dynamics/retarget metrics, kinematic auxiliary metrics, visual validation, and latency.

## Non-Goals

- Do not build a separate non-Sonic OnlineRetarget model as the formal training path.
- Do not train on G1 target body pose as if it were human source input.
- Do not add diffusion, flow matching, or large transformer models before the Sonic-native MLP/adapter/expert baselines are measured.
- Do not start simulator-heavy dynamics work outside Sonic/Isaac Lab integration.
