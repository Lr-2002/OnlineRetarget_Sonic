# Sonic Multi-Skeleton Encoder Pivot

Date: 2026-05-19
Local time: 12:19 CST

## State Snapshot

Current OnlineRetarget state has been preserved, not cleared.

- Active OMX runtime modes: none.
- Remote 8-GPU host: no active `torchrun`, `train_agent_trl.py`, or OnlineRetarget `scripts/train.py` training process.
- Remote GPUs: 8 x RTX 5090 idle at the time of this snapshot.
- Latest OnlineRetarget baseline commit: `6c4e870`.
- Latest completed OnlineRetarget run: filtered BONES Sonic TXT `token_transformer` long run, not VAE.
- Current data rule remains unchanged: `/home/user/data/motion_data` is read-only.

This pivot is also saved in `.omx/notepad.md` working memory for compaction resilience.

## Full SONIC Baseline Run Started

The first full official SONIC training run was launched after the pivot, before
the multi-skeleton encoder implementation work.

Run identity:

- Remote host: `106.14.35.26`, SSH port `1050`.
- Tmux session: `sonic_no_eef_body_xyz_8gpu`.
- Control repo: `/mnt/data_cpfs/code/wxh/mask_controller`.
- SONIC source repo:
  `/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training`.
- Config:
  `configs/p3_official_sonic_filtered_bones_8gpu_wandb_4096_scratch_indexed_no_eef_body_xyz.json`.
- Output directory:
  `/mnt/data_cpfs/code/wxh/mask_controller/outputs/masked_motion_encoder/official_sonic_runs/filtered_bones_8gpu_wandb_4096_scratch_indexed_no_eef_body_xyz`.
- Launch manifest:
  `outputs/masked_motion_encoder/official_sonic_runs/filtered_bones_8gpu_wandb_4096_scratch_indexed_no_eef_body_xyz/launch_manifest.json`.
- Launch log:
  `outputs/masked_motion_encoder/official_sonic_runs/filtered_bones_8gpu_wandb_4096_scratch_indexed_no_eef_body_xyz/logs/launch.log`.
- W&B run:
  `https://wandb.ai/world_model_xh/TRL_G1_Track/runs/42gb2c7z`.

Training data:

- Motion index:
  `/mnt/data_cpfs/code/wxh/mask_controller/outputs/masked_motion_encoder/sonic_motionlib/robot_filtered_v1_index/robot_filtered_v1.index.pkl`.
- Source filtered motion dir:
  `/mnt/data_cpfs/code/wxh/mask_controller/outputs/masked_motion_encoder/sonic_motionlib/robot_filtered_v1`.
- Motion count: `129785`.
- Total frames: `47621019`.
- `smpl_motion_file`: `dummy`; this run is official SONIC G1 filtered-data
  training, not the future SOMA/SMPL multi-skeleton encoder lane.

Training scale:

- Processes: `8`.
- Envs per process: `4096`.
- Global envs: `32768`.
- Learning iterations: `100000`.
- Steps per env: `24`.
- Epochs per update: `5`.
- Minibatches: `4`.
- Actor LR: `2e-5`.
- Critic LR: `1e-3`.

Termination override:

```text
++manager_env.terminations.ee_body_pos=null
++manager_env.terminations.foot_pos_xyz=null
```

The remaining early terminations seen in the training log are:

```text
Env/Episode_Termination/time_out
Env/Episode_Termination/anchor_pos
Env/Episode_Termination/anchor_ori_full
```

Startup verification:

- Tmux session exists.
- `accelerate launch --num_processes=8` and eight `train_agent_trl.py`
  workers are active.
- W&B authenticated as `xhwang_2002 (world_model_xh)`.
- Log loaded `129785 motions`.
- Training reached `Learning iteration 14`.
- At iteration 14, example metrics were:
  - `Total timesteps: 11010048`
  - `Computation: 174122 steps/s`
  - `Mean rewards: 0.70623`
  - `Env/Metrics/motion/error_body_pos: 0.1647`
  - `Env/Metrics/motion/error_joint_pos: 0.3664`
  - `Env/Episode_Termination/anchor_pos: 0.1125`
  - `Env/Episode_Termination/anchor_ori_full: 0.9127`

## User Decision

The next lane should use the SONIC codebase as the reference implementation
instead of continuing to grow a separate ad-hoc retargeter first.

Near-term target:

1. Load prior SONIC training experience.
2. Confirm whether SONIC already exposes different encoders and a complete
   training pipeline.
3. Plan the smallest SONIC-compatible change that can train different skeleton
   encoders and align them into one latent/token space.
4. Prepare for VAE/AE-style pretraining experiments after the loss paths are
   explicit.

## Prior SONIC Experience Loaded

Local control repo:

- `/home/user/project/mask_controller`

Important notes read:

- `/home/user/project/mask_controller/docs/sonic-training-release-audit.md`
- `/home/user/project/mask_controller/docs/sonic-encoder-only-training-knowhow.md`
- `/home/user/project/mask_controller/docs/sonic_low_level_controller_training_experience.md`
- `/home/user/project/mask_controller/docs/sonic_release_category_eval_2026-05-18.md`
- `/home/user/project/mask_controller/docs/sonic_noearly_trace_eval_result_2026-05-18.md`

Key loaded experience:

- Official upstream SONIC training code exists in
  `/home/user/repos/GR00T-WholeBodyControl-upstream-training` at audited rev
  `0a87181c9106d0e49293400714b157676e0ec664`.
- The older local deploy snapshot `/home/user/repos/GR00T-WholeBodyControl`
  is not the source of truth for training-code inspection.
- The official release controller is useful as a reference, but previous
  category eval on our derived filtered BONES distribution had poor full-data
  executability:
  - overall success about `0.191`
  - dominant virtual failure was `foot_pos_xyz`
  - hard categories included jog/run, crawl/kneel, crouch/squat/stoop, dance,
    jump/hop, and walk.
- Encoder-only probes against released SONIC ONNX showed that a small MLP
  student can fit frozen encoder tokens, but that was a deploy-encoder
  distillation probe, not full SONIC retraining.

## SONIC Pipeline Confirmed

The upstream training path is centered on `UniversalTokenModule`:

```text
tokenizer observations
  -> one or more named encoders
  -> shared latent
  -> optional FSQ quantizer
  -> shared token tensor
  -> G1 dynamic decoder -> action
  -> G1 kinematic decoder -> reconstruction auxiliary losses
```

Relevant files:

- `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/train_agent_trl.py`
- `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
- `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`
- `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/trl/modules/universal_token_modules.py`
- `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/trl/losses/token_losses.py`

Confirmed encoder support:

- `sonic_release`: `g1`, `teleop`, `smpl`
- `sonic_bones_seed`: `g1`, `teleop`, `smpl`, `soma`

Confirmed SOMA inputs:

```text
soma_joints_multi_future_local_nonflat
soma_root_ori_b_multi_future
joint_pos_multi_future_wrist_for_soma
```

Confirmed G1 path:

```text
g1 encoder:
  command_multi_future_nonflat
  motion_anchor_ori_b_mf_nonflat

g1_kin decoder:
  reconstructs the same future G1 motion fields

g1_dyn decoder:
  outputs the policy action
```

## Existing Latent Alignment

SONIC already implements the latent-alignment pattern we need.

Existing losses include:

- `G1SmplLatentLoss`
- `G1TeleopLatentLoss`
- `TeleopSmplLatentLoss`
- `G1SomaLatentLoss`
- `ReencodedSmplG1LatentLoss`
- `G1ReconLoss`

The current SOMA-specific alignment is:

```text
z_g1 = E_g1(g1 future motion)
z_soma = E_soma(soma future joints/root/wrist-conditioned fields)
L_g1_soma = MSE(z_g1, z_soma)
```

The config path is:

- `gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent_soma.yaml`
- `gear_sonic/config/aux_losses/terms/g1_soma_latent.yaml`

This means the first implementation should reuse SONIC's encoder/decoder/loss
surface where possible, instead of adding a parallel architecture unless we need
an offline-only simplified trainer.

## What Is Still Missing For Our Goal

SONIC has a SOMA encoder, but it does not yet mean it has our desired
multi-skeleton encoder policy.

Current SONIC SOMA handling is one named `soma` encoder for a uniform selected
26-joint SOMA representation. Our goal is different:

```text
SOMA proportional skeleton family / actor skeleton
  -> skeleton-specific or skeleton-conditioned encoder
  -> unified latent/token
  -> G1 decoder / G1 target motion
```

So the open design question is not "can SONIC have multiple encoders?" It can.
The real question is how to represent multiple human skeletons:

Option A - one shared SOMA encoder with skeleton metadata:

```text
E_soma_shared(soma joints, root, wrists, actor morphology)
```

Option B - one encoder per skeleton family / actor group:

```text
E_soma_A001(...)
E_soma_A002(...)
...
all aligned to E_g1 latent
```

Option C - shared trunk plus lightweight per-skeleton adapter:

```text
E_soma_shared(...)
  + adapter(actor_uid or skeleton_cluster)
  -> latent
```

Current recommendation:

- Start with Option A or C.
- Do not start with hundreds of actor-specific full MLP encoders.
- Use skeleton cluster or morphology-conditioned adapters only if shared SOMA
  encoder underfits by actor/skeleton group.

## VAE / AE Pretraining Loss Paths

Before full PPO/IsaacLab training, run an offline token/latent pretraining lane.
This can be built either as a lightweight OnlineRetarget trainer or as a SONIC
compatible pretraining wrapper around `UniversalTokenModule`.

Use filtered paired data only.

Definitions:

```text
B = SOMA/BVH-derived human skeleton motion feature
G = G1 target future motion feature
S = skeleton / morphology / skeleton-family feature
z_b = E_b(B, S)
z_g = E_g(G)
G_hat_from_b = D_g(z_b)
B_hat = D_b(z_b)
G_hat = D_g(z_g)
```

Losses for the first deterministic AE-style run:

```text
L_b_rec = MSE(B_hat, B)
L_g_rec = MSE(G_hat, G)
L_b2g   = MSE(G_hat_from_b, G)
L_lat   = MSE(z_b, z_g)

L_total = w_b * L_b_rec
        + w_g * L_g_rec
        + w_x * L_b2g
        + w_z * L_lat
```

Use VAE KL only after the MSE paths are stable:

```text
L_kl = KL(q_b(z | B, S) || N(0, I)) + KL(q_g(z | G) || N(0, I))
```

## Two Initial Experiments

Both experiments use the same filtered dataset and the same actor-heldout
`8/1/1` train/val/test split.

Experiment 1 - single frame:

```text
B_t, S -> z_b -> B_t
G_t    -> z_g -> G_t
B_t, S -> z_b -> G_t
z_b aligned with z_g
```

Experiment 2 - temporal 4 past + 1 current:

```text
[B_{t-4}, ..., B_t], S -> z_b -> target current frame
[G_{t-4}, ..., G_t]    -> z_g -> target current frame
z_b aligned with z_g
```

Initial recommendation:

- Decode the current frame first, not the whole window.
- Add whole-window reconstruction only as a later auxiliary loss if current
  frame metrics are stable.

## Implementation Direction

Decision correction on 2026-05-19:

- Do not build the multi-skeleton baseline through OnlineRetarget
  `token_transformer`/TF code.
- Do not introduce a standalone downstream retargeter decoder while the project
  direction is SONIC-first.
- Keep OnlineRetarget's registry/data-audit artifacts, because SONIC still
  needs a stable `actor_uid` / proportional skeleton mapping.
- Implement model changes in the SONIC codebase:
  `/home/user/project/GR00T-WholeBodyControl-upstream-training`.

SONIC-native implementation plan:

1. Keep SONIC's existing `UniversalTokenModule` shape:
   `g1/smpl/teleop/soma encoders -> shared FSQ/token -> g1_kin/g1_dyn decoders`.
2. Treat the current `soma` encoder as the extension point for proportional
   skeleton adaptation.
3. Feed actor/proportional skeleton identity or shape features into SONIC's
   tokenizer observations, not into a separate OnlineRetarget model.
4. Start with the smallest SONIC-compatible design:
   - one shared `soma` MLP plus lightweight actor/shape conditioning or adapter;
   - only escalate to one full MLP encoder per actor if this underfits.
5. Keep SONIC's existing losses as the reference objective:
   - `G1SomaLatentLoss` for `z_g1` / `z_soma` alignment;
   - `G1ReconLoss` for G1 kinematic decoder reconstruction;
   - `g1_dyn` decoder remains the controller-action path.
6. Use OnlineRetarget registry only to generate/audit the mapping from filtered
   BONES Sonic rows to proportional skeleton ids and shape files.

## Immediate Next Checks

Before training:

- Verify exact local availability of SONIC-style SOMA PKL data or build it
  from `soma_proportional.tar` into a derived output directory.
- Verify shape compatibility against SONIC's expected SOMA 26-joint format.
- Confirm whether the filtered `data.txt` can be mapped losslessly to both G1
  NPZ and SOMA/BVH source paths.
- Create one small paired dataset smoke and print:
  - actor id
  - skeleton/morphology vector
  - B shape
  - G shape
  - split
  - source and target paths
- Only then launch two 4-GPU experiments.

## Proportional Skeleton Registry v0

Generated from the filtered BONES Sonic curated index:

```bash
PYTHONPATH=src python3 -m online_retarget.cli build-skeleton-registry \
  --index-csv runs/curated/bones_sonic_txt_filtered_pair_s17/curated_index.csv \
  --data-root /home/user/data/motion_data \
  --output-root runs \
  --run-name bones_sonic_txt_filtered_skeleton_registry_v0 \
  --action-column merged_quality_action
```

Outputs:

- `runs/skeleton_registry/bones_sonic_txt_filtered_skeleton_registry_v0/skeleton_registry.csv`
- `runs/skeleton_registry/bones_sonic_txt_filtered_skeleton_registry_v0/skeleton_registry_report.json`

Key result:

- Rows seen: `112723`
- Rows used for first training lane (`keep` + `downweight`): `110697`
- Rows excluded by action filter (`quarantine`): `2026`
- Actor/proportional skeleton count in full table: `515`
- Effective actor/proportional skeleton count for first encoder bank: `489`
- Actors excluded by action filter: `26`
- Missing proportional shape files among effective actors: `0`
- Split clips: train `87394`, val `12758`, test `10545`
- Split actors: train `387`, val `50`, test `52`
- Clip count per effective actor: min `2`, mean `226.374233`, max `889`
- Shape header signature is consistent for all effective actors:
  `face_expr_params:<f4:1x72|identity_params:<f4:1x45|pose_params:<f4:1x136|scale_params:<f4:1x68`

Interpretation:

- The first practical skeleton id is `actor_uid`.
- The first practical encoder id is also `actor_uid`.
- The proportional skeleton source is `move_soma_proportional_path`.
- The actor shape source is `move_soma_proportional_shape_path`.
- The paired robot target remains the G1 motion in the same curated row.
- Do not allocate first-version actor encoders for the 26 actors that have no
  `keep` or `downweight` rows.

Baseline-2 minimal implementation boundary:

```text
SONIC motion library row
  -> actor_uid / proportional skeleton id from registry
  -> SONIC tokenizer obs:
       soma_joints_multi_future_local_nonflat
       soma_root_ori_b_multi_future
       joint_pos_multi_future_wrist_for_soma
       actor/shape conditioning feature
  -> SONIC soma encoder or soma adapter
  -> shared SONIC latent/token Z
  -> shared SONIC g1_kin decoder
  -> shared SONIC g1_dyn decoder / low-level controller path
```

The registry is only the routing and audit artifact. It does not change
training behavior by itself, and it is not a reason to add a TF baseline.
