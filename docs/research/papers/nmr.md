# NMR Reading Note

Primary source: https://arxiv.org/abs/2603.22201

## Core Claim

Optimization-based retargeting is non-convex and sensitive to initialization. NMR instead learns a distribution mapping from human motion to feasible humanoid robot motion, using physics-refined supervision.

## Deep-Read Insights

### 1. NMR's main contribution is data construction, not just network size

The paper's strongest insight is that a neural retargeter is only as good as its supervision. If the model is trained only on kinematic retargeting output, it learns the same IK local-minimum artifacts: joint jumps, self-penetration, foot sliding, and joint-limit saturation.

NMR solves this with CEPR: Clustered-Expert Physics Refinement.

CEPR has three gates:

1. Physics-aware human motion curation.
2. Kinematic retargeting plus robot-motion quality filtering.
3. Physics-based humanoid motion refinement through simulator tracking experts.

Implication for OnlineRetarget: our first formal milestone cannot be "train bigger model" before source/G1 quality is understood. BONES source quality, G1 target quality, pair consistency, and later simulator-refined provenance need to remain separate labels.

### 2. Their filtering hierarchy matches our M2Q direction

NMR filters source human motions for excessive jerk, bad support-base relation, insufficient foot-ground contact, floating, and penetration. After GMR-like kinematic retargeting, it filters robot targets by joint velocity discontinuity, MuJoCo self-intersection, and foot floating.

Concrete thresholds reported in the paper:

- self-intersection frame tolerance: `cross_ratio = 0.05`
- floating sequence threshold: mean lowest-foot elevation above ground `> 0.10 m`
- joint jump / discontinuity: hardware-specific peak joint velocity threshold during data filtering
- evaluation joint jump: max single-step joint angle change `> 0.5 rad`
- joint limit metric: any joint within `0.05 rad` of hardware boundary

Implication for OnlineRetarget: use these as metric definitions or sanity references, not direct BONES thresholds. BONES contains jumps, kicks, sitting, floor motions, and actor-specific SOMA skeletons, so thresholds must be category-aware.

### 3. CEPR is expensive but tells us the target provenance hierarchy

NMR first creates a broad kinematic dataset, then creates about 30K physics-refined SMPL-G1 pairs through clustered RL expert rollouts. The refined pairs are treated as pseudo ground truth because they have been executed in simulation.

Implication for OnlineRetarget:

- Stage A target: existing BONES/SONIC kinematic G1 targets.
- Stage B target: simulator-replayed or simulator-refined G1 targets with separate provenance.
- Do not mix kinematic and physics-refined targets without labels; they encode different objectives.

### 4. Motion clustering is used to make physics refinement tractable

The paper argues that one global tracking policy over all motions suffers from distribution conflict, while one policy per clip is too expensive. CEPR clusters motions by semantic/latent motion features, then trains one expert policy per cluster.

Implication for OnlineRetarget: we do not need clustering for the first supervised BVH->G1 baseline. We will need it later if we generate simulator-refined targets at scale, because locomotion, sitting, martial arts, dance, and floor motions should not be forced through one refinement policy.

### 5. Their model representation is simple and useful

Human motion representation:

- root planar velocity
- root orientation in 6D
- local joint positions
- local joint velocities

Robot motion representation:

- root planar velocity
- root orientation
- local robot body positions
- local robot body velocities
- robot DoFs

Implication for OnlineRetarget: this supports our current observation contract: root/body-local BVH FK positions and velocities plus morphology, predicting G1 joint positions or deltas. We do not need SMPL-X specifically for this insight.

### 6. Bidirectional temporal context is essential

NMR uses a non-autoregressive CNN-Transformer with full self-attention rather than causal attention because input and output are temporally aligned one-to-one. This lets each frame use future and past context, suppress local source jitters, and avoid frame-wise IK discontinuities.

Implication for OnlineRetarget: our baseline should not be single-frame. A compact temporal MLP/TCN over a short history is the first latency-safe version. A small bidirectional temporal Transformer is a second-stage model if MLP/TCN underfits.

### 7. The two-stage training scheme matters more than exact architecture

NMR's two stages:

1. Pretrain on large kinematic retargeting data for broad coverage.
2. Fine-tune on smaller physics-refined CEPR data for physical grounding.

Their argument is that kinematic data alone is broad but physically flawed; physics data alone is high-quality but too narrow and overfits.

Implication for OnlineRetarget:

1. Train on broad BONES/SONIC kinematic pairs first.
2. Evaluate failure modes.
3. Add simulator-refined subsets later for fine-tuning, not at the beginning.

### 8. Evaluation is artifact-first plus downstream tracking

NMR evaluates both direct retargeting artifacts and downstream tracking:

- joint jump
- self-collision
- joint-limit proximity
- downstream tracking success rate
- MPJPE
- W-MPJPE

Reported result direction: NMR reaches zero joint jumps on their test suite, reduces self-collision frames relative to GMR, and reduces joint-limit violation frames. Their ablation without physics refinement remains much worse on self-collision, confirming that model architecture alone does not solve physical feasibility.

Implication for OnlineRetarget: offline eval must include artifact metrics before Isaac Lab tracking. MPJPE alone is insufficient because a motion can be close to source but unusable on G1.

### 9. NMR has a morphology limitation

The paper notes that CEPR is morphology-specific: extending to a new robot requires regenerating the data pipeline, and morphology-conditioned architectures remain future work.

Implication for OnlineRetarget: since our first robot is G1, this is acceptable. For human-side actor morphology, BONES actor-specific SOMA proportional skeletons are valuable and should be kept in the input rather than collapsed into uniform skeleton only.

## Observation / Input

- Human SMPL/SMPL-X motion sequences.
- Global temporal context, not frame-only IK.
- Physics-refined target pairs generated by CEPR.

## Output

- Unitree G1 humanoid motion sequence, mainly G1 joint angles / generalized coordinates for downstream tracking.

## Model

- Non-autoregressive CNN-Transformer.
- Pretraining on large kinematic retargeting data.
- Post-training/fine-tuning on about 30K physically consistent pairs generated by clustered expert policies.

## Data

- Large human motion data retargeted initially by optimization-based methods.
- CEPR data pipeline:
  - curate physical human motions
  - cluster motion motifs with a VAE-like motion representation
  - train RL experts in simulation
  - use rollouts as high-quality physical supervision

## Loss / Reward

- Neural retargeter: supervised regression against kinematic then physically refined G1 targets.
- CEPR experts: RL tracking reward used to project/repair reference motions into feasible robot trajectories.

## Loss / Reward Inventory

NMR uses three different objective-like layers. They should not be mixed together in our baseline config.

### 1. Neural retargeter supervised loss

The explicit retargeting-network objective in Eq. 9 is an L1 sequence regression loss:

```text
L = sum_t || m_t^bot - m_hat_t^bot ||_1
```

where the humanoid motion representation is:

```text
m_i^bot = { r_bot^x, r_bot^z, r_bot, j_bot^p, j_bot^v, q }
```

This means the paper supervises more than just joint angles in the full representation:

- root planar linear velocities
- root orientation
- local robot body positions
- local robot body velocities
- robot DoFs / joint values

The experiment section also describes the kinematic pretraining stage as minimizing regression between predicted G1 joint angles and kinematic reference targets. The safest interpretation is:

- full NMR target representation: L1 over the robot motion representation in Eq. 8/9;
- minimum practical baseline: L1 over G1 joint positions if the dataset only exposes reliable `q`;
- next supervised upgrade: add body position / velocity targets once the G1 FK feature export is stable.

### 2. Two-stage data objective, not a different model loss

NMR keeps the same supervised regression idea across two training stages:

- Stage 1: pretrain on large-scale kinematic retargeting pairs.
- Stage 2: fine-tune on about 30K CEPR physics-refined pairs.

Reported optimizer settings:

- AdamW
- batch size 128
- kinematic stage: learning rate `2e-4`, cosine annealing, 500 epochs
- CEPR fine-tune: learning rate `1e-5`, warm start, 50 epochs

For OnlineRetarget, this maps to:

- Stage A: BONES/SONIC kinematic G1 targets.
- Stage B: simulator-replayed or simulator-refined G1 targets, stored with different provenance.

### 3. CEPR expert policy rewards

These are not losses for the neural retargeter. They are rewards used to train simulator tracking experts that generate physically refined supervision.

NMR lists the expert policy tracking reward terms with weight `1.0`:

- anchor position
- anchor orientation
- anchor velocity
- body link position, relative
- body link orientation, relative
- body link linear velocity
- body link angular velocity

It lists minimal penalty terms with weight `-0.1`:

- action rate
- undesired contacts

NMR also uses an adaptive tracking-reward tolerance schedule:

```text
sigma(i) = sigma_start + (sigma_end - sigma_start) * (i - i0) / (imax - i0)
```

The policy first learns with relaxed tolerances, then the tracking reward is tightened during training. This belongs to the future Isaac Lab / CEPR-like refinement stage, not the first BONES supervised baseline.

### 4. Data filters and evaluation metrics

These are gate and metric definitions, not training losses in NMR:

- source human curation: excessive jerk, CoM far outside support base, insufficient foot-ground contact, floating, penetration
- GMR target filtering: inter-frame joint velocity above hardware-specific `qdot_max`
- self-intersection filter: reject if self-intersecting frame ratio exceeds `cross_ratio = 0.05`
- floating filter: reject if mean lowest-foot elevation exceeds `0.10 m`
- eval joint jump: max single-step joint angle change exceeds `0.5 rad`
- eval joint limit: any joint comes within `0.05 rad` of hardware boundary
- eval self-collision: non-hand body segment contact via MuJoCo FK/collision checks
- downstream tracking metrics: success rate, MPJPE, W-MPJPE

### 5. OnlineRetarget implementation decision

Immediate baseline loss should match NMR's simplest supervised interpretation:

```yaml
loss:
  l1: 1.0
  mse: 0.0
  smooth_l1: 0.0
```

Keep MSE and SmoothL1 available as ablations, but do not treat them as NMR-derived. Add body position / velocity L1 terms only after we export trustworthy G1 FK body features for every target frame.

Do not add contact, joint-limit, self-collision, or action-rate penalties into the first training loss. For now they should remain as independent validation metrics and data-quality gates. They can become regularizers later only after the model outputs sequences or simulator-rollout states rather than isolated single-frame `q`.

## Evaluation

- Joint jumps.
- Self-collisions.
- Joint-limit violations.
- Downstream BeyondMimic tracking success.
- MPJPE and W-MPJPE over short/medium/long sequences.

## Code-Level Encoder / Training Reading

Source checked: local official NMR inference repo at `/home/user/repos/MakeTrackingEasy` on 2026-05-18. The repo README marks CEPR dataset and training code as not released, so the implementation details below are exact for the released inference/model path and paper-backed for training.

### Released input encoder

NMR inference converts AMASS/SMPL-X parameters into a dense per-frame motion vector:

```text
SMPL-X params -> SMPL-X joints -> x_t in R^140
```

The 140 dimensions are:

- root planar velocity: `2`
- root orientation in 6D rotation representation: `6`
- local positions for 22 SMPL-X joints: `22 * 3 = 66`
- local joint velocities for 22 SMPL-X joints: `22 * 3 = 66`

The preprocessing also:

- converts AMASS Z-up input to NMR's Y-up convention when needed;
- downsamples high-FPS input to 30 FPS;
- subtracts an initial ground/root origin;
- keeps local X/Z joint positions relative to the root;
- canonicalizes yaw per chunk before inference, then rotates predictions back afterward;
- normalizes input with `weights/smplx_mean.npy` and `weights/smplx_std.npy`.

This is the most important representation-level lesson for OnlineRetarget: the encoder is not raw joint Euler angles. It is root-motion plus root orientation plus local joint position/velocity features.

### Released neural encoder stack

The released model config uses:

```text
SMPL-X motion (B, T, 140)
  -> VQVAE.preprocess: (B, 140, T)
  -> EncoderAttn
  -> encoded feature (B, 512, T/4)
  -> permute (B, T/4, 512)
  -> Linear(512 -> 512), SiLU, Linear(512 -> 512)
  -> LLaMAHF_Fwd transformer
  -> upsample x2, Conv1d, upsample x2, Conv1d
  -> Linear(512 -> 217)
  -> G1 motion (B, T, 217)
```

`EncoderAttn` is a temporal Conv1D encoder with two stride-2 stages. Each stage contains:

- Conv1D temporal downsampling;
- `Resnet1D` with dilated residual Conv1D blocks;
- a small full-attention block over the downsampled time axis;
- final Conv1D projection to 512 channels.

The repo includes an FSQ quantizer and VQ-VAE decode path, but the released retarget model class is `RetargetTransformerPredMotion_no_smplvq`. In that model, retarget inference uses the SMPL-X VQ-VAE encoder output directly as continuous features; it does not quantize the source feature before the transformer.

### Transformer behavior

The released transformer is `LLaMAHF_Fwd`, a forward/non-autoregressive variant:

- no token embedding table is used;
- input is already a continuous embedding sequence;
- it applies LLaMA-style blocks with RMSNorm, RoPE, full self-attention over valid condition frames, and SwiGLU MLP;
- config in the released checkpoint is `n_layer=8`, `n_head=8`, `n_embd=512`, `block_size=1024`.

Although the base attention class is named causal, its mask explicitly allows all valid condition frames to attend to each other. This implements the paper's bidirectional temporal-context claim: each predicted frame can use both past and future frames inside the current chunk.

### Released output representation

The network predicts a 217D G1 motion representation:

- root planar velocity: `2`
- root orientation in 6D: `6`
- local/world-restored positions for 30 G1 bodies: `30 * 3 = 90`
- body velocities for 30 G1 bodies: `30 * 3 = 90`
- G1 joint DoFs: `29`

Postprocess extracts:

- `joint_pos = pred_motion[:, -29:]`
- root translation from predicted body-0 position plus integrated planar velocity;
- root quaternion from predicted 6D root orientation;
- optional Butterworth low-pass filtering for translation, root quaternion, and DoFs.

The CLI then converts 30 FPS network output to bmimic-style G1 NPZ at 50 FPS with `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.

### Training scheme from paper / README

The official training script is not released. The paper describes the train logic as two-stage supervised regression:

1. Kinematic pretraining on large SMPL -> G1 kinematic retarget pairs.
2. Physics grounding fine-tune on about 30K CEPR simulator-refined SMPL -> G1 pairs.

The paper reports:

- optimizer: AdamW;
- batch size: 128;
- stage-1 learning rate: `2e-4`;
- schedule: cosine annealing;
- stage-1 length: 500 epochs;
- stage-2 learning rate: `1e-5`;
- stage-2 length: 50 epochs;
- objective: L1 sequence regression against the robot motion representation.

The CEPR expert policies are separate data-generation controllers, not the neural retargeter's training loss. Their reward terms are used to produce physically refined supervision, then the neural retargeter is trained by supervised L1.

### Direct mapping to OnlineRetarget

For our `SOMA proportional -> G1` baseline, the closest NMR-faithful design is:

- replace SMPL-X 140D with SOMA FK root/local joint position and velocity features plus morphology;
- keep temporal context;
- start with direct L1 over G1 `joint_pos` because BONES/SONIC reliably exposes 29D joint targets;
- later extend the target to NMR-like 217D G1 motion once our G1 FK/body velocity export is stable;
- keep bidirectional offline sequence models for quality experiments, but separately build a causal/short-window deployment model if strict online inference latency requires it.

## Implications

- Strong evidence for our direct neural retargeter baseline.
- Physics refinement is important, but should come after the baseline because it requires simulator-scale infrastructure.
- Use NMR's artifact metrics early, before downstream policy training.
- For BONES, implement an NMR-like supervised retargeter over SOMA BVH FK features rather than forcing BONES through SMPL-X just to match NMR's released inference path.
- Keep target provenance explicit: kinematic BONES/SONIC targets first, simulator-refined targets later.
