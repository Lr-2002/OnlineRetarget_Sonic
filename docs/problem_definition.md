# Problem Definition

Current target: learn an online mapping from **BONES-SEED SOMA proportional human motion** to **Unitree G1 motion references**.

```text
SOMA proportional BVH motion
+ actor morphology / SOMA shape metadata
+ optional current G1 state
    -> online retargeter
    -> Unitree G1 29-DoF joint target
```

## Source

The source side is BONES-SEED `SOMA proportional`, not `SOMA uniform`.

- Motion path: `/home/user/data/motion_data/soma_proportional.tar`
- Metadata path: `/home/user/data/motion_data/metadata/seed_metadata_v003.csv`
- Metadata column: `move_soma_proportional_path`
- Actor shape column: `move_soma_proportional_shape_path`
- Grouping key: `actor_uid`

`SOMA proportional` keeps the same SOMA topology across actors, but preserves actor-specific offsets, limb lengths, and shape parameters. Local metadata references 522 actor-specific proportional shape files. This is the intended source of cross-person / cross-skeleton variation.

`SOMA uniform` is a later ablation only. It uses one shared skeleton and does not test the main heterogeneous-skeleton problem.

## Target

The target side is Unitree G1 motion.

Primary active target lane:

- Path: `/home/user/data/motion_data/bones_sonic/**/*.npz`
- Joined from metadata by converting:

```text
g1/csv/<date>/<name>.csv -> bones_sonic/<date>/<name>.npz
```

Target tensors:

- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`

Initial supervised output is direct 29D G1 joint position, with velocity/body/contact terms used mainly as evaluation diagnostics until the target provenance and simulator gate are stronger.

Legacy `g1.tar` CSV files remain useful for provenance and parser regression, but the current baseline should use `bones_sonic` NPZ targets unless a specific experiment says otherwise.

## Learning Task

For each paired metadata row, build fixed-window samples:

```text
input_t =
  SOMA proportional source history around frame t
  + source velocities
  + actor morphology / shape conditioning
  + optional G1 current state side channel

target_t =
  G1 joint_pos at aligned frame t
```

The first model family is a compact temporal MLP / small temporal encoder. A mid-term skeleton encoder is a planned design branch: it should encode SOMA topology, bone proportions, local motion/contact features, and shape metadata into model-facing features before G1 prediction. Transformer, flow matching, VAE, and diffusion variants remain research branches after the direct supervised baseline is measurable.

## Split And Evaluation

Splits must be actor-heldout by `actor_uid`. Clip-level random split is not acceptable because it leaks the same proportional skeleton into train and eval.

Primary eval answers:

- Does the model map unseen SOMA proportional actors to G1 joint targets?
- Does morphology / SOMA shape conditioning improve actor-heldout metrics?
- Does the model introduce temporal artifacts beyond those already present in the target?

Initial metrics:

- G1 joint MAE/RMSE and max joint absolute error
- action similarity
- joint velocity RMSE
- predicted vs target joint jump rate
- optional body MPJPE / contact artifact metrics when body positions are emitted

## Out Of Scope For The Baseline

- `SOMA uniform -> G1` as the main task
- `G1 body_pos_w -> G1 joint_pos` target-state reconstruction as proof of retargeting
- AMASS/GMR retargeted NPZ as source-side human skeleton data
- simulator physics refinement as a replacement for target provenance
- latent/VAE/flow/diffusion before the direct mapping baseline is measured
