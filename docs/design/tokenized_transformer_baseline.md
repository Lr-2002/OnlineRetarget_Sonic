# Tokenized Transformer Baseline Design

Date: 2026-05-18

Status: design checkpoint for the next implementation pass.

## Goal

Build the next retargeting baseline around continuous tokens:

```text
SOMA proportional skeleton and motion tokens
+ previous G1 reference/state token
+ next-frame query token
    -> Transformer decoder
    -> Unitree G1 29D next-frame joint target
```

The immediate implementation target is still offline supervised training on BONES
`SOMA proportional BVH -> BONES-SONIC G1 joint_pos` pairs. The design must remain
compatible with online inference, autoregressive rollout, and a later Flow
Matching decoder.

Default latent token size: `128`.

## Core Representation Choices

### Continuous Tokens, Not Discrete Tokens

Motion and robot actions are continuous control signals. The first baseline should
use continuous token embeddings, not a VQ/discrete tokenizer.

Reasons:

- Joint-space and FK-MPJPE errors are sensitive to small continuous errors.
- Discrete codebooks add quantization error and codebook-collapse failure modes.
- Online deployment needs smooth control and debuggable residuals.
- The current data already provides numeric skeleton, motion, and G1 state values.

Discrete or VQ tokenization remains a later motion-prior branch only if continuous
encoders fail to model useful motion semantics.

### Skeleton Token

The skeleton token is a projection of physically meaningful numeric values. It
does not need a complicated encoder in the first pass.

Candidate raw features:

- Rest/T-pose or proposal-file body positions.
- Bone vector and bone length.
- Parent index or parent-relative offset.
- Body type or body semantic id when available.
- Actor morphology and SOMA shape measurements when available.

First implementation:

```text
raw skeleton numeric vector -> MLP encoder -> z_skel
z_skel -> MLP decoder -> reconstructed skeleton vector
```

The skeleton autoencoder is intentionally simple because the local data has about
500 actor skeletons and the static numeric structure should be easy to compress.

### Motion Token

Motion tokens should also be continuous.

Candidate raw features:

- Source body positions over a history window.
- Source body velocities.
- Optional rotations/contact proxies when the sample builder emits them.

First implementation:

```text
raw motion window -> MLP or TCN encoder -> z_motion
z_motion -> MLP decoder -> reconstructed motion window
```

Temporal relationships can be modeled either inside a TCN encoder or by the
Transformer consuming frame/body tokens. The first code path can start with an
MLP encoder over the current 1,547D observation, then split into a more structured
body/time encoder after rollout eval exposes the failure modes.

### Previous G1 State / Action Token

This token must be interpreted carefully.

Offline training generally does not have real robot measured state. It has the
paired G1 reference trajectory. Therefore:

```text
training:
  prev-state token = ground-truth G1 q[t-1], optionally dq[t-1]

inference:
  prev-state token = previous predicted q_hat[t-1]
  or measured robot q[t-1] if the deployed controller provides it
```

This creates a teacher-forcing versus autoregressive inference gap. It must be
tracked as a known risk. Mitigations:

- Add noise to the previous-state token during training.
- Run closed-loop rollout eval, not only one-step eval.
- Later add scheduled sampling or rollout loss if drift is observed.

First implementation:

```text
prev G1 q -> MLP encoder -> z_state
z_state -> MLP decoder -> reconstructed q
```

### Query Token

The query token is the learned "next frame" request. It is analogous to a next
frame token in an autoregressive model, but it is continuous and task-specific.

For deterministic Transformer prediction:

```text
query token + prev-state token cross-attend source memory -> q[t] or delta_q[t]
```

For Flow Matching:

```text
x_tau token + time embedding + prev-state token cross-attend source memory
    -> vector field v_theta
```

## Transformer Design Choices

### Preferred First Transformer: Cross-Attention Decoder

Use source skeleton/motion as memory, and use robot/query tokens as decoder
queries.

```text
source memory:
  z_skel + z_motion

decoder query:
  z_prev_state + learned next-frame query

decoder output:
  MLP head -> 29D q[t] or delta_q[t]
```

This is preferred over early fusion because retargeting is naturally a query over
source motion and skeleton context. The generated G1 state asks what source
motion/skeleton information is relevant for the next G1 pose.

### Early Fusion Alternative

```text
[skeleton tokens, motion tokens, prev-state token, query token]
    -> self-attention encoder
    -> MLP head -> 29D q[t]
```

This is simpler, but less structured. It should be kept as a fallback or ablation.

### Joint Query Alternative

A later, more interpretable variant can use one query per G1 joint:

```text
29 G1 joint query tokens cross-attend source memory
    -> per-joint head -> 29D q[t]
```

This may help inspect whether ankle queries attend to legs/feet and shoulder
queries attend to arms, but it adds complexity. It is not required for the first
Transformer baseline.

## Shared Latent Direction

The long-term representation goal is a shared motion/action latent:

```text
E_h(human skeleton motion) -> z
E_g(G1 motion/state)       -> z
D_h(z, human skeleton)     -> human motion
D_g(z, G1 skeleton/state)  -> G1 motion
```

Possible losses:

```text
human reconstruction: D_h(E_h(human)) ~= human
robot reconstruction: D_g(E_g(g1)) ~= g1
paired alignment:     E_h(human_paired) ~= E_g(g1_paired)
retargeting:          D_g(E_h(human), prev_g1) ~= g1_target
```

Do not start with a full VAE/shared autoencoder unless the deterministic
encoder-decoder exposes a clear need. The first implementation should use simple
autoencoder heads and optional latent alignment so the training path stays
debuggable.

## Flow Matching Branch

Flow Matching is a decoder/training objective, not a replacement for tokenization.
It can reuse the same source memory and state tokens.

Preferred branch:

```text
condition c_t = z_skel + z_motion + z_prev_state
x0 = q[t-1] + sigma * epsilon
x1 = q[t]
x_tau = (1 - tau) * x0 + tau * x1

v_theta(x_tau, tau, c_t) ~= x1 - x0
```

This is residual conditional Flow Matching. It is preferred over pure
noise-to-motion because online retargeting is strongly conditioned and requires
temporal continuity.

Frame-level interpretation:

```text
within frame: Flow Matching maps x0 -> q[t]
between frames: autoregressive rollout feeds q_hat[t] into the next step
```

Known risks:

- Multi-step inference may violate the 1 ms latency target.
- One-step Flow Matching can collapse into a delta predictor.
- Training with ground-truth previous state can still drift during rollout.

Required eval:

- One-step joint metrics.
- Closed-loop rollout metrics.
- FK-MPJPE and end-effector MPJPE.
- Latency for 1, 2, 4, and 8 FM steps.

## First Implementation Plan

### Stage 1: Independent 128D Encoders

Train simple autoencoder losses inside the supervised run:

```text
z_skel  = E_skel(skeleton_features)
z_motion = E_motion(observation_or_motion_features)
z_state = E_state(prev_g1_q)

reconstruct skeleton_features, motion_features, prev_g1_q
```

If explicit skeleton features are not yet emitted by the dataset, use the current
morphology/static side channel as the first skeleton vector and keep the interface
ready for richer T-pose/proposal features.

### Stage 2: Transformer Retargeter

```text
memory = [z_skel, z_motion]
query = [z_state + learned next query]
decoder(query, memory) -> z_next
head(z_next) -> delta_q or q[t]
```

First output should be direct `q[t]` or `delta_q[t]`. Prefer `delta_q[t]` once the
dataset reliably emits `prev_g1_q`.

### Stage 3: Joint Training

The simplest joint objective:

```text
L = L_retarget
  + lambda_skel * L_skeleton_recon
  + lambda_motion * L_motion_recon
  + lambda_state * L_state_recon
```

Optional later:

```text
L += lambda_align * ||z_motion - z_state||^2
```

Do not add KL/VAE terms until reconstruction, retargeting, and rollout eval are
stable.

### Stage 4: Rollout And Remote Training

Before remote 8-GPU training:

- Commit the local implementation.
- Record exact config, git SHA, data artifacts, and quality gate status.
- Verify WandB logging locally in offline or online mode.
- Prepare a handoff note for the remote server with environment, commands, and
  expected output paths.

Remote training can then use the provided SSH target after the local scaffold and
tests are green.

## Acceptance For This Design Branch

The Transformer baseline branch is useful only if it produces comparable or
better evidence than the current MLP:

- Actor-heldout validation joint metrics.
- FK-MPJPE auto eval or a clearly documented supplemental eval.
- Closed-loop rollout eval on sequence samples.
- Batch-1 latency benchmark on the target GPU.
- WandB run with config, git SHA, dataset paths, and eval artifacts.

Until those exist, it should be reported as a scaffold, not as an improved
retargeter.
