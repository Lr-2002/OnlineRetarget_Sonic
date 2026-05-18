# Selection: Collision- and Contact-Aware Embodiment Retargeting

Date: 2026-05-18

## Why This Selection Exists

This project direction is not simply "make retargeted motion look closer to
human motion." The stronger selection is:

> Learn an embodiment-aware retargeting / motion generation system that keeps
> source motion semantics while avoiding unwanted self-collision and unwanted
> environment collision, preserving wanted contacts, and producing robot motion
> that is easier for a downstream controller to track.

The key distinction is:

```text
Retargeting asks: what trajectory should the robot be asked to follow?
Tracking asks: how should the robot follow that trajectory?
```

If the reference trajectory already asks the robot to put an arm through its
torso, penetrate a backpack, exceed joint limits, or slide a foot through the
ground, then a stronger tracker is forced into a bad tradeoff:

```text
track the reference accurately -> illegal collision/contact
avoid collision/contact         -> large tracking error
```

In that case, improving the tracker does not solve the real problem. The
reference itself is outside the target robot's feasible motion manifold. A
collision/contact-aware retargeter tries to avoid handing the controller an
impossible target in the first place.

## Project Goal

Given:

- source motion,
- source body / skeleton / mesh information,
- target robot URDF/MJCF,
- target robot collision geometry, contact sites, limits, and possibly
  attachments,
- optional environment / object geometry,

generate:

- target robot `q`,
- target robot `qvel`,
- action or PD target when appropriate,
- contact state / contact intent,
- collision risk / feasibility score,

such that:

- source motion semantics are preserved,
- target joint, velocity, and torque limits are respected,
- unwanted self-collision is avoided,
- unwanted environment/object penetration is avoided,
- wanted contacts are preserved or created when semantically necessary,
- downstream tracking is easier and more stable,
- inference can remain online / low latency.

This makes the research target:

```text
collision- and contact-aware embodiment-conditioned retargeting
```

not just generic motion retargeting.

## Why URDF / MJCF Matters

URDF/MJCF is valuable because it gives the model and system access to the
target embodiment:

- joint tree and parent-child structure,
- joint type, axis, and limits,
- link frames and transforms,
- collision meshes or primitive collision geometry,
- DAE/STL mesh-derived geometry,
- mass and inertia,
- actuator or PD-control metadata when available,
- foot/hand/contact sites,
- self-collision masks and prohibited collision pairs,
- attachments such as backpacks, hands, tools, or added collision bodies.

The important point is that URDF/MJCF-derived tokens are not magic. Encoding
robot metadata into tokens does not by itself teach the model what collision
means. The system also needs explicit feasibility supervision and evaluation.

## Wanted Contact vs Unwanted Collision

The goal is not to avoid all contacts. That would destroy many useful motions.

Wanted contact examples:

- feet on ground,
- hand on object,
- hand on wall,
- knee on ground,
- sitting contact,
- object carrying contact.

Unwanted collision examples:

- hand through torso,
- elbow through backpack,
- foot penetrating ground,
- thigh crossing through the other leg,
- large hand geometry colliding with waist,
- object penetration caused by bad retargeting.

The system should therefore learn and evaluate:

```text
avoid unwanted collision
preserve wanted contact
```

## Complete System Features

A future complete system should include at least these layers.

### 1. Source Representation

Source-side information should not be reduced too early to a fixed small human
skeleton. The source body itself matters.

Possible source inputs:

- source motion tokens,
- source skeleton topology,
- bone lengths,
- SMPL/SMPL-X shape parameters when available,
- source mesh or surface anchors,
- marker/BVH convention metadata,
- source velocity and timing,
- inferred foot/hand/body contact evidence.

Purpose:

- understand what the source motion means on that body,
- separate motion semantics from source-specific body proportions,
- avoid mistaking body-shape differences for motion intent.

### 2. Target Robot Representation

The target robot should be represented as a variable-size embodiment token set,
not only as a single robot ID.

Possible target tokens:

- joint tokens: type, axis, limits, parent, child, damping, actuator limits,
- link tokens: length, frame, mass, inertia, local geometry descriptors,
- collision tokens: mesh proxy, SDF latent, capsules, convex parts, spheres,
- contact-site tokens: foot sole, toe, heel, palm, fingertips, knee, torso,
- attachment tokens: backpack, hand variant, tool, external payload,
- global robot token: total mass, scale, nominal stance, DoF summary.

Purpose:

- support different humanoids,
- support changed hands or hand sizes,
- support backpacks and added collision bodies,
- support robot-specific collision/contact constraints.

### 3. Correspondence / Affordance Layer

Retargeting cannot rely only on fixed "left hand to left hand" mappings.
Different target robots may have:

- no dexterous hand,
- a large hand,
- a two-finger gripper,
- a five-finger hand,
- extra tools,
- limited shoulder range,
- different feet or support geometry.

The system needs to represent which source parts correspond to which target
links, sites, or affordances, and when exact correspondence should be relaxed.

### 4. Feasibility Evaluator

This is the critical layer. It must be independent from the neural model.

It should measure:

- joint limit violation,
- velocity / acceleration violation,
- self-collision,
- environment/object penetration,
- penetration depth,
- foot sliding,
- contact mismatch,
- wanted contact preservation,
- support/contact consistency,
- action smoothness,
- downstream tracking success,
- latency.

Without this evaluator, robot tokens and attention can become decorative
metadata rather than a constraint-aware model.

### 5. Retargeting Model

The model should be target-embodiment-conditioned. It may output:

- target `q`,
- target `qvel`,
- delta `q`,
- action / PD target,
- target link/site trajectories,
- contact probabilities,
- collision/contact risk,
- confidence or OOD score.

A useful conceptual model is:

```text
source motion + source body + target robot + optional environment
  -> target motion + contact intent + feasibility estimates
```

The exact architecture is a design choice. Possible options include:

- deterministic temporal MLP / TCN / Transformer,
- robot-token cross-attention model,
- graph transformer over body and robot tokens,
- residual correction / refiner,
- one-step or few-step projector,
- diffusion / flow matching as an offline teacher or later-stage projector.

### 6. Online Safety / Recovery Layer

Even a good model should not be trusted blindly. A low-latency safety layer may
include:

- joint-limit projection,
- velocity smoothing,
- contact-state hysteresis,
- local collision-aware correction,
- rejection of unsafe outputs,
- fallback to previous safe motion,
- bounded IK/projection with a fixed compute budget.

This is not meant to replace the model. It keeps the online system from failing
catastrophically when the model is uncertain or out of distribution.

### 7. Offline Physics / Collision Refinement

Offline processing can be more expensive and should generate supervision for
the online model.

Possible offline sources:

- MuJoCo collision checks,
- Isaac Lab tracking evaluation,
- FCL / mesh collision labels,
- signed distance labels,
- trajectory optimization,
- RL-based physics refinement,
- generated negative samples near feasibility boundaries.

The online model should learn from these labels, rather than doing expensive
per-frame optimization during inference.

## Architecture Discussion

One candidate sketch was:

```text
source motion tokens
+ source body tokens
        |
        v
source-aware motion encoder
        |
        v
canonical / semantic motion latent
        |
        v
target proposal decoder  <---- cross attention ---- target robot tokens
        |
        v
robot-aware refiner      <---- cross attention ---- target robot tokens
        |
        v
final q, qvel, contact, collision-risk
```

This should be treated as a design candidate, not a fixed answer.

The reason to consider attention in more than one place is that the query
changes:

- semantic motion tokens query the target robot to decide how an action should
  be expressed on this embodiment;
- predicted target state tokens query robot geometry/contact tokens to decide
  whether the current proposal is feasible and how to correct it.

But the system does not require exactly two cross-attention blocks. A single
iterative target-aware block, graph transformer, or deterministic refiner could
serve the same purpose if it provides the same information flow.

The principle is:

```text
target embodiment information should affect both generation and feasibility
assessment/refinement.
```

## Main Risks

### Robot Tokens May Not Become Constraint Understanding

The model may learn robot IDs or morphology correlations, not true geometry
constraints. This is especially likely if training data does not include
negative poses or OOD embodiment tests.

Mitigation:

- add collision/contact labels,
- add near-boundary negative samples,
- hold out full robot/attachment/hand variants,
- test whether changed geometry changes the output.

### Contact-Rich Motion May Be Under-Specified

Many contact intents are not visible from skeleton motion alone. Sitting,
kneeling, hand support, carrying, and object interaction require explicit
contact evidence or environment/object tokens.

Mitigation:

- infer or annotate contact labels,
- evaluate by contact category,
- include environment/object geometry when contact matters.

### Mesh Representation Can Be Too Heavy Or Too Weak

Full DAE/STL tokenization is likely too expensive and hard to train. But overly
coarse capsules can miss important geometry, especially for hands, feet,
backpacks, and tools.

Mitigation:

- start with multi-resolution geometry proxies,
- use collision primitives, convex parts, selected SDF queries, contact patches,
  and high-risk collision-pair features,
- reserve full mesh checks for offline labeling and validation.

### Diffusion / Flow Matching May Conflict With Online Latency

Generative projectors can be useful, but multi-step sampling may violate the
online budget and introduce nondeterminism.

Mitigation:

- first test deterministic models and one-step refiners,
- use diffusion/FM as offline teacher or data generator,
- only consider online use after distillation to a small number of steps.

### Tracking Quality May Hide Reference Quality

A strong tracker may compensate for bad references, making retargeting quality
look less important. But that compensation may require extra reward shaping,
training time, or unsafe behavior.

Mitigation:

- evaluate the same tracker on baseline references and collision-aware
  references,
- compare convergence, tracking success, collision rate, and failure cases.

## Minimum Proof Demo

The most convincing demo should show that the problem cannot be solved cleanly
by only improving the tracker.

The core demo:

```text
same source motion
same tracker
same control budget
different target embodiment geometry
```

Example setup:

- source motion: crossed arms, hugging chest, arm swing near torso, reaching
  behind body, hand support, sitting or kneeling;
- target robot A: normal Unitree G1;
- target robot B: G1 plus backpack collision mesh, larger hand, changed foot
  sole, or changed forearm geometry.

Compare:

```text
baseline reference + same tracker
ours reference     + same tracker
```

The expected result:

- baseline reference collides with torso/backpack/ground or creates contact
  inconsistency;
- the tracker either follows the bad reference into collision or deviates from
  it and produces high tracking error;
- our reference changes the arm/shoulder/elbow/contact trajectory according to
  target geometry;
- the same tracker succeeds more often with lower collision/contact violations.

Recommended metrics:

- self-collision frames,
- max penetration depth,
- unwanted environment collision,
- wanted contact preservation,
- foot sliding,
- joint limit violation,
- tracking success,
- inference latency.

Video layout:

```text
source human motion | baseline retarget reference | baseline tracked robot | ours tracked robot
```

This directly tests the value claim:

> A feasible reference reduces the burden on the downstream tracker.

## Minimum Falsifiable Experiments

### Experiment A: Embodiment OOD Generalization

Train on several robot variants and hold out one full variant:

- link length scale,
- joint limit changes,
- hand size/topology,
- foot geometry,
- backpack collision mesh.

Compare:

- per-target direct baseline,
- IK/GMR-style baseline,
- robot-feature-conditioned MLP,
- robot-token attention model,
- robot-token model plus one-step refiner.

Fail condition:

- proposed model does not reduce constraint violations on held-out embodiment,
  or only improves visual similarity while feasibility remains poor.

### Experiment B: Geometry Sensitivity

Use the same source motion and same robot, then add a backpack or enlarge the
hand.

Fail condition:

- model output does not change in response to the changed geometry, and
  collisions increase.

### Experiment C: Contact-Rich Stress Test

Evaluate sitting, kneeling, hand support, jumping/landing, carrying, and
foot-to-platform motions.

Fail condition:

- normal locomotion works, but contact-rich motions fail systematically.

### Experiment D: Latency / Quality Pareto

Compare:

- direct model,
- direct model plus one-step refiner,
- diffusion/FM with 2, 4, and 8 steps.

Fail condition:

- quality gains require too many steps for online inference.

## Practical Selection Statement

This direction is worth pursuing if the following hypothesis can be validated:

> Source-body and target-robot embodiment representations, combined with
> explicit feasibility supervision, allow retargeting to generalize to unseen
> robot geometry, attachments, hands, and body shapes better than naive
> q-regression or tracker-only compensation.

The first milestone should not be a large diffusion/FM model. It should be:

1. define source and target embodiment schemas,
2. build collision/contact/feasibility evaluation,
3. create simple robot-conditioned baselines,
4. design OOD robot/attachment tests,
5. prove whether embodiment information actually changes output and reduces
   unwanted collision/contact violations.

Only after that should more complex cross-attention, refiner, diffusion, or flow
matching models be introduced.
