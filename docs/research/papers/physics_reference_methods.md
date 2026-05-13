# Physics-Consistent Reference Methods Reading Note

Primary sources:

- OmniTrack: https://arxiv.org/abs/2602.23832
- ULTRA: https://arxiv.org/abs/2603.03279
- OmniRetarget: https://arxiv.org/abs/2509.26633
- ReActor: https://arxiv.org/abs/2605.06593
- ProtoMotions G1 workflow: https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html

## Shared Pattern

These works treat raw or kinematic retargeted references as insufficient. They introduce a simulator or constrained optimization layer that converts source motion into physically plausible robot references, then train policies on those references.

## Observation / Input Patterns

- Proprioception: joint positions/velocities, previous action.
- IMU/base state: torso orientation, pelvis angular velocity, projected gravity.
- Reference features: current/future body or joint references.
- Residual features: simulation-reference deltas in heading-aligned frames.
- Contact/object features for loco-manipulation variants.

## Output Patterns

- PD position targets.
- Joint-level actions.
- Physics-consistent reference rollouts.

## Reward Patterns

- Body tracking: position, orientation, linear velocity, angular velocity.
- Object tracking for interaction tasks.
- Action rate penalty.
- Soft joint-limit penalty.
- Self-collision penalty.
- Contact/foot stability terms.
- Domain randomization for friction, mass, COM, pushes, and observation noise.

## Evaluation Patterns

- Success / fall rate.
- Episode length.
- MPJPE / W-MPJPE / mean body position error.
- Joint position/velocity/acceleration error.
- Ground penetration.
- Foot slide / foot float.
- Self-penetration/collision.
- Sim-to-sim and sim-to-real transfer.
- Online teleoperation robustness under jitter/latency.

## Implications

- M7 must evaluate physical feasibility, not just reconstruction error.
- Before M7, M4 should already compute jump/limit/body errors so failure attribution is possible.
- The online retargeter should emit a clear deployment sidecar: input schema, output schema, normalization, frequency, and latency.
