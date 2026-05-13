# GMR / BeyondMimic Reading Note

Primary sources:

- GMR / Retargeting Matters: https://arxiv.org/abs/2510.02252
- BeyondMimic code: https://github.com/HybridRobotics/whole_body_tracking
- BeyondMimic paper: https://arxiv.org/abs/2508.08241

## Core Claim

Retargeting quality directly affects downstream motion tracking. Policies can sometimes compensate for imperfect references, but artifacts such as ground penetration, self-intersection, and sudden joint jumps reduce robustness and fidelity.

## Observation / Input

- Retargeted reference motion.
- Robot proprioception and reference features inside the tracking policy.
- BeyondMimic uses Isaac Lab-style tracking tasks and WandB motion artifact management.

## Output

- Retargeting: G1 reference trajectories.
- Tracking: policy actions / joint targets.

## Model

- GMR: optimization-based retargeting with body correspondences and IK objectives.
- BeyondMimic: RL motion tracking; later guided diffusion for versatile control.

## Data

- LAFAN1 retargets and Unitree/GMR/PHC/ProtoMotions references.

## Reward / Loss

- GMR: IK objective over Cartesian positions/orientations.
- BeyondMimic: DeepMimic-style body pose/velocity/orientation tracking and smoothing/regularization.

## Evaluation

- Success rate.
- Global body position error.
- Root-relative body position error.
- Joint rotation error.
- Robustness under observation noise, model mismatch, and latency.
- User study for perceptual faithfulness.

## Implications

- Our eval suite must expose artifacts before simulator training.
- M7 should align with BeyondMimic-style tracking metrics.
- WandB artifact discipline is important because retargeted motion data and policy runs are tightly linked.
