# PDF-HR Reading Note

Primary source: https://arxiv.org/abs/2602.04851

## Core Claim

A lightweight pose-distance field can model the plausible G1 pose manifold and act as a differentiable prior for tracking, retargeting, optimization, or scoring.

## Observation / Input

- Query robot pose `q`.
- The model is not a source-to-target retargeter by itself.

## Output

- Scalar distance / plausibility score: larger values indicate farther from the learned pose manifold.

## Model

- MLP distance field.

## Data

- Large corpus of retargeted humanoid robot poses.
- Positive/near/far pose samples for distance-field training.

## Loss / Reward

- Supervised regression to distance-to-manifold targets.
- Used downstream as regularization or reward shaping.

## Evaluation

- Single-trajectory tracking.
- General motion tracking.
- Style-conditioned mimicry.
- Motion retargeting quality.

## Implications

- Best used as a later regularizer/scorer for predicted G1 poses.
- Compatible with online latency if implemented as a small MLP.
- Requires a trusted positive pose set and confirmed G1 joint limits.
