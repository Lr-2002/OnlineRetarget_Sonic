# ADR 0001 - Start With Direct G1 Supervision

Date: 2026-05-13

## Context

The project needs a learning-based online retargeter from heterogeneous human skeleton data to Unitree G1. The user cares about retarget quality, physics-aware evaluation, cross-skeleton generalization, and sub-1 ms inference on an RTX 4090.

The local data root contains BONES-SEED metadata, SOMA source skeleton paths, G1 CSV targets, and processed G1 NPZ targets with joint and body states.

## Decision

Start with a compact direct-output supervised retargeter that predicts G1 joint targets or deltas from source skeleton history plus morphology and robot state.

Keep latent, diffusion, flow matching, and physics-refined target generation as later branches gated by measured baseline failures.

## Rationale

- Direct G1 output is easiest to evaluate and deploy under the 1 ms latency budget.
- BONES-SEED already contains G1 targets for every metadata row.
- NMR supports the learned-retargeter direction, but its CEPR pipeline is too heavy for the first repo milestone.
- PDF-HR supports a lightweight pose-prior branch, but that branch needs a trusted positive pose set and should be added after baseline metrics are stable.

## Consequences

- The first code emphasis is data inventory, actor-level splits, independent metrics, and traceable training.
- Simulator integration is deferred until offline prediction quality is measurable.
- Latent models remain research branches, not baseline infrastructure.
