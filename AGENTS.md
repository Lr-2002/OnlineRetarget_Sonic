# OnlineRetarget Agent Contract

This repository targets a learning-based online retargeter from heterogeneous human skeleton motion to Unitree G1 robot motion.

## Non-Negotiables

- Treat `/home/user/data/motion_data` as read-only. Derived data, caches, checkpoints, and logs must go under `runs/`, `outputs/`, or another explicit repo-local/output path.
- Keep code simple before adding abstractions. Prefer direct supervised baselines and independent evaluation modules before generative or simulator-heavy variants.
- Cite papers or source code in docs when a design is derived from prior work.
- Keep training runs traceable: commit before meaningful training, log config and git SHA to WandB, and launch long training inside tmux.
- Use Isaac Lab for simulation stages. This repo may contain adapters and configs, but simulator-specific assets should remain clearly separated.
- Preserve DDP readiness for training code: use rank/world-size aware logging, deterministic config capture, and output paths that do not collide across ranks.

## Current Technical Direction

- First output target: direct G1 generalized motion, beginning with joint position/velocity supervision from existing G1 targets.
- First model family: compact MLP/temporal MLP or small temporal transformer only after the MLP baseline is measured against the 1 ms inference budget on a 4090.
- First metrics: MPJPE/body position error, G1 joint RMSE, action similarity, joint jump rate, joint limit violation rate, and downstream tracking success once Isaac Lab integration exists.
- Latent/VAE, flow matching, and diffusion are research branches, not the baseline path, unless evaluation shows the compact direct model is insufficient.

## Documentation Discipline

- Update `docs/research/literature_review.md` when new papers materially change architecture or evaluation choices.
- Update `docs/research/data_inventory.md` when data layout, skeleton grouping, or loader assumptions change.
- Update `docs/milestones.md` when a milestone gate is completed or re-scoped.
