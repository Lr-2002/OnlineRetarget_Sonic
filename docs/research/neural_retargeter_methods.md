# Neural Motion Retargeter Method Taxonomy

Date: 2026-05-18.

Purpose: keep the model-design discussion paper-backed. This note separates
methods that can be the online retargeter itself from methods that are better
treated as regularizers, data generators, controllers, or physics-refinement
stages.

## Method Families

| Family | Paper support | Input | Output | Role for OnlineRetarget |
| --- | --- | --- | --- | --- |
| Direct supervised sequence retargeter | NMR: Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control, arXiv:2603.22201 | Human SMPL/SMPL-X or source skeleton motion sequence, temporal context, optional morphology | Unitree G1 feasible motion sequence / joint angles | Primary research reference for a learned human-to-G1 retargeter. Our first version should be a compact temporal MLP/TCN/small Transformer with the same supervised mapping idea, not the full NMR pipeline. |
| Skeleton-topology neural retargeting | Neural Kinematic Networks for Unsupervised Motion Retargetting; Skeleton-Aware Networks for Deep Motion Retargeting | Source skeleton joint rotations/positions plus target skeleton topology or offsets | Retargeted target-skeleton motion | Supports the general idea that skeleton structure and bone lengths should be explicit inputs/features. These are character-animation papers, so their outputs are not G1 physics-valid by themselves. |
| Shared latent retargeting | Nonparametric Motion Retargeting for Humanoid Robots on Shared Latent Space, RSS 2020 | Mocap Cartesian joints and robot joint-space poses | Robot pose decoded/regressed from shared latent space | Valid latent branch if direct output underfits cross-skeleton morphology. Not first baseline because it adds encoder/decoder ambiguity before direct G1 metrics are stable. |
| Online / unpaired human-to-robot latent retargeting | ImitationNet: Unsupervised Human-to-Robot Motion Retargeting via Shared Latent Space, arXiv:2309.05310 / Humanoids 2023 | Human pose stream, robot pose space, cross-domain similarity features | Robot pose or latent command decoded to robot motion | Supports unpaired human-to-robot retargeting and online-style control. Relevant if BONES source-target pairing is incomplete or if we add new robots later, but weaker G1-specific evidence than NMR. |
| Physics-aware RL-coupled retargeting | ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting, arXiv:2605.06593; OmniTrack: General Motion Tracking via Physics-Consistent Reference, arXiv:2602.23832; ULTRA, arXiv:2603.03279 | Human/reference motion plus simulator state, correspondences, or privileged observations | Physics-consistent reference rollout or policy actions | Later-stage physics target generation/refinement. Too heavy to be the first 1 ms online retargeter, but essential for simulator-validated target provenance. |
| Motion-tracking policy as retarget consumer | PHC, arXiv:2305.06456; BeyondMimic, arXiv:2508.08241; ProtoMotions docs | Robot proprioception, IMU/base twist, previous action, future/reference body features | PD targets / actions | Not the source-to-G1 retargeter itself. Provides observation sidecars, action-smoothness losses, and simulator evaluation metrics for the retargeted output. |
| Generative latent / diffusion / masked motion controller | BeyondMimic guided diffusion; MaskedMimic, arXiv:2409.14393 | Motion latent, partial goals, masks, current/past state | Completed trajectory, latent/action sequence, or policy action | Useful for partial observations, completion, robustness, and control. Not first-line for sub-1 ms online retargeting unless distilled into a compact one-step model. |
| Robot pose prior / scorer | PDF-HR: Pose Distance Fields for Humanoid Robots, arXiv:2602.04851 | Robot pose | Plausibility/distance score | Not a retargeter body. Use as G1 pose regularizer, loss term, or evaluation score after a direct retargeter exists. |
| Optimization or simulator refinement feeding a neural retargeter | GMR/Retargeting Matters, arXiv:2510.02252; DynaRetarget, arXiv:2602.06827; SPIDER, arXiv:2511.09484; OmniRetarget, arXiv:2509.26633 | Kinematic human/robot references, contact/object constraints, simulator state | Kinematic or dynamically feasible robot trajectory | Usually not neural inference at runtime. Use to generate labels, provenance, or quality flags for later supervised neural training. |

## Current Ranking For This Repo

1. Start with a direct supervised temporal retargeter:
   `source skeleton history + velocities + morphology -> 29D G1 joint target`
   or `joint delta`.
2. Add small TCN / small Transformer as controlled ablations with the same
   input/output contract.
3. Add a PDF-HR-style G1 pose prior only after direct output has stable offline
   metrics.
4. Try shared latent / VAE only if direct G1 output fails to generalize across
   actor skeletons.
5. Treat NMR CEPR, ReActor, OmniTrack, DynaRetarget, and SPIDER-style physics
   refinement as a later target-provenance lane, not as the first online model.
6. Defer diffusion/flow/masked controllers until there is evidence that partial
   observation or long-horizon completion is the limiting problem.

## Design Implications

- A method is only the retargeter body if it maps source human/skeleton motion
  directly to G1 pose/reference/action at inference.
- Pose priors, motion trackers, and physics refiners are valuable, but they are
  separate modules in the pipeline.
- The 1 ms RTX 4090 target strongly favors compact MLP/TCN/small Transformer
  inference and disfavors multi-step diffusion or full simulator-in-the-loop
  methods for the first online model.
- Every training target must carry provenance: kinematic G1 target, simulator
  replayed target, RL-refined target, sampling-refined target, or generated
  target. These should not be mixed silently.

## Primary References

- Qingrui Zhao et al. Make Tracking Easy: Neural Motion Retargeting for
  Humanoid Whole-body Control. arXiv:2603.22201.
- PDF-HR: Pose Distance Fields for Humanoid Robots. arXiv:2602.04851.
- Retargeting Matters: General Motion Retargeting for Humanoid Motion
  Tracking. arXiv:2510.02252.
- Chen Tessler et al. MaskedMimic: Unified Physics-Based Character Control
  Through Masked Motion Inpainting. ACM TOG 2024 / arXiv:2409.14393.
- BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided
  Diffusion. arXiv:2508.08241.
- Zhengyi Luo et al. Perpetual Humanoid Control for Real-time Simulated
  Avatars. arXiv:2305.06456.
- ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting.
  arXiv:2605.06593.
- OmniTrack: General Motion Tracking via Physics-Consistent Reference.
  arXiv:2602.23832.
- Sungjoon Choi, Matthew Pan, and Joohyung Kim. Nonparametric Motion
  Retargeting for Humanoid Robots on Shared Latent Space. RSS 2020.
- Yashuai Yan, Esteve Valls Mascaro, and Dongheui Lee. ImitationNet:
  Unsupervised Human-to-Robot Motion Retargeting via Shared Latent Space.
  Humanoids 2023 / arXiv:2309.05310.
- Villegas et al. Neural Kinematic Networks for Unsupervised Motion
  Retargetting. CVPR 2018.
- Skeleton-Aware Networks for Deep Motion Retargeting.
- DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based
  Trajectory Optimization. arXiv:2602.06827.
- SPIDER: Scalable Physics-Informed Dexterous Retargeting. arXiv:2511.09484.
