# Paper Matrix: Learning-Based / Physics-Aware Humanoid Retargeting

Search date: 2026-05-13.

Purpose: make model/data/eval choices traceable. This matrix tracks what each relevant paper or codebase uses for observations, rewards, losses, data, model, output, and evaluation.

## Reading Coverage

| ID | Work | Primary source | Status |
| --- | --- | --- | --- |
| NMR | Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control | https://arxiv.org/abs/2603.22201 | Read project/arXiv excerpts, matrixed |
| PDF-HR | Pose Distance Fields for Humanoid Robots | https://arxiv.org/abs/2602.04851 | Read arXiv/project excerpts, matrixed |
| GMR | Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking | https://arxiv.org/abs/2510.02252 | Read arXiv excerpts, matrixed |
| BeyondMimic | From Motion Tracking to Versatile Humanoid Control via Guided Diffusion | https://arxiv.org/abs/2508.08241 and https://github.com/HybridRobotics/whole_body_tracking | Read repo/project excerpts, matrixed |
| ProtoMotions-G1 | G1 Whole-Body Tracker workflow | https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html | Read workflow excerpts, matrixed |
| OmniTrack | General Motion Tracking via Physics-Consistent Reference | https://arxiv.org/abs/2602.23832 | Read project/arXiv excerpts, matrixed |
| ULTRA | Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation | https://arxiv.org/abs/2603.03279 | Read project/arXiv excerpts, matrixed |
| OmniRetarget | Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction | https://arxiv.org/abs/2509.26633 | Read paper excerpts, matrixed |
| ReActor | Reinforcement Learning for Physics-Aware Motion Retargeting | https://arxiv.org/abs/2605.06593 | Read arXiv excerpts, matrixed |
| DynaRetarget | Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization | https://arxiv.org/abs/2602.06827 | Read PDF/OpenAlex/project excerpts, matrixed for simulator refinement |
| SPIDER | Scalable Physics-Informed Dexterous Retargeting | https://arxiv.org/abs/2511.09484 | Read PDF/OpenAlex excerpts, matrixed for physics-informed refinement |
| Shared Latent Retargeting | Nonparametric Motion Retargeting for Humanoid Robots on Shared Latent Space | https://doi.org/10.15607/RSS.2020.XVI.071 | Read abstract/excerpts, matrixed |
| Motion Quality Curation | Cross-paper filtering and physics-quality synthesis | `docs/research/motion_quality_curation.md` | Added as M2Q gate |
| Citation Usage Map | Cited/discussed/actually-used relationship map for core references | `docs/research/citation_usage_map.md` | Added to distinguish citation metadata from implementation evidence |

## Comparative Matrix

| Work | Input / Observation | Output | Model / Solver | Data | Loss / Reward | Evaluation | Key implications for OnlineRetarget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NMR | Human SMPL/SMPL-X motion sequence; global temporal context; CEPR physically validated pairs | Feasible Unitree G1 motion sequence / joint angles | Non-autoregressive CNN-Transformer; two-stage pretrain on kinematic data then finetune on CEPR | Large kinematic retargeting data + about 30K physics-refined pairs | Supervised regression to G1 kinematic/physical targets; CEPR uses RL expert tracking to generate physical targets | Joint jumps, self-collisions, joint-limit violations, downstream BeyondMimic success, MPJPE, W-MPJPE | Strong support for direct neural retargeting. Use temporal context, but keep model small until latency benchmark passes. Physics-refined targets are later milestone. |
| PDF-HR | Query robot pose `q`; no source skeleton at inference for the prior itself | Scalar pose distance/plausibility | Lightweight MLP unsigned distance field | Retargeted G1 pose corpus with positive/near/far samples | Supervised distance-to-manifold regression | Tracking and retargeting improvements as reward/regularizer/scorer | Add later as G1 pose prior regularizer/scorer; compatible with low latency if MLP is compact. |
| GMR / Retargeting Matters | Source BVH/LAFAN1 motion; body correspondence; IK objective | G1 retargeted motion reference | Optimization-based GMR/PHC/ProtoMotions comparison | LAFAN1 subset; Unitree G1 retarget data; BeyondMimic eval | IK geometric objectives; downstream policy rewards from BeyondMimic | Success rate, global/root-relative body position error, joint rotation error, robustness to noise/model mismatch/latency, user study | Eval must target retarget artifacts: ground penetration, self-intersection, sudden joint jumps, and start/end stability. |
| BeyondMimic | Reference motion, robot proprioception, future/reference features; Isaac Lab task observations | Tracking policy action / PD targets; later diffusion latent state-action controller | PPO tracking policy; VAE + guided diffusion for broader control | Retargeted LAFAN1 and other G1 references | DeepMimic-style tracking rewards and regularization; diffusion/VAE objectives for control stage | Motion tracking success, real deployment, guided diffusion task performance | Use as simulator eval style and WandB/artifact discipline, not as first retargeter model. |
| ProtoMotions-G1 | Joint pos/vel, torso IMU orientation, pelvis angular velocity, previous action, future motion reference frames | ONNX policy outputs PD position targets | Actor network exported as unified ONNX with observation computation/action processing | Full BONES-SEED G1 CSV converted to MotionLib; pretrained on large GPU cluster | BeyondMimic-style body tracking reward, global anchor orientation, action rate, soft joint limit; L2C2 smoothness regularization | MuJoCo-first validation, real G1 deployment | Excellent deployment contract. Our online retargeter should expose clear obs/output sidecars and latency benchmark like this. |
| OmniTrack | Raw retargeted or teleop motion; physical motion generation policy observation; later general tracking policy | Physics-consistent reference then joint-level actions | Two-stage policy: privileged generalist rollout then general control policy | Unitree-retargeted LAFAN1 and AMASS; online MoCap/VR teleop | Tracking rewards over body links; physical feasibility from simulator rollout | Floating/penetration elimination, smoothness, MPJPE, long-horizon/online teleop, real G1 | Confirms M7: simulator-backed physical reference generation can sit between retargeter and tracker for online robustness. |
| ULTRA | SMPL-X motion + object trajectory; privileged sim/ref/residual observation; multimodal student later | Physically feasible humanoid rollout; later multimodal control actions | RL-based physics-driven neural retargeting; teacher-student latent controller | Contact-rich loco-manipulation MoCap, object trajectories | Rewards encode tracking plus kinematic/dynamic/contact constraints; distillation and latent losses later | Sim and real Unitree G1 success on loco-manipulation and goal-following | Useful for contact-rich future branch; too complex for first baseline. Observation decomposition `sim/ref/delta` informs Isaac eval. |
| OmniRetarget | Human/object/terrain interaction mesh and kinematic constraints | Kinematically feasible G1 references preserving interactions | Constrained optimization / SOCP-style retargeting data engine | OMOMO, LAFAN1, in-house MoCap; G1 | RL policies trained on only five terms: body tracking, object tracking, action rate, soft joint limit, self-collision | Kinematic quality, penetration, foot skating, contact preservation, policy success | Interaction preservation matters later. For now, add artifact metrics and keep reward design minimal. |
| ReActor | Sparse semantic rigid-body correspondences; AMASS filtered data | Retargeted references + tracking policy | Bilevel optimization with RL lower level and retarget parameter upper level | AMASS, Unitree G1 and smaller robot | RL tracking regularization: acceleration, torque, action rate; upper-level retarget objective | Ground/self penetration, foot slide/float, success, root/joint tracking RMSE | Good physics-aware comparator. Highlights morphology generalization and sparse correspondences. |
| KDMR | Human marker motion, estimated/contact-labeled heel-toe states, and GRF signals | Dynamically feasible humanoid trajectory with multi-contact constraints | Multi-contact whole-body trajectory optimization | Motion capture plus force/contact measurements | Kinematic matching with joint/velocity limits, rigid-body dynamics, contact complementarity, and GRF tracking | Dynamic feasibility, GRF tracking, smoothness, downstream BeyondMimic policy efficiency | Add acceleration/jerk, contact consistency, and torque/limit proxies before Isaac. Future force/contact labels should be quality signals. |
| DynaRetarget | Imperfect kinematic human-object demonstrations; contact/object references; G1 state and control trajectory in MuJoCo | Dynamically feasible G1 loco-manipulation trajectory and downstream tracking policy data | Full-horizon sampling-based trajectory optimization followed by RL tracking | OmniRetarget-style G1 object-interaction demonstrations | Tracking/object/velocity costs with contact-related terms; refinement success defined by object position/rotation error and smoothness by joint acceleration ratio | Refinement success rate, compute efficiency, smoothness, downstream policy success, real-robot transfer | Treat simulator refinement success/failure as a quality label. Failed refinement can quarantine a clip even if kinematic metrics look acceptable. |
| SPIDER | Kinematic human/object demonstrations, reconstructed objects, intended contacts, robot/object reference states | Dynamically feasible robot trajectory / control sequence | Physics-based sampling with annealed kernels and virtual contact guidance | Six human demonstration datasets, nine humanoid/dexterous embodiments | State/control tracking costs plus virtual contact guidance and contact filtering for unstable contacts | Task success, speedup over RL, contact sequence correctness, dataset scale, real rollout | Useful M7 comparator for contact-rich refinement. Do not mix generated dynamic-feasible trajectories with kinematic G1 CSV labels. |
| KungfuBot | Processed SMPL-format video motions, contact masks, robot proprioception/reference features for tracking | IK-retargeted G1 references and policy actions | Physics-based motion processing plus adaptive tracking policy | Highly dynamic video motions plus existing motion datasets | Filtering by physical metrics, contact-mask correction, IK retargeting; policy reward includes tracking plus smoothness/stability regularization | Dynamic skill tracking, contact correctness, fall-resilient behavior | Dynamic categories need category-aware filtering and correction; do not over-smooth high-energy clips. |
| RoboForge | Text-conditioned motion latent, generated reference, simulator execution feedback | Refined policy rollout and filtered high-quality motion data | Physical Plausibility Optimization coupled to teacher-student tracking | Text-guided/generated humanoid locomotion data | Plausibility reward penalizes skating, floating, and penetration; retained refined clips fine-tune generator | Tracking error, success rate, penetration, float, skate, distribution/diversity metrics | Supports M7 backward loop: simulator-refined outputs become a distinct training target provenance. |
| Shared latent retargeting | Mocap Cartesian joint positions and robot joint space | Robot joint pose via latent regression/decoder | WAE/shared latent space + locally weighted regression + graph heuristics | Paired and domain-specific mocap/robot datasets | Latent reconstruction/alignment; nonparametric regression; transition feasibility heuristics | Feasibility/self-collision and tracking in simulation/online puppeteering | Valid latent branch, but not baseline because it adds encoder/decoder ambiguity before direct G1 metrics exist. |

## Motion Quality / Filtering Matrix

| Work | Source curation | Target/humanoid curation | Physics refinement | Quality metrics / filters to borrow |
| --- | --- | --- | --- | --- |
| NMR / CEPR | Filters noisy or semantically incompatible SMPL-style motions; cites excessive jerk, bad support-base relation, insufficient foot contact, float/penetration. | Hard-threshold filters retargeted motion for joint jumps, self-intersection, floating feet, and joint-limit artifacts. | Uses clustered RL expert policies to project kinematic references onto a physically feasible G1 manifold. | Separate source quality, G1 target quality, and physics-refined provenance. Use joint jump, self-intersection proxy, float, and joint-limit metrics. |
| PHUMA / PhySINK | Low-pass filtering, ground/contact inference, clip-level rejection for root jerk, support-base/CoM plausibility, foot-contact score, and pelvis height. Public docs state default thresholds preserve airborne phases rather than deleting all float-like clips. | Retargeting objective adds feasibility, grounding, and skating terms. | Physics-constrained optimization, not policy rollout. | Category-aware contact thresholds, foot skate, float, penetration, joint feasibility, retained-diversity report. Avoid global thresholds that erase jumps/kicks/flips. |
| GMR / Retargeting Matters | Uses curated benchmark subsets; focuses more on retargeting artifacts than raw source filtering. | Addresses scaling, foot sliding, ground penetration, self-intersection, floating, and height artifacts through optimization/FK post-processing. | Evaluates downstream with BeyondMimic rather than generating physics-refined targets. | FK body-height checks, global/root-relative tracking error, policy success sensitivity to artifact quality. |
| OmniTrack | Takes raw retargeted/teleop references that may contain artifacts. | Uses simulator-generated physically consistent references to eliminate penetration/floating and improve smoothness. | RL physical motion generator in simulation. | Track fidelity-physics tradeoff: MPJPE may rise while feasibility improves. Record physics-refined provenance and evaluate penetration/floating/smoothness separately from MPJPE. |
| OmniRetarget | Preserves human/object/environment interactions rather than only keypoints. | Enforces collision avoidance, joint and velocity limits, and foot sticking as hard constraints. | Downstream RL can repair minor remaining violations. | Interaction/contact preservation, penetration, foot skating, joint/velocity limit violation. |
| KDMR | Uses motion and force/contact measurements for locomotion. | Dynamic retargeting removes foot slip, penetration, and high-acceleration infeasibility. | Multi-contact whole-body trajectory optimization with dynamics/contact constraints. | Contact timing, GRF/contact consistency, acceleration and torque-limit proxies. |
| ReActor | Uses filtered AMASS-style data but still accounts for residual source floating/penetration with per-motion vertical offsets. | Evaluates and reduces ground penetration, self-penetration, foot sliding, and foot floating. | Bilevel RL retargeting and downstream tracking. | Ground/self penetration, foot slide/float definitions, torque/action-rate/action-acceleration smoothness. |
| DynaRetarget | Starts from imperfect kinematic demonstrations and explicitly handles missing contacts, penetrations, and discontinuities. | Refines G1 trajectories in MuJoCo; failures remain tied to poor reference quality such as abrupt contact/object changes. | Full-horizon sampling-based trajectory optimization; refined data improves RL tracking. | Refinement success/failure, object tracking error, smoothness by joint acceleration ratio, compute cost. |
| SPIDER | Accepts noisy kinematic human demonstrations that lack force/contact information. | Converts references to dynamically feasible robot trajectories with corrected contacts. | Sampling-based physics optimization with virtual contact guidance and contact filters. | Contact sequence correctness, task success, speed, dynamic feasibility, contact-filtered demonstrations. |
| KungfuBot | Filters reconstructed video motion with physics-based metrics and extracts contact masks. | Corrects minor floating from contact masks, smooths correction jitter, then retargets with IK respecting joint limits. | Adaptive tracking reward tolerance for hard motions. | Contact-mask correction for quarantined clips; dynamic-category thresholds. |
| RoboForge | Filters generated/refined rollouts after simulation execution. | Penalizes skating, floating, and ground penetration during physical plausibility optimization. | Closed-loop generate, execute/refine, filter, and re-train. | Simulator-refined provenance and M7 backward data-generation path. |
| Contact/dynamics and self-contact retargeting | Detects foot/self contacts and physically implausible motion from video/mocap. | Optimizes contact preservation and interpenetration reduction on target skeleton/geometry. | Physics-based trajectory optimization or encoder-space optimization. | Contact foot float/penetration/skate definitions; self-contact/interpenetration proxies. |

## Filtering Milestone Implications

M2Q must be completed before formal training. The concrete data filtering plan is:

1. Measure source quality, G1 target quality, pair consistency, and physics provenance independently.
2. Use `keep/downweight/quarantine/exclude`, not binary keep/drop, so dynamic but valuable data is not lost prematurely.
3. Calibrate thresholds from BONES-SEED distributions by category, actor, and skeleton. Smoke thresholds are only scanner tests.
4. Generate worst-clip manifests for manual review by failure type: joint jump, twist, float, foot slide, penetration, joint-limit violation, self-intersection proxy, and start/end instability.
5. Report diversity loss. A curation policy that removes most actors, skeletons, or dynamic categories is not acceptable even if remaining clips look clean.
6. Treat simulator-replayed or RL-refined motion as different target provenance, following NMR/CEPR and OmniTrack, not as interchangeable labels with kinematic G1 CSV targets.
7. Add jerk/support-base/pelvis-height and contact-mask/correction review after the current source/G1/pair scanners, following PHUMA and KungfuBot, before promoting a formal policy.
8. Record simulator refinement success/failure as a later quality signal, following DynaRetarget and SPIDER. A clip that cannot be refined or tracked in physics should remain quarantined even if static FK metrics are borderline acceptable.

## Design Conclusions

1. Baseline should be direct G1 output with temporal context.
2. Actor-heldout split is mandatory because BONES-SEED has 522 source skeletons; clip-level split would overstate generalization.
3. Offline metrics must include both geometric fidelity and artifact metrics; tracking-only success is too late for debugging.
4. Robot proprioception/IMU/history are important for a deployable tracker, but first retargeter can separate source-to-reference generation from controller action.
5. PDF-HR-style pose prior is the lowest-complexity next regularizer after direct baseline.
6. Physics-refined data generation is required for final quality but should be introduced after direct baseline failure modes are measured.
7. Diffusion/flow models are not first-line online retargeters under a 1 ms budget unless distilled into a single-step compact network.
8. Motion filtering is a formal M2Q gate: source quality, G1 target quality, pair consistency, and physics provenance must be measured before formal training.
9. Simulator-refined or sampling-refined trajectories are not replacements for kinematic labels; they are separate target provenance and must carry refinement success/failure metadata.

## Baseline Observation Candidate

First trainable observation:

- Source body positions/history in source-local heading frame.
- Source velocities over 8-frame history.
- Actor morphology vector from BONES-SEED metadata.
- Current G1 joint position/velocity and previous action, zero-filled if producing offline references.

First output:

- 29D G1 joint target or joint delta.

Later output:

- root + 29D generalized coordinates for offline reference generation.
- physics-consistent rollout state from Isaac Lab / MuJoCo.

## Evaluation Candidate

Offline:

- joint RMSE
- MPJPE/body position error
- W-MPJPE once global body positions are available
- action similarity
- joint jump rate
- joint-limit violation
- per-category and per-actor breakdown

Simulator:

- success/fall rate
- episode length
- global/root-relative body error
- foot slide, foot float, ground penetration
- self-collision
- robustness to observation noise, latency, and domain randomization
