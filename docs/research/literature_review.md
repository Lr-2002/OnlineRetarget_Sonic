# Literature Review

Search date: 2026-05-13.

Sources used: Exa web search/crawl, arXiv pages, OpenAlex lookup, project pages, local BONES-SEED README/metadata.

## Seed References

### NMR: Neural Motion Retargeting for Humanoid Whole-body Control

Primary links:

- Project page: https://nju3dv-humanoidgroup.github.io/nmr.github.io/
- arXiv: https://arxiv.org/abs/2603.22201

The user supplied `https://arxiv.org/abs/2603.23983`; live lookup did not resolve that ID as the NMR paper. The NMR project page and paper consistently cite `arXiv:2603.22201`, and the provided NMR PDF URL points to the same project.

Core idea: NMR reframes retargeting as a learned motion-distribution mapping instead of per-frame optimization. It uses Clustered-Expert Physics Refinement (CEPR): VAE-style motion clustering, RL expert policies in simulation to project noisy or kinematic references onto the robot feasible manifold, then a non-autoregressive CNN/Transformer retargeting model. Reported target robot is Unitree G1.

Implication for this repo:

- Direct supervised neural retargeting is a credible target, but NMR's CEPR is expensive and belongs after a simple baseline.
- We should track artifacts NMR calls out: joint jumps, self-collision, joint-limit violations, floating feet, and self-intersection.
- A small temporal model is the right first step; large global-context transformers must be justified against the 1 ms inference budget.

### PDF-HR: Pose Distance Fields for Humanoid Robots

Primary links:

- Project page section: https://gaoyukang33.github.io/PDF-HR/#retargeting
- arXiv: https://arxiv.org/abs/2602.04851

PDF-HR learns a lightweight MLP pose prior for humanoid robot poses. The model maps an arbitrary robot pose to a scalar distance from a corpus of retargeted robot poses, creating a differentiable plausibility score. The paper applies this as a regularizer, reward shaping term, or standalone scorer across tracking and retargeting.

Implication for this repo:

- Add a pose-prior branch only after the direct baseline exists.
- The first practical use is an evaluation/regularization score on G1 joint poses, not a new generative model.
- The pose-prior concept is compatible with the 1 ms target because the proposed prior is an MLP.

### Retargeting Matters / GMR

Primary links:

- arXiv: https://arxiv.org/abs/2510.02252
- Code: https://github.com/YanjieZe/GMR

The paper evaluates how retargeting quality affects humanoid motion tracking, using BeyondMimic as the downstream policy training/evaluation harness. It identifies ground penetration, self-intersections, and sudden joint jumps as artifacts that can make policy learning harder even when a clip is not impossible to track.

Implication for this repo:

- Evaluation must be retargeting-first, not only training-loss-first.
- Metrics should include local and world MPJPE, joint jump rate, limit violations, contact/ground artifacts, and later downstream tracking success.
- Start/end pose stability is part of retargeted-data quality.

### BeyondMimic

Primary links:

- arXiv: https://arxiv.org/abs/2508.08241
- Code: https://github.com/HybridRobotics/whole_body_tracking

BeyondMimic provides the motion tracking training/evaluation stack used by NMR and GMR-related work. The public repo targets Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, and Unitree G1. It expects retargeted generalized coordinates and uses WandB registry/logging for motion artifacts.

Implication for this repo:

- Isaac Lab integration should follow the BeyondMimic-style separation: reference motion preprocessing, tracking task config, train, play/eval.
- WandB is required from the start because motion artifacts, configs, and code revisions need traceability.
- Offline retargeter metrics should align with downstream tracking metrics to avoid optimizing the wrong target.

### Nonparametric Motion Retargeting on Shared Latent Space

Primary links:

- RSS page: http://roboticsproceedings.org/rss16/p071.html
- DOI: https://doi.org/10.15607/RSS.2020.XVI.071

This work uses a shared latent space between mocap and robot poses, with locally weighted regression and graph heuristics to preserve feasibility and transitions. It is relevant to the user's latent-output option and online puppeteering goal.

Implication for this repo:

- Latent alignment is a valid research branch, especially if direct G1 output overfits actor morphology.
- It should not be the first baseline because it adds encoder/decoder complexity and evaluation ambiguity.

### BONES-SEED and SOMA Retargeter

Primary links:

- BONES-SEED Hugging Face: https://huggingface.co/datasets/bones-studio/seed
- BONES-SEED website: https://bones.studio/datasets/seed
- Seed viewer: https://github.com/bones-studio/seed-viewer
- NVIDIA SOMA Retargeter: https://github.com/NVIDIA/soma-retargeter

BONES-SEED provides 142,220 annotated motions, 522 performers, SOMA uniform/proportional skeleton formats, and Unitree G1 MuJoCo-compatible CSV targets. The SOMA Retargeter docs state that it uses SOMA input and currently supports Unitree G1 29 DoF output, with scaling, IK, feet stabilization, and joint-limit clamping.

Implication for this repo:

- BONES-SEED can provide both heterogeneous source skeletons and G1 supervision.
- Actor-level splits are mandatory to test cross-skeleton generalization.
- The source skeleton should include morphology features from metadata, not only per-frame motion.

## Data Curation / Filtering References

These works are the closest matches to the user's question about how to filter "good enough" data before or during retargeting.

Detailed implementation mapping now lives in `docs/research/motion_quality_curation.md`. The important planning change is that motion quality is not a side note under data loading; it is a formal M2Q gate between data inventory and formal training.

### NMR / CEPR

Primary links:

- Project page: https://nju3dv-humanoidgroup.github.io/nmr.github.io/
- arXiv: https://arxiv.org/abs/2603.22201

NMR makes filtering explicit in its CEPR pipeline. The paper description shows three relevant stages: physics-aware human motion curation, humanoid motion curation after kinematic retargeting, and physics-based humanoid motion refinement. The paper text also describes hard-threshold filtering for retargeted clips: reject excessive joint-velocity jumps, reject sequences with geometric self-intersections above a cross-ratio threshold, and prune floating-foot motions above a foot-clearance threshold. It also notes that raw SMPL-style source motion can contain ground penetration or temporal jitter, so source motion is curated before retargeting.

Implication for this repo:

- Use NMR as the strongest direct reference for our M2 quality flags.
- Separate source-motion curation from target-motion curation.
- Keep "quality-flagged but useful" motions instead of only binary keep/drop when diversity matters.

### PHUMA

Primary links:

- arXiv: https://arxiv.org/abs/2510.26236
- Project page: https://davian-robotics.github.io/PHUMA/

PHUMA is a physically grounded humanoid locomotion dataset built from large-scale human video. The abstract explicitly says it addresses floating, penetration, and foot skating through careful data curation and physics-constrained retargeting. It enforces joint limits, ensures ground contact, and removes foot skating. This is the cleanest recent example of treating dataset construction as a physics-quality problem rather than a pure labeling problem.

Implication for this repo:

- Treat joint limits, ground contact, and foot skating as first-class curation gates.
- Use physics-constrained retargeting as a data-quality upgrade path, not only as a training target.

### ExBody2

Primary links:

- arXiv: https://arxiv.org/abs/2412.13196
- Project page: https://exbody2.github.io/

ExBody2 describes automatic dataset curation via a feasibility-diversity principle. The available paper snippets show that the method uses a teacher policy to generate intermediate motions, then filters infeasible whole-body motions with a threshold chosen against base-policy tracking performance. The tradeoff matters: too strict a threshold removes useful diversity, while too loose a threshold keeps unstable motions.

Implication for this repo:

- Do not choose a single hard threshold without checking how much motion diversity is lost.
- Consider a scalar quality score or keep/downweight/quarantine policy instead of only keep/drop.

### A Kung Fu Athlete Bot That Can Do It All Day

Primary links:

- arXiv: https://arxiv.org/abs/2602.13656

This paper is not a retargeting paper per se, but it is useful as a motion-cleaning reference for very dynamic data. The summary describes a video-to-motion correction pipeline that fixes root-height drift, reconstructs airborne phases with a physics-based parabola, and applies Savitzky-Golay smoothing to reduce jitter while keeping key motion peaks. It also uses low-kinetic-energy anchors when sampling episodes.

Implication for this repo:

- Dynamic clips need cleaning rules that preserve peak motion instead of over-smoothing them away.
- For highly dynamic categories, quality filtering should be category-aware rather than global.

## Working Design Choices

1. Baseline output is direct G1 joint reference, not latent.
2. Baseline input includes source skeleton motion history, actor morphology, and robot proprioception placeholders.
3. Evaluation module is independent from training.
4. Physics-refined targets are a later milestone, likely via Isaac Lab/BeyondMimic-style tracking or NMR-like CEPR.
5. Diffusion and flow matching are not first-line online models unless distilled into a low-step or direct network that meets the latency budget.
6. Data filtering should preserve diversity as well as feasibility; a strict binary filter is usually too aggressive for locomotion/dance/interactions.
7. Formal training must record a quality policy ID and curated index path; debug training can use smoke data only when clearly labeled as debug.

## Open Questions

- Confirm whether the user intended `arXiv:2603.23983` or `arXiv:2603.22201`.
- Decide whether to train first on `g1.tar` CSV targets or already-converted `bones_sonic` G1 `npz` targets.
- Confirm exact G1 joint limits and body-name ordering from the simulator asset before using joint-limit and collision metrics as gates.
- Decide whether first inference target is joint position, joint delta/action, or short horizon of generalized coordinates.
