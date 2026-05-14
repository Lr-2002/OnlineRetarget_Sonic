# Citation and Usage Map

Search/read date: 2026-05-14.

Purpose: expand M1 beyond a flat paper list. This note separates three evidence levels:

- `cited`: the work appears in references or citation metadata.
- `discussed`: the paper uses the work as related work, motivation, or comparison context.
- `used`: the paper actually uses the method, data, codebase, simulator, metric, or benchmark in its pipeline or experiments.

This distinction matters because OnlineRetarget should borrow implementation details only from papers that actually use the referenced component, not from generic related-work citations.

## Source Reliability

| Source | What was checked | Result / limitation |
| --- | --- | --- |
| OpenAlex | `NMR` as `W7140212981`, `GMR` as `W4414927347`, `PDF-HR` as `W7128096233` | All are new arXiv records with `cited_by=0` in OpenAlex at query time. OpenAlex is useful for stable IDs but not enough for usage discovery here. |
| OpenAlex references endpoint | NMR, GMR, PDF-HR | Returned empty output for these records in this pass, likely because reference metadata is incomplete for fresh arXiv records. Do not infer "no references" from this. |
| Local PDFs in `runs/papers/` | NMR, GMR, PHUMA, OmniTrack, plus filtering/refinement PDFs listed in `pdf_manifest.md` | Primary evidence for references, comparisons, actual datasets, and actual evaluation usage. |
| Project/code pages | PHUMA repository/project page, GMR repository/project page, NMR project page, BeyondMimic/ProtoMotions docs | Used when paper metadata is missing or when code exposes thresholds/pipeline steps. |

## Core Usage Map

| Work | Cited / discussed dependencies | Actually used in experiments or pipeline | OnlineRetarget implication |
| --- | --- | --- | --- |
| NMR / CEPR | Discusses GMR, PHUMA, PHC, SMPL/AMASS, shared-latent retargeting, ImitationNet, TMR, BeyondMimic, and classical motion retargeting. | Uses PHUMA-style source filtering, GMR to obtain initial robot motions, TMR features for semantic motion clustering, AMASS for test motions, CEPR RL expert rollouts to create physics-refined G1 pairs, and BeyondMimic for downstream tracking evaluation. | M1 must treat NMR as the direct architecture/data-pipeline reference. M2Q should mirror its source-curation vs humanoid-filtering vs physics-refinement separation. M7 should keep CEPR-like outputs as separate provenance. |
| GMR / Retargeting Matters | Discusses PHC, ProtoMotions, Unitree retarget data, LAFAN1, SMPL/SMPL-X conversion, BeyondMimic, VideoMimic, and contact-aware retargeting. | Retargets a diverse LAFAN1 subset to Unitree G1 using PHC, ProtoMotions, GMR, and a Unitree baseline; trains/evaluates tracking policies with BeyondMimic; evaluates success under sim, domain randomization, sim2sim/MuJoCo, observation noise, model mismatch, and network latency. | M4/M7 must report artifact metrics and tracking success, not just supervised loss. The artifact classes to track are physically inconsistent height/ground penetration, self-intersection, and sudden joint jumps. |
| PHUMA / PhySINK | Discusses AMASS, LAFAN1, Motion-X, Humanoid-X, video-to-motion noise, MaskedMimic, SINK/IK, and G1/H1-2 embodiments. | Filters Humanoid-X/Motion-X-style SMPL-X motions, applies physics-aware curation thresholds, retargets with PhySINK losses, trains policies with MaskedMimic in IsaacGym, and evaluates Unitree G1/H1-2 imitation and pelvis path following. | M2Q threshold policy must be category-aware. PHUMA explicitly preserves airborne phases while filtering physically implausible float/penetration/skating. Its thresholds are signal evidence, not OnlineRetarget defaults. |
| OmniTrack | Discusses PHUMA, GMR, BeyondMimic, ExBody2, OmniH2O, MaskedMimic, IsaacLab, MuJoCo, LAFAN1, and AMASS. | Uses Unitree-retargeted LAFAN1, AMASS motions retargeted using GMR, simulation-generated physics-consistent references, IsaacLab training, MuJoCo and real-robot checks, and MPJPE/success/floating/penetration/smoothness metrics. | M7 should evaluate the fidelity-feasibility tradeoff explicitly: MPJPE can increase while physics feasibility improves. Simulator-generated references must not overwrite kinematic G1 CSV labels. |
| DynaRetarget | Discusses imperfect kinematic humanoid/object demonstrations, MuJoCo rollout refinement, contact/object discontinuities, and downstream policy training. | Uses sampling-based full-horizon trajectory optimization in MuJoCo to refine imperfect G1 trajectories and reports refinement success, compute efficiency, smoothness, and downstream policy benefit. | Add `sim_refine_success`, `sim_refine_failed`, compute budget, and refinement provenance fields once simulator refinement exists. |
| SPIDER | Discusses noisy kinematic demonstrations, contact ambiguity, missing force/contact labels, virtual contact guidance, and physics-based sampling. | Converts kinematic demonstrations for multiple embodiments with annealed physics sampling and contact filtering before executing refined trajectories. | Contact sequence correctness should become an M7 quality label; generated dynamic-feasible trajectories are a separate target provenance. |
| KungfuBot | Discusses dynamic video motions, contact masks, minor floating, correction jitter, IK retargeting, and adaptive tracking. | Processes video-to-SMPL motions with physics-based filtering, contact-mask extraction, contact-aware vertical correction, smoothing, and IK retargeting respecting joint limits. | M2Q contact-correction candidates should remain review items until repaired/evaluated. Dynamic categories need filtering rules that preserve high-energy motions. |
| RoboForge | Discusses generated motions, physical plausibility optimization, skating/floating/penetration penalties, and simulator feedback loops. | Generates/refines/filters simulated humanoid rollouts and retrains on accepted physically plausible motions. | Useful as a later M7 feedback-loop design; refined simulator rollouts belong in a new dataset lineage, not the original BONES-SEED label set. |

## Reference Expansion Priorities

These are the next papers/codebases to inspect before freezing model ablations:

| Priority | Reference | Why it matters |
| --- | --- | --- |
| High | BeyondMimic | Defines downstream G1 tracking evaluation, motion artifact logging, success rate, robustness, and WandB artifact discipline used by GMR/NMR. |
| High | TMR | NMR uses it for motion clustering in CEPR. Needed if we implement behavior-clustered quality review or simulator refinement. |
| High | PHC / ProtoMotions retargeting paths | GMR compares against them and finds concrete artifact patterns. Their failure modes should inform M2Q review families. |
| Medium | MaskedMimic | PHUMA uses it for policy training and pelvis-only path following. Useful for M7 and possible partial-observation eval. |
| Medium | Contact and Dynamics from Monocular Video / UNDERPRESSURE | Needed to make foot float, penetration, and skate formulas precise rather than heuristic-only. |
| Medium | Shared latent retargeting / ImitationNet / unsupervised neural retargeting | Needed before investing in a VAE/shared-latent branch. |
| Low until M7 | OmniH2O, ExBody2, ASAP, GMT, AnyTrack, Sonic | Useful simulator/controller comparators, but they do not block the first direct supervised retargeter. |

## Concrete Design Decisions Supported

1. Start with direct G1 joint output and compact temporal models. NMR supports learned retargeting, but CEPR-scale physics refinement is too expensive for the first baseline.
2. Keep M2Q as a hard gate. GMR, PHUMA, NMR, and OmniTrack all show that bad references can make downstream tracking unstable or misleading.
3. Use staged quality labels instead of binary deletion. PHUMA and OmniTrack both show that physically useful corrections can trade off against raw MPJPE or apparent contact metrics.
4. Require provenance fields for target labels: `kinematic_g1_csv`, `filtered_kinematic`, `sim_replayed`, `rl_refined`, or `policy_generated`.
5. Include robustness/eval axes from GMR and OmniTrack later: observation noise, controller latency, sim2sim/MuJoCo, success/fall, MPJPE, velocity error, acceleration/smoothness, and contact artifacts.

## Gaps Still Open

- OpenAlex citation graph is not informative yet for the newest 2025/2026 arXiv papers, so future M1 passes should retry cited-by after metadata catches up.
- BeyondMimic, TMR, MaskedMimic, PHC, and ProtoMotions need deeper per-paper/code notes before final M6 ablation choices.
- Contact metric formulas need a dedicated note that maps foot points, ground plane, contact confidence, and category exceptions to exact M4/M2Q computations.
