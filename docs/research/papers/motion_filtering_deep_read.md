# Motion Filtering Deep-Read Notes

Search and read date: 2026-05-14.

Purpose: capture paper-backed evidence for how OnlineRetarget should filter imperfect source motions and imperfect G1 targets before formal training.

Local scratch PDFs, not committed:

- `runs/papers/nmr-2603.22201.pdf`
- `runs/papers/phuma-2510.26236.pdf`
- `runs/papers/gmr-2510.02252.pdf`
- `runs/papers/kdmr-2603.09956.pdf`
- `runs/papers/reactor-2605.06593.pdf`
- `runs/papers/omniretarget-2509.26633.pdf`
- `runs/papers/kungfubot-2506.12851.pdf`
- `runs/papers/roboforge-2603.17927.pdf`
- `runs/papers/dynaretarget-2602.06827.pdf`
- `runs/papers/spider-2511.09484.pdf`

OpenAlex resolution status:

| Work | OpenAlex status |
| --- | --- |
| NMR | Resolved as `W7140212981` / `W7140347402`, arXiv `2603.22201` |
| GMR / Retargeting Matters | Resolved as `W4414927347`, arXiv `2510.02252` |
| KDMR | Resolved as `W7134921642` / `W7134992137`, arXiv `2603.09956` |
| ReActor | Resolved as `W7160710656` / `W7160637686`, arXiv `2605.06593` |
| KungfuBot | Resolved as `W4415112555`, arXiv `2506.12851` |
| RoboForge | Resolved as `W7140001425` / `W7139011383`, arXiv `2603.17927` |
| DynaRetarget | Resolved as `W7128373084` / `W7128408694`, arXiv `2602.06827` |
| SPIDER | Resolved as `W4416307812`, arXiv `2511.09484` |
| PHUMA | Not reliably resolved by title in this pass; use arXiv/OpenReview/project page as primary sources |
| OmniRetarget | Not reliably resolved by title in this pass; use arXiv/project page as primary sources |

## Filtering Evidence Matrix

| Work | Source-motion filtering | Humanoid-target filtering / repair | Physics refinement | Direct OnlineRetarget mapping |
| --- | --- | --- | --- | --- |
| NMR / CEPR | Separates physics-aware human motion curation from humanoid motion curation. Filters source motions with excessive jerk, poor support-base relation, insufficient foot-ground contact, float, and penetration. | Uses hard-threshold filtering after kinematic retargeting for joint velocity jumps, geometric self-intersection, floating feet, and joint-limit failures. | Uses clustered RL expert policies to project references into a physics-feasible G1 data manifold. | Keep M2Q split into source quality, G1 quality, pair/provenance, and physics provenance. Treat physics-refined data as a separate target class. |
| PHUMA / PhySINK | Applies low-pass filtering, consistent ground-plane estimation, root jerk checks, foot-contact scoring, pelvis-height bounds, and base-of-support checks. The paper reports thresholds including root jerk below 50 m/s^3, foot contact score above 0.6, pelvis height between 0.6 m and 1.5 m, pelvis distance to support base below 6 cm, and spine1 distance below 11 cm. | PhySINK adds joint feasibility, grounding, and skating losses; reported metrics include joint feasibility, non-floating, non-penetration, and non-skating percentages. | Optimization-based physics-constrained retargeting, not simulator rollout. | Add jerk/support-base/pelvis-height signals after current source/G1 FK scanners. Keep contact thresholds category-aware because airborne clips can still be valid if takeoff/landing contact is consistent. |
| GMR / Retargeting Matters | Focuses less on raw source filtering and more on how retargeting artifacts affect downstream tracking. | Identifies floating, foot penetration, foot sliding, self-intersection, and abrupt velocity spikes as harmful retarget artifacts. Some post-processing that fixes penetration can introduce severe floating, so repairs must be validated. | Evaluates retarget quality through BeyondMimic tracking rather than generating refined targets. | M4 must report artifact metrics beside MPJPE/joint RMSE. M2Q should avoid blindly shifting height to fix penetration without measuring float. |
| KDMR | Uses marker motion plus GRF/contact information to estimate heel/toe contact events. | Converts kinematic references into dynamically feasible robot trajectories with joint/velocity limits, contact complementarity, and dynamics constraints. Calls out foot slip, ground penetration, and high acceleration beyond actuator capability. | Multi-contact whole-body trajectory optimization. | Add acceleration/jerk, contact-state inconsistency, and torque/limit proxies before Isaac; later use force/contact labels if available. |
| DynaRetarget | Uses kinematically retargeted human-object demonstrations as imperfect references; explicitly calls out missing contacts and trajectory imperfections. | Refines imperfect kinematic trajectories into dynamically feasible humanoid loco-manipulation trajectories with sampling-based trajectory optimization; failures remain tied to poor reference quality such as abrupt contact/object changes. | MuJoCo rollout-based full-horizon refinement before downstream RL tracking. | Treat simulator refinement as a separate repair/provenance stage. Add "refine success/failure" as a later quality label instead of overwriting kinematic targets. |
| SPIDER | Uses kinematic human/object demonstrations that are noisy, embodiment-mismatched, and usually missing force/contact information. | Converts robot/object references into dynamically feasible trajectories with annealed sampling and virtual contact guidance; filters unstable contacts by duration/drift before applying guidance. | Parallel physics-based sampling rather than RL policy training. | Add a future simulator-refinement quality record with contact-sequence correctness, success/failure, compute cost, and dynamic-feasible provenance. |
| ReActor | Uses AMASS-style data and acknowledges source references can contain floating/penetration artifacts. Introduces per-motion vertical offsets to correct noisy contacts. | Measures ground penetration, self-penetration, foot sliding, and foot floating; reports downstream success and root/joint tracking. Regularizes acceleration, torque, action rate, and action acceleration in RL. | Bilevel optimization couples retarget parameters with RL policy training. | Keep sparse correspondence / morphology-generalization branch as a later comparator. Borrow metric definitions for penetration, self-penetration, slide, float, torque/action smoothness. |
| OmniRetarget | Retargets human-object-terrain demonstrations and preserves interaction mesh relationships rather than filtering all complex contacts away. | Enforces hard constraints for collision avoidance, joint limits, velocity limits, and foot sticking / non-penetration. | Downstream RL with minimal rewards validates generated references. | Interaction/contact preservation is a future branch; current M2Q should retain contact-rich categories in quarantine/downweight rather than deleting them with global contact filters. |
| KungfuBot | Uses a multi-step motion processing pipeline: video motion extraction, physics-based filtering, contact-mask extraction, contact-aware correction, smoothing, then IK retargeting. | Corrects minor floating using foot contact masks; smooths correction-induced jitter with EMA; retargeting respects joint limits. | Adaptive motion tracking changes reward tolerance for difficult/high-dynamic motions. | Add a future contact-mask correction branch for quarantined clips. For dynamic categories, avoid over-smoothing and record whether filtering removed high-energy motion. |
| RoboForge | Uses simulation refinement and then quality control to keep physically plausible rollouts. | Its PP-Opt objective penalizes skating, floating, and ground penetration; quality control accepts refined motions only when an MPJPE-like stability threshold passes. | Closed loop: generate, execute/refine, filter, and fine-tune the generator on retained motions. | M7 should support a backward refinement dataset path: simulator-executed/refined G1 states are not interchangeable with kinematic CSV targets. |

## Implementation-Level Evidence

This pass inspected released PHUMA code in addition to paper/project text. The PHUMA `preprocess_smplx.py` curation script exposes concrete knobs for `foot_contact_threshold=0.6`, `root_jerk_threshold=50`, pelvis height bounds `[0.6, 1.5]`, pelvis-to-base-of-support distance `0.06`, and spine1-to-base-of-support distance `0.11`. It segments motions into 4-second chunks with 0.5-second overlap, applies low-pass filtering, estimates a robust ground height from foot vertices, labels foot contact from vertex height near the ground, then filters chunks by contact score, root jerk, pelvis height, and support-base distance. PHUMA's retargeting script adds joint feasibility, grounding, skating, velocity-limit, and smoothing losses, and its G1 config explicitly defines heel/toe keypoints and joint velocity limits.

These PHUMA values are useful evidence for what to measure, not OnlineRetarget defaults. BONES-SEED uses BVH source skeletons and G1 CSV targets at 120 Hz, so thresholds must be recalibrated on BONES-SEED distributions by category, actor, and source skeleton before becoming a policy.

NMR adds a direct G1-specific hierarchy: source human curation, retargeted humanoid filtering, then physics-based refinement. Its paper text gives target-side filters for peak joint velocity, self-intersection frame ratio, and mean foot clearance, and reports retargeting quality with joint jump, self-collision, joint-limit, MPJPE, W-MPJPE, and downstream tracking success.

GMR/Retargeting Matters is the strongest warning against trusting kinematic target files blindly. It connects ground penetration, self-intersection, sudden joint jumps, and bad start/end poses to downstream tracking failure or worse robustness under noise, model mismatch, and controller latency.

KDMR, DynaRetarget, and SPIDER all argue that kinematic artifacts are not only visual defects. KDMR formalizes multi-contact dynamic feasibility with contact complementarity, GRF/contact timing, velocity/acceleration, and torque/limit constraints. DynaRetarget treats kinematic trajectories as imperfect references and uses simulator rollouts to refine them; it also records that poor reference quality can still cause refinement failures. SPIDER shows a related sampling-based path where virtual contact guidance and contact filtering help convert noisy kinematic human demonstrations into dynamic-feasible robot trajectories. This supports a later M7 label such as `sim_refine_success`, `sim_refine_failed`, or `sim_refined_target` instead of silently replacing `kinematic_g1_csv`.

## OnlineRetarget Filter Design

The current filter design remains staged, not binary:

- `exclude`: parse failures, nonfinite required values, missing targets, unrecoverable pair/provenance mismatch, severe confirmed simulator-infeasible geometry.
- `quarantine`: joint/root jumps, high jerk, suspicious support/contact, float/penetration, joint-limit stress, source/G1 disagreement, contact-rich or airborne clips needing review.
- `downweight`: mirror variants, mild foot slide, mild start/end instability, mild threshold outliers where diversity is valuable.
- `keep`: clips passing source, target, pair, and provenance checks under the named policy.

The evidence above changes the implementation priority:

1. Keep current source/G1/pair scanners as the first curation layer.
2. Add source and G1 acceleration/jerk summaries before promoting a formal policy.
3. Add support-base and pelvis/root-height metrics when body points and contact confidence are reliable enough.
4. Keep `contact_frame_ratio` as a lower-tail proposal, but never use a single global low-contact rule for jumps, flips, kicks, crawls, sits, get-ups, or object/terrain interactions.
5. Require diversity-loss reports before threshold promotion, because PHUMA, ExBody2-style feasibility-diversity, and GMR all imply that "clean" but narrow data is not a sufficient dataset.
6. For M7, label simulator-replayed, RL-refined, and policy-generated outputs separately from `kinematic_g1_csv`; simulator repair success/failure should become quality evidence, not an overwrite.

## Remaining Reading Tasks

- Read NMR references around PHUMA, GMR, BeyondMimic, TMR, and CEPR-style RL refinement to decide which methods are cited versus actually used.
- Expand contact metric notes from Contact and Dynamics from Monocular Video and self-contact retargeting work into precise foot-float/penetration/skate formulas.
- Inspect OmniRetarget code or supplementary material for actual hard-constraint implementation details.
- For every future formal threshold, link back to this note and to the generated threshold artifact rather than hard-coding paper thresholds directly.
