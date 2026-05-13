# Motion Quality Curation

Search date: 2026-05-14.

Purpose: define how OnlineRetarget decides whether a human/G1 motion pair is suitable for training, evaluation, or physics refinement. This is a hard gate because the model should not learn artifacts from noisy human motion or flawed retargeted targets.

## Principle

Use quality labels before deletion. The default action set is:

- `keep`: usable for train/eval.
- `downweight`: likely useful but lower confidence, for example mirrored variants or mild artifacts.
- `quarantine`: do not train on this clip until thresholds or manual review confirm it.
- `exclude`: unrecoverable parse/provenance failure or severe physical impossibility.

This follows the pattern in recent humanoid motion work: preserve diversity when possible, but do not silently mix physically invalid motion with clean targets.

## Paper Evidence

| Work | What they filter or repair | Mechanism / thresholds reported | Implication for OnlineRetarget |
| --- | --- | --- | --- |
| NMR / CEPR | Raw SMPL noise, semantically incompatible motion, excessive jerk, bad support-base relation, insufficient foot contact, G1 joint jumps, self-intersection, floating feet | Three-stage pipeline: physics-aware human motion curation, humanoid motion curation, and physics-based humanoid refinement through RL expert policies. The arXiv HTML reports hard-threshold filtering for retargeted segments, including inter-frame joint velocity limits, self-intersection frame-ratio tolerance near 0.05, and floating-foot pruning around mean foot clearance above 0.10 m. | Treat source and G1 target quality separately. Use hard `exclude` only for severe failures; use physics-refined provenance labels when Isaac/RL refinement arrives. |
| PHUMA / PhySINK | Floating, penetration, foot skating, joint-limit stress, unnatural pelvis height, root jerk, weak foot contact | Public docs describe tunable foot-contact thresholding and explicitly state defaults preserve airborne phases. Available paper summaries report 4-second clips, root jerk threshold around 50 m/s^3, support-base distance checks, average foot-contact score threshold around 0.6, pelvis-height bounds, Butterworth smoothing, and retargeting feasibility/ground/skating losses. | Add category-aware thresholds. Locomotion can use stricter contact gates than jumps, kicks, sits, or airborne motion. Report diversity lost per category. |
| GMR / Retargeting Matters | Foot sliding, ground penetration, self-intersections, floating, start/end instability, scaling artifacts | Uses retargeting method choices and FK post-processing to reduce height artifacts; evaluates how retarget quality affects BeyondMimic tracking. | Split/eval cannot rely only on supervised loss. FK-based body-height checks and per-category artifact metrics must be in M2/M4. |
| OmniTrack | Physically infeasible raw retargeted references: inconsistent CoM, foot skating, floating, penetration, jitter | Generates physics-consistent references through simulator rollouts; evaluates penetration duration, floating duration, smoothness/jerk, style fidelity via MPJPE, success rate, MPJPE, velocity error, and acceleration error. It explicitly accepts that MPJPE can increase when physics feasibility improves. | M7 should label simulator-executed/reference data separately from kinematic targets and compare fidelity against physical feasibility. Do not optimize only raw-reference MPJPE. |
| OmniRetarget | Foot skating, penetration, joint and velocity limit violations, loss of interaction/contact relationships | Interaction mesh optimization with hard constraints for collision avoidance, joint/velocity limits, and foot sticking. | For future interaction data, contact preservation and penetration are part of data quality, not only eval cosmetics. |
| KDMR | Foot slip, ground penetration, high acceleration exceeding actuator capability, inaccurate contact timing | Multi-contact whole-body trajectory optimization with dynamics and contact complementarity; uses GRF/contact information where available. | Add dynamic feasibility proxies before Isaac: acceleration spikes, contact-state inconsistency, and torque/limit proxies if model data supports them. |
| Contact and Dynamics from Monocular Video | Foot floating, foot penetration, foot skate, unnatural leaning | Defines contact-based metrics: floating when contact foot is above ground, penetration when below ground, skating when a contact foot moves horizontally. | Use these definitions for source/FK target metrics once foot points and ground plane are available. |
| Contact-aware motion retargeting / self-contact work | Lost self-contact, foot-contact errors, interpenetration | Detects foot and self contacts, then optimizes geometry/contact terms to reduce penetration and preserve contacts. | Self-contact and self-collision need proxy metrics first, then geometry-aware checks if meshes/capsules are available. |
| UNDERPRESSURE / foot contact detection | Footskate cleanup depends on reliable contact labels | Learns vertical GRF/contact labels from pressure insole data and shows threshold heuristics can be noisy. | Start with height/velocity heuristics, but keep contact confidence separate from hard labels. |

## Quality Signal Plan

| Signal | Applies to | Current status | First implementation | Action policy |
| --- | --- | --- | --- | --- |
| Nonfinite values | source, G1 | Implemented for source BVH and G1 CSV scanners where parse reaches numeric frames | Count frames/channels with NaN/Inf | `exclude` if any nonfinite affects required fields |
| Frame/channel mismatch | source | Implemented for BVH debug scanner | Compare declared BVH channels to frame width | `exclude` unless recoverable with documented parser fix |
| Joint/channel velocity jump | source, G1 | Implemented smoke scanner | Max per-frame absolute delta times fps; later use per-action percentiles | `quarantine` above provisional threshold, `exclude` for extreme parser-like jumps |
| Root discontinuity/speed | source, G1 | Implemented smoke scanner for root-like fields | Max root delta/speed and discontinuity count | `quarantine` until category-aware thresholds are calibrated |
| Acceleration/jerk | source, G1 | Pending | Second/third finite differences after unit normalization | `quarantine` high jerk; category-aware for jumps/fights |
| Joint-limit margin | G1 | G1 MJCF FK smoke scanner implemented | Load G1 joint limits from a MuJoCo/MJCF asset; count violations and max margin beyond limits | `quarantine` for mild margin stress, `exclude` for severe violations after calibration |
| Foot float | source, G1 FK | Source FK and G1 MJCF FK smoke scanners implemented | Estimate ground plane and foot height during contact; source scanner supports fixed ground or foot-percentile ground; G1 scanner uses MJCF foot bodies when `--model-xml` is supplied | `quarantine`; `exclude` if clip is globally floating like invalid sit/lie after calibration |
| Foot slide/skate | source, G1 FK | Source FK and G1 MJCF FK smoke scanners implemented | Horizontal foot velocity while contact confidence is high; current scanners use foot height contact heuristics | `downweight` mild, `quarantine` severe |
| Ground penetration | source, G1 FK | Source FK and G1 MJCF FK smoke scanners implemented | Minimum foot/body height below ground plane; fixed `ground_height=0.0` keeps global float/penetration visible | `quarantine` mild, `exclude` severe |
| Self-collision / self-intersection proxy | G1 FK, later source mesh | G1 lightweight MJCF geom-point distance proxy implemented; simulator collision labels pending | Non-adjacent FK body geom-point minimum distance, ignoring auxiliary mocap/IMU bodies and close kinematic neighbors; later MuJoCo/Isaac mesh/contact check | `quarantine`; only `exclude` after simulator proxy is validated |
| Support-base / CoM plausibility | source, G1 sim/FK | Pending | Pelvis/root projection relative to support polygon when contact is known | `quarantine`; review category-specific exceptions |
| Start/end instability | G1 | Pending | Windowed velocity/acceleration at first/last frames | `downweight` or `quarantine`; useful for trimming policy later |
| Source-target pair mismatch | pair | Pair scanner implemented; representative category/split artifact exists | Compare fps/length/action/category/provenance/missing files and target provenance | `exclude` for missing target, `quarantine` for suspicious mismatch |
| Mirrored-pair leakage | pair/split | Implemented metadata curation downweights mirrors and actor split prevents actor leakage | Keep mirror variants in same split and downweight by default | `downweight` unless a no-mirror policy is selected |

## Threshold Policy

1. Smoke thresholds are allowed only for scanner debugging.
2. Formal thresholds must be proposed from full or representative split distributions, with p90/p95/p99 summaries per category and per skeleton group.
3. Thresholds must report retained clips/hours and diversity loss by actor, skeleton, package, category, and action label.
4. Airborne/dynamic categories need separate thresholds from walking/idling categories; otherwise the filter will erase useful jumps, kicks, and acrobatics.
5. A threshold can only become a training gate after it has a named policy ID, generated artifacts, and a short rationale linked to this document.
6. Contact thresholds must distinguish "no support when support is expected" from intentional flight or floor interaction. A low contact ratio is not automatically bad for jumps, flips, cartwheels, crawls, sits, or get-up motions.
7. Filtering should default to `quarantine` plus manual/simulator review for ambiguous categories. `exclude` is reserved for parse/provenance failures, nonfinite required data, severe simulator-impossible geometry, or confirmed data corruption.
8. A policy is rejected if it improves aggregate quality only by collapsing actor/skeleton/category coverage.
9. Bounded smoke scans should use explicit stratified sampling, for example `--sample-by category --sample-by split`, when the goal is threshold calibration rather than parser debugging. First-N scans remain useful for fast regression checks but are not representative enough to promote a formal policy.
10. Threshold direction is metric-specific. Most artifact metrics are high-is-bad upper-tail proposals, but support/contact metrics such as `contact_frame_ratio` are low-is-bad lower-tail proposals. A threshold artifact must record `tail` and `comparison` so review code does not accidentally quarantine high-contact clips.
11. Time scale is part of data quality. BONES-SEED representative pair scans observed BVH `Frame Time` near `0.008333` and matching G1 frame counts, so G1 and pair quality defaults use 120 Hz. Source FK/contact scans derive FPS from BVH headers unless an override is explicitly passed.

## M2Q Execution Checklist

1. Run metadata split and inventory.
2. Run source BVH discontinuity scanner.
3. Run source FK/contact scanner.
4. Run G1 CSV/FK/contact/joint-limit scanner with the G1 MJCF.
5. Merge all source/G1/pair quality signals into a curated index.
6. Generate percentile threshold proposals by metric and category.
7. Generate diversity-loss reports by actor, source skeleton, package, category, split, mirrored status, and motion provenance.
8. Inspect worst clips by failure family: jump, twist, float, slide, penetrate, joint-limit, unstable start/end, parser mismatch.
9. Promote a named curation policy only after the retained/quarantined/excluded tradeoff is documented.
10. Allow formal training only when the run config records the policy ID, curated index, quality reports, and git SHA.

## Current Design Answer: How to Filter Good Data

The first formal policy should not be a single hard threshold. Use a staged action policy:

- `exclude`: corrupted files, nonfinite required channels, missing target, unrecoverable source-target mismatch, severe confirmed simulator-infeasible geometry.
- `quarantine`: high root/joint jumps, suspicious contact, float/penetration, joint-limit stress, source/G1 disagreement, category-ambiguous cases pending visual or simulator review.
- `downweight`: mirror variants, mild foot slide, mild start/end instability, mild threshold outliers where diversity is valuable.
- `keep`: clips passing source, target, pair, and provenance checks under the chosen policy.

This follows the strongest common pattern across NMR/CEPR, PHUMA, GMR, and OmniTrack: clean obvious defects early, preserve diversity with labels when possible, and use simulation or downstream tracking to resolve physically ambiguous clips.

## Required Artifacts

- `runs/quality/<policy>/source_*_stats.jsonl`
- `runs/quality/<policy>/source_fk_quality_stats.jsonl`
- `runs/quality/<policy>/g1_*_stats.jsonl`
- `runs/quality/<policy>/threshold_proposals.json`
- `runs/quality/<policy>/curation_report.json`
- `runs/quality/<policy>/worst_clips.csv`
- `runs/curated/<policy>/curated_index.csv`

Each artifact must record data root, git SHA, policy ID, split ID, thresholds, scanner versions, timestamp, and whether the data is kinematic, simulator-replayed, RL-refined, or policy-generated.

## Immediate Milestone Impact

- M2 is not complete until M2Q has source, target, pair, and at least initial FK/contact quality reports.
- Current source FK/contact smoke artifact: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit100/source_fk_quality_report.json`. It scanned 100 clips with `frame_stride=2`, `max_frames=256`, and fixed `ground_height=0.0`; result was keep/downweight/quarantine = 42/20/38 with flags `source_foot_slide=20` and `source_low_foot_contact=38`. This first-100 artifact predates the 120 Hz cleanup and is a scanner regression signal, not a calibration policy.
- Current G1 MJCF FK/contact/self-collision-proxy smoke artifact: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_report.json`. It scanned 100 clips with `/home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml`, `frame_stride=2`, `max_frames=256`, and fixed `ground_height=0.0`; result was keep/downweight/quarantine = 18/36/46 with flags `g1_foot_slide=70`, `g1_ground_penetration=41`, `g1_joint_limit_violation=18`, `g1_unstable_start_end=8`, `g1_foot_float=1`, and `g1_self_collision_proxy=1`. This first-100 artifact predates the 120 Hz cleanup and is weaker than the representative scan below; the self-collision proxy remains weaker than a MuJoCo/Isaac collision label.
- Current curated smoke artifact: `runs/curated/smoke_source_g1_limit100/curated_report.json` merges source BVH discontinuity stats, source FK/contact stats, and G1 FK/contact/self-collision-proxy stats. The latest three-way merge records keep/downweight/quarantine/exclude = 71,088/71,048/83/1, with `merged_source_rows=100`, `merged_source_fk_rows=100`, and `merged_g1_rows=100`.
- The same curated report now includes `diversity_loss` by actor, source skeleton, height bin, gender, package, category, split, and mirror status. In the current smoke policy, all 522 actor/source-skeleton groups retain at least one keep/downweight clip; 84 rows are quarantine/exclude. This is a smoke-scale coverage check, not a final threshold decision.
- Group-aware threshold proposal is now implemented for M2Q.5 via `propose-thresholds --group-by`. Current smoke artifacts include `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit100/source_fk_threshold_proposals_grouped_p95.json` grouped by category/package/split, plus a small G1 metadata verification run at `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit6/g1_threshold_proposals_grouped_p95.json`. These remain proposals only; full or representative scans are still required before a formal policy ID is promoted.
- Stratified scan selection is implemented for source BVH, source FK/contact, and G1 scanners via repeated `--sample-by` fields. Current smoke evidence: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit24_by-category-split/source_fk_quality_report.json` selected 24 rows across 24 category/split groups and records candidate/selected group counts in the report. This reduces first-N ordering bias for calibration experiments but is still not a substitute for a full formal scan.
- Manual review manifests are implemented from `worst_clips.csv` via `build-review-manifest`. Current smoke artifacts under `runs/curated/smoke_source_g1_limit100/manual_review/` include JSONL and Markdown review sheets for parser, mirror, jump, foot-slide, penetration, float, joint-limit, and self-collision families. Each item carries actor/category/split, source BVH path, G1 CSV path, flags, metrics, and blank reviewer decision fields.
- Pair quality scanning is implemented via `scan-pair-quality`. The current representative artifact `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit560_by-category-split/pair_quality_report.json` uses `expected_source_frame_time=0.008333333333333333`, `g1_fps=120`, strict zero frame-count delta, and target provenance `kinematic_g1_csv`. It scanned 560 category/split rows, found 494 keep and 66 quarantine, with only `pair_duration_mismatch` flags. Source/G1 frame counts match exactly in this sample; absolute duration deltas are tiny but can exceed the current 1 ms tolerance over long clips.
- Representative category/split scans now exist at 560 rows each for source BVH, source FK/contact, G1 FK/contact/self-collision-proxy, and pair/provenance. The four-way representative merge `runs/curated/representative_source_g1_pair_limit560_by_category_split/curated_report.json` records keep/downweight/quarantine/exclude = 70,858/70,876/479/7 across the full split index after merging those sampled rows. It records 0 lost actor groups, 0 lost source-skeleton groups, 0 lost category groups, and 0 lost split groups. This is stronger than first-N smoke evidence, but it is still not a promoted policy because thresholds and review decisions remain provisional.
- Lower-tail threshold proposal support is implemented for `contact_frame_ratio`. Representative source FK and G1 grouped p95 proposal artifacts record `lower_metrics=["contact_frame_ratio"]`, using p5 lower-tail contact thresholds while keeping slide, penetration, clearance, joint-limit, joint-jump, and self-collision metrics as upper-tail proposals.
- Representative manual-review artifacts under `runs/curated/representative_source_g1_pair_limit560_by_category_split/manual_review/` contain 35 rows across float, foot-slide, joint-limit, jump, mirror, parser, and penetration families. They are review sheets, not completed reviewer decisions.
- M5 formal training must require a curated index and policy ID. Tiny debug training may use raw or smoke data only when the output path and log state that it is a debug run.
- M4 must break down metrics by quality flags, because a model can improve average loss by learning bad target artifacts.
- M7 must keep physics-refined targets separate from kinematic G1 targets, even if both share the same source motion.
