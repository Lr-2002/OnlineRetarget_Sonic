# Repository Filtering Code Review

Search date: 2026-05-14.

Purpose: record code-level evidence for how related humanoid motion repositories filter, repair, or mine problematic motions, and map those choices to OnlineRetarget's BONES-SEED `soma_proportional` -> Unitree G1 curation pipeline.

## Repositories Inspected

| Repository | Local revision inspected | Relevant files |
| --- | --- | --- |
| DAVIAN-Robotics/PHUMA | `7a10155c0b8959f7de6bdfd551867bd7ff7fd28a` | `src/curation/preprocess_smplx_folder.py`, `src/curation/preprocess_g1_folder.py`, `src/utils/smpl.py` |
| NVlabs/ProtoMotions | `a87d2f3a0be91fc1ab3485ef4ace2e92ef40b0fd` | `data/scripts/motion_filter.py`, `data/scripts/contact_detection.py`, `data/scripts/convert_g1_csv_to_proto.py`, `protomotions/agents/evaluators/*` |
| YanjieZe/GMR | `f74056992debac94ec153c5e818509143ceb5226` | `general_motion_retargeting/motion_retarget.py`, `TEST_MOTIONS.md` |
| NVIDIA/soma-retargeter | `b3ef2708d84bfd1314ddb52d0db6c9c211df1f57` | `soma_retargeter/pipelines/feet_stabilizer.py`, `soma_retargeter/configs/unitree_g1/*.json` |
| ZhengyiLuo/PHC | `846988d433ce1f341e85ac6fbd2cd51911bb3341` | `phc/learning/im_amp_players.py`, retargeting docs |

## Main Finding

The repositories split quality handling into three different layers:

1. **Offline hard filtering**: PHUMA and ProtoMotions contain explicit pre-training filters. These are the strongest references for M2Q scanner design.
2. **Retarget-time correction and constraints**: GMR and SOMA Retargeter mostly constrain or repair IK outputs. These are not dataset filters by themselves, but their constraints tell us which metrics should be exposed in quality reports.
3. **Simulation/evaluation failure mining**: PHC and ProtoMotions record failed tracking motions and success rates during policy evaluation. This should become a later Isaac/MuJoCo label path, not an early replacement for source/G1 geometry scans.

## PHUMA

PHUMA has the most complete code-level filtering recipe.

### Human/SMPL-X Curation

`src/curation/preprocess_smplx_folder.py` does:

- Applies a Butterworth low-pass filter when the sequence is long enough.
- Estimates a robust ground height from toe/heel vertices.
- Computes foot-contact labels from the fraction of foot vertices close to the robust ground.
- Splits motions into 4-second chunks with 0.5-second overlap and 1-second minimum length.
- Rejects chunks using:
  - foot contact score `<= 0.6`;
  - mean root jerk `>= 50`;
  - pelvis height `<= 0.6 m` or `>= 1.5 m`;
  - pelvis distance to base of support `>= 0.06 m`;
  - spine1 distance to base of support `>= 0.11 m`.

`src/utils/smpl.py` provides the implementation details:

- translation cutoff is 3 Hz;
- global/body rotation cutoff is 6 Hz;
- robust ground is the densest height window over toe/heel contact heights;
- base of support is a convex hull over ankle/foot joints.

### G1 Curation

`src/curation/preprocess_g1_folder.py` does target-side filtering after MuJoCo FK:

- optionally estimates a ground offset from foot heights;
- computes COM through MuJoCo;
- computes a toe/heel support polygon;
- rejects chunks when:
  - minimum COM height `< 0.4 m`;
  - mean COM-to-base-of-support distance `> 0.16 m`.

### OnlineRetarget Mapping

Adopt PHUMA's **signals**, not its thresholds, as defaults. The thresholds were tuned for PHUMA's data, 30 Hz processing, SMPL-X representation, and locomotion-heavy curation. BONES-SEED uses `soma_proportional` BVH, actor-specific skeletons, G1 CSV targets, and 120 Hz timing, so thresholds need category/skeleton calibration before promotion.

Immediate mapping:

- Keep root acceleration/jerk, root height, support distance, contact ratio, foot float, penetration, and slide in scanner output.
- Use PHUMA-like COM/BoS as a stronger G1 metric once BONES-SEED categories have been calibrated.
- Keep airborne/sit/get-up/fight actions out of global hard-contact filters. PHUMA's public docs explicitly tune defaults to preserve airborne phases, which matches our staged `keep/downweight/quarantine/exclude` policy.

## ProtoMotions

ProtoMotions has a deliberately simple motion filter plus contact heuristics. It is valuable because it directly supports BONES-SEED and G1 conversion flows.

`data/scripts/motion_filter.py` rejects a `RobotState` when:

- any rigid body height falls below `min_height_threshold=-0.05`;
- finite-difference global body velocity exceeds `max_velocity_threshold=15.0`;
- DOF velocity exceeds `max_dof_vel_threshold=40.0`;
- the lowest body point stays more than `duration_height_filter` above the estimated floor for `duration_height_seconds`.

The conversion scripts expose these knobs. For example, `convert_g1_csv_to_proto.py` has:

- `--apply-motion-filter`;
- `--min-height-threshold -0.05`;
- `--max-velocity-threshold 15.0`;
- `--max-dof-vel-threshold 40.0`;
- `--duration-height-filter 0.1`;
- `--duration-height-seconds 0.6` for G1 CSV conversion.

`data/scripts/contact_detection.py` labels contact by combining low body velocity and low body height:

- default velocity threshold around `0.1` to `0.15 m/s` depending call site;
- height threshold around `0.08` to `0.1 m`.

`convert_g1_csv_to_proto.py` also normalizes height before filtering through per-frame and global height fixes, then recomputes velocities. That is a correction step, not a pure measurement step, so OnlineRetarget should not silently apply it to read-only BONES-SEED data. If we add correction, it should write a separate derived target directory with provenance.

### OnlineRetarget Mapping

ProtoMotions supports keeping our first filter simple:

- hard reject parser-like defects: nonfinite data, severe below-ground body positions, huge body or joint velocity spikes;
- keep long-duration floating as a quarantine/exclude candidate;
- keep contact labels as confidence heuristics, not ground truth;
- expose filter knobs and disabled/enabled state in artifacts, because ProtoMotions itself makes filtering optional in some conversion commands.

## GMR

GMR is mainly an IK/retargeting constraint reference, not a dataset filter.

`general_motion_retargeting/motion_retarget.py`:

- loads a MuJoCo robot model;
- uses Mink `ConfigurationLimit`;
- optionally adds a velocity limit of `3*pi`;
- applies human scale tables and ground offsets;
- exposes IK task errors through two staged match tables.

`TEST_MOTIONS.md` is useful as a manual failure catalog: it lists known hard motions and notes examples such as body jittering, lying-ground mismatch, and bad arm twisting on G1.

### OnlineRetarget Mapping

Use GMR to justify target-side validation metrics:

- joint limit violation;
- velocity jump;
- start/end instability;
- ground offset side effects;
- task/body match error if we later run optimization baselines.

Do not treat GMR's retarget result as automatically clean. Its own failure list argues for visual/manual review and downstream tracking checks.

## SOMA Retargeter

SOMA Retargeter is also mainly correction/constraint code.

`soma_retargeter/pipelines/feet_stabilizer.py` builds a Newton IK solver for Unitree G1:

- uses pelvis, hip, knee, and ankle effectors;
- adds an IK joint-limit objective;
- solves batched two-bone IK for leg stabilization.

`soma_retargeter/configs/unitree_g1/g1_feet_stabilizer_config.json` sets:

- `ik_iterations=20`;
- `joint_limit_weight=10.0`;
- high pelvis position weight `30.0`;
- ankle position weight `10.0`.

### OnlineRetarget Mapping

This is evidence that feet/pelvis stabilization and joint limits are first-class retarget constraints. It does not replace dataset filtering. For our pipeline:

- keep G1 foot slide, foot float, penetration, and joint-limit stress metrics visible;
- treat feet stabilization as a future repair branch for quarantined clips;
- never mutate BONES-SEED inputs in place. A stabilized output should be a separate derived artifact with a policy ID.

## PHC

PHC provides an evaluation-stage quality path.

`phc/learning/im_amp_players.py` in imitation evaluation mode:

- tracks termination state per motion;
- computes success rate as one minus failed-motion rate;
- records MPJPE over predicted versus ground-truth body positions;
- writes failed and successful motion keys;
- can dump observations, clean actions, environment actions, reset masks, motion lengths, and normalization statistics into a PHC action dataset.

### OnlineRetarget Mapping

This should become a later simulator-backed quality label:

- `sim_tracking_success`;
- `sim_tracking_mpjpe`;
- `sim_early_termination`;
- `sim_failed_motion_key`;
- `sim_refined_or_policy_generated_target`.

It should not be used before we have a stable Isaac Lab/MuJoCo replay path, because a tracking policy failure can reflect controller weakness, not only bad motion data.

## Recommended OnlineRetarget Filter Design

Short-term M2Q policy should stay staged:

- `exclude`: parse/provenance corruption, nonfinite required fields, missing target, severe impossible geometry, extreme velocity spikes.
- `quarantine`: foot slide, penetration, float, joint-limit stress, start/end instability, support/height outliers, long-duration floating, ambiguous source-target mismatch.
- `downweight`: mild artifacts, mirrored variants, mild start/end issues, valuable but noisy high-dynamic categories.
- `keep`: clips passing source, G1, pair, provenance, and policy-specific thresholds.

Concrete migrations from repo review:

1. Keep current metric-only support/root-height/contact-skate fields until BONES-SEED calibration is complete.
2. Add or promote COM/BoS only after validating category exceptions for airborne, sit, get-up, fight, and floor-contact motions.
3. Add ProtoMotions-style long-duration floating and severe body-velocity filters as proposal metrics, not immediate global hard filters.
4. Keep correction separate from filtering: height normalization, feet stabilization, and contact-mask correction must write derived data with provenance.
5. Add simulator/evaluation failure mining later, following PHC/ProtoMotions: failed motion IDs, success rate, MPJPE, reset/termination reasons, and sampling weights.
6. Manual review should include both metric-worst samples and category-balanced examples, because strict physical filters can accidentally erase high-value dynamic motions.

## Relation To Current OnlineRetarget Artifacts

The current G1 flags already align with the external repo evidence:

- `g1_foot_slide`: supported by PHUMA/PhySINK, GMR, SOMA feet stabilization, and ProtoMotions contact heuristics.
- `g1_joint_limit_violation`: supported by GMR, SOMA Retargeter, PHUMA retarget constraints, and ProtoMotions DOF velocity checks.
- `g1_unstable_start_end`: supported by GMR issue patterns and ProtoMotions velocity-spike filtering.
- `g1_ground_penetration`: supported by PHUMA, ProtoMotions min-height filtering, GMR, and SOMA.
- `joint_velocity_jump`: supported by ProtoMotions DOF velocity filtering and GMR optional velocity limit.
- `g1_foot_float` / `g1_low_foot_contact`: supported by PHUMA foot-contact score and ProtoMotions long-duration height filter.
- `g1_self_collision_proxy`: supported more by paper-level humanoid curation evidence than these repo filters; keep as proxy until simulator collision labels are available.

The immediate conclusion is that OnlineRetarget should not jump to a binary drop list. The repo evidence supports our current path: scan source, G1 target, pair/provenance, and later simulator execution independently; expose metrics; review balanced demos; then promote a named threshold policy only after diversity loss and manual/simulator evidence are acceptable.
