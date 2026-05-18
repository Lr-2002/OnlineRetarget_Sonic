# Data Experience Summary

Date: 2026-05-18.

Decision: pause/defer the current SONIC visual-QA / full-curation milestone. This is not a success mark and not a formal data-quality policy. It means we have enough data-source and skeleton-semantics evidence to stop blocking on this lane and move the next work toward train/test/eval split, sample extraction, and a small supervised baseline.

## Current Ground Truth

The active dataset for the SONIC lane is:

- `/home/user/data/motion_data/bones_sonic`

The legacy `soma_proportional.tar + g1.tar` path is useful only as historical BONES-SEED debugging evidence. It is not valid evidence for current SONIC quality, current SONIC visualization, or current G1 target semantics.

The full BONES-SONIC index found:

- 142,220 NPZ files.
- 142,220 metadata matches.
- 522 actors.
- 71,088 mirrored clips.
- all clips at 50 Hz.
- all files with schema status `ok`.
- frame count min / mean / max: 29 / 364.952742 / 9007.

Canonical artifacts:

- index report: `runs/indices/bones_sonic_index_full_v0/sonic_index_report.json`
- data-source note: `docs/status/sonic_data_source.md`
- skeleton note: `docs/status/sonic_skeleton_semantics.md`
- corrected videos: `runs/review_clips/sonic_quality_3d_capsules_v3_isaaclab_order/README.md`

## Data Semantics Learned

`bones_sonic` is robot-state data, not raw SMPL data. Each NPZ contains:

- `fps`: `(1,)`
- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`

The local `bones_sonic` files do not contain `smpl_joints` or `smpl_pose`. SONIC has a separate SMPL input lane in its repo conventions, where SMPL joints are 24 points and the topology is fixed, but that lane is not present inside these NPZ target files.

The `joint_pos` and `body_pos_w` arrays are already in SONIC/IsaacLab G1 order. They must not be interpreted as legacy BONES-SEED CSV order or MuJoCo pre-order. The decisive FK check on `bones_sonic/240529/macarena_001__A545.npz` gave near-zero FK error only for IsaacLab joint order plus IsaacLab body order; other order assumptions were wrong by roughly 0.1-1.0 m.

Practical rule: when using `bones_sonic`, read `body_pos_w` and `joint_pos` as G1 robot-state target data in IsaacLab order.

## Visualization Lessons

The bad capsule videos were mostly caused by wrong skeleton/order assumptions, not necessarily by bad motions.

The corrected visualization rules are:

- render SONIC target videos directly from `body_pos_w`;
- use the G1 kinematic tree reindexed into IsaacLab body order;
- never connect bodies by adjacent array index;
- prune hands/wrists/head-like distal clutter only after the parent tree is correct;
- render full-length videos when doing manual motion review;
- keep derived videos under `runs/`, never under the source data tree.

The current corrected review artifact has 6 full-length 3D capsule videos. It is enough to prove the visualization path can be made semantically correct, but it is not enough to certify the whole dataset.

## Quality And Filtering Lessons

The motion data should not be treated as perfect. The current quality stance is:

- use `keep`, `downweight`, `quarantine`, and `exclude` rather than a binary keep/drop policy;
- keep quality flags as metadata until thresholds are calibrated;
- treat body-origin contact, XML joint-limit, and body-origin self-collision metrics as provisional unless validated against the actual SONIC exporter and simulator;
- do not promote representative scans into a training policy without a policy audit.

Current bounded SONIC-native quality evidence:

- 256-row smoke: 186 keep, 47 downweight, 23 quarantine.
- 512-row smoke: 384 keep, 78 downweight, 50 quarantine.
- active smoke flags: `sonic_unstable_start_end`, `sonic_joint_velocity_jump`, `sonic_joint_position_jump`, `sonic_ground_penetration`.

This is useful for debugging and next-step prioritization. It is not a promoted curation policy.

The legacy BONES-SEED quality lane was still useful because it clarified which metric families matter:

- joint velocity / acceleration jumps;
- root start/end instability;
- ground penetration;
- foot slide and foot float;
- low contact ratio;
- joint-limit stress;
- self-collision proxy;
- source-target pair timing/provenance mismatch;
- diversity loss by actor, skeleton, category, and split.

But those legacy scan counts and legacy videos must not be used as current SONIC claims.

## Reliable vs Provisional

Reliable:

- `bones_sonic` is the current SONIC target source.
- The NPZ schema, row count, actor count, mirror count, and 50 Hz FPS are known from full header indexing.
- `joint_pos` and `body_pos_w` are SONIC/IsaacLab G1 order.
- `bones_sonic` is not raw SMPL; SMPL is a separate lane.
- The corrected 3D capsule renderer can produce semantically correct full-length videos from `body_pos_w`.

Provisional:

- final SONIC quality thresholds;
- formal `keep/downweight/quarantine/exclude` policy for training;
- whether all SONIC targets are simulator-feasible;
- exact contact and collision labels;
- final source-human input lane for learning;
- final split policy and training sample extraction from SONIC NPZ targets.

## Skip/Defer Decision

The current milestone being skipped is the full SONIC visual-QA / full-curation milestone. It is deferred because it would continue to consume time on visualization, threshold calibration, full scans, and manual review before the train/test/eval path exists.

This skip has a narrow meaning:

- do not block next work on more SONIC videos;
- do not block next work on full SONIC threshold promotion;
- do not claim the data is clean;
- do not use unpromoted quality smoke labels as formal training gates;
- do carry the source/order/visualization lessons into the next split and baseline work.

The next productive lane is to build the SONIC-native train/test/eval path with explicit provenance and quality metadata, then run a small baseline/eval loop. Full curation can be resumed once model/eval artifacts expose which quality labels actually affect training and validation.

## Resume Checklist For This Deferred Lane

When we resume SONIC full curation:

1. Run a full or accepted representative SONIC scan from `sonic_index.csv`.
2. Calibrate thresholds by category/actor/skeleton/split rather than from one smoke run.
3. Export balanced review videos for each active `sonic_*` flag.
4. Fill manual review decisions for worst clips.
5. Promote a named curation policy only through audit/preflight.
6. Add simulator replay/refinement labels before treating target motion as physically valid ground truth.
