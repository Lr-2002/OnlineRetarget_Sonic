# Data Inventory

Scan date: 2026-05-13.

Correction on 2026-05-15: the current SONIC lane must use `/home/user/data/motion_data/bones_sonic` as the target data source. Earlier `soma_proportional.tar + g1.tar` archive scans remain legacy BONES-SEED evidence and must not be used as SONIC quality or video evidence. See `docs/status/sonic_data_source.md`.

Local data root: `/home/user/data/motion_data`.

Rule: this path is read-only. Do not unpack archives, write caches, or create derived datasets under this directory.

## SMPL-Family Availability

AMASS in its original release is a SMPL-family motion dataset: the official AMASS project describes MoSh++ fitting mocap into a common rigged body-model parameterization, and the AMASS loader documents SMPL+H with compatibility to SMPL and SMPL-X.

That is not what the local `/home/user/data/motion_data/AMASS/GMR_retarget_data` directory contains. Local scans found G1/robot-retargeted schemas there, not raw AMASS SMPL/SMPL-H/SMPL-X motion parameters. In particular, the observed local files do not expose NMR-style human input keys such as `trans`, `root_orient`, `pose_body`, `transl`, `global_orient`, or `body_pose`.

The official BONES-SEED public dataset is also not distributed as SMPL/SMPL-X motion parameters. Its documented release formats are:

- SOMA Uniform BVH
- SOMA Proportional BVH plus actor shape metadata
- Unitree G1 MuJoCo-compatible trajectories

Public documentation from BONES/Hugging Face and NVIDIA Kimodo describes BONES-SEED as SOMA/G1 data, including SOMA 77-joint BVH. This means official NMR-on-BONES still requires an adapter:

```text
BONES SOMA BVH + SOMA actor shape metadata -> SMPL-X/AMASS-style npz -> NMR
```

Search note, 2026-05-18: a third-party Hugging Face dataset `zirobtc/bone` appears to contain a `bones-seed-smplx.tar.gz` WebDataset with visible `betas`-like fields. It has no dataset card and is not the official `bones-studio/seed` release, so it must be treated as unverified provenance until we inspect samples, keys, frame alignment, license compatibility, and correspondence to official BONES metadata/G1 targets.

## Top-Level Layout

Observed files:

- `README.md`
- `LICENSE.md`
- `metadata/seed_metadata_v003.csv`
- `metadata/seed_metadata_v003.parquet`
- `metadata/seed_metadata_v002_temporal_labels.jsonl`
- `g1.tar`
- `soma_uniform.tar`
- `soma_proportional.tar`
- `soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh`
- `soma_shapes/soma_base_rig/soma_base_skel_minimal.usd`
- `bones_sonic/.../*.npz`
- `AMASS/GMR_retarget_data/.../*.npz`

Size: about 733G.

Extension counts from scan:

- `npz`: 142,834
- `txt`: 45
- `pkl`: 20
- `mp4`: 20
- `tar`: 3
- `md`: 2
- `usd`: 1
- `parquet`: 1
- `jsonl`: 1
- `csv`: 1
- `bvh`: 1

## BONES-SEED Metadata

Local `metadata/seed_metadata_v003.csv` summary:

- Rows: 142,220
- Actor/proportional skeleton IDs: 522
- Mirrored rows: 71,088
- Missing G1 target paths: 0
- Missing SOMA proportional paths: 0
- Actor height range: 145-199 cm

Top packages:

- Locomotion: 74,488
- Communication: 21,493
- Interactions: 14,643
- Dances: 11,006
- Gaming: 8,700
- Everyday: 5,816
- Sport: 3,993
- Other: 2,081

Top categories:

- Basic Locomotion Neutral: 33,430
- Baseline: 22,878
- Gestures: 17,590
- Object Manipulation: 11,620
- Dancing: 11,006
- Object Interaction: 10,817
- Basic Locomotion Styles: 10,746
- Advanced Locomotion: 6,036
- Sports: 3,973
- Communication: 3,723

## Skeleton Grouping

Use `actor_uid` as the first skeleton/person grouping key. The initial learning and curation path must use BONES-SEED `soma_proportional`, not `soma_uniform`: `move_soma_proportional_path` points into `soma_proportional.tar`, and `move_soma_proportional_shape_path` points to a per-actor shape under `soma_shapes/soma_proportion_fit_mhr_params/`. The `soma_uniform` path uses the shared `soma_base_fit_mhr_params.npz` standard skeleton and should only be used later as an ablation, not as the main heterogeneous-skeleton input.

Current learning target definition: **SOMA proportional -> G1**. Source samples must be built from `move_soma_proportional_path` and actor morphology/shape metadata; target samples must be built from the matched G1 target lane, currently `bones_sonic` NPZ files joined from metadata. `SOMA uniform`, AMASS/GMR retargeted NPZ, and G1 body-state reconstruction are not valid substitutes for this main mapping unless the experiment is explicitly labeled as an ablation/debug lane.

Each actor has one proportional shape path, and metadata includes morphology dimensions:

- height
- foot length
- collarbone height/span
- elbow span
- wrist span
- shoulder span
- hip height/span
- knee height
- ankle height
- weight, age, gender

Raw shape files under `soma_shapes/soma_proportion_fit_mhr_params/{actor_id}.npz`
provide a richer per-actor SOMA shape parameterization. A spot check of actors
such as `A001`, `A002`, `A548`, and `A551` found these keys:

- `identity_params`: `(1, 45)` float32
- `scale_params`: `(1, 68)` float32
- `pose_params`: `(1, 136)` float32
- `face_expr_params`: `(1, 72)` float32, currently all zeros in checked files

The local directory has 529 proportional shape files, while metadata references
522 actor shape paths; all 522 referenced paths exist. The extra local shape
files are not referenced by `seed_metadata_v003.csv` and should not be sampled
unless a later provenance check explains them.

The SOMA BVH skeleton topology is stable across base and proportional samples:
the checked base rig and `A001` proportional sample both expose 78 named joints.
The heterogeneity comes from per-actor offsets/limb lengths and per-actor shape
parameters, not from a different joint-name topology. For example, the checked
`A001` proportional sample has shorter arm/leg offsets than the base rig, while
preserving names such as `Hips`, `Spine1`, `Chest`, `LeftArm`, `LeftForeArm`,
`LeftHand`, `LeftLeg`, `LeftShin`, `LeftFoot`, and `LeftToeBase`.

Current trainable samples use a reduced 30-body source subset from SOMA BVH FK,
not all 78 joints. That subset keeps trunk, neck/head, arms, legs, toes, and a
few first finger joints; it does not include face/eye/jaw/end-site clutter. The
current morphology side channel uses the 11 metadata measurements plus weight
and age, for 13 numeric dimensions. It does not yet consume the raw 45/68/136
SOMA shape parameters.

First split policy: split by `actor_uid`, not by clip, to measure cross-person/cross-skeleton generalization.

## Target Motion Formats

### G1 CSV Archive

`g1.tar` contains 142,220 CSV files under `g1/csv/{date}/{filename}.csv`.

Sample CSV header has 36 columns:

- `Frame`
- root translation: `root_translateX`, `root_translateY`, `root_translateZ`
- root rotation: `root_rotateX`, `root_rotateY`, `root_rotateZ`
- 29 G1 joint columns:
  - `left_hip_pitch_joint_dof`
  - `left_hip_roll_joint_dof`
  - `left_hip_yaw_joint_dof`
  - `left_knee_joint_dof`
  - `left_ankle_pitch_joint_dof`
  - `left_ankle_roll_joint_dof`
  - `right_hip_pitch_joint_dof`
  - `right_hip_roll_joint_dof`
  - `right_hip_yaw_joint_dof`
  - `right_knee_joint_dof`
  - `right_ankle_pitch_joint_dof`
  - `right_ankle_roll_joint_dof`
  - `waist_yaw_joint_dof`
  - `waist_roll_joint_dof`
  - `waist_pitch_joint_dof`
  - `left_shoulder_pitch_joint_dof`
  - `left_shoulder_roll_joint_dof`
  - `left_shoulder_yaw_joint_dof`
  - `left_elbow_joint_dof`
  - `left_wrist_roll_joint_dof`
  - `left_wrist_pitch_joint_dof`
  - `left_wrist_yaw_joint_dof`
  - `right_shoulder_pitch_joint_dof`
  - `right_shoulder_roll_joint_dof`
  - `right_shoulder_yaw_joint_dof`
  - `right_elbow_joint_dof`
  - `right_wrist_roll_joint_dof`
  - `right_wrist_pitch_joint_dof`
  - `right_wrist_yaw_joint_dof`

### BONES-SONIC NPZ Targets

`bones_sonic` contains 142,220 `npz` files. Sample files contain:

- `fps`: `(1,)`
- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`

`AMASS/GMR_retarget_data` contains 25,379 `npz` files with the same observed key schema.

These `npz` targets are the active SONIC target lane. The full read-only index at `runs/indices/bones_sonic_index_full_v0` found 142,220 files, 142,220 metadata matches, 522 actors, 71,088 mirrored files, all at 50 Hz, and all with schema status `ok`.

The `joint_pos` and `body_pos_w` arrays use SONIC/IsaacLab G1 order, not legacy G1 CSV/MJCF pre-order. See `docs/status/sonic_skeleton_semantics.md` for the FK verification and canonical index examples.

Metadata provenance is joined by converting each metadata `move_g1_path`:

```text
g1/csv/<date>/<name>.csv -> bones_sonic/<date>/<name>.npz
```

The legacy G1 CSV path is retained only as provenance in the SONIC index. New SONIC quality scans, review videos, and supervised target extraction must read the NPZ tensors directly.

`AMASS/GMR_retarget_data` contains 25,379 `npz` files with the same observed key schema. That path is not the current SONIC lane.

## Reproducible Inventory Command

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py inventory --data-root /home/user/data/motion_data
```

This command reads only `metadata/seed_metadata_v003.csv` and prints JSON.

## Reproducible Split Index Command

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py split-index \
  --data-root /home/user/data/motion_data \
  --output-root runs \
  --seed 17 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --policy-name metadata_balanced_v0 \
  --min-duration-frames 60
```

Observed result on 2026-05-13:

- Rows: 142,220
- Actors: 522
- Actor split train/val/test: 417 / 52 / 53
- Row split train/val/test: 112,789 / 15,760 / 13,671
- Curation actions: 71,132 keep, 71,088 downweight
- Quality flags: 71,088 `mirror_variant`

Current scope: this is metadata-level curation only. Clip-level physical quality flags are handled by the M2Q scanners and remain provisional until a named policy is promoted.

## Timing Assumption

Representative pair scans found the source BVH `Frame Time` is `0.008333333333333333` for sampled BONES-SEED clips, i.e. about 120 Hz. The paired G1 CSVs have the same frame counts as the source BVHs in the 560-row category/split sample.

Implication:

- G1 quality scans should use 120 Hz unless a specific target source proves otherwise. The scanner default is now 120 Hz.
- Source FK/contact scans derive FPS from each BVH `Frame Time` by default. Do not pass `--fps 30` for BONES-SEED calibration unless explicitly testing an override.
- Older 30 Hz smoke artifacts are useful for scanner regression only; velocity, slide, duration, and start/end metrics from those runs are time-scale suspect.

## G1 Target Quality Smoke

Command:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-g1-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --fps 120 \
  --max-joint-velocity 20 \
  --max-root-speed 8
```

Observed result on the first 100 indexed G1 CSV targets:

- Actions: 29 keep, 71 quarantine
- Flags: 71 `joint_velocity_jump`, 28 `root_discontinuity`
- `max_abs_joint_velocity` p50/p95/max: 70.045590 / 444.556812 / 538.624500
- `max_root_speed` p50/p95/max: 3.199689 / 35.318570 / 36.870913

Interpretation: this old first-100 smoke proves the parser path works, but it was run with a 30 Hz assumption and should not be used to calibrate velocity or contact thresholds. Use the 120 Hz representative artifacts below for current M2Q discussion.

## Source BVH Quality Smoke

Command:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-source-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --max-channel-velocity 3000 \
  --max-root-speed 500
```

Observed result on the first 100 indexed SOMA proportional BVH source motions:

- Actions: 59 keep, 40 quarantine, 1 exclude
- Flags: 40 `source_channel_jump`, 1 `nonfinite_value`, 1 `channel_width_mismatch`
- `max_abs_channel_velocity` p50/p95/max: 586.721289 / 43245.538600 / 43396.294372
- `max_root_speed` p50/p95/max: 16.008700 / 170.172050 / 174.569584

Interpretation: source clips also need curation. Some very large channel velocities are likely angular wrap/discontinuity artifacts, while the excluded sample has real nonfinite/channel-width issues in the BVH motion rows. Thresholds are still provisional.

## Threshold Proposal Smoke

Command examples:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py propose-thresholds \
  --stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_quality_stats.jsonl \
  --output-json runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit100/g1_threshold_proposals_p95.json \
  --metric max_abs_joint_velocity \
  --metric joint_jump_rate \
  --metric max_root_speed \
  --percentile 0.95 \
  --action quarantine
```

Current p95 proposals from 100-sample smoke scans:

- G1 `max_abs_joint_velocity`: 444.556812
- G1 `joint_jump_rate`: 0.158423
- G1 `max_root_speed`: 35.318570
- Source BVH `max_abs_channel_velocity`: 43245.538600
- Source BVH `channel_jump_rate`: 0.000011
- Source BVH `max_root_speed`: 170.172050

These are not final thresholds. They are traceable proposals to motivate larger scans and per-category calibration.

## Pair Quality Representative Scan

Command:

```bash
PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py scan-pair-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 560 \
  --sample-by category \
  --sample-by split \
  --expected-source-frame-time 0.008333333333333333 \
  --g1-fps 120 \
  --max-frame-count-delta 0 \
  --max-duration-delta-sec 0.001 \
  --target-provenance kinematic_g1_csv
```

Observed artifact:

- Report: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit560_by-category-split/pair_quality_report.json`
- Stats: `runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit560_by-category-split/pair_quality_stats.jsonl`
- Scanned rows: 560 category/split-stratified rows.
- Actions: 494 keep, 66 quarantine.
- Flags: 66 `pair_duration_mismatch`.
- Frame-count delta: max 0, p95 0.
- Absolute duration delta: p50 0.000368 s, p95 0.001768 s, max 0.002807 s.
- Target provenance label: `kinematic_g1_csv`.

Interpretation: source and G1 frame counts match in the representative sample, but strict 1 ms duration tolerance quarantines 66 rows because BVH frame time and exact `1 / 120` are not bit-identical over long clips. This should be treated as a provenance/timing calibration issue, not evidence that the files are missing frames.

## Representative Four-Way Curation Merge

Current strongest M2Q artifact merges source BVH, source FK/contact, G1 FK/contact/self-collision proxy, and pair/provenance stats:

```bash
PYTHONPATH=src:. python3 scripts/inspect_bones_seed.py merge-quality \
  --split-index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --source-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_limit560_by-category-split/source_bvh_quality_stats.jsonl \
  --source-fk-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_source_fk_limit560_by-category-split/source_fk_quality_stats.jsonl \
  --g1-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_limit560_by-category-split/g1_quality_stats.jsonl \
  --pair-stats-jsonl runs/quality/actor_split_t80_v10_x10_s17_metadata_balanced_v0_pair_limit560_by-category-split/pair_quality_stats.jsonl \
  --output-root runs \
  --run-name representative_source_g1_pair_limit560_by_category_split
```

Observed result:

- Report: `runs/curated/representative_source_g1_pair_limit560_by_category_split/curated_report.json`
- Curated index: `runs/curated/representative_source_g1_pair_limit560_by_category_split/curated_index.csv`
- Merged sampled rows: 560 source, 560 source-FK, 560 G1, 560 pair.
- Full-index actions after sampled-row merge: 70,858 keep, 70,876 downweight, 479 quarantine, 7 exclude.
- Main flags: 71,088 `mirror_variant`, 485 `g1:g1_foot_slide`, 244 `source_fk:source_foot_slide`, 238 `g1:g1_joint_limit_violation`, 234 `g1:g1_ground_penetration`, 168 `g1:g1_unstable_start_end`, 96 `g1:joint_velocity_jump`, 66 `pair:pair_duration_mismatch`.
- Diversity-loss summary: 0 lost actor groups out of 522, 0 lost source-skeleton groups out of 522, 0 lost category groups out of 20, and 0 lost split groups out of 3.
- Manual review manifest: `runs/curated/representative_source_g1_pair_limit560_by_category_split/manual_review/review_manifest.md`, 35 review rows across float, foot-slide, joint-limit, jump, mirror, parser, and penetration families.

This is still a representative calibration artifact, not a promoted formal curation policy.
