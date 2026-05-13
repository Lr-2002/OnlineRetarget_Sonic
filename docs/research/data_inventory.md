# Data Inventory

Scan date: 2026-05-13.

Local data root: `/home/user/data/motion_data`.

Rule: this path is read-only. Do not unpack archives, write caches, or create derived datasets under this directory.

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

Use `actor_uid` as the first skeleton/person grouping key. Each actor has one `move_soma_proportional_shape_path`, and metadata includes morphology dimensions:

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

### Existing G1 NPZ Targets

`bones_sonic` contains 142,220 `npz` files. Sample files contain:

- `fps`: `(1,)`
- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`

`AMASS/GMR_retarget_data` contains 25,379 `npz` files with the same observed key schema.

These `npz` targets are useful for first offline supervised learning because they already include body-space physical state features. The open question is whether they are canonical BONES-SEED outputs, GMR outputs, or a downstream processed version; do not assume provenance without tracing filenames and scripts.

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

Current scope: this is metadata-level curation only. Clip-level physical quality flags still need a motion-stat scanner.

## G1 Target Quality Smoke

Command:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py scan-g1-quality \
  --data-root /home/user/data/motion_data \
  --index-csv runs/indices/actor_split_t80_v10_x10_s17_metadata_balanced_v0/split_index.csv \
  --output-root runs \
  --limit 100 \
  --fps 30 \
  --max-joint-velocity 20 \
  --max-root-speed 8
```

Observed result on the first 100 indexed G1 CSV targets:

- Actions: 29 keep, 71 quarantine
- Flags: 71 `joint_velocity_jump`, 28 `root_discontinuity`
- `max_abs_joint_velocity` p50/p95/max: 70.045590 / 444.556812 / 538.624500
- `max_root_speed` p50/p95/max: 3.199689 / 35.318570 / 36.870913

Interpretation: the scanner works, but these temporary thresholds are not calibrated. The metric summary should be used to choose per-category or percentile-based thresholds before filtering training data aggressively.

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
