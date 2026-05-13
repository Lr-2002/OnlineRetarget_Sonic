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
