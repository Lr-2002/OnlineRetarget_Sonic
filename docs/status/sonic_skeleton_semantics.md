# SONIC Skeleton Semantics

Date: 2026-05-15.

This note is the durable lookup point for the current BONES-SONIC skeleton conclusion.

## What `bones_sonic` Contains

`/home/user/data/motion_data/bones_sonic/*.npz` is a robot-state dataset, not raw SMPL:

- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`
- no `smpl_joints`
- no `smpl_pose`

These arrays are G1-aligned. They are useful for target-side data usability and visual QA, but they do not by themselves expose the 24-joint human SMPL skeleton.

## SONIC SMPL Lane

SONIC still has a real SMPL path, but it is separate from `bones_sonic`:

- ZMQ protocol docs define `smpl_joints` as `[N, 24, 3]` and `smpl_pose` as `[N, 21, 3]`.
- `gear_sonic/scripts/pico_manager_thread_server.py` computes local SMPL joints and SMPL pose for teleop.
- `gear_sonic/utils/teleop/vis/vr3pt_pose_visualizer.py` defines the standard 24-joint SMPL topology:

```text
[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
```

The official SONIC release config uses both:

- robot motion file: `data/motion_lib_bones_seed/robot_filtered`
- SMPL motion file: `data/bones_seed_smpl`

So the correct mental model is not "SONIC equals SMPL 24 joints"; it is "SONIC can use a SMPL input lane, while the local `bones_sonic` NPZ directory is already G1 robot-state data."

2026-05-18 clarification: the local SONIC training code also contains a SOMA lane. `all_mlp_v1.yaml` uses `g1`, `teleop`, and `smpl` encoders, while `all_mlp_v1_soma.yaml` adds a `soma` encoder whose inputs are `soma_joints_multi_future_local_nonflat`, `soma_root_ori_b_multi_future`, and wrist joint features. The preprocessing script `extract_soma_joints_from_bvh.py` parses BONES/SOMA BVH and writes `soma_joints`, `soma_root_quat`, and `soma_transl` PKLs. Therefore SONIC's training stack can consume BONES in its released SOMA/BVH form after preprocessing; a separate SMPL-converted lane may exist for SMPL encoder training/eval, but it is not the format of the official public BONES-SEED files.

## G1 Order

`bones_sonic` arrays use IsaacLab G1 order.

The order source is the official SONIC G1 config:

```text
/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/envs/manager_env/robots/g1.py
```

Important body indices:

```text
0  pelvis
1  left_hip_pitch_link
2  right_hip_pitch_link
3  waist_yaw_link
9  torso_link
18 left_ankle_roll_link
19 right_ankle_roll_link
28 left_wrist_yaw_link
29 right_wrist_yaw_link
```

Important joint indices:

```text
0  left_hip_pitch_joint
1  right_hip_pitch_joint
2  waist_yaw_joint
17 left_ankle_roll_joint
18 right_ankle_roll_joint
23 left_wrist_roll_joint
24 right_wrist_roll_joint
25 left_wrist_pitch_joint
26 right_wrist_pitch_joint
27 left_wrist_yaw_joint
28 right_wrist_yaw_joint
```

The G1 capsule visualization should use the MJCF kinematic tree reindexed into this IsaacLab body order. Do not connect bodies by raw adjacent array index.

## FK Sanity Check

The discriminating check was: compare `body_pos_w` against FK from the official G1 MJCF under four interpretations.

Sample:

```text
/home/user/data/motion_data/bones_sonic/240529/macarena_001__A545.npz
```

Result:

```text
mode joint/body median mean p95 max
mujoco   mujoco   0.313941 0.393422 0.882623 1.065037
mujoco   isaaclab 0.113680 0.117818 0.359019 0.388547
isaaclab mujoco   0.297972 0.372290 0.930714 1.064054
isaaclab isaaclab 0.000000 0.000000 0.000001 0.000001
```

Conclusion: both `joint_pos` and `body_pos_w` are IsaacLab order. The old candidate order explains the visually wrong capsule links.

## Practical Rules

- For direct `bones_sonic` visual QA, render from `body_pos_w` using `SONIC_BODY_NAMES` in IsaacLab order.
- For source-human visualization, do not assume `bones_sonic` has SMPL. Use a real SMPL file/lane or the linked SOMA/BVH provenance separately.
- When pruning clutter, remove distal hands/fingers and face/head markers only after the correct parent tree is established.
- Derived artifacts go under `runs/`; never modify `/home/user/data/motion_data`.
