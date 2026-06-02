# Accepted SOMA/G1 Validation Service

This runbook fixes the LR-106 accepted visualization path as the standard delta validation entrypoint for training owners.

## One Command

Run on delta, not on a local Mac:

```bash
cd /home/user/project/OnlineRetarget
/home/user/venvs/isaaclab-210/bin/python scripts/run_accepted_soma_g1_validation.py \
  --sample-limit 1 \
  --run-name lr117_smoke_accepted_soma_g1
```

For the full LR-106 fixed set, omit `--sample-limit 1`.
The default target view is LR-117-grade: IsaacLab G1 playback uses root-zeroed
relative XY, `--camera-mode follow`, smoothed root-XY framing, orientation
overlays, and an 80m blue ground plane. Override `--ground-size`,
`--camera-offset`, or `--camera-framing-margin` only when the resulting manifest
still records a readable follow/framing policy.

## Input Schema

Default samples are the 8 LR-106 fixed validation clips. Custom samples can be supplied with either:

```bash
--sample 220720:itching_neck_003__A032_M:200
```

or a CSV/JSON manifest:

```csv
date,stem,frames
220720,itching_neck_003__A032_M,200
```

Required path contract:

| Field | Default |
| --- | --- |
| source BVH | `/home/user/data/motion_data/clean_data/soma_proportional/bvh/<date>/<stem>.bvh` |
| source SOMA motionlib | `outputs/lr106_stage/soma_filtered_v1/<date>__<stem>.pkl` |
| target G1 motionlib | `outputs/lr106_stage/robot_soma_paired_v1/<date>__<stem>.pkl` |
| G1 USD | `runs/isaaclab_urdf_cache/g1_main/main.usd` |
| G1 MJCF for overlay FK | `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml` |

## Output Contract

Each clip directory contains:

| Artifact | Meaning |
| --- | --- |
| `source_somabvh_somamesh.mp4/json` | SomaMesh/global-SOMA source panel |
| `target_g1_isaaclab.mp4/json` | IsaacLab G1 kinematic playback |
| `overlay/orientation/.../target_g1_isaaclab_orientation_debug.mp4/json` | target playback with world/root axes and semantic front/L/R markers |
| `<stem>_somamesh_source_g1_isaac_with_axes.mp4` | final side-by-side accepted validation video |
| `final_report.json` | per-clip accepted contract and ffprobe evidence |

Run root contains:

| Artifact | Meaning |
| --- | --- |
| `summary.json` | machine-readable manifest and standard output contract |
| `accepted_soma_g1_validation_report.md` | human-readable run report |
| `batch_render.log` | reproducible command log |
| `agenthub_upload.json` | Agent Hub upload result when enabled |

## Accepted Visualization Contract

- Source: SomaBVH/SomaMesh LBS, not capsule/stick BVH.
- Source display conversion: `(x, y, z)_display = (x, -z, y)_soma`.
- Source camera/root: `Hips`/pelvis reference, smoothed horizontal follow, fixed look-at height.
- Target: IsaacLab G1 kinematic playback from motionlib PKL.
- Target ground/framing: root-zeroed relative XY by default, 80m IsaacLab ground plane, follow camera with smoothed root XY and stable look-at height.
- Root quaternion: motionlib `root_rot` is `xyzw`; IsaacLab write uses `wxyz`.
- Overlays: world axes and root local axes are visible.
- L/R markers: semantic body names only; never screen-side inference.
- `changed_frames`, fps, frame count, joint order, G1 asset path, and ffprobe video stats are recorded.

## What This Gate Does Not Prove

This service proves a standardized kinematic visualization path and semantic playback evidence. It does not prove policy tracking, dynamics, balance, torque feasibility, sim2sim, sim2real, or training convergence.

## Troubleshooting

| Symptom | Likely owner / cause | Minimum next check |
| --- | --- | --- |
| `data_or_asset_missing` | dataset/stage artifact missing | check source BVH, `outputs/lr106_stage`, G1 USD, G1 MJCF paths |
| source render fails | SOMA renderer dependency | verify `/home/user/project/ContextRetarget/third_party/soma-retargeter/.venv/bin/python` and SOMA USD |
| target render fails or hangs | IsaacLab/G1 renderer | rerun one clip with `--fast-exit-after-report`; check IsaacLab venv and GPU |
| overlay fails | G1 FK/body order | check `SONIC_BODY_NAMES`, `sonic_joint_values_to_g1_columns`, and G1 MJCF |
| Agent Hub upload `401`/failed | AgentHub auth/upload config | keep remote output path as valid smoke evidence; ask AgentHub owner for token/config |
