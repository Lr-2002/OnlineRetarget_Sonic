# BONES-SONIC Data Source Correction

Date: 2026-05-15.

Current source of truth for the SONIC lane is:

- data root: `/home/user/data/motion_data/bones_sonic`
- metadata join: `/home/user/data/motion_data/metadata/seed_metadata_v003.csv`
- full index: `runs/indices/bones_sonic_index_full_v0/sonic_index.csv`
- full report: `runs/indices/bones_sonic_index_full_v0/sonic_index_report.json`

The earlier `soma_proportional.tar + g1.tar` scans, review CSVs, MuJoCo videos, and 3D capsule videos are legacy BONES-SEED archive evidence. They are not valid evidence for the SONIC NPZ target lane.

## Observed SONIC Schema

The full read-only header scan found 142,220 NPZ files. Every file matched the required schema:

- `fps`: `(1,)`, value `50`
- `joint_pos`: `(T, 29)`
- `joint_vel`: `(T, 29)`
- `body_pos_w`: `(T, 30, 3)`
- `body_quat_w`: `(T, 30, 4)`
- `body_lin_vel_w`: `(T, 30, 3)`
- `body_ang_vel_w`: `(T, 30, 3)`

Full index summary:

- scanned files: 142,220
- metadata matches: 142,220
- actor count: 522
- mirrored files: 71,088
- fps counts: 50 Hz for all files
- frame count min / mean / max: 29 / 364.952742 / 9007
- schema status: 142,220 `ok`

## Provenance Join

The metadata join maps each metadata row's legacy G1 CSV path to the SONIC NPZ path:

```text
g1/csv/<date>/<name>.csv -> bones_sonic/<date>/<name>.npz
```

The index keeps the legacy CSV path only as provenance metadata in `legacy_g1_csv_path`. New SONIC quality scans, review clips, training targets, and reports must read the NPZ arrays directly from `sonic_path`.

## Body And Joint Order

The SONIC NPZ tensors are in IsaacLab G1 order, not legacy BONES-SEED CSV/MJCF pre-order.

Evidence:

- SONIC official G1 config defines `G1_ISAACLAB_JOINTS` and the MuJoCo/IsaacLab reorder tables in `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/envs/manager_env/robots/g1.py`.
- SONIC motionlib converts MuJoCo-order motionlib data to IsaacLab order before command use in `/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/utils/motion_lib/motion_lib_base.py`.
- A direct FK sanity check on `/home/user/data/motion_data/bones_sonic/240529/macarena_001__A545.npz` gave near-zero FK-to-`body_pos_w` error only when both `joint_pos` and `body_pos_w` were interpreted in IsaacLab order:

```text
mode joint/body median mean p95 max
mujoco   mujoco   0.313941 0.393422 0.882623 1.065037
mujoco   isaaclab 0.113680 0.117818 0.359019 0.388547
isaaclab mujoco   0.297972 0.372290 0.930714 1.064054
isaaclab isaaclab 0.000000 0.000000 0.000001 0.000001
```

Implication:

- `src/online_retarget/data/bones_sonic.py::SONIC_JOINT_NAMES` and `SONIC_BODY_NAMES` must stay in IsaacLab order.
- `src/online_retarget/data/sonic_quality.py::SONIC_BODY_PARENTS` is the G1 MJCF tree reindexed into IsaacLab body order.
- The earlier mask-controller P1 `pelvis + child links of actuated joints` body candidate is invalid for direct `bones_sonic` visualization.

SMPL is a separate SONIC input lane. SONIC docs and teleop code use `smpl_joints` with shape `[N, 24, 3]` and `smpl_pose` with shape `[N, 21, 3]`, but those keys are not present in `/home/user/data/motion_data/bones_sonic/*.npz`.

## Current Valid Commands

Build or refresh the SONIC index:

```bash
PYTHONPATH=src python3 scripts/inspect_bones_seed.py build-sonic-index \
  --sonic-root /home/user/data/motion_data/bones_sonic \
  --metadata-csv /home/user/data/motion_data/metadata/seed_metadata_v003.csv \
  --output-root runs \
  --run-name bones_sonic_index_full_v0
```

Inspect the full report:

```bash
jq '{sonic_root,metadata_csv,scanned_files,metadata_found_count,actor_count,mirror_count,fps_counts,schema_status_counts,frame_count_summary,body_order_note,joint_order_note}' \
  runs/indices/bones_sonic_index_full_v0/sonic_index_report.json
```

Run a bounded SONIC-native quality smoke scan:

```bash
PYTHONPATH=src /home/user/repos/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/inspect_bones_seed.py scan-sonic-quality \
  --index-csv runs/indices/bones_sonic_index_full_v0/sonic_index.csv \
  --output-root runs \
  --limit 512 \
  --sample-by category \
  --sample-by date \
  --model-xml /home/user/repos/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --frame-stride 2
```

Current smoke artifact:

- report: `runs/quality/bones_sonic_index_full_v0_sonic_limit512_by-category-date/sonic_quality_report.json`
- stats: `runs/quality/bones_sonic_index_full_v0_sonic_limit512_by-category-date/sonic_quality_stats.jsonl`
- scanned rows: 512
- action counts: 384 keep, 78 downweight, 50 quarantine
- active flags: `sonic_unstable_start_end=92`, `sonic_joint_velocity_jump=34`, `sonic_joint_position_jump=33`, `sonic_ground_penetration=13`

The scanner also records body-origin foot/contact, XML joint-limit, and body-origin self-collision proxy metrics, but these are metric-only by default. They require explicit `--enable-*` flags because SONIC body origins are not sole contact points or collision geometry, and the XML limit source may differ from the SONIC exporter.

## Invalidated Evidence

Treat these as legacy-only for the current SONIC lane:

- `runs/review_clips/*/target_g1.csv`
- `runs/review_clips/*/target_g1_mujoco.mp4`
- `runs/review_clips/*/target_g1_3d_capsules.mp4`
- `runs/quality/*/g1_quality_stats.jsonl` when generated by `scan-g1-quality`
- `runs/quality/*/source_bvh_quality_stats.jsonl`
- `runs/quality/*/source_fk_quality_stats.jsonl`
- `runs/quality/*/pair_quality_stats.jsonl`

These artifacts may still be useful for parser regression or historical notes, but they must not be used to answer "what does SONIC do?" or "which SONIC clips are bad?".

## Next Work

The next curation lane should remain SONIC-native:

1. calibrate SONIC thresholds from representative and then full NPZ scans;
2. export full-length SONIC 3D capsule videos from `body_pos_w`, not from BVH FK or legacy G1 CSV;
3. build train/val/test splits from the SONIC index while preserving actor-heldout grouping;
4. keep all derived artifacts under `runs/`.
