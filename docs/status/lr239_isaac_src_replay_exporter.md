# LR-239 Isaac/SRC Replay Exporter Contract

This is the minimal patch surface for turning the verified IsaacLab G1 USD playback path into a contact/SRC metric packet exporter.

## Reachable Review Path

Reviewers can inspect the patch after fetching the branch that contains this file:

```bash
cd /mnt/data_cpfs/code/wxh/OnlineRetarget
git fetch origin agent/isaac-lab-humanoid-rl-training/b6126f16
git checkout FETCH_HEAD
```

If the canonical 5090 worktree is dirty, create a separate worktree/clone first and run the same commands there.

## Command Path

Dry-run contract smoke:

```bash
cd /mnt/data_cpfs/code/wxh/OnlineRetarget
PYTHONPATH=src:. /workspace/isaaclab/_isaac_sim/python.sh scripts/export_lr239_isaac_src_packets.py \
  --paired-state-h5 /mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/lr239_staged_g1_state_fk_export_20260602T180710Z/soma_uniform/paired_g1_state.h5 \
  --robot-usd /mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd \
  --output-dir /mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/lr239_isaac_src_replay_smoke/soma_uniform \
  --variant soma_uniform \
  --max-frames 64 \
  --dry-run
```

This command intentionally does not depend on `direnv` or `conda`; it uses the already-verified Isaac launcher directly.

Expected dry-run artifacts:

- `replay_manifest.json`
- `packet_schema.json`
- empty placeholder `isaac_src_packets.jsonl`

Packet metric aggregation:

```bash
cd /mnt/data_cpfs/code/wxh/OnlineRetarget
PYTHONPATH=src:. python3 scripts/summarize_lr239_isaac_src_packets.py \
  --input-jsonl /path/to/isaac_src_packets.jsonl \
  --output-dir /path/to/packet_metrics
```

Expected metric artifacts:

- `packet_metric_summary.json`
- `per_frame_side_metrics.json`
- `per_frame_side_metrics.jsonl`
- `per_frame_side_metrics.csv`
- `per_frame_side_foot_metrics.csv`

## Packet Contract

`online_retarget.isaac_src_replay` declares schema `lr239.isaac_src_contact_packets.v1`.

Each JSONL packet is one frame with paired `pred` and `target` state packets. State packets carry root pose, 29-DOF joint state, foot contact force/contact flags, support margin/floating guard, rolling foot artifact fields, foot-ground support pairs, body-pair/self-collision availability fields, and cross-ratio availability fields. The packet schema pins the joint order, foot links, disabled collision pairs, contact filters, ground frame, support thresholds, foot artifact thresholds, and cross-ratio contract.

Contact families are intentionally separated:

- `foot_ground_contact_pairs` is support-only evidence from each single-body foot sensor's filtered `force_matrix_w` entry for the configured ground prim. Aggregate `net_forces_w` is not accepted as `/World/Ground` evidence.
- `foot_ground_contact_status=blocked` means filtered `force_matrix_w` is absent or unusable; in that case foot-ground pairs are empty, foot force entries are `null`, and `floating_guard` is `null`.
- `foot_slide_speed_mps`, `foot_slide_flags`, `foot_skate_distance_m`, `foot_skate_flags`, `foot_float_clearance_m`, and `foot_float_flags` are produced by exporter-local `FootArtifactTracker` rolling state. They are available only when the current per-foot `/World/Ground` contact source is verified through filtered `force_matrix_w`; if foot-ground contact is blocked, these fields are `null` per foot with `foot_artifact_status=blocked`.
- Foot slide uses horizontal foot speed only when the current and previous frames for that side are both verified in contact. Foot skate is rolling horizontal displacement within the current verified contact segment. Foot float is foot clearance during a verified contact frame. These are foot-ground artifacts only, not body-body collision evidence.
- `contact_pairs` is a compatibility alias for `foot_ground_contact_pairs`; it is not a body-body or self-collision source.
- `body_pair_contacts` is produced only from separate single-body body-pair sensors filtered to candidate robot bodies through IsaacLab `force_matrix_w`. Aggregate `net_forces_w`, foot-ground support pairs, and `/World/Ground` filters are not accepted as body-pair evidence.
- `body_pair_contact_status=blocked` means at least one required body-body filtered matrix is absent or unusable; in that case `body_pair_contacts` and `self_collision_count` stay `null`.
- `self_collision_count` is computed only after all configured body-body filtered matrices are verified and disabled pairs are excluded. A numeric `0` is valid only in that available state, when every remaining filtered body-body pair is below threshold.
- `cross_ratio` and `cross_ratio_guard` are `null` with `cross_ratio_status=blocked` until an SRC geometry checker is bound.

## Packet Metric Aggregator

`online_retarget.isaac_src_metrics` consumes `isaac_src_packets.jsonl` and writes summary plus per-frame/per-side CSV/JSON artifacts for review and LR-235 ingestion. The first implementation aggregates only the clear metric families:

- Foot-ground contact: foot-ground availability rate, per-foot contact rate, force summary, and support pair count.
- Footskate / slide: `foot_slide_speed_mps`, `foot_skate_distance_m`, and their flags summarized as max/mean/rate where values are available.
- Floating/support: `floating_guard`, `support_margin_m`, and `foot_float_clearance_m` summarized as guard violation/pass rates and clearance/support statistics. This does not attempt an airborne/action mask.
- Blocked/null accounting: `body_pair_contacts`, `self_collision_count`, `cross_ratio`, and `cross_ratio_guard` are summarized through status counts and null/non-null counts. Blocked/null values are never converted into numeric zero metrics. `self_collision_count=0` is meaningful only when `self_collision_status=available`.

## Isaac Binding

The non-dry implementation is now bound to IsaacLab behind the existing CLI. It:

- Spawn the verified G1 USD with `enabled_self_collisions=True`.
- Ensure the spawner activates PhysX contact reporters/contact sensors; IsaacLab contact sensors require contact reporter activation on the rigid bodies.
- Instantiate one `ContactSensorCfg` per declared foot link, using a single foot body prim per sensor, and filter each sensor to `/World/Ground`.
- Compute foot-ground support only from filtered `force_matrix_w`; if IsaacLab does not provide that filtered matrix, leave foot-ground support blocked instead of falling back to aggregate `net_forces_w`.
- Instantiate separate single-body `ContactSensorCfg` body-pair sensors for unique candidate body pairs. Each source-body sensor filters only to candidate robot target bodies after excluding `/World/Ground`, configured contact filter prims, duplicate reverse pairs, and configured disabled collision pairs.
- Compute `body_pair_contacts` and `self_collision_count` only from those body-pair filtered `force_matrix_w` matrices. If any required filtered body-body matrix is missing, leave the body-pair/self-collision fields blocked/null. If all matrices are present and no filtered pair exceeds threshold, emit `body_pair_contacts=[]` and `self_collision_count=0`.
- Compute footskate/slide/floating artifact fields only from verified foot-ground flags plus foot body poses; do not infer them from aggregate forces or proxy collision geometry.
- Replay `pred_g1_state` and `target_g1_state` from `paired_g1_state.h5` in SONIC joint order.
- Serialize `packet_schema.json`-compatible JSONL for LR-235 consumption, with SRC/cross-ratio fields blocked/null unless their verified source is present. The exporter does not substitute FK/body-origin/sphere proxy metrics for formal SRC or cross-ratio values.

Non-dry execution still requires Code Reviewer approval before running on the 10h LR-239 artifacts. Before review, use only dry-run or import/preflight smoke checks. The implementation fails closed with a blocked manifest if IsaacLab/SRC imports, the G1 USD, or required HDF5 state fields are unavailable; it does not fabricate contact, self-collision, support, or cross-ratio values on machines that cannot run the Isaac/SRC contact path.
