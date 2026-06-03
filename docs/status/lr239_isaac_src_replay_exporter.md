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

## Packet Contract

`online_retarget.isaac_src_replay` declares schema `lr239.isaac_src_contact_packets.v1`.

Each JSONL packet is one frame with paired `pred` and `target` state packets. State packets carry root pose, 29-DOF joint state, foot contact force/contact flags, support margin/floating guard, self-collision count, contact pairs, and cross-ratio fields. The packet schema pins the joint order, foot links, disabled collision pairs, contact filters, ground frame, support thresholds, and cross-ratio contract.

## Isaac Binding

The non-dry implementation is now bound to IsaacLab behind the existing CLI. It:

- Spawn the verified G1 USD with `enabled_self_collisions=True`.
- Ensure the spawner activates PhysX contact reporters/contact sensors; IsaacLab contact sensors require contact reporter activation on the rigid bodies.
- Instantiate `ContactSensorCfg` for the declared foot links and filter them to `/World/Ground`.
- Replay `pred_g1_state` and `target_g1_state` from `paired_g1_state.h5` in SONIC joint order.
- Serialize `packet_schema.json`-compatible JSONL for LR-235 consumption.

Non-dry execution still requires Code Reviewer approval before running on the 10h LR-239 artifacts. Before review, use only dry-run or import/preflight smoke checks. The implementation fails closed with a blocked manifest if IsaacLab/SRC imports, the G1 USD, or required HDF5 state fields are unavailable; it does not fabricate contact, self-collision, support, or cross-ratio values on machines that cannot run the Isaac/SRC contact path.
