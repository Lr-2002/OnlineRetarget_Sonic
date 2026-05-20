# Orientation Visual Debugging Notes

Date: 2026-05-20

## Conclusion

For SOMA proportional BVH -> BONES-SONIC G1 visual review, apparent left/right or front/back reversal is not enough evidence for a bad pair or bad retarget. In the checked samples, the stronger explanation is camera/viewpoint plus facing-convention ambiguity.

The pair mapping itself has been checked:

- Mapping CSV: `runs/indices/seed_clean_pair_mapping_v0/seed_clean_pair_mapping.csv`
- Full rows: `142,220`
- Filename stem mismatches: `0`
- Date directory mismatches: `0`
- Five diagnostic samples passed duration/hand/yaw checks.

## Standard Overlay

Use orientation overlays before discussing mirror or facing problems.

- red `+X`: world/render `+X`
- green `+Y`: world/render `+Y`
- blue `+Z`: up
- orange `FRONT`: body-facing direction, computed as `cross(left_shoulder - right_shoulder, torso_up)`
- purple `L`: source `LeftHand` or target `left_wrist_yaw_link`
- cyan `R`: source `RightHand` or target `right_wrist_yaw_link`

Judgment order:

1. Check whether purple `L` and cyan `R` attach to the correct physical hands/wrists.
2. Check whether orange `FRONT` differs because the camera is looking from a different side.
3. Only suspect pair mismatch if labels attach to wrong limbs or temporal action labels disagree after coordinate/camera ambiguity is removed.

## Artifacts

Capsule orientation debug:

- Hub: `http://100.76.129.28:5175/runs/online-retarget/20260520-113159-soma-g1-orientation-debug-visuals`
- Local: `runs/vis_pair_check/seed_orientation_debug_5motion_20260520`
- Content: 5 source SOMA BVH capsule videos and 5 target G1 SONIC capsule videos, all with axes, `FRONT`, `L`, and `R`.

IsaacLab orientation debug:

- Hub: `http://100.76.129.28:5175/runs/online-retarget/20260520-123952-soma-g1-isaac-orientation-debug-visuals`
- Local: `runs/vis_pair_check/seed_isaac_orientation_debug_5motion_20260520`
- Content: same 5 BONES-SONIC G1 target motions in IsaacLab kinematic playback, with post-process axes, `FRONT`, `L`, and `R` overlay.

## Scripts

- `scripts/export_orientation_debug_vis.py`: renders source SOMA BVH and target G1 capsule videos with coordinate/facing/hand labels.
- `scripts/overlay_isaac_orientation_debug.py`: overlays coordinate/facing/hand labels on existing IsaacLab playback MP4s using the same BONES-SONIC `body_pos_w`. It does not modify the motion or the original Isaac video.
- `scripts/render_g1_isaac_pair.py`: renders BONES-SONIC G1 NPZ in IsaacLab as kinematic playback.

## Isaac Caveat

The Isaac overlay is a diagnostic post-process. It uses `body_pos_w` and the root-follow camera offset `(2.5, -3.0, 1.6)` to project labels onto the already-rendered video. It is good for visual debugging, but rigorous claims should use 3D `body_pos_w`, body names, joint names, and mapping metadata rather than pixel appearance alone.

## Validation Evidence

Capsule orientation run:

- Published 10 videos, 10 thumbnails, report, summary, and integrity tables.
- Agent Hub manifest valid.
- Tailscale page HTTP 200.

IsaacLab orientation run:

- Published 5 MP4s, 5 thumbnails, report, summary, and integrity tables.
- Videos are 50 FPS with frame counts `182 / 328 / 654 / 166 / 435`.
- Agent Hub manifest valid.
- Tailscale page HTTP 200.

## Future Rule

For motion retarget data review, do not compare unlabelled software capsule and Isaac videos by eye. Always generate coordinate/`FRONT`/`L`/`R` overlays for both views before concluding mirror, rotation, or pair mismatch.
