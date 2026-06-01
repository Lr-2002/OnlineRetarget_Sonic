# LR-177 Accepted Visualization Patch

This patch prepares the LR-177 post-training visual package without launching training or GPU render jobs.

## Contract

- Source panel: `scripts/render_somamesh_source.py` loads the Soma USD skeletal mesh and renders CPU LBS triangles. Accepted source reports must include `vertices > 0`, `triangles_loaded > 0`, `renderer`, and `not_capsule_bvh_visualizer=true`.
- Target panel: `scripts/render_g1_isaac_pair.py` renders IsaacLab G1 kinematic playback with parameterized blue ground and camera framing metadata.
- LR-177 runner: `scripts/run_lr177_accepted_clean_validation.py` defaults to preserving world root, drawing orientation labels, true SomaMesh source rendering, blue large ground, and follow-camera auto-framing.

## Required Render Host Assets

Do not claim final SomaMesh render unless these paths exist on the render host:

```bash
test -d /home/user/project/ContextRetarget/third_party/soma-retargeter
test -x /home/user/project/ContextRetarget/third_party/soma-retargeter/.venv/bin/python
test -f /home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd
test -d /home/user/data/motion_data/clean_data/soma_proportional/bvh
```

## One-Clip Smoke Command

This is the command contract for the first MLOps smoke after assets and a clean GPU window are available. It is intentionally not executed by the code patch task.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lr177_accepted_clean_validation.py \
  --config configs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_uniform_4gpu.json \
  --run-dir /mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_runs/lr177_a0_no_skel_formal_20260601T0353Z/uniform \
  --checkpoint /mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_kin_soma_motionlib_a0_no_skeleton_encoder_runs/lr177_a0_no_skel_formal_20260601T0353Z/uniform/checkpoints/latest.pt \
  --output-dir /mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/lr177_visual_smoke_somamesh_bluefloor_no_skel_uniform \
  --sample-limit 1 \
  --sample-index 0 \
  --duration-sec 4 \
  --width 1280 \
  --height 720 \
  --device cpu \
  --render \
  --isaac-python /workspace/isaaclab/_isaac_sim/python.sh \
  --soma-python /home/user/project/ContextRetarget/third_party/soma-retargeter/.venv/bin/python \
  --soma-retargeter-root /home/user/project/ContextRetarget/third_party/soma-retargeter \
  --somamesh-usd /home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd \
  --robot-usd /mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd \
  --preserve-world-root \
  --draw-orientation-labels \
  --ground-size 28 \
  --ground-color 0.08 0.20 0.72 \
  --camera-mode follow \
  --camera-offset 3.4 -4.4 2.2 \
  --camera-follow-smoothing 4 \
  --camera-framing-margin 1.35
```
