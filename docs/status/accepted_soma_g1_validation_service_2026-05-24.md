# Accepted SOMA/G1 Validation Service Status

Date: 2026-05-24

## Result

LR-117 standardized the LR-106 accepted visualization path as a reusable delta validation entrypoint:

```bash
cd /home/user/project/OnlineRetarget
/home/user/venvs/isaaclab-210/bin/python scripts/run_accepted_soma_g1_validation.py \
  --sample-limit 1 \
  --run-name lr117_smoke_accepted_soma_g1_20260524_1630
```

The full LR-106 fixed sample set is the default; omit `--sample-limit 1` to render all 8 clips.

## Smoke Evidence

| Field | Value |
| --- | --- |
| Gate | smoke pass |
| Execution host | `lr-2002delta` |
| Output dir | `/home/user/project/OnlineRetarget/outputs/lr117_smoke_accepted_soma_g1_20260524_1630` |
| AgentHub | `http://10.1.11.30:5175/runs/online-retarget/20260524-162911-lr-117-accepted-soma-g1-validation-smoke-v2` |
| Sample | `220720__itching_neck_003__A032_M` |
| Source | SomaBVH/SomaMesh LBS |
| Target | IsaacLab G1 kinematic playback |
| Frames / fps | `200` frames reported, final ffprobe `199` video frames at `50/1` fps, duration `3.980000` |
| Final video | `00_itching_neck_003__A032_M/itching_neck_003__A032_M_somamesh_source_g1_isaac_with_axes.mp4` |
| Changed frames | source `199`, target `199`, overlay `199` |
| Final video shape | `1920x540` |
| G1 asset | `/home/user/project/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd` |

Both AgentHub URLs returned HTTP 200 during verification:

- `http://10.1.11.30:5175/runs/online-retarget/20260524-162911-lr-117-accepted-soma-g1-validation-smoke-v2`
- `http://100.76.129.28:5175/runs/online-retarget/20260524-162911-lr-117-accepted-soma-g1-validation-smoke-v2`

## Scope Boundary

This service proves standardized kinematic visualization and semantic playback evidence only. It does not prove policy tracking, dynamics, balance, torque feasibility, sim2sim, sim2real, or training convergence.
