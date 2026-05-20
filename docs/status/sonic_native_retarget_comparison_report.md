# Sonic-Native Retarget Comparison Report

Status: pending formal run completion

Run group:

`sonic_native_retarget_1m_20260520T220222Z`

This report is the final decision surface for `goal.md`. Fill it only with
evidence from the formal A1/A2/B1/B2 runs, not from smoke runs or dry runs.

## Run Inventory

| Variant | W&B run | Encoder | GPU | Status | Final step | Checkpoint |
| --- | --- | --- | ---: | --- | ---: | --- |
| A1 | `rcuzxotj` | Concat SOMA encoder | 0 | running | pending | pending |
| A2 | `o1ldyppd` | FiLM/contact SOMA encoder | 1 | running | pending | pending |
| B1 | `ctkd8d87` | Adapter SOMA encoder | 2 | running | pending | pending |
| B2 | `2r8c0hs0` | Expert SOMA encoder | 3 | running | pending | pending |

## Traceability

| Field | Value |
| --- | --- |
| OnlineRetarget launch commit | `de7ff733edf5b8cd978882826229b0a7400ac0d2` |
| Sonic launch commit | `53e5a44f6373fe70b2bc62c934fa8f98ee810062` |
| Monitor ETA script commit | `6d5c468` |
| Report scaffold first commit | `8808244` |
| Launcher manifest | `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_launcher/launch_manifest.json` |
| Monitor summary | `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_monitor/latest_status.md` |

## Required Completion Evidence

| Gate | Required evidence | Status |
| --- | --- | --- |
| 20k visual validation | 8 synchronized 4s videos per formal run at step 20000, uploaded to W&B | pending |
| 1M training outcome | Each variant reaches 1M Sonic training steps, or has a reproducible failure log with W&B run and git SHA | pending |
| Dynamics objective | W&B has `online_retarget_g1_dyn_action` metrics for all variants | pending |
| Kinematics auxiliary | W&B has G1 kinematic auxiliary metrics for all variants | pending |
| Latency | Batch size 1 inference latency measured for all viable variants | pending |
| Checkpoints | Final or best checkpoint path recorded for each variant | pending |
| Decision | One recommended next-line variant with rationale and risks | pending |

## Metrics Table

Use the same step window for all variants when comparing unfinished runs.

| Variant | Step | g1_dyn action loss | action cosine | G1 joint RMSE | FK MPJPE | smoothness | foot/contact artifact | latency ms | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| A2 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B1 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B2 | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Visual Validation Review

| Variant | Step | Video count | Source/target/inference time-aligned | Target speed plausible | Inference follows target | Main artifact |
| --- | ---: | ---: | --- | --- | --- | --- |
| A1 | pending | pending | pending | pending | pending | pending |
| A2 | pending | pending | pending | pending | pending | pending |
| B1 | pending | pending | pending | pending | pending | pending |
| B2 | pending | pending | pending | pending | pending | pending |

## Decision

Pending.

Decision rule:

1. Reject any variant that fails the source-feature contract or cannot produce
   synchronized validation videos.
2. Among viable variants, prioritize lower `g1_dyn` action loss, then lower G1
   joint RMSE/FK MPJPE, then lower visual artifacts.
3. If metrics are close, prefer the simpler variant unless B1/B2 shows a clear
   skeleton-family benefit.
4. Do not choose a variant whose batch size 1 latency violates the deployment
   budget without a concrete simplification path.

## Open Risks

- Formal runs are still far below 20k steps, so no formal validation video has
  been generated yet.
- Smoke-run videos prove callback mechanics only; they do not prove formal
  4-second validation quality.
- Current ETA to 1M steps is measured in weeks, so the final decision may need
  an intermediate checkpoint comparison before full completion if training cost
  becomes unacceptable.
