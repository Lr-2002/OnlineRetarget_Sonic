# Sonic-Native Retarget Comparison Report

LR-185 supersession note (2026-05-28): this A1/A2/B1/B2 comparison report is
historical and must not be used as the current run matrix. The active formal
comparison is `sonic_kin_only_soma_encoder_uniform` versus
`sonic_kin_only_soma_encoder_proportional`, each as one 4-GPU kin-only job.

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
| Dynamics objective | W&B has `online_retarget_g1_dyn_action` metrics for all variants | streaming; final pending |
| Kinematics auxiliary | W&B has G1 kinematic auxiliary metrics for all variants | streaming; final pending |
| Latency | Batch size 1 inference latency measured for all viable variants | pending |
| Checkpoints | Final or best checkpoint path recorded for each variant | pending |
| Decision | One recommended next-line variant with rationale and risks | pending |

## Early Checkpoint Evidence

Latest monitor snapshot at `2026-05-21T01:40:24Z` confirmed all four formal
variants crossed the first numbered checkpoint gate at step `2000`:

| Variant | Iteration | Checkpoint count | Latest checkpoint | Meaning |
| --- | ---: | ---: | --- | --- |
| A1 | `2029` | `2` | `model_step_002000.pt` | First numbered checkpoint |
| A2 | `2000` | `2` | `model_step_002000.pt` | First numbered checkpoint |
| B1 | `2011` | `2` | `model_step_002000.pt` | First numbered checkpoint |
| B2 | `2015` | `2` | `model_step_002000.pt` | First numbered checkpoint |

These files prove regular Sonic checkpoint writing is active for all four
formal variants. They are still early progress artifacts, not final or best
checkpoints for the completion table.

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
- Current ETA to 1M Sonic trainer iterations is roughly 70+ days at the latest
  monitor rate. An intermediate checkpoint comparison may be useful for a
  research decision, but it must not be treated as `goal.md` completion unless
  the user explicitly re-scopes the 1M-step acceptance criterion.

## Latest Monitor Snapshot

Manual refresh at `2026-05-21T01:40:24Z` for run group
`sonic_native_retarget_1m_20260520T220222Z`:

| Variant | Iteration | Iter/hr | ETA 20k | ETA 1M | Validation files | Hard error |
| --- | ---: | ---: | --- | --- | ---: | --- |
| A1 | `2029` | `563.2` | `1d 7h` | `73d 19h` | `0` | `none` |
| A2 | `2000` | `556.4` | `1d 8h` | `74d 17h` | `0` | `none` |
| B1 | `2011` | `559.0` | `1d 8h` | `74d 9h` | `0` | `none` |
| B2 | `2015` | `560.0` | `1d 8h` | `74d 6h` | `0` | `none` |

No 20k validation evidence is expected yet. This report remains pending until
formal validation videos, W&B uploads, checkpoints, metrics, and latency
measurements exist.

Runtime health check at the same audit point confirmed A1/A2/B1/B2 are actively
computing on GPUs `0/1/2/3`. Local and remote OnlineRetarget checkouts are clean
at `9fea953b1a3fc7735c6b6d3e89f9ac348a10bb0a`; the diff from launch commit
`de7ff733edf5b8cd978882826229b0a7400ac0d2` to current head only updates status
docs, monitor/watch scripts, and regression tests for launcher/watcher
guardrails, not training runtime code.

## W&B Config Evidence

Remote W&B API check at the current audit point:

| Variant | Run ID | State | Last history step | Config path |
| --- | --- | --- | ---: | --- |
| A1 | `rcuzxotj` | `running` | `1090` | `/mnt/data_cpfs/code/wxh/OnlineRetarget/configs/sonic_native_retarget_a1_concat_1gpu.json` |
| A2 | `o1ldyppd` | `running` | `1070` | `/mnt/data_cpfs/code/wxh/OnlineRetarget/configs/sonic_native_retarget_a2_film_contact_1gpu.json` |
| B1 | `ctkd8d87` | `running` | `1078` | `/mnt/data_cpfs/code/wxh/OnlineRetarget/configs/sonic_native_retarget_b1_adapter_1gpu.json` |
| B2 | `2r8c0hs0` | `running` | `1082` | `/mnt/data_cpfs/code/wxh/OnlineRetarget/configs/sonic_native_retarget_b2_expert_1gpu.json` |

The traceability fields are stored under W&B config key `online_retarget`, not
as top-level W&B config keys. That nested object records:

- `contract=sonic_native_retarget`
- `run_group=sonic_native_retarget_1m_20260520T220222Z`
- OnlineRetarget launch commit
  `de7ff733edf5b8cd978882826229b0a7400ac0d2`
- Sonic launch commit `53e5a44f6373fe70b2bc62c934fa8f98ee810062`
- The per-run encoder variant and formal config path

Launcher logs also confirm all four variants initialized the Sonic `g1_dyn`
decoder and filtered active decoders to `g1_dyn` plus `g1_kin`.

## W&B Metric Stream Evidence

Remote W&B API check for project `world_model_xh/OnlineRetarget`:

| Variant | Run ID | State | Last history step | Retarget metric keys present |
| --- | --- | --- | ---: | --- |
| A1 | `rcuzxotj` | `running` | `1488` | yes |
| A2 | `o1ldyppd` | `running` | `1464` | yes |
| B1 | `ctkd8d87` | `running` | `1475` | yes |
| B2 | `2r8c0hs0` | `running` | `1481` | yes |

Observed metric keys include:

- `loss/aux_online_retarget_g1_dyn_action_avg`
- `loss/aux_online_retarget_action_smoothness_avg`
- `loss/aux_g1_recon_avg`
- `loss/total_aux_loss_avg`
- PPO loss keys

This proves the formal W&B runs are streaming the primary dynamics auxiliary
loss key. It is still not final comparison evidence because all variants are
below the 20k visual-validation gate and far below 1M steps.

Latest W&B API sample at around `2026-05-21T01:10Z`:

| Variant | Run ID | State | Last history step | g1_dyn action loss | smoothness | g1_recon | videos uploaded |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| A1 | `rcuzxotj` | `running` | `1748` | `0.4249298573` | `1.5132879019` | `0.9607078433` | pending |
| A2 | `o1ldyppd` | `running` | `1724` | `0.3671081066` | `1.5134330988` | `0.9453714490` | pending |
| B1 | `ctkd8d87` | `running` | `1736` | `0.4007171690` | `1.4459726810` | `0.9570316672` | pending |
| B2 | `2r8c0hs0` | `running` | `1739` | `0.4280258715` | `1.5081157684` | `0.8584457636` | pending |

All four W&B configs still report the formal OnlineRetarget launch commit
`de7ff733edf5b8cd978882826229b0a7400ac0d2` and Sonic commit
`53e5a44f6373fe70b2bc62c934fa8f98ee810062`. The visual-validation upload
metric is still absent, which is expected before step `20000`.
