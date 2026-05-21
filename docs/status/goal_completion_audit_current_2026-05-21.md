# OnlineRetarget Goal Completion Audit - Current Snapshot

## Objective

`goal.md` 的目标是用 Sonic 原生训练路径，固定 Sonic 面向 Unitree G1 的
`g1_dyn` Dynamics Decoder 作为主目标，训练并比较 A1/A2/B1/B2 四种
source-side Encoder variant，用于 online human/SOMA/BVH to G1 retargeting。

本文件是当前状态审计，不是完成报告。

## Current Verdict

**Not complete.**

代码级 contract、四个正式 run、W&B metrics、monitor/watch 自动化都已启动并有证据；
但 `goal.md` 的完成 gate 还缺真实 `step_00020000` video validation、W&B video upload
证据、每个 variant 的 `1M` 结果或可复现失败证据，以及最终 A1/A2/B1/B2 对比结论。

## Evidence Snapshot

| Item | Current evidence |
| --- | --- |
| Synced checkout SHA | local and 5090 checkout clean at `fe5bcbf43ed76b7769928671b1944c21ea47fff5` |
| Active training launch SHA | process env `ONLINE_RETARGET_GIT_SHA=de7ff733edf5b8cd978882826229b0a7400ac0d2` |
| Sonic SHA | `53e5a44f6373fe70b2bc62c934fa8f98ee810062` |
| Run group | `sonic_native_retarget_1m_20260520T220222Z` |
| W&B project | `world_model_xh/OnlineRetarget` |
| Latest local focused regression | `37` tests passed, `3` skipped |
| Remote formal config path check | 5090 Isaac Python validation passed with `--require-formal --check-paths`; all warnings empty |
| AgentHub cadence report | `http://10.1.11.30:5175/runs/online-retarget/20260521-094531-onlineretarget-20k-video-validation-cadence` |

## Prompt-to-Artifact Checklist

| Requirement from `goal.md` | Evidence inspected | Status |
| --- | --- | --- |
| Use Sonic-native training path | Four running processes invoke `gear_sonic/train_agent_trl.py` with Sonic Hydra overrides | Covered for active runs |
| Main target is Sonic `g1_dyn` Dynamics Decoder | Run commands include `active_decoders=[g1_dyn,g1_kin]` and `online_retarget_g1_dyn_action` loss | Covered at config/run-command level |
| Do not use standalone AutoEncoder as formal target | Active runs use Sonic train entrypoint and Sonic backbone overrides | Covered for current formal runs |
| A1 Concat variant | Active tmux/process for `A1_concat`; W&B run `rcuzxotj` state `running` | Running |
| A2 FiLM/Contact variant | Active tmux/process for `A2_film_contact`; W&B run `o1ldyppd` state `running` | Running |
| B1 Adapter variant | Active tmux/process for `B1_adapter`; W&B run `ctkd8d87` state `running` | Running |
| B2 Expert variant | Active tmux/process for `B2_expert`; W&B run `2r8c0hs0` state `running` | Running |
| Source inputs exclude target-only `body_pos_w` / `body_quat_w` | Focused regression `tests.test_sonic_native_contract` passed as part of 37-test suite | Covered by test/contract level |
| Training/validation/inference share feature contract | Focused contract/callback regression passed; formal configs wire common callback and feature terms | Covered at code/config level |
| Sonic target timeline is 50Hz | Formal configs set `motion_lib_cfg.target_fps=50`; callback uses `target_fps=50`; remote `--check-paths` validator passed | Covered at config/path level |
| Video validation integrated into training | `SonicVisualValidationCallback` is configured in all formal Hydra args | Covered at config/code level |
| Video validation every 20k steps | Callback `should_run_visual_validation()` gates on positive multiples of `every_steps`; configs set `every_steps=20000` | Covered at code/config level |
| 8 videos per validation, 4s inference window | Formal configs set `num_videos=8`, `duration_sec=4.0` | Covered at config level |
| W&B video upload path exists | Callback uses `wandb.Video`; formal configs set `wandb_upload=true` | Covered at code/config level; real 20k upload missing |
| Remote launcher checks committed/pushed/synced code | Focused launcher guardrail tests passed | Covered at test level |
| Long training in tmux | 5090 has four variant tmux sessions plus monitor/watch sessions | Covered |
| `/home/user/data/motion_data` remains read-only | No writes observed in current commands; formal output roots under `outputs/`; remote path validator passed against configured output paths | Covered for current configs; ongoing runtime audit only |
| W&B config/metrics visible | W&B API shows all four runs `running` with loss metrics present | Covered for live metric stream |
| First 20k video bundle exists | `validation_file_count=0`; no `validation_20k_ready.md` | Missing |
| W&B 20k videos uploaded | no `online_retarget_visual_validation/videos_uploaded` yet | Missing |
| Each variant reaches 1M or reproducibly fails | current steps are around `2k`, far below `1M` | Missing |
| Final A1/A2/B1/B2 comparison selects next-line variant | comparison report scaffold exists, but no final metrics/videos/latency evidence | Missing |

## Latest Run Health

Remote monitor at `2026-05-21T01:46:20Z`:

| Variant | Monitor iteration | Hard error | Latest checkpoint |
| --- | ---: | --- | --- |
| A1 | `2084` | `none` | `model_step_002000.pt` |
| A2 | `2054` | `none` | `model_step_002000.pt` |
| B1 | `2066` | `none` | `model_step_002000.pt` |
| B2 | `2071` | `none` | `model_step_002000.pt` |

W&B API check around `2026-05-21T01:48Z`:

| Variant | W&B run | State | W&B step | `loss/aux_online_retarget_g1_dyn_action_avg` |
| --- | --- | --- | ---: | ---: |
| A1 | `rcuzxotj` | `running` | `2109` | `0.3988` |
| A2 | `o1ldyppd` | `running` | `2080` | `0.3621` |
| B1 | `ctkd8d87` | `running` | `2091` | `0.3870` |
| B2 | `2r8c0hs0` | `running` | `2096` | `0.3878` |

Remote config path validation on 5090 using the actual training Python:

```bash
PYTHONPATH=src /workspace/isaaclab/_isaac_sim/python.sh \
  scripts/validate_sonic_native_retarget_config.py \
  --require-formal --check-paths --json \
  configs/sonic_native_retarget_a1_concat_1gpu.json \
  configs/sonic_native_retarget_a2_film_contact_1gpu.json \
  configs/sonic_native_retarget_b1_adapter_1gpu.json \
  configs/sonic_native_retarget_b2_expert_1gpu.json
```

Result: all four configs returned `formal=true`, `target_decoder=g1_dyn`,
`training_lane=sonic_native_retarget`, and `warnings=[]`.

## Active Automation

| Automation | Evidence | Status |
| --- | --- | --- |
| Remote 20k watcher | tmux `sonic_native_retarget_20k_watcher`; `pgrep` shows `watch_sonic_native_retarget_20k_validation.sh` running | Alive |
| Remote 1M monitor | tmux `sonic_native_retarget_monitor_1m` | Alive |
| Local AgentHub 20k upload sidecar | tmux `sonic_native_retarget_20k_agenthub_upload`; latest pane says not ready and sleeping | Alive |

## Stop Condition

Do not mark the goal complete until the audit can cite:

1. Real `step_00020000` validation videos for all four variants.
2. W&B upload reports with no failed/skipped/other upload status.
3. 1M-step completion or reproducible failure evidence for each variant.
4. A final comparison using dynamics loss, kinematic auxiliary metrics, visual validation, and latency.
