# OnlineRetarget Sonic-native Run Status

更新时间：2026-05-21 00:33 UTC

## 结论

`goal.md` 还没有完成。四个正式 Sonic-native retarget 训练已经启动并持续运行，
但还没到第一个 `20k` step visual validation，也没有 1M-step 结果或最终对比结论。

当前状态适合继续等待训练，同时保留 monitor / watcher 自动检查。

## 正式 Run

Run group: `sonic_native_retarget_1m_20260520T220222Z`

| Variant | W&B run | Latest iteration | State | Hard error |
| --- | --- | ---: | --- | --- |
| A1_concat | `rcuzxotj` | `1408` | running | none |
| A2_film_contact | `o1ldyppd` | `1384` | running | none |
| B1_adapter | `ctkd8d87` | `1395` | running | none |
| B2_expert | `2r8c0hs0` | `1400` | running | none |

## Traceability

- OnlineRetarget launch commit: `de7ff733edf5b8cd978882826229b0a7400ac0d2`
- Sonic launch commit: `53e5a44f6373fe70b2bc62c934fa8f98ee810062`
- Current repo / GitHub / remote checkout: `50c81516df3867ab674df630a7368bffdcab4c11`
- W&B nested config key: `online_retarget`
- Formal decoder path: `g1_dyn` primary, `g1_kin` auxiliary
- Formal callback: `SonicVisualValidationCallback`
- Launch-to-current diff: status docs plus monitor/watch scripts only; no runtime
  training code/config/callback/loss/encoder changes after launch.

## Runtime Health

- Local and remote worktrees: clean on `main`, tracking `origin/main`.
- Current resource use: A1/A2/B1/B2 are actively computing on GPUs `0/1/2/3`.
- Contract regression: `30` tests passed, `3` skipped via
  `PYTHONPATH=src python3 -m unittest tests.test_sonic_native_contract tests.test_sonic_validation_callback -q`.
- W&B history is still updating for all four runs. Latest API check saw
  `lastHistoryStep` around `1464-1488`, with metric keys including
  `loss/aux_online_retarget_g1_dyn_action_avg`,
  `loss/aux_online_retarget_action_smoothness_avg`, and
  `loss/aux_g1_recon_avg`.

## Validation Gate

Video validation is integrated in training but gated at `every_steps=20000`.

Current validation state:

- `validation_file_count`: `0`
- `validation_20k_ready.md`: not created yet
- Expected first validation: around 1 day 8-9 hours from the latest monitor snapshot
- Watcher requirement: wait for `4` W&B upload reports, `32` MP4 files,
  `4` successful upload statuses, zero failed/skipped/other uploads, and at
  least `32` uploaded W&B videos

## Remaining Completion Gates

- Formal 20k validation videos exist for all four variants.
- W&B upload reports show videos uploaded successfully.
- Each variant reaches 1M Sonic training steps, or has a reproducible failure report with W&B run and git SHA.
- Final A1/A2/B1/B2 comparison report selects a next-line variant using dynamics loss, kinematic auxiliary metrics, visual validation, and latency.

## Evidence Files

- `docs/status/goal_completion_audit_2026-05-20.md`
- `docs/status/sonic_native_retarget_comparison_report.md`
- Remote monitor:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_monitor/latest_status.md`
