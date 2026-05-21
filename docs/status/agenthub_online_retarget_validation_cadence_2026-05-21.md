# OnlineRetarget 20k Video Validation Status

## Conclusion

当前正式训练没有在每次 evaluation 时渲染视频。`SonicVisualValidationCallback`
挂在 training loop 的 `on_step_end`，但只在 `global_step` 为 `20000` 的正整数倍时触发。

截至 `2026-05-21T01:44:31Z`，四个正式 run 仍在约 `2k` iteration，尚未到第一次
`20k` video validation gate，所以 `validation_file_count=0` 是预期状态。

## Evidence

| Item | Evidence |
| --- | --- |
| 触发条件 | `src/online_retarget/sonic_validation_callback.py` 中 `should_run_visual_validation()` 要求 `global_step > 0` 且 `global_step % every_steps == 0` |
| 正式配置 | 四个正式 config 均设置 `every_steps=20000`、`num_videos=8`、`duration_sec=4.0`、`target_fps=50`、`wandb_upload=true` |
| 当前进度 | `A1=2067`、`A2=2038`、`B1=2050`、`B2=2054` |
| 20k artifact | `validation_file_count=0`，`validation_20k_ready.md` 尚未生成 |
| 训练状态 | 四个 tmux training session、远程 `20k_watcher`、本地 AgentHub upload sidecar 均已确认存活 |

## Current Run Group

- Run group: `sonic_native_retarget_1m_20260520T220222Z`
- W&B project: `world_model_xh/OnlineRetarget`
- OnlineRetarget synced SHA: `ca0872649f7eef6b81c888cc27af8bd7b9b00b52`

## Next Gate

下一项真实验收证据不是当前 checkpoint，而是自动生成的
`step_00020000` video validation bundle：

- 四个 run 合计至少 `32` 个 MP4。
- 每个 run 有 `main_upload_report.json`。
- W&B upload status 全部为 `ok`。
- W&B videos uploaded 总数至少 `32`。
- 每个视频包含 source capsule、dataset G1 target、inferred G1 三路同步结果。

该 gate 尚未达成，因此 `goal.md` 不能标记完成。
