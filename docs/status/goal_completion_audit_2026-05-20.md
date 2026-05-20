# Goal Completion Audit

Date: 2026-05-21

Objective: implement `goal.md`, meaning a SONIC-native retargeting lane where
heterogeneous human/SOMA/BVH source encoders drive SONIC's G1 decoder path,
with `g1_dyn` as the primary target and `g1_kin` only auxiliary.

## Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Formal lane is not the standalone G1 reconstruction trainer | `configs/sonic_kin_skeleton_*` are marked `legacy_kin_diagnostic`; formal configs use `training_lane: sonic_native_retarget` | Partial |
| Formal configs target SONIC native decoder path | `configs/sonic_native_retarget_*` require `target_decoder.primary: g1_dyn` and `decoder_targets: ["g1_dyn", "g1_kin"]` | Covered at config-contract level |
| Source inputs are human/SOMA/BVH and skeleton features | Formal configs list SOMA joints/root orientation; `sonic_morphology.py` maps registry rows into actor/skeleton/morphology source keys | Covered in OnlineRetarget feature layer |
| `body_pos_w/body_quat_w` cannot enter source encoder input | `sonic_native_contract.py`; `test_sonic_native_contract.py` rejects those fields in source context | Covered |
| Same feature contract for train/validation/inference | `sonic_native_features.py`; `test_sonic_native_features.py` verifies training and inference pack the same source payload and digest | Covered for OnlineRetarget packer |
| Four variants A1/A2/B1/B2 exist | Four `configs/sonic_native_retarget_*_1gpu.json` configs exist and validate | Configs covered |
| Four variants are actually implemented in SONIC model code | `src/online_retarget/sonic_encoder_modules.py` provides Hydra-compatible modules; formal configs wire each `module_target` through `sonic_hydra.args`; contract validation requires `sonic_hydra.variant_wired: true` | Covered at code/config level |
| B1/B2 route skeletons through adapter/expert branches explicitly | B1/B2 modules use deterministic routes from the normalized skeleton-cluster scalar in `soma_morphology`; validation reports include per-clip encoder route sequences and counts | Covered at code/config level |
| Dynamics decoder imitation, latent alignment, kinematic aux, smoothness losses exist | `src/online_retarget/sonic_losses.py` implements `G1DynamicsActionLoss` and `ActionSmoothnessLoss`; configs still rely on Sonic's existing `g1_soma_latent` / `g1_recon` losses for latent/kin aux terms | Partial |
| Sonic 50 Hz timing is enforced | Config validator requires `frequency.target_fps: 50`; formal configs set `motion_lib_cfg.target_fps=50`; visual callback renders `duration_sec * target_fps` frames | Covered at config/callback level |
| Integrated validation runs inside formal Sonic training | `src/online_retarget/sonic_validation_callback.py` implements a Sonic `TrainerCallback`; all formal configs inject it under `callbacks.online_retarget_visual_val`; contract validation now requires this callback wiring | Covered at code/config level |
| W&B video upload inside formal training | `SonicVisualValidationCallback` collects rank-local videos into the shared step directory and main rank logs them with `wandb.Video`; formal configs require `wandb_upload=true` | Covered at code/config level |
| Remote launch checks committed/latest git | `remote_start_sonic_native_retarget_4x1gpu.sh` validates formal configs and requires committed/latest git before execution | Covered |
| A1/A2/B1/B2 can launch under assigned GPU budget | Launcher allocates one GPU per formal config and validates `sonic_hydra.variant_wired=true`; dry-run passed after commit `cd7bdb71473434f8ea0197a6b34b07cf038c03ae`; execution still requires the remote Sonic/Isaac environment and four visible GPUs | Partial until remote launch |
| Formal Sonic runs use 1M trainer iterations | Formal configs and validator require `algo.config.num_learning_iterations=1000000`, which is the Sonic trainer loop field; sidecar `training.max_steps` alone is not trusted | Covered at config-contract level |
| 1M-step comparison report selects best variant | No 1M-step runs or comparison report exist | Missing |

## Verification Run

```bash
python3 -m py_compile \
  src/online_retarget/sonic_native_contract.py \
  src/online_retarget/sonic_native_features.py \
  src/online_retarget/sonic_encoder_modules.py \
  src/online_retarget/sonic_morphology.py \
  src/online_retarget/sonic_observation_terms.py \
  src/online_retarget/sonic_losses.py \
  src/online_retarget/sonic_validation_callback.py \
  scripts/validate_sonic_native_retarget_config.py

PYTHONPATH=src python3 -m unittest \
  tests.test_sonic_native_contract \
  tests.test_sonic_native_features \
  tests.test_sonic_morphology \
  tests.test_sonic_validation_callback

python3 scripts/validate_sonic_native_retarget_config.py --require-formal \
  configs/sonic_native_retarget_a1_concat_1gpu.json \
  configs/sonic_native_retarget_a2_film_contact_1gpu.json \
  configs/sonic_native_retarget_b1_adapter_1gpu.json \
  configs/sonic_native_retarget_b2_expert_1gpu.json

bash -n scripts/remote_start_sonic_native_retarget_4x1gpu.sh
```

Observed result on 2026-05-21: all commands passed; the focused unittest run
executed 21 tests, including the callback scheduling/rank-splitting checks and
the formal-config callback wiring rejection check.

Additional verification after deterministic B-route hardening:

```bash
python3 -m py_compile \
  src/online_retarget/sonic_native_contract.py \
  src/online_retarget/sonic_native_features.py \
  src/online_retarget/sonic_encoder_modules.py \
  src/online_retarget/sonic_morphology.py \
  src/online_retarget/sonic_observation_terms.py \
  src/online_retarget/sonic_losses.py \
  src/online_retarget/sonic_validation_callback.py \
  scripts/validate_sonic_native_retarget_config.py

PYTHONPATH=src python3 -m unittest \
  tests.test_sonic_native_contract \
  tests.test_sonic_native_features \
  tests.test_sonic_morphology \
  tests.test_sonic_encoder_modules \
  tests.test_sonic_validation_callback

PYTHONPATH=src python3 scripts/validate_sonic_native_retarget_config.py --require-formal \
  configs/sonic_native_retarget_a1_concat_1gpu.json \
  configs/sonic_native_retarget_a2_film_contact_1gpu.json \
  configs/sonic_native_retarget_b1_adapter_1gpu.json \
  configs/sonic_native_retarget_b2_expert_1gpu.json

bash -n scripts/remote_start_sonic_native_retarget_4x1gpu.sh

PYTHONPATH=src bash scripts/remote_start_sonic_native_retarget_4x1gpu.sh
```

Observed result: compile passed; focused unittest ran 28 tests with 4
torch-dependent encoder tensor tests skipped because local `torch` is not
installed; formal config validator passed; launcher dry-run passed and wrote:

`outputs/sonic_native_retarget_runs/sonic_native_retarget_20260520T165510Z/_launcher/launch_manifest.json`

The dry-run manifest records:

- `online_retarget_commit`: `cd7bdb71473434f8ea0197a6b34b07cf038c03ae`
- `executed`: `false`
- configs: A1/A2/B1/B2 formal configs
- GPU assignments: `0 1 2 3`

This dry-run also verified `HEAD == origin/main` before writing the manifest.

Local execution blocker on this machine:

- only one GPU is visible through `nvidia-smi`
- `torch` is unavailable in the local Python environment
- formal configs point to remote Sonic root
  `/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training`, which is
  not mounted in this container

## Conclusion

The goal is still not complete. The repo now has a correct formal contract,
four variant configs, source-feature guardrails, a shared train/inference
feature packer, registry-backed morphology feature extraction, Hydra-compatible
encoder module classes, Sonic observation/loss injections, and an integrated
Sonic visual validation callback with W&B video upload.

The remaining decisive work is external execution evidence: run the four formal
Sonic/Isaac jobs in the remote environment, confirm that the callback renders
real videos at step 20k, train the variants to 1M steps, and write the
comparison report selecting the best variant. Commit/sync is done for the local
implementation commit listed above, but the goal is not complete until the
training and comparison evidence exists.

## Remote Execution Update: 2026-05-21

Current formal run group:

`sonic_native_retarget_1m_20260520T203518Z`

Remote evidence collected from `ssh 5090`:

| Item | Evidence | Status |
| --- | --- | --- |
| OnlineRetarget checkout | `/mnt/data_cpfs/code/wxh/OnlineRetarget` is clean on `main`, commit `ea87ad0cd6986c4493856dd1fb030cf543537797` | Covered |
| Sonic checkout | `/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training`, commit `53e5a44f6373fe70b2bc62c934fa8f98ee810062` | Covered |
| Manifest | `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T203518Z/_launcher/launch_manifest.json` records `executed: true`, four configs, four GPUs, four tmux sessions | Covered |
| A1 run | tmux `sonic_native_sonic_native_retarget_1m_20260520T203518Z_A1_concat`, W&B `zbtshia6`, running, W&B config has encoder `A1_concat`, `num_learning_iterations: 1000000`, git SHA fields | Running |
| A2 run | tmux `sonic_native_sonic_native_retarget_1m_20260520T203518Z_A2_film_contact`, W&B `c9tg1s9d`, running, W&B config has encoder `A2_film_contact`, visual callback config | Running |
| B1 run | tmux `sonic_native_sonic_native_retarget_1m_20260520T203518Z_B1_adapter`, W&B `8fj9qby9`, running, W&B config has encoder `B1_adapter`, deterministic route config | Running |
| B2 run | tmux `sonic_native_sonic_native_retarget_1m_20260520T203518Z_B2_expert`, W&B `zsgqemib`, running, W&B config has encoder `B2_expert`, deterministic route config | Running |
| Remote contract validation | `PYTHONPATH=src /workspace/isaaclab/_isaac_sim/python.sh scripts/validate_sonic_native_retarget_config.py --require-formal --check-paths ...` passed for all four configs | Covered |
| Runtime errors | Grep over four launcher logs found no `Traceback`, `RuntimeError`, `KeyError`, or `Error executing job` at the audit time | Covered so far |
| Current progress | Logs showed A1/B1/B2 around learning iteration `127`, A2 around `128`; W&B API showed lastHistoryStep around `145-148` | In progress |
| 20k-step visual validation | Runs have not reached step `20000`; no real W&B validation videos are expected yet | Missing until reached |
| 1M-step completion | Runs are far below `1000000` configured trainer iterations | Missing |

Additional Sonic-side evidence:

- `gear_sonic/train_agent_trl.py` instantiates `config.callbacks`.
- `gear_sonic/trl/callbacks/hv_callback_handler.py` calls callback `on_step_end`
  with `env`, `model`, and `accelerator`, so
  `SonicVisualValidationCallback` is not merely dead Hydra config.
- Sonic motion command / wrapper exposes the panel fields used by the callback:
  `soma_joints`, `body_pos_w`, `robot_body_pos_w`, `motion_start_time_steps`,
  and `time_steps`.

Updated conclusion:

The code and remote launch path now satisfy the formal implementation contract
at config, import, callback, git, W&B-config, and motionlib-path levels. The
goal is still not complete because the decisive runtime artifacts are not
available yet: first integrated W&B validation videos at 20k steps, full 1M-step
completion or reproducible failure reports, and the final A1/A2/B1/B2 comparison
report.

## Local Loss Guardrail Update: 2026-05-21

One implementation gap was found after the remote launch: the formal
`G1DynamicsActionLoss` accepted `loss_inputs["action_mean"]` as a fallback when
`decoded_outputs["g1_dyn"]` was absent. That fallback was too weak for this
Goal, because it did not force the supervised objective to prove that Sonic's
Dynamics Decoder actually ran.

The loss now fails fast unless `decoded_outputs` contains the requested
`g1_dyn` decoder and that decoder emits `action`, `body_action`, or
`meta_action`. `tests/test_sonic_losses.py` covers the accepted output keys,
temporal target alignment, missing-decoder failure, missing-action failure, and
smoothness loss on the `g1_dyn` prediction.

Local verification:

```bash
PYTHONPATH=src python3 scripts/validate_sonic_native_retarget_config.py --require-formal \
  configs/sonic_native_retarget_a1_concat_1gpu.json \
  configs/sonic_native_retarget_a2_film_contact_1gpu.json \
  configs/sonic_native_retarget_b1_adapter_1gpu.json \
  configs/sonic_native_retarget_b2_expert_1gpu.json

PYTHONPATH=src python3 -m unittest \
  tests.test_sonic_losses \
  tests.test_sonic_encoder_modules \
  tests.test_sonic_native_contract \
  tests.test_sonic_native_features \
  tests.test_sonic_validation_callback

python3 -m compileall -q src/online_retarget tests/test_sonic_losses.py
```

Observed result: formal config validation passed; compileall passed; local
unittest reported `43` tests OK with `14` skipped because this local Python does
not provide `torch`. The Torch-backed loss tests still need remote Isaac/Sonic
Python verification after the commit is pushed and the remote checkout is
synced.

## Current Formal Run Update: 2026-05-21

Current formal run group:

`sonic_native_retarget_1m_20260520T220222Z`

Remote launcher manifest:

`/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_launcher/launch_manifest.json`

| Item | Evidence | Status |
| --- | --- | --- |
| OnlineRetarget commit | Manifest and W&B args record `de7ff733edf5b8cd978882826229b0a7400ac0d2` | Covered for launched runs |
| Sonic commit | Manifest, W&B metadata, and W&B args record `53e5a44f6373fe70b2bc62c934fa8f98ee810062` | Covered |
| Remote checkout at launch audit | `/mnt/data_cpfs/code/wxh/OnlineRetarget` was clean; `HEAD` and upstream both resolved to `de7ff73` | Covered at audit time |
| Launcher execution | Manifest records `executed: true`, configs A1/A2/B1/B2, GPUs `0 1 2 3`, and four tmux sessions | Covered |
| A1 run | W&B local run `rcuzxotj`; args include `encoder_variant=A1_concat`, `num_learning_iterations=1000000`, `g1_dyn` loss, and visual callback settings | Running |
| A2 run | W&B local run `o1ldyppd`; args include `encoder_variant=A2_film_contact`, `num_learning_iterations=1000000`, `g1_dyn` loss, and visual callback settings | Running |
| B1 run | W&B local run `ctkd8d87`; args include `encoder_variant=B1_adapter`, deterministic adapter module, `num_learning_iterations=1000000`, and visual callback settings | Running |
| B2 run | W&B local run `2r8c0hs0`; args include `encoder_variant=B2_expert`, deterministic expert module, `num_learning_iterations=1000000`, and visual callback settings | Running |
| Sonic callback integration | Sonic `HVCallbackHandler` passes `env`, `model`, and `accelerator` to callback `on_step_end`; `PPOTrainer` increments `global_step` before calling callbacks | Covered at source-audit level |
| 1M-step semantics | Sonic config maps `num_total_batches` to `algo.config.num_learning_iterations` and comments it as total training steps | Covered |
| Health scan | Grep over the four launcher logs found no `Traceback`, `CUDA out of memory`, `RuntimeError`, import error, or validation/W&B failure markers at the audit time | Covered so far |
| Current progress sample | Logs showed A1 around learning iteration `205`, A2 around `198`, B1 around `201`, and B2 around `202` | In progress |
| 20k-step visual validation | Runs are far below global step `20000`, so no integrated validation video should exist yet | Missing until reached |
| 1M-step completion | Runs are far below `1000000` Sonic training steps | Missing |
| Final comparison report | Requires completed or reproducibly failed A1/A2/B1/B2 runs with metrics, validation videos, and latency evidence | Missing |

Latest focused local verification:

```bash
PYTHONPATH=src python3 scripts/validate_sonic_native_retarget_config.py --require-formal \
  configs/sonic_native_retarget_a1_concat_1gpu.json \
  configs/sonic_native_retarget_a2_film_contact_1gpu.json \
  configs/sonic_native_retarget_b1_adapter_1gpu.json \
  configs/sonic_native_retarget_b2_expert_1gpu.json

PYTHONPATH=src python3 -m unittest \
  tests.test_sonic_native_contract \
  tests.test_sonic_validation_callback

bash -n scripts/remote_start_sonic_native_retarget_4x1gpu.sh
```

Observed result: formal config validation passed for all four configs; focused
unittest ran `30` tests with `3` skips; launcher shell syntax check passed.

Current conclusion remains unchanged: the implementation and launch contract are
in place, and the four formal Sonic-native runs are alive, but the Goal is not
complete until the 20k-step videos, 1M-step completion or reproducible failure
evidence, and final A1/A2/B1/B2 comparison report exist.

## Visual Callback Smoke Evidence: 2026-05-21

A short A1 smoke run was found under:

`/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_vis_smoke_20260520T215549Z`

This smoke run is not a substitute for the formal 20k-step validation videos,
but it does prove that the integrated callback can execute inside Sonic
training, render mp4 files, write per-clip reports, and upload videos through
W&B.

Observed artifacts:

| Evidence | Result |
| --- | --- |
| Per-clip reports | `6` reports across steps `1`, `2`, and `3` |
| Videos | `6` mp4 files under `online_retarget_visual_validation/step_*` |
| Upload reports | `3` `main_upload_report.json` files |
| W&B upload status | Every upload report records `online_retarget_visual_validation/wandb_upload_status: ok` and `videos_uploaded: 2` |
| Time alignment metadata | Clip reports record `source_fps: 50.0`, `target_fps: 50.0`, `duration_sec: 0.2`, `target_frame_count: 10`, and `physical_time_aligned: true` |
| Motion keys | Sample reports include `210531__jump_and_land_heavy_001__A001` and `210531__jump_and_land_heavy_001__A001_M` |

Limitations:

- This was a short smoke configuration (`duration_sec: 0.2`, `num_videos: 2`)
  rather than the formal 20k-step validation configuration
  (`duration_sec: 4.0`, `num_videos: 8`).
- Remote `ffprobe` is not installed, so video duration/fps were verified from
  callback reports and file existence, not container metadata.
- The formal run group
  `sonic_native_retarget_1m_20260520T220222Z` still has `0` validation files
  because it has not reached step `20000`.

## Remote Monitor Update: 2026-05-21

A no-GPU monitor was added and started on 5090:

- Script: `scripts/monitor_sonic_native_retarget_runs.sh`
- tmux session: `sonic_native_retarget_monitor_1m`
- Summary path:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_monitor/latest_status.md`
- JSONL path:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_monitor/status.jsonl`
- Interval: `1800` seconds
- Validation artifact search roots:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z`
  and
  `/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training/logs_rl/OnlineRetarget`

Initial monitor snapshot:

| Variant | Iteration | Hard error | Validation files |
| --- | ---: | --- | ---: |
| A1 | `306` | `none` | `0` |
| A2 | `298` | `none` | `0` |
| B1 | `301` | `none` | `0` |
| B2 | `302` | `none` | `0` |

This monitor is status evidence only. It does not satisfy the Goal's 20k-step
video, 1M-step, or final comparison-report acceptance criteria.

Monitor correction:

The first monitor version only searched the OnlineRetarget launcher run root for
`online_retarget_visual_validation` files. Formal Sonic configs set callback
`output_dir` to `${experiment_dir}/online_retarget_visual_validation`, which is
expected to resolve under Sonic's `logs_rl/OnlineRetarget` tree. Commit
`0aa5482` extended the monitor to search both locations while filtering by
`run_group`, so formal 20k validation artifacts should not be missed.

## 20k Validation Watcher Update: 2026-05-21

A no-GPU watcher was added and started on 5090:

- Script: `scripts/watch_sonic_native_retarget_20k_validation.sh`
- tmux session: `sonic_native_retarget_20k_watcher`
- Ready report path:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z/_monitor/validation_20k_ready.md`
- Interval: `1800` seconds
- Search roots:
  `/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/sonic_native_retarget_runs/sonic_native_retarget_1m_20260520T220222Z`
  and
  `/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training/logs_rl/OnlineRetarget`

The watcher exits after it sees any formal
`online_retarget_visual_validation` file and writes a compact ready report with
mp4, clip/rank report, and upload report counts plus sample paths.

Important limitation:

`validation_20k_ready.md` is a trigger for follow-up audit only. It does not
prove the Goal is complete. The follow-up audit still must verify the formal
20k step, expected 8 clips per run, W&B video upload, time alignment metadata,
and absence of render/upload errors.
