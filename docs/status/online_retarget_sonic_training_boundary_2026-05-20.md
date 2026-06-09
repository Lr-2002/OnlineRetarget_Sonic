# OnlineRetarget Sonic Training Boundary

Date: 2026-05-20. Updated for LR-185 on 2026-05-28 and LR-291 on
2026-06-09.

OnlineRetarget is the owning project for the current Sonic-based retargeting
experiments. Sonic is the upstream code/data reference, but training runs for
this work should be launched from the OnlineRetarget repository and logged under
the W&B project `OnlineRetarget`.

Current remote training root:

```text
/mnt/data_cpfs/code/wxh/OnlineRetarget
```

Current Sonic source root:

```text
/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training
```

Rules for current kinematics-only runs:

- Use `scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh` from
  OnlineRetarget. The older
  `scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh` name is now a
  compatibility wrapper into this strict supervised launcher.
- Launch exactly one `training_lane=soma_motionlib_kin_only` config per run.
- Current LR-280 kin/walk data-package configs:
  - `configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json`
  - `configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json`
- Current broader proportional treatment/baseline configs:
  - `configs/sonic_kin_soma_motionlib_proportional_4gpu.json`
  - `configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json`
- Supporting uniform treatment config:
  - `configs/sonic_kin_soma_motionlib_uniform_4gpu.json`
- Each config is one 4-GPU job. Do not split the current requirement into
  A1/A2/B1/B2 one-GPU sessions unless a task explicitly asks for historical
  archaeology and unlocks the historical launcher guard.
- The old SONIC-native shared-token configs
  `configs/sonic_kin_only_soma_encoder_uniform.json` and
  `configs/sonic_kin_only_soma_encoder_proportional.json` remain contract
  artifacts, not the default current launch surface.
- Commit and push OnlineRetarget before launching.
- The remote OnlineRetarget checkout must be clean and at its latest upstream
  commit. The launcher fetches its tracking branch and refuses to start if
  `HEAD` does not match upstream.
- The strict supervised launcher rejects configs that contain PPO, Isaac,
  reward, episode-length, SONIC Hydra, `KinematicActionUniversalTokenModule`, or
  `g1_dyn` training tokens outside descriptive fields.
- LR-280 data-package configs must validate the paired SOMA motionlib kin/walk
  package indicator, expected row count, and package-row digest before training.
- The configured Sonic source checkout is referenced for data/model provenance
  when present, but the strict supervised entrypoint does not execute external
  Sonic code directly.
- Keep output under `outputs/` in the OnlineRetarget remote checkout.
- Log W&B runs to project `OnlineRetarget`.
- Record the OnlineRetarget commit, config, run group, data package summary when
  applicable, and external source commit status in each launch manifest.
- Stop when the selected treatment/baseline pair has comparable kin loss, joint
  RMSE, velocity RMSE, root pose metrics, readable visual artifacts, and
  sliding/jitter review notes, or when each has a reproducible failure report
  with run group, W&B run, config path, and git SHA.
