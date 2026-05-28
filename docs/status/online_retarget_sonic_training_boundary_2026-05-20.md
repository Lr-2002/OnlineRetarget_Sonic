# OnlineRetarget Sonic Training Boundary

Date: 2026-05-20. Updated for LR-185 on 2026-05-28.

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

Rules for the next kinematics-only runs:

- Use `scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh` from OnlineRetarget.
- Launch exactly one config per run:
  - `configs/sonic_kin_only_soma_encoder_uniform.json`
  - `configs/sonic_kin_only_soma_encoder_proportional.json`
- Each config is one 4-GPU job. Do not split the current requirement into
  A1/A2/B1/B2 one-GPU sessions.
- Commit and push OnlineRetarget before launching.
- The remote OnlineRetarget checkout must be clean and at its latest upstream
  commit. The launcher fetches its tracking branch and refuses to start if
  `HEAD` does not match upstream.
- The configured Sonic source checkout must be clean and its exact commit must
  be recorded in the manifest and W&B summary. If that checkout later gets a
  configured upstream, add the same latest-upstream guard there too.
- Keep output under `outputs/` in the OnlineRetarget remote checkout.
- Log W&B runs to project `OnlineRetarget`.
- Record both the OnlineRetarget commit and Sonic source commit in each run
  manifest and W&B summary.
- Stop when both baselines reach 1M steps with kin loss/MPJPE/readable visual
  artifacts and sliding/jitter review notes, or when each has a reproducible
  failure report with run group, W&B run, config path, and git SHAs.
