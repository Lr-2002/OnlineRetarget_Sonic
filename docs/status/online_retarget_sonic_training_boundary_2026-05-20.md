# OnlineRetarget Sonic Training Boundary

Date: 2026-05-20

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

- Use `scripts/remote_start_sonic_kin_skeleton_4x1gpu.sh` from OnlineRetarget.
- Commit and push OnlineRetarget before launching.
- Keep output under `outputs/` in the OnlineRetarget remote checkout.
- Log W&B runs to project `OnlineRetarget`.
- Record both the OnlineRetarget commit and Sonic source commit in each run
  manifest and W&B summary.
