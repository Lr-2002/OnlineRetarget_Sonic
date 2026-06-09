# LR-291 Repo Cleanup Note

Date: 2026-06-09

## Conclusion

The repo docs now distinguish the current strict supervised
`soma_motionlib_kin_only` launch surface from older SONIC-native shared-token
contract artifacts.

## Current Launch Surface

Use `scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh` with one explicit
config per 4-GPU job.

Active package-focused configs:

- `configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json`
- `configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json`

Broader proportional treatment/baseline configs:

- `configs/sonic_kin_soma_motionlib_proportional_4gpu.json`
- `configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json`

The compatibility wrapper
`scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh` still exists for old
instructions, but it forwards into the strict supervised SOMA motionlib
launcher.

## Guardrails

- Current formal configs keep `training_lane=soma_motionlib_kin_only`.
- Current active targets are `g1_kin` command/root-pose fields only.
- Current launcher rejects PPO, Isaac rollout, reward, episode-length,
  SONIC-Hydra, `KinematicActionUniversalTokenModule`, and `g1_dyn` training
  tokens outside descriptive fields.
- LR-280 data-package configs pin the paired SOMA motionlib kin/walk package
  indicator, expected row count, and digest.

## Weekly Walk-Data Outcome Target

online retargeter walk should be non-jittery, non-drifting, avoid self-collision, and be at least similar to the source walk data.

## Verification Targets

- `PYTHONPATH=src:. python3 -m unittest tests.test_remote_launcher_guardrails tests.test_data_package_indicator -q`
- `python3 -m compileall -q src scripts`
- `git diff --check`
