# OnlineRetarget LR-147 Code Update Audit

Date: 2026-05-25

LR-177 correction note (2026-05-29): the active execution surface is now the
strict supervised SOMA motionlib lane:
`configs/sonic_kin_soma_motionlib_uniform_4gpu.json` and
`configs/sonic_kin_soma_motionlib_proportional_4gpu.json`. The run names remain
`sonic_kin_only_soma_encoder_uniform` and
`sonic_kin_only_soma_encoder_proportional`; both are reconstruction-only 4-GPU
baselines. The A1/A2/B1/B2 entries below are historical LR-147 evidence, not
the current run matrix.

## Conclusion

LR-147 requested a code-level check and update after confirming the SONIC-based
code path is equivalent to Yuhan/SONIC for the baseline lane. The repo already
had the main formal OnlineRetarget pieces: W&B traceability, integrated
source/dataset/inference visual validation, and four source-encoder variants.
This pass tightened two missing pieces:

- visual validation can now run on a wall-clock cadence, so long runs can emit
  validation videos every 30-60 minutes instead of waiting only for a large step
  multiple;
- kin-only diagnostic training can now supervise G1 `root_pos_w` plus
  `root_rot_w` alongside joint position/velocity targets.

## Feature Matrix

| LR-147 item | Status | Code/config evidence |
| --- | --- | --- |
| W&B traceability | Present | formal and kin configs have `wandb.enabled=true`; trainer logs config, commit, metrics, checkpoints |
| Continuous source/dataset/inference visual validation | Updated | `SonicVisualValidationCallback` supports `every_minutes` / `every_seconds`; formal configs request hourly validation while retaining the 20k gate |
| Four encoder variants | Present | A1 concat, A2 FiLM/contact, B1 adapter, B2 expert configs and modules |
| Historical SONIC-native retarget route | Superseded | active LR-177 configs use the strict supervised SOMA motionlib lane |
| Kin-only root pose supervision | Updated | kin trainer supports `include_root_pos_target=true`, target layout `command + root_pos_w_mf + root_rot_w_mf` |
| Validation guardrails | Updated | formal config contract now requires the hourly callback wiring in addition to the 20k step gate |

## Training Notes

The 20k validation watcher remains valid: the step-gated `step_00020000`
artifacts are still required for the formal 20k readiness gate. Wall-clock
validation creates additional intermediate step directories for faster visual
feedback and W&B video review.

For kin-only configs, old target layouts remain readable when
`include_root_pos_target` is absent or false. The updated baseline configs opt in
explicitly, which changes the target dimension from:

- legacy: `window * (29 joint_pos + 29 joint_vel) + window * 6 root_rot`
- updated: `window * (29 joint_pos + 29 joint_vel) + window * 3 root_pos + window * 6 root_rot`

Existing checkpoints trained with the legacy target layout should not be resumed
with the updated configs unless the normalization stats and model head dimension
match.

## Verification

- `PYTHONPATH=src python3 -m unittest tests.test_sonic_validation_callback -q`
- `PYTHONPATH=src python3 -m unittest tests.test_sonic_native_20k_watcher -q`
- `conda run -n torch env PYTHONPATH=src python -m unittest tests.test_sonic_kin_train -q`
- `conda run -n base_dev env PYTHONPATH=src python -m unittest tests.test_sonic_native_contract -q`
