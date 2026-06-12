# LR-313 Ablation Config Examples

The supervised soma-motionlib trainer preserves legacy behavior when
`training.loss_mode` is omitted. Existing `*_loss_enabled` and `*_loss_weight`
fields continue to work unchanged.

Use `training.loss_mode` only when the ablation should be explicit:

```json
{
  "training": {
    "loss_mode": "none",
    "temporal_consistency_loss_enabled": false,
    "temporal_consistency_loss_weight": 0.0,
    "ab_overlap_loss_enabled": false,
    "ab_overlap_loss_weight": 0.0
  }
}
```

```json
{
  "training": {
    "loss_mode": "a_only",
    "temporal_consistency_loss_weight": 0.01,
    "ab_overlap_loss_weight": 0.0
  }
}
```

```json
{
  "training": {
    "loss_mode": "a_plus_b",
    "temporal_consistency_loss_weight": 0.01,
    "ab_overlap_loss_weight": 0.01
  }
}
```

The first LR-313 input candidate is previous G1 base-up, derived from the
existing `root_rot` motionlib field. It is opt-in and adds 3 motion-token dims:
`motion_token` 871 to 874, `model_input` 975 to 978.

```json
{
  "features": {
    "previous_g1_action_condition": true,
    "previous_g1_root_roll_pitch_condition": true,
    "previous_g1_base_up_condition": true,
    "expected_dims": {
      "motion_token": 874,
      "x_skel": 104,
      "z_skel": 104,
      "model_input": 978,
      "target": 670
    }
  }
}
```

Launch handoff commands:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
CONFIG=configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_base_up_4gpu.json scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh
```
