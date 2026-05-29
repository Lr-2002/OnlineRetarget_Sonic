# SOMA Motionlib Reconstruction Baseline Contract

Date: 2026-05-20. Corrected for LR-177 on 2026-05-29.

## Decision

The active OnlineRetarget baseline surface is the strict supervised SOMA
motionlib lane, not the older SONIC-native launcher surface. There are exactly
two active 4-GPU configs:

- `configs/sonic_kin_soma_motionlib_uniform_4gpu.json`
- `configs/sonic_kin_soma_motionlib_proportional_4gpu.json`

Both keep the run names:

- `sonic_kin_only_soma_encoder_uniform`
- `sonic_kin_only_soma_encoder_proportional`

The train path is:

```text
SOMA motionlib source features + SOMA skeleton features
  -> supervised SOMA encoder MLP
  -> G1 g1_kin target fields
  -> reconstruction objective
```

## Baseline Guardrails

- `training_lane` must be `soma_motionlib_kin_only`.
- Each active config is one 4-GPU job.
- `target_decoder.primary` and `decoder_targets` must be exactly `g1_kin`.
- `losses.primary` must be exactly `["reconstruction"]`.
- `losses.auxiliary` must be empty.
- The encoder MLP hidden dimensions must be `[512, 2048, 512]`.
- Target frequency remains 50 Hz.
- Long-run output stays under the OnlineRetarget `outputs/` tree.

The fields under `losses.reported_metrics` are reporting metrics for judging
quality after reconstruction training. They are not additional training losses.

## Launch Surface

Use the wrapper:

```bash
CONFIG=configs/sonic_kin_soma_motionlib_uniform_4gpu.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh

CONFIG=configs/sonic_kin_soma_motionlib_proportional_4gpu.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh
```

The wrapper delegates to
`scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh`, which uses
`scripts/train_sonic_kin_skeleton_ae.py` through `torch.distributed.run`.

## Morphology Bucket Note

`num_clusters` is a legacy config/API key. In short: source skeleton/morphology bucket count, not actuator grouping.
The implementation hashes `skeleton_id` into `0..num_clusters-1`, stores that
as `skeleton_cluster_id`, and appends the normalized bucket scalar to
`soma_morphology`.

The clearer future name is `num_skeleton_buckets`. Until the config/API surface
is migrated, keep `num_clusters` for compatibility and document it as skeleton
bucket count where it appears.

## Stop Condition

Both baselines should either reach the planned step budget with reconstruction
loss curves, G1 kinematic metrics, readable visual artifacts, and run
manifests, or produce reproducible failure reports with run group, config path,
git SHA, and logs.
