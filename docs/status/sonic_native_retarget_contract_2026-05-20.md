# SONIC Kin-Only SOMA Encoder Contract

Date: 2026-05-20. Updated for LR-185 on 2026-05-28.

## Decision

OnlineRetarget's formal training lane is now `sonic_kin_only_soma_encoder`.

This lane is defined as:

```text
SOMA/BVH source motion + skeleton/morphology features
  -> SONIC-compatible SOMA encoder baseline
  -> SONIC shared token/latent space
  -> SONIC g1_kin decoder as the only active retarget target
  -> readable kin-loss / MPJPE / sliding-jitter validation
```

Historical A1/A2/B1/B2 configs are not the current target scope. They remain as
health/history artifacts only and are rejected by strict `--require-formal`
validation.

## Guardrails Added

- Formal configs must set `training_lane: sonic_kin_only_soma_encoder`.
- Formal configs must set `sonic_native: true`.
- Formal config names must be exactly:
  - `sonic_kin_only_soma_encoder_uniform`
  - `sonic_kin_only_soma_encoder_proportional`
- Each formal config must request one 4-GPU launch (`required_gpu_count=4`,
  `sonic_hydra.accelerate_num_processes=4`).
- Formal source features must include SOMA/BVH motion, root
  orientation, and skeleton/morphology conditioning.
- `body_pos_w` and `body_quat_w` are forbidden in formal source encoder inputs.
- `body_pos_w` and `body_quat_w` remain allowed as target labels,
  visualization targets, FK checks, and diagnostics.
- `target_decoder.primary` and `active_decoders` must be exactly `g1_kin`.
- `g1_dyn`, `g1_target_action`, and action/dynamics auxiliary losses are
  forbidden in active formal runs.
- Sonic target frequency must be 50 Hz.
- Visual validation must run every 20k steps, render 8 clips of 4 seconds, and
  upload video to W&B.
- Remote training launch must require committed code and a latest-git check.

## Shared Feature Packer

`src/online_retarget/sonic_native_features.py` defines the shared
train/inference feature contract for the formal lane.

- `SonicNativeFeatureContract.from_config_path(...)` derives source and target
  roles from a formal config after running the strict config validator.
- `pack_training_pair(...)` and `pack_inference_features(...)` use the same
  deployable source keys and emit the same contract digest.
- Source payloads are rejected if they contain target-only fields such as
  `body_pos_w` or `body_quat_w`.
- `assert_matching_contracts(...)` rejects train/inference feature-contract
  drift before a run can be treated as comparable.

## Morphology Features

`src/online_retarget/sonic_morphology.py` converts skeleton-registry rows into
formal source morphology features:

- `actor_uid`
- `skeleton_id`
- `skeleton_cluster_id`
- `height`
- `bone_lengths`
- `body_proportions`
- `foot_leg_arm_torso_measurements`

`pack_source_motion_with_morphology(...)` merges these morphology features with
SOMA/BVH motion features before applying the shared deployable source contract.

## Two Formal Configs

- `configs/sonic_kin_only_soma_encoder_uniform.json`
- `configs/sonic_kin_only_soma_encoder_proportional.json`

Each config reserves one 4-GPU job and targets a 1M-step formal comparison.
Both configs name a Hydra-compatible encoder module target under
`online_retarget.sonic_encoder_modules`; the comparison is uniform versus
proportional SOMA source topology, not A/B architecture variants.

## Encoder Module Surface

`src/online_retarget/sonic_encoder_modules.py` adds four SONIC-compatible
module classes:

- `ConcatSomaEncoderModule`
- `FilmSomaEncoderModule`
- `AdapterSomaEncoderModule`
- `ExpertSomaEncoderModule`

The remote launcher exports `PYTHONPATH=${ROOT}/src` so a SONIC process can
import these module targets after the SONIC-side observation dimensions and
Hydra overrides are wired.

## Launcher Surface

`scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh` launches one active
baseline config as a single 4-GPU job via
`scripts/remote_start_sonic_native_retarget_4gpu.sh`. The historical
`4x1gpu` launcher now refuses to default-launch A1/A2/B1/B2 unless explicitly
unlocked for archaeology, and it rejects active kin-only SOMA encoder configs.

Dry-run command surface:

```bash
PYTHONPATH=src PYTHON_BIN=python3 \
  CONFIG=configs/sonic_kin_only_soma_encoder_uniform.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh

PYTHONPATH=src PYTHON_BIN=python3 \
  CONFIG=configs/sonic_kin_only_soma_encoder_proportional.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh
```

Remote execution surface:

```bash
CHECK_SONIC_PATHS=1 EXECUTE_SONIC_NATIVE_TRAINING=1 \
  CONFIG=configs/sonic_kin_only_soma_encoder_uniform.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh

CHECK_SONIC_PATHS=1 EXECUTE_SONIC_NATIVE_TRAINING=1 \
  CONFIG=configs/sonic_kin_only_soma_encoder_proportional.json \
  scripts/remote_start_sonic_kin_only_soma_encoder_4gpu.sh
```

Stop condition for MLOps: both baselines either reach 1M training steps with
kin loss, MPJPE, readable validation artifacts, and sliding/jitter review cases,
or produce a reproducible failure report with run group, W&B run, config path,
OnlineRetarget git SHA, and SONIC git SHA.

## Remaining Work

- Confirm the remote uniform and proportional SOMA motionlib directories exist
  and pass `--check-paths` immediately before execution.
- After launch, validate the resolved Hydra runtime config for
  `active_decoders=[g1_kin]` and absence of inherited `g1_dyn` blocks.
