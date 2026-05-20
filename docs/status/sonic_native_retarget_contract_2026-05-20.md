# SONIC-Native Retarget Contract

Date: 2026-05-20

## Decision

OnlineRetarget's formal training lane is now `sonic_native_retarget`.

This lane is defined as:

```text
SOMA/BVH proportional source motion + skeleton/morphology features
  -> SONIC-compatible source encoder variant
  -> SONIC shared token/latent space
  -> SONIC g1_dyn decoder as the primary retarget target
  -> SONIC g1_kin only as auxiliary diagnostics / visualization
```

The older `sonic_kin_skeleton_*` configs are explicitly marked
`legacy_kin_diagnostic`. They are not formal retargeting because they use G1
target robot state as input and reconstruct G1 kinematic targets.

## Guardrails Added

- Formal configs must set `training_lane: sonic_native_retarget`.
- Formal configs must set `sonic_native: true`.
- Formal source features must include SOMA/BVH/proportional motion, root
  orientation, and skeleton/morphology conditioning.
- `body_pos_w` and `body_quat_w` are forbidden in formal source encoder inputs.
- `body_pos_w` and `body_quat_w` remain allowed as target labels,
  visualization targets, FK checks, and diagnostics.
- `target_decoder.primary` must be `g1_dyn`.
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

## Four Formal Configs

- `configs/sonic_native_retarget_a1_concat_1gpu.json`
- `configs/sonic_native_retarget_a2_film_contact_1gpu.json`
- `configs/sonic_native_retarget_b1_adapter_1gpu.json`
- `configs/sonic_native_retarget_b2_expert_1gpu.json`

Each config reserves one GPU and targets a 1M-step formal comparison.
Each config also names a Hydra-compatible encoder module target under
`online_retarget.sonic_encoder_modules`.

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

`scripts/remote_start_sonic_native_retarget_4x1gpu.sh` validates the four formal
configs, checks both OnlineRetarget and SONIC git state, and writes a launch
manifest. By default it does not start tmux sessions. Even with
`EXECUTE_SONIC_NATIVE_TRAINING=1`, it refuses to launch unless each config sets
`sonic_hydra.variant_wired: true`, so a run cannot silently ignore the requested
encoder variant.

## Remaining Work

- Wire the encoder module targets into actual SONIC Hydra overrides after
  actor/skeleton morphology observation dimensions are known.
- Wire actor/skeleton morphology observations into SONIC tokenizer terms.
- Integrate the three-panel validation renderer into the SONIC training loop,
  not the legacy standalone trainer.
- Confirm that A1/A2/B1/B2 change the actual encoder architecture rather than
  only logging variant metadata.
