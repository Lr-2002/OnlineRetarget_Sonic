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
| Dynamics decoder imitation, latent alignment, kinematic aux, smoothness losses exist | `src/online_retarget/sonic_losses.py` implements `G1DynamicsActionLoss` and `ActionSmoothnessLoss`; configs still rely on Sonic's existing `g1_soma_latent` / `g1_recon` losses for latent/kin aux terms | Partial |
| Sonic 50 Hz timing is enforced | Config validator requires `frequency.target_fps: 50`; formal configs set `motion_lib_cfg.target_fps=50`; visual callback renders `duration_sec * target_fps` frames | Covered at config/callback level |
| Integrated validation runs inside formal Sonic training | `src/online_retarget/sonic_validation_callback.py` implements a Sonic `TrainerCallback`; all formal configs inject it under `callbacks.online_retarget_visual_val`; contract validation now requires this callback wiring | Covered at code/config level |
| W&B video upload inside formal training | `SonicVisualValidationCallback` collects rank-local videos into the shared step directory and main rank logs them with `wandb.Video`; formal configs require `wandb_upload=true` | Covered at code/config level |
| Remote launch checks committed/latest git | `remote_start_sonic_native_retarget_4x1gpu.sh` validates formal configs and requires committed/latest git before execution | Covered |
| A1/A2/B1/B2 can launch under assigned GPU budget | Launcher allocates one GPU per formal config and validates `sonic_hydra.variant_wired=true`; execution still requires a committed/latest clean checkout and the remote Sonic/Isaac environment | Partial until remote dry-run/launch |
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

## Conclusion

The goal is still not complete. The repo now has a correct formal contract,
four variant configs, source-feature guardrails, a shared train/inference
feature packer, registry-backed morphology feature extraction, Hydra-compatible
encoder module classes, Sonic observation/loss injections, and an integrated
Sonic visual validation callback with W&B video upload.

The remaining decisive work is external execution evidence: commit/sync the
repo, run the four formal Sonic/Isaac jobs in the remote environment, confirm
that the callback renders real videos at step 20k, train the variants to 1M
steps, and write the comparison report selecting the best variant.
