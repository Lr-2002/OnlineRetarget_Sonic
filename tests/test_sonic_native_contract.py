import unittest
from copy import deepcopy
from pathlib import Path

from online_retarget.sonic_native_contract import ContractError, validate_config, validate_file


ROOT = Path(__file__).resolve().parents[1]
SONIC_CONFIG = "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml"
SONIC_ACTOR_CRITIC_CONFIG = (
    "gear_sonic/config/actor_critic/universal_token/all_mlp_v1_soma.yaml"
)


class SonicNativeContractTests(unittest.TestCase):
    def test_new_formal_configs_pass_contract(self):
        configs = sorted(ROOT.glob("configs/sonic_native_retarget_*_1gpu.json"))

        self.assertEqual(len(configs), 4)
        for config_path in configs:
            with self.subTest(config=config_path.name):
                result = validate_file(config_path, require_formal=True)

                self.assertTrue(result.formal)
                self.assertEqual(result.training_lane, "sonic_native_retarget")
                self.assertEqual(result.target_decoder, "g1_dyn")

    def test_target_body_pose_labels_are_allowed_when_not_source_inputs(self):
        config = _base_formal_config()

        result = validate_config(config, require_formal=True)

        self.assertTrue(result.formal)

    def test_rejects_body_pose_as_formal_source_feature(self):
        config = _base_formal_config()
        config["source_features"].append("body_pos_w")

        with self.assertRaisesRegex(ContractError, "body_pos_w is forbidden"):
            validate_config(config, require_formal=True)

    def test_rejects_body_quat_as_formal_source_encoder_input(self):
        config = _base_formal_config()
        config["source_encoder"]["inputs"].append("body_quat_w")

        with self.assertRaisesRegex(ContractError, "body_quat_w"):
            validate_config(config, require_formal=True)

    def test_rejects_target_derived_soma_wrist_teacher_forcing(self):
        config = _base_formal_config()
        config["source_encoder"]["inputs"].append("joint_pos_multi_future_wrist_for_soma")

        with self.assertRaisesRegex(ContractError, "teacher forcing"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_g1_dyn_primary_decoder(self):
        config = _base_formal_config()
        config["target_decoder"]["primary"] = "g1_kin"
        config["decoder_targets"] = ["g1_kin"]

        with self.assertRaisesRegex(ContractError, "g1_dyn"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_integrated_visual_callback(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if "online_retarget_visual_val" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "visual validation callback"):
            validate_config(config, require_formal=True)

    def test_adapter_and_expert_configs_require_deterministic_route_wiring(self):
        config = _base_formal_config()
        config["variant"]["type"] = "adapter"
        config["source_encoder"]["module_target"] = (
            "online_retarget.sonic_encoder_modules.AdapterSomaEncoderModule"
        )
        config["sonic_hydra"]["args"] = [
            arg.replace("ConcatSomaEncoderModule", "AdapterSomaEncoderModule")
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "deterministic skeleton-cluster routing"):
            validate_config(config, require_formal=True)

        config["sonic_hydra"]["args"].append(
            "++algo.config.actor.backbone.encoders.soma.params.routing=deterministic_cluster"
        )
        result = validate_config(config, require_formal=True)
        self.assertTrue(result.formal)

    def test_legacy_diagnostic_config_is_not_formal(self):
        config = {
            "schema_version": "1.0.0",
            "owner": "OnlineRetarget",
            "training_lane": "legacy_kin_diagnostic",
            "purpose": "Kinematics only g1_kin diagnostic",
            "features": {"motion_feature": "body_pos_w"},
        }

        result = validate_config(config)

        self.assertFalse(result.formal)
        with self.assertRaisesRegex(ContractError, "training_lane"):
            validate_config(config, require_formal=True)


def _base_formal_config():
    return deepcopy(
        {
            "schema_version": "2.0.0",
            "owner": "OnlineRetarget",
            "training_lane": "sonic_native_retarget",
            "sonic_native": True,
            "source_repo": "/tmp/sonic",
            "sonic_config": SONIC_CONFIG,
            "base_actor_critic_config": SONIC_ACTOR_CRITIC_CONFIG,
            "source_features": [
                "soma_joints_multi_future_local_nonflat",
                "soma_root_ori_b_multi_future",
                "actor_uid",
                "skeleton_id",
                "bone_lengths",
                "body_proportions",
            ],
            "source_encoder": {
                "module_target": "online_retarget.sonic_encoder_modules.ConcatSomaEncoderModule",
                "inputs": [
                    "soma_joints_multi_future_local_nonflat",
                    "soma_root_ori_b_multi_future",
                    "soma_morphology",
                ]
            },
            "target_decoder": {"primary": "g1_dyn", "auxiliary": ["g1_kin"]},
            "decoder_targets": ["g1_dyn", "g1_kin"],
            "target_features": ["action", "joint_pos", "body_pos_w", "body_quat_w"],
            "frequency": {"target_fps": 50},
            "training": {"max_steps": 1000000},
            "visual_validation": {
                "enabled": True,
                "every_steps": 20000,
                "num_videos": 8,
                "duration_sec": 4.0,
                "wandb_upload": True,
            },
            "runtime": {
                "require_committed_code": True,
                "require_latest_code": True,
            },
            "wandb": {
                "enabled": True,
                "project": "OnlineRetarget",
                "log_git_sha": True,
            },
            "sonic_hydra": {
                "variant_wired": True,
                "args": [
                    "++manager_env.observations.tokenizer.soma_morphology.func=online_retarget.sonic_observation_terms:soma_morphology",
                    "++manager_env.observations.tokenizer.g1_target_action.func=online_retarget.sonic_observation_terms:g1_target_action",
                    "++algo.config.actor.backbone.encoders.soma.params._target_=online_retarget.sonic_encoder_modules.ConcatSomaEncoderModule",
                    "++algo.config.actor.backbone.aux_loss_func.online_retarget_g1_dyn_action._target_=online_retarget.sonic_losses.G1DynamicsActionLoss",
                    "algo.config.num_learning_iterations=1000000",
                    "++callbacks.online_retarget_visual_val._target_=online_retarget.sonic_validation_callback.SonicVisualValidationCallback",
                    "++callbacks.online_retarget_visual_val.every_steps=20000",
                    "++callbacks.online_retarget_visual_val.num_videos=8",
                    "++callbacks.online_retarget_visual_val.duration_sec=4.0",
                ],
            },
            "variant": {"name": "test_variant"},
        }
    )
