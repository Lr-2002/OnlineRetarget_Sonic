import importlib
import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

from online_retarget.sonic_native_contract import (
    ContractError,
    FORMAL_BASELINE_NAMES,
    FORMAL_TRAINING_LANE,
    validate_config,
    validate_file,
    validate_resolved_runtime_config,
)


ROOT = Path(__file__).resolve().parents[1]
SONIC_CONFIG = "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml"
SONIC_ACTOR_CRITIC_CONFIG = (
    "gear_sonic/config/actor_critic/universal_token/all_mlp_v1_soma.yaml"
)


class SonicNativeContractTests(unittest.TestCase):
    def test_new_formal_configs_pass_contract(self):
        configs = sorted(ROOT.glob("configs/sonic_kin_only_soma_encoder_*.json"))

        self.assertEqual({path.stem for path in configs}, set(FORMAL_BASELINE_NAMES))
        for config_path in configs:
            with self.subTest(config=config_path.name):
                result = validate_file(config_path, require_formal=True)

                self.assertTrue(result.formal)
                self.assertEqual(result.training_lane, FORMAL_TRAINING_LANE)
                self.assertEqual(result.target_decoder, "g1_kin")

    def test_historical_ab_configs_are_not_current_formal_baselines(self):
        configs = sorted(ROOT.glob("configs/sonic_native_retarget_*_1gpu.json"))

        self.assertEqual(len(configs), 4)
        for config_path in configs:
            with self.subTest(config=config_path.name):
                with self.assertRaisesRegex(ContractError, "training_lane"):
                    validate_file(config_path, require_formal=True)

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

    def test_rejects_g1_encoder_sampling_in_formal_retarget(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace(
                "++manager_env.commands.motion.encoder_sample_probs.g1=0.0",
                "++manager_env.commands.motion.encoder_sample_probs.g1=1.0",
            )
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "encoder_sample_probs.g1=0.0"):
            validate_config(config, require_formal=True)

    def test_rejects_g1_as_active_source_encoder_in_formal_retarget(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace(
                "++algo.config.actor.backbone.active_encoders=[soma]",
                "++algo.config.actor.backbone.active_encoders=[g1,soma]",
            )
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "active_encoders=\\[soma\\]"):
            validate_config(config, require_formal=True)

    def test_rejects_g1_soma_latent_aux_loss_in_formal_retarget(self):
        config = _base_formal_config()
        config["sonic_hydra"]["online_retarget_aux_losses"] = [
            "g1_soma_latent",
        ]

        with self.assertRaisesRegex(ContractError, "g1_soma_latent"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_g1_kin_primary_decoder(self):
        config = _base_formal_config()
        config["target_decoder"]["primary"] = "g1_dyn"
        config["decoder_targets"] = ["g1_dyn", "g1_kin"]

        with self.assertRaisesRegex(ContractError, "g1_kin"):
            validate_config(config, require_formal=True)

    def test_formal_config_rejects_g1_dyn_active_decoder(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace(
                "++algo.config.actor.backbone.active_decoders=[g1_kin]",
                "++algo.config.actor.backbone.active_decoders=[g1_dyn,g1_kin]",
            )
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "active_decoders=\\[g1_kin\\]"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_deleting_inherited_g1_dyn_decoder(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if arg != "~algo.config.actor.backbone.decoders.g1_dyn"
        ]

        with self.assertRaisesRegex(ContractError, "delete inherited decoder"):
            validate_config(config, require_formal=True)

    def test_formal_config_allows_only_g1_dyn_delete_reference(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"].append(
            "++algo.config.actor.backbone.decoders.g1_dyn.outputs=[action]"
        )

        with self.assertRaisesRegex(ContractError, "g1_dyn is forbidden"):
            validate_config(config, require_formal=True)

    def test_formal_config_rejects_action_dynamics_losses(self):
        config = _base_formal_config()
        config["losses"]["primary"] = ["dynamics_action_mse"]

        with self.assertRaisesRegex(ContractError, "action/dynamics losses"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_kinematic_action_wrapper(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if "KinematicActionUniversalTokenModule" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "kinematic action"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_g1_kin_action_bridge(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace(
                "++algo.config.actor.backbone.kinematic_action_decoder=g1_kin",
                "++algo.config.actor.backbone.kinematic_action_decoder=soma",
            )
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "derive actions from the g1_kin"):
            validate_config(config, require_formal=True)

    def test_formal_config_rejects_g1_target_action_observation(self):
        config = _base_formal_config()
        config["target_features"].append("g1_target_action")

        with self.assertRaisesRegex(ContractError, "action/dynamics targets"):
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

    def test_formal_config_requires_hourly_visual_callback_cadence(self):
        config = _base_formal_config()
        config["visual_validation"].pop("every_minutes")
        config["sonic_hydra"]["args"] = [
            arg for arg in config["sonic_hydra"]["args"] if "every_minutes" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "60 minute"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_raw_readable_visual_export(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if "persist_raw_trajectories" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "persist raw trajectories"):
            validate_config(config, require_formal=True)

        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg for arg in config["sonic_hydra"]["args"] if "readable_render=true" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "readable soma-G1"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_readable_clip_00_and_06(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace("readable_clip_indices=[0,6]", "readable_clip_indices=[0]")
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "clip_00 and clip_06"):
            validate_config(config, require_formal=True)

    def test_formal_config_requires_validation_runtime_memory_guardrails(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if "use_evaluation_mode" not in arg and "empty_cuda_cache" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "evaluation-mode motion loading"):
            validate_config(config, require_formal=True)

    def test_resolved_runtime_config_rejects_inherited_g1_dyn_decoder(self):
        config = _base_resolved_runtime_config()
        config["algo"]["config"]["actor"]["backbone"]["decoders"]["g1_dyn"] = {
            "outputs": ["action"],
        }

        with self.assertRaisesRegex(ContractError, "decoders.g1_dyn"):
            validate_resolved_runtime_config(config)

    def test_resolved_runtime_config_accepts_kin_only_evidence(self):
        validate_resolved_runtime_config(_base_resolved_runtime_config())

    def test_formal_config_requires_motionlib_tracking_body_subset(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg
            for arg in config["sonic_hydra"]["args"]
            if "motion_lib_cfg.body_names" not in arg
        ]

        with self.assertRaisesRegex(ContractError, "motion_lib_cfg.body_names"):
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

    def test_check_paths_verifies_motionlib_and_registry_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            result = validate_config(config, require_formal=True, check_paths=True)

        self.assertTrue(result.formal)

    def test_check_paths_rejects_missing_motionlib_before_isaac_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "missing_robot_motionlib"
            soma_dir = root / "soma_motionlib"
            soma_dir.mkdir()
            (soma_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            with self.assertRaisesRegex(ContractError, "robot_motion_file does not exist"):
                validate_config(config, require_formal=True, check_paths=True)

    @unittest.skipUnless(
        importlib.util.find_spec("joblib"),
        "joblib is required for metadata fixtures",
    )
    def test_check_paths_rejects_metadata_only_motionlib_before_isaac_launch(self):
        joblib = importlib.import_module("joblib")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            joblib.dump({"clip": {"num_frames": 10}}, robot_dir / "metadata.pkl")
            (soma_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            with self.assertRaisesRegex(ContractError, "Sonic-loadable per-motion PKL"):
                validate_config(config, require_formal=True, check_paths=True)

    @unittest.skipUnless(
        importlib.util.find_spec("joblib"),
        "joblib is required for metadata fixtures",
    )
    def test_check_paths_rejects_metadata_file_key_mismatch(self):
        joblib = importlib.import_module("joblib")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            joblib.dump({"other_clip": {"num_frames": 10}}, robot_dir / "metadata.pkl")
            (soma_dir / "clip.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            with self.assertRaisesRegex(ContractError, "metadata keys must match"):
                validate_config(config, require_formal=True, check_paths=True)

    def test_check_paths_rejects_unpaired_robot_and_soma_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            (robot_dir / "clip_b.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            with self.assertRaisesRegex(ContractError, "final robot motionlib keys"):
                validate_config(config, require_formal=True, check_paths=True)

    def test_check_paths_allows_unused_soma_extra_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "clip_b.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            result = validate_config(config, require_formal=True, check_paths=True)

        self.assertTrue(result.formal)

    def test_check_paths_accepts_explicit_sonic_remove_motion_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            (robot_dir / "clip_b.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "clip_a.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)
            config["input_data"]["robot_remove_motion_keys"] = ["clip_b"]
            config["sonic_hydra"]["args"].append(
                "++manager_env.commands.motion.motion_lib_cfg.remove_motion_keys=[clip_b]"
            )

            result = validate_config(config, require_formal=True, check_paths=True)

        self.assertTrue(result.formal)

    def test_rejects_missing_hydra_remove_motion_key_wiring(self):
        config = _base_formal_config()
        config["input_data"]["robot_remove_motion_keys"] = ["clip_b"]

        with self.assertRaisesRegex(ContractError, "remove_motion_keys"):
            validate_config(config, require_formal=True)

    def test_check_paths_rejects_motions_missing_registry_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "move__A100.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "move__A100.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)

            with self.assertRaisesRegex(ContractError, "skeleton_registry"):
                validate_config(config, require_formal=True, check_paths=True)

    def test_check_paths_accepts_filter_for_registry_covered_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _base_formal_config()
            config["source_repo"] = str(root / "sonic")
            _write_sonic_config_files(Path(config["source_repo"]))
            robot_dir = root / "robot_motionlib"
            soma_dir = root / "soma_motionlib"
            robot_dir.mkdir()
            soma_dir.mkdir()
            (robot_dir / "move__A001.pkl").write_bytes(b"not-used-by-contract-test")
            (robot_dir / "move__A100.pkl").write_bytes(b"not-used-by-contract-test")
            (soma_dir / "move__A001.pkl").write_bytes(b"not-used-by-contract-test")
            registry = root / "skeleton_registry.csv"
            registry.write_text("actor_uid,actor_height_cm\nA001,170\n", encoding="utf-8")
            _set_data_paths(config, robot_dir, soma_dir, registry)
            filter_regex = r"^(?!.*__(?:A100)(?:_M)?$).*$"
            config["input_data"]["robot_filter_motion_keys"] = filter_regex
            config["sonic_hydra"]["args"].append(
                "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
                + filter_regex
            )

            result = validate_config(config, require_formal=True, check_paths=True)

        self.assertTrue(result.formal)

    def test_rejects_hydra_motion_path_mismatch(self):
        config = _base_formal_config()
        config["sonic_hydra"]["args"] = [
            arg.replace("/tmp/robot_motionlib", "/tmp/other_robot_motionlib")
            for arg in config["sonic_hydra"]["args"]
        ]

        with self.assertRaisesRegex(ContractError, "does not match input_data"):
            validate_config(config, require_formal=True)


def _base_formal_config():
    return deepcopy(
        {
            "schema_version": "2.0.0",
            "owner": "OnlineRetarget",
            "training_lane": FORMAL_TRAINING_LANE,
            "sonic_native": True,
            "source_repo": "/tmp/sonic",
            "sonic_config": SONIC_CONFIG,
            "base_actor_critic_config": SONIC_ACTOR_CRITIC_CONFIG,
            "input_data": {
                "soma_topology": "proportional",
                "robot_motion_file": "/tmp/robot_motionlib",
                "soma_motion_file": "/tmp/soma_motionlib",
                "skeleton_registry": "/tmp/skeleton_registry.csv",
            },
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
            "target_decoder": {"primary": "g1_kin", "auxiliary": []},
            "decoder_targets": ["g1_kin"],
            "target_features": ["joint_pos", "body_pos_w", "body_quat_w"],
            "losses": {
                "primary": ["g1_kin_joint_rmse", "g1_kin_body_mpjpe"],
                "alignment": [],
                "auxiliary": ["temporal_smoothness"],
            },
            "frequency": {"target_fps": 50},
            "training": {"max_steps": 1000000, "required_gpu_count": 4},
            "visual_validation": {
                "enabled": True,
                "every_steps": 20000,
                "every_minutes": 60,
                "num_videos": 8,
                "duration_sec": 4.0,
                "wandb_upload": True,
            },
            "runtime": {
                "required_gpu_count": 4,
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
                "accelerate_num_processes": 4,
                "args": [
                    "++manager_env.observations.tokenizer.soma_morphology.func=online_retarget.sonic_observation_terms:soma_morphology",
                    "++manager_env.observations.tokenizer.soma_morphology.params.registry_csv=/tmp/skeleton_registry.csv",
                    "++manager_env.commands.motion.motion_lib_cfg.motion_file=/tmp/robot_motionlib",
                    "++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/tmp/soma_motionlib",
                    "++manager_env.commands.motion.motion_lib_cfg.body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]",
                    "++manager_env.commands.motion.encoder_sample_probs.g1=0.0",
                    "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0",
                    "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0",
                    "++manager_env.commands.motion.encoder_sample_probs.soma=1.0",
                    "++algo.config.actor.backbone._target_=online_retarget.sonic_runtime_modules.KinematicActionUniversalTokenModule",
                    "++algo.config.actor.backbone.active_encoders=[soma]",
                    "++algo.config.actor.backbone.active_decoders=[g1_kin]",
                    "++algo.config.actor.backbone.kinematic_action_decoder=g1_kin",
                    "++algo.config.actor.backbone.kinematic_action_output=command_multi_future_nonflat",
                    "++algo.config.actor.backbone.kinematic_action_frame_index=0",
                    "++algo.config.actor.backbone.reencode_smpl_g1_recon=false",
                    "~algo.config.actor.backbone.decoders.g1_dyn",
                    "~algo.config.actor.backbone.aux_loss_func.g1_smpl_latent",
                    "~algo.config.actor.backbone.aux_loss_coef.g1_smpl_latent",
                    "~algo.config.actor.backbone.aux_loss_func.g1_teleop_latent",
                    "~algo.config.actor.backbone.aux_loss_coef.g1_teleop_latent",
                    "~algo.config.actor.backbone.aux_loss_func.teleop_smpl_latent",
                    "~algo.config.actor.backbone.aux_loss_coef.teleop_smpl_latent",
                    "~algo.config.actor.backbone.aux_loss_func.reencoded_smpl_g1_latent",
                    "~algo.config.actor.backbone.aux_loss_coef.reencoded_smpl_g1_latent",
                    "~algo.config.actor.backbone.aux_loss_func.g1_soma_latent",
                    "~algo.config.actor.backbone.aux_loss_coef.g1_soma_latent",
                    "++algo.config.actor.backbone.encoders.soma.params._target_=online_retarget.sonic_encoder_modules.ConcatSomaEncoderModule",
                    "algo.config.num_learning_iterations=1000000",
                    "++callbacks.online_retarget_visual_val._target_=online_retarget.sonic_validation_callback.SonicVisualValidationCallback",
                    "++callbacks.online_retarget_visual_val.every_steps=20000",
                    "++callbacks.online_retarget_visual_val.every_minutes=60",
                    "++callbacks.online_retarget_visual_val.num_videos=8",
                    "++callbacks.online_retarget_visual_val.duration_sec=4.0",
                    "++callbacks.online_retarget_visual_val.persist_raw_trajectories=true",
                    "++callbacks.online_retarget_visual_val.readable_render=true",
                    "++callbacks.online_retarget_visual_val.readable_clip_indices=[0,6]",
                    "++callbacks.online_retarget_visual_val.use_evaluation_mode=false",
                    "++callbacks.online_retarget_visual_val.empty_cuda_cache=true",
                ],
            },
            "variant": {"name": "sonic_kin_only_soma_encoder_proportional"},
        }
    )


def _base_resolved_runtime_config():
    return {
        "online_retarget": {
            "contract": FORMAL_TRAINING_LANE,
            "encoder_variant": "sonic_kin_only_soma_encoder_proportional",
        },
        "algo": {
            "config": {
                "actor": {
                    "backbone": {
                        "_target_": (
                            "online_retarget.sonic_runtime_modules."
                            "KinematicActionUniversalTokenModule"
                        ),
                        "active_encoders": ["soma"],
                        "active_decoders": ["g1_kin"],
                        "kinematic_action_decoder": "g1_kin",
                        "decoders": {
                            "g1_kin": {
                                "outputs": ["command_multi_future_nonflat"],
                            },
                        },
                    },
                },
            },
        },
        "callbacks": {
            "online_retarget_visual_val": {
                "use_evaluation_mode": False,
                "empty_cuda_cache": True,
            },
        },
    }


def _write_sonic_config_files(source_repo: Path) -> None:
    for rel in (SONIC_CONFIG, SONIC_ACTOR_CRITIC_CONFIG):
        path = source_repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# contract test fixture\n", encoding="utf-8")


def _set_data_paths(config, robot_dir: Path, soma_dir: Path, registry: Path) -> None:
    replacements = {
        "/tmp/robot_motionlib": str(robot_dir),
        "/tmp/soma_motionlib": str(soma_dir),
        "/tmp/skeleton_registry.csv": str(registry),
    }
    config["input_data"] = {
        "soma_topology": "proportional",
        "robot_motion_file": str(robot_dir),
        "soma_motion_file": str(soma_dir),
        "skeleton_registry": str(registry),
    }
    config["sonic_hydra"]["args"] = [
        _replace_all(arg, replacements) for arg in config["sonic_hydra"]["args"]
    ]


def _replace_all(value: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value
