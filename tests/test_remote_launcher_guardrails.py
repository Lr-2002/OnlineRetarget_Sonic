from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_native_retarget_4x1gpu.sh"
DDP_LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_native_retarget_4gpu.sh"
KIN_ONLY_LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_kin_only_soma_encoder_4gpu.sh"
KIN_SKELETON_LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_kin_skeleton_4x1gpu.sh"
SUPERVISED_DDP_LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_kin_soma_motionlib_4gpu.sh"
SUPERVISED_TRAINER = REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py"
SUPERVISED_CONFIGS = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_uniform_4gpu.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_proportional_4gpu.json",
)
LOSS_OFF_BASELINE_CONFIG = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json"
)
FINAL_KIN_WALK_DATA_PACKAGE_A_ONLY_CONFIG = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json"
)
FINAL_KIN_WALK_DATA_PACKAGE_A_PLUS_B_CONFIG = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json"
)
FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS = (
    FINAL_KIN_WALK_DATA_PACKAGE_A_ONLY_CONFIG,
    FINAL_KIN_WALK_DATA_PACKAGE_A_PLUS_B_CONFIG,
)
FINAL_KIN_WALK_PACKAGE_INDICATOR = (
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/"
    "lr280_data_package_inventory_20260609T0420Z/indicators/soma_motionlib/kin/walk.txt"
)
FINAL_KIN_WALK_ROW_COUNT = 11248
FINAL_KIN_WALK_ROWS_SHA256 = "2fb36f38d023752e2d1113b1c3455dcb98d1c82318262bde6dfc9c3d34fd79cd"
FINAL_KIN_WALK_VALIDATION_RATIO = 0.015
ACTIVE_FOUR_GPU_SOMA_MOTIONLIB_CONFIGS = (
    *SUPERVISED_CONFIGS,
    LOSS_OFF_BASELINE_CONFIG,
    *FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS,
)
PROPORTIONAL_TREATMENT_AND_BASELINE_CONFIGS = (SUPERVISED_CONFIGS[1], LOSS_OFF_BASELINE_CONFIG)
ACCEPTED_SOMAMESH_VISUAL_FIELDS = {
    "soma_retargeter_root": "/home/user/project/ContextRetarget/third_party/soma-retargeter",
    "somamesh_usd": "/home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd",
    "g1_robot_usd": "/mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd",
    "isaac_python_bin": "/workspace/isaaclab/_isaac_sim/python.sh",
    "isaac_render_script": "scripts/render_g1_isaac_pair.py",
}
ACCEPTED_VISUAL_METRIC_VALIDATION = {
    "enabled": True,
    "every_steps": 20000,
    "output_dir": "metrics",
    "primary": "mpjpe",
    "requested_metrics": ["mpjpe", "w_mpjpe", "context_compositing"],
}
LR270_SHARED_EVAL_COHORT = {
    "enabled": True,
    "id": "lr270_shared_eval_v1",
    "seed": 20260608,
    "include_run_group": True,
    "visual_num_samples": 8,
    "metric_num_samples": 100,
    "manifest_path": "eval_cohort_manifest.json",
}
FINAL_KIN_WALK_EXPECTED_DIMS = {
    "motion_token": 840,
    "x_skel": 104,
    "z_skel": 104,
    "model_input": 944,
    "target": 670,
}
A0_FOUR_GPU_CONFIGS = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_no_skeleton_encoder_uniform_4gpu.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_no_skeleton_encoder_proportional_4gpu.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_uniform_4gpu.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_proportional_4gpu.json",
)
A0_TWO_GPU_2K_VIS_CONFIGS = (
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_no_skeleton_encoder_uniform_2gpu_2kvis.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_no_skeleton_encoder_proportional_2gpu_2kvis.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_uniform_2gpu_2kvis.json",
    REPO_ROOT / "configs" / "sonic_kin_soma_motionlib_a0_frozen_ae_proportional_2gpu_2kvis.json",
)
A0_TWO_GPU_TO_FOUR_GPU_CONFIGS = tuple(zip(A0_TWO_GPU_2K_VIS_CONFIGS, A0_FOUR_GPU_CONFIGS))


class RemoteLauncherGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = LAUNCHER.read_text(encoding="utf-8")

    def test_online_retarget_must_be_clean_and_latest(self) -> None:
        text = self.launcher_text
        self.assertIn("git diff --quiet && git diff --cached --quiet", text)
        self.assertIn("OnlineRetarget repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git "${ROOT}" "OnlineRetarget repo"', text)

    def test_latest_check_fetches_upstream_and_requires_exact_head(self) -> None:
        text = self.launcher_text
        self.assertIn("rev-parse --abbrev-ref --symbolic-full-name '@{u}'", text)
        self.assertIn('git -C "${repo}" fetch --quiet "${remote}" "${branch}"', text)
        self.assertIn('head="$(git -C "${repo}" rev-parse HEAD)"', text)
        self.assertIn('upstream_head="$(git -C "${repo}" rev-parse FETCH_HEAD)"', text)
        self.assertIn('if [[ "${head}" != "${upstream_head}" ]]; then', text)
        self.assertIn("refusing to train without a latest-code check", text)

    def test_sonic_repo_is_checked_for_real_training(self) -> None:
        text = self.launcher_text
        self.assertIn('if [[ "${EXECUTE_SONIC_NATIVE_TRAINING}" == "1" ]]; then', text)
        self.assertIn("SONIC source repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git_if_configured "${SONIC_ROOT}" "SONIC source repo"', text)

    def test_launch_records_online_retarget_and_sonic_commits(self) -> None:
        text = self.launcher_text
        self.assertIn('CONTROL_COMMIT="$(git rev-parse HEAD)"', text)
        self.assertIn('SONIC_COMMIT="$(git -C "${SONIC_ROOT}" rev-parse HEAD)"', text)
        self.assertIn('++online_retarget.git_sha={online_retarget_commit}', text)
        self.assertIn('++online_retarget.sonic_git_sha={sonic_commit}', text)
        self.assertIn("ONLINE_RETARGET_GIT_SHA", text)
        self.assertIn("SONIC_GIT_SHA", text)

    def test_historical_four_by_one_launcher_has_no_default_ab_launch(self) -> None:
        text = self.launcher_text
        self.assertIn("ALLOW_HISTORICAL_A_B_4X1GPU", text)
        self.assertIn("A1/A2/B1/B2 4x1-GPU launching is historical", text)
        self.assertIn("LR-273 loss-on config", text)
        self.assertIn("LR-274 loss-off baseline config", text)
        self.assertIn("active kin-only SOMA encoder treatment/baseline configs must run as one 4-GPU job", text)


class NativeRetargetFourGpuLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = DDP_LAUNCHER.read_text(encoding="utf-8")

    def test_single_config_multi_gpu_launcher_uses_accelerate_processes(self) -> None:
        text = self.launcher_text
        self.assertIn('CONFIG="${CONFIG:-configs/sonic_kin_only_soma_encoder_proportional.json}"', text)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-4}"', text)
        self.assertIn('--num_processes="${NPROC_PER_NODE}"', text)
        self.assertNotIn("--num_processes=1 gear_sonic/train_agent_trl.py", text)
        self.assertNotIn("sonic_native_retarget_a1_concat_1gpu", text)

    def test_single_config_launcher_rejects_multiple_configs(self) -> None:
        text = self.launcher_text
        self.assertIn("CONFIG must name exactly one formal config", text)
        self.assertIn('if [[ "${CONFIG}" == *" "* ]]; then', text)
        self.assertIn("required_gpu_count", text)

    def test_single_config_launcher_preserves_training_guardrails(self) -> None:
        text = self.launcher_text
        self.assertIn("OnlineRetarget repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git "${ROOT}" "OnlineRetarget repo"', text)
        self.assertIn("SONIC source repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git_if_configured "${SONIC_ROOT}" "SONIC source repo"', text)
        self.assertIn("ONLINE_RETARGET_GIT_SHA", text)
        self.assertIn("SONIC_GIT_SHA", text)

    def test_single_config_launcher_uses_configured_entrypoint(self) -> None:
        text = self.launcher_text
        self.assertIn("SONIC_ENTRYPOINT", text)
        self.assertIn("sonic_entrypoint", text)
        self.assertIn("sonic_entrypoint_quoted", text)
        self.assertIn('"entrypoint": sys.argv[14]', text)

    def test_single_config_launcher_defaults_to_nccl_workarounds(self) -> None:
        text = self.launcher_text
        self.assertIn('NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"', text)
        self.assertIn('NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"', text)
        self.assertIn('NCCL_ALGO="${NCCL_ALGO:-Ring}"', text)
        self.assertIn('export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE}"', text)
        self.assertIn('export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"', text)
        self.assertIn('export NCCL_ALGO="${NCCL_ALGO}"', text)
        self.assertIn('"nccl_shm_disable": sys.argv[10]', text)
        self.assertIn('"nccl_ib_disable": sys.argv[11]', text)
        self.assertIn('"nccl_algo": sys.argv[12]', text)
        self.assertIn('"contract": sys.argv[13]', text)


class KinOnlySomaEncoderLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = KIN_ONLY_LAUNCHER.read_text(encoding="utf-8")

    def test_wrapper_defaults_to_proportional_loss_on_treatment_config(self) -> None:
        text = self.launcher_text
        self.assertIn("LR-273 temporal-consistency loss-on treatment", text)
        self.assertIn("LR-274 loss-off baseline", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_4gpu.json", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json", text)
        self.assertIn("remote_start_sonic_kin_soma_motionlib_4gpu.sh", text)


class SupervisedSomaMotionlibFourGpuConfigTests(unittest.TestCase):
    def test_configs_are_strict_supervised_four_gpu_loss_on_treatments(self) -> None:
        expected = {
            "uniform": "sonic_kin_only_soma_encoder_uniform_temporal_consistency",
            "proportional": "sonic_kin_only_soma_encoder_proportional_temporal_consistency",
        }
        for path in SUPERVISED_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                topology = config["input_data"]["soma_topology"]
                self.assertEqual(config["training_lane"], "soma_motionlib_kin_only")
                self.assertIn("temporal-consistency loss-on treatment", config["purpose"])
                self.assertIn("not the", config["purpose"])
                self.assertEqual(config["variant"]["name"], expected[topology])
                self.assertEqual(config["variant"]["family"], "soma_encoder_temporal_consistency_treatment")
                self.assertEqual(config["variant"]["soma_topology"], topology)
                self.assertEqual(config["runtime"]["required_gpu_count"], 4)
                self.assertEqual(config["training"]["required_gpu_count"], 4)
                self.assertEqual(config["target_decoder"]["primary"], "g1_kin")
                self.assertEqual(config["decoder_targets"], ["g1_kin"])
                self.assertEqual(
                    config["losses"]["auxiliary"],
                    [
                        "g1_kin_command_temporal_consistency_delta_mse",
                        "g1_kin_command_ab_overlap_mse",
                    ],
                )
                self.assertIs(config["training"]["temporal_consistency_loss_enabled"], True)
                self.assertEqual(config["training"]["temporal_consistency_loss_weight"], 0.01)
                self.assertIs(config["training"]["ab_overlap_loss_enabled"], True)
                self.assertEqual(config["training"]["ab_overlap_loss_weight"], 0.01)
                self.assertIn(f"soma_{topology}_filtered_v1", config["input_data"]["soma_motion_dir"])
                self.assertIn("temporal-consistency-loss-on", config["wandb"]["tags"])
                self.assertIn("ab-overlap-loss-on", config["wandb"]["tags"])
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("sonic_hydra", text)
                self.assertNotIn("train_agent_trl.py", text)
                self.assertNotIn("KinematicActionUniversalTokenModule", text)
                self.assertNotIn("g1_dyn", text)
                self.assertNotIn("g1_target_action", text)
                self.assertNotIn("episode_length", text)

    def test_final_kin_walk_data_package_configs_are_present_and_pinned(self) -> None:
        self.assertEqual(len(FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS), 2)
        for path in FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"missing final kin/walk package config: {path}")
                config = json.loads(path.read_text(encoding="utf-8"))
                data_package = config["input_data"].get("data_package")

                self.assertIsInstance(data_package, dict)
                self.assertEqual(config["input_data"]["format"], "soma_motionlib")
                self.assertEqual(config["input_data"]["soma_topology"], "proportional")
                self.assertIn("soma_proportional_filtered_v1", config["input_data"]["soma_motion_dir"])
                self.assertEqual(data_package["spec"], "kin")
                self.assertEqual(data_package["category"], "walk")
                self.assertEqual(data_package["indicator"], FINAL_KIN_WALK_PACKAGE_INDICATOR)
                self.assertEqual(data_package["missing_policy"], "error")
                self.assertEqual(data_package["expected_row_count"], FINAL_KIN_WALK_ROW_COUNT)
                self.assertEqual(data_package["package_rows_sha256"], FINAL_KIN_WALK_ROWS_SHA256)
                self.assertNotIn("raw_sonic", data_package["indicator"])
                self.assertEqual(config["training"]["required_gpu_count"], 4)
                self.assertEqual(config["runtime"]["required_gpu_count"], 4)
                self.assertEqual(config["split"]["validation_ratio"], FINAL_KIN_WALK_VALIDATION_RATIO)
                self.assertGreaterEqual(
                    FINAL_KIN_WALK_ROW_COUNT * config["split"]["validation_ratio"],
                    LR270_SHARED_EVAL_COHORT["metric_num_samples"],
                )
                self.assertEqual(
                    config["validation_command"],
                    f"CONFIG=configs/{path.name} scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh",
                )
                self.assertEqual(config["features"]["expected_dims"], FINAL_KIN_WALK_EXPECTED_DIMS)
                self.assertIn("data-package", config["wandb"]["tags"])
                self.assertIn("kin-walk-package", config["wandb"]["tags"])

    def test_final_kin_walk_data_package_loss_toggles_match_contract(self) -> None:
        a_only = json.loads(FINAL_KIN_WALK_DATA_PACKAGE_A_ONLY_CONFIG.read_text(encoding="utf-8"))
        a_plus_b = json.loads(FINAL_KIN_WALK_DATA_PACKAGE_A_PLUS_B_CONFIG.read_text(encoding="utf-8"))

        self.assertIs(a_only["training"]["temporal_consistency_loss_enabled"], True)
        self.assertEqual(a_only["training"]["temporal_consistency_loss_weight"], 0.01)
        self.assertIs(a_only["training"]["ab_overlap_loss_enabled"], False)
        self.assertEqual(a_only["training"]["ab_overlap_loss_weight"], 0.0)
        self.assertEqual(
            a_only["losses"]["auxiliary"],
            ["g1_kin_command_temporal_consistency_delta_mse"],
        )
        self.assertIn("temporal-consistency-loss-on", a_only["wandb"]["tags"])
        self.assertIn("ab-overlap-loss-off", a_only["wandb"]["tags"])
        self.assertIn("a-only", a_only["wandb"]["tags"])

        self.assertIs(a_plus_b["training"]["temporal_consistency_loss_enabled"], True)
        self.assertEqual(a_plus_b["training"]["temporal_consistency_loss_weight"], 0.01)
        self.assertIs(a_plus_b["training"]["ab_overlap_loss_enabled"], True)
        self.assertEqual(a_plus_b["training"]["ab_overlap_loss_weight"], 0.01)
        self.assertEqual(
            a_plus_b["losses"]["auxiliary"],
            [
                "g1_kin_command_temporal_consistency_delta_mse",
                "g1_kin_command_ab_overlap_mse",
            ],
        )
        self.assertIn("temporal-consistency-loss-on", a_plus_b["wandb"]["tags"])
        self.assertIn("ab-overlap-loss-on", a_plus_b["wandb"]["tags"])
        self.assertIn("a-plus-b", a_plus_b["wandb"]["tags"])

    def test_final_kin_walk_a_plus_b_preserves_a_only_contract_except_overlap_identity(self) -> None:
        a_only = json.loads(FINAL_KIN_WALK_DATA_PACKAGE_A_ONLY_CONFIG.read_text(encoding="utf-8"))
        a_plus_b = json.loads(FINAL_KIN_WALK_DATA_PACKAGE_A_PLUS_B_CONFIG.read_text(encoding="utf-8"))
        preserved_keys = (
            "source_repo",
            "source_rev",
            "input_data",
            "source_features",
            "target_decoder",
            "decoder_targets",
            "target_features",
            "features",
            "split",
            "normalization",
            "model",
            "visual_validation",
            "metric_validation",
            "evaluation_cohort",
            "runtime",
            "expected_artifacts",
        )
        for key in preserved_keys:
            self.assertEqual(a_plus_b[key], a_only[key], key)

        a_only_training = dict(a_only["training"])
        a_plus_b_training = dict(a_plus_b["training"])
        for key in ("ab_overlap_loss_enabled", "ab_overlap_loss_weight"):
            a_only_training.pop(key)
            a_plus_b_training.pop(key)
        self.assertEqual(a_plus_b_training, a_only_training)
        self.assertEqual(a_plus_b["losses"]["primary"], a_only["losses"]["primary"])
        self.assertEqual(
            a_plus_b["losses"]["auxiliary"],
            [*a_only["losses"]["auxiliary"], "g1_kin_command_ab_overlap_mse"],
        )

    def test_final_kin_walk_package_configs_preserve_shared_metric_eval_cohort(self) -> None:
        for path in FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["metric_validation"], ACCEPTED_VISUAL_METRIC_VALIDATION)
                self.assertEqual(config["evaluation_cohort"], LR270_SHARED_EVAL_COHORT)
                self.assertEqual(config["visual_validation"]["num_videos"], LR270_SHARED_EVAL_COHORT["visual_num_samples"])
                self.assertEqual(
                    config["metric_validation"]["every_steps"],
                    config["visual_validation"]["every_steps"],
                )

    def test_loss_off_baseline_config_matches_proportional_treatment_except_loss_contract(self) -> None:
        treatment = json.loads(SUPERVISED_CONFIGS[1].read_text(encoding="utf-8"))
        baseline = json.loads(LOSS_OFF_BASELINE_CONFIG.read_text(encoding="utf-8"))

        self.assertIn("LR-274", baseline["purpose"])
        self.assertIn("loss-off baseline", baseline["purpose"])
        self.assertEqual(baseline["training_lane"], "soma_motionlib_kin_only")
        for key in (
            "source_repo",
            "source_rev",
            "input_data",
            "source_features",
            "target_decoder",
            "decoder_targets",
            "target_features",
            "features",
            "split",
            "normalization",
            "model",
            "visual_validation",
            "metric_validation",
            "evaluation_cohort",
            "runtime",
        ):
            self.assertEqual(baseline[key], treatment[key])
        treatment_training = dict(treatment["training"])
        baseline_training = dict(baseline["training"])
        for key in (
            "temporal_consistency_loss_enabled",
            "temporal_consistency_loss_weight",
            "ab_overlap_loss_enabled",
            "ab_overlap_loss_weight",
        ):
            treatment_training.pop(key)
            baseline_training.pop(key)
        self.assertEqual(baseline_training, treatment_training)
        self.assertEqual(baseline["losses"]["primary"], treatment["losses"]["primary"])
        self.assertEqual(baseline["losses"]["auxiliary"], [])
        self.assertIs(baseline["training"]["temporal_consistency_loss_enabled"], False)
        self.assertEqual(baseline["training"]["temporal_consistency_loss_weight"], 0.0)
        self.assertIs(baseline["training"]["ab_overlap_loss_enabled"], False)
        self.assertEqual(baseline["training"]["ab_overlap_loss_weight"], 0.0)
        self.assertEqual(baseline["training"]["seed"], treatment["training"]["seed"])
        self.assertEqual(baseline["training"]["max_steps"], 1000000)
        self.assertEqual(baseline["training"]["required_gpu_count"], 4)
        self.assertEqual(baseline["runtime"]["required_gpu_count"], 4)
        self.assertEqual(
            baseline["validation_command"],
            "CONFIG=configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json "
            "scripts/remote_start_sonic_kin_soma_motionlib_4gpu.sh",
        )
        self.assertNotIn("g1_kin_command_temporal_consistency_delta_mse", LOSS_OFF_BASELINE_CONFIG.read_text())
        self.assertNotIn("g1_kin_command_ab_overlap_mse", LOSS_OFF_BASELINE_CONFIG.read_text())
        self.assertIn("temporal-consistency-loss-off", baseline["wandb"]["tags"])
        self.assertIn("ab-overlap-loss-off", baseline["wandb"]["tags"])

    def test_active_four_gpu_configs_use_accepted_somamesh_visual_backend(self) -> None:
        for path in ACTIVE_FOUR_GPU_SOMA_MOTIONLIB_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                visual = config["visual_validation"]
                self.assertIs(visual["enabled"], True)
                self.assertIs(visual["acceptance_backend"], True)
                self.assertEqual(visual["every_steps"], 20000)
                self.assertEqual(visual["every_minutes"], 60)
                self.assertEqual(visual["num_videos"], 8)
                for key, expected in ACCEPTED_SOMAMESH_VISUAL_FIELDS.items():
                    self.assertEqual(visual[key], expected)
                self.assertIn("acceptance-backend", config["wandb"]["tags"])

    def test_proportional_treatment_and_baseline_log_visual_body_metrics(self) -> None:
        for path in PROPORTIONAL_TREATMENT_AND_BASELINE_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["metric_validation"], ACCEPTED_VISUAL_METRIC_VALIDATION)
                self.assertEqual(
                    config["metric_validation"]["every_steps"],
                    config["visual_validation"]["every_steps"],
                )
                self.assertEqual(config["training"]["validate_every"], 200)
                self.assertEqual(config["metric_validation"]["every_steps"] % config["training"]["validate_every"], 0)

    def test_proportional_treatment_and_baseline_share_fixed_eval_cohort(self) -> None:
        treatment = json.loads(PROPORTIONAL_TREATMENT_AND_BASELINE_CONFIGS[0].read_text(encoding="utf-8"))
        baseline = json.loads(PROPORTIONAL_TREATMENT_AND_BASELINE_CONFIGS[1].read_text(encoding="utf-8"))
        self.assertEqual(treatment["evaluation_cohort"], LR270_SHARED_EVAL_COHORT)
        self.assertEqual(baseline["evaluation_cohort"], LR270_SHARED_EVAL_COHORT)
        self.assertEqual(treatment["evaluation_cohort"], baseline["evaluation_cohort"])
        self.assertEqual(treatment["visual_validation"]["num_videos"], LR270_SHARED_EVAL_COHORT["visual_num_samples"])
        self.assertEqual(baseline["visual_validation"]["num_videos"], LR270_SHARED_EVAL_COHORT["visual_num_samples"])

    def test_supervised_trainer_gates_shared_eval_cohort_and_preserves_legacy_salt(self) -> None:
        text = SUPERVISED_TRAINER.read_text(encoding="utf-8")
        self.assertIn("build_evaluation_cohort(validation_dataset.rows, config, run_group=run_group)", text)
        self.assertIn("def select_visual_validation_rows(", text)
        self.assertIn("if evaluation_cohort_config(config):", text)
        self.assertIn("evaluation_cohort_manifest_payload", text)
        self.assertIn("metric_val_loader", text)
        self.assertIn("metric_rows_sha256", text)
        self.assertIn("visual_rows_sha256", text)
        self.assertIn("max_batches=0", text)
        self.assertIn('"variant.name"', text)
        self.assertIn("salt=f\"{config['variant']['name']}:{config['training']['seed']}\"", text)


class A0TwoGpuAcceptedVisualizationConfigTests(unittest.TestCase):
    def test_original_accepted_a0_four_gpu_configs_remain_formal_records(self) -> None:
        for path in A0_FOUR_GPU_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["training"]["required_gpu_count"], 4)
                self.assertEqual(config["runtime"]["required_gpu_count"], 4)
                self.assertEqual(config["visual_validation"]["every_steps"], 20000)
                self.assertNotIn("acceptance_backend", config["visual_validation"])
                self.assertEqual(config["variant"]["gpu_topology"], "single_4gpu_ddp_job")

    def test_two_gpu_2k_visual_configs_are_committed_5090_validation_profiles(self) -> None:
        for path in A0_TWO_GPU_2K_VIS_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                visual = config["visual_validation"]
                self.assertEqual(config["training_lane"], "soma_motionlib_kin_only")
                self.assertEqual(config["training"]["required_gpu_count"], 2)
                self.assertEqual(config["runtime"]["required_gpu_count"], 2)
                self.assertEqual(config["variant"]["gpu_topology"], "single_2gpu_ddp_job")
                self.assertIn("accepted SomaMesh/SOMA Shapes + G1 Isaac visual validation", config["purpose"])
                self.assertNotIn("accepted SOMA Skeleton", config["purpose"])
                self.assertEqual(visual["every_steps"], 2000)
                self.assertEqual(visual["every_minutes"], 0)
                self.assertIs(visual["acceptance_backend"], True)
                self.assertEqual(visual["isaac_python_bin"], "/workspace/isaaclab/_isaac_sim/python.sh")
                self.assertEqual(visual["isaac_render_script"], "scripts/render_g1_isaac_pair.py")
                self.assertEqual(
                    visual["soma_retargeter_root"],
                    "/home/user/project/ContextRetarget/third_party/soma-retargeter",
                )
                self.assertEqual(
                    visual["somamesh_usd"],
                    "/home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd",
                )
                self.assertIn("source_bvh_roots", visual)
                self.assertIn("source_bvh_cache", visual)
                self.assertIn("source_bvh_tar", visual)
                self.assertIn("g1_robot_usd", visual)
                self.assertIn("--nproc-per-node=2", config["validation_command"])
                self.assertIn(f"--config configs/{path.name}", config["validation_command"])
                self.assertIn("2gpu", config["wandb"]["tags"])
                self.assertIn("2kvis", config["wandb"]["tags"])
                self.assertIn("acceptance-backend", config["wandb"]["tags"])
                self.assertNotIn("4gpu", config["wandb"]["tags"])

    def test_two_gpu_profiles_only_change_launch_and_visual_validation_contract(self) -> None:
        preserved_keys = (
            "training_lane",
            "input_data",
            "source_features",
            "target_decoder",
            "decoder_targets",
            "target_features",
            "losses",
            "evaluation_metrics",
            "features",
            "split",
            "normalization",
            "model",
            "ddp",
        )
        for two_gpu_path, four_gpu_path in A0_TWO_GPU_TO_FOUR_GPU_CONFIGS:
            with self.subTest(path=two_gpu_path.name):
                two_gpu = json.loads(two_gpu_path.read_text(encoding="utf-8"))
                four_gpu = json.loads(four_gpu_path.read_text(encoding="utf-8"))
                for key in preserved_keys:
                    self.assertEqual(two_gpu[key], four_gpu[key])
                self.assertEqual(two_gpu["training"]["seed"], four_gpu["training"]["seed"])
                self.assertEqual(two_gpu["training"]["max_steps"], four_gpu["training"]["max_steps"])
                self.assertEqual(two_gpu["output_dir"], four_gpu["output_dir"])


class SupervisedSomaMotionlibFourGpuLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = SUPERVISED_DDP_LAUNCHER.read_text(encoding="utf-8")

    def _run_supervised_launcher_until_cuda_smoke(
        self, config: dict[str, object], temp_root: Path
    ) -> subprocess.CompletedProcess[str]:
        (temp_root / "configs").mkdir()
        robot_motion_dir = temp_root / "robot_motion"
        soma_motion_dir = temp_root / "soma_motion"
        robot_motion_dir.mkdir()
        soma_motion_dir.mkdir()
        input_data = dict(config.get("input_data", {}))
        input_data["robot_motion_dir"] = str(robot_motion_dir)
        input_data["soma_motion_dir"] = str(soma_motion_dir)
        config["input_data"] = input_data
        config_path = temp_root / "configs" / "guardrail_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        fake_python = temp_root / "python.sh"
        python_bin = shlex.quote(sys.executable)
        fake_python.write_text(
            "\n".join(
                (
                    "#!/usr/bin/env bash",
                    "set -uo pipefail",
                    'if [[ "${1:-}" == "-" ]]; then',
                    '  payload="$(mktemp)"',
                    '  cat > "${payload}"',
                    '  if grep -q "import torch" "${payload}"; then',
                    '    echo "CUDA is required for strict supervised 4-GPU smoke" >&2',
                    '    rm -f "${payload}"',
                    "    exit 1",
                    "  fi",
                    f"  {python_bin} \"$@\" < \"${{payload}}\"",
                    "  status=$?",
                    '  rm -f "${payload}"',
                    '  exit "${status}"',
                    "fi",
                    f"exec {python_bin} \"$@\"",
                    "",
                )
            ),
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(temp_root),
                "PYTHON_BIN": str(fake_python),
                "CONFIG": "configs/guardrail_config.json",
                "NPROC_PER_NODE": "4",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            }
        )
        return subprocess.run(
            ["bash", str(SUPERVISED_DDP_LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_launcher_uses_torch_distributed_supervised_entrypoint(self) -> None:
        text = self.launcher_text
        self.assertIn("LR-273 temporal-consistency loss-on treatment", text)
        self.assertIn("LR-274 loss-off baseline", text)
        self.assertIn("LR-280 kin/walk data-package smoke targets", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_4gpu.json", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_loss_off_baseline_4gpu.json", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_only_4gpu.json", text)
        self.assertIn("configs/sonic_kin_soma_motionlib_kin_walk_data_package_a_plus_b_4gpu.json", text)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-4}"', text)
        self.assertIn("torch.distributed.run", text)
        self.assertIn("scripts/train_sonic_kin_skeleton_ae.py", text)
        self.assertNotIn("gear_sonic/train_agent_trl.py", text)
        self.assertNotIn("accelerate.commands.launch", text)

    def test_launcher_rejects_non_supervised_or_rollout_configs(self) -> None:
        text = self.launcher_text
        self.assertIn("training_lane=soma_motionlib_kin_only", text)
        self.assertIn("CONFIG must use training_lane=soma_motionlib_kin_only", text)
        self.assertIn("KinematicActionUniversalTokenModule", text)
        self.assertIn("sonic_hydra", text)
        self.assertIn("episode_length", text)
        self.assertIn("NPROC_PER_NODE must match required_gpu_count", text)

    def test_launcher_guard_allows_descriptive_reward_prose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "training_lane": "soma_motionlib_kin_only",
                "purpose": "Strict supervised baseline with no rollout or reward surface.",
                "runtime": {"required_gpu_count": 4},
                "training": {"required_gpu_count": 4},
            }
            result = self._run_supervised_launcher_until_cuda_smoke(config, Path(tmp))

        self.assertEqual(result.returncode, 1)
        self.assertIn("CUDA is required for strict supervised 4-GPU smoke", result.stderr)
        self.assertNotIn("CONFIG contains PPO/Isaac/reward/episode-length tokens", result.stderr)

    def test_launcher_guard_allows_loss_off_baseline_config_purpose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(LOSS_OFF_BASELINE_CONFIG.read_text(encoding="utf-8"))
            result = self._run_supervised_launcher_until_cuda_smoke(config, Path(tmp))

        self.assertEqual(result.returncode, 1)
        self.assertIn("reward surface", config["purpose"])
        self.assertIn("CUDA is required for strict supervised 4-GPU smoke", result.stderr)
        self.assertNotIn("CONFIG contains PPO/Isaac/reward/episode-length tokens", result.stderr)

    def test_launcher_guard_accepts_final_kin_walk_package_configs_until_cuda_smoke(self) -> None:
        for path in FINAL_KIN_WALK_DATA_PACKAGE_CONFIGS:
            with self.subTest(path=path.name), tempfile.TemporaryDirectory() as tmp:
                config = json.loads(path.read_text(encoding="utf-8"))
                result = self._run_supervised_launcher_until_cuda_smoke(config, Path(tmp))

                self.assertEqual(result.returncode, 1)
                self.assertIn("CUDA is required for strict supervised 4-GPU smoke", result.stderr)
                self.assertNotIn("CONFIG contains PPO/Isaac/reward/episode-length tokens", result.stderr)
                self.assertNotIn("NPROC_PER_NODE must match required_gpu_count", result.stderr)

    def test_launcher_guard_still_rejects_reward_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "training_lane": "soma_motionlib_kin_only",
                "purpose": "Strict supervised baseline.",
                "reward": {"scale": 1.0},
                "runtime": {"required_gpu_count": 4},
                "training": {"required_gpu_count": 4},
            }
            result = self._run_supervised_launcher_until_cuda_smoke(config, Path(tmp))

        self.assertEqual(result.returncode, 1)
        self.assertIn("forbidden strict-supervised token 'reward' in key at reward", result.stderr)
        self.assertIn("CONFIG contains PPO/Isaac/reward/episode-length tokens", result.stderr)
        self.assertNotIn("CUDA is required for strict supervised 4-GPU smoke", result.stderr)

    def test_launcher_supports_short_smoke_overrides(self) -> None:
        text = self.launcher_text
        self.assertIn("MAX_STEPS", text)
        self.assertIn("--max-steps", text)
        self.assertIn("WANDB_MODE", text)
        self.assertIn("--wandb-mode", text)
        self.assertIn("DISABLE_VISUAL_VALIDATION", text)
        self.assertIn("--disable-visual-validation", text)

    def test_launcher_supports_supervised_resume_checkpoint_override(self) -> None:
        text = self.launcher_text
        self.assertIn("RESUME_CHECKPOINT", text)
        self.assertIn("missing supervised resume checkpoint", text)
        self.assertIn("--resume-checkpoint", text)
        self.assertIn('"resume_checkpoint": sys.argv[13]', text)

    def test_launcher_accepts_2gpu_configs_by_matching_required_gpu_count(self) -> None:
        text = self.launcher_text
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-4}"', text)
        self.assertIn('REQUIRED_GPU_COUNT="$("${PYTHON_BIN}" -c', text)
        self.assertIn('if [[ "${REQUIRED_GPU_COUNT}" != "${NPROC_PER_NODE}" ]]; then', text)
        self.assertIn("NPROC_PER_NODE must match required_gpu_count", text)

    def test_launcher_preflight_scope_matches_strict_supervised_smoke(self) -> None:
        text = self.launcher_text
        self.assertIn('require_latest_git "${ROOT}" "OnlineRetarget repo"', text)
        self.assertIn("robot_motion_dir", text)
        self.assertIn("soma_motion_dir", text)
        self.assertIn("torch.cuda.is_available", text)
        self.assertIn("torch.distributed.is_available", text)
        self.assertIn("external_source_guard", text)
        self.assertIn("not_required_supervised_entrypoint_no_external_import_exec", text)
        self.assertNotIn("SONIC_ROOT", text)
        self.assertNotIn("SONIC source repo", text)
        self.assertNotIn('require_latest_git "${SOURCE_ROOT}"', text)


class SupervisedTrainerDdpGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trainer_text = SUPERVISED_TRAINER.read_text(encoding="utf-8")

    def test_trainer_initializes_torchrun_ddp_runtime(self) -> None:
        text = self.trainer_text
        self.assertIn("RANK", text)
        self.assertIn("WORLD_SIZE", text)
        self.assertIn("LOCAL_RANK", text)
        self.assertIn("torch.distributed.init_process_group", text)
        self.assertIn("DistributedDataParallel", text)
        self.assertIn("DistributedSampler", text)
        self.assertIn("--local-rank", text)
        self.assertIn("expected torchrun WORLD_SIZE", text)

    def test_trainer_keeps_side_effects_on_main_rank(self) -> None:
        text = self.trainer_text
        self.assertIn("is_main_process", text)
        self.assertIn("write_loss_header(loss_curve)", text)
        self.assertIn("manifest = write_manifest", text)
        self.assertIn("wandb_run = init_wandb(config, manifest, output_dir, run_group) if is_main else None", text)
        self.assertIn("run_visual_validation", text)
        self.assertIn("save_checkpoint(output_dir, unwrap_model(model)", text)
        self.assertIn("distributed_barrier(runtime)", text)

    def test_trainer_exposes_supervised_resume_checkpoint_restore_path(self) -> None:
        text = self.trainer_text
        self.assertIn("--resume-checkpoint", text)
        self.assertIn("load_supervised_resume_checkpoint", text)
        self.assertIn('required_keys = ("model", "optimizer", "step")', text)
        self.assertIn("unwrap_model(model).load_state_dict(payload[\"model\"])", text)
        self.assertIn("optimizer.load_state_dict(payload[\"optimizer\"])", text)
        self.assertIn('step = int(resume_state["step"]) if resume_state is not None else 0', text)
        self.assertIn('start_event["resume_checkpoint"] = resume_state["path"]', text)

    def test_trainer_data_package_stays_paired_soma_motionlib_only(self) -> None:
        text = self.trainer_text
        self.assertIn("filter_rows_by_data_package_config", text)
        self.assertIn("manifest_summary_from_selected_rows", text)
        self.assertIn("apply_max_clips=package_cfg is None", text)
        self.assertIn("input_data.data_package is currently supported only for paired soma_motionlib configs", text)
        self.assertIn("raw NPZ package indicators require explicit approval", text)
        self.assertIn("data_package_summary=data_package_summary", text)

    def test_trainer_waits_on_rank0_visual_status_before_ddp_barrier(self) -> None:
        text = self.trainer_text
        self.assertIn("rank0_stage_status_path", text)
        self.assertIn("write_rank0_stage_status", text)
        self.assertIn("wait_for_rank0_stage_status", text)
        self.assertIn("rank0_stage_sync_timeout", text)
        self.assertIn("accepted_visual_metrics_failed", text)

        visual_block_start = text.index('rank0_stage_status_path(output_dir, "visual_validation", step=step)')
        visual_wait = text.index("rank0_status = wait_for_rank0_stage_status", visual_block_start)
        visual_barrier = text.index("distributed_barrier(runtime)", visual_wait)
        self.assertLess(visual_wait, visual_barrier)

        finalize_block_start = text.index('rank0_stage_status_path(output_dir, "training_finalize")')
        finalize_wait = text.index("rank0_status = wait_for_rank0_stage_status", finalize_block_start)
        finalize_barrier = text.index("distributed_barrier(runtime)", finalize_wait)
        self.assertLess(finalize_wait, finalize_barrier)

    def test_trainer_passes_configured_acceptance_backend_to_visual_hook(self) -> None:
        text = self.trainer_text
        self.assertIn('visual_cfg = config.get("visual_validation", {})', text)
        self.assertIn('acceptance_backend=bool(visual_cfg.get("acceptance_backend", False))', text)
        self.assertIn('isaac_python_bin=visual_cfg.get("isaac_python_bin")', text)
        self.assertIn('isaac_render_script=visual_cfg.get("isaac_render_script")', text)

    def test_trainer_promotes_visual_body_metrics_to_wandb_scalars(self) -> None:
        text = self.trainer_text
        self.assertIn("visual_validation_wandb_payload", text)
        self.assertIn("visual_wandb_payload = visual_validation_wandb_payload(summary, summary_path=report_path)", text)
        self.assertIn("**visual_wandb_payload", text)

    def test_trainer_supports_short_smoke_cli_overrides(self) -> None:
        text = self.trainer_text
        self.assertIn("--max-steps", text)
        self.assertIn("--wandb-mode", text)
        self.assertIn("--disable-visual-validation", text)
        self.assertIn("apply_cli_overrides", text)

    def test_trainer_keeps_external_source_as_metadata_not_runtime_guard(self) -> None:
        text = self.trainer_text
        self.assertIn('"source_revision_actual": git_revision(source_root)', text)
        self.assertIn('"source_status_short": git_status_short(source_root)', text)
        self.assertIn("source_repo_git_commit", text)
        self.assertIn("source_repo_commit", text)
        self.assertNotIn("git_revision(source_root) is None", text)
        self.assertNotIn("git_has_tracked_changes(source_root)", text)
        self.assertNotIn("require_latest_git(source_root", text)


class HistoricalKinSkeletonLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = KIN_SKELETON_LAUNCHER.read_text(encoding="utf-8")

    def test_historical_kin_skeleton_four_by_one_has_no_default_ab_launch(self) -> None:
        text = self.launcher_text
        self.assertIn("ALLOW_HISTORICAL_A_B_4X1GPU", text)
        self.assertIn("kin-skeleton 4x1-GPU launching is historical", text)
        self.assertIn("LR-273 loss-on config", text)
        self.assertIn("LR-274 loss-off baseline config", text)
        self.assertIn("active kin-only SOMA encoder treatment/baseline configs must run as one 4-GPU job", text)


if __name__ == "__main__":
    unittest.main()
