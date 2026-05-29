from __future__ import annotations

import json
from pathlib import Path
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
ACTIVE_BASELINE_DOCS = (
    REPO_ROOT / "docs" / "status" / "online_retarget_code_update_audit_2026-05-25.md",
    REPO_ROOT / "docs" / "status" / "online_retarget_sonic_training_boundary_2026-05-20.md",
    REPO_ROOT / "docs" / "status" / "sonic_native_retarget_contract_2026-05-20.md",
)
MORPHOLOGY_SOURCE = REPO_ROOT / "src" / "online_retarget" / "sonic_morphology.py"
OBSERVATION_TERMS_SOURCE = REPO_ROOT / "src" / "online_retarget" / "sonic_observation_terms.py"


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
        self.assertIn("active kin-only SOMA encoder baselines must run as one 4-GPU job", text)


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

    def test_wrapper_defaults_to_proportional_four_gpu_config(self) -> None:
        text = self.launcher_text
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_4gpu.json", text)
        self.assertIn("remote_start_sonic_kin_soma_motionlib_4gpu.sh", text)


class SupervisedSomaMotionlibFourGpuConfigTests(unittest.TestCase):
    def test_configs_are_strict_supervised_four_gpu_baselines(self) -> None:
        expected = {
            "uniform": "sonic_kin_only_soma_encoder_uniform",
            "proportional": "sonic_kin_only_soma_encoder_proportional",
        }
        for path in SUPERVISED_CONFIGS:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                topology = config["input_data"]["soma_topology"]
                self.assertEqual(config["training_lane"], "soma_motionlib_kin_only")
                self.assertEqual(config["variant"]["name"], expected[topology])
                self.assertEqual(config["variant"]["soma_topology"], topology)
                self.assertEqual(config["runtime"]["required_gpu_count"], 4)
                self.assertEqual(config["training"]["required_gpu_count"], 4)
                self.assertEqual(config["target_decoder"]["primary"], "g1_kin")
                self.assertEqual(config["decoder_targets"], ["g1_kin"])
                self.assertEqual(config["losses"]["primary"], ["reconstruction"])
                self.assertEqual(config["losses"]["auxiliary"], [])
                self.assertEqual(config["model"]["hidden_dims"], [512, 2048, 512])
                self.assertNotIn("hidden_dim", config["model"])
                self.assertNotIn("num_layers", config["model"])
                self.assertEqual(
                    config["training"]["reconstruction_weights"],
                    {"command": 1.0, "root_pos": 0.25, "root_rot": 0.5},
                )
                self.assertIn(f"soma_{topology}_filtered_v1", config["input_data"]["soma_motion_dir"])
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("Isaac", text)
                self.assertNotIn("isaac", text)
                self.assertNotIn("reward", text)
                self.assertNotIn("sonic_hydra", text)
                self.assertNotIn("train_agent_trl.py", text)
                self.assertNotIn("KinematicActionUniversalTokenModule", text)
                self.assertNotIn("g1_dyn", text)
                self.assertNotIn("g1_target_action", text)
                self.assertNotIn("dynamics", text)
                self.assertNotIn("action", text)
                self.assertNotIn("episode_length", text)

    def test_active_baseline_docs_have_no_isaac_wording(self) -> None:
        for path in ACTIVE_BASELINE_DOCS:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("Isaac", text)
                self.assertNotIn("isaac", text)

    def test_num_clusters_comment_clarifies_morphology_bucket_not_actuators(self) -> None:
        phrase = "source skeleton/morphology bucket count, not actuator grouping"
        for path in (MORPHOLOGY_SOURCE, OBSERVATION_TERMS_SOURCE, ACTIVE_BASELINE_DOCS[-1]):
            with self.subTest(path=path.name):
                text = " ".join(path.read_text(encoding="utf-8").split())
                self.assertIn(phrase, text)


class SupervisedSomaMotionlibFourGpuLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = SUPERVISED_DDP_LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_uses_torch_distributed_supervised_entrypoint(self) -> None:
        text = self.launcher_text
        self.assertIn("configs/sonic_kin_soma_motionlib_proportional_4gpu.json", text)
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python3}"', text)
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
        self.assertIn("forbidden rollout-loop tokens", text)
        self.assertIn("NPROC_PER_NODE must match required_gpu_count", text)

    def test_launcher_supports_short_smoke_overrides(self) -> None:
        text = self.launcher_text
        self.assertIn("MAX_STEPS", text)
        self.assertIn("--max-steps", text)
        self.assertIn("WANDB_MODE", text)
        self.assertIn("--wandb-mode", text)
        self.assertIn("DISABLE_VISUAL_VALIDATION", text)
        self.assertIn("--disable-visual-validation", text)

    def test_launcher_preflight_scope_matches_strict_supervised_smoke(self) -> None:
        text = self.launcher_text
        self.assertIn('require_latest_git "${ROOT}" "OnlineRetarget repo"', text)
        self.assertIn("robot_motion_dir", text)
        self.assertIn("soma_motion_dir", text)
        self.assertIn("torch.cuda.is_available", text)
        self.assertIn("torch.distributed.is_available", text)
        self.assertIn("external_source_guard", text)
        self.assertIn("not_required_supervised_entrypoint_no_external_import_exec", text)
        self.assertIn("strict_supervised_soma_motionlib_kin_only_reconstruction", text)
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
        self.assertIn("active kin-only SOMA encoder baselines must run as one 4-GPU job", text)


if __name__ == "__main__":
    unittest.main()
