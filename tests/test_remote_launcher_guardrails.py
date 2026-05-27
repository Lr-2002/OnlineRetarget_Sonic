from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_native_retarget_4x1gpu.sh"
DDP_LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_native_retarget_4gpu.sh"


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


class NativeRetargetFourGpuLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = DDP_LAUNCHER.read_text(encoding="utf-8")

    def test_single_config_multi_gpu_launcher_uses_accelerate_processes(self) -> None:
        text = self.launcher_text
        self.assertIn('CONFIG="${CONFIG:-configs/sonic_native_retarget_a1_concat_1gpu.json}"', text)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-4}"', text)
        self.assertIn('--num_processes="${NPROC_PER_NODE}"', text)
        self.assertNotIn("--num_processes=1 gear_sonic/train_agent_trl.py", text)

    def test_single_config_launcher_rejects_multiple_configs(self) -> None:
        text = self.launcher_text
        self.assertIn("CONFIG must name exactly one formal config", text)
        self.assertIn('if [[ "${CONFIG}" == *" "* ]]; then', text)

    def test_single_config_launcher_preserves_training_guardrails(self) -> None:
        text = self.launcher_text
        self.assertIn("OnlineRetarget repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git "${ROOT}" "OnlineRetarget repo"', text)
        self.assertIn("SONIC source repo has uncommitted tracked changes", text)
        self.assertIn('require_latest_git_if_configured "${SONIC_ROOT}" "SONIC source repo"', text)
        self.assertIn("ONLINE_RETARGET_GIT_SHA", text)
        self.assertIn("SONIC_GIT_SHA", text)

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


if __name__ == "__main__":
    unittest.main()
