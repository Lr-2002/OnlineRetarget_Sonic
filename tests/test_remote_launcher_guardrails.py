from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "remote_start_sonic_native_retarget_4x1gpu.sh"


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


if __name__ == "__main__":
    unittest.main()
