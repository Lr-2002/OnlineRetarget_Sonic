from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHER = REPO_ROOT / "scripts" / "watch_sonic_native_retarget_20k_validation.sh"


class SonicNative20kWatcherTests(unittest.TestCase):
    def _make_layout(self, *, step_dir: str = "step_00020000") -> tuple[Path, Path, Path]:
        tmp = Path(tempfile.mkdtemp(prefix="online-retarget-watcher-test-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = tmp / "root"
        sonic_root = tmp / "sonic"
        run_root = root / "outputs" / "sonic_native_retarget_runs" / "test_group"
        (run_root / "_launcher").mkdir(parents=True)
        (run_root / "_launcher" / "A.log").write_text("", encoding="utf-8")
        sonic_root.mkdir(parents=True)

        for run_index in range(4):
            (
                run_root
                / f"run_{run_index}"
                / "online_retarget_visual_validation"
                / step_dir
            ).mkdir(parents=True)
        return root, sonic_root, run_root

    def _write_complete_artifacts(
        self,
        run_root: Path,
        *,
        step_dir: str = "step_00020000",
        status: str = "ok",
        videos_uploaded: int = 8,
        mp4s_per_run: int = 8,
    ) -> None:
        for run_index in range(4):
            val_dir = (
                run_root
                / f"run_{run_index}"
                / "online_retarget_visual_validation"
                / step_dir
            )
            val_dir.mkdir(parents=True, exist_ok=True)
            for clip_index in range(mp4s_per_run):
                (val_dir / f"clip_{run_index}_{clip_index}.mp4").write_bytes(b"")
            report = {
                "online_retarget_visual_validation/wandb_upload_status": status,
                "online_retarget_visual_validation/videos_uploaded": videos_uploaded,
            }
            (val_dir / "main_upload_report.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )

    def _run_watcher(
        self,
        root: Path,
        sonic_root: Path,
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(root),
                "SONIC_ROOT": str(sonic_root),
                "RETARGET_RUN_GROUP": "test_group",
                "WATCH_INTERVAL_SECONDS": "60",
                "EXPECTED_UPLOAD_REPORTS": "4",
                "EXPECTED_MP4_COUNT": "32",
            }
        )
        return subprocess.run(
            ["bash", str(WATCHER)],
            check=False,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    def test_ready_only_when_20k_artifacts_and_uploads_are_complete(self) -> None:
        root, sonic_root, run_root = self._make_layout()
        self._write_complete_artifacts(run_root)

        result = self._run_watcher(root, sonic_root, timeout=10)

        self.assertEqual(result.returncode, 0, result.stderr)
        ready = run_root / "_monitor" / "validation_20k_ready.md"
        self.assertTrue(ready.exists())
        content = ready.read_text(encoding="utf-8")
        self.assertIn("validation_step_dir: `step_00020000`", content)
        self.assertIn("mp4_count: `32`", content)
        self.assertIn("wandb_upload_ok: `4`", content)
        self.assertIn("wandb_videos_uploaded_total: `32`", content)

    def test_complete_non_20k_artifacts_do_not_mark_20k_ready(self) -> None:
        root, sonic_root, run_root = self._make_layout(step_dir="step_00040000")
        self._write_complete_artifacts(run_root, step_dir="step_00040000")

        with self.assertRaises(subprocess.TimeoutExpired):
            self._run_watcher(root, sonic_root, timeout=1)
        self.assertFalse((run_root / "_monitor" / "validation_20k_ready.md").exists())

    def test_failed_upload_report_does_not_mark_ready(self) -> None:
        root, sonic_root, run_root = self._make_layout()
        self._write_complete_artifacts(run_root, status="failed")

        with self.assertRaises(subprocess.TimeoutExpired):
            self._run_watcher(root, sonic_root, timeout=1)
        self.assertFalse((run_root / "_monitor" / "validation_20k_ready.md").exists())


if __name__ == "__main__":
    unittest.main()
