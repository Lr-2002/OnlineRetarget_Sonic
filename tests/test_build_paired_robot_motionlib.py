import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_paired_robot_motionlib.py"


@unittest.skipUnless(importlib.util.find_spec("joblib"), "joblib is required for motionlib metadata")
class BuildPairedRobotMotionlibTests(unittest.TestCase):
    def test_filters_robot_metadata_to_reference_pkl_keys(self):
        joblib = importlib.import_module("joblib")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robot = root / "robot"
            reference = root / "soma"
            output = root / "paired"
            robot.mkdir()
            reference.mkdir()
            joblib.dump(
                {
                    "keep_a": {"num_frames": 10},
                    "keep_b": {"num_frames": 20},
                    "drop_c": {"num_frames": 30},
                },
                robot / "metadata.pkl",
            )
            (robot / "keep_a.pkl").write_bytes(b"robot-a")
            (robot / "keep_b.pkl").write_bytes(b"robot-b")
            (robot / "drop_c.pkl").write_bytes(b"robot-c")
            (reference / "keep_a.pkl").write_bytes(b"not-read")
            (reference / "keep_b.pkl").write_bytes(b"not-read")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--robot-motion-dir",
                    str(robot),
                    "--reference-motion-dir",
                    str(reference),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn('"paired_key_count": 2', completed.stdout)
            paired = joblib.load(output / "metadata.pkl")
            self.assertEqual(sorted(paired), ["keep_a", "keep_b"])
            self.assertEqual(
                (output / "keys.txt").read_text(encoding="utf-8").splitlines(),
                ["keep_a", "keep_b"],
            )
            self.assertTrue((output / "keep_a.pkl").is_symlink())
            self.assertTrue((output / "keep_b.pkl").is_symlink())
            self.assertFalse((output / "drop_c.pkl").exists())

    def test_rejects_metadata_keys_without_source_motion_files(self):
        joblib = importlib.import_module("joblib")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robot = root / "robot"
            reference = root / "soma"
            output = root / "paired"
            robot.mkdir()
            reference.mkdir()
            joblib.dump({"clip": {"num_frames": 10}}, robot / "metadata.pkl")
            (reference / "clip.pkl").write_bytes(b"not-read")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--robot-motion-dir",
                    str(robot),
                    "--reference-motion-dir",
                    str(reference),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("paired robot motion files missing", completed.stderr)


if __name__ == "__main__":
    unittest.main()
