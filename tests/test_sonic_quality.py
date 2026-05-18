from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal stdlib envs.
    np = None

from online_retarget.data.bones_sonic import SONIC_BODY_NAMES
from online_retarget.data.sonic_quality import (
    SONIC_BODY_PARENTS,
    SonicQualityConfig,
    scan_sonic_quality_from_index,
    summarize_sonic_arrays,
)


def _synthetic_arrays(frames: int = 6) -> dict[str, object]:
    assert np is not None
    joint_pos = np.zeros((frames, 29), dtype=np.float32)
    joint_vel = np.zeros((frames, 29), dtype=np.float32)
    joint_vel[2, 0] = 30.0
    body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
    body_lin_vel = np.zeros((frames, 30, 3), dtype=np.float32)
    for body_index in range(30):
        body_pos[:, body_index, 0] = body_index * 0.1
        body_pos[:, body_index, 2] = 0.8
    left_foot = SONIC_BODY_NAMES.index("left_ankle_roll_link")
    right_foot = SONIC_BODY_NAMES.index("right_ankle_roll_link")
    body_pos[:, left_foot, 2] = 0.0
    body_pos[:, right_foot, 1] = 0.2
    body_pos[:, right_foot, 2] = 0.0
    body_pos[:, left_foot, 0] = np.arange(frames) * 0.02
    body_lin_vel[:, left_foot, 0] = 1.0
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_lin_vel_w": body_lin_vel,
    }


@unittest.skipUnless(np is not None, "numpy required for SONIC quality tests")
class SonicQualityTest(unittest.TestCase):
    def test_sonic_body_parents_match_isaaclab_g1_tree(self) -> None:
        parent_names = [
            None if parent is None else SONIC_BODY_NAMES[parent]
            for parent in SONIC_BODY_PARENTS
        ]

        self.assertEqual(len(SONIC_BODY_PARENTS), len(SONIC_BODY_NAMES))
        self.assertIsNone(parent_names[SONIC_BODY_NAMES.index("pelvis")])
        self.assertEqual(
            parent_names[SONIC_BODY_NAMES.index("torso_link")],
            "waist_roll_link",
        )
        self.assertEqual(
            parent_names[SONIC_BODY_NAMES.index("left_ankle_roll_link")],
            "left_ankle_pitch_link",
        )
        self.assertEqual(
            parent_names[SONIC_BODY_NAMES.index("right_ankle_roll_link")],
            "right_ankle_pitch_link",
        )
        self.assertEqual(
            parent_names[SONIC_BODY_NAMES.index("left_elbow_link")],
            "left_shoulder_yaw_link",
        )
        self.assertEqual(
            parent_names[SONIC_BODY_NAMES.index("right_wrist_yaw_link")],
            "right_wrist_pitch_link",
        )

    def test_summarize_sonic_arrays_flags_npz_quality(self) -> None:
        assert np is not None
        result = summarize_sonic_arrays(
            {
                "sonic_relative_path": "230101/test__A001.npz",
                "sonic_path": "/tmp/test.npz",
                "filename": "test__A001",
            },
            _synthetic_arrays(),
            fps=50.0,
            config=SonicQualityConfig(enable_body_origin_contact_flags=True),
            np=np,
        )

        self.assertEqual(result["quality_action"], "quarantine")
        self.assertIn("sonic_joint_velocity_jump", result["quality_flags"])
        self.assertIn("sonic_foot_slide", result["quality_flags"])
        self.assertEqual(result["frame_count"], 6)
        self.assertGreater(float(result["max_abs_joint_velocity"]), 20.0)
        self.assertGreater(float(result["contact_slide_rate"]), 0.0)

    def test_scan_sonic_quality_from_index_reads_npz(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "bones_sonic" / "230101" / "test__A001.npz"
            npz_path.parent.mkdir(parents=True)
            arrays = _synthetic_arrays()
            np.savez(
                npz_path,
                fps=np.asarray([50], dtype=np.int64),
                joint_pos=arrays["joint_pos"],
                joint_vel=arrays["joint_vel"],
                body_pos_w=arrays["body_pos_w"],
                body_quat_w=np.zeros((6, 30, 4), dtype=np.float32),
                body_lin_vel_w=arrays["body_lin_vel_w"],
                body_ang_vel_w=np.zeros((6, 30, 3), dtype=np.float32),
            )
            index_csv = root / "sonic_index.csv"
            with index_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sonic_relative_path",
                        "sonic_path",
                        "date",
                        "filename",
                        "actor_uid",
                        "package",
                        "category",
                        "is_mirror",
                        "schema_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sonic_relative_path": "230101/test__A001.npz",
                        "sonic_path": str(npz_path),
                        "date": "230101",
                        "filename": "test__A001",
                        "actor_uid": "A001",
                        "package": "Locomotion",
                        "category": "Baseline",
                        "is_mirror": "False",
                        "schema_status": "ok",
                    }
                )

            result = scan_sonic_quality_from_index(
                index_csv=index_csv,
                output_root=root / "runs",
                config=SonicQualityConfig(enable_body_origin_contact_flags=True),
                limit=1,
            )

            self.assertEqual(result.scanned_rows, 1)
            self.assertEqual(result.flag_counts["sonic_joint_velocity_jump"], 1)
            self.assertTrue(result.report_json.exists())
            with result.stats_jsonl.open(encoding="utf-8") as handle:
                line = handle.readline()
            self.assertIn("sonic_foot_slide", line)


if __name__ == "__main__":
    unittest.main()
