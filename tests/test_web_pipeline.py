import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from online_retarget.web_pipeline import run_web_pipeline


class WebPipelineTests(unittest.TestCase):
    def test_bvh_pipeline_generates_retarget_and_kinematic_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            result = run_web_pipeline(
                _bvh_text().encode("utf-8"),
                "walk.bvh",
                output_root=Path(tmp) / "web_runs",
                model_xml=model_xml,
                max_frames=3,
            )

            payload = result.to_dict()
            retarget_csv = Path(payload["artifacts"]["retargeted_g1_csv"])
            retarget_csv_exists = retarget_csv.exists()
            physics = payload["stages"]["physics_sim"]
            report = json.loads((result.output_dir / "pipeline_result.json").read_text())

        self.assertEqual(payload["input_format"], "bvh")
        self.assertEqual(payload["stages"]["load"]["status"], "ok")
        self.assertEqual(payload["stages"]["retarget"]["status"], "ok")
        self.assertEqual(payload["stages"]["kinematic_sim"]["status"], "ok")
        self.assertIn(payload["stages"]["physics_sim"]["status"], {"ok", "blocked", "failed"})
        self.assertTrue(retarget_csv_exists)
        self.assertEqual(len(payload["preview"]["source"]["frames"]), 3)
        self.assertEqual(len(payload["preview"]["robot"]["frames"]), 3)
        self.assertEqual(report["run_id"], payload["run_id"])
        if physics["status"] == "ok":
            video_path = Path(payload["artifacts"]["mujoco_g1_render_mp4"])
            self.assertTrue(video_path.exists())
            self.assertGreater(video_path.stat().st_size, 0)
            self.assertTrue(physics["details"]["rendered_by_mujoco"])
            ground_alignment = physics["details"]["ground_alignment"]
            self.assertTrue(ground_alignment["applied"])
            self.assertEqual(ground_alignment["frames"], 3)
            self.assertEqual(ground_alignment["post_min_foot_z"], 0.0)
            self.assertTrue(physics["details"]["render"]["root_z_aligned_to_ground"])

    def test_smpl_like_npz_pipeline_generates_preview_when_numpy_is_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "motion.npz"
            np.savez(
                npz,
                poses=np.zeros((3, 72), dtype=float),
                trans=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float),
                mocap_framerate=np.array([30.0], dtype=float),
            )
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            result = run_web_pipeline(
                npz.read_bytes(),
                "motion.npz",
                output_root=Path(tmp) / "web_runs",
                model_xml=model_xml,
                max_frames=3,
                render_frames=True,
            )
            payload = result.to_dict()

        self.assertEqual(payload["input_format"], "smpl")
        self.assertEqual(payload["stages"]["load"]["status"], "ok")
        self.assertEqual(payload["stages"]["retarget"]["status"], "ok")
        self.assertEqual(payload["stages"]["kinematic_sim"]["status"], "ok")
        self.assertEqual(payload["stages"]["load"]["details"]["pose_key"], "poses")
        self.assertEqual(len(payload["preview"]["source"]["frames"]), 3)
        if payload["stages"]["physics_sim"]["status"] == "ok":
            self.assertTrue(payload["stages"]["physics_sim"]["details"]["render_frames"])

    def test_malformed_smpl_like_upload_fails_before_retarget(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken_npz = _broken_npz_bytes()
            result = run_web_pipeline(
                broken_npz,
                "motion.npz",
                output_root=Path(tmp) / "web_runs",
                model_xml=Path(tmp) / "missing.xml",
            )
            payload = result.to_dict()

        self.assertEqual(payload["input_format"], "smpl")
        self.assertIn(payload["stages"]["load"]["status"], {"failed", "blocked"})
        self.assertEqual(payload["stages"]["retarget"]["status"], "blocked")

def _bvh_text() -> str:
    return """HIERARCHY
ROOT Root
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips
  {
    OFFSET 0.000000 1.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftFoot
    {
      OFFSET 0.000000 -1.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
    JOINT RightFoot
    {
      OFFSET 0.000000 -1.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
    JOINT LeftHand
    {
      OFFSET -1.000000 0.500000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
    JOINT RightHand
    {
      OFFSET 1.000000 0.500000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.100000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 2.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""


def _minimal_g1_mjcf() -> str:
    return """<mujoco model="minimal_g1">
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <freejoint name="pelvis"/>
      <body name="torso_link" pos="0 0 0.5">
        <body name="head_link" pos="0 0 0.2"/>
      </body>
      <body name="left_ankle_roll_link" pos="0 0.1 -0.6">
        <joint name="left_hip_pitch_joint" axis="0 1 0" range="-1 1"/>
        <geom pos="0 0 0"/>
        <body name="left_toe_link" pos="0.1 0 0"/>
      </body>
      <body name="right_ankle_roll_link" pos="0 -0.1 -0.6">
        <geom pos="0 0 0"/>
        <body name="right_toe_link" pos="0.1 0 0"/>
      </body>
      <body name="left_rubber_hand" pos="0 0.4 0.2"/>
      <body name="right_rubber_hand" pos="0 -0.4 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _broken_npz_bytes() -> bytes:
    path = Path(tempfile.mkdtemp()) / "broken.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "not an npy array")
    return path.read_bytes()


if __name__ == "__main__":
    unittest.main()
