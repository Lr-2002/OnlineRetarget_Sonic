from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile

from online_retarget import web_pipeline
from online_retarget.web_pipeline import _adapt_soma_bvh_for_gmr, run_web_pipeline


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
            self.assertEqual(ground_alignment["mode"], "sequence_foot_geom_min_z")
            self.assertEqual(ground_alignment["frames"], 3)
            self.assertEqual(ground_alignment["post_min_foot_z"], 0.0)
            self.assertEqual(ground_alignment["root_z_offset_delta_abs_max"], 0.0)
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

    def test_soma_gmr_bvh_adapter_promotes_hips_and_drops_dummy_root_channels(self):
        adapted, report = _adapt_soma_bvh_for_gmr(_soma_gmr_bvh_text())
        motion_lines = adapted.splitlines()
        frame_line = motion_lines[-1]

        self.assertTrue(report["applied"])
        self.assertEqual(report["dropped_dummy_root_channels"], 6)
        self.assertEqual(report["dummy_root_nonzero_frames"], 0)
        self.assertEqual(report["aliases"]["LeftLeg"], "LeftUpLeg")
        self.assertEqual(report["aliases"]["LeftShin"], "LeftLeg")
        self.assertIn("ROOT Hips", adapted)
        self.assertNotIn("ROOT Root", adapted)
        self.assertIn("JOINT LeftUpLeg", adapted)
        self.assertIn("JOINT RightUpLeg", adapted)
        self.assertNotIn("JOINT LeftShin", adapted)
        self.assertNotIn("JOINT RightShin", adapted)
        self.assertEqual(len(frame_line.split()), report["output_channels"])
        self.assertEqual(frame_line.split()[:6], ["1.000000", "2.000000", "3.000000", "10.000000", "20.000000", "30.000000"])

    def test_raw_soma_bvh_web_pipeline_uses_gmr_adapter_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            observations: dict[str, object] = {}
            original_root = web_pipeline.DEFAULT_GMR_ROOT
            fake_modules = _install_fake_gmr_modules(observations)
            try:
                web_pipeline.DEFAULT_GMR_ROOT = tmp_path / "gmr"
                web_pipeline.DEFAULT_GMR_ROOT.mkdir()
                result = run_web_pipeline(
                    _soma_gmr_bvh_text().encode("utf-8"),
                    "jump_sideway_135_001__A023.bvh",
                    output_root=tmp_path / "web_runs",
                    model_xml=tmp_path / "missing_g1.xml",
                    max_frames=0,
                )
            finally:
                web_pipeline.DEFAULT_GMR_ROOT = original_root
                _restore_modules(fake_modules)

            payload = result.to_dict()
            report = json.loads((result.output_dir / "retarget_report.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["stages"]["retarget"]["status"], "ok")
        self.assertEqual(report["selected_retargeter"], "gmr")
        self.assertNotEqual(report["selected_retargeter"], "rule_based_preview")
        self.assertEqual(report["src_human"], "bvh_nokov")
        self.assertEqual(report["gmr_config_source"], "bvh_nokov")
        self.assertEqual(report["frames"], 1)
        self.assertAlmostEqual(report["source_fps"], 120.0)
        self.assertTrue(report["gmr_bvh_adapter"]["applied"])
        self.assertEqual(observations["bvh_format"], "nokov")
        self.assertNotIn("ROOT Root", str(observations["adapted_bvh_text"]))
        self.assertIn("ROOT Hips", str(observations["adapted_bvh_text"]))
        self.assertIn("JOINT LeftUpLeg", str(observations["adapted_bvh_text"]))
        self.assertIn("LeftFootMod", observations["frame_keys"])


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


def _soma_gmr_bvh_text() -> str:
    row = [0.0] * 6
    row += [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    row += [0.0] * (15 * 3)
    frame = " ".join(f"{value:.6f}" for value in row)
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips
  {{
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Spine2
    {{
      OFFSET 0.000000 15.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }}
    JOINT LeftLeg
    {{
      OFFSET -8.000000 -10.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
      JOINT LeftShin
      {{
        OFFSET 0.000000 -38.000000 0.000000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftFoot
        {{
          OFFSET 0.000000 -38.000000 0.000000
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT LeftToeBase
          {{
            OFFSET 0.000000 -8.000000 12.000000
            CHANNELS 3 Zrotation Yrotation Xrotation
          }}
        }}
      }}
    }}
    JOINT RightLeg
    {{
      OFFSET 8.000000 -10.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
      JOINT RightShin
      {{
        OFFSET 0.000000 -38.000000 0.000000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightFoot
        {{
          OFFSET 0.000000 -38.000000 0.000000
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT RightToeBase
          {{
            OFFSET 0.000000 -8.000000 12.000000
            CHANNELS 3 Zrotation Yrotation Xrotation
          }}
        }}
      }}
    }}
    JOINT LeftArm
    {{
      OFFSET -15.000000 10.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
      JOINT LeftForeArm
      {{
        OFFSET -25.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftHand
        {{
          OFFSET -20.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Yrotation Xrotation
        }}
      }}
    }}
    JOINT RightArm
    {{
      OFFSET 15.000000 10.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
      JOINT RightForeArm
      {{
        OFFSET 25.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightHand
        {{
          OFFSET 20.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Yrotation Xrotation
        }}
      }}
    }}
  }}
}}
MOTION
Frames: 1
Frame Time: 0.008333333333333333
{frame}
"""


def _install_fake_gmr_modules(observations: dict[str, object]) -> dict[str, types.ModuleType | None]:
    module_names = [
        "numpy",
        "general_motion_retargeting",
        "general_motion_retargeting.utils",
        "general_motion_retargeting.utils.lafan1",
        "general_motion_retargeting.motion_retarget",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}
    numpy = types.ModuleType("numpy")
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = []  # type: ignore[attr-defined]
    utils_package = types.ModuleType("general_motion_retargeting.utils")
    utils_package.__path__ = []  # type: ignore[attr-defined]
    lafan1 = types.ModuleType("general_motion_retargeting.utils.lafan1")
    motion_retarget = types.ModuleType("general_motion_retargeting.motion_retarget")

    def load_bvh_file(path: str, format: str = "lafan1"):
        text = Path(path).read_text(encoding="utf-8")
        observations["bvh_format"] = format
        observations["adapted_bvh_text"] = text
        frame = {
            key: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            for key in (
                "Hips",
                "Spine2",
                "LeftUpLeg",
                "LeftLeg",
                "LeftFoot",
                "LeftToeBase",
                "LeftFootMod",
                "RightUpLeg",
                "RightLeg",
                "RightFoot",
                "RightToeBase",
                "RightFootMod",
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
                "RightArm",
                "RightForeArm",
                "RightHand",
            )
        }
        return [frame], 1.75

    class FakeGeneralMotionRetargeting:
        def __init__(self, src_human: str, tgt_robot: str, actual_human_height: float, verbose: bool):
            observations["src_human"] = src_human
            observations["tgt_robot"] = tgt_robot
            observations["actual_human_height"] = actual_human_height
            observations["verbose"] = verbose

        def retarget(self, human_frame: dict[str, object]) -> list[float]:
            observations["frame_keys"] = set(human_frame)
            return [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0] + [0.0] * len(web_pipeline.G1_JOINT_COLUMNS)

    class FakeArray(list):
        def reshape(self, *_shape: object) -> "FakeArray":
            return self

    def asarray(values: object, dtype: object = None) -> FakeArray:
        return FakeArray(values)  # type: ignore[arg-type]

    numpy.asarray = asarray  # type: ignore[attr-defined]
    lafan1.load_bvh_file = load_bvh_file  # type: ignore[attr-defined]
    motion_retarget.GeneralMotionRetargeting = FakeGeneralMotionRetargeting  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy
    sys.modules["general_motion_retargeting"] = package
    sys.modules["general_motion_retargeting.utils"] = utils_package
    sys.modules["general_motion_retargeting.utils.lafan1"] = lafan1
    sys.modules["general_motion_retargeting.motion_retarget"] = motion_retarget
    return saved


def _restore_modules(saved: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


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
