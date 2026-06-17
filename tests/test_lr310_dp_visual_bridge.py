import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import rerender_lr310_dp_visual_validation as bridge


class LR310DPVisualBridgeTests(unittest.TestCase):
    def test_body_root_prediction_row_writes_accepted_motion_npz_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bvh = root / "clip.bvh"
            source_bvh.write_text("HIERARCHY\n", encoding="utf-8")
            target_npz = root / "target.npz"
            body_pos = np.zeros((3, 30, 3), dtype=np.float32)
            body_pos[:, 0, :] = np.asarray(
                [[0.0, 0.0, 0.8], [0.1, 0.0, 0.8], [0.2, 0.0, 0.8]],
                dtype=np.float32,
            )
            body_quat = np.zeros((3, 30, 4), dtype=np.float32)
            body_quat[:, :, 0] = 1.0
            target_joints = np.asarray([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]], dtype=np.float32)
            np.savez(
                target_npz,
                body_pos_w=body_pos,
                body_quat_w=body_quat,
                joint_pos=target_joints,
                fps=np.asarray([50.0], dtype=np.float32),
            )
            row = {
                "sample_id": "walk/probe",
                "source_motion_path": "soma_proportional/bvh/clip.bvh",
                "target_g1_path": str(target_npz),
                "target_joint_names": ["left", "right"],
                "predicted_joints": [[3.0, 4.0], [3.1, 4.1], [3.2, 4.2]],
            }

            result = bridge.rerender_prediction_row(
                row=row,
                index=0,
                predictions_jsonl=root / "predictions.jsonl",
                output_dir=root / "visual_validation",
                config={},
                target_g1_roots=[],
                step=123,
                execute_renderers=False,
                root_source="body_root",
                root_body_index=0,
                root_body_name="pelvis",
                root_quat_format="wxyz",
                allow_root_fixed_fallback=False,
                source_bvh_resolver=lambda _row, _config, _output_dir: source_bvh,
            )

            self.assertFalse(result["acceptance_ok"])
            self.assertFalse(result["root_fixed_fallback"])
            row2_npz = Path(result["target_motion_npz"])
            row3_npz = Path(result["prediction_motion_npz"])
            with np.load(row2_npz) as loaded:
                np.testing.assert_allclose(loaded["root_pos"], body_pos[:, 0, :])
                np.testing.assert_allclose(loaded["root_quat"], body_quat[:, 0, :])
                np.testing.assert_allclose(loaded["joint_pos"], target_joints)
                self.assertEqual([str(name) for name in loaded["joint_names"]], ["left", "right"])
            with np.load(row3_npz) as loaded:
                np.testing.assert_allclose(loaded["root_pos"], body_pos[:, 0, :])
                np.testing.assert_allclose(loaded["joint_pos"], np.asarray(row["predicted_joints"], dtype=np.float32))

            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            bridge_meta = metadata["lr310_dp_prediction_bridge"]
            self.assertEqual(bridge_meta["root_pose"]["root_pose_source"], "body_root")
            self.assertEqual(
                bridge_meta["prediction_root_pose"]["prediction_root_pose_source"],
                "target_root_pose_reused",
            )
            self.assertIn("target-root reuse", bridge_meta["lr290_contract_parity_note"])
            self.assertEqual(metadata["visual_backend"]["accepted_vertical_v2_status"], "failed")
            self.assertEqual(
                Path(result["combined_video"]).name,
                "probe__step_00000123__vertical_somamesh_g1target_g1kinematics.mp4",
            )
            self.assertEqual(
                Path(result["row1_soma_somamesh_video"]).name,
                "probe__step_00000123__row1_soma_somamesh.mp4",
            )
            self.assertEqual(
                Path(result["row2_g1_target_isaaclab_video"]).name,
                "probe__step_00000123__row2_g1_target_isaaclab.mp4",
            )
            self.assertEqual(
                Path(result["row3_g1_kinematics_isaaclab_video"]).name,
                "probe__step_00000123__row3_g1_kinematics_isaaclab.mp4",
            )

    def test_noncontiguous_target_frame_indices_align_joint_and_root_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bvh = root / "clip.bvh"
            source_bvh.write_text("HIERARCHY\n", encoding="utf-8")
            target_npz = root / "target.npz"
            frames = 6
            joint_pos = np.asarray([[frame, frame + 0.5] for frame in range(frames)], dtype=np.float32)
            root_pos = np.asarray(
                [[frame, frame + 10.0, frame + 20.0] for frame in range(frames)],
                dtype=np.float32,
            )
            root_quat = np.zeros((frames, 4), dtype=np.float32)
            for frame in range(frames):
                theta = frame * 0.1
                root_quat[frame] = [np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0]
            body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
            body_pos[:, 0, :] = root_pos
            body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
            body_quat[:, 0, :] = root_quat
            np.savez(
                target_npz,
                joint_pos=joint_pos,
                root_pos=root_pos,
                root_quat=root_quat,
                body_pos_w=body_pos,
                body_quat_w=body_quat,
                fps=np.asarray([50.0], dtype=np.float32),
            )
            row = {
                "sample_id": "windowed",
                "source_motion_path": "soma_proportional/bvh/clip.bvh",
                "target_g1_path": str(target_npz),
                "target_joint_names": ["left", "right"],
                "target_frame_indices": [4, 1, 5],
                "predicted_joints": [[30.0, 40.0], [31.0, 41.0], [32.0, 42.0]],
            }

            result = bridge.rerender_prediction_row(
                row=row,
                index=0,
                predictions_jsonl=root / "predictions.jsonl",
                output_dir=root / "visual_validation",
                config={},
                target_g1_roots=[],
                step=123,
                execute_renderers=False,
                root_source="target_npz_root",
                root_body_index=0,
                root_body_name="pelvis",
                root_quat_format="wxyz",
                allow_root_fixed_fallback=False,
                source_bvh_resolver=lambda _row, _config, _output_dir: source_bvh,
            )

            selected = np.asarray([4, 1, 5], dtype=np.int64)
            with np.load(Path(result["target_motion_npz"])) as loaded:
                np.testing.assert_allclose(loaded["joint_pos"], joint_pos[selected])
                np.testing.assert_allclose(loaded["root_pos"], root_pos[selected])
                np.testing.assert_allclose(loaded["root_quat"], root_quat[selected])
            with np.load(Path(result["prediction_motion_npz"])) as loaded:
                np.testing.assert_allclose(loaded["root_pos"], root_pos[selected])
                np.testing.assert_allclose(loaded["root_quat"], root_quat[selected])
            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            bridge_meta = metadata["lr310_dp_prediction_bridge"]
            self.assertEqual(bridge_meta["target_frame_indices"], [4, 1, 5])
            self.assertEqual(bridge_meta["root_pose"]["root_pose_source"], "target_npz_root")
            self.assertEqual(
                bridge_meta["prediction_root_pose"]["prediction_root_pose_source"],
                "target_root_pose_reused",
            )

    def test_target_frame_indices_validate_npz_range(self) -> None:
        arrays = {
            "joint_pos": np.zeros((2, 2), dtype=np.float32),
            "root_pos": np.zeros((2, 3), dtype=np.float32),
            "root_quat": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "out of range"):
            bridge.root_pose_from_target_arrays(
                arrays,
                frame_indices=np.asarray([0, 2], dtype=np.int64),
                root_source="target_npz_root",
                root_body_index=0,
                root_body_name="pelvis",
                root_quat_format="wxyz",
                allow_root_fixed_fallback=False,
            )

    def test_auto_root_requires_explicit_root_fixed_fallback_marker(self) -> None:
        arrays = {"joint_pos": np.zeros((2, 2), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "fixed-root"):
            bridge.root_pose_from_target_arrays(
                arrays,
                frame_indices=np.asarray([0, 1], dtype=np.int64),
                root_source="auto",
                root_body_index=0,
                root_body_name="pelvis",
                root_quat_format="wxyz",
                allow_root_fixed_fallback=False,
            )

        root_pos, root_quat, semantics = bridge.root_pose_from_target_arrays(
            arrays,
            frame_indices=np.asarray([0, 1], dtype=np.int64),
            root_source="auto",
            root_body_index=0,
            root_body_name="pelvis",
            root_quat_format="wxyz",
            allow_root_fixed_fallback=True,
        )

        np.testing.assert_allclose(root_pos, np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_allclose(root_quat, np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32))
        self.assertTrue(semantics["root_fixed_fallback"])
        self.assertIn("not LR-290", semantics["root_semantics"])


if __name__ == "__main__":
    unittest.main()
