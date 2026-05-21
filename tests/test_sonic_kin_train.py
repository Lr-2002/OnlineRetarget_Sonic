import unittest

import numpy as np

import scripts.train_sonic_kin_skeleton_ae as sonic_train


class SonicKinTrainTimingTests(unittest.TestCase):
    def test_robot_root_rot_to_wxyz_converts_gmr_xyzw_motionlib_quat(self):
        root_rot_xyzw = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        converted = sonic_train.robot_root_rot_to_wxyz(
            root_rot_xyzw,
            {"input_data": {"robot_root_rot_format": "xyzw"}},
        )

        np.testing.assert_allclose(converted, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    def test_time_align_frame_maps_maps_target_time_to_source_time(self):
        frames = [{"Hips": (float(index), 0.0, 0.0)} for index in range(20)]

        aligned, indices = sonic_train._time_align_frame_maps(
            frames,
            source_fps=120.0,
            target_fps=50.0,
            frame_count=5,
        )

        self.assertEqual(indices, [0, 2, 4, 7, 9])
        self.assertEqual([frame["Hips"][0] for frame in aligned], [0.0, 2.0, 4.0, 7.0, 9.0])

    def test_source_target_timing_summary_accepts_120hz_source_to_50hz_target(self):
        summary = sonic_train.source_target_timing_summary(
            {"move_duration_frames": "120", "fps": "50"},
            frame_count=50,
            indexing={"source_fps": 120.0, "max_duration_delta_sec": 0.02},
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["flags"], [])

    def test_source_target_timing_summary_rejects_target_that_is_not_slower(self):
        summary = sonic_train.source_target_timing_summary(
            {"move_duration_frames": "50", "fps": "120"},
            frame_count=50,
            indexing={"source_fps": 120.0, "max_duration_delta_sec": 0.02},
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertIn("target_fps_not_below_source_fps", summary["flags"])

    def test_soma_resample_matches_sonic_target_timeline_length(self):
        source = np.arange(330, dtype=np.float32).reshape(330, 1)

        resampled = sonic_train.resample_soma_array(source, 120.0, 50.0)

        self.assertEqual(resampled.shape, (138, 1))
        self.assertEqual(float(resampled[0, 0]), 0.0)
        self.assertGreater(float(resampled[-1, 0]), 320.0)

    def test_soma_motionlib_feature_builder_uses_soma_source_and_g1_kin_target(self):
        frames = 8
        soma_joints = np.zeros((frames, 26, 3), dtype=np.float32)
        soma_joints[:, :, 0] = np.arange(26, dtype=np.float32)
        soma_joints[:, :, 1] = np.arange(frames, dtype=np.float32)[:, None]
        identity = np.zeros((frames, 4), dtype=np.float32)
        identity[:, 0] = 1.0
        dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.01
        arrays = {
            "soma_joints": soma_joints,
            "soma_root_quat": identity.copy(),
            "joint_pos": dof,
            "joint_vel": sonic_train.finite_difference_velocity(dof, 50.0),
            "root_rot": identity.copy(),
        }

        motion, skeleton, target = sonic_train.build_soma_motionlib_features(
            arrays,
            np.asarray([0, 1], dtype=np.int64),
            window=3,
            step=1,
        )

        self.assertEqual(motion.shape, (2, 3 * 26 * 3 + 3 * 6))
        self.assertEqual(skeleton.shape, (2, 26 * 3 + 26))
        self.assertEqual(target.shape, (2, 3 * (29 + 29) + 3 * 6))
        np.testing.assert_allclose(target[0, :29], dof[0])
