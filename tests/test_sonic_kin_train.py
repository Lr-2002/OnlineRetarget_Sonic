import unittest

import scripts.train_sonic_kin_skeleton_ae as sonic_train


class SonicKinTrainTimingTests(unittest.TestCase):
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
