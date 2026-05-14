import math
import unittest

from online_retarget.metrics import (
    action_similarity,
    contact_artifact_metrics,
    joint_jump_rate,
    joint_limit_violation_rate,
    joint_rmse,
    mpjpe,
)


class MetricTests(unittest.TestCase):
    def test_mpjpe(self):
        pred = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        target = [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        self.assertEqual(mpjpe(pred, target), 0.5)

    def test_joint_rmse(self):
        self.assertTrue(math.isclose(joint_rmse([[1.0, 3.0]], [[1.0, 1.0]]), math.sqrt(2.0)))

    def test_action_similarity(self):
        self.assertEqual(action_similarity([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]), 1.0)

    def test_joint_jump_rate(self):
        positions = [[0.0, 0.0], [0.2, 0.01]]
        self.assertEqual(joint_jump_rate(positions, fps=10.0, max_velocity=1.0), 0.5)

    def test_joint_limit_violation_rate(self):
        positions = [[0.0, 2.0], [-2.0, 0.0]]
        self.assertEqual(joint_limit_violation_rate(positions, [-1.0, -1.0], [1.0, 1.0]), 0.5)

    def test_contact_artifact_metrics_use_reference_contact_frames(self):
        target = [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.2]],
        ]
        predicted = [
            [[0.0, 0.0, 0.08]],
            [[0.2, 0.0, 0.08]],
            [[0.2, 0.0, -0.01]],
        ]

        metrics = contact_artifact_metrics(
            predicted,
            fps=10.0,
            foot_indices=(0,),
            contact_reference=target,
            contact_height_threshold=0.04,
            max_contact_slide_speed=0.25,
        )

        self.assertEqual(metrics["contact_frame_ratio"], 2 / 3)
        self.assertEqual(metrics["foot_float_rate"], 1.0)
        self.assertEqual(metrics["contact_slide_rate"], 1.0)
        self.assertEqual(metrics["max_contact_slide_speed"], 2.0)
        self.assertEqual(metrics["contact_skate_rate"], 1.0)
        self.assertEqual(metrics["max_contact_skate_distance"], 0.2)
        self.assertEqual(metrics["ground_penetration_rate"], 1 / 3)
        self.assertEqual(metrics["penetration_depth"], 0.01)

    def test_contact_artifact_metrics_separates_slow_skate_from_speed_slide(self):
        target = [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
        predicted = [
            [[0.0, 0.0, 0.0]],
            [[0.015, 0.0, 0.0]],
            [[0.03, 0.0, 0.0]],
        ]

        metrics = contact_artifact_metrics(
            predicted,
            fps=1.0,
            foot_indices=(0,),
            contact_reference=target,
            contact_height_threshold=0.04,
            max_contact_slide_speed=0.25,
            max_contact_skate_distance=0.02,
        )

        self.assertEqual(metrics["contact_slide_rate"], 0.0)
        self.assertEqual(metrics["max_contact_slide_speed"], 0.015)
        self.assertEqual(metrics["contact_skate_rate"], 1.0)
        self.assertEqual(metrics["max_contact_skate_distance"], 0.03)


if __name__ == "__main__":
    unittest.main()
