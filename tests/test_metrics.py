import math
import unittest

from online_retarget.metrics import (
    action_similarity,
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


if __name__ == "__main__":
    unittest.main()
