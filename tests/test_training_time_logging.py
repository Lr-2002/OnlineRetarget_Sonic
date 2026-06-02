import math
import unittest

import scripts.train as train_entry
from online_retarget.metrics import DEFAULT_ONLINE_METRIC_NAMES, compute_online_metrics


class TrainingTimeLoggingTests(unittest.TestCase):
    def test_online_metric_names_use_config_or_default_registry_names(self):
        self.assertEqual(train_entry._online_metric_names({}), DEFAULT_ONLINE_METRIC_NAMES)
        self.assertEqual(
            train_entry._online_metric_names(
                {"evaluation": {"online_metrics": ["loss", "joint_rotation_error"]}}
            ),
            ("loss", "joint_rotation_error"),
        )
        self.assertEqual(
            train_entry._online_metric_names({"evaluation": {"online_metrics": "loss"}}),
            ("loss",),
        )

    def test_compute_online_metrics_returns_wandb_ready_scalars(self):
        scalars = compute_online_metrics(
            {
                "loss": 1.25,
                "predicted_joints": [[0.0, 1.0], [0.2, 1.0]],
                "target_joints": [[0.0, 0.0], [0.0, 0.0]],
                "fps": 10.0,
            },
            ("loss", "g1_joint_pos_rmse_rad", "joint_velocity_rmse", "root_position_rmse"),
            prefix="train/",
            include_availability=True,
        )

        self.assertEqual(scalars["train/loss"], 1.25)
        self.assertTrue(math.isclose(scalars["train/g1_joint_pos_rmse_rad"], math.sqrt(0.51)))
        self.assertIn("train/joint_velocity_rmse", scalars)
        self.assertEqual(scalars["train/root_position_rmse_available"], 0.0)


if __name__ == "__main__":
    unittest.main()
