import math
import unittest

from online_retarget.metrics import (
    action_similarity,
    compute_metric_bundle,
    contact_artifact_metrics,
    joint_jump_rate,
    joint_limit_violation_rate,
    joint_mae,
    joint_mse,
    joint_rmse,
    joint_velocity_rmse,
    max_joint_abs_error,
    metric_metadata,
    mpjpe,
    weighted_mpjpe,
)


class MetricTests(unittest.TestCase):
    def test_mpjpe(self):
        pred = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        target = [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        self.assertEqual(mpjpe(pred, target), 0.5)

    def test_metric_registry_exposes_mpjpe_and_w_mpjpe_contract_status(self):
        fields = {
            "predicted_g1_body_pos": [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            "target_g1_body_pos": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            "body_position_weights": [2.0, 1.0],
        }

        blocked = compute_metric_bundle(fields, ("mpjpe", "w_mpjpe"))
        self.assertEqual(blocked["mpjpe"].status, "blocked")
        self.assertIn("pinned FK/link/root-alignment contract", blocked["mpjpe"].reason)
        self.assertEqual(blocked["w_mpjpe"].status, "blocked")

        fields["body_position_mpjpe_contract"] = {
            "pinned": True,
            "link_order": ["pelvis", "left_foot"],
            "units": "m",
            "root_alignment": "world_g1_root",
        }
        available = compute_metric_bundle(fields, ("mpjpe", "w_mpjpe"))

        self.assertEqual(available["mpjpe"].status, "available")
        self.assertEqual(available["mpjpe"].value, 0.5)
        self.assertEqual(available["w_mpjpe"].status, "available")
        self.assertEqual(available["w_mpjpe"].value, 1.0 / 3.0)
        self.assertEqual(
            weighted_mpjpe(
                fields["predicted_g1_body_pos"],
                fields["target_g1_body_pos"],
                [2.0, 1.0],
            ),
            1.0 / 3.0,
        )
        metadata = metric_metadata(("mpjpe", "w_mpjpe"))
        self.assertEqual(metadata["mpjpe"]["source_ref"], "LR-239 shared online/offline metric registry")
        self.assertIn("body_position_weights", metadata["w_mpjpe"]["required_fields"])

    def test_joint_rmse(self):
        self.assertTrue(math.isclose(joint_rmse([[1.0, 3.0]], [[1.0, 1.0]]), math.sqrt(2.0)))

    def test_joint_loss_metrics(self):
        predicted = [[1.0, 3.0], [2.0, -1.0]]
        target = [[0.0, 1.0], [2.5, 1.0]]

        self.assertEqual(joint_mae(predicted, target), 1.375)
        self.assertEqual(joint_mse(predicted, target), 2.3125)
        self.assertEqual(max_joint_abs_error(predicted, target), 2.0)

    def test_joint_velocity_rmse(self):
        predicted = [[0.0, 0.0], [0.2, 0.0]]
        target = [[0.0, 0.0], [0.0, 0.1]]

        self.assertTrue(math.isclose(joint_velocity_rmse(predicted, target, fps=10.0), 1.5811388300841898))

    def test_joint_velocity_rmse_single_frame_is_zero(self):
        self.assertEqual(joint_velocity_rmse([[0.0, 0.0]], [[1.0, 1.0]], fps=10.0), 0.0)

    def test_action_similarity(self):
        self.assertEqual(action_similarity([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]), 1.0)

    def test_joint_jump_rate(self):
        positions = [[0.0, 0.0], [0.2, 0.01]]
        self.assertEqual(joint_jump_rate(positions, fps=10.0, max_velocity=1.0), 0.5)

    def test_joint_jump_rate_single_frame_is_zero(self):
        self.assertEqual(joint_jump_rate([[0.0, 0.0]], fps=10.0, max_velocity=1.0), 0.0)

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
