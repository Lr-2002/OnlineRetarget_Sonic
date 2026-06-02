import math
import unittest

from online_retarget.metrics import (
    action_similarity,
    compute_metric_bundle,
    compute_online_metrics,
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

        self.assertTrue(
            math.isclose(
                joint_velocity_rmse(predicted, target, fps=10.0),
                1.5811388300841898,
            )
        )

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

    def test_metric_bundle_reports_metadata_and_unavailable_velocity(self):
        bundle = compute_metric_bundle(
            {
                "predicted_joints": [[0.0, 1.0]],
                "target_joints": [[0.0, 3.0]],
                "loss": 2.5,
            },
            ("loss", "g1_joint_pos_rmse_rad", "joint_velocity_rmse", "root_position_rmse"),
        )

        self.assertEqual(bundle["loss"].status, "available")
        self.assertEqual(bundle["loss"].value, 2.5)
        self.assertEqual(bundle["g1_joint_pos_rmse_rad"].metadata.unit, "rad")
        self.assertEqual(bundle["g1_joint_pos_rmse_rad"].metadata.direction, "lower_is_better")
        self.assertEqual(bundle["joint_velocity_rmse"].status, "unavailable")
        self.assertIn("at least two frames", bundle["joint_velocity_rmse"].reason)
        self.assertEqual(bundle["root_position_rmse"].status, "unavailable")

        independent_batch = compute_metric_bundle(
            {
                "independent_batch": True,
                "predicted_joints": [[0.0, 1.0], [0.2, 1.0]],
                "target_joints": [[0.0, 0.0], [0.0, 0.0]],
            },
            ("joint_velocity_rmse",),
        )
        self.assertEqual(independent_batch["joint_velocity_rmse"].status, "unavailable")
        self.assertIn("explicit joint velocity", independent_batch["joint_velocity_rmse"].reason)

        metadata = metric_metadata(("g1_joint_pos_rmse_rad",))
        self.assertEqual(metadata["g1_joint_pos_rmse_rad"]["unit"], "rad")
        self.assertIn("mask_semantics", metadata["g1_joint_pos_rmse_rad"])

    def test_mpjpe_requires_pinned_contract_for_registry_value(self):
        fields = {
            "predicted_g1_body_pos": [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            "target_g1_body_pos": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            "body_position_weights": [2.0, 1.0],
        }

        blocked = compute_metric_bundle(fields, ("mpjpe", "w_mpjpe"))
        self.assertEqual(blocked["mpjpe"].status, "blocked")
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

    def test_paper_body_metrics_use_global_and_root_relative_formulas(self):
        fields = {
            "predicted_g1_body_pos": [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            "target_g1_body_pos": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            "predicted_root_pos": [[1.0, 0.0, 0.0]],
            "target_root_pos": [[0.0, 0.0, 0.0]],
            "body_position_mpjpe_contract": {
                "pinned": True,
                "link_order": ["pelvis", "left_foot"],
                "units": "m",
                "root_alignment": "world_g1_root",
            },
        }

        bundle = compute_metric_bundle(
            fields,
            ("global_body_position_error", "E_g_mpbpe", "root_relative_MPJPE", "E_mpbpe"),
        )

        self.assertEqual(bundle["global_body_position_error"].value, 1.0)
        self.assertEqual(bundle["E_g_mpbpe"].value, 1.0)
        self.assertEqual(bundle["root_relative_MPJPE"].value, 0.0)
        self.assertEqual(bundle["E_mpbpe"].value, 0.0)

        metadata = metric_metadata(("root_relative_body_position_error",))
        self.assertIn("E_mpbpe", metadata["root_relative_body_position_error"]["paper_labels"])
        self.assertIn("body_position_minus_root_position", metadata["root_relative_body_position_error"]["formula"])
        self.assertIn("nmr", metadata["root_relative_body_position_error"]["method_coverage"])

    def test_method_coverage_marks_metric_not_applicable(self):
        bundle = compute_metric_bundle(
            {"method_id": "nmr"},
            ("global_body_position_error", "joint_jump_count"),
        )

        self.assertEqual(bundle["global_body_position_error"].status, "not_applicable")
        self.assertIn("method_id=nmr", bundle["global_body_position_error"].reason)
        self.assertEqual(bundle["joint_jump_count"].status, "unavailable")

    def test_joint_rotation_error_uses_full_joint_geodesic_angle(self):
        bundle = compute_metric_bundle(
            {
                "predicted_joint_rotations": [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
                "target_joint_rotations": [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            },
            ("joint_rotation_error",),
        )

        self.assertTrue(math.isclose(bundle["joint_rotation_error"].value, math.pi / 2.0))

    def test_phc_velocity_and_acceleration_terms_are_registry_metrics(self):
        bundle = compute_metric_bundle(
            {
                "predicted_joint_velocities": [[0.0, 0.0], [2.0, 0.0]],
                "target_joint_velocities": [[0.0, 0.0], [0.0, 0.0]],
                "fps": 1.0,
            },
            ("joint_velocity_error", "joint_acceleration_error"),
        )

        self.assertEqual(bundle["joint_velocity_error"].value, 1.0)
        self.assertTrue(math.isclose(bundle["joint_acceleration_error"].value, math.sqrt(2.0)))

        independent = compute_metric_bundle(
            {
                "independent_batch": True,
                "predicted_joints": [[0.0], [1.0], [3.0]],
                "target_joints": [[0.0], [0.0], [0.0]],
            },
            ("joint_acceleration_error",),
        )
        self.assertEqual(independent["joint_acceleration_error"].status, "unavailable")
        self.assertIn("explicit joint acceleration", independent["joint_acceleration_error"].reason)

    def test_nmr_count_metrics_use_position_thresholds_not_velocity_rate(self):
        jump = compute_metric_bundle(
            {
                "qpos": [[0.0, 0.0], [0.6, 0.1], [0.7, -0.6]],
                "joint_jump_threshold_rad": 0.5,
                "fps": 120.0,
            },
            ("joint_jump_count",),
        )
        self.assertEqual(jump["joint_jump_count"].value, 2.0)

        proximity = compute_metric_bundle(
            {
                "qpos": [[0.96, 0.0], [0.0, -0.96]],
                "joint_lower_limits": [-1.0, -1.0],
                "joint_upper_limits": [1.0, 1.0],
                "joint_limit_proximity_threshold_rad": 0.05,
            },
            ("joint_limit_proximity_count",),
        )
        self.assertEqual(proximity["joint_limit_proximity_count"].value, 2.0)

    def test_collision_count_and_physical_guards_consume_explicit_inputs(self):
        bundle = compute_metric_bundle(
            {
                "mujoco_self_contacts": [
                    [],
                    [{"geom1": "torso", "geom2": "arm"}],
                    [{"geom1": "left_hand", "geom2": "right_hand"}],
                ],
                "allowed_contact_pairs": [("left_hand", "right_hand")],
                "mean_lowest_foot_height": [0.12, 0.14],
                "self_intersection_frames": [False, True, False, False],
                "phc_avg_body_joint_distance": 0.4,
            },
            ("self_collision_count", "floating_guard", "cross_ratio_guard", "phc_failure_guard"),
        )

        self.assertEqual(bundle["self_collision_count"].value, 1.0)
        self.assertEqual(bundle["floating_guard"].value, 0.0)
        self.assertEqual(bundle["cross_ratio_guard"].value, 0.0)
        self.assertEqual(bundle["phc_failure_guard"].value, 1.0)

        unavailable = compute_metric_bundle({}, ("self_collision_count", "floating_guard"))
        self.assertEqual(unavailable["self_collision_count"].status, "unavailable")
        self.assertEqual(unavailable["floating_guard"].status, "unavailable")

    def test_compute_online_metrics_returns_scalar_payload(self):
        metrics = compute_online_metrics(
            {
                "loss": 1.25,
                "predicted_joints": [[0.0, 1.0], [0.2, 1.0]],
                "target_joints": [[0.0, 0.0], [0.0, 0.0]],
                "fps": 10.0,
            },
            ("loss", "g1_joint_pos_rmse_rad", "joint_velocity_rmse"),
            prefix="train/",
        )

        self.assertEqual(metrics["train/loss"], 1.25)
        self.assertTrue(math.isclose(metrics["train/g1_joint_pos_rmse_rad"], math.sqrt(0.51)))
        self.assertIn("train/joint_velocity_rmse", metrics)


if __name__ == "__main__":
    unittest.main()
