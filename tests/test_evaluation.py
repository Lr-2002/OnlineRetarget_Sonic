import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.evaluation import EvaluationConfig, evaluate_jsonl
from online_retarget.metrics import compute_online_metrics


class EvaluationTests(unittest.TestCase):
    def test_evaluate_jsonl_writes_summary_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_jsonl = root / "predictions.jsonl"
            samples = [
                {
                    "sample_id": "s1",
                    "actor_uid": "A001",
                    "category": "Baseline",
                    "package": "Locomotion",
                    "quality_flags": [],
                    "predicted_joints": [[0.0, 0.0], [0.0, 0.0]],
                    "target_joints": [[0.0, 0.0], [0.0, 0.0]],
                    "predicted_body_pos": [[[0.0, 0.0, 0.0]]],
                    "target_body_pos": [[[0.0, 0.0, 0.0]]],
                },
                {
                    "sample_id": "s2",
                    "actor_uid": "A002",
                    "category": "Baseline",
                    "package": "Locomotion",
                    "quality_flags": ["joint_velocity_jump"],
                    "predicted_joints": [[0.0, 0.0], [2.0, 0.0]],
                    "target_joints": [[0.0, 0.0], [0.0, 0.0]],
                    "predicted_body_pos": [
                        [[0.0, 0.0, 0.08]],
                        [[0.2, 0.0, 0.08]],
                    ],
                    "target_body_pos": [
                        [[0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0]],
                    ],
                    "body_names": ["left_foot"],
                    "foot_body_names": ["left_foot"],
                    "fps": 10.0,
                },
            ]
            with input_jsonl.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample))
                    f.write("\n")

            result = evaluate_jsonl(
                input_jsonl=input_jsonl,
                output_root=root / "runs",
                config=EvaluationConfig(
                    run_name="fixture_eval",
                    max_failures=1,
                    joint_jump_velocity=1.0,
                ),
            )

            summary = json.loads(result.summary_json.read_text())
            failure_rows = result.failure_manifest_csv.read_text().strip().splitlines()
            per_sample_rows = result.per_sample_csv.read_text().strip().splitlines()

        self.assertEqual(result.sample_count, 2)
        self.assertEqual(summary["overall"]["joint_mae"], 0.25)
        self.assertEqual(summary["overall"]["joint_mse"], 0.5)
        self.assertEqual(summary["overall"]["joint_rmse"], 0.5)
        self.assertEqual(summary["overall"]["max_joint_abs_error"], 1.0)
        self.assertEqual(summary["by_quality_flag"]["joint_velocity_jump"]["joint_rmse"], 1.0)
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"]["predicted_joint_jump_rate"],
            0.5,
        )
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"]["target_joint_jump_rate"],
            0.0,
        )
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"][
                "predicted_minus_target_joint_jump_rate"
            ],
            0.5,
        )
        self.assertEqual(summary["by_quality_flag"]["none"]["joint_rmse"], 0.0)
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"]["predicted_foot_float_rate"],
            1.0,
        )
        self.assertIn("joint_velocity_rmse", per_sample_rows[0])
        self.assertIn("predicted_contact_slide_rate", per_sample_rows[0])
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"]["predicted_contact_skate_rate"],
            1.0,
        )
        self.assertIn("predicted_max_contact_skate_distance", per_sample_rows[0])
        self.assertEqual(len(failure_rows), 2)
        self.assertIn("s2", failure_rows[1])
        self.assertEqual(summary["metric_availability"]["mpjpe"]["counts"]["blocked"], 2)
        self.assertNotIn("mpjpe", summary["overall"])
        self.assertIn("g1_joint_pos_rmse_rad", summary["metric_metadata"])

    def test_evaluate_jsonl_keeps_not_applicable_out_of_numeric_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_jsonl = root / "predictions.jsonl"
            input_jsonl.write_text(json.dumps({"sample_id": "s1", "method_id": "nmr"}) + "\n")

            result = evaluate_jsonl(
                input_jsonl=input_jsonl,
                output_root=root / "runs",
                config=EvaluationConfig(
                    metrics=("global_body_position_error",),
                    run_name="not_applicable",
                ),
            )
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))

        self.assertNotIn("global_body_position_error", summary["overall"])
        self.assertEqual(
            summary["metric_availability"]["global_body_position_error"]["counts"][
                "not_applicable"
            ],
            1,
        )

    def test_online_and_offline_metric_paths_match_same_inputs(self):
        metric_names = (
            "loss",
            "g1_joint_pos_rmse_rad",
            "joint_velocity_rmse",
            "root_position_rmse",
            "root_rot6d_rmse",
            "global_body_position_error",
            "root_relative_body_position_error",
            "joint_rotation_error",
            "joint_velocity_error",
            "joint_acceleration_error",
            "joint_jump_count",
            "joint_limit_proximity_count",
            "self_collision_count",
            "floating_guard",
            "cross_ratio_guard",
            "phc_failure_guard",
            "mpjpe",
            "w_mpjpe",
        )
        sample = {
            "sample_id": "same",
            "loss": 0.25,
            "predicted_joints": [[0.0, 1.0], [0.2, 1.1]],
            "target_joints": [[0.0, 0.0], [0.0, 1.0]],
            "predicted_joint_velocities": [[0.0, 0.0], [2.0, 1.0]],
            "target_joint_velocities": [[0.0, 0.0], [0.0, 0.0]],
            "predicted_joint_accelerations": [[1.0, 0.0]],
            "target_joint_accelerations": [[0.0, 0.0]],
            "predicted_joint_rotations": [[[1.0, 0.0, 0.0, 0.0]]],
            "target_joint_rotations": [[[1.0, 0.0, 0.0, 0.0]]],
            "joint_lower_limits": [-2.0, -2.0],
            "joint_upper_limits": [2.0, 2.0],
            "self_collision_count": 0,
            "mean_lowest_foot_height": 0.02,
            "cross_ratio": 0.0,
            "phc_avg_body_joint_distance": 0.1,
            "pred_root_pos_w": [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
            "target_root_pos_w": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "predicted_root_rot6d": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            "target_root_rot6d": [[1.0, 0.0, 0.0, 0.0, 0.5, 0.0]],
            "predicted_g1_body_pos": [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            "target_g1_body_pos": [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            "body_position_weights": [2.0, 1.0],
            "body_position_mpjpe_contract": {
                "pinned": True,
                "link_order": ["pelvis", "left_foot"],
                "units": "m",
                "root_alignment": "world_g1_root",
            },
            "fps": 10.0,
        }

        online = compute_online_metrics(sample, metric_names)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_jsonl = root / "predictions.jsonl"
            input_jsonl.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            result = evaluate_jsonl(
                input_jsonl=input_jsonl,
                output_root=root / "runs",
                config=EvaluationConfig(
                    metrics=metric_names,
                    run_name="same_inputs",
                    fps=10.0,
                ),
            )
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))

        for name in metric_names:
            self.assertIn(name, online)
            self.assertEqual(summary["metric_availability"][name]["counts"]["available"], 1)
            self.assertAlmostEqual(online[name], summary["overall"][name])


if __name__ == "__main__":
    unittest.main()
