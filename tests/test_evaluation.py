import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.evaluation import EvaluationConfig, evaluate_jsonl


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
        self.assertEqual(summary["by_quality_flag"]["joint_velocity_jump"]["predicted_joint_jump_rate"], 0.5)
        self.assertEqual(summary["by_quality_flag"]["joint_velocity_jump"]["target_joint_jump_rate"], 0.0)
        self.assertEqual(
            summary["by_quality_flag"]["joint_velocity_jump"]["predicted_minus_target_joint_jump_rate"],
            0.5,
        )
        self.assertEqual(summary["by_quality_flag"]["none"]["joint_rmse"], 0.0)
        self.assertEqual(summary["by_quality_flag"]["joint_velocity_jump"]["predicted_foot_float_rate"], 1.0)
        self.assertIn("joint_velocity_rmse", per_sample_rows[0])
        self.assertIn("predicted_contact_slide_rate", per_sample_rows[0])
        self.assertEqual(summary["by_quality_flag"]["joint_velocity_jump"]["predicted_contact_skate_rate"], 1.0)
        self.assertIn("predicted_max_contact_skate_distance", per_sample_rows[0])
        self.assertEqual(len(failure_rows), 2)
        self.assertIn("s2", failure_rows[1])


if __name__ == "__main__":
    unittest.main()
