import csv
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.review_manifest import build_review_manifest


class ReviewManifestTests(unittest.TestCase):
    def test_build_review_manifest_groups_failure_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worst = root / "worst_clips.csv"
            _write_worst_clips(worst)

            result = build_review_manifest(
                worst_clips_csv=worst,
                output_root=root / "review",
                run_name="fixture",
                max_per_family=1,
            )
            items = [
                json.loads(line)
                for line in result.manifest_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            markdown = result.manifest_md.read_text(encoding="utf-8")

        families = {item["failure_family"] for item in items}
        self.assertIn("parser", families)
        self.assertIn("foot_slide", families)
        self.assertIn("penetration", families)
        self.assertIn("joint_limit", families)
        self.assertIn("self_collision", families)
        self.assertEqual(result.family_counts["parser"], 1)
        self.assertEqual(report["reviewed_rows"], len(items))
        self.assertEqual(items[0]["motion_paths"]["source_bvh"], "soma/a.bvh")
        self.assertEqual(items[0]["motion_paths"]["g1_csv"], "g1/a.csv")
        self.assertIn("review_fields", items[0])
        self.assertIn("Manual Motion Review Manifest", markdown)
        self.assertIn("Recommended action", markdown)


def _write_worst_clips(path: Path) -> None:
    rows = [
        {
            "row_index": "1",
            "split": "train",
            "actor_uid": "A001",
            "package": "Locomotion",
            "category": "Run",
            "filename": "a",
            "move_soma_proportional_path": "soma/a.bvh",
            "move_g1_path": "g1/a.csv",
            "merged_quality_action": "exclude",
            "merged_quality_flags": "source:nonfinite_value|source:channel_width_mismatch",
            "source_channel_jump_rate": "",
            "source_max_abs_channel_velocity": "",
            "g1_joint_limit_violation_rate": "",
            "g1_penetration_depth": "",
            "g1_contact_slide_rate": "",
        },
        {
            "row_index": "2",
            "split": "train",
            "actor_uid": "A002",
            "package": "Communication",
            "category": "Gestures",
            "filename": "b",
            "move_soma_proportional_path": "soma/b.bvh",
            "move_g1_path": "g1/b.csv",
            "merged_quality_action": "quarantine",
            "merged_quality_flags": "source_fk:source_foot_slide|g1:g1_ground_penetration|g1:g1_joint_limit_violation",
            "source_channel_jump_rate": "0.0",
            "source_max_abs_channel_velocity": "10.0",
            "g1_joint_limit_violation_rate": "0.5",
            "g1_penetration_depth": "0.07",
            "g1_contact_slide_rate": "0.25",
        },
        {
            "row_index": "3",
            "split": "train",
            "actor_uid": "A003",
            "package": "Locomotion",
            "category": "Run",
            "filename": "c",
            "move_soma_proportional_path": "soma/c.bvh",
            "move_g1_path": "g1/c.csv",
            "merged_quality_action": "quarantine",
            "merged_quality_flags": "g1:g1_self_collision_proxy",
            "source_channel_jump_rate": "0.0",
            "source_max_abs_channel_velocity": "10.0",
            "g1_joint_limit_violation_rate": "0.0",
            "g1_penetration_depth": "0.0",
            "g1_contact_slide_rate": "0.0",
            "g1_self_collision_proxy_rate": "0.5",
            "g1_min_self_collision_distance": "0.01",
            "g1_mean_min_self_collision_distance": "0.03",
        },
    ]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
