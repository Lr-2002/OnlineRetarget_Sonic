import csv
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.review_manifest import (
    build_review_decision_template,
    build_review_manifest,
    merge_review_decisions,
)


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
        self.assertIn("contact_correction", families)
        self.assertIn("joint_limit", families)
        self.assertIn("self_collision", families)
        self.assertEqual(result.family_counts["parser"], 1)
        self.assertEqual(report["reviewed_rows"], len(items))
        self.assertEqual(items[0]["motion_paths"]["source_bvh"], "soma/a.bvh")
        self.assertEqual(items[0]["motion_paths"]["g1_csv"], "g1/a.csv")
        self.assertIn("review_fields", items[0])
        correction_item = next(item for item in items if item["failure_family"] == "contact_correction")
        self.assertEqual(correction_item["metrics"]["g1_contact_correction_candidate"], "1")
        self.assertEqual(
            correction_item["metrics"]["g1_contact_correction_reason"],
            "vertical_penetration_offset",
        )
        self.assertEqual(
            correction_item["metrics"]["source_fk_contact_correction_reason"],
            "vertical_penetration_offset",
        )
        self.assertIn("Manual Motion Review Manifest", markdown)
        self.assertIn("Recommended action", markdown)

    def test_build_review_decision_template_can_feed_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worst = root / "worst_clips.csv"
            _write_worst_clips(worst)
            manifest = build_review_manifest(
                worst_clips_csv=worst,
                output_root=root / "review",
                run_name="fixture",
                max_per_family=1,
            )

            template = build_review_decision_template(
                review_manifest_jsonl=manifest.manifest_jsonl,
                output_csv=root / "decisions_template.csv",
                output_report_json=root / "decisions_template_report.json",
            )
            rows = _read_csv(template.output_csv)
            report = json.loads(template.report_json.read_text(encoding="utf-8"))
            for row in rows:
                row["decision"] = "confirmed"
                row["reviewer"] = "unit-test"
                row["confirmed_issue"] = "yes"
                row["recommended_action"] = "quarantine"
                row["notes"] = "filled from template"
            filled = root / "decisions_filled.csv"
            _write_rows(filled, rows)

            result = merge_review_decisions(
                review_manifest_jsonl=manifest.manifest_jsonl,
                decisions_file=filled,
                output_jsonl=root / "reviewed.jsonl",
                output_report_json=root / "decision_report.json",
            )

        self.assertEqual(template.manifest_items, len(rows))
        self.assertEqual(report["manifest_items"], len(rows))
        self.assertIn("metric_summary", rows[0])
        self.assertEqual(rows[0]["source_bvh"], "soma/a.bvh")
        self.assertEqual(rows[0]["g1_csv"], "g1/a.csv")
        self.assertEqual(result.complete_decisions, len(rows))

    def test_build_review_decision_template_refuses_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "review_manifest.jsonl"
            manifest.write_text(
                json.dumps({"review_id": "known", "review_fields": {}}) + "\n",
                encoding="utf-8",
            )
            output = root / "template.csv"
            output.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                build_review_decision_template(manifest, output_csv=output)

    def test_merge_review_decisions_writes_reviewed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worst = root / "worst_clips.csv"
            _write_worst_clips(worst)
            manifest = build_review_manifest(
                worst_clips_csv=worst,
                output_root=root / "review",
                run_name="fixture",
                max_per_family=1,
            )
            items = _read_jsonl(manifest.manifest_jsonl)
            decisions = root / "decisions.csv"
            _write_decisions_csv(
                decisions,
                rows=[
                    {
                        "review_id": item["review_id"],
                        "decision": "confirmed",
                        "reviewer": "unit-test",
                        "notes": f"checked {item['failure_family']}",
                        "confirmed_issue": "yes",
                        "recommended_action": "quarantine",
                    }
                    for item in items
                ],
            )

            result = merge_review_decisions(
                review_manifest_jsonl=manifest.manifest_jsonl,
                decisions_file=decisions,
                output_jsonl=root / "reviewed.jsonl",
                output_report_json=root / "decision_report.json",
            )
            reviewed = _read_jsonl(result.output_jsonl)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual(result.manifest_items, len(items))
        self.assertEqual(result.complete_decisions, len(items))
        self.assertEqual(result.incomplete_decisions, 0)
        self.assertEqual(result.matched_decisions, len(items))
        self.assertEqual(reviewed[0]["review_fields"]["reviewer"], "unit-test")
        self.assertEqual(reviewed[0]["review_fields"]["recommended_action"], "quarantine")
        self.assertEqual(report["complete_decisions"], len(items))

    def test_merge_review_decisions_rejects_unknown_review_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "review_manifest.jsonl"
            manifest.write_text(
                json.dumps({"review_id": "known", "review_fields": {}}) + "\n",
                encoding="utf-8",
            )
            decisions = root / "decisions.csv"
            _write_decisions_csv(
                decisions,
                rows=[
                    {
                        "review_id": "unknown",
                        "decision": "confirmed",
                        "recommended_action": "quarantine",
                    }
                ],
            )

            with self.assertRaises(ValueError) as raised:
                merge_review_decisions(manifest, decisions)

        self.assertIn("unknown review_id", str(raised.exception))

    def test_merge_review_decisions_rejects_invalid_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "review_manifest.jsonl"
            manifest.write_text(
                json.dumps({"review_id": "known", "review_fields": {}}) + "\n",
                encoding="utf-8",
            )
            decisions = root / "decisions.jsonl"
            decisions.write_text(
                json.dumps(
                    {
                        "review_id": "known",
                        "review_fields": {
                            "decision": "confirmed",
                            "recommended_action": "needs_magic",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as raised:
                merge_review_decisions(manifest, decisions)

        self.assertIn("invalid recommended_action", str(raised.exception))


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
            "g1_contact_correction_candidate": "",
            "g1_contact_correction_reason": "",
            "g1_contact_correction_offset": "",
            "g1_contact_correction_abs_offset": "",
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
            "g1_contact_correction_candidate": "1",
            "g1_contact_correction_reason": "vertical_penetration_offset",
            "g1_contact_correction_offset": "0.07",
            "g1_contact_correction_abs_offset": "0.07",
            "g1_contact_slide_rate": "0.25",
            "source_fk_contact_correction_candidate": "1",
            "source_fk_contact_correction_reason": "vertical_penetration_offset",
            "source_fk_contact_correction_offset": "0.02",
            "source_fk_contact_correction_abs_offset": "0.02",
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


def _write_decisions_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "review_id",
        "decision",
        "reviewer",
        "notes",
        "confirmed_issue",
        "recommended_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
