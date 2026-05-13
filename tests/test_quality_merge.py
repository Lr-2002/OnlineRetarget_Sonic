import csv
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.quality_merge import merge_quality_stats


class QualityMergeTests(unittest.TestCase):
    def test_merge_quality_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_index = root / "split_index.csv"
            source_stats = root / "source.jsonl"
            source_fk_stats = root / "source_fk.jsonl"
            g1_stats = root / "g1.jsonl"
            _write_split_index(split_index)
            source_stats.write_text(
                json.dumps(
                    {
                        "row_index": "1",
                        "quality_action": "quarantine",
                        "quality_flags": "source_channel_jump",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_fk_stats.write_text(
                json.dumps(
                    {
                        "row_index": "1",
                        "quality_action": "downweight",
                        "quality_flags": "source_foot_slide",
                        "contact_frame_ratio": 0.8,
                        "contact_slide_rate": 0.15,
                        "max_contact_slide_speed": 0.4,
                        "mean_foot_clearance": 0.05,
                        "penetration_depth": 0.02,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            g1_stats.write_text(
                json.dumps(
                    {
                        "row_index": "1",
                        "quality_action": "keep",
                        "quality_flags": "",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "row_index": "2",
                        "quality_action": "exclude",
                        "quality_flags": "missing_g1_csv_member",
                        "penetration_depth": 0.07,
                        "contact_slide_rate": 0.25,
                        "joint_limit_violation_rate": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = merge_quality_stats(
                split_index_csv=split_index,
                source_stats_jsonl=source_stats,
                source_fk_stats_jsonl=source_fk_stats,
                g1_stats_jsonl=g1_stats,
                output_root=root / "runs",
                run_name="fixture",
            )
            rows = _read_csv(result.curated_index_csv)
            worst = _read_csv(result.worst_clips_csv)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.merged_source_rows, 1)
        self.assertEqual(result.merged_source_fk_rows, 1)
        self.assertEqual(result.merged_g1_rows, 2)
        self.assertEqual(rows[0]["merged_quality_action"], "quarantine")
        self.assertEqual(
            rows[0]["merged_quality_flags"],
            "source:source_channel_jump|source_fk:source_foot_slide",
        )
        self.assertEqual(rows[0]["source_fk_quality_action"], "downweight")
        self.assertEqual(rows[0]["source_fk_quality_flags"], "source_foot_slide")
        self.assertEqual(rows[1]["merged_quality_action"], "exclude")
        self.assertEqual(rows[1]["merged_quality_flags"], "mirror_variant|g1:missing_g1_csv_member")
        self.assertEqual(len(worst), 2)
        self.assertEqual(worst[0]["merged_quality_action"], "exclude")
        self.assertEqual(worst[0]["g1_penetration_depth"], "0.07")
        self.assertEqual(worst[0]["g1_contact_slide_rate"], "0.25")
        self.assertEqual(worst[0]["g1_joint_limit_violation_rate"], "0.5")
        self.assertEqual(worst[1]["source_fk_contact_slide_rate"], "0.15")
        self.assertEqual(worst[1]["source_fk_penetration_depth"], "0.02")
        self.assertEqual(report["source_fk_stats_jsonl"], str(source_fk_stats))
        self.assertEqual(report["merged_source_fk_rows"], 1)
        self.assertEqual(report["breakdown"]["split"]["train"]["exclude"], 1)
        self.assertEqual(report["worst_clips_csv"], str(result.worst_clips_csv))


def _write_split_index(path: Path) -> None:
    rows = [
        {
            "row_index": "1",
            "split": "train",
            "actor_uid": "A001",
            "package": "Locomotion",
            "category": "Idle",
            "filename": "motion_a",
            "curation_action": "keep",
            "quality_flags": "",
        },
        {
            "row_index": "2",
            "split": "train",
            "actor_uid": "A002",
            "package": "Locomotion",
            "category": "Run",
            "filename": "motion_b",
            "curation_action": "downweight",
            "quality_flags": "mirror_variant",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
