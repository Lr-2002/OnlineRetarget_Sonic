import csv
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.quality_review_exports import export_balanced_quality_review_csv


class QualityReviewExportTests(unittest.TestCase):
    def test_export_balanced_quality_review_csv_samples_each_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "g1_stats.jsonl"
            _write_jsonl(
                stats,
                [
                    _row("1", "clip_a", "quarantine", "g1_ground_penetration", penetration=0.1),
                    _row("2", "clip_b", "quarantine", "g1_ground_penetration", penetration=0.5),
                    _row("3", "clip_c", "quarantine", "g1_self_collision_proxy", collision=0.2),
                    _row("4", "clip_d", "downweight", "g1_foot_slide", slide=0.9),
                ],
            )

            result = export_balanced_quality_review_csv(
                stats_jsonl=stats,
                output_csv=root / "review.csv",
                flags=("g1_ground_penetration", "g1_self_collision_proxy", "g1_foot_slide"),
                max_per_flag=1,
            )
            rows = _read_csv(result.output_csv)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual([row["filename"] for row in rows], ["clip_b", "clip_c"])
        self.assertEqual(result.family_counts["g1_ground_penetration"], 1)
        self.assertEqual(result.family_counts["g1_self_collision_proxy"], 1)
        self.assertNotIn("g1_foot_slide", result.family_counts)
        self.assertEqual(report["exported_rows"], 2)

    def test_export_balanced_quality_review_csv_can_include_downweight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "g1_stats.jsonl"
            _write_jsonl(
                stats,
                [_row("4", "clip_d", "downweight", "g1_foot_slide", slide=0.9)],
            )

            result = export_balanced_quality_review_csv(
                stats_jsonl=stats,
                output_csv=root / "review.csv",
                flags=("g1_foot_slide",),
                max_per_flag=1,
                include_downweight=True,
            )
            rows = _read_csv(result.output_csv)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_family"], "g1_foot_slide")
        self.assertEqual(rows[0]["quality_action"], "downweight")

    def test_export_balanced_quality_review_csv_backfills_split_index_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "g1_stats.jsonl"
            split_index = root / "split_index.csv"
            target_only_row = _row(
                "9",
                "clip_stats_name",
                "quarantine",
                "g1_ground_penetration",
                penetration=0.9,
            )
            target_only_row["move_soma_proportional_path"] = ""
            _write_jsonl(
                stats,
                [target_only_row],
            )
            _write_split_index(split_index)

            result = export_balanced_quality_review_csv(
                stats_jsonl=stats,
                split_index_csv=split_index,
                output_csv=root / "review.csv",
                flags=("g1_ground_penetration",),
                max_per_flag=1,
            )
            rows = _read_csv(result.output_csv)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["row_index"], "9")
        self.assertEqual(rows[0]["filename"], "clip_stats_name")
        self.assertEqual(
            rows[0]["move_soma_proportional_path"],
            "soma_proportional/bvh/240101/clip_index_name.bvh",
        )
        self.assertEqual(report["split_index_csv"], str(split_index))

    def test_export_balanced_quality_review_csv_writes_headers_when_no_rows_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "g1_stats.jsonl"
            _write_jsonl(
                stats,
                [_row("4", "clip_d", "downweight", "g1_foot_slide", slide=0.9)],
            )

            result = export_balanced_quality_review_csv(
                stats_jsonl=stats,
                output_csv=root / "review.csv",
                flags=("g1_foot_slide",),
                max_per_flag=1,
            )
            rows = _read_csv(result.output_csv)
            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            output_text = result.output_csv.read_text(encoding="utf-8")

        self.assertEqual(rows, [])
        self.assertEqual(report["exported_rows"], 0)
        self.assertIn("quality_action", output_text)
        self.assertIn("review_family", output_text)

    def test_unstable_start_end_uses_start_end_speed_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "g1_stats.jsonl"
            _write_jsonl(
                stats,
                [
                    _row(
                        "1",
                        "high_joint_velocity",
                        "quarantine",
                        "g1_unstable_start_end",
                        joint_velocity=200.0,
                        start_end_root_speed=0.21,
                    ),
                    _row(
                        "2",
                        "high_start_end_speed",
                        "quarantine",
                        "g1_unstable_start_end",
                        joint_velocity=5.0,
                        start_end_root_speed=1.2,
                    ),
                ],
            )

            result = export_balanced_quality_review_csv(
                stats_jsonl=stats,
                output_csv=root / "review.csv",
                flags=("g1_unstable_start_end",),
                max_per_flag=1,
            )
            rows = _read_csv(result.output_csv)

        self.assertEqual(rows[0]["filename"], "high_start_end_speed")
        self.assertEqual(rows[0]["max_start_end_root_speed"], "1.2")


def _row(
    row_index: str,
    filename: str,
    action: str,
    flags: str,
    *,
    penetration: float = 0.0,
    slide: float = 0.0,
    collision: float = 0.0,
    joint_velocity: float = 0.0,
    start_end_root_speed: float = 0.0,
) -> dict[str, object]:
    return {
        "quality_action": action,
        "quality_flags": flags,
        "row_index": row_index,
        "split": "train",
        "category": "Fixture",
        "actor_uid": "A001",
        "filename": filename,
        "move_soma_proportional_path": f"soma_proportional/bvh/240101/{filename}.bvh",
        "move_g1_path": f"g1/csv/240101/{filename}.csv",
        "penetration_depth": penetration,
        "contact_slide_rate": slide,
        "self_collision_proxy_rate": collision,
        "max_abs_joint_velocity": joint_velocity,
        "max_start_end_root_speed": start_end_root_speed,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_split_index(path: Path) -> None:
    fieldnames = [
        "row_index",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "row_index": "9",
                "filename": "clip_index_name",
                "move_soma_proportional_path": "soma_proportional/bvh/240101/clip_index_name.bvh",
                "move_g1_path": "g1/csv/240101/clip_index_name.csv",
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
