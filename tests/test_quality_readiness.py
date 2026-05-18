import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.quality_readiness import (
    QualityLaneInput,
    check_quality_lane_readiness,
)


class QualityReadinessTests(unittest.TestCase):
    def test_reports_ready_when_required_lanes_are_full_with_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _write_index(root / "split_index.csv", rows=3)
            lanes = []
            for name in ("source", "g1"):
                stats = _write_stats(root / f"{name}.jsonl", rows=3)
                report = _write_report(root / f"{name}.json", stats, scanned_rows=3, limit=None)
                lanes.append(QualityLaneInput(name=name, stats_jsonl=stats, report_json=report))

            result = check_quality_lane_readiness(
                index_csv=index,
                output_json=root / "readiness.json",
                lanes=lanes,
            )
            payload = json.loads(result.output_json.read_text(encoding="utf-8"))

        self.assertTrue(result.ready)
        self.assertEqual(result.status, "ready")
        self.assertEqual(payload["expected_rows"], 3)
        self.assertEqual(result.lanes[0].coverage_ratio, 1.0)

    def test_blocks_partial_lane_without_final_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _write_index(root / "split_index.csv", rows=4)
            stats = _write_stats(root / "g1.jsonl", rows=2)

            result = check_quality_lane_readiness(
                index_csv=index,
                output_json=root / "readiness.json",
                lanes=(QualityLaneInput(name="g1", stats_jsonl=stats, report_json=root / "missing.json"),),
            )

        self.assertFalse(result.ready)
        blockers = "\n".join(result.blockers)
        self.assertIn("stats cover 2/4 expected rows", blockers)
        self.assertIn("final report JSON is missing", blockers)
        self.assertEqual(result.lanes[0].status, "partial")

    def test_blocks_limited_report_even_when_rows_match_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _write_index(root / "split_index.csv", rows=3)
            stats = _write_stats(root / "source.jsonl", rows=3)
            report = _write_report(root / "source.json", stats, scanned_rows=3, limit=3)

            result = check_quality_lane_readiness(
                index_csv=index,
                output_json=root / "readiness.json",
                lanes=(QualityLaneInput(name="source", stats_jsonl=stats, report_json=report),),
            )

        self.assertFalse(result.ready)
        self.assertIn("report limit is 3", "\n".join(result.blockers))
        self.assertIn("rerun source without --limit", "\n".join(result.next_actions))

    def test_optional_missing_lane_warns_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _write_index(root / "split_index.csv", rows=1)

            result = check_quality_lane_readiness(
                index_csv=index,
                output_json=root / "readiness.json",
                lanes=(QualityLaneInput(name="source_fk", required=False),),
            )

        self.assertTrue(result.ready)
        self.assertIn("optional lane has no artifacts", "\n".join(result.warnings))


def _write_index(path: Path, rows: int) -> Path:
    lines = ["row_index,curation_action\n"]
    for index in range(rows):
        lines.append(f"{index},keep\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_stats(path: Path, rows: int) -> Path:
    path.write_text(
        "".join(json.dumps({"quality_action": "keep"}) + "\n" for _ in range(rows)),
        encoding="utf-8",
    )
    return path


def _write_report(path: Path, stats: Path, scanned_rows: int, limit: int | None) -> Path:
    path.write_text(
        json.dumps(
            {
                "stats_jsonl": str(stats),
                "scanned_rows": scanned_rows,
                "limit": limit,
                "action_counts": {"keep": scanned_rows},
                "flag_counts": {},
                "git_sha": "test",
                "git_dirty": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
