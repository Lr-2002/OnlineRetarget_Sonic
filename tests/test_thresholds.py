import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import contextlib
import io

from online_retarget import cli
from online_retarget.data.thresholds import propose_thresholds_from_jsonl, write_threshold_proposals


class ThresholdProposalTests(unittest.TestCase):
    def test_propose_thresholds_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats.jsonl"
            rows = [
                {"sample_id": "a", "max_root_speed": 1.0},
                {"sample_id": "b", "max_root_speed": 3.0},
                {"sample_id": "c", "max_root_speed": 5.0},
            ]
            stats.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            payload = propose_thresholds_from_jsonl(
                stats,
                metrics=("max_root_speed",),
                percentile=0.5,
                action="quarantine",
            )

            self.assertEqual(payload["sample_count"], 3)
            self.assertEqual(payload["proposals"][0]["value"], 3.0)
            self.assertEqual(payload["proposals"][0]["action"], "quarantine")
            self.assertEqual(payload["proposals"][0]["tail"], "upper")
            self.assertEqual(payload["proposals"][0]["comparison"], ">")
            self.assertEqual(payload["group_by"], [])
            self.assertEqual(payload["groups"], {})

    def test_propose_grouped_thresholds_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats.jsonl"
            rows = [
                {"sample_id": "a", "category": "Walk", "max_root_speed": 1.0},
                {"sample_id": "b", "category": "Walk", "max_root_speed": 3.0},
                {"sample_id": "c", "category": "Jump", "max_root_speed": 8.0},
                {"sample_id": "d", "category": "Jump", "max_root_speed": 12.0},
                {"sample_id": "e", "category": "Idle", "max_root_speed": 0.5},
            ]
            stats.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            payload = propose_thresholds_from_jsonl(
                stats,
                metrics=("max_root_speed",),
                percentile=0.5,
                action="quarantine",
                group_by=("category",),
                min_group_size=2,
            )

            groups = payload["groups"]["category"]
            self.assertEqual([group["value"] for group in groups], ["Jump", "Walk"])
            self.assertEqual(groups[0]["sample_count"], 2)
            self.assertEqual(groups[0]["proposals"][0]["value"], 10.0)
            self.assertEqual(groups[1]["proposals"][0]["value"], 2.0)
            self.assertEqual(payload["grouped_rows"]["category"], 4)

    def test_propose_lower_tail_thresholds_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats = Path(tmp) / "stats.jsonl"
            rows = [
                {"sample_id": "a", "contact_frame_ratio": 0.0},
                {"sample_id": "b", "contact_frame_ratio": 0.5},
                {"sample_id": "c", "contact_frame_ratio": 1.0},
            ]
            stats.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            payload = propose_thresholds_from_jsonl(
                stats,
                metrics=(),
                lower_metrics=("contact_frame_ratio",),
                percentile=0.95,
                action="quarantine",
            )

            proposal = payload["proposals"][0]
            self.assertEqual(proposal["metric"], "contact_frame_ratio")
            self.assertEqual(proposal["tail"], "lower")
            self.assertEqual(proposal["comparison"], "<")
            self.assertAlmostEqual(proposal["percentile"], 0.05)
            self.assertEqual(proposal["value"], 0.05)
            self.assertEqual(payload["lower_metrics"], ["contact_frame_ratio"])

    def test_write_threshold_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            output = root / "thresholds.json"
            stats.write_text('{"joint_jump_rate": 0.0}\n{"joint_jump_rate": 0.2}\n', encoding="utf-8")

            write_threshold_proposals(
                stats_jsonl=stats,
                output_json=output,
                metrics=("joint_jump_rate",),
                percentile=0.5,
                group_by=("split",),
            )

            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text())["proposals"][0]["value"], 0.1)

    def test_cli_accepts_only_lower_tail_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            output = root / "thresholds.json"
            stats.write_text(
                '{"contact_frame_ratio": 0.0}\n{"contact_frame_ratio": 1.0}\n',
                encoding="utf-8",
            )

            argv = [
                "online-retarget",
                "propose-thresholds",
                "--stats-jsonl",
                str(stats),
                "--output-json",
                str(output),
                "--lower-metric",
                "contact_frame_ratio",
                "--percentile",
                "0.95",
            ]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                cli.main()

            proposal = json.loads(output.read_text(encoding="utf-8"))["proposals"][0]

        self.assertEqual(proposal["metric"], "contact_frame_ratio")
        self.assertEqual(proposal["tail"], "lower")
        self.assertEqual(proposal["comparison"], "<")

    def test_cli_requires_at_least_one_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            output = root / "thresholds.json"
            stats.write_text('{"contact_frame_ratio": 0.0}\n', encoding="utf-8")

            argv = [
                "online-retarget",
                "propose-thresholds",
                "--stats-jsonl",
                str(stats),
                "--output-json",
                str(output),
            ]
            with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as raised:
                cli.main()

        self.assertIn("at least one --metric or --lower-metric", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
