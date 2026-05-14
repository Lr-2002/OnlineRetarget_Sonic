import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.quality_summary import summarize_quality_jsonl


class QualitySummaryTests(unittest.TestCase):
    def test_summarize_quality_jsonl_counts_flags_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            stats.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "quality_action": "keep",
                                "quality_flags": "",
                                "split": "train",
                                "category": "Walk",
                                "penetration_depth": 0.0,
                                "contact_slide_rate": 0.1,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "quality_action": "quarantine",
                                "quality_flags": "g1_ground_penetration|g1_foot_slide",
                                "split": "val",
                                "category": "Stunts",
                                "penetration_depth": 0.4,
                                "contact_slide_rate": 0.9,
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = summarize_quality_jsonl(
                stats_jsonl=stats,
                output_json=root / "summary.json",
                metrics=("penetration_depth", "contact_slide_rate"),
                group_by=("split", "category"),
                quantiles=(0.5, 1.0),
            )
            payload = json.loads(result.output_json.read_text(encoding="utf-8"))

        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.action_counts, {"keep": 1, "quarantine": 1})
        self.assertEqual(result.flag_counts["g1_ground_penetration"], 1)
        self.assertEqual(payload["group_counts"]["split"], {"train": 1, "val": 1})
        self.assertEqual(payload["metric_summary"]["penetration_depth"]["max"], 0.4)
        self.assertEqual(payload["metric_summary"]["contact_slide_rate"]["p100"], 0.9)

    def test_summarize_quality_jsonl_uses_default_metrics_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            stats.write_text(
                json.dumps(
                    {
                        "quality_action": "downweight",
                        "quality_flags": "g1_foot_slide",
                        "split": "train",
                        "category": "Walk",
                        "contact_slide_rate": 0.25,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = summarize_quality_jsonl(
                stats_jsonl=stats,
                output_json=root / "summary.json",
                metrics=(),
                group_by=(),
                quantiles=(),
            )
            payload = json.loads(result.output_json.read_text(encoding="utf-8"))

        self.assertIn("contact_slide_rate", payload["metric_summary"])
        self.assertIn("split", payload["group_counts"])
        self.assertIn("p95", payload["metric_summary"]["contact_slide_rate"])

    def test_summarize_quality_jsonl_rejects_invalid_quantile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.jsonl"
            stats.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                summarize_quality_jsonl(
                    stats_jsonl=stats,
                    output_json=root / "summary.json",
                    quantiles=(1.5,),
                )


if __name__ == "__main__":
    unittest.main()
