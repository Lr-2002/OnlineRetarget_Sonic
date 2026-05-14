import json
from pathlib import Path
import tempfile
import unittest

import scripts.train as train_entry


class TrainEntryTests(unittest.TestCase):
    def test_load_supervised_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"
            samples.write_text(
                json.dumps({"observation": [0.0, 1.0], "target_joints": [0.5]}) + "\n",
                encoding="utf-8",
            )

            loaded = train_entry._load_supervised_samples(samples)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["observation"], [0.0, 1.0])

    def test_load_supervised_samples_requires_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"
            samples.write_text(json.dumps({"observation": [0.0]}) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_entry._load_supervised_samples(samples)

    def test_write_prediction_jsonl_matches_eval_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.jsonl"

            train_entry._write_prediction_jsonl(
                output,
                samples=[
                    {
                        "sample_id": "s1",
                        "actor_uid": "A001",
                        "category": "Baseline",
                        "package": "Locomotion",
                        "quality_flags": ["source_foot_slide"],
                        "target_joints": [0.0, 1.0],
                    }
                ],
                predictions=[[0.5, 1.5]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0]])
        self.assertEqual(payload["quality_flags"], ["source_foot_slide"])

    def test_wandb_disabled_by_default(self):
        run = train_entry._init_wandb(
            config={},
            quality_gate={},
            output_dir=Path("."),
            enabled=True,
        )

        self.assertIsNone(run)

    def test_build_train_report_records_trace_artifacts(self):
        report = train_entry._build_train_report(
            samples_jsonl=Path("runs/samples.jsonl"),
            output_dir=Path("runs/train/run"),
            checkpoint=Path("runs/train/run/checkpoint.pt"),
            predictions_jsonl=Path("runs/train/run/train_predictions.jsonl"),
            offline_eval={"summary_json": "runs/train/run/eval/train_offline_eval/eval_summary.json"},
            sample_count=2,
            input_dim=1547,
            output_dim=29,
            max_steps=1,
            batch_size=2,
            learning_rate=3e-4,
            hidden_dims=(32,),
            dropout=0.0,
            quality_gate={"policy_id": "policy"},
            device="cpu",
            world_size=1,
            rank=0,
            final_train_mse=0.1,
            wandb_summary={"enabled": False},
        )

        self.assertEqual(report["predictions_jsonl"], "runs/train/run/train_predictions.jsonl")
        self.assertEqual(
            report["offline_eval"]["summary_json"],
            "runs/train/run/eval/train_offline_eval/eval_summary.json",
        )
        self.assertEqual(report["quality_gate"]["policy_id"], "policy")
        self.assertFalse(report["wandb"]["enabled"])

    def test_quality_gate_blocks_formal_training_without_policy(self):
        context = train_entry._quality_gate_context(
            {},
            index_csv=Path("runs/indices/split_index.csv"),
            samples_jsonl=None,
        )

        with self.assertRaises(SystemExit) as raised:
            train_entry._validate_quality_gate(context)

        self.assertIn("quality gate failed", str(raised.exception))

    def test_quality_gate_allows_debug_override(self):
        context = train_entry._quality_gate_context(
            {},
            index_csv=Path("runs/indices/split_index.csv"),
            samples_jsonl=None,
            allow_debug_data=True,
        )

        train_entry._validate_quality_gate(context)

    def test_quality_gate_reads_supervised_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_report = root / "curated_report.json"
            quality_report.write_text("{}\n", encoding="utf-8")
            quality_audit = root / "policy_audit.json"
            _write_policy_audit(quality_audit, policy_id="policy", promotable=True)
            sample_dir = root / "supervised"
            sample_dir.mkdir()
            samples = sample_dir / "samples.jsonl"
            samples.write_text("", encoding="utf-8")
            (sample_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "builder": "bvh_fk_30body_window",
                        "index_csv": "runs/curated/policy/curated_index.csv",
                        "config": {"action_column": "merged_quality_action"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            context = train_entry._quality_gate_context(
                {
                    "data": {
                        "quality_policy_id": "policy",
                        "quality_report": str(quality_report),
                        "quality_policy_audit": str(quality_audit),
                    }
                },
                index_csv=None,
                samples_jsonl=samples,
            )

        self.assertTrue(context["uses_curated_index"])
        self.assertTrue(context["uses_merged_action"])
        self.assertTrue(context["quality_report_exists"])
        self.assertTrue(context["quality_policy_audit_exists"])
        self.assertTrue(context["quality_policy_audit_promotable"])
        self.assertTrue(context["samples_builder_is_formal"])
        train_entry._validate_quality_gate(context)

    def test_quality_gate_requires_policy_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_report = root / "curated_report.json"
            quality_report.write_text("{}\n", encoding="utf-8")
            context = train_entry._quality_gate_context(
                {
                    "data": {
                        "quality_policy_id": "policy",
                        "quality_report": str(quality_report),
                    }
                },
                index_csv=Path("runs/curated/policy/curated_index.csv"),
                samples_jsonl=None,
                action_column="merged_quality_action",
            )

        with self.assertRaises(SystemExit) as raised:
            train_entry._validate_quality_gate(context)

        self.assertIn("quality_policy_audit", str(raised.exception))

    def test_quality_gate_blocks_unpromotable_policy_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_report = root / "curated_report.json"
            quality_report.write_text("{}\n", encoding="utf-8")
            quality_audit = root / "policy_audit.json"
            _write_policy_audit(
                quality_audit,
                policy_id="policy",
                promotable=False,
                blockers=["review decisions missing"],
            )
            context = train_entry._quality_gate_context(
                {
                    "data": {
                        "quality_policy_id": "policy",
                        "quality_report": str(quality_report),
                        "quality_policy_audit": str(quality_audit),
                    }
                },
                index_csv=Path("runs/curated/policy/curated_index.csv"),
                samples_jsonl=None,
                action_column="merged_quality_action",
            )

        with self.assertRaises(SystemExit) as raised:
            train_entry._validate_quality_gate(context)

        self.assertIn("promotable quality policy audit", str(raised.exception))
        self.assertIn("review decisions missing", str(raised.exception))

    def test_quality_gate_blocks_policy_audit_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_report = root / "curated_report.json"
            quality_report.write_text("{}\n", encoding="utf-8")
            quality_audit = root / "policy_audit.json"
            _write_policy_audit(quality_audit, policy_id="other_policy", promotable=True)
            context = train_entry._quality_gate_context(
                {
                    "data": {
                        "quality_policy_id": "policy",
                        "quality_report": str(quality_report),
                        "quality_policy_audit": str(quality_audit),
                    }
                },
                index_csv=Path("runs/curated/policy/curated_index.csv"),
                samples_jsonl=None,
                action_column="merged_quality_action",
            )

        with self.assertRaises(SystemExit) as raised:
            train_entry._validate_quality_gate(context)

        self.assertIn("audit matching policy_id policy", str(raised.exception))

    def test_quality_gate_blocks_raw_debug_samples_for_formal_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_report = root / "curated_report.json"
            quality_report.write_text("{}\n", encoding="utf-8")
            quality_audit = root / "policy_audit.json"
            _write_policy_audit(quality_audit, policy_id="policy", promotable=True)
            sample_dir = root / "supervised"
            sample_dir.mkdir()
            samples = sample_dir / "samples.jsonl"
            samples.write_text("", encoding="utf-8")
            (sample_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "builder": "raw_bvh_channel_debug",
                        "index_csv": "runs/curated/policy/curated_index.csv",
                        "config": {"action_column": "merged_quality_action"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            context = train_entry._quality_gate_context(
                {
                    "data": {
                        "quality_policy_id": "policy",
                        "quality_report": str(quality_report),
                        "quality_policy_audit": str(quality_audit),
                    }
                },
                index_csv=None,
                samples_jsonl=samples,
            )

        with self.assertRaises(SystemExit) as raised:
            train_entry._validate_quality_gate(context)

        self.assertIn("formal samples built by bvh_fk_30body_window", str(raised.exception))

    def test_quality_gate_allows_raw_debug_samples_with_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "supervised"
            sample_dir.mkdir()
            samples = sample_dir / "samples.jsonl"
            samples.write_text("", encoding="utf-8")
            (sample_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "builder": "raw_bvh_channel_debug",
                        "index_csv": "runs/indices/debug/split_index.csv",
                        "config": {"action_column": "curation_action"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            context = train_entry._quality_gate_context(
                {},
                index_csv=None,
                samples_jsonl=samples,
                allow_debug_data=True,
            )

        train_entry._validate_quality_gate(context)


def _write_policy_audit(
    path: Path,
    *,
    policy_id: str,
    promotable: bool,
    blockers: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "policy_id": policy_id,
                "promotable": promotable,
                "status": "promotable" if promotable else "blocked",
                "blockers": blockers or [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
