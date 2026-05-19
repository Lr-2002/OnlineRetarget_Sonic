import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
                        "fps": 50.0,
                        "target_frame": 17,
                        "target_g1_path": "bones_sonic/clip.npz",
                        "target_joints": [0.0, 1.0],
                    }
                ],
                predictions=[[0.5, 1.5]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0]])
        self.assertEqual(payload["quality_flags"], ["source_foot_slide"])
        self.assertEqual(payload["fps"], 50.0)
        self.assertEqual(payload["target_frame"], 17)
        self.assertEqual(payload["sequence_id"], "bones_sonic/clip.npz")

    def test_wandb_disabled_by_default(self):
        run = train_entry._init_wandb(
            config={},
            quality_gate={},
            output_dir=Path("."),
            enabled=True,
        )

        self.assertIsNone(run)

    def test_apply_wandb_mode_override(self):
        config = {"tracking": {"wandb_mode": "disabled"}}

        updated = train_entry._apply_wandb_mode_override(config, "offline")

        self.assertEqual(updated["tracking"]["wandb_mode"], "offline")
        self.assertEqual(config["tracking"]["wandb_mode"], "disabled")

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
            resume_checkpoint="runs/train/prev/checkpoint.pt",
            encoder_context={"enabled": True, "num_actor_encoders": 2},
        )

        self.assertEqual(report["predictions_jsonl"], "runs/train/run/train_predictions.jsonl")
        self.assertEqual(
            report["offline_eval"]["summary_json"],
            "runs/train/run/eval/train_offline_eval/eval_summary.json",
        )
        self.assertEqual(report["quality_gate"]["policy_id"], "policy")
        self.assertEqual(report["resume_checkpoint"], "runs/train/prev/checkpoint.pt")
        self.assertEqual(report["encoder_context"]["num_actor_encoders"], 2)
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

    def test_supervised_loss_uses_explicit_l1_without_implicit_mse(self):
        prediction = _FakeTensor()
        target = object()
        calls = []
        torch = mock.Mock()
        torch.nn.functional.l1_loss.side_effect = lambda pred, tgt: calls.append("l1") or _FakeTensor()
        torch.nn.functional.mse_loss.side_effect = lambda pred, tgt: calls.append("mse") or _FakeTensor()
        torch.nn.functional.smooth_l1_loss.side_effect = (
            lambda pred, tgt: calls.append("smooth_l1") or _FakeTensor()
        )

        train_entry._supervised_loss(torch, prediction, target, {"loss": {"l1": 1.0}})

        self.assertEqual(calls, ["l1"])

    def test_previous_target_joints_prefers_sample_field(self):
        sample = {"prev_target_joints": [1, 2, 3]}

        self.assertEqual(train_entry._previous_target_joints(sample, 3), [1.0, 2.0, 3.0])

    def test_previous_target_joints_falls_back_to_zeros(self):
        self.assertEqual(train_entry._previous_target_joints({}, 2), [0.0, 0.0])

    def test_encoder_bank_context_reads_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "skeleton_registry.csv"
            with registry.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["actor_uid", "encoder_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"actor_uid": "A001", "encoder_id": "A001"},
                        {"actor_uid": "A002", "encoder_id": "A002"},
                    ]
                )
            config = {
                "data": {"skeleton_registry_csv": str(registry)},
                "model": {"skeleton_encoder_mode": "actor_bank"},
            }

            context = train_entry._encoder_bank_context(
                config,
                [{"sample_id": "s1", "actor_uid": "A002"}],
            )
            updated = train_entry._config_with_encoder_bank(config, context)

        self.assertTrue(context["enabled"])
        self.assertEqual(context["num_actor_encoders"], 2)
        self.assertEqual(context["encoder_id_to_index"]["A002"], 1)
        self.assertEqual(updated["model"]["num_actor_encoders"], 2)

    def test_encoder_bank_context_rejects_missing_registry_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "skeleton_registry.csv"
            with registry.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["actor_uid", "encoder_id"])
                writer.writeheader()
                writer.writerow({"actor_uid": "A001", "encoder_id": "A001"})
            config = {
                "data": {"skeleton_registry_csv": str(registry)},
                "model": {"skeleton_encoder_mode": "actor_bank"},
            }

            with self.assertRaises(SystemExit) as raised:
                train_entry._encoder_bank_context(
                    config,
                    [{"sample_id": "s1", "actor_uid": "A404"}],
                )

        self.assertIn("outside registry", str(raised.exception))

    def test_filter_finite_supervised_tensors_drops_bad_rows(self):
        class TorchStub:
            @staticmethod
            def isfinite(tensor):
                return tensor.isfinite()

        samples = [
            {"sample_id": "good", "source_motion_path": "a.bvh", "target_g1_path": "a.csv"},
            {"sample_id": "bad", "source_motion_path": "b.bvh", "target_g1_path": "b.csv"},
        ]
        x = _FakeMatrix([[1.0, 2.0], [math.nan, 3.0]])
        y = _FakeMatrix([[0.0], [1.0]])
        prev_y = _FakeMatrix([[0.0], [1.0]])

        filtered_samples, filtered_x, filtered_y, filtered_prev_y, report = (
            train_entry._filter_finite_supervised_tensors(
                TorchStub,
                samples=samples,
                x=x,
                y=y,
                prev_y=prev_y,
            )
        )

        self.assertEqual([sample["sample_id"] for sample in filtered_samples], ["good"])
        self.assertEqual(filtered_x.rows, [[1.0, 2.0]])
        self.assertEqual(filtered_y.rows, [[0.0]])
        self.assertEqual(filtered_prev_y.rows, [[0.0]])
        self.assertEqual(report["input_count"], 2)
        self.assertEqual(report["filtered_count"], 1)
        self.assertEqual(report["dropped_count"], 1)
        self.assertEqual(report["dropped_examples"][0]["sample_id"], "bad")
        self.assertEqual(report["dropped_examples"][0]["reasons"], ["observation_nonfinite"])


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


class _FakeTensor:
    def new_tensor(self, value):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self


class _FakeBoolVector:
    def __init__(self, values):
        self.values = [bool(value) for value in values]

    def __and__(self, other):
        return _FakeBoolVector([left and right for left, right in zip(self.values, other.values)])

    def __invert__(self):
        return _FakeBoolVector([not value for value in self.values])

    def __getitem__(self, index):
        return self.values[index]

    def sum(self):
        return _FakeScalar(sum(self.values))

    def nonzero(self, as_tuple=False):
        indices = [index for index, value in enumerate(self.values) if value]
        if as_tuple:
            return (_FakeIndexVector(indices),)
        return _FakeIndexMatrix(indices)


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeIndexVector:
    def __init__(self, indices):
        self.indices = list(indices)

    def flatten(self):
        return self

    def tolist(self):
        return list(self.indices)


class _FakeIndexMatrix(_FakeIndexVector):
    pass


class _FakeMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def isfinite(self):
        return _FakeFiniteMatrix(
            [[math.isfinite(value) for value in row] for row in self.rows]
        )

    def index_select(self, dim, indices):
        self.assert_dim_zero(dim)
        return _FakeMatrix([self.rows[index] for index in indices.tolist()])

    @staticmethod
    def assert_dim_zero(dim):
        if dim != 0:
            raise AssertionError(f"expected dim 0, got {dim}")


class _FakeFiniteMatrix:
    def __init__(self, rows):
        self.rows = rows

    def all(self, dim):
        if dim != 1:
            raise AssertionError(f"expected dim 1, got {dim}")
        return _FakeBoolVector([all(row) for row in self.rows])


if __name__ == "__main__":
    unittest.main()
