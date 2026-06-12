import contextlib
import io
import json
import math
from pathlib import Path
import sys
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

    def test_config_preset_switches_flat_and_route_b_settings(self):
        config = _preset_switch_config("flat_diffusion_policy")

        flat = train_entry._apply_config_preset(config)
        route_b = train_entry._apply_config_preset(
            {**config, "policy_preset": "route_b_temporal_diffusion"}
        )

        self.assertEqual(flat["data"]["samples_jsonl"], "runs/supervised/flat/samples.jsonl")
        self.assertEqual(flat["data"]["target_horizon_frames"], 10)
        self.assertEqual(flat["data"]["target_future_step"], 1)
        self.assertEqual(flat["data"]["source_body_token_dim"], 15)
        self.assertEqual(flat["data"]["source_rotation"], "rot6d")
        self.assertEqual(flat["data"]["action_dim"], 29)
        self.assertEqual(flat["data"]["build"]["target_future_step"], 1)
        self.assertEqual(flat["data"]["build"]["source_rotation"], "rot6d")
        self.assertEqual(flat["model"]["family"], "diffusion_policy")
        self.assertEqual(flat["model"]["hidden_dims"], [512, 512, 256])
        self.assertEqual(flat["model"]["output"], "g1_joint_position_future_window")
        self.assertEqual(flat["loss"], {"diffusion_policy": 1.0})

        self.assertEqual(route_b["data"]["samples_jsonl"], "runs/supervised/route_b/samples.jsonl")
        self.assertEqual(route_b["data"]["target_horizon_frames"], 10)
        self.assertEqual(route_b["data"]["target_future_step"], 5)
        self.assertEqual(route_b["data"]["source_body_token_dim"], 15)
        self.assertEqual(route_b["data"]["source_rotation"], "rot6d")
        self.assertEqual(route_b["data"]["action_dim"], 29)
        self.assertEqual(route_b["data"]["build"]["target_future_step"], 5)
        self.assertEqual(route_b["data"]["build"]["source_rotation"], "rot6d")
        self.assertEqual(route_b["model"]["family"], "temporal_diffusion_policy")
        self.assertEqual(route_b["model"]["action_dim"], 29)
        self.assertEqual(route_b["model"]["d_model"], 128)
        self.assertEqual(route_b["model"]["nhead"], 4)
        self.assertEqual(route_b["model"]["num_layers"], 2)
        self.assertEqual(route_b["model"]["dim_feedforward"], 256)
        self.assertEqual(route_b["model"]["robot_state_dim"], 0)
        self.assertEqual(route_b["model"]["output_mode"], "residual_prev_action")
        self.assertEqual(route_b["model"]["output"], "g1_joint_position_future_window")
        self.assertEqual(
            route_b["loss"],
            {
                "temporal_diffusion_policy": 1.0,
                "denoise": 1.0,
                "x0_reconstruction": 0.25,
                "velocity": 0.1,
                "acceleration": 0.05,
                "jerk": 0.0,
                "delta_smoothness": 0.05,
                "joint_jump": 0.02,
                "joint_jump_velocity": 20.0,
                "joint_jump_fps": 50.0,
                "joint_limit": 0.0,
            },
        )

    def test_config_preset_preserves_old_config_without_preset(self):
        config = {
            "data": {"samples_jsonl": "runs/supervised/flat/samples.jsonl"},
            "model": {"family": "diffusion_policy", "hidden_dims": [8]},
        }

        self.assertIs(train_entry._apply_config_preset(config), config)

    def test_config_preset_rejects_incomplete_route_b_group(self):
        with self.assertRaises(ValueError) as raised:
            train_entry._apply_config_preset(
                {
                    "policy_preset": "route_b_temporal_diffusion",
                    "policy_presets": {
                        "route_b_temporal_diffusion": {
                            "data": {"samples_jsonl": "runs/samples.jsonl"},
                            "model": {"family": "temporal_diffusion_policy"},
                        }
                    },
                }
            )

        self.assertIn("missing controlled keys", str(raised.exception))

    def test_load_config_fails_fast_without_pyyaml(self):
        with mock.patch.object(train_entry, "yaml", None):
            with self.assertRaises(SystemExit) as raised:
                train_entry._load_config(Path("configs/bones_sonic_diffusion_policy_debug.yaml"))

        self.assertIn("PyYAML is required to read --config", str(raised.exception))

    def test_dry_run_reports_future_horizon_output_dim_from_config(self):
        config = {
            "data": {
                "target_horizon_frames": 10,
                "allow_debug_data": True,
            },
            "model": {
                "output": "g1_joint_position_future_window",
            },
        }
        fake_yaml = mock.Mock()
        fake_yaml.safe_load.return_value = config
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("ignored: true\n", encoding="utf-8")
            with mock.patch.object(train_entry, "yaml", fake_yaml):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["train.py", "--config", str(config_path), "--dry-run"],
                ):
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        train_entry.main()

        self.assertIn("output_dim=290", stdout.getvalue())

    def test_build_train_report_records_trace_artifacts(self):
        report = train_entry._build_train_report(
            samples_jsonl=Path("runs/samples.jsonl"),
            output_dir=Path("runs/train/run"),
            checkpoint=Path("runs/train/run/checkpoint.pt"),
            predictions_jsonl=Path("runs/train/run/train_predictions.jsonl"),
            offline_eval={"summary_json": "runs/train/run/eval/train_offline_eval/eval_summary.json"},
            visualization={"enabled": True, "summary_json": "runs/train/run/visualization/visual_manifest.json"},
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
        )

        self.assertEqual(report["predictions_jsonl"], "runs/train/run/train_predictions.jsonl")
        self.assertEqual(
            report["offline_eval"]["summary_json"],
            "runs/train/run/eval/train_offline_eval/eval_summary.json",
        )
        self.assertEqual(
            report["visualization"]["summary_json"],
            "runs/train/run/visualization/visual_manifest.json",
        )
        self.assertEqual(report["quality_gate"]["policy_id"], "policy")
        self.assertEqual(report["resume_checkpoint"], "runs/train/prev/checkpoint.pt")
        self.assertFalse(report["wandb"]["enabled"])

    def test_visualization_artifacts_write_traceable_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "sequence_id": "bones_sonic/230101/walk.npz",
                        "actor_uid": "A001",
                        "category": "walk",
                        "package": "Locomotion",
                        "target_frame_indices": [10, 15],
                        "target_joint_names": ["hip", "knee"],
                        "predicted_joints": [[0.5, 1.5], [0.75, 1.75]],
                        "target_joints": [[0.0, 1.0], [1.0, 2.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {
                        "enabled": True,
                        "artifact_name": "route_b_probe",
                        "num_samples": 1,
                        "max_joints": 2,
                    }
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            csv_text = Path(result["trajectory_csv"]).read_text(encoding="utf-8")
            svg_exists = Path(result["trajectory_svg"]).exists()
            svg_text = Path(result["trajectory_svg"]).read_text(encoding="utf-8")
            html_text = Path(result["trajectory_html"]).read_text(encoding="utf-8")

        self.assertTrue(result["enabled"])
        self.assertEqual(summary["artifact_version"], "route_b_joint_trajectory_v1")
        self.assertEqual(summary["trajectory_row_count"], 4)
        self.assertIn("s1", csv_text)
        self.assertTrue(svg_exists)
        self.assertIn("trajectory preview", html_text)

    def test_visualization_artifacts_escape_markup_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "<script>alert(1)</script>",
                        "target_joint_names": ["hip<script>"],
                        "predicted_joints": [[0.5]],
                        "target_joints": [[0.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {
                        "enabled": True,
                        "artifact_name": "route_b_probe",
                        "num_samples": 1,
                        "max_joints": 1,
                    }
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )
            svg_text = Path(result["trajectory_svg"]).read_text(encoding="utf-8")
            html_text = Path(result["trajectory_html"]).read_text(encoding="utf-8")

        self.assertNotIn("<script>", svg_text)
        self.assertNotIn("<script>", html_text)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", svg_text)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html_text)

    def test_capsule_visualization_writes_blocked_manifest_without_model_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "sequence_id": "clip.npz",
                        "target_joint_names": ["j0", "j1"],
                        "predicted_joints": [[0.5, 1.5], [0.75, 1.75]],
                        "target_joints": [[0.0, 1.0], [1.0, 2.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {
                        "enabled": True,
                        "artifact_name": "route_b_probe",
                        "num_samples": 1,
                        "max_joints": 2,
                        "capsule": {
                            "enabled": True,
                            "model_xml": root / "missing_g1.xml",
                            "num_samples": 1,
                            "max_frames": 1,
                        },
                    }
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )

            capsule = result["capsule_visualization"]
            manifest = json.loads(Path(capsule["manifest_json"]).read_text(encoding="utf-8"))
            html_text = Path(capsule["html"]).read_text(encoding="utf-8")
            trajectory = json.loads(
                Path(manifest["samples"][0]["trajectory_json"]).read_text(encoding="utf-8")
            )

        self.assertEqual(capsule["status"], "blocked")
        self.assertEqual(manifest["artifact_version"], "route_b_g1_capsule_visualization_v1")
        self.assertIn("g1_model_xml is missing", manifest["message"])
        self.assertIn("Route B 3D capsule preview", html_text)
        self.assertEqual(len(trajectory["predicted_joints"]), 1)
        self.assertEqual(manifest["samples"][0]["target_render"]["status"], "blocked")

    def test_capsule_visualization_uses_sonic_capsule_renderer_hooks(self):
        class FakeRenderConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        render_calls = []

        def fake_render(**kwargs):
            video_path = kwargs["video_path"]
            video_path.write_bytes(b"mp4")
            render_calls.append(kwargs)
            return {
                "status": "ok",
                "message": "rendered",
                "video_path": str(video_path),
                "render_backend": "software_perspective_capsules",
                "frames": len(kwargs["frames"]),
            }

        deps = {
            "ReviewClipExportConfig": FakeRenderConfig,
            "load_g1_kinematic_model": lambda path: {"model_xml": str(path)},
            "_g1_capsule_edges": lambda model: (("pelvis", "torso"),),
            "_g1_capsule_frames": lambda model, trajectory: [
                {"pelvis": (0.0, 0.0, 0.0), "torso": (0.0, 0.0, 1.0)}
                for _ in trajectory
            ],
            "_render_capsule_3d_video": fake_render,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_xml = root / "g1.xml"
            model_xml.write_text("<mujoco/>\n", encoding="utf-8")
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "predicted_joints": [[0.5, 1.5], [0.75, 1.75]],
                        "target_joints": [[0.0, 1.0], [1.0, 2.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(train_entry, "_load_route_b_capsule_render_deps", return_value=deps):
                result = train_entry._write_visualization_artifacts(
                    config={
                        "visualization": {
                            "enabled": True,
                            "artifact_name": "route_b_probe",
                            "num_samples": 1,
                            "capsule": {
                                "enabled": True,
                                "model_xml": model_xml,
                                "num_samples": 1,
                                "max_frames": 2,
                            },
                        }
                    },
                    predictions_jsonl=predictions,
                    output_dir=root / "train",
                    eval_result=None,
                    run_name="train_visualization",
                )

            capsule = result["capsule_visualization"]
            videos = train_entry._capsule_video_paths(capsule)

        self.assertEqual(capsule["status"], "ok")
        self.assertEqual(len(render_calls), 2)
        self.assertEqual(len(videos), 2)
        self.assertEqual({call["label"] for call in render_calls}, {
            "Route B target G1 FK capsules",
            "Route B predicted G1 FK capsules",
        })

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

    def test_sample_manifest_contract_blocks_target_future_step_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"

            with self.assertRaises(SystemExit) as raised:
                train_entry._validate_sample_manifest_contract(
                    {"data": {"target_future_step": 5}},
                    {"target_future_step": 1},
                    samples,
                )

        self.assertIn("target_future_step mismatch", str(raised.exception))

    def test_sample_manifest_contract_requires_declared_target_future_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"

            with self.assertRaises(SystemExit) as raised:
                train_entry._validate_sample_manifest_contract(
                    {"data": {"target_future_step": 5}},
                    {},
                    samples,
                )

        self.assertIn("lacks target_future_step", str(raised.exception))

    def test_sample_manifest_contract_allows_legacy_default_future_step(self):
        train_entry._validate_sample_manifest_contract(
            {"data": {"target_future_step": 1}},
            {},
            Path("runs/samples.jsonl"),
        )

    def test_sample_manifest_contract_preserves_old_config_without_future_step(self):
        train_entry._validate_sample_manifest_contract(
            {"data": {}},
            {},
            Path("runs/samples.jsonl"),
        )

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

    def test_target_vector_flattens_future_targets(self):
        sample = {"future_target_joints": [[1, 2], [3, 4]], "target_joints": [1, 2]}

        self.assertEqual(train_entry._target_vector(sample), [1.0, 2.0, 3.0, 4.0])

    def test_temporal_diffusion_policy_keeps_future_targets_nonflat(self):
        sample = {"future_target_joints": [[1, 2], [3, 4]], "target_joints": [1, 2]}

        self.assertEqual(train_entry._configured_model_family({"model": {"family": "dp-temporal"}}), "temporal_diffusion_policy")
        self.assertEqual(train_entry._target_action_shape(sample), (2, 2))
        self.assertEqual(
            train_entry._target_action_sequence(sample),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_previous_target_vector_repeats_single_frame_for_horizon(self):
        sample = {"prev_target_joints": [1, 2]}

        self.assertEqual(train_entry._previous_target_vector(sample, 4), [1.0, 2.0, 1.0, 2.0])

    def test_write_prediction_jsonl_reshapes_future_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.jsonl"

            train_entry._write_prediction_jsonl(
                output,
                samples=[
                    {
                        "sample_id": "s1",
                        "target_joints": [0.0, 1.0],
                        "future_target_joints": [[0.0, 1.0], [2.0, 3.0]],
                        "target_frame_indices": [10, 11],
                    }
                ],
                predictions=[[0.5, 1.5, 2.5, 3.5]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5], [2.5, 3.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(payload["target_frame_indices"], [10, 11])

    def test_write_prediction_jsonl_preserves_nested_temporal_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.jsonl"

            train_entry._write_prediction_jsonl(
                output,
                samples=[
                    {
                        "sample_id": "s1",
                        "target_joints": [0.0, 1.0],
                        "future_target_joints": [[0.0, 1.0], [2.0, 3.0]],
                        "target_frame_indices": [10, 15],
                    }
                ],
                predictions=[[[0.5, 1.5], [2.5, 3.5]]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5], [2.5, 3.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(payload["target_frame_indices"], [10, 15])

    def test_previous_target_joints_falls_back_to_zeros(self):
        self.assertEqual(train_entry._previous_target_joints({}, 2), [0.0, 0.0])

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

    def test_temporal_feature_contract_reports_p0_contract(self):
        samples = [_temporal_sample()]
        tensors = _temporal_tensors()
        config = _temporal_feature_contract_config()

        report = train_entry._temporal_feature_contract_report(config, samples, tensors)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["dimensions"]["source_body_count"], 2)
        self.assertEqual(report["dimensions"]["source_body_token_dim"], 3)
        self.assertEqual(report["dimensions"]["model_robot_state_dim"], 0)
        self.assertEqual(report["dimensions"]["robot_state_tensor_dim"], 5)
        self.assertEqual(report["output_contract"]["model_output_mode"], "residual_prev_action")
        self.assertEqual(
            report["actual_condition_sample_fields"],
            [
                "source_body_tokens",
                "source_skeleton",
                "morphology",
                "prev_target_joints",
                "previous_target_joints",
                "prev_g1_joints",
            ],
        )
        self.assertEqual(
            report["actual_model_condition_tensor_keys"],
            ["source_body_tokens", "source_skeleton", "morphology", "prev_action"],
        )
        self.assertFalse(report["actor_uid_used_as_input"])
        self.assertEqual(len(report["digest"]), 64)

    def test_temporal_feature_contract_rejects_declared_target_only_condition_key(self):
        samples = [_temporal_sample()]
        tensors = _temporal_tensors()
        config = _temporal_feature_contract_config()
        config["feature_contract"]["condition_sample_keys"].append("future_target_joints")

        with self.assertRaises(SystemExit) as raised:
            train_entry._temporal_feature_contract_report(config, samples, tensors)

        self.assertIn("future_target_joints", str(raised.exception))

    def test_temporal_feature_contract_rejects_actual_target_only_condition_field(self):
        samples = [_temporal_sample()]
        tensors = _temporal_tensors()
        config = _temporal_feature_contract_config()
        actual_fields = (
            *train_entry.TEMPORAL_MODEL_CONDITION_SAMPLE_FIELDS,
            "target_frame_indices",
        )

        with mock.patch.object(
            train_entry,
            "TEMPORAL_MODEL_CONDITION_SAMPLE_FIELDS",
            actual_fields,
        ):
            with self.assertRaises(SystemExit) as raised:
                train_entry._temporal_feature_contract_report(config, samples, tensors)

        self.assertIn("actual temporal condition source fields", str(raised.exception))
        self.assertIn("target_frame_indices", str(raised.exception))

    def test_temporal_training_dataset_keeps_fps_before_target_action(self):
        tensors = {key: _FakeDeviceTensor(key) for key in train_entry.TEMPORAL_BATCH_KEYS}

        batch = train_entry._temporal_training_dataset_tensors(tensors)
        condition = train_entry._temporal_batch_to_device(batch, device="cpu")

        self.assertEqual([tensor.name for tensor in batch], list(train_entry.TEMPORAL_BATCH_KEYS))
        self.assertEqual(condition["fps"].name, "fps")
        self.assertEqual(condition["target_action"].name, "target_action")


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


def _preset_switch_config(policy_preset: str) -> dict:
    return {
        "policy_preset": policy_preset,
        "data": {
            "root": "/data",
            "index_csv": "runs/index.csv",
            "build": {"target_future_step": 5, "source_rotation": "quat"},
        },
        "policy_presets": {
            "flat_diffusion_policy": {
                "data": {
                    "samples_jsonl": "runs/supervised/flat/samples.jsonl",
                    "target_format": "bones_sonic_joint_pos_future_window",
                    "history_frames": 8,
                    "target_horizon_frames": 10,
                    "target_future_step": 1,
                    "source_body_count": 30,
                    "source_body_token_dim": 15,
                    "source_rotation": "rot6d",
                    "action_dim": 29,
                },
                "model": {
                    "family": "diffusion_policy",
                    "hidden_dims": [512, 512, 256],
                    "dropout": 0.0,
                    "time_embed_dim": 32,
                    "diffusion_steps": 32,
                    "inference_steps": 8,
                    "output": "g1_joint_position_future_window",
                },
                "loss": {"diffusion_policy": 1.0},
            },
            "route_b_temporal_diffusion": {
                "data": {
                    "samples_jsonl": "runs/supervised/route_b/samples.jsonl",
                    "target_format": "bones_sonic_joint_pos_future_window",
                    "history_frames": 8,
                    "target_horizon_frames": 10,
                    "target_future_step": 5,
                    "source_body_count": 30,
                    "source_body_token_dim": 15,
                    "source_rotation": "rot6d",
                    "action_dim": 29,
                },
                "model": {
                    "family": "temporal_diffusion_policy",
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 256,
                    "dropout": 0.0,
                    "time_embed_dim": 32,
                    "diffusion_steps": 32,
                    "inference_steps": 8,
                    "action_dim": 29,
                    "source_body_count": 30,
                    "source_body_token_dim": 15,
                    "source_skeleton_dim": 120,
                    "morphology_dim": 13,
                    "robot_state_dim": 0,
                    "output_mode": "residual_prev_action",
                    "output": "g1_joint_position_future_window",
                },
                "loss": {
                    "temporal_diffusion_policy": 1.0,
                    "denoise": 1.0,
                    "x0_reconstruction": 0.25,
                    "velocity": 0.1,
                    "acceleration": 0.05,
                    "jerk": 0.0,
                    "delta_smoothness": 0.05,
                    "joint_jump": 0.02,
                    "joint_jump_velocity": 20.0,
                    "joint_jump_fps": 50.0,
                    "joint_limit": 0.0,
                },
            },
        },
    }


def _temporal_sample() -> dict:
    return {
        "sample_id": "s1",
        "actor_uid": "A001",
        "observation": [0.0],
        "source_body_tokens": [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.1, 0.0, 0.0], [0.0, 1.1, 0.0]],
        ],
        "source_skeleton": [1.0, 2.0, 3.0, 4.0],
        "morphology": [1.7, 70.0],
        "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0],
        "prev_target_joints": [0.0, 0.1],
        "fps": 50.0,
        "target_joints": [0.1, 0.2],
        "future_target_joints": [[0.1, 0.2], [0.2, 0.4]],
        "target_frame_indices": [10, 15],
    }


def _temporal_feature_contract_config() -> dict:
    return {
        "data": {"target_format": "bones_sonic_joint_pos_future_window"},
        "model": {
            "output": "g1_joint_position_future_window",
            "output_mode": "residual_prev_action",
            "robot_state_dim": 0,
        },
        "evaluation": {
            "metrics": [
                "joint_rmse",
                "joint_velocity_rmse",
                "predicted_minus_target_joint_jump_rate",
                "max_joint_abs_error",
            ]
        },
        "feature_contract": {
            "enabled": True,
            "enforce": True,
            "condition_sample_keys": [
                "source_body_tokens",
                "source_skeleton",
                "morphology",
                "prev_target_joints",
                "previous_target_joints",
                "prev_g1_joints",
            ],
            "forbid_condition_sample_keys": [
                "target_joints",
                "future_target_joints",
                "target_frame",
                "target_frame_indices",
                "target_g1_path",
                "actor_uid",
            ],
            "robot_state_policy": "disabled",
            "expected": {
                "target_horizon_frames": 2,
                "source_body_count": 2,
                "source_body_token_dim": 3,
                "source_skeleton_dim": 4,
                "morphology_dim": 2,
                "robot_state_dim": 0,
                "action_dim": 2,
            },
            "required_eval_metrics": [
                "joint_rmse",
                "joint_velocity_rmse",
                "predicted_minus_target_joint_jump_rate",
                "max_joint_abs_error",
            ],
        },
    }


def _temporal_tensors() -> dict:
    return {
        "source_body_tokens": _FakeTemporalTensor((1, 2, 2, 3)),
        "source_skeleton": _FakeTemporalTensor((1, 4)),
        "morphology": _FakeTemporalTensor((1, 2)),
        "robot_state": _FakeTemporalTensor((1, 5)),
        "prev_action": _FakeTemporalTensor((1, 2)),
        "fps": _FakeTemporalTensor((1,)),
        "target_action": _FakeTemporalTensor((1, 2, 2)),
    }


class _FakeTemporalTensor:
    def __init__(self, shape: tuple[int, ...], abs_sum: float = 0.0):
        self.shape = shape
        self._abs_sum = abs_sum

    def abs(self):
        return self

    def sum(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def __float__(self):
        return float(self._abs_sum)


class _FakeDeviceTensor:
    def __init__(self, name: str):
        self.name = name

    def to(self, device, non_blocking=False):
        return self


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
