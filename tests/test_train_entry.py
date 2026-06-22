import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

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

    def test_load_supervised_samples_shards_ddp_without_parsing_other_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"
            lines = [
                _sample_json("s0"),
                _sample_json("s1"),
                "{not-json-on-other-rank}",
                _sample_json("s3"),
                _sample_json("s4"),
                _sample_json("s5"),
                _sample_json("s6"),
                _sample_json("s7"),
            ]
            samples.write_text("\n".join(lines) + "\n", encoding="utf-8")

            loaded, report = train_entry._load_supervised_samples_with_report(
                samples,
                rank=1,
                world_size=4,
            )

        self.assertEqual([sample["sample_id"] for sample in loaded], ["s1", "s5"])
        self.assertTrue(report["sharded"])
        self.assertEqual(report["assignment"], "nonempty_jsonl_row_index_mod_world_size")
        self.assertEqual(report["total_nonempty_rows_seen"], 8)
        self.assertEqual(report["parsed_count"], 2)
        self.assertEqual(report["materialized_count"], 2)
        self.assertEqual(report["skipped_by_shard_count"], 6)

    def test_load_supervised_samples_drops_uneven_ddp_tail_for_equal_rank_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.jsonl"
            samples.write_text(
                "\n".join(_sample_json(f"s{index}") for index in range(8)) + "\n",
                encoding="utf-8",
            )

            shards = [
                train_entry._load_supervised_samples_with_report(
                    samples,
                    rank=rank,
                    world_size=3,
                )
                for rank in range(3)
            ]

        self.assertEqual([[sample["sample_id"] for sample in loaded] for loaded, _report in shards], [
            ["s0", "s3"],
            ["s1", "s4"],
            ["s2", "s5"],
        ])
        self.assertEqual([report["materialized_count"] for _loaded, report in shards], [2, 2, 2])
        self.assertEqual([report["dropped_uneven_tail_count"] for _loaded, report in shards], [1, 1, 0])

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
        self.assertTrue(route_b["data"]["include_target_root_pose"])
        self.assertEqual(route_b["data"]["source_body_token_dim"], 15)
        self.assertEqual(route_b["data"]["source_rotation"], "rot6d")
        self.assertEqual(route_b["data"]["action_dim"], 36)
        self.assertEqual(route_b["data"]["build"]["target_future_step"], 5)
        self.assertTrue(route_b["data"]["build"]["include_target_root_pose"])
        self.assertEqual(route_b["data"]["build"]["source_rotation"], "rot6d")
        self.assertEqual(route_b["model"]["family"], "temporal_diffusion_policy")
        self.assertEqual(route_b["model"]["action_dim"], 36)
        self.assertEqual(route_b["model"]["d_model"], 128)
        self.assertEqual(route_b["model"]["nhead"], 4)
        self.assertEqual(route_b["model"]["num_layers"], 2)
        self.assertEqual(route_b["model"]["dim_feedforward"], 256)
        self.assertEqual(route_b["model"]["robot_state_dim"], 0)
        self.assertEqual(route_b["model"]["output_mode"], "residual_prev_action")
        self.assertEqual(route_b["model"]["output"], "g1_joint_root_position_future_window")
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
        self.assertFalse(report["step_profiler"]["enabled"])

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

    def test_visualization_artifacts_export_accepted_vertical_v2_bridge_assets_without_accepting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_npz = root / "target.npz"
            np.savez(
                target_npz,
                joint_pos=np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
                root_pos=np.asarray(
                    [[0.0, 0.0, 0.7], [0.1, 0.0, 0.7], [0.2, 0.0, 0.7]],
                    dtype=np.float32,
                ),
                root_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32),
                fps=np.asarray([50.0], dtype=np.float32),
            )
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "walk/probe",
                        "source_motion_path": "soma_proportional/bvh/clip.bvh",
                        "target_g1_path": str(target_npz),
                        "target_frame_indices": [0, 2],
                        "target_joint_names": ["hip", "knee"],
                        "predicted_joints": [[0.25, 1.25], [4.25, 5.25]],
                        "target_joints": [[0.0, 1.0], [4.0, 5.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            raw_trajectory = root / "train" / "visualization" / "route_b_probe" / "accepted_vertical_v2" / "clip_00_walk_probe_trajectory.npz"
            fake_bridge = types.SimpleNamespace(
                read_prediction_rows=lambda path, count=None: [json.loads(predictions.read_text(encoding="utf-8").strip())],
                rerender_prediction_row=lambda **kwargs: _fake_dp_bridge_result(
                    raw_trajectory_path=raw_trajectory,
                    metadata_path=raw_trajectory.with_suffix(".json"),
                    step=kwargs["step"],
                ),
            )

            with mock.patch.object(train_entry, "_load_lr310_dp_visual_bridge", return_value=fake_bridge):
                accepted = train_entry._write_accepted_vertical_v2_artifacts(
                    config={"visualization": {}},
                    visual_cfg={
                        "enabled": True,
                        "num_samples": 1,
                        "execute_renderers": False,
                        "skip_source_bvh_resolve": False,
                        "root_source": "target_npz_root",
                    },
                    predictions_jsonl=predictions,
                    artifact_dir=root / "train",
                    checkpoint=root / "train" / "checkpoint.pt",
                    checkpoint_step=17,
                )

            accepted_summary = json.loads(Path(accepted["summary_json"]).read_text(encoding="utf-8"))
            clip = accepted_summary["clips"][0]
            metadata = json.loads(Path(clip["metadata"]).read_text(encoding="utf-8"))
            target_motion_exists = Path(clip["target_motion_npz"]).exists()
            prediction_motion_exists = Path(clip["prediction_motion_npz"]).exists()
            raw_trajectory_exists = Path(clip["raw_trajectory_path"]).exists()
            primary_raw_trajectory_exists = Path(accepted["primary_raw_trajectory_path"]).exists()

        self.assertEqual(accepted["status"], "blocked")
        self.assertEqual(accepted["export_status"], "ok")
        self.assertFalse(accepted["review_contract"]["final_review_eligible"])
        self.assertEqual(accepted["review_contract"]["mode"], "metric_horizon_bridge_only")
        self.assertTrue(accepted["review_contract"]["native_fps_review_required"])
        self.assertEqual(len(accepted["review_contract"]["evidence"]), 1)
        self.assertFalse(accepted["review_contract"]["evidence"][0]["final_review_eligible"])
        self.assertTrue(accepted["review_contract"]["evidence"][0]["raw_trajectory_path"].endswith("_trajectory.npz"))
        self.assertEqual(accepted["clip_count"], 1)
        self.assertEqual(accepted["error_count"], 0)
        self.assertEqual(accepted["accepted_vertical_v2_ok_count"], 0)
        self.assertFalse(clip["acceptance_ok"])
        self.assertEqual(clip["accepted_vertical_v2_status"], "failed")
        self.assertTrue(target_motion_exists)
        self.assertTrue(prediction_motion_exists)
        self.assertTrue(raw_trajectory_exists)
        self.assertTrue(primary_raw_trajectory_exists)
        self.assertEqual(metadata["accepted_visual_contract"]["panels"][2]["checkpoint_step"], 17)
        self.assertEqual(metadata["inference_render"]["checkpoint_step"], 17)
        self.assertFalse(metadata["lr310_dp_prediction_bridge"]["source_bvh_resolution_skipped"])
        self.assertEqual(accepted_summary["visualization_core"], "scripts.rerender_lr310_dp_visual_validation.rerender_prediction_row")

    def test_visualization_artifacts_report_accepted_vertical_v2_export_failure_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "missing-target",
                        "target_g1_path": str(root / "missing.npz"),
                        "predicted_joints": [[0.25, 1.25]],
                        "target_joints": [[0.0, 1.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {
                        "enabled": True,
                        "accepted_vertical_v2": {
                            "enabled": True,
                            "continue_on_error": True,
                            "execute_renderers": False,
                        },
                    },
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )

        accepted = result["accepted_vertical_v2"]
        self.assertEqual(result["primary_backend"], "accepted_vertical_v2")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(accepted["status"], "failed")
        self.assertEqual(accepted["export_status"], "failed")
        self.assertEqual(accepted["clip_count"], 0)
        self.assertEqual(accepted["error_count"], 1)
        self.assertIn("target_g1_path does not exist", accepted["errors"][0]["error"])

    def test_visualization_artifacts_capsule_only_cannot_satisfy_accepted_vertical_v2_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "capsule-only",
                        "target_g1_path": str(root / "missing_target.npz"),
                        "predicted_joints": [[0.25, 1.25]],
                        "target_joints": [[0.0, 1.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {
                        "enabled": True,
                        "num_samples": 1,
                        "max_joints": 2,
                        "capsule": {"enabled": True},
                        "accepted_vertical_v2": {
                            "enabled": True,
                            "continue_on_error": True,
                            "execute_renderers": False,
                        },
                    },
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )

        self.assertEqual(result["route_visualization_status"], "ok")
        self.assertEqual(result["primary_backend"], "accepted_vertical_v2")
        self.assertEqual(result["accepted_vertical_v2"]["status"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertNotEqual(result["status"], "ok")

    def test_accepted_vertical_v2_status_helper_uses_clip_acceptance(self):
        accepted_clip = {"acceptance_ok": True}
        rejected_clip = {"acceptance_ok": False}

        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[accepted_clip],
                errors=[],
                requested_clip_count=1,
                execute_renderers=True,
                skip_source_bvh_resolve=False,
            ),
            "ok",
        )
        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[accepted_clip],
                errors=[],
                requested_clip_count=1,
                execute_renderers=True,
                skip_source_bvh_resolve=True,
            ),
            "blocked",
        )
        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[accepted_clip],
                errors=[],
                requested_clip_count=1,
                execute_renderers=False,
                skip_source_bvh_resolve=False,
            ),
            "blocked",
        )
        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[accepted_clip, rejected_clip],
                errors=[],
                requested_clip_count=2,
                execute_renderers=True,
            ),
            "partial",
        )
        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[rejected_clip],
                errors=[],
                requested_clip_count=1,
                execute_renderers=True,
            ),
            "failed",
        )
        self.assertEqual(
            train_entry._accepted_vertical_v2_summary_status(
                clips=[rejected_clip],
                errors=[],
                requested_clip_count=1,
                execute_renderers=False,
            ),
            "blocked",
        )

    def test_accepted_vertical_v2_completion_requires_renderers_resolution_and_ok_clip(self):
        self.assertTrue(
            train_entry._accepted_vertical_v2_completion_ok(
                {
                    "enabled": True,
                    "status": "ok",
                    "execute_renderers": True,
                    "skip_source_bvh_resolve": False,
                    "accepted_vertical_v2_ok_count": 1,
                }
            )
        )
        for rejected in (
            {"execute_renderers": False, "skip_source_bvh_resolve": False, "accepted_vertical_v2_ok_count": 1},
            {"execute_renderers": True, "skip_source_bvh_resolve": True, "accepted_vertical_v2_ok_count": 1},
            {"execute_renderers": True, "skip_source_bvh_resolve": False, "accepted_vertical_v2_ok_count": 0},
        ):
            payload = {"enabled": True, "status": "ok", **rejected}
            self.assertFalse(train_entry._accepted_vertical_v2_completion_ok(payload))

    def test_top_level_visual_validation_does_not_opt_in_train_closeout_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "train_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "target_g1_path": str(root / "missing.npz"),
                        "predicted_joints": [[0.25, 1.25]],
                        "target_joints": [[0.0, 1.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = train_entry._write_visualization_artifacts(
                config={
                    "visualization": {"enabled": True},
                    "visual_validation": {"enabled": True},
                },
                predictions_jsonl=predictions,
                output_dir=root / "train",
                eval_result=None,
                run_name="train_visualization",
            )

        self.assertEqual(result["accepted_vertical_v2"], {"enabled": False})

    def test_route_b_debug_config_enables_accepted_vertical_v2_export(self):
        config_text = Path("configs/bones_sonic_diffusion_policy_debug.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("visual_validation:\n  enabled: true", config_text)
        self.assertIn("output_dir: online_retarget_visual_validation", config_text)
        self.assertIn("primary_backend: accepted_vertical_v2", config_text)
        self.assertIn("capsule:\n    enabled: false", config_text)
        self.assertIn("accepted_vertical_v2:\n    enabled: true", config_text)
        self.assertIn("execute_renderers: true", config_text)
        self.assertIn("skip_source_bvh_resolve: false", config_text)
        self.assertIn("soma_render_script: scripts/render_somamesh_source.py", config_text)
        self.assertIn("isaac_render_script: scripts/render_g1_isaac_pair.py", config_text)
        self.assertIn("soma_retargeter_root:", config_text)
        self.assertIn("somamesh_usd:", config_text)
        self.assertIn("source_bvh_tar:", config_text)
        if train_entry.yaml is None:
            return
        config = train_entry._load_config(Path("configs/bones_sonic_diffusion_policy_debug.yaml"))

        native_visual_cfg = config["visual_validation"]
        visual_cfg = config["visualization"]["accepted_vertical_v2"]

        self.assertTrue(native_visual_cfg["enabled"])
        self.assertEqual(native_visual_cfg["num_videos"], 2)
        self.assertEqual(native_visual_cfg["output_dir"], "online_retarget_visual_validation")
        self.assertEqual(native_visual_cfg["readable_clip_indices"], [0])
        self.assertEqual(config["visualization"]["primary_backend"], "accepted_vertical_v2")
        self.assertFalse(config["visualization"]["capsule"]["enabled"])
        self.assertTrue(visual_cfg["enabled"])
        self.assertTrue(visual_cfg["execute_renderers"])
        self.assertFalse(visual_cfg["skip_source_bvh_resolve"])
        self.assertEqual(visual_cfg["output_dir"], "accepted_vertical_v2")
        self.assertEqual(visual_cfg["soma_render_script"], "scripts/render_somamesh_source.py")
        self.assertEqual(visual_cfg["isaac_render_script"], "scripts/render_g1_isaac_pair.py")
        self.assertIn("soma_retargeter_root", visual_cfg)
        self.assertIn("somamesh_usd", visual_cfg)
        self.assertIn("source_bvh_tar", visual_cfg)

    def test_route_b_debug_config_checkpoint_cadence_is_5000_steps(self):
        config_path = Path("configs/bones_sonic_diffusion_policy_debug.yaml")
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("checkpoint_every_steps: 5000", config_text)
        self.assertNotIn("checkpoint_every_steps: 100\n", config_text)
        if train_entry.yaml is None:
            return

        config = train_entry._load_config(config_path)
        checkpointing = train_entry._checkpointing_config(config)

        self.assertEqual(config["train"]["checkpoint_every_steps"], 5000)
        self.assertEqual(checkpointing["every_steps"], 5000)
        self.assertTrue(checkpointing["enabled"])

    def test_route_b_debug_config_periodic_eval_cadence_is_5000_steps(self):
        config_path = Path("configs/bones_sonic_diffusion_policy_debug.yaml")
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("eval_every: 5000", config_text)
        self.assertIn("every_steps: 5000", config_text)
        self.assertNotIn("eval_every: 200\n", config_text)
        if train_entry.yaml is None:
            return

        config = train_entry._load_config(config_path)
        periodic_eval = train_entry._periodic_eval_config(config)

        self.assertEqual(config["train"]["eval_every"], 5000)
        self.assertEqual(config["evaluation"]["every_steps"], 5000)
        self.assertEqual(periodic_eval["source_key"], "evaluation.every_steps")
        self.assertEqual(periodic_eval["every_steps"], 5000)
        self.assertTrue(periodic_eval["enabled"])

    def test_periodic_eval_triggers_at_5000_not_adjacent_steps(self):
        periodic_eval = train_entry._periodic_eval_config(
            {"train": {"eval_every": 200}, "evaluation": {"every_steps": 5000}}
        )

        self.assertFalse(train_entry._should_run_periodic_eval(4999, periodic_eval))
        self.assertTrue(train_entry._should_run_periodic_eval(5000, periodic_eval))
        self.assertFalse(train_entry._should_run_periodic_eval(5001, periodic_eval))
        self.assertTrue(train_entry._should_run_periodic_eval(10000, periodic_eval))

    def test_route_b_visual_smoke_config_enables_single_clip_accepted_vertical_v2_export(self):
        config_path = Path("configs/bones_sonic_temporal_diffusion_policy_vis_smoke.yaml")
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("visual_validation:\n  enabled: true", config_text)
        self.assertIn("output_dir: online_retarget_visual_validation", config_text)
        self.assertIn("primary_backend: accepted_vertical_v2", config_text)
        self.assertIn("artifact_name: route_b_walk_visual_smoke", config_text)
        self.assertIn("wandb_upload: true", config_text)
        self.assertIn("num_samples: 1", config_text)
        self.assertIn("accepted_vertical_v2:\n    enabled: true", config_text)
        self.assertIn("execute_renderers: true", config_text)
        self.assertIn("skip_source_bvh_resolve: false", config_text)
        self.assertIn("continue_on_error: true", config_text)
        self.assertIn("capsule:\n    enabled: false", config_text)
        if train_entry.yaml is None:
            return

        config = train_entry._load_config(config_path)
        native_visual_cfg = config["visual_validation"]
        visual_cfg = config["visualization"]
        accepted_cfg = visual_cfg["accepted_vertical_v2"]

        self.assertEqual(config["experiment"]["name"], "bones_sonic_temporal_diffusion_policy_vis_smoke")
        self.assertEqual(config["tracking"]["wandb_mode"], "online")
        self.assertTrue(config["data"]["allow_debug_data"])
        self.assertTrue(native_visual_cfg["enabled"])
        self.assertEqual(native_visual_cfg["num_videos"], 1)
        self.assertEqual(native_visual_cfg["output_dir"], "online_retarget_visual_validation")
        self.assertEqual(native_visual_cfg["readable_clip_indices"], [0])
        self.assertEqual(visual_cfg["primary_backend"], "accepted_vertical_v2")
        self.assertEqual(visual_cfg["artifact_name"], "route_b_walk_visual_smoke")
        self.assertEqual(visual_cfg["output_dir"], "visualization/route_b_walk_visual_smoke")
        self.assertEqual(visual_cfg["num_samples"], 1)
        self.assertTrue(visual_cfg["wandb_upload"])
        self.assertFalse(visual_cfg["capsule"]["enabled"])
        self.assertTrue(accepted_cfg["enabled"])
        self.assertEqual(accepted_cfg["num_samples"], 1)
        self.assertTrue(accepted_cfg["execute_renderers"])
        self.assertFalse(accepted_cfg["skip_source_bvh_resolve"])
        self.assertTrue(accepted_cfg["continue_on_error"])

    def test_route_b_visual_smoke_config_runs_periodic_eval_checkpoint_and_sample_cap_at_20_steps(self):
        config_path = Path("configs/bones_sonic_temporal_diffusion_policy_vis_smoke.yaml")
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn("max_steps: 20", config_text)
        self.assertIn("eval_every: 20", config_text)
        self.assertIn("checkpoint_every_steps: 20", config_text)
        self.assertIn("every_steps: 20", config_text)
        self.assertIn("periodic_max_samples: 2", config_text)
        if train_entry.yaml is None:
            return

        config = train_entry._load_config(config_path)
        periodic_eval = train_entry._periodic_eval_config(config)
        checkpointing = train_entry._checkpointing_config(config)

        self.assertEqual(config["train"]["batch_size"], 8)
        self.assertEqual(config["train"]["max_steps"], 20)
        self.assertEqual(config["train"]["eval_every"], 20)
        self.assertEqual(config["train"]["checkpoint_every_steps"], 20)
        self.assertEqual(periodic_eval["every_steps"], 20)
        self.assertEqual(periodic_eval["source_key"], "evaluation.every_steps")
        self.assertEqual(periodic_eval["max_samples"], 2)
        self.assertTrue(periodic_eval["enabled"])
        self.assertEqual(checkpointing["every_steps"], 20)
        self.assertTrue(checkpointing["enabled"])
        self.assertFalse(train_entry._should_run_periodic_eval(19, periodic_eval))
        self.assertTrue(train_entry._should_run_periodic_eval(20, periodic_eval))
        self.assertFalse(train_entry._should_run_periodic_eval(21, periodic_eval))

    def test_run_periodic_eval_if_due_calls_artifact_writer_at_5000_only(self):
        class FakeTensor:
            def __getitem__(self, _item):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[0.25, 1.25]]

        class FakeNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            @staticmethod
            def no_grad():
                return FakeNoGrad()

        class FakeModel:
            training = True

            def train(self):
                self.training = True

        calls = []

        def fake_predict(*_args, **_kwargs):
            calls.append("predict")
            return FakeTensor()

        def fake_write(**kwargs):
            calls.append(
                {
                    "step": kwargs["step"],
                    "sample_count": len(kwargs["samples"]),
                    "predictions": kwargs["predictions"],
                }
            )
            return {"step": kwargs["step"]}

        periodic_eval = train_entry._periodic_eval_config(
            {"evaluation": {"every_steps": 5000, "periodic_max_samples": 1}}
        )
        samples = [
            {"sample_id": "s1", "target_joints": [0.0, 1.0]},
            {"sample_id": "s2", "target_joints": [2.0, 3.0]},
        ]
        with mock.patch.object(train_entry, "_predict_tensor", side_effect=fake_predict):
            with mock.patch.object(
                train_entry,
                "_write_periodic_eval_artifacts",
                side_effect=fake_write,
            ):
                self.assertIsNone(
                    train_entry._run_periodic_eval_if_due(
                        torch=FakeTorch,
                        config={},
                        output_dir=Path("/tmp/out"),
                        step=4999,
                        samples=samples,
                        model=FakeModel(),
                        model_family="temporal_mlp",
                        prediction_inputs=FakeTensor(),
                        batch_size=64,
                        device="cpu",
                        periodic_eval=periodic_eval,
                        wandb_run=None,
                        sample_loader={"sharded": False, "materialized_count": len(samples)},
                        rank=0,
                        world_size=1,
                    )
                )
                summary = train_entry._run_periodic_eval_if_due(
                    torch=FakeTorch,
                    config={},
                    output_dir=Path("/tmp/out"),
                    step=5000,
                    samples=samples,
                    model=FakeModel(),
                    model_family="temporal_mlp",
                    prediction_inputs=FakeTensor(),
                    batch_size=64,
                    device="cpu",
                    periodic_eval=periodic_eval,
                    wandb_run=None,
                    sample_loader={"sharded": False, "materialized_count": len(samples)},
                    rank=0,
                    world_size=1,
                )

        self.assertEqual(summary, {"step": 5000})
        self.assertEqual(calls[0], "predict")
        self.assertEqual(
            calls[1],
            {
                "step": 5000,
                "sample_count": 1,
                "predictions": [[0.25, 1.25]],
            },
        )

    def test_run_periodic_eval_if_due_invokes_native_fps_backend_for_temporal_dp(self):
        class FakeTensor:
            def __getitem__(self, _item):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[0.25, 1.25]]

        class FakeNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            @staticmethod
            def no_grad():
                return FakeNoGrad()

        class FakeModel:
            training = True

            def train(self):
                self.training = True

        native_calls = []

        def fake_native(**kwargs):
            native_calls.append(
                {
                    "step": kwargs["step"],
                    "model_family": kwargs["model_family"],
                    "sample_count": len(kwargs["samples"]),
                }
            )
            return {
                "enabled": True,
                "status": "ok",
                "summary_json": "/tmp/out/online_retarget_visual_validation/step_00005000/summary.json",
                "videos_ok": 1,
                "videos_failed": 0,
            }

        def fake_write(**kwargs):
            return {
                "step": kwargs["step"],
                "native_fps_visual_validation": kwargs["native_fps_visual_validation"],
            }

        periodic_eval = train_entry._periodic_eval_config(
            {"evaluation": {"every_steps": 5000, "periodic_max_samples": 1}}
        )
        samples = [
            {"sample_id": "s1", "target_joints": [0.0, 1.0]},
            {"sample_id": "s2", "target_joints": [2.0, 3.0]},
        ]
        with mock.patch.object(train_entry, "_predict_tensor", return_value=FakeTensor()):
            with mock.patch.object(
                train_entry,
                "_run_temporal_native_fps_visual_validation",
                side_effect=fake_native,
            ):
                with mock.patch.object(
                    train_entry,
                    "_write_periodic_eval_artifacts",
                    side_effect=fake_write,
                ):
                    summary = train_entry._run_periodic_eval_if_due(
                        torch=FakeTorch,
                        config={"visual_validation": {"enabled": True}},
                        output_dir=Path("/tmp/out"),
                        step=5000,
                        samples=samples,
                        model=FakeModel(),
                        model_family="temporal_diffusion_policy",
                        prediction_inputs=FakeTensor(),
                        batch_size=64,
                        device="cpu",
                        periodic_eval=periodic_eval,
                        wandb_run=None,
                        sample_loader={"sharded": False, "materialized_count": len(samples)},
                        rank=0,
                        world_size=1,
                    )

        self.assertEqual(
            native_calls,
            [{"step": 5000, "model_family": "temporal_diffusion_policy", "sample_count": 1}],
        )
        self.assertTrue(summary["native_fps_visual_validation"]["enabled"])
        self.assertEqual(summary["native_fps_visual_validation"]["status"], "ok")

    def test_periodic_eval_artifacts_are_step_scoped_and_use_eval_path(self):
        class FakeEvalResult:
            def __init__(self, output_root: Path, run_name: str):
                self.output_dir = output_root / "eval" / run_name
                self.summary_json = self.output_dir / "eval_summary.json"
                self.per_sample_csv = self.output_dir / "per_sample_metrics.csv"
                self.failure_manifest_csv = self.output_dir / "failure_manifest.csv"
                self.sample_count = 1
                self.overall = {"joint_rmse": 0.25}
                self.git_sha = "fake"
                self.git_dirty = False
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.summary_json.write_text("{}\n", encoding="utf-8")

            def to_dict(self):
                return {
                    "output_dir": str(self.output_dir),
                    "summary_json": str(self.summary_json),
                    "per_sample_csv": str(self.per_sample_csv),
                    "failure_manifest_csv": str(self.failure_manifest_csv),
                    "sample_count": self.sample_count,
                    "overall": self.overall,
                    "git_sha": self.git_sha,
                    "git_dirty": self.git_dirty,
                }

        eval_calls = []

        def fake_evaluate_jsonl(*, input_jsonl, output_root, config):
            eval_calls.append(
                {
                    "input_jsonl": input_jsonl,
                    "output_root": output_root,
                    "run_name": config.run_name,
                }
            )
            return FakeEvalResult(output_root, config.run_name)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"
            periodic_eval = train_entry._periodic_eval_config(
                {"evaluation": {"every_steps": 5000}}
            )
            with mock.patch.object(train_entry, "evaluate_jsonl", side_effect=fake_evaluate_jsonl):
                with mock.patch.object(
                    train_entry,
                    "_write_visualization_artifacts",
                    return_value={"enabled": False},
                ) as visual_mock:
                    summary = train_entry._write_periodic_eval_artifacts(
                        config={"evaluation": {"every_steps": 5000}},
                        output_dir=output_dir,
                        step=5000,
                        samples=[
                            {
                                "sample_id": "s1",
                                "target_joints": [0.0, 1.0],
                            }
                        ],
                        predictions=[[0.25, 1.25]],
                        periodic_eval=periodic_eval,
                        wandb_run=None,
                        sample_scope=train_entry._periodic_eval_sample_scope(
                            rank=0,
                            world_size=1,
                            sample_loader={"sharded": False, "materialized_count": 1},
                            loaded_sample_count=1,
                            evaluated_sample_count=1,
                        ),
                        checkpoint=output_dir / "checkpoints" / "step_00005000.pt",
                    )

            predictions_path = output_dir / "periodic_eval" / "step_00005000" / "predictions.jsonl"
            summary_path = output_dir / "periodic_eval" / "step_00005000" / "periodic_eval_summary.json"
            eval_summary_path = output_dir / "eval" / "periodic_step_00005000" / "eval_summary.json"

            self.assertEqual(summary["step"], 5000)
            self.assertEqual(summary["predictions_jsonl"], str(predictions_path))
            self.assertTrue(predictions_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(eval_summary_path.exists())
            self.assertEqual(eval_calls[0]["run_name"], "periodic_step_00005000")
            self.assertEqual(eval_calls[0]["input_jsonl"], predictions_path)
            visual_mock.assert_called_once()
            self.assertEqual(
                visual_mock.call_args.kwargs["run_name"],
                "periodic_step_00005000_visualization",
            )

    def test_periodic_eval_artifacts_persist_native_fps_visual_validation_summary(self):
        class FakeEvalResult:
            def __init__(self, output_root: Path, run_name: str):
                self.output_dir = output_root / "eval" / run_name
                self.summary_json = self.output_dir / "eval_summary.json"
                self.per_sample_csv = self.output_dir / "per_sample_metrics.csv"
                self.failure_manifest_csv = self.output_dir / "failure_manifest.csv"
                self.sample_count = 1
                self.overall = {"joint_rmse": 0.25}
                self.git_sha = "fake"
                self.git_dirty = False
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.summary_json.write_text("{}\n", encoding="utf-8")

            def to_dict(self):
                return {
                    "output_dir": str(self.output_dir),
                    "summary_json": str(self.summary_json),
                    "per_sample_csv": str(self.per_sample_csv),
                    "failure_manifest_csv": str(self.failure_manifest_csv),
                    "sample_count": self.sample_count,
                    "overall": self.overall,
                    "git_sha": self.git_sha,
                    "git_dirty": self.git_dirty,
                }

        native_summary = {
            "enabled": True,
            "status": "ok",
            "summary_json": "/tmp/out/online_retarget_visual_validation/step_00005000/summary.json",
            "videos_ok": 1,
            "videos_failed": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"
            periodic_eval = train_entry._periodic_eval_config(
                {"evaluation": {"every_steps": 5000}}
            )
            with mock.patch.object(
                train_entry,
                "evaluate_jsonl",
                side_effect=lambda **kwargs: FakeEvalResult(kwargs["output_root"], kwargs["config"].run_name),
            ):
                with mock.patch.object(
                    train_entry,
                    "_write_visualization_artifacts",
                    return_value={"enabled": False},
                ):
                    summary = train_entry._write_periodic_eval_artifacts(
                        config={"evaluation": {"every_steps": 5000}},
                        output_dir=output_dir,
                        step=5000,
                        samples=[
                            {
                                "sample_id": "s1",
                                "target_joints": [0.0, 1.0],
                            }
                        ],
                        predictions=[[0.25, 1.25]],
                        periodic_eval=periodic_eval,
                        wandb_run=None,
                        sample_scope=train_entry._periodic_eval_sample_scope(
                            rank=0,
                            world_size=1,
                            sample_loader={"sharded": False, "materialized_count": 1},
                            loaded_sample_count=1,
                            evaluated_sample_count=1,
                        ),
                        native_fps_visual_validation=native_summary,
                    )

            payload = train_entry._wandb_periodic_eval_payload(
                step=5000,
                summary=summary,
                eval_result=types.SimpleNamespace(overall={"joint_rmse": 0.25}, summary_json="eval.json"),
                visualization={"enabled": False},
            )
            persisted = json.loads(
                (output_dir / "periodic_eval" / "step_00005000" / "periodic_eval_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(persisted["native_fps_visual_validation"]["status"], "ok")
        self.assertEqual(
            payload["periodic_eval/native_fps_visual_validation_status"],
            "ok",
        )
        self.assertEqual(
            payload["periodic_eval/native_fps_visual_validation_summary_json"],
            native_summary["summary_json"],
        )
        self.assertEqual(payload["periodic_eval/native_fps_visual_validation_videos_ok"], 1)

    def test_periodic_eval_summary_log_and_wandb_payload_mark_shard_scope(self):
        class FakeEvalResult:
            output_dir = Path("/tmp/out/eval/periodic_step_00005000")
            summary_json = output_dir / "eval_summary.json"
            per_sample_csv = output_dir / "per_sample_metrics.csv"
            failure_manifest_csv = output_dir / "failure_manifest.csv"
            sample_count = 2
            overall = {"joint_rmse": 0.5}
            git_sha = "fake"
            git_dirty = False

            def to_dict(self):
                return {
                    "output_dir": str(self.output_dir),
                    "summary_json": str(self.summary_json),
                    "per_sample_csv": str(self.per_sample_csv),
                    "failure_manifest_csv": str(self.failure_manifest_csv),
                    "sample_count": self.sample_count,
                    "overall": self.overall,
                    "git_sha": self.git_sha,
                    "git_dirty": self.git_dirty,
                }

        logged_payloads = []

        class FakeRun:
            def log(self, payload, step=None):
                logged_payloads.append((payload, step))

            def save(self, _path):
                return None

        sample_scope = train_entry._periodic_eval_sample_scope(
            rank=0,
            world_size=4,
            sample_loader={
                "rank": 0,
                "world_size": 4,
                "sharded": True,
                "materialized_count": 2,
            },
            loaded_sample_count=2,
            evaluated_sample_count=2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"
            fake_eval_summary = output_dir / "eval" / "periodic_step_00005000" / "eval_summary.json"
            fake_eval_summary.parent.mkdir(parents=True, exist_ok=True)
            fake_eval_summary.write_text("{}\n", encoding="utf-8")
            FakeEvalResult.output_dir = fake_eval_summary.parent
            FakeEvalResult.summary_json = fake_eval_summary
            FakeEvalResult.per_sample_csv = fake_eval_summary.parent / "per_sample_metrics.csv"
            FakeEvalResult.failure_manifest_csv = fake_eval_summary.parent / "failure_manifest.csv"
            with mock.patch.object(train_entry, "evaluate_jsonl", return_value=FakeEvalResult()):
                with mock.patch.object(
                    train_entry,
                    "_write_visualization_artifacts",
                    return_value={"enabled": False},
                ):
                    with mock.patch.object(train_entry, "_wandb_log_visualization"):
                        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                            summary = train_entry._write_periodic_eval_artifacts(
                                config={"evaluation": {"every_steps": 5000}},
                                output_dir=output_dir,
                                step=5000,
                                samples=[
                                    {"sample_id": "s1", "target_joints": [0.0]},
                                    {"sample_id": "s2", "target_joints": [1.0]},
                                ],
                                predictions=[[0.0], [1.0]],
                                periodic_eval=train_entry._periodic_eval_config(
                                    {"evaluation": {"every_steps": 5000}}
                                ),
                                wandb_run=FakeRun(),
                                sample_scope=sample_scope,
                            )

            summary_path = output_dir / "periodic_eval" / "step_00005000" / "periodic_eval_summary.json"
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            eval_summary = json.loads(fake_eval_summary.read_text(encoding="utf-8"))
            event = json.loads(
                [line for line in stdout.getvalue().splitlines() if line.strip()][-1]
            )
            wandb_payload, wandb_step = logged_payloads[0]

        self.assertEqual(summary["sample_scope"]["scope"], "rank_shard")
        self.assertEqual(persisted["sample_scope"]["rank"], 0)
        self.assertEqual(persisted["sample_scope"]["world_size"], 4)
        self.assertTrue(persisted["sample_scope"]["sample_loader_sharded"])
        self.assertEqual(persisted["sample_scope"]["loaded_sample_count"], 2)
        self.assertEqual(persisted["sample_scope"]["evaluated_sample_count"], 2)
        self.assertEqual(eval_summary["metric_scope"], "rank_shard")
        self.assertEqual(eval_summary["sample_scope"]["rank"], 0)
        self.assertTrue(eval_summary["sample_scope"]["sample_loader"]["sharded"])
        self.assertEqual(event["sample_scope"]["scope"], "rank_shard")
        self.assertTrue(event["sample_scope"]["sample_loader_sharded"])
        self.assertEqual(wandb_step, 5000)
        self.assertEqual(wandb_payload["periodic_eval/scope"], "rank_shard")
        self.assertEqual(wandb_payload["periodic_eval/rank"], 0)
        self.assertEqual(wandb_payload["periodic_eval/world_size"], 4)
        self.assertTrue(wandb_payload["periodic_eval/sample_loader_sharded"])
        self.assertTrue(wandb_payload["periodic_eval/sample_loader/sharded"])
        self.assertEqual(wandb_payload["periodic_eval/loaded_sample_count"], 2)
        self.assertEqual(wandb_payload["periodic_eval/evaluated_sample_count"], 2)
        self.assertEqual(wandb_payload["periodic_eval/rank_loaded_sample_count"], 2)
        self.assertEqual(wandb_payload["periodic_eval/rank_evaluated_sample_count"], 2)
        self.assertEqual(wandb_payload["periodic_eval/rank_materialized_sample_count"], 2)

    def test_periodic_eval_writes_step_scoped_visualization_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"
            config = {
                "evaluation": {"every_steps": 5000},
                "visualization": {
                    "enabled": True,
                    "num_samples": 1,
                    "max_joints": 2,
                    "wandb_upload": False,
                },
            }
            summary = train_entry._write_periodic_eval_artifacts(
                config=config,
                output_dir=output_dir,
                step=5000,
                samples=[
                    {
                        "sample_id": "s1",
                        "target_joints": [0.0, 1.0],
                    }
                ],
                predictions=[[0.25, 1.25]],
                periodic_eval=train_entry._periodic_eval_config(config),
                wandb_run=None,
                sample_scope=train_entry._periodic_eval_sample_scope(
                    rank=0,
                    world_size=1,
                    sample_loader={"sharded": False, "materialized_count": 1},
                    loaded_sample_count=1,
                    evaluated_sample_count=1,
                ),
            )

            visual_dir = output_dir / "periodic_eval" / "step_00005000" / "visualization"
            visual_summary = visual_dir / "visual_manifest.json"

            self.assertTrue(visual_summary.exists())
            self.assertTrue((visual_dir / "trajectory_preview.csv").exists())
            self.assertTrue((visual_dir / "trajectory_preview.svg").exists())
            self.assertTrue((visual_dir / "trajectory_preview.html").exists())
            self.assertEqual(
                Path(summary["visualization"]["summary_json"]).resolve(),
                visual_summary.resolve(),
            )
            self.assertEqual(
                Path(summary["visualization"]["output_dir"]).resolve(),
                visual_dir.resolve(),
            )

    def test_periodic_visualization_wandb_uses_periodic_namespace(self):
        logged_payloads = []

        class FakeRun:
            def log(self, payload, step=None):
                logged_payloads.append((payload, step))

            def save(self, _path):
                return None

        visualization = {
            "enabled": True,
            "status": "ok",
            "primary_backend": "accepted_vertical_v2",
            "route_visualization_status": "ok",
            "summary_json": "/tmp/periodic/visual_manifest.json",
            "sample_count": 1,
            "trajectory_row_count": 2,
            "capsule_visualization": {"status": "blocked", "manifest_json": ""},
        }

        train_entry._wandb_log_visualization(
            FakeRun(),
            visualization,
            {"visualization": {"wandb_upload": True}},
            key_prefix="periodic_eval/visualization",
        )

        payload, step = logged_payloads[0]
        self.assertIsNone(step)
        self.assertIn("periodic_eval/visualization/status", payload)
        self.assertEqual(
            payload["periodic_eval/visualization/primary_backend"],
            "accepted_vertical_v2",
        )
        self.assertEqual(payload["periodic_eval/visualization/route_status"], "ok")
        self.assertIn("periodic_eval/visualization/summary_json", payload)
        self.assertNotIn("visualization/status", payload)
        self.assertNotIn("visualization/summary_json", payload)

    def test_periodic_visualization_wandb_logs_accepted_vertical_v2_media(self):
        logged_payloads = []
        saved_paths = []

        class FakeRun:
            def log(self, payload, step=None):
                logged_payloads.append((payload, step))

            def save(self, path):
                saved_paths.append(path)

        class FakeVideo:
            def __init__(self, path):
                self.path = path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "summary_json": root / "lr310_dp_visual_validation_summary.json",
                "metadata": root / "walk__step_00005000__vertical_somamesh_g1target_g1kinematics.json",
                "combined_video": root / "walk__step_00005000__vertical_somamesh_g1target_g1kinematics.mp4",
                "row1_soma_somamesh_video": root / "walk__step_00005000__row1_soma_somamesh.mp4",
                "row2_g1_target_isaaclab_video": root / "walk__step_00005000__row2_g1_target_isaaclab.mp4",
                "row3_g1_kinematics_isaaclab_video": root / "walk__step_00005000__row3_g1_kinematics_isaaclab.mp4",
            }
            for path in paths.values():
                path.write_bytes(b"artifact")
            visualization = {
                "enabled": True,
                "status": "ok",
                "primary_backend": "accepted_vertical_v2",
                "route_visualization_status": "ok",
                "summary_json": str(root / "visual_manifest.json"),
                "sample_count": 1,
                "trajectory_row_count": 0,
                "accepted_vertical_v2": {
                    "enabled": True,
                    "status": "ok",
                    "summary_json": str(paths["summary_json"]),
                    "primary_metadata": str(paths["metadata"]),
                    "primary_video": str(paths["combined_video"]),
                    "primary_row1_soma_somamesh_video": str(paths["row1_soma_somamesh_video"]),
                    "primary_row2_g1_target_isaaclab_video": str(paths["row2_g1_target_isaaclab_video"]),
                    "primary_row3_g1_kinematics_isaaclab_video": str(paths["row3_g1_kinematics_isaaclab_video"]),
                    "clips": [
                        {
                            "metadata": str(paths["metadata"]),
                            "combined_video": str(paths["combined_video"]),
                            "row1_soma_somamesh_video": str(paths["row1_soma_somamesh_video"]),
                            "row2_g1_target_isaaclab_video": str(paths["row2_g1_target_isaaclab_video"]),
                            "row3_g1_kinematics_isaaclab_video": str(paths["row3_g1_kinematics_isaaclab_video"]),
                        }
                    ],
                },
                "capsule_visualization": {"status": "disabled", "manifest_json": ""},
            }
            fake_wandb = types.SimpleNamespace(Video=FakeVideo)
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                train_entry._wandb_log_visualization(
                    FakeRun(),
                    visualization,
                    {"visualization": {"wandb_upload": True}},
                    key_prefix="periodic_eval/visualization",
                )

        payload, step = logged_payloads[0]
        self.assertIsNone(step)
        self.assertEqual(
            payload["periodic_eval/visualization/accepted_vertical_v2/primary_video"],
            str(paths["combined_video"]),
        )
        for key in (
            "periodic_eval/visualization/accepted_vertical_v2/primary",
            "periodic_eval/visualization/accepted_vertical_v2/row1_soma_somamesh",
            "periodic_eval/visualization/accepted_vertical_v2/row2_g1_target",
            "periodic_eval/visualization/accepted_vertical_v2/row3_g1_kinematics",
        ):
            self.assertIsInstance(payload[key], FakeVideo)
        self.assertTrue(any(str(paths["combined_video"]) == path for path in saved_paths))
        self.assertTrue(any(str(paths["row1_soma_somamesh_video"]) == path for path in saved_paths))
        self.assertTrue(any(str(paths["row2_g1_target_isaaclab_video"]) == path for path in saved_paths))
        self.assertTrue(any(str(paths["row3_g1_kinematics_isaaclab_video"]) == path for path in saved_paths))

    def test_periodic_visualization_prefers_native_fps_primary_when_available(self):
        native = {
            "enabled": True,
            "status": "ok",
            "summary_json": "/tmp/out/online_retarget_visual_validation/step_00005000/summary.json",
            "clips": [
                {
                    "status": "ok",
                    "readable_video_path": "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
                    "triplet_video_path": "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00.mp4",
                    "frame_count": 200,
                    "physical_time_aligned": True,
                    "prediction_root_pose_source": "predictions_jsonl",
                    "review_contract": {"mode": "native_fps_contiguous_rollout"},
                }
            ],
        }
        selected = train_entry._select_periodic_visualization_primary(
            visualization={
                "enabled": True,
                "status": "failed",
                "primary_backend": "accepted_vertical_v2",
                "route_visualization_status": "ok",
                "accepted_vertical_v2": {
                    "enabled": True,
                    "primary_video": "bridge.mp4",
                    "primary_metadata": "bridge.json",
                },
            },
            native_fps_visual_validation=native,
        )

        self.assertEqual(selected["primary_backend"], "native_fps_contiguous_rollout")
        self.assertEqual(selected["status"], "ok")
        self.assertEqual(
            selected["primary_video"],
            "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
        )
        self.assertEqual(
            selected["primary_readable_video"],
            "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
        )
        self.assertEqual(selected["primary_triplet_video"], "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00.mp4")
        self.assertEqual(selected["primary_frame_count"], 200)
        self.assertTrue(selected["primary_physical_time_aligned"])
        self.assertEqual(selected["primary_prediction_root_pose_source"], "predictions_jsonl")
        self.assertEqual(
            selected["accepted_vertical_v2"]["primary_video"],
            "/tmp/out/online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
        )
        self.assertEqual(selected["accepted_vertical_v2"]["bridge_primary_video"], "bridge.mp4")
        self.assertEqual(selected["accepted_vertical_v2"]["primary_frame_count"], 200)
        self.assertTrue(selected["accepted_vertical_v2"]["primary_physical_time_aligned"])
        self.assertEqual(
            selected["accepted_vertical_v2"]["primary_prediction_root_pose_source"],
            "predictions_jsonl",
        )

    def test_periodic_visualization_wandb_logs_native_fps_primary_without_bridge_primary_alias(self):
        logged_payloads = []
        saved_paths = []

        class FakeRun:
            def log(self, payload, step=None):
                logged_payloads.append((payload, step))

            def save(self, path, base_path=None):
                saved_paths.append((path, base_path))

        class FakeVideo:
            def __init__(self, path):
                self.path = path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_video = root / "online_retarget_visual_validation" / "step_00005000" / "rank_000" / "clip_00.mp4"
            primary_readable = root / "online_retarget_visual_validation" / "step_00005000" / "rank_000" / "clip_00_readable.mp4"
            bridge_video = root / "accepted_vertical_v2.mp4"
            primary_video.parent.mkdir(parents=True, exist_ok=True)
            for path in (primary_video, primary_readable, bridge_video):
                path.write_bytes(b"artifact")
            visualization = {
                "enabled": True,
                "status": "ok",
                "primary_backend": "native_fps_contiguous_rollout",
                "route_visualization_status": "ok",
                "summary_json": str(root / "visual_manifest.json"),
                "primary_video": str(primary_video),
                "primary_readable_video": str(primary_readable),
                "accepted_vertical_v2": {
                    "enabled": True,
                    "status": "failed",
                    "primary_video": str(primary_readable),
                    "bridge_primary_video": str(bridge_video),
                    "primary_frame_count": 200,
                    "primary_physical_time_aligned": True,
                    "primary_prediction_root_pose_source": "predictions_jsonl",
                },
                "capsule_visualization": {"status": "disabled", "manifest_json": ""},
            }
            fake_wandb = types.SimpleNamespace(Video=FakeVideo)
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                train_entry._wandb_log_visualization(
                    FakeRun(),
                    visualization,
                    {"visualization": {"wandb_upload": True}},
                    key_prefix="periodic_eval/visualization",
                )

        payload, step = logged_payloads[0]
        self.assertIsNone(step)
        self.assertEqual(payload["periodic_eval/visualization/primary_video"], str(primary_video))
        self.assertEqual(
            payload["periodic_eval/visualization/primary_readable_video"],
            str(primary_readable),
        )
        self.assertIsInstance(payload["periodic_eval/visualization/primary"], FakeVideo)
        self.assertIsInstance(payload["periodic_eval/visualization/primary_readable"], FakeVideo)
        self.assertIsInstance(payload["periodic_eval/visualization/accepted_vertical_v2/primary"], FakeVideo)
        self.assertEqual(payload["periodic_eval/visualization/accepted_vertical_v2/primary_video"], str(primary_readable))
        self.assertEqual(payload["periodic_eval/visualization/accepted_vertical_v2/primary_frame_count"], 200)
        self.assertTrue(payload["periodic_eval/visualization/accepted_vertical_v2/primary_physical_time_aligned"])
        self.assertEqual(
            payload["periodic_eval/visualization/accepted_vertical_v2/primary_prediction_root_pose_source"],
            "predictions_jsonl",
        )
        self.assertTrue(
            any(
                path == "clip_00.mp4"
                and base_path is not None
                and base_path.endswith(
                    "online_retarget_visual_validation/step_00005000/rank_000"
                )
                for path, base_path in saved_paths
            )
        )
        self.assertTrue(
            any(
                path == "clip_00_readable.mp4"
                and base_path is not None
                and base_path.endswith(
                    "online_retarget_visual_validation/step_00005000/rank_000"
                )
                for path, base_path in saved_paths
            )
        )

    def test_wandb_save_avoids_same_base_path_glob_for_external_single_file(self):
        saved_calls = []

        class FakeRun:
            def save(self, path, base_path=None):
                if base_path is None:
                    raise AssertionError("expected base_path for single-file artifact save")
                if Path(path) == Path(base_path):
                    raise ValueError("Glob cannot be the same as the base path")
                saved_calls.append((path, base_path))

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "online_retarget_visual_validation" / "step_00000020" / "rank_000" / "clip_00_readable.mp4"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"artifact")

            train_entry._wandb_save(FakeRun(), artifact)

        self.assertEqual(
            saved_calls,
            [("clip_00_readable.mp4", str(artifact.parent.resolve()))],
        )

    def test_periodic_visualization_wandb_skips_empty_media_paths_instead_of_saving_cwd(self):
        logged_payloads = []
        saved_paths = []

        class FakeRun:
            def log(self, payload, step=None):
                logged_payloads.append((payload, step))

            def save(self, path, base_path=None):
                if base_path is None:
                    raise AssertionError("expected base_path for save call")
                if Path(path) == Path(base_path):
                    raise ValueError("Glob cannot be the same as the base path")
                saved_paths.append((path, base_path))

        visualization = {
            "enabled": True,
            "status": "ok",
            "primary_backend": "native_fps_contiguous_rollout",
            "route_visualization_status": "ok",
            "summary_json": "",
            "primary_video": "",
            "primary_readable_video": "",
            "trajectory_html": "",
            "capsule_visualization": {"status": "disabled", "manifest_json": "", "html": ""},
            "accepted_vertical_v2": {
                "enabled": True,
                "status": "failed",
                "primary_video": "",
            },
        }

        train_entry._wandb_log_visualization(
            FakeRun(),
            visualization,
            {"visualization": {"wandb_upload": True}},
            key_prefix="periodic_eval/visualization",
        )

        self.assertEqual(saved_paths, [])
        payload, step = logged_payloads[0]
        self.assertIsNone(step)
        self.assertEqual(payload["periodic_eval/visualization/primary_video"], "")
        self.assertEqual(payload["periodic_eval/visualization/primary_readable_video"], "")

    def test_periodic_eval_wandb_payload_records_visualization_primary_backend(self):
        payload = train_entry._wandb_periodic_eval_payload(
            step=5000,
            summary={
                "predictions_jsonl": "predictions.jsonl",
                "summary_json": "periodic_eval_summary.json",
                "sample_count": 1,
                "status": "ok",
            },
            eval_result=None,
            visualization={
                "enabled": True,
                "status": "failed",
                "primary_backend": "accepted_vertical_v2",
                "route_visualization_status": "ok",
                "summary_json": "visual_manifest.json",
                "accepted_vertical_v2": {
                    "enabled": True,
                    "status": "failed",
                    "export_status": "failed",
                    "primary_video": "vertical_somamesh_g1target_g1kinematics.mp4",
                    "accepted_vertical_v2_ok_count": 0,
                },
            },
        )

        self.assertEqual(payload["periodic_eval/visualization_status"], "failed")
        self.assertEqual(
            payload["periodic_eval/visualization_primary_backend"],
            "accepted_vertical_v2",
        )
        self.assertEqual(payload["periodic_eval/visualization_route_status"], "ok")
        self.assertEqual(payload["periodic_eval/accepted_vertical_v2_status"], "failed")
        self.assertEqual(payload["periodic_eval/accepted_vertical_v2_export_status"], "failed")
        self.assertEqual(
            payload["periodic_eval/accepted_vertical_v2/primary_video"],
            "vertical_somamesh_g1target_g1kinematics.mp4",
        )

    def test_periodic_eval_wandb_payload_exposes_selected_primary_video(self):
        payload = train_entry._wandb_periodic_eval_payload(
            step=5000,
            summary={
                "predictions_jsonl": "predictions.jsonl",
                "summary_json": "periodic_eval_summary.json",
                "sample_count": 1,
                "status": "ok",
            },
            eval_result=None,
            visualization={
                "enabled": True,
                "status": "ok",
                "primary_backend": "native_fps_contiguous_rollout",
                "route_visualization_status": "ok",
                "summary_json": "visual_manifest.json",
                "primary_video": "online_retarget_visual_validation/step_00005000/rank_000/clip_00.mp4",
                "primary_readable_video": "online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
                "accepted_vertical_v2": {
                    "enabled": True,
                    "status": "failed",
                    "export_status": "failed",
                    "primary_video": "online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
                    "primary_frame_count": 200,
                    "primary_physical_time_aligned": True,
                    "primary_prediction_root_pose_source": "predictions_jsonl",
                    "accepted_vertical_v2_ok_count": 0,
                },
            },
        )

        self.assertEqual(
            payload["periodic_eval/visualization_primary_backend"],
            "native_fps_contiguous_rollout",
        )
        self.assertEqual(
            payload["periodic_eval/visualization/primary_video"],
            "online_retarget_visual_validation/step_00005000/rank_000/clip_00.mp4",
        )
        self.assertEqual(
            payload["periodic_eval/visualization/primary_readable_video"],
            "online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
        )
        self.assertEqual(
            payload["periodic_eval/accepted_vertical_v2/primary_video"],
            "online_retarget_visual_validation/step_00005000/rank_000/clip_00_readable.mp4",
        )
        self.assertEqual(payload["periodic_eval/accepted_vertical_v2/primary_frame_count"], 200)
        self.assertTrue(payload["periodic_eval/accepted_vertical_v2/primary_physical_time_aligned"])
        self.assertEqual(
            payload["periodic_eval/accepted_vertical_v2/primary_prediction_root_pose_source"],
            "predictions_jsonl",
        )

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
        sample = {
            "future_target_joints": [[1, 2], [3, 4]],
            "future_target_root_pos_w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "future_target_root_quat_w": [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            "target_joints": [1, 2],
        }

        self.assertEqual(train_entry._configured_model_family({"model": {"family": "dp-temporal"}}), "temporal_diffusion_policy")
        self.assertEqual(train_entry._target_action_shape(sample), (2, 9))
        self.assertEqual(
            train_entry._target_action_sequence(sample),
            [
                [1.0, 2.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 0.4, 0.5, 0.6, 0.9, 0.1, 0.0, 0.0],
            ],
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
                        "future_target_root_pos_w": [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
                        "future_target_root_quat_w": [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                        "target_frame_indices": [10, 11],
                    }
                ],
                predictions=[[0.5, 1.5, 20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0, 2.5, 3.5, 23.0, 24.0, 25.0, 0.9, 0.1, 0.0, 0.0]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5], [2.5, 3.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(payload["pred_root_pos_w"], [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]])
        self.assertEqual(payload["prediction_root_pose_source"], "predictions_jsonl")
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
                        "future_target_root_pos_w": [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
                        "future_target_root_quat_w": [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                        "target_frame_indices": [10, 15],
                    }
                ],
                predictions=[[[0.5, 1.5, 20.0, 21.0, 22.0, 1.0, 0.0, 0.0, 0.0], [2.5, 3.5, 23.0, 24.0, 25.0, 0.9, 0.1, 0.0, 0.0]]],
            )

            payload = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["predicted_joints"], [[0.5, 1.5], [2.5, 3.5]])
        self.assertEqual(payload["target_joints"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(payload["pred_root_pos_w"], [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]])
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
        self.assertEqual(report["output_contract"]["prediction_export"], "absolute_g1_joint_root_position_future_window")
        self.assertEqual(
            report["actual_condition_sample_fields"],
            [
                "source_body_tokens",
                "source_skeleton",
                "morphology",
                "prev_target_action",
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

    def test_temporal_condition_tensors_emits_tensorization_progress(self):
        samples = [_temporal_sample(), {**_temporal_sample(), "sample_id": "s2"}]
        events = []

        tensors = train_entry._temporal_condition_tensors(
            _FakeTensorizeTorch,
            samples,
            progress=lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        self.assertEqual(
            [event["stage"] for event in events],
            [
                "shape_infer_done",
                "rows_begin",
                "rows_progress",
                "rows_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_begin",
                "tensor_convert_key_done",
                "tensor_convert_done",
            ],
        )
        self.assertEqual(events[0]["total_count"], 2)
        self.assertEqual(events[0]["target_horizon"], 2)
        self.assertEqual(events[2]["processed_count"], 2)
        converted = [event for event in events if event["stage"] == "tensor_convert_key_done"]
        self.assertEqual([event["key"] for event in converted], list(train_entry.TEMPORAL_BATCH_KEYS))
        self.assertEqual(converted[0]["tensor"]["shape"], [2, 2, 2, 3])
        self.assertEqual(tensors["target_action"].shape, (2, 2, 2))
        self.assertTrue(all("elapsed_seconds" in event for event in events))

    def test_temporal_training_dataset_keeps_fps_before_target_action(self):
        tensors = {key: _FakeDeviceTensor(key) for key in train_entry.TEMPORAL_BATCH_KEYS}

        batch = train_entry._temporal_training_dataset_tensors(tensors)
        condition = train_entry._temporal_batch_to_device(batch, device="cpu")

        self.assertEqual([tensor.name for tensor in batch], list(train_entry.TEMPORAL_BATCH_KEYS))
        self.assertEqual(condition["fps"].name, "fps")
        self.assertEqual(condition["target_action"].name, "target_action")

    def test_temporal_batch_to_device_uses_configured_non_blocking_flag(self):
        batch = tuple(_FakeDeviceTensor(key) for key in train_entry.TEMPORAL_BATCH_KEYS)

        train_entry._temporal_batch_to_device(batch, device="cuda:0", non_blocking=False)

        for tensor in batch:
            self.assertEqual(tensor.to_calls, [{"device": "cuda:0", "non_blocking": False}])

    def test_temporal_feed_config_enables_prebatch_only_with_preload(self):
        feed = train_entry._temporal_feed_config(
            {
                "train": {
                    "batch_to_device": {
                        "preload_tensors": True,
                        "prebatch_in_memory": True,
                    }
                }
            },
            {"device_type": "cuda"},
        )

        self.assertTrue(feed["preload_tensors"])
        self.assertTrue(feed["prebatch_in_memory"])
        self.assertTrue(feed["prebatch_in_memory_requested"])

    def test_temporal_feed_config_skips_prebatch_without_preload(self):
        feed = train_entry._temporal_feed_config(
            {
                "train": {
                    "batch_to_device": {
                        "prebatch_in_memory": True,
                    }
                }
            },
            {"device_type": "cuda"},
        )

        self.assertFalse(feed["prebatch_in_memory"])
        self.assertEqual(feed["prebatch_in_memory_skip_reason"], "preload_tensors_disabled")

    def test_train_dataloader_kwargs_exposes_worker_and_prefetch_knobs(self):
        kwargs, report = train_entry._train_dataloader_kwargs(
            {
                "train": {
                    "dataloader": {
                        "num_workers": 4,
                        "prefetch_factor": 3,
                        "persistent_workers": True,
                        "pin_memory": True,
                        "drop_last": True,
                    }
                }
            },
            {"device_type": "cuda"},
            dataset_length=1000,
            batch_size=128,
        )

        self.assertEqual(kwargs["batch_size"], 128)
        self.assertEqual(kwargs["num_workers"], 4)
        self.assertEqual(kwargs["prefetch_factor"], 3)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["drop_last"])
        self.assertEqual(report["requested_num_workers"], 4)
        self.assertEqual(report["prefetch_factor"], 3)

    def test_train_dataloader_kwargs_forces_single_process_for_preloaded_tensors(self):
        kwargs, report = train_entry._train_dataloader_kwargs(
            {
                "train": {
                    "dataloader": {
                        "num_workers": 4,
                        "prefetch_factor": 2,
                        "persistent_workers": True,
                        "pin_memory": True,
                    }
                }
            },
            {"device_type": "cuda"},
            dataset_length=64,
            batch_size=256,
            force_single_process=True,
        )

        self.assertEqual(kwargs["batch_size"], 64)
        self.assertEqual(kwargs["num_workers"], 0)
        self.assertFalse(kwargs["pin_memory"])
        self.assertNotIn("prefetch_factor", kwargs)
        self.assertEqual(
            report["forced_single_process_reason"],
            "materialized_tensors_preloaded_to_device",
        )

    def test_step_profiler_config_defaults_cuda_sync_when_enabled(self):
        report = train_entry._step_profiler_config(
            {"train": {"step_profiler": {"enabled": True, "active_steps": 300, "warmup_steps": 10}}},
            {"device_type": "cuda"},
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["active_steps"], 300)
        self.assertEqual(report["warmup_steps"], 10)
        self.assertTrue(report["synchronize_cuda"])
        self.assertEqual(report["summary_filename"], "step_profiler_rank{rank}.json")

    def test_temporal_step_profiler_summary_reports_percentiles_and_share(self):
        profiler = train_entry._init_temporal_step_profiler(
            {"train": {"step_profiler": {"enabled": True, "warmup_steps": 1, "active_steps": 2}}},
            {"device_type": "cuda"},
        )

        skipped = train_entry._begin_temporal_step_profile(profiler, 101)
        self.assertIsNone(skipped)
        first = train_entry._begin_temporal_step_profile(profiler, 102)
        second = train_entry._begin_temporal_step_profile(profiler, 103)
        ignored = train_entry._begin_temporal_step_profile(profiler, 104)
        self.assertEqual(first, {})
        self.assertEqual(second, {})
        self.assertIsNone(ignored)

        train_entry._finish_temporal_step_profile(
            profiler,
            {
                "dataloader_next": 0.01,
                "cpu_batch_materialize_or_cache": 0.03,
                "h2d_to_device": 0.02,
                "forward": 0.04,
                "backward": 0.05,
                "optimizer": 0.01,
                "logging_checkpoint": 0.0,
            },
        )
        train_entry._finish_temporal_step_profile(
            profiler,
            {
                "dataloader_next": 0.03,
                "cpu_batch_materialize_or_cache": 0.01,
                "h2d_to_device": 0.02,
                "forward": 0.02,
                "backward": 0.01,
                "optimizer": 0.01,
                "logging_checkpoint": 0.0,
            },
        )

        summary = train_entry._temporal_step_profiler_summary(profiler)

        self.assertEqual(summary["recorded_steps"], 2)
        self.assertEqual(summary["first_profiled_step"], 102)
        self.assertEqual(summary["last_profiled_step"], 103)
        self.assertAlmostEqual(summary["phases"]["dataloader_next"]["mean_ms"], 20.0)
        self.assertAlmostEqual(summary["phases"]["dataloader_next"]["p50_ms"], 20.0)
        self.assertAlmostEqual(summary["phases"]["dataloader_next"]["p95_ms"], 29.0)
        self.assertAlmostEqual(summary["phases"]["forward"]["share_percent"], 23.077, places=3)
        self.assertAlmostEqual(summary["step_total"]["mean_ms"], 130.0)
        self.assertAlmostEqual(summary["step_total"]["share_percent"], 100.0)

    def test_emit_temporal_step_profiler_summary_writes_json_artifact(self):
        profiler = train_entry._init_temporal_step_profiler(
            {"train": {"step_profiler": {"enabled": True, "active_steps": 1}}},
            {"device_type": "cpu"},
        )
        train_entry._begin_temporal_step_profile(profiler, 11)
        train_entry._finish_temporal_step_profile(
            profiler,
            {
                "dataloader_next": 0.01,
                "cpu_batch_materialize_or_cache": 0.02,
                "h2d_to_device": 0.03,
                "forward": 0.04,
                "backward": 0.05,
                "optimizer": 0.01,
                "logging_checkpoint": 0.01,
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                summary = train_entry._emit_temporal_step_profiler_summary(
                    output_dir=Path(tmp),
                    rank=3,
                    step_profiler=profiler,
                )

            summary_path = Path(tmp) / "step_profiler_rank3.json"
            self.assertEqual(summary["summary_path"], str(summary_path))
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["recorded_steps"], 1)
            self.assertEqual(payload["summary_path"], str(summary_path))
            self.assertIn("step_profiler_summary=", stream.getvalue())

    def test_setup_runtime_can_disable_ddp_but_keep_rank_sharding(self):
        torch = _FakeRuntimeTorch(cuda_available=True)

        runtime = train_entry._setup_torch_runtime(
            torch,
            config={"train": {"ddp": False}},
            rank=1,
            world_size=2,
        )

        self.assertFalse(runtime["distributed"])
        self.assertFalse(runtime["ddp_enabled"])
        self.assertTrue(runtime["sample_sharded"])
        self.assertEqual(runtime["world_size"], 2)
        self.assertEqual(torch.cuda.set_devices, [1])
        self.assertEqual(torch.distributed.init_backends, [])

    def test_ddp_constructor_kwargs_are_configurable_for_cuda_runtime(self):
        kwargs, report = train_entry._ddp_constructor_kwargs(
            _FakeDDPTorch,
            {
                "train": {
                    "ddp_options": {
                        "broadcast_buffers": False,
                        "find_unused_parameters": True,
                        "static_graph": False,
                        "gradient_as_bucket_view": True,
                        "init_sync": False,
                        "bucket_cap_mb": 8,
                    }
                }
            },
            {
                "distributed": True,
                "distributed_backend": "nccl",
                "device_type": "cuda",
                "local_rank": 1,
            },
        )

        self.assertEqual(kwargs["device_ids"], [1])
        self.assertEqual(kwargs["output_device"], 1)
        self.assertFalse(kwargs["broadcast_buffers"])
        self.assertTrue(kwargs["find_unused_parameters"])
        self.assertFalse(kwargs["static_graph"])
        self.assertTrue(kwargs["gradient_as_bucket_view"])
        self.assertFalse(kwargs["init_sync"])
        self.assertEqual(kwargs["bucket_cap_mb"], 8.0)
        self.assertEqual(report["backend"], "nccl")
        self.assertEqual(report["unsupported_kwargs"], [])

    def test_ddp_constructor_kwargs_preserve_torch_defaults_without_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            kwargs, report = train_entry._ddp_constructor_kwargs(
                _FakeDDPTorch,
                {},
                {
                    "distributed": True,
                    "distributed_backend": "nccl",
                    "device_type": "cuda",
                    "local_rank": 0,
                },
            )

        self.assertEqual(kwargs, {"device_ids": [0], "output_device": 0})
        self.assertEqual(report["kwargs"], {"device_ids": [0], "output_device": 0})
        self.assertNotIn("broadcast_buffers", kwargs)
        self.assertNotIn("find_unused_parameters", kwargs)
        self.assertNotIn("static_graph", kwargs)
        self.assertNotIn("gradient_as_bucket_view", kwargs)
        self.assertNotIn("bucket_cap_mb", kwargs)
        self.assertNotIn("init_sync", kwargs)

    def test_forward_microbatch_config_splits_large_temporal_batch(self):
        report = train_entry._forward_microbatch_config(
            {"train": {"forward_microbatch_size": 8192}},
            logical_batch_size=32768,
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(report["requested_size"], 8192)
        self.assertEqual(report["size"], 8192)
        self.assertEqual(report["logical_batch_size"], 32768)

    def test_temporal_microbatches_slice_logical_batch_before_device_transfer(self):
        batch = tuple(_FakeSliceTensor(key, 32768) for key in train_entry.TEMPORAL_BATCH_KEYS)

        chunks = list(train_entry._iter_temporal_microbatches(batch, 8192))

        self.assertEqual([count for _chunk, count, _total in chunks], [8192, 8192, 8192, 8192])
        self.assertEqual({total for _chunk, _count, total in chunks}, {32768})
        self.assertEqual(
            batch[0].slices,
            [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)],
        )
        first_condition = train_entry._temporal_batch_to_device(
            chunks[0][0],
            device="cuda:0",
            non_blocking=True,
        )
        self.assertEqual(first_condition["source_body_tokens"].shape[0], 8192)
        self.assertEqual(
            first_condition["source_body_tokens"].to_calls,
            [{"device": "cuda:0", "non_blocking": True}],
        )

    def test_temporal_prebatched_epoch_shuffles_and_batches_in_memory(self):
        tensors = {key: _FakeSliceTensor(key, 8) for key in train_entry.TEMPORAL_BATCH_KEYS}

        batches = list(
            train_entry._temporal_prebatched_epoch(
                tensors,
                batch_size=3,
                seed=11,
                epoch=2,
                shuffle=True,
                drop_last=False,
            )
        )

        self.assertEqual(len(batches), 3)
        self.assertTrue(all(isinstance(batch, tuple) for batch in batches))
        self.assertEqual(batches[0][0].shape[0], 3)
        self.assertEqual(len(tensors["source_body_tokens"].slices), 3)

    def test_temporal_pre_cuda_diagnostics_print_before_model_device_transfer(self):
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            train_entry._print_temporal_pre_cuda_diagnostics(
                rank=1,
                runtime={
                    "distributed": False,
                    "ddp_enabled": False,
                    "sample_sharded": True,
                    "rank": 1,
                    "world_size": 2,
                    "local_rank": 1,
                    "device_type": "cuda",
                },
                tensors=_temporal_tensors(),
                feed={"non_blocking": True},
                data_loader={"num_workers": 0, "pin_memory": True},
                microbatching={
                    "enabled": True,
                    "size": 8192,
                    "logical_batch_size": 32768,
                },
                checkpointing={"enabled": True, "every_steps": 100},
            )

        output = stream.getvalue()
        self.assertIn("rank=1 data_loader.num_workers=0", output)
        self.assertIn(
            "rank=1 forward_microbatch enabled=true size=8192 logical_batch_size=32768",
            output,
        )
        self.assertIn('"sample_sharded": true', output)

    def test_temporal_ddp_diagnostics_include_model_signature(self):
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            train_entry._print_temporal_ddp_diagnostics(
                rank=0,
                stage="pre_ddp_wrap",
                runtime={
                    "distributed": True,
                    "ddp_enabled": True,
                    "sample_sharded": True,
                    "rank": 0,
                    "world_size": 2,
                    "local_rank": 0,
                    "device_type": "cuda",
                    "distributed_backend": "nccl",
                },
                model=_FakeModule(),
                ddp={
                    "enabled": True,
                    "backend": "nccl",
                    "kwargs": {"broadcast_buffers": False},
                    "unsupported_kwargs": [],
                },
            )

        payload = json.loads(stream.getvalue().split("=", 1)[1])
        self.assertEqual(payload["stage"], "pre_ddp_wrap")
        self.assertTrue(payload["runtime"]["distributed"])
        self.assertEqual(payload["ddp"]["kwargs"]["broadcast_buffers"], False)
        self.assertEqual(payload["model"]["parameter_count"], 1)
        self.assertEqual(payload["model"]["buffer_count"], 1)
        self.assertEqual(len(payload["model"]["tensor_signature_sha256"]), 64)

    def test_temporal_startup_stage_prints_rank_local_json_marker(self):
        stream = io.StringIO()

        with mock.patch.dict(
            os.environ,
            {
                "RANK": "1",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "4",
                "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                "ONLINE_RETARGET_DDP": "1",
                "WANDB_MODE": "offline",
            },
            clear=True,
        ):
            with contextlib.redirect_stdout(stream):
                train_entry._print_temporal_startup_stage(
                    rank=1,
                    stage="checkpoint_load_begin",
                    runtime={
                        "distributed": True,
                        "ddp_enabled": True,
                        "sample_sharded": True,
                        "rank": 1,
                        "world_size": 4,
                        "local_rank": 1,
                        "device_type": "cuda",
                        "device": "cuda:1",
                        "distributed_backend": "nccl",
                    },
                    checkpoint=Path("/tmp/checkpoints/step_00000300.pt"),
                    tensor=_FakeReportTensor((2, 3), dtype="float32", device="cpu"),
                )

        output = stream.getvalue().strip()
        self.assertTrue(output.startswith("temporal_startup_state="))
        payload = json.loads(output.split("=", 1)[1])
        self.assertEqual(payload["rank"], 1)
        self.assertEqual(payload["stage"], "checkpoint_load_begin")
        self.assertEqual(payload["runtime"]["world_size"], 4)
        self.assertEqual(payload["device"], "cuda:1")
        self.assertEqual(payload["env"]["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(payload["checkpoint"], "/tmp/checkpoints/step_00000300.pt")
        self.assertEqual(payload["tensor"]["shape"], [2, 3])
        self.assertIn("pid", payload)
        self.assertIn("ppid", payload)

    def test_periodic_training_checkpoint_writes_latest_and_prunes_old_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpointing = {
                "dir": "checkpoints",
                "keep_last": 2,
                "latest_manifest": "latest_checkpoint.json",
            }
            for step in (10, 20, 30):
                train_entry._save_periodic_training_checkpoint(
                    _FakeTorchIO(),
                    output_dir=output_dir,
                    model=_FakeStateful("model"),
                    optimizer=_FakeStateful("optimizer"),
                    step=step,
                    epoch=1,
                    loss=0.5,
                    checkpointing=checkpointing,
                    sample_loader={"materialized_count": 16},
                    data_loader={"num_workers": 2},
                    feed={"non_blocking": True},
                    runtime={"device_type": "cuda", "rank": 0, "world_size": 1},
                )

            checkpoint_dir = output_dir / "checkpoints"
            self.assertFalse((checkpoint_dir / "step_00000010.pt").exists())
            self.assertTrue((checkpoint_dir / "step_00000020.pt").exists())
            self.assertTrue((checkpoint_dir / "step_00000030.pt").exists())
            latest = json.loads((output_dir / "latest_checkpoint.json").read_text())
            self.assertEqual(latest["step"], 30)
            self.assertEqual(latest["checkpoint"], str(checkpoint_dir / "step_00000030.pt"))

    def test_temporal_resume_position_advances_periodic_checkpoint_from_saved_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            saved_checkpoint = checkpoint_dir / "step_00010000.pt"
            saved_checkpoint.write_text("existing checkpoint\n", encoding="utf-8")
            checkpointing = {
                "dir": "checkpoints",
                "every_steps": 100,
                "keep_last": 2,
                "latest_manifest": "latest_checkpoint.json",
            }

            resume = train_entry._resume_training_position({"step": 10000, "epoch": 7})

            self.assertTrue(resume["resumed"])
            self.assertEqual(resume["step"], 10000)
            self.assertEqual(resume["epoch"], 7)
            self.assertEqual(resume["step"] + 1, 10001)
            self.assertFalse(
                train_entry._should_save_periodic_checkpoint(10001, 10200, checkpointing)
            )
            self.assertTrue(
                train_entry._should_save_periodic_checkpoint(10100, 10200, checkpointing)
            )

            train_entry._save_periodic_training_checkpoint(
                _FakeTorchIO(),
                output_dir=output_dir,
                model=_FakeStateful("model"),
                optimizer=_FakeStateful("optimizer"),
                step=10100,
                epoch=resume["epoch"],
                loss=0.25,
                checkpointing=checkpointing,
                sample_loader={"materialized_count": 16},
                data_loader={"num_workers": 2},
                feed={"non_blocking": True},
                runtime={"device_type": "cuda", "rank": 0, "world_size": 1},
            )

            self.assertEqual(saved_checkpoint.read_text(encoding="utf-8"), "existing checkpoint\n")
            advanced_checkpoint = checkpoint_dir / "step_00010100.pt"
            self.assertTrue(advanced_checkpoint.exists())
            latest = json.loads((output_dir / "latest_checkpoint.json").read_text())
            self.assertEqual(latest["step"], 10100)
            self.assertEqual(latest["checkpoint"], str(advanced_checkpoint))


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


def _sample_json(sample_id: str) -> str:
    return json.dumps(
        {
            "sample_id": sample_id,
            "observation": [0.0, 1.0],
            "target_joints": [0.5],
        }
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
                    "target_format": "bones_sonic_joint_root_pos_future_window",
                    "history_frames": 8,
                    "target_horizon_frames": 10,
                    "target_future_step": 5,
                    "include_target_root_pose": True,
                    "source_body_count": 30,
                    "source_body_token_dim": 15,
                    "source_rotation": "rot6d",
                    "action_dim": 36,
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
                    "action_dim": 36,
                    "source_body_count": 30,
                    "source_body_token_dim": 15,
                    "source_skeleton_dim": 120,
                    "morphology_dim": 13,
                    "robot_state_dim": 0,
                    "output_mode": "residual_prev_action",
                    "output": "g1_joint_root_position_future_window",
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
        "prev_target_action": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0],
        "prev_target_joints": [0.0, 0.1],
        "fps": 50.0,
        "target_joints": [0.1, 0.2],
        "target_root_pos_w": [0.2, 0.3, 0.4],
        "target_root_quat_w": [1.0, 0.0, 0.0, 0.0],
        "future_target_joints": [[0.1, 0.2], [0.2, 0.4]],
        "future_target_root_pos_w": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
        "future_target_root_quat_w": [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        "target_frame_indices": [10, 15],
    }


def _temporal_feature_contract_config() -> dict:
    return {
        "data": {"target_format": "bones_sonic_joint_root_pos_future_window"},
        "model": {
            "output": "g1_joint_root_position_future_window",
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
                "prev_target_action",
                "prev_target_joints",
                "previous_target_joints",
                "prev_g1_joints",
            ],
            "forbid_condition_sample_keys": [
                "target_joints",
                "future_target_joints",
                "target_root_pos_w",
                "target_root_quat_w",
                "future_target_root_pos_w",
                "future_target_root_quat_w",
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
                "action_dim": 9,
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
        "prev_action": _FakeTemporalTensor((1, 9)),
        "fps": _FakeTemporalTensor((1,)),
        "target_action": _FakeTemporalTensor((1, 2, 9)),
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


class _FakeTensorizeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(value, dtype=None):
        return _FakeReportTensor(_nested_shape(value), dtype=str(dtype), device="cpu")


def _nested_shape(value) -> tuple[int, ...]:
    shape = []
    cursor = value
    while isinstance(cursor, list):
        shape.append(len(cursor))
        cursor = cursor[0] if cursor else []
    return tuple(shape)


class _FakeDeviceTensor:
    def __init__(self, name: str):
        self.name = name
        self.to_calls = []

    def to(self, device, non_blocking=False):
        self.to_calls.append({"device": device, "non_blocking": non_blocking})
        return self


class _FakeSliceTensor(_FakeDeviceTensor):
    def __init__(self, name: str, rows: int):
        super().__init__(name)
        self.shape = (rows,)
        self.slices = []

    def __getitem__(self, item):
        if isinstance(item, list):
            self.slices.append(list(item))
            return _FakeSliceTensor(
                f"{self.name}[{','.join(str(index) for index in item)}]",
                len(item),
            )
        if not isinstance(item, slice):
            raise AssertionError(f"expected slice, got {item!r}")
        start, stop, step = item.indices(self.shape[0])
        if step != 1:
            raise AssertionError(f"expected contiguous slice, got step {step}")
        self.slices.append((start, stop))
        return _FakeSliceTensor(f"{self.name}[{start}:{stop}]", max(0, stop - start))


class _FakeTorchIO:
    def save(self, payload, path):
        path.write_text(json.dumps({"step": payload["step"], "loss": payload["loss"]}) + "\n")


class _FakeStateful:
    def __init__(self, name: str):
        self.name = name

    def state_dict(self):
        return {"name": self.name}


class _FakeRuntimeTorch:
    def __init__(self, *, cuda_available: bool):
        self.cuda = _FakeRuntimeCuda(cuda_available=cuda_available)
        self.distributed = _FakeRuntimeDistributed()

    @staticmethod
    def device(*args):
        return args


class _FakeRuntimeCuda:
    def __init__(self, *, cuda_available: bool):
        self._available = cuda_available
        self.set_devices = []

    def is_available(self):
        return self._available

    def set_device(self, index):
        self.set_devices.append(index)


class _FakeRuntimeDistributed:
    def __init__(self):
        self.init_backends = []

    @staticmethod
    def is_initialized():
        return False

    def init_process_group(self, *, backend):
        self.init_backends.append(backend)


class _FakeDDPTorch:
    class nn:
        class parallel:
            class DistributedDataParallel:
                def __init__(
                    self,
                    module,
                    *,
                    device_ids=None,
                    output_device=None,
                    broadcast_buffers=True,
                    find_unused_parameters=False,
                    static_graph=False,
                    gradient_as_bucket_view=False,
                    init_sync=True,
                    bucket_cap_mb=None,
                ):
                    self.module = module


class _FakeModule:
    def named_parameters(self):
        return [("weight", _FakeNamedTensor((2, 3), requires_grad=True))]

    def named_buffers(self):
        return [("running", _FakeNamedTensor((3,), requires_grad=False))]


class _FakeNamedTensor:
    def __init__(self, shape, *, requires_grad: bool):
        self.shape = shape
        self.requires_grad = requires_grad

    def numel(self):
        total = 1
        for value in self.shape:
            total *= value
        return total


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


class _FakeReportTensor:
    def __init__(self, shape: tuple[int, ...], *, dtype: str, device: str):
        self.shape = shape
        self.dtype = dtype
        self.device = device


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


def _minimal_dp_bridge_bvh(*, frames: int) -> str:
    motion_lines = []
    for frame_idx in range(frames):
        motion_lines.append(f"{frame_idx}.0 0.0 0.0 0.0 0.0 0.0")
    return (
        "HIERARCHY\n"
        "ROOT Hips\n"
        "{\n"
        "  OFFSET 0.0 0.0 0.0\n"
        "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        "}\n"
        "MOTION\n"
        f"Frames: {frames}\n"
        "Frame Time: 0.02\n"
        + "\n".join(motion_lines)
        + "\n"
    )


def _fake_dp_bridge_result(*, raw_trajectory_path: Path, metadata_path: Path, step: int) -> dict[str, object]:
    from online_retarget.sonic_validation_export import save_raw_validation_trajectory

    import numpy as np

    source_soma = np.zeros((3, 14, 3), dtype=np.float32)
    target_g1 = np.zeros((3, 14, 3), dtype=np.float32)
    inferred_g1 = np.zeros((3, 14, 3), dtype=np.float32)
    raw_report = save_raw_validation_trajectory(
        trajectory={
            "clip_index": 0,
            "local_env_index": 0,
            "motion_id": "walk/probe",
            "motion_key": "walk/probe",
            "source_soma": source_soma,
            "target_g1": target_g1,
            "inferred_g1": inferred_g1,
            "source_frame_indices": [0, 1, 2],
            "encoder_routes": [],
            "source_fps": 50.0,
            "target_fps": 50.0,
            "physical_time_aligned": False,
            "root_rot_format": "wxyz",
            "initial_root_xy_zeroed": False,
            "source_soma_names": [f"body_{index}" for index in range(14)],
            "g1_body_names": [f"body_{index}" for index in range(14)],
        },
        output_path=raw_trajectory_path,
        target_fps=50.0,
        duration_sec=3 / 50.0,
    )
    row2_motion = raw_trajectory_path.with_name("row2_motion.npz")
    row3_motion = raw_trajectory_path.with_name("row3_motion.npz")
    np.savez(row2_motion, joint_pos=np.zeros((3, 2), dtype=np.float32), root_pos=np.zeros((3, 3), dtype=np.float32), root_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32))
    np.savez(row3_motion, joint_pos=np.zeros((3, 2), dtype=np.float32), root_pos=np.zeros((3, 3), dtype=np.float32), root_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32))
    metadata = {
        "accepted_visual_contract": {"panels": [{}, {}, {"checkpoint_step": int(step)}]},
        "inference_render": {"checkpoint_step": int(step)},
        "visual_backend": {"accepted_vertical_v2_status": "failed"},
        "lr310_dp_prediction_bridge": {"source_bvh_resolution_skipped": False},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "sample_id": "walk/probe",
        "index": 0,
        "fps": 50.0,
        "frames": 3,
        "target_motion_npz": str(row2_motion),
        "prediction_motion_npz": str(row3_motion),
        "metadata": str(metadata_path),
        "combined_video": str(raw_trajectory_path.with_suffix(".mp4")),
        "row1_soma_somamesh_video": str(raw_trajectory_path.with_name("row1.mp4")),
        "row2_g1_target_isaaclab_video": str(raw_trajectory_path.with_name("row2.mp4")),
        "row3_g1_kinematics_isaaclab_video": str(raw_trajectory_path.with_name("row3.mp4")),
        "acceptance_ok": False,
        "accepted_vertical_v2_status": "failed",
        "acceptance_failure_reasons": ["stubbed"],
        "root_fixed_fallback": False,
        "execute_renderers": False,
        "raw_trajectory_path": str(raw_trajectory_path),
        "raw_trajectory_metadata_path": str(raw_trajectory_path.with_suffix(".json")),
        "review_contract": {
            **raw_report,
            "final_review_eligible": False,
            "physical_time_aligned": False,
            "blocked_reason": "metric-horizon sparse/window stub",
        },
    }


if __name__ == "__main__":
    unittest.main()
