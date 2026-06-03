import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from online_retarget.a0_visual_validation import (
    ACCEPTANCE_G1_BACKEND,
    ACCEPTANCE_SOURCE_BACKEND,
    DEBUG_CAPSULE_BACKEND,
    DEFAULT_G1_USD,
    PRIMARY_VISUAL_BACKEND,
    SOMA_DISPLAY_TRANSFORM,
    A0VisualValidationRenderer,
    accepted_vertical_v2_artifact_paths,
)


class A0VisualValidationRendererTests(unittest.TestCase):
    def test_soma_root_target_composes_local_xy_to_world(self) -> None:
        renderer = A0VisualValidationRenderer(
            {
                "input_data": {"format": "soma_motionlib"},
                "features": {"include_root_pos_target": True},
            }
        )

        root = renderer.compose_prediction_root(
            np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32),
            np.asarray([[10.0, 20.0, 0.3]], dtype=np.float32),
        )

        np.testing.assert_allclose(root, np.asarray([[10.25, 19.5, 1.2]], dtype=np.float32))

    def test_non_soma_root_target_keeps_predicted_world_root(self) -> None:
        renderer = A0VisualValidationRenderer(
            {
                "input_data": {"format": "npz"},
                "features": {"include_root_pos_target": True},
            }
        )

        root = renderer.compose_prediction_root(
            np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32),
            np.asarray([[10.0, 20.0, 0.3]], dtype=np.float32),
        )

        np.testing.assert_allclose(root, np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32))

    def test_soma_without_root_target_keeps_predicted_root(self) -> None:
        renderer = A0VisualValidationRenderer(
            {
                "input_data": {"format": "soma_motionlib"},
                "features": {"include_root_pos_target": False},
            }
        )

        root = renderer.compose_prediction_root(
            np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32),
            np.asarray([[10.0, 20.0, 0.3]], dtype=np.float32),
        )

        np.testing.assert_allclose(root, np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32))

    def test_soma_display_transform_is_x_negative_z_y(self) -> None:
        self.assertEqual(
            A0VisualValidationRenderer.soma_point_to_display((1.0, 2.0, 3.0)),
            (1.0, -3.0, 2.0),
        )
        frames = A0VisualValidationRenderer.soma_motionlib_source_frames(
            np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
            ["Hips"],
        )
        self.assertEqual(frames, [{"Hips": (1.0, -3.0, 2.0)}])

    def test_backend_manifest_records_primary_and_debug_backends(self) -> None:
        renderer = A0VisualValidationRenderer({"visual_validation": {}})

        manifest = renderer.backend_manifest(active_backend=DEBUG_CAPSULE_BACKEND)

        self.assertIn("target_and_kinematic", PRIMARY_VISUAL_BACKEND)
        self.assertEqual(manifest["primary_backend"], PRIMARY_VISUAL_BACKEND)
        self.assertEqual(manifest["active_backend"], DEBUG_CAPSULE_BACKEND)
        self.assertEqual(manifest["source_display_transform"], SOMA_DISPLAY_TRANSFORM)
        self.assertEqual(manifest["g1_asset_usd"], str(renderer.g1_usd_path))
        self.assertEqual(manifest["g1_asset_usd_resolution"]["path"], str(renderer.g1_usd_path))
        self.assertFalse(manifest["active_backend_is_acceptance_backend"])
        self.assertFalse(manifest["debug_fallback_is_acceptance_backend"])

        acceptance = renderer.acceptance_backend_manifest()
        self.assertEqual(acceptance["active_backend"], PRIMARY_VISUAL_BACKEND)
        self.assertEqual(acceptance["source_human_backend"], ACCEPTANCE_SOURCE_BACKEND)
        self.assertEqual(acceptance["g1_backend"], ACCEPTANCE_G1_BACKEND)
        self.assertTrue(acceptance["active_backend_is_acceptance_backend"])

    def test_g1_usd_derives_from_online_retarget_output_root(self) -> None:
        renderer = A0VisualValidationRenderer(
            {
                "output_dir": "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/"
                "sonic_kin_soma_motionlib_a0_frozen_ae_runs/{run_group}/uniform"
            }
        )

        expected = Path("/mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd")
        self.assertEqual(renderer.g1_usd_path, expected)
        self.assertEqual(renderer.g1_usd_resolution["source"], "output_dir")
        self.assertEqual(renderer.g1_usd_resolution["path"], str(expected))

    def test_explicit_g1_robot_usd_wins_over_runtime_root(self) -> None:
        explicit = Path("/tmp/explicit_g1/main.usd")
        renderer = A0VisualValidationRenderer(
            {
                "output_dir": "/mnt/data_cpfs/code/wxh/OnlineRetarget/outputs/run",
                "visual_validation": {"g1_robot_usd": str(explicit)},
            }
        )

        self.assertEqual(renderer.g1_usd_path, explicit)
        self.assertEqual(renderer.g1_usd_resolution["source"], "visual_validation.g1_robot_usd")

    def test_isaaclab_g1_render_command_preserves_world_root_and_asset(self) -> None:
        renderer = A0VisualValidationRenderer({"visual_validation": {}})

        command = renderer.isaaclab_g1_render_command(
            python_bin="/workspace/isaaclab/_isaac_sim/python.sh",
            script_path="scripts/render_g1_isaac_pair.py",
            motion_path="clip/inference_g1_isaac_input.npz",
            output_path="clip/inference_g1_isaac.mp4",
            duration_sec=4.0,
            width=960,
            height=540,
        )

        self.assertIn("--preserve-world-root", command)
        self.assertIn("--overlay-world-root-axes", command)
        self.assertIn("--overlay-semantic-lr", command)
        self.assertIn("--robot-usd", command)
        self.assertIn(str(DEFAULT_G1_USD), command)
        self.assertEqual(command[command.index("--format") + 1], "npz")

    def test_rerender_cli_command_requests_acceptance_backend(self) -> None:
        command = A0VisualValidationRenderer({"visual_validation": {"g1_robot_usd": "/tmp/g1/main.usd"}}).rerender_cli_command(
            config_path="config.json",
            checkpoint_path="step.pt",
            output_dir="rerender",
            rows_cache="rows.json",
            stats_path="normalization.pt",
            step=119500,
            count=8,
            python_bin="python",
            script_path="scripts/rerender_a0_visual_validation.py",
            isaac_python_bin="/workspace/isaaclab/_isaac_sim/python.sh",
        )

        self.assertIn("--acceptance-backend", command)
        self.assertEqual(command[command.index("--checkpoint") + 1], "step.pt")
        self.assertEqual(command[command.index("--rows-cache") + 1], "rows.json")
        self.assertEqual(command[command.index("--step") + 1], "119500")
        self.assertEqual(command[command.index("--g1-robot-usd") + 1], "/tmp/g1/main.usd")

    def test_g1_motion_npz_and_fake_isaaclab_playback_record_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = A0VisualValidationRenderer({})
            motion_path = root / "clip" / "g1_input.npz"
            motion_report = renderer.write_g1_motion_npz(
                path=motion_path,
                joint_pos=np.zeros((3, 29), dtype=np.float32),
                root_pos=np.asarray([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [2.0, 0.0, 0.8]], dtype=np.float32),
                root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)),
                fps=50.0,
                joint_names=[f"joint_{index}" for index in range(29)],
            )
            self.assertEqual(motion_report["status"], "ok")
            self.assertTrue(motion_path.exists())
            with np.load(motion_path) as loaded:
                self.assertEqual(tuple(loaded["joint_pos"].shape), (3, 29))
                self.assertEqual(tuple(loaded["root_quat"].shape), (3, 4))
            self.assertEqual(len(motion_report["sha256"]), 64)

            fake_script = root / "fake_isaac.py"
            fake_script.write_text(
                "import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--g1-motion'); p.add_argument('--format'); "
                "p.add_argument('--output'); p.add_argument('--duration-sec'); p.add_argument('--robot-usd'); "
                "p.add_argument('--preserve-world-root', action='store_true'); p.add_argument('--width'); "
                "p.add_argument('--height'); p.add_argument('--overlay-world-root-axes', action='store_true'); "
                "p.add_argument('--overlay-semantic-lr', action='store_true'); a=p.parse_args()\n"
                "open(a.output, 'wb').write(b'mp4')\n"
                "open(a.output.rsplit('.',1)[0]+'.json', 'w').write(json.dumps({'status':'ok','backend':'isaaclab_usd_g1_kinematic_playback'}))\n",
                encoding="utf-8",
            )

            report = renderer.render_g1_isaaclab_playback(
                python_bin=sys.executable,
                script_path=fake_script,
                motion_path=motion_path,
                output_path=root / "g1.mp4",
                duration_sec=4.0,
                width=320,
                height=180,
                execute=True,
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["backend"], ACCEPTANCE_G1_BACKEND)
            self.assertTrue(report["output_exists"])
            self.assertGreater(report["output_bytes"], 0)
            self.assertTrue((root / "g1.mp4.command.json").exists())
            command_record = json.loads((root / "g1.mp4.command.json").read_text(encoding="utf-8"))
            self.assertEqual(command_record["expected_output_path"], str(root / "g1.mp4"))
            self.assertTrue(command_record["preserve_world_root"])
            self.assertIn("semantic_left_right", command_record["overlays"])

    def test_accepted_vertical_v2_two_sample_npz_manifest_paths_are_unique_and_hashes_match(self) -> None:
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = A0VisualValidationRenderer({})
            base_dir = root / "visual_validation" / "step_00002000"
            manifests = []
            for sample_id, value in (("sample_a", 1.0), ("sample_b", 2.0)):
                paths = accepted_vertical_v2_artifact_paths(base_dir, sample_id=sample_id, step=2000)
                row2_report = renderer.write_g1_motion_npz(
                    path=paths["row2_motion_npz"],
                    joint_pos=np.full((2, 2), value, dtype=np.float32),
                    root_pos=np.asarray([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]], dtype=np.float32),
                    root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                    fps=50.0,
                    joint_names=["left", "right"],
                )
                row3_report = renderer.write_g1_motion_npz(
                    path=paths["row3_motion_npz"],
                    joint_pos=np.full((2, 2), value + 10.0, dtype=np.float32),
                    root_pos=np.asarray([[0.2, 0.0, 0.8], [0.3, 0.0, 0.8]], dtype=np.float32),
                    root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                    fps=50.0,
                    joint_names=["left", "right"],
                )
                manifest = {
                    "sample_id": sample_id,
                    "accepted_visual_contract": {
                        "panels": [
                            {"name": "Soma"},
                            {
                                "name": "G1 Target Playback",
                                "motion_path": row2_report["path"],
                                "motion_sha256": row2_report["sha256"],
                            },
                            {
                                "name": "G1 Kinematics Playback",
                                "motion_path": row3_report["path"],
                                "motion_sha256": row3_report["sha256"],
                            },
                        ],
                    },
                    "g1_isaaclab_target_motion_asset": row2_report,
                    "g1_isaaclab_motion_asset": row3_report,
                }
                paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                manifests.append(json.loads(paths["manifest_json"].read_text(encoding="utf-8")))

            row2_paths = [
                Path(manifest["accepted_visual_contract"]["panels"][1]["motion_path"])
                for manifest in manifests
            ]
            row3_paths = [
                Path(manifest["accepted_visual_contract"]["panels"][2]["motion_path"])
                for manifest in manifests
            ]
            self.assertEqual(len(set(row2_paths)), 2)
            self.assertEqual(len(set(row3_paths)), 2)
            for manifest, row2_path, row3_path in zip(manifests, row2_paths, row3_paths):
                row2 = manifest["accepted_visual_contract"]["panels"][1]
                row3 = manifest["accepted_visual_contract"]["panels"][2]
                self.assertEqual(
                    row2_path.name,
                    f"{manifest['sample_id']}__step_00002000__row2_g1_target_isaaclab_input.npz",
                )
                self.assertEqual(
                    row3_path.name,
                    f"{manifest['sample_id']}__step_00002000__row3_g1_kinematics_isaaclab_input.npz",
                )
                self.assertTrue(row2_path.exists())
                self.assertTrue(row3_path.exists())
                self.assertEqual(row2["motion_sha256"], sha256(row2_path))
                self.assertEqual(row3["motion_sha256"], sha256(row3_path))
                self.assertEqual(manifest["g1_isaaclab_target_motion_asset"]["path"], str(row2_path))
                self.assertEqual(manifest["g1_isaaclab_motion_asset"]["path"], str(row3_path))

    def test_fake_isaaclab_returncode_zero_without_mp4_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = A0VisualValidationRenderer({})
            motion_path = root / "g1_input.npz"
            renderer.write_g1_motion_npz(
                path=motion_path,
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                root_pos=np.asarray([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]], dtype=np.float32),
                root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                fps=50.0,
            )
            fake_script = root / "fake_isaac_no_video.py"
            fake_script.write_text(
                "import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--g1-motion'); p.add_argument('--format'); "
                "p.add_argument('--output'); p.add_argument('--duration-sec'); p.add_argument('--robot-usd'); "
                "p.add_argument('--preserve-world-root', action='store_true'); p.add_argument('--width'); "
                "p.add_argument('--height'); p.add_argument('--overlay-world-root-axes', action='store_true'); "
                "p.add_argument('--overlay-semantic-lr', action='store_true'); a=p.parse_args()\n"
                "open(a.output.rsplit('.',1)[0]+'.json', 'w').write(json.dumps({'status':'ok'}))\n",
                encoding="utf-8",
            )

            report = renderer.render_g1_isaaclab_playback(
                python_bin=sys.executable,
                script_path=fake_script,
                motion_path=motion_path,
                output_path=root / "missing.mp4",
                duration_sec=1.0,
                width=160,
                height=90,
                execute=True,
            )

            self.assertEqual(report["returncode"], 0)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["output_exists"])
            self.assertEqual(report["output_bytes"], 0)
            self.assertIn("expected_output_mp4_missing", report["failure_reasons"])
            self.assertEqual(report["expected_output_path"], str(root / "missing.mp4"))

    def test_fake_isaaclab_linger_after_valid_artifacts_is_terminated_and_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = A0VisualValidationRenderer({})
            motion_path = root / "g1_input.npz"
            renderer.write_g1_motion_npz(
                path=motion_path,
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                root_pos=np.asarray([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]], dtype=np.float32),
                root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                fps=50.0,
            )
            fake_script = root / "fake_isaac_linger.py"
            fake_script.write_text(
                "import argparse, json, time\n"
                "p=argparse.ArgumentParser(); p.add_argument('--g1-motion'); p.add_argument('--format'); "
                "p.add_argument('--output'); p.add_argument('--duration-sec'); p.add_argument('--robot-usd'); "
                "p.add_argument('--preserve-world-root', action='store_true'); p.add_argument('--width'); "
                "p.add_argument('--height'); p.add_argument('--overlay-world-root-axes', action='store_true'); "
                "p.add_argument('--overlay-semantic-lr', action='store_true'); a=p.parse_args()\n"
                "open(a.output, 'wb').write(b'mp4')\n"
                "open(a.output.rsplit('.',1)[0]+'.json', 'w').write(json.dumps({"
                "'status':'ok','backend':'isaaclab_usd_g1_kinematic_playback','failure_reasons':[]}))\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )

            started = time.monotonic()
            report = renderer.render_g1_isaaclab_playback(
                python_bin=sys.executable,
                script_path=fake_script,
                motion_path=motion_path,
                output_path=root / "linger.mp4",
                duration_sec=1.0,
                width=160,
                height=90,
                execute=True,
                timeout_sec=5.0,
                success_artifact_grace_sec=0.1,
                terminate_grace_sec=1.0,
            )

            self.assertLess(time.monotonic() - started, 3.0)
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["output_exists"])
            self.assertGreater(report["output_bytes"], 0)
            self.assertTrue(report["terminated_after_success_artifacts"])
            self.assertEqual(report["failure_reasons"], [])
            self.assertEqual(report["isaaclab_report"]["status"], "ok")

    def test_fake_isaaclab_bad_json_is_failed_even_with_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = A0VisualValidationRenderer({})
            motion_path = root / "g1_input.npz"
            renderer.write_g1_motion_npz(
                path=motion_path,
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                root_pos=np.asarray([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]], dtype=np.float32),
                root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
                fps=50.0,
            )
            fake_script = root / "fake_isaac_bad_json.py"
            fake_script.write_text(
                "import argparse\n"
                "p=argparse.ArgumentParser(); p.add_argument('--g1-motion'); p.add_argument('--format'); "
                "p.add_argument('--output'); p.add_argument('--duration-sec'); p.add_argument('--robot-usd'); "
                "p.add_argument('--preserve-world-root', action='store_true'); p.add_argument('--width'); "
                "p.add_argument('--height'); p.add_argument('--overlay-world-root-axes', action='store_true'); "
                "p.add_argument('--overlay-semantic-lr', action='store_true'); a=p.parse_args()\n"
                "open(a.output, 'wb').write(b'mp4')\n"
                "open(a.output.rsplit('.',1)[0]+'.json', 'w').write('{bad json')\n",
                encoding="utf-8",
            )

            report = renderer.render_g1_isaaclab_playback(
                python_bin=sys.executable,
                script_path=fake_script,
                motion_path=motion_path,
                output_path=root / "bad_json.mp4",
                duration_sec=1.0,
                width=160,
                height=90,
                execute=True,
            )

            self.assertEqual(report["returncode"], 0)
            self.assertEqual(report["status"], "failed")
            self.assertIn("renderer_report_bad_json", report["failure_reasons"])

    def test_render_g1_missing_usd_fails_before_app_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_package = root / "fake_modules" / "isaaclab"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text("", encoding="utf-8")
            (fake_package / "app.py").write_text(
                "import os\n"
                "from pathlib import Path\n"
                "class AppLauncher:\n"
                "    @staticmethod\n"
                "    def add_app_launcher_args(parser):\n"
                "        parser.add_argument('--fake-app-arg', default='')\n"
                "    def __init__(self, args):\n"
                "        Path(os.environ['APP_LAUNCHER_SENTINEL']).write_text('started', encoding='utf-8')\n"
                "        self.app = object()\n",
                encoding="utf-8",
            )
            output = root / "out.mp4"
            sentinel = root / "app_launcher_started.txt"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root / "fake_modules")
            env["APP_LAUNCHER_SENTINEL"] = str(sentinel)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_g1_isaac_pair.py",
                    "--g1-motion",
                    str(root / "motion.npz"),
                    "--format",
                    "npz",
                    "--output",
                    str(output),
                    "--robot-usd",
                    str(root / "missing.usd"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(sentinel.exists())
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["preflight_stage"], "before_app_launcher")
            self.assertFalse(report["app_launcher_started"])
            self.assertIn("robot_usd_missing", report["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
