import json
from pathlib import Path
import sys
import tempfile
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

        self.assertEqual(manifest["primary_backend"], PRIMARY_VISUAL_BACKEND)
        self.assertEqual(manifest["active_backend"], DEBUG_CAPSULE_BACKEND)
        self.assertEqual(manifest["source_display_transform"], SOMA_DISPLAY_TRANSFORM)
        self.assertEqual(manifest["g1_asset_usd"], str(DEFAULT_G1_USD))
        self.assertFalse(manifest["active_backend_is_acceptance_backend"])
        self.assertFalse(manifest["debug_fallback_is_acceptance_backend"])

        acceptance = renderer.acceptance_backend_manifest()
        self.assertEqual(acceptance["active_backend"], PRIMARY_VISUAL_BACKEND)
        self.assertEqual(acceptance["source_human_backend"], ACCEPTANCE_SOURCE_BACKEND)
        self.assertEqual(acceptance["g1_backend"], ACCEPTANCE_G1_BACKEND)
        self.assertTrue(acceptance["active_backend_is_acceptance_backend"])

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
        command = A0VisualValidationRenderer({}).rerender_cli_command(
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


if __name__ == "__main__":
    unittest.main()
