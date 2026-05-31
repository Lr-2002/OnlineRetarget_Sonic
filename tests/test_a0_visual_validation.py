import unittest

import numpy as np

from online_retarget.a0_visual_validation import (
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
        self.assertIn("--robot-usd", command)
        self.assertIn(str(DEFAULT_G1_USD), command)
        self.assertEqual(command[command.index("--format") + 1], "npz")


if __name__ == "__main__":
    unittest.main()
