from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAAC_RENDERER = REPO_ROOT / "scripts" / "render_g1_isaac_pair.py"
SOMAMESH_RENDERER = REPO_ROOT / "scripts" / "render_somamesh_source.py"
LR177_RUNNER = REPO_ROOT / "scripts" / "run_lr177_accepted_clean_validation.py"


class LR177VisualizationContractTests(unittest.TestCase):
    def test_somamesh_renderer_reports_true_lbs_schema(self) -> None:
        text = SOMAMESH_RENDERER.read_text(encoding="utf-8")

        self.assertIn("load_skeletal_mesh_from_usd", text)
        self.assertIn("skin_vertices", text)
        self.assertIn('"vertices"', text)
        self.assertIn('"triangles_loaded"', text)
        self.assertIn('"renderer"', text)
        self.assertIn('"not_capsule_bvh_visualizer"', text)

    def test_isaac_renderer_exposes_ground_and_framing_contract(self) -> None:
        text = ISAAC_RENDERER.read_text(encoding="utf-8")

        for token in (
            "--source-renderer",
            "--ground-size",
            "--ground-color",
            "--camera-follow-smoothing",
            "--camera-framing-margin",
        ):
            self.assertIn(token, text)
        self.assertIn("default=\"somamesh\"", text)
        self.assertIn("ground_size=args_cli.ground_size", text)
        self.assertIn('"ground_size"', text)
        self.assertIn('"ground_color"', text)
        self.assertIn('"camera_policy"', text)
        self.assertIn("_smooth_root_xy_targets", text)
        self.assertIn("_camera_envelope", text)

    def test_lr177_runner_defaults_and_passes_visual_args(self) -> None:
        text = LR177_RUNNER.read_text(encoding="utf-8")

        self.assertIn("DEFAULT_GROUND_COLOR = (0.08, 0.20, 0.72)", text)
        self.assertIn(
            'parser.add_argument("--preserve-world-root", action="store_true", default=True)',
            text,
        )
        self.assertIn(
            'parser.add_argument("--draw-orientation-labels", action="store_true", default=True)',
            text,
        )
        for token in (
            "--source-renderer",
            "somamesh",
            "--ground-size",
            "--ground-color",
            "--camera-mode",
            "--camera-offset",
            "--camera-follow-smoothing",
            "--camera-framing-margin",
            "--preserve-world-root",
            "--draw-orientation-labels",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
