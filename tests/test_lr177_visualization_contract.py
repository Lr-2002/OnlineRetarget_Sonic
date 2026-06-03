from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from scripts import render_somamesh_source


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

    def test_somamesh_renderer_resolves_src_layout_retargeter_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retargeter_root = Path(tmp) / "soma-retargeter"
            package_root = retargeter_root / "src" / "soma_retargeter"
            package_root.mkdir(parents=True)

            paths = render_somamesh_source.soma_retargeter_python_paths(retargeter_root)

            self.assertEqual(paths, [retargeter_root, retargeter_root / "src"])
            with mock.patch.object(sys, "path", []):
                render_somamesh_source.load_soma_retargeter(retargeter_root)
                self.assertIn(str(retargeter_root), sys.path)
                self.assertIn(str(retargeter_root / "src"), sys.path)

    def test_somamesh_renderer_preflight_blocks_missing_dependency_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = render_somamesh_source.preflight_somamesh_renderer(
                retargeter_root=root / "missing-soma-retargeter",
                usd_path=root / "missing-soma-base.usd",
            )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("soma_retargeter_root_missing", report["failure_reasons"])
            self.assertIn("soma_usd_missing", report["failure_reasons"])
            self.assertIn("soma_retargeter_import_failed", report["failure_reasons"])
            message = render_somamesh_source.format_somamesh_preflight_error(report)
            self.assertIn("SomaMeshShapes renderer preflight blocked before rendering", message)
            self.assertIn("missing-soma-retargeter", message)

    def test_training_visual_gate_passes_somamesh_pythonpath_env(self) -> None:
        text = (REPO_ROOT / "scripts" / "train_sonic_kin_skeleton_ae.py").read_text(encoding="utf-8")

        self.assertIn("env = _somamesh_renderer_env(cfg)", text)
        self.assertIn("env=env", text)
        self.assertIn("def _somamesh_renderer_env", text)
        self.assertIn('retargeter_root / "src"', text)
        self.assertIn("paths.extend([str(ROOT), str(SRC_ROOT)])", text)
        self.assertIn('env["PYTHONPATH"] = os.pathsep.join(deduped_paths)', text)
        self.assertIn("def preflight_acceptance_somamesh_visual_validation", text)
        self.assertIn("preflight_acceptance_somamesh_visual_validation(config, output_dir, runtime)", text)
        self.assertIn("--preflight-only", text)
        self.assertIn("soma_retargeter_root", text)
        self.assertIn("somamesh_usd", text)
        self.assertIn("accepted SomaMeshShapes renderer preflight blocked before training", text)

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
        self.assertIn('default="follow"', text)
        self.assertIn("default=80.0", text)
        self.assertIn("ground_size=args_cli.ground_size", text)
        self.assertIn('"ground_size"', text)
        self.assertIn('"ground_color"', text)
        self.assertIn('"camera_policy"', text)
        self.assertIn("_smooth_root_xy_targets", text)
        self.assertIn("_camera_envelope", text)

    def test_lr177_runner_defaults_and_passes_visual_args(self) -> None:
        text = LR177_RUNNER.read_text(encoding="utf-8")

        self.assertIn("DEFAULT_GROUND_COLOR = (0.08, 0.20, 0.72)", text)
        self.assertIn("DEFAULT_GROUND_SIZE = 80.0", text)
        self.assertIn(
            'parser.add_argument("--preserve-world-root", action="store_true", default=False)',
            text,
        )
        self.assertIn('"preserve_world_root": bool(args.preserve_world_root)', text)
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
