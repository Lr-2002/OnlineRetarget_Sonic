from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_SCRIPT = REPO_ROOT / "scripts" / "run_accepted_soma_g1_validation.py"
ISAAC_RENDERER = REPO_ROOT / "scripts" / "render_g1_isaac_pair.py"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "accepted_soma_g1_validation.md"


def load_service() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("accepted_soma_g1_validation", SERVICE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AcceptedSomaG1ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = load_service()

    def test_default_lr106_manifest_is_fixed_eight_samples(self) -> None:
        samples = self.service.DEFAULT_LR106_SAMPLES
        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0].key, "220720__itching_neck_003__A032_M")
        self.assertEqual(samples[-1].key, "221115__walk_ff_loop_180_R_001__A060_M")
        self.assertTrue(all(sample.frames > 0 for sample in samples))

    def test_sample_parser_accepts_cli_schema(self) -> None:
        sample = self.service.parse_sample("220720:itching_neck_003__A032_M:200")

        self.assertEqual(sample.date, "220720")
        self.assertEqual(sample.stem, "itching_neck_003__A032_M")
        self.assertEqual(sample.frames, 200)
        self.assertEqual(sample.key, "220720__itching_neck_003__A032_M")

    def test_summary_records_accepted_visualization_contract(self) -> None:
        sample = self.service.DEFAULT_LR106_SAMPLES[0]
        summary = self.service.summary(
            Path("/tmp/out"),
            [sample],
            [{"status": "ok"}],
            {"status": "ok", "dashboard_url": "http://10.1.11.30:5175/runs/online-retarget/test"},
        )

        self.assertEqual(summary["status"], "ok")
        self.assertIn("SomaBVH/SomaMesh LBS", summary["source_contract"])
        self.assertEqual(
            summary["standard_output_contract"]["source_display_conversion"],
            "(x, y, z)_display = (x, -z, y)_soma",
        )
        self.assertIn("follow", summary["standard_output_contract"]["target_camera"])
        self.assertIn("root-zeroed", summary["standard_output_contract"]["target_root_xy_policy"])
        self.assertIn("80.0m", summary["standard_output_contract"]["target_ground"])
        self.assertIn("xyzw", summary["standard_output_contract"]["root_quaternion"])
        self.assertFalse(summary["retarget_mapping_changed"])

    def test_agenthub_upload_output_parser_extracts_dashboard_url(self) -> None:
        parsed = self.service.parse_agenthub_upload_output(
            '{"data":{"project":"online-retarget","run_id":"abc","dashboard_url":"http://hub/run"},"error":null}'
        )

        self.assertEqual(parsed["run_id"], "abc")
        self.assertEqual(parsed["dashboard_url"], "http://hub/run")

    def test_isaac_renderer_supports_fast_exit_for_headless_delta(self) -> None:
        renderer_text = ISAAC_RENDERER.read_text(encoding="utf-8")
        service_text = SERVICE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--fast-exit-after-report", renderer_text)
        self.assertIn("os._exit(0)", renderer_text)
        self.assertIn("max_frames=0", service_text)
        self.assertIn('default="follow"', renderer_text)
        self.assertIn("DEFAULT_GROUND_SIZE = 80.0", service_text)
        self.assertIn("--ground-size", service_text)
        self.assertIn("--draw-orientation-labels", service_text)

    def test_runbook_states_non_dynamics_scope(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("SomaMesh/global-SOMA", text)
        self.assertIn("(x, y, z)_display = (x, -z, y)_soma", text)
        self.assertIn("root-zeroed", text)
        self.assertIn("80m", text)
        self.assertIn("not prove policy tracking", text)


if __name__ == "__main__":
    unittest.main()
