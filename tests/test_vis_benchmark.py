from __future__ import annotations

import importlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vis_benchmark import BenchmarkConfig, run_phase3_benchmark


class VisBenchmarkTests(unittest.TestCase):
    def test_import_does_not_import_optional_renderer_backends(self) -> None:
        for module_name in tuple(sys.modules):
            if module_name == "vis_benchmark" or module_name.startswith("vis_benchmark."):
                del sys.modules[module_name]
        guard = _BlockedImportGuard({"isaaclab", "isaacsim", "mujoco", "newton", "warp"})
        sys.meta_path.insert(0, guard)
        try:
            module = importlib.import_module("vis_benchmark")
        finally:
            sys.meta_path.remove(guard)

        self.assertTrue(hasattr(module, "run_phase3_benchmark"))

    def test_synthetic_smoke_reports_observed_throughput_without_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_phase3_benchmark(
                BenchmarkConfig(
                    manifest_path=None,
                    output_dir=Path(tmp),
                    synthetic_smoke=True,
                    packets=2,
                    workers=1,
                )
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["benchmark_scope"], "synthetic_smoke")
        self.assertIsNone(report["performance_conclusion"])
        self.assertEqual(report["throughput"]["observed"]["completed_packets"], 2)
        self.assertIn("effective_fps", report["throughput"]["observed"])
        self.assertIn("timeline_align_wall_sec", report["stage_times_sec"]["unavailable"])
        self.assertIn("encode_wall_sec", report["stage_times_sec"]["unavailable"])

    def test_dry_run_builds_isaac_adapter_command_without_executing_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_phase3_benchmark(
                BenchmarkConfig(
                    manifest_path=None,
                    output_dir=Path(tmp),
                    adapter="isaac_render",
                    dry_run=True,
                    synthetic_smoke=True,
                )
            )

        command = report["adapter_plan"]["command"]
        self.assertEqual(report["status"], "dry_run")
        self.assertIsNotNone(command)
        self.assertIn("render_g1_isaac_pair.py", command["argv"][1])
        self.assertIn("--g1-motion", command["argv"])
        self.assertEqual(report["throughput"]["observed"], {})
        unavailable = report["resource_metrics"]["unavailable"].values()
        self.assertIn("dry_run_does_not_execute_renderer", unavailable)

    def test_cli_writes_json_report_for_synthetic_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "benchmark_vis_phase3.py"
        with tempfile.TemporaryDirectory() as tmp:
            output_json = Path(tmp) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--synthetic-smoke",
                    "--dry-run",
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--output-json",
                    str(output_json),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["benchmark_scope"], "synthetic_smoke")


class _BlockedImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_roots: set[str]) -> None:
        self._blocked_roots = blocked_roots

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        root_name = fullname.partition(".")[0]
        if root_name in self._blocked_roots:
            raise AssertionError(f"optional backend imported by benchmark harness: {fullname}")
        return None


if __name__ == "__main__":
    unittest.main()
