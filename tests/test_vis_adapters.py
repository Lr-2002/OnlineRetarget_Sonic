from __future__ import annotations

import csv
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest

from vis_core import RenderRequest, load_vis_packet


class VisAdaptersTests(unittest.TestCase):
    def test_vis_adapters_import_without_optional_backend_imports(self) -> None:
        for module_name in tuple(sys.modules):
            if module_name == "vis_adapters" or module_name.startswith("vis_adapters."):
                del sys.modules[module_name]
        guard = _BlockedImportGuard({"isaaclab", "isaacsim", "mujoco", "newton", "warp"})
        sys.meta_path.insert(0, guard)
        try:
            module = importlib.import_module("vis_adapters")
        finally:
            sys.meta_path.remove(guard)

        self.assertTrue(hasattr(module, "IsaacRenderAdapter"))
        self.assertTrue(hasattr(module, "SOMANewtonDynamicSessionAdapter"))

    def test_vis_core_does_not_import_adapter_layer(self) -> None:
        vis_core_root = Path(__file__).resolve().parents[1] / "src" / "vis_core"

        for path in vis_core_root.glob("*.py"):
            self.assertNotIn("vis_adapters", path.read_text(encoding="utf-8"))

    def test_registry_marks_connected_vs_interface_only_adapters(self) -> None:
        from vis_adapters import adapter_descriptors

        descriptors = {descriptor.name: descriptor for descriptor in adapter_descriptors()}

        self.assertEqual(descriptors["isaac_render"].status, "connected_script")
        self.assertEqual(descriptors["somamesh_source_render"].status, "connected_script")
        self.assertEqual(descriptors["gmr_mujoco_kinematic_playback"].status, "interface_only")
        self.assertEqual(descriptors["soma_newton_dynamic_session"].status, "interface_only")

    def test_isaac_render_adapter_builds_command_for_existing_script(self) -> None:
        from vis_adapters import IsaacRenderAdapter

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(Path(tmp))
            adapter = IsaacRenderAdapter(python="python")

            preflight = adapter.preflight(request)
            command = adapter.build_command(request)

        self.assertEqual(preflight.status, "ok")
        self.assertIn("render_g1_isaac_pair.py", command.argv[1])
        self.assertIn("--g1-motion", command.argv)
        self.assertEqual(_arg_path_name(command.argv, "--g1-motion"), "play.csv")
        self.assertIn("--bvh", command.argv)
        self.assertEqual(_arg_path_name(command.argv, "--bvh"), "source.bvh")
        self.assertIn("--fast-exit-after-report", command.argv)
        self.assertFalse(command.interface_only)

    def test_somamesh_source_adapter_builds_command_for_existing_script(self) -> None:
        from vis_adapters import SomaMeshSourceRenderAdapter

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(Path(tmp))
            adapter = SomaMeshSourceRenderAdapter(python="python")

            preflight = adapter.preflight(request)
            command = adapter.build_command(request)

        self.assertEqual(preflight.status, "ok")
        self.assertIn("render_somamesh_source.py", command.argv[1])
        self.assertIn("--bvh", command.argv)
        self.assertEqual(_arg_path_name(command.argv, "--bvh"), "source.bvh")
        self.assertIn("--frame-count", command.argv)
        self.assertFalse(command.interface_only)

    def test_gmr_mujoco_adapter_is_interface_only_migration_point(self) -> None:
        from vis_adapters import GMRMujocoKinematicPlaybackAdapter

        with tempfile.TemporaryDirectory() as tmp:
            request = _request(Path(tmp))
            adapter = GMRMujocoKinematicPlaybackAdapter()
            preflight = adapter.preflight(request)
            command = adapter.command_plan(request)

        self.assertEqual(preflight.status, "interface_only")
        self.assertFalse(preflight.connected)
        self.assertIn("interface_only_unbounded_viewer_loop", preflight.reasons)
        self.assertTrue(command.interface_only)
        self.assertIn("vis_robot_motion.py", command.argv[1])

    def test_soma_newton_adapter_is_interface_only_dynamic_session_boundary(self) -> None:
        from vis_adapters import SOMANewtonDynamicSessionAdapter

        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_vis_packet(_write_fixture_packet(Path(tmp)))
            adapter = SOMANewtonDynamicSessionAdapter()

            reset_report = adapter.reset(loaded)
            command = adapter.recorder_command_plan(output_csv=Path(tmp) / "online.csv")

        self.assertEqual(reset_report["status"], "interface_only")
        self.assertIn("interface_only_external_runtime_session", reset_report["reasons"])
        self.assertTrue(command.interface_only)
        self.assertIn("record_online_retarget_output.py", command.argv[1])
        with self.assertRaises(NotImplementedError):
            adapter.step({}, dt=0.02)


class _BlockedImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_roots: set[str]) -> None:
        self._blocked_roots = blocked_roots

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        root_name = fullname.partition(".")[0]
        if root_name in self._blocked_roots:
            raise AssertionError(f"optional backend imported at adapter import time: {fullname}")
        return None


def _request(root: Path) -> RenderRequest:
    loaded = load_vis_packet(_write_fixture_packet(root))
    return RenderRequest(loaded_packet=loaded, output_dir=root / "out")


def _write_fixture_packet(root: Path) -> Path:
    (root / "source.bvh").write_text("HIERARCHY\n", encoding="utf-8")
    (root / "human.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"root_pos": [0, 0, 0], "root_rot": [0, 0, 0, 1], "pelvis": [0, 0, 0]},
                    {"root_pos": [0, 0, 1], "root_rot": [0, 0, 0, 1], "pelvis": [0, 0, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_csv(root / "target.csv")
    _write_csv(root / "play.csv")
    manifest_path = root / "packet.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    return manifest_path


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "VisPacket v0.1",
        "coordinate_standard": "isaac",
        "timeline": {"fps": 50, "dt": 0.02, "frame_count": 2, "sim_dt": 0.005, "dte": 4},
        "tracks": {
            "human": {
                "uri": "human.json",
                "format": "json",
                "coordinate": {
                    "standard": "soma_bvh",
                    "up_axis": "Y",
                    "forward_axis": "Z",
                    "handedness": "right",
                    "unit_length": "meter",
                    "unit_angle": "degree",
                    "root_rotation": "euler_xyz",
                },
                "joint_names": ["pelvis"],
            },
            "target_g1": {
                "uri": "target.csv",
                "format": "csv",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
            "play_g1": {
                "uri": "play.csv",
                "format": "csv",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
        },
        "render": {
            "enabled": True,
            "interface": "StaticRenderer",
            "config": {"source_bvh_uri": "source.bvh", "width": 320, "height": 240},
        },
        "physics": {"enabled": False},
        "diagnose": {"enabled": True},
    }


def _write_csv(path: Path) -> None:
    rows = [
        {"root_pos": "0 0 0", "root_rot": "0 0 0 1", "pelvis": "0 0 0"},
        {"root_pos": "0 0 1", "root_rot": "0 0 0 1", "pelvis": "0 0 1"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _arg_path_name(argv: tuple[str, ...], flag: str) -> str:
    return Path(argv[argv.index(flag) + 1]).name


if __name__ == "__main__":
    unittest.main()
