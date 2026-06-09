from __future__ import annotations

import csv
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest

from vis_core import (
    SCHEMA_VERSION,
    CoordinateValidationError,
    TimelineValidationError,
    VisPacketSchemaError,
    coordinate_from_mapping,
    load_vis_packet,
    parse_vis_packet_manifest,
    run_static_packet,
)


class VisCoreTests(unittest.TestCase):
    def test_loads_vis_packet_v01_with_json_and_csv_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_fixture_packet(Path(tmp))

            loaded = load_vis_packet(manifest_path)

        self.assertEqual(loaded.packet.schema_version, SCHEMA_VERSION)
        self.assertEqual(loaded.packet.timeline.frame_count, 2)
        self.assertEqual(loaded.packet.timeline.dte, 4)
        self.assertEqual(set(loaded.tracks), {"human", "target_g1", "play_g1"})
        self.assertEqual(loaded.tracks["target_g1"].frame_count, 2)
        self.assertFalse(loaded.packet.physics.enabled)
        self.assertTrue(loaded.packet.diagnose.enabled)

    def test_schema_rejects_metrics_inside_vis_core(self) -> None:
        payload = _manifest_payload()
        payload["metrics"] = {"mpjpe": "external"}

        with self.assertRaisesRegex(VisPacketSchemaError, "metrics stay outside vis_core"):
            parse_vis_packet_manifest(payload)

        payload = _manifest_payload()
        payload["tracks"]["target_g1"]["metrics"] = {"score": 1.0}

        with self.assertRaisesRegex(VisPacketSchemaError, "metrics stay outside vis_core"):
            parse_vis_packet_manifest(payload)

    def test_timeline_validates_fps_dt_and_sim_dte_alignment(self) -> None:
        payload = _manifest_payload()
        payload["timeline"]["dt"] = 0.01

        with self.assertRaisesRegex(TimelineValidationError, "fps and timeline.dt"):
            parse_vis_packet_manifest(payload)

        payload = _manifest_payload()
        payload["timeline"]["dte"] = 3

        with self.assertRaisesRegex(TimelineValidationError, "sim_dt \\* timeline.dte"):
            parse_vis_packet_manifest(payload)

    def test_coordinate_validation_requires_isaac_standard_and_explicit_bvh_fields(self) -> None:
        coordinate = coordinate_from_mapping(
            {
                "standard": "soma_bvh",
                "up_axis": "Y",
                "forward_axis": "Z",
                "handedness": "right",
                "unit_length": "meter",
                "unit_angle": "degree",
                "root_rotation": "euler_xyz",
            }
        )
        self.assertFalse(coordinate.is_isaac)
        self.assertEqual(coordinate.up_axis, "Y")

        with self.assertRaises(CoordinateValidationError):
            coordinate_from_mapping({"standard": "isaac", "up_axis": "Y"})

        payload = _manifest_payload()
        payload["coordinate_standard"] = {
            "standard": "soma_bvh",
            "up_axis": "Y",
            "forward_axis": "Z",
            "handedness": "right",
            "unit_length": "meter",
            "unit_angle": "degree",
            "root_rotation": "euler_xyz",
        }
        with self.assertRaisesRegex(CoordinateValidationError, "Isaac coordinate standard"):
            parse_vis_packet_manifest(payload)

    def test_load_validation_requires_all_tracks_to_match_timeline_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_fixture_packet(root)
            human = root / "human.json"
            human.write_text(json.dumps({"frames": [{"frame": 0}]}), encoding="utf-8")

            with self.assertRaisesRegex(TimelineValidationError, "timeline.frame_count"):
                load_vis_packet(manifest_path)

    def test_static_runner_reports_diagnostics_without_physics_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_fixture_packet(Path(tmp))
            guard = _BlockedImportGuard({"mujoco", "isaaclab", "isaacsim", "newton", "warp"})
            sys.meta_path.insert(0, guard)
            try:
                result = run_static_packet(manifest_path)
            finally:
                sys.meta_path.remove(guard)

        diagnostics = result.diagnostics.as_dict()
        self.assertEqual(diagnostics["status"], "ok")
        self.assertEqual(diagnostics["frame_count"], 2)
        self.assertEqual(diagnostics["tracks"], ["human", "target_g1", "play_g1"])
        self.assertFalse(diagnostics["physics_executed"])
        self.assertIsNone(diagnostics["renderer"])
        self.assertGreaterEqual(diagnostics["load_sec"], 0.0)
        self.assertGreaterEqual(diagnostics["validate_sec"], 0.0)


class _BlockedImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_roots: set[str]) -> None:
        self._blocked_roots = blocked_roots

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        root_name = fullname.partition(".")[0]
        if root_name in self._blocked_roots:
            raise AssertionError(f"vis_core static path imported optional dependency: {fullname}")
        return None


def _write_fixture_packet(root: Path) -> Path:
    (root / "human.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"frame": 0, "root_pos": [0.0, 0.0, 0.0], "root_rot": [0.0, 0.0, 0.0]},
                    {"frame": 1, "root_pos": [0.0, 0.0, 1.0], "root_rot": [0.0, 0.0, 0.1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "target.csv",
        [
            {"frame": "0", "root_x": "0.0", "root_y": "0.0", "root_z": "0.9"},
            {"frame": "1", "root_x": "0.0", "root_y": "0.0", "root_z": "0.91"},
        ],
    )
    (root / "play.json").write_text(
        json.dumps(
            [
                {"frame": 0, "root_pos": [0.0, 0.0, 0.9], "root_rot": [0.0, 0.0, 0.0, 1.0]},
                {"frame": 1, "root_pos": [0.0, 0.0, 0.91], "root_rot": [0.0, 0.0, 0.0, 1.0]},
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = root / "packet.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    return manifest_path


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_standard": "isaac",
        "timeline": {
            "fps": 50,
            "dt": 0.02,
            "frame_count": 2,
            "sim_dt": 0.005,
            "dte": 4,
        },
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
                "root_fields": ["root_pos", "root_rot"],
            },
            "target_g1": {
                "uri": "target.csv",
                "format": "csv",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
            "play_g1": {
                "uri": "play.json",
                "format": "json",
                "coordinate": "isaac",
                "joint_names": ["pelvis"],
            },
        },
        "render": {"enabled": False, "interface": "StaticRenderer"},
        "physics": {"enabled": False, "interface": "PhysicsAdapter"},
        "diagnose": {"enabled": True, "interface": "DiagnoseAdapter"},
        "metadata": {"case": "fixture"},
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
