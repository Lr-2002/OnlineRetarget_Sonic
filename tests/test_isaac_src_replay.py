from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.isaac_src_replay import (
    SCHEMA_VERSION,
    SONIC_JOINT_NAMES,
    default_replay_config,
    inspect_paired_state_h5,
    main,
    packet_schema_payload,
    replay_config_from_mapping,
    run_dry_or_blocked_export,
    validate_artifact_summary,
    validate_replay_config,
)


class IsaacSrcReplayContractTests(unittest.TestCase):
    def test_default_contract_declares_contact_semantics(self) -> None:
        config = default_replay_config(
            paired_state_h5=Path("/tmp/paired_g1_state.h5"),
            robot_usd=Path("/tmp/g1.usd"),
        )
        schema = packet_schema_payload(config)

        self.assertEqual(config.schema_version, SCHEMA_VERSION)
        self.assertTrue(config.contact.enable_self_collisions)
        self.assertTrue(config.contact.activate_contact_sensors)
        self.assertEqual(
            config.contact.foot_links,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
        )
        self.assertIn("self_collision_count", schema["state_packet_fields"])
        self.assertIn("cross_ratio_contract", schema)

    def test_validation_rejects_visual_playback_contact_settings(self) -> None:
        config = replay_config_from_mapping(
            {
                "paired_state_h5": "/tmp/paired_g1_state.h5",
                "robot_usd": "/tmp/g1.usd",
                "contact": {
                    "enable_self_collisions": False,
                    "activate_contact_sensors": False,
                    "contact_filter_prim_paths": [],
                },
            }
        )

        errors = validate_replay_config(config)

        self.assertIn(
            "contact.enable_self_collisions must be true for self_collision_count packets",
            errors,
        )
        self.assertIn(
            "contact.activate_contact_sensors must be true for PhysX contact packets",
            errors,
        )
        self.assertIn("contact.contact_filter_prim_paths must include ground/filter prims", errors)

    def test_dry_run_writes_manifest_and_packet_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "paired_g1_state.h5"
            usd_path = root / "main.usd"
            output_dir = root / "out"
            usd_path.write_bytes(b"PXR-USDC")

            try:
                import h5py  # type: ignore
            except ImportError:
                self.skipTest("h5py is required for the minimal HDF5 dry-run fixture")

            _write_valid_paired_h5(h5_path, frames=1)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--paired-state-h5",
                        str(h5_path),
                        "--robot-usd",
                        str(usd_path),
                        "--output-dir",
                        str(output_dir),
                        "--variant",
                        "soma_uniform",
                        "--dry-run",
                    ]
                )

            manifest = json.loads((output_dir / "replay_manifest.json").read_text(encoding="utf-8"))
            schema = json.loads((output_dir / "packet_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(manifest["variant"], "soma_uniform")
        self.assertEqual(schema["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["paired_state_h5"]["hdf5_status"], "ok")
        self.assertEqual(manifest["paired_state_h5"]["frame_count"], 1)
        self.assertEqual(manifest["paired_state_h5"]["joint_count"], 29)
        self.assertEqual(manifest["artifact_errors"], [])
        self.assertIn("acceptance_smoke", manifest)

    def test_artifact_validation_requires_root_pose_for_replay_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "paired_g1_state.h5"
            try:
                import h5py  # type: ignore
            except ImportError:
                self.skipTest("h5py is required for the minimal HDF5 fixture")
            with h5py.File(h5_path, "w") as handle:
                pred = handle.create_group("pred_g1_state")
                target = handle.create_group("target_g1_state")
                pred.create_dataset("joint_q_rad", data=[[0.0] * 29])
                target.create_dataset("joint_q_rad", data=[[0.0] * 29])

            summary = inspect_paired_state_h5(h5_path)
            errors = validate_artifact_summary(summary)

        self.assertIn("pred root_pos_world_m", summary.missing_required_datasets)
        self.assertTrue(any("missing required datasets" in error for error in errors))

    def test_preflight_rejects_root_rot_alias_and_mismatched_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "paired_g1_state.h5"
            try:
                import h5py  # type: ignore
            except ImportError:
                self.skipTest("h5py is required for the malformed HDF5 fixture")
            with h5py.File(h5_path, "w") as handle:
                pred = handle.create_group("pred_g1_state")
                target = handle.create_group("target_g1_state")
                pred.create_dataset(
                    "root_pos_world_m",
                    data=[[0.0, 0.0, 0.8], [0.0, 0.0, 0.81]],
                )
                target.create_dataset("root_pos_world_m", data=[[0.0, 0.0, 0.8]])
                pred.create_dataset(
                    "root_rot",
                    data=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                )
                target.create_dataset("root_rot", data=[[1.0, 0.0, 0.0, 0.0]])
                pred.create_dataset("joint_q_rad", data=[[0.0] * 29, [0.1] * 29])
                target.create_dataset("joint_q_rad", data=[[0.0] * 29])

            summary = inspect_paired_state_h5(h5_path)
            errors = validate_artifact_summary(summary)

        self.assertIn("pred root_quat_wxyz", summary.missing_required_datasets)
        self.assertIn("target root_quat_wxyz", summary.missing_required_datasets)
        self.assertTrue(any("frame counts must match" in error for error in errors))

    def test_preflight_rejects_wrong_dataset_widths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "paired_g1_state.h5"
            try:
                import h5py  # type: ignore
            except ImportError:
                self.skipTest("h5py is required for the malformed HDF5 fixture")
            with h5py.File(h5_path, "w") as handle:
                pred = handle.create_group("pred_g1_state")
                target = handle.create_group("target_g1_state")
                pred.create_dataset("root_pos_world_m", data=[[0.0, 0.0]])
                target.create_dataset("root_pos_world_m", data=[[0.0, 0.0, 0.8]])
                pred.create_dataset("root_quat_wxyz", data=[[1.0, 0.0, 0.0]])
                target.create_dataset("root_quat_wxyz", data=[[1.0, 0.0, 0.0, 0.0]])
                pred.create_dataset("joint_q_rad", data=[[0.0] * 28])
                target.create_dataset("joint_q_rad", data=[[0.0] * 29])

            summary = inspect_paired_state_h5(h5_path)
            errors = validate_artifact_summary(summary)

        self.assertTrue(any("pred root_pos_world_m width must be 3" in error for error in errors))
        self.assertTrue(any("pred root_quat_wxyz width must be 4" in error for error in errors))
        self.assertTrue(any("pred joint_q_rad width must be 29" in error for error in errors))

    def test_non_dry_export_writes_real_jsonl_with_backend_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "paired_g1_state.h5"
            usd_path = root / "main.usd"
            output_dir = root / "out"
            usd_path.write_bytes(b"PXR-USDC")
            try:
                import h5py  # noqa: F401
            except ImportError:
                self.skipTest("h5py is required for the minimal HDF5 fixture")
            _write_valid_paired_h5(h5_path, frames=2)
            args = argparse.Namespace(
                paired_state_h5=h5_path,
                robot_usd=usd_path,
                output_dir=output_dir,
                config=None,
                variant="soma_uniform",
                max_frames=2,
                dry_run=False,
                device="cpu",
                backend_factory=lambda config: _FakeReplayBackend(config),
            )

            manifest = run_dry_or_blocked_export(args)
            packets = [
                json.loads(line)
                for line in (output_dir / "isaac_src_packets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["export_result"]["packets_written"], 2)
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0]["pred"]["contact_pairs"][0]["body_a"], "left_ankle_roll_link")
        self.assertAlmostEqual(packets[0]["target"]["body_pos_world_m"][0][2], 0.8)

def _write_valid_paired_h5(path: Path, *, frames: int) -> None:
    import h5py  # type: ignore

    root_pos = [[0.0, 0.0, 0.8 + frame * 0.01] for frame in range(frames)]
    root_quat = [[1.0, 0.0, 0.0, 0.0] for _frame in range(frames)]
    joint_q = [[frame * 0.1 for _joint in range(29)] for frame in range(frames)]
    with h5py.File(path, "w") as handle:
        pred = handle.create_group("pred_g1_state")
        target = handle.create_group("target_g1_state")
        pred.create_dataset("root_pos_world_m", data=root_pos)
        target.create_dataset("root_pos_world_m", data=root_pos)
        pred.create_dataset("root_quat_wxyz", data=root_quat)
        target.create_dataset("root_quat_wxyz", data=root_quat)
        pred.create_dataset("joint_q_rad", data=joint_q)
        target.create_dataset("joint_q_rad", data=joint_q)
        handle.attrs["fps"] = 50.0
        handle.attrs["joint_names_json"] = json.dumps(list(SONIC_JOINT_NAMES))


class _FakeReplayBackend:
    def __init__(self, config):
        self.config = config
        self.frames: list[tuple[str, int]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def collect_state_packet(self, state):
        self.frames.append((state.label, state.frame_idx))
        return {
            "root_pos_world_m": list(state.root_pos_world_m),
            "root_quat_wxyz": list(state.root_quat_wxyz),
            "joint_q_rad": list(state.joint_q_rad),
            "body_names": list(self.config.body_names),
            "body_pos_world_m": [list(state.root_pos_world_m) for _name in self.config.body_names],
            "foot_contact_force_n": [42.0, 0.0],
            "foot_in_contact": [True, False],
            "support_margin_m": 0.0,
            "floating_guard": False,
            "self_collision_count": 0,
            "contact_pairs": [
                {
                    "body_a": "left_ankle_roll_link",
                    "body_b": "/World/Ground",
                    "force_n": 42.0,
                    "position_world_m": list(state.root_pos_world_m),
                    "source": "fake_contact_sensor",
                }
            ],
            "cross_ratio": None,
            "cross_ratio_guard": None,
        }

    def report(self):
        return {
            "backend": "fake_isaaclab_contact_backend",
            "frames_collected": len(self.frames),
        }


if __name__ == "__main__":
    unittest.main()
