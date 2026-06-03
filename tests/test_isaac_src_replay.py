from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from online_retarget.isaac_src_replay import (
    SCHEMA_VERSION,
    default_replay_config,
    inspect_paired_state_h5,
    main,
    packet_schema_payload,
    replay_config_from_mapping,
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

            with h5py.File(h5_path, "w") as handle:
                pred = handle.create_group("pred_g1_state")
                target = handle.create_group("target_g1_state")
                pred.create_dataset("root_pos_world_m", data=[[0.0, 0.0, 0.8]])
                target.create_dataset("root_pos_world_m", data=[[0.0, 0.0, 0.8]])
                pred.create_dataset("root_quat_wxyz", data=[[1.0, 0.0, 0.0, 0.0]])
                target.create_dataset("root_quat_wxyz", data=[[1.0, 0.0, 0.0, 0.0]])
                pred.create_dataset("joint_q_rad", data=[[0.0] * 29])
                target.create_dataset("joint_q_rad", data=[[0.0] * 29])
                handle.attrs["fps"] = 50.0
                handle.attrs["joint_names_json"] = json.dumps(
                    [
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "waist_yaw_joint",
                        "left_hip_roll_joint",
                        "right_hip_roll_joint",
                        "waist_roll_joint",
                        "left_hip_yaw_joint",
                        "right_hip_yaw_joint",
                        "waist_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint",
                        "left_shoulder_roll_joint",
                        "right_shoulder_roll_joint",
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "right_elbow_joint",
                        "left_wrist_roll_joint",
                        "right_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                        "right_wrist_pitch_joint",
                        "left_wrist_yaw_joint",
                        "right_wrist_yaw_joint",
                    ]
                )

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

        self.assertIn("pred root_pos", summary.missing_required_datasets)
        self.assertTrue(any("missing required datasets" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
