from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import online_retarget.isaac_src_replay as isaac_src_replay_module
from online_retarget.isaac_src_replay import (
    SCHEMA_VERSION,
    SONIC_JOINT_NAMES,
    default_replay_config,
    export_replay_packets,
    inspect_paired_state_h5,
    main,
    packet_schema_payload,
    PairedStateData,
    FootArtifactTracker,
    replay_config_from_mapping,
    run_dry_or_blocked_export,
    validate_artifact_summary,
    validate_replay_config,
    filtered_ground_contact_force_norm,
    foot_ground_contact_sensor_prim_paths,
    IsaacLabReplayBackend,
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
        self.assertEqual(
            foot_ground_contact_sensor_prim_paths(config),
            {
                "left_ankle_roll_link": "/World/Robot/left_ankle_roll_link",
                "right_ankle_roll_link": "/World/Robot/right_ankle_roll_link",
            },
        )
        self.assertIn("foot_ground_contact_pairs", schema["state_packet_fields"])
        self.assertIn("foot_ground_contact_status", schema["state_packet_fields"])
        self.assertIn("foot_slide_speed_mps", schema["state_packet_fields"])
        self.assertIn("foot_skate_distance_m", schema["state_packet_fields"])
        self.assertIn("foot_float_clearance_m", schema["state_packet_fields"])
        self.assertIn("foot_artifact_status", schema["state_packet_fields"])
        self.assertIn("body_pair_contacts", schema["state_packet_fields"])
        self.assertIn("body_pair_contact_status", schema["state_packet_fields"])
        self.assertIn("self_collision_count", schema["state_packet_fields"])
        self.assertIn("self_collision_status", schema["state_packet_fields"])
        self.assertIn("cross_ratio_status", schema["state_packet_fields"])
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
            "contact.enable_self_collisions must be true for body-body contact readiness",
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

    def test_foot_artifact_tracker_rolls_verified_contact_only(self) -> None:
        config = replay_config_from_mapping(
            {
                "fps": 50.0,
                "contact": {
                    "foot_slide_speed_threshold_mps": 0.25,
                    "foot_skate_distance_threshold_m": 0.02,
                    "support": {
                        "ground_height_m": 0.0,
                        "floating_clearance_threshold_m": 0.04,
                    },
                },
            }
        )
        tracker = FootArtifactTracker(config)

        first = tracker.update(
            label="pred",
            foot_positions=((0.0, 0.0, 0.01), (1.0, 0.0, 0.01)),
            foot_in_contact=(True, False),
            foot_ground_contact_status="available",
        )
        second = tracker.update(
            label="pred",
            foot_positions=((0.03, 0.0, 0.05), (1.0, 0.0, 0.01)),
            foot_in_contact=(True, False),
            foot_ground_contact_status="available",
        )

        self.assertEqual(first.foot_artifact_status, "available")
        self.assertEqual(first.foot_slide_speed_mps, (None, None))
        self.assertEqual(first.foot_skate_distance_m, (0.0, None))
        self.assertEqual(first.foot_float_flags, (False, None))
        self.assertEqual(second.foot_artifact_status, "available")
        self.assertAlmostEqual(second.foot_slide_speed_mps[0] or 0.0, 1.5)
        self.assertEqual(second.foot_slide_flags, (True, None))
        self.assertAlmostEqual(second.foot_skate_distance_m[0] or 0.0, 0.03)
        self.assertEqual(second.foot_skate_flags, (True, None))
        self.assertAlmostEqual(second.foot_float_clearance_m[0] or 0.0, 0.05)
        self.assertEqual(second.foot_float_flags, (True, None))

    def test_foot_artifact_tracker_blocks_and_clears_state_when_contact_unverified(self) -> None:
        config = default_replay_config()
        tracker = FootArtifactTracker(config)
        tracker.update(
            label="target",
            foot_positions=((0.0, 0.0, 0.01), (1.0, 0.0, 0.01)),
            foot_in_contact=(True, True),
            foot_ground_contact_status="available",
        )

        blocked = tracker.update(
            label="target",
            foot_positions=((0.05, 0.0, 0.01), (1.0, 0.0, 0.01)),
            foot_in_contact=(True, True),
            foot_ground_contact_status="blocked",
            foot_ground_contact_reason="force_matrix_w missing",
        )
        resumed = tracker.update(
            label="target",
            foot_positions=((0.10, 0.0, 0.01), (1.0, 0.0, 0.01)),
            foot_in_contact=(True, True),
            foot_ground_contact_status="available",
        )

        self.assertEqual(blocked.foot_artifact_status, "blocked")
        self.assertEqual(blocked.foot_artifact_reason, "force_matrix_w missing")
        self.assertEqual(blocked.foot_slide_speed_mps, (None, None))
        self.assertEqual(resumed.foot_slide_speed_mps, (None, None))
        self.assertEqual(resumed.foot_skate_distance_m, (0.0, 0.0))

    def test_export_replay_packets_emits_safe_foot_artifacts_without_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = default_replay_config()
            state = _minimal_paired_state(frames=2)

            result = export_replay_packets(
                config=config,
                state=state,
                output_dir=output_dir,
                max_frames=2,
                backend_factory=lambda replay_config: _FakeReplayBackend(replay_config),
            )
            packets = [
                json.loads(line)
                for line in (output_dir / "isaac_src_packets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(result["packets_written"], 2)
        self.assertEqual(packets[0]["pred"]["foot_artifact_status"], "available")
        self.assertEqual(packets[0]["pred"]["foot_slide_speed_mps"], [None, None])
        self.assertEqual(packets[1]["pred"]["foot_slide_flags"], [True, None])
        self.assertEqual(packets[1]["pred"]["foot_skate_flags"], [True, None])
        self.assertEqual(packets[1]["pred"]["foot_float_flags"], [True, None])
        self.assertIsNone(packets[1]["pred"]["body_pair_contacts"])
        self.assertEqual(packets[1]["pred"]["body_pair_contact_status"], "blocked")
        self.assertIsNone(packets[1]["pred"]["self_collision_count"])
        self.assertEqual(packets[1]["pred"]["self_collision_status"], "blocked")
        self.assertIsNone(packets[1]["pred"]["cross_ratio"])
        self.assertEqual(packets[1]["pred"]["cross_ratio_status"], "blocked")

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
            persisted_manifest = json.loads(
                (output_dir / "replay_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["export_result"]["packets_written"], 2)
        self.assertEqual(manifest["export_result"]["lifecycle_exit_strategy"], "normal_context_exit")
        self.assertEqual(persisted_manifest["status"], "completed")
        self.assertEqual(persisted_manifest["runtime_blocker"], "")
        self.assertEqual(persisted_manifest["export_result"]["packets_written"], 2)
        self.assertEqual(
            persisted_manifest["export_result"]["lifecycle_exit_strategy"],
            "normal_context_exit",
        )
        self.assertTrue(persisted_manifest["packet_jsonl_exists"])
        self.assertGreater(persisted_manifest["packet_jsonl_bytes"], 0)
        self.assertEqual(len(packets), 2)
        self.assertEqual(
            packets[0]["pred"]["foot_ground_contact_pairs"][0]["body_a"],
            "left_ankle_roll_link",
        )
        self.assertEqual(
            packets[0]["pred"]["contact_pairs"],
            packets[0]["pred"]["foot_ground_contact_pairs"],
        )
        self.assertEqual(packets[0]["pred"]["foot_artifact_status"], "available")
        self.assertEqual(packets[0]["pred"]["foot_slide_speed_mps"], [None, None])
        self.assertEqual(packets[1]["pred"]["foot_slide_flags"], [True, None])
        self.assertEqual(packets[1]["pred"]["foot_skate_flags"], [True, None])
        self.assertEqual(packets[1]["pred"]["foot_float_flags"], [True, None])
        self.assertIsNone(packets[0]["pred"]["body_pair_contacts"])
        self.assertEqual(packets[0]["pred"]["body_pair_contact_status"], "blocked")
        self.assertIsNone(packets[0]["pred"]["self_collision_count"])
        self.assertEqual(packets[0]["pred"]["self_collision_status"], "blocked")
        self.assertIsNone(packets[0]["pred"]["cross_ratio"])
        self.assertEqual(packets[0]["pred"]["cross_ratio_status"], "blocked")
        self.assertAlmostEqual(packets[0]["target"]["body_pos_world_m"][0][2], 0.8)

    def test_filtered_ground_force_ignores_aggregate_net_force(self) -> None:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            self.skipTest("numpy is required for the contact force regression fixture")
        config = default_replay_config()
        sensor = _FakeContactSensor(
            net_forces_w=np.asarray([[[100.0, 0.0, 0.0]]], dtype=np.float32),
            force_matrix_w=np.asarray([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
        )

        reading = filtered_ground_contact_force_norm(sensor, config)

        self.assertEqual(reading.status, "available")
        self.assertEqual(reading.force_n, 0.0)

    def test_zero_filtered_ground_force_emits_no_support_pair(self) -> None:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            self.skipTest("numpy is required for the contact force regression fixture")
        config = default_replay_config()
        backend = IsaacLabReplayBackend(config, device="cpu")
        backend.body_names = config.body_names
        backend.foot_contact_sensors = {
            foot_link: _FakeContactSensor(
                net_forces_w=np.asarray([[[100.0, 0.0, 0.0]]], dtype=np.float32),
                force_matrix_w=np.asarray([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
            )
            for foot_link in config.contact.foot_links
        }
        body_pos = [[0.0, 0.0, 0.8] for _name in config.body_names]

        foot_ground_state = backend._foot_contact_state()
        pairs = backend._foot_ground_contact_pairs(
            body_pos=body_pos,
            foot_ground_state=foot_ground_state,
        )

        self.assertEqual(foot_ground_state.status, "available")
        self.assertEqual(foot_ground_state.forces_n, (0.0, 0.0))
        self.assertEqual(foot_ground_state.flags, (False, False))
        self.assertEqual(pairs, [])

    def test_filtered_ground_force_blocks_when_force_matrix_missing(self) -> None:
        try:
            import numpy as np  # type: ignore
        except ImportError:
            self.skipTest("numpy is required for the contact force regression fixture")
        config = default_replay_config()
        sensor = _FakeContactSensor(
            net_forces_w=np.asarray([[[100.0, 0.0, 0.0]]], dtype=np.float32),
            force_matrix_w=None,
        )

        reading = filtered_ground_contact_force_norm(sensor, config)

        self.assertEqual(reading.status, "blocked")
        self.assertIsNone(reading.force_n)

    def test_hard_exit_backend_persists_completed_manifest_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            packet_jsonl = output_dir / "isaac_src_packets.jsonl"
            manifest = {
                "status": "ready",
                "runtime_blocker": "pre-export",
                "packet_jsonl": str(packet_jsonl),
            }
            config = default_replay_config()
            state = _minimal_paired_state(frames=1)
            exit_calls = 0
            original_exit = isaac_src_replay_module._exit_process_success

            def completion_callback(export_result):
                manifest["status"] = "completed"
                manifest["runtime_blocker"] = ""
                manifest["export_result"] = export_result
                manifest["packet_jsonl_exists"] = packet_jsonl.exists()
                manifest["packet_jsonl_bytes"] = (
                    packet_jsonl.stat().st_size if packet_jsonl.exists() else 0
                )
                (output_dir / "replay_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            def fake_exit() -> None:
                nonlocal exit_calls
                exit_calls += 1
                raise SystemExit(0)

            try:
                isaac_src_replay_module._exit_process_success = fake_exit
                with self.assertRaises(SystemExit) as raised:
                    export_replay_packets(
                        config=config,
                        state=state,
                        output_dir=output_dir,
                        max_frames=1,
                        backend_factory=lambda replay_config: _HardExitReplayBackend(
                            replay_config
                        ),
                        completion_callback=completion_callback,
                    )
            finally:
                isaac_src_replay_module._exit_process_success = original_exit

            persisted_manifest = json.loads(
                (output_dir / "replay_manifest.json").read_text(encoding="utf-8")
            )
            packets = (output_dir / "isaac_src_packets.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(exit_calls, 1)
        self.assertEqual(persisted_manifest["status"], "completed")
        self.assertEqual(persisted_manifest["runtime_blocker"], "")
        self.assertEqual(persisted_manifest["export_result"]["packets_written"], 1)
        self.assertEqual(
            persisted_manifest["export_result"]["lifecycle_exit_strategy"],
            "os._exit(0)_after_completed_manifest",
        )
        self.assertTrue(persisted_manifest["packet_jsonl_exists"])
        self.assertGreater(persisted_manifest["packet_jsonl_bytes"], 0)
        self.assertEqual(len(packets), 1)


def _minimal_paired_state(*, frames: int) -> PairedStateData:
    root_pos = [[0.0, 0.0, 0.8 + frame * 0.01] for frame in range(frames)]
    root_quat = [[1.0, 0.0, 0.0, 0.0] for _frame in range(frames)]
    joint_q = [[frame * 0.1 for _joint in range(29)] for frame in range(frames)]
    return PairedStateData(
        frame_count=frames,
        fps=50.0,
        joint_names=SONIC_JOINT_NAMES,
        pred_root_pos_world_m=root_pos,
        target_root_pos_world_m=root_pos,
        pred_root_quat_wxyz=root_quat,
        target_root_quat_wxyz=root_quat,
        pred_joint_q_rad=joint_q,
        target_joint_q_rad=joint_q,
    )


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
        self.foot_artifact_tracker = FootArtifactTracker(config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def collect_state_packet(self, state):
        self.frames.append((state.label, state.frame_idx))
        body_pos = [list(state.root_pos_world_m) for _name in self.config.body_names]
        left_foot_index = self.config.body_names.index("left_ankle_roll_link")
        right_foot_index = self.config.body_names.index("right_ankle_roll_link")
        body_pos[left_foot_index] = [
            float(state.root_pos_world_m[0]) + state.frame_idx * 0.03,
            float(state.root_pos_world_m[1]),
            0.05 if state.frame_idx else 0.01,
        ]
        body_pos[right_foot_index] = [1.0, 0.0, 0.01]
        foot_artifacts = self.foot_artifact_tracker.update(
            label=state.label,
            foot_positions=(body_pos[left_foot_index], body_pos[right_foot_index]),
            foot_in_contact=(True, False),
            foot_ground_contact_status="available",
        )
        return {
            "root_pos_world_m": list(state.root_pos_world_m),
            "root_quat_wxyz": list(state.root_quat_wxyz),
            "joint_q_rad": list(state.joint_q_rad),
            "body_names": list(self.config.body_names),
            "body_pos_world_m": body_pos,
            "foot_contact_force_n": [42.0, 0.0],
            "foot_in_contact": [True, False],
            "foot_ground_contact_status": "available",
            "foot_ground_contact_reason": "",
            "support_margin_m": 0.0,
            "floating_guard": False,
            "floating_guard_status": "available",
            "floating_guard_reason": "",
            "foot_slide_speed_mps": list(foot_artifacts.foot_slide_speed_mps),
            "foot_slide_flags": list(foot_artifacts.foot_slide_flags),
            "foot_skate_distance_m": list(foot_artifacts.foot_skate_distance_m),
            "foot_skate_flags": list(foot_artifacts.foot_skate_flags),
            "foot_float_clearance_m": list(foot_artifacts.foot_float_clearance_m),
            "foot_float_flags": list(foot_artifacts.foot_float_flags),
            "foot_artifact_status": foot_artifacts.foot_artifact_status,
            "foot_artifact_reason": foot_artifacts.foot_artifact_reason,
            "foot_ground_contact_pairs": [
                {
                    "body_a": "left_ankle_roll_link",
                    "body_b": "/World/Ground",
                    "force_n": 42.0,
                    "position_world_m": list(state.root_pos_world_m),
                    "source": "fake_single_body_foot_ground_filtered_force_matrix",
                }
            ],
            "contact_pairs": [
                {
                    "body_a": "left_ankle_roll_link",
                    "body_b": "/World/Ground",
                    "force_n": 42.0,
                    "position_world_m": list(state.root_pos_world_m),
                    "source": "fake_single_body_foot_ground_filtered_force_matrix",
                }
            ],
            "body_pair_contacts": None,
            "body_pair_contact_status": "blocked",
            "body_pair_contact_reason": "fake backend has no verified body-body source",
            "self_collision_count": None,
            "self_collision_status": "blocked",
            "self_collision_reason": "fake backend has no verified body-body source",
            "cross_ratio": None,
            "cross_ratio_guard": None,
            "cross_ratio_status": "blocked",
            "cross_ratio_reason": "fake backend has no SRC geometry checker",
        }

    def report(self):
        return {
            "backend": "fake_isaaclab_contact_backend",
            "frames_collected": len(self.frames),
        }


class _HardExitReplayBackend(_FakeReplayBackend):
    requires_hard_exit_after_success = True


class _FakeContactSensor:
    def __init__(self, *, net_forces_w, force_matrix_w):
        self.data = _FakeContactData(
            net_forces_w=net_forces_w,
            force_matrix_w=force_matrix_w,
        )


class _FakeContactData:
    def __init__(self, *, net_forces_w, force_matrix_w):
        self.net_forces_w = net_forces_w
        self.force_matrix_w = force_matrix_w


if __name__ == "__main__":
    unittest.main()
