import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from online_retarget.sonic_validation_callback import (
    SonicVisualValidationCallback,
    _clip_report,
    _current_soma_routes,
    _render_triplet_video,
    _reset_policy_rollout_buffer,
    rank_video_indices,
    should_run_visual_validation,
    validation_frame_count,
)
from online_retarget.sonic_validation_export import (
    DEFAULT_READABLE_CLIP_INDICES,
    export_readable_validation_pack,
    load_raw_validation_trajectory,
    native_fps_review_evidence,
    parse_clip_indices,
    render_readable_validation_video,
    save_raw_validation_trajectory,
)

RENDER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("imageio", "matplotlib", "numpy")
)
RAW_DEPS_AVAILABLE = importlib.util.find_spec("numpy") is not None

if RAW_DEPS_AVAILABLE:
    import numpy as np


class SonicValidationCallbackTests(unittest.TestCase):
    def test_visual_validation_runs_only_on_positive_interval(self):
        self.assertFalse(should_run_visual_validation(0, 20000))
        self.assertFalse(should_run_visual_validation(19999, 20000))
        self.assertTrue(should_run_visual_validation(20000, 20000))
        self.assertFalse(should_run_visual_validation(20000, 20000, last_step=20000))

    def test_visual_validation_can_run_on_wall_clock_interval(self):
        self.assertFalse(
            should_run_visual_validation(
                499,
                20000,
                now=100.0,
                every_seconds=3600.0,
                last_time=99.0,
            )
        )
        self.assertTrue(
            should_run_visual_validation(
                500,
                20000,
                now=3700.0,
                every_seconds=3600.0,
                last_time=99.0,
            )
        )
        self.assertFalse(
            should_run_visual_validation(
                500,
                20000,
                last_step=500,
                now=3700.0,
                every_seconds=3600.0,
                last_time=99.0,
            )
        )

    def test_rank_video_indices_split_global_clips(self):
        self.assertEqual(rank_video_indices(8, 0, 4), (0, 4))
        self.assertEqual(rank_video_indices(8, 1, 4), (1, 5))
        self.assertEqual(rank_video_indices(8, 2, 4), (2, 6))
        self.assertEqual(rank_video_indices(8, 3, 4), (3, 7))

    def test_validation_frame_count_uses_sonic_target_frequency(self):
        self.assertEqual(validation_frame_count(4.0, 50), 200)
        self.assertEqual(validation_frame_count(0.0, 50), 1)

    def test_callback_defaults_avoid_evaluation_motion_loading(self):
        callback = SonicVisualValidationCallback()

        self.assertFalse(callback.use_evaluation_mode)
        self.assertTrue(callback.empty_cuda_cache)

    def test_current_soma_routes_uses_last_temporal_route(self):
        policy = _PolicyWithRoutes([[0, 1, 2], [3, 2, 1]])

        routes = _current_soma_routes(policy)

        self.assertEqual(list(routes), [2, 1])

    def test_validation_resets_rollout_buffer_without_clearing_aux_losses(self):
        policy = _PolicyWithAuxState()

        _reset_policy_rollout_buffer(policy)

        self.assertEqual(policy.init_rollout_calls, 1)
        self.assertEqual(policy.clear_rollout_calls, 0)
        self.assertEqual(policy.aux_losses, {"loss": 1.0})

    def test_clip_report_includes_encoder_route_counts(self):
        report = _clip_report(
            trajectory={
                "clip_index": 0,
                "local_env_index": 0,
                "target_g1": [1, 2, 3],
                "encoder_routes": [2, 2, 3],
            },
            video_path=__file__,
            step=20000,
            rank=0,
            world_size=1,
            target_fps=50,
            duration_sec=4.0,
        )

        self.assertEqual(report["encoder_route_first"], 2)
        self.assertEqual(report["encoder_route_last"], 3)
        self.assertEqual(report["encoder_route_counts"], {"2": 2, "3": 1})

    def test_parse_readable_clip_indices(self):
        self.assertEqual(parse_clip_indices(None), DEFAULT_READABLE_CLIP_INDICES)
        self.assertEqual(parse_clip_indices("[0,6]"), (0, 6))
        self.assertEqual(parse_clip_indices(["0", 6]), (0, 6))

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_raw_validation_trajectory_roundtrip(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "clip_00_fixture_trajectory.npz"
            report = save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_path,
                target_fps=50,
                duration_sec=3 / 50,
            )
            loaded = load_raw_validation_trajectory(raw_path)

        self.assertEqual(report["raw_trajectory_frames"], 3)
        self.assertEqual(loaded["clip_index"], 0)
        self.assertEqual(loaded["source_frame_indices"], [10, 11, 12])
        self.assertEqual(loaded["review_mode"], "native_fps_contiguous_rollout")
        self.assertEqual(loaded["fps"], 50.0)
        self.assertEqual(loaded["frame_count"], 3)
        self.assertEqual(loaded["source_frame_range"], [10, 12])
        evidence = native_fps_review_evidence(loaded)
        self.assertEqual(evidence["fps"], 50.0)
        self.assertEqual(evidence["frame_count"], 3)
        self.assertEqual(evidence["source_frame_range"], [10, 12])
        self.assertEqual(evidence["source_frame_indices_count"], 3)
        self.assertEqual(evidence["source_frame_indices_covered"], 3)
        self.assertTrue(evidence["physical_time_aligned"])
        self.assertTrue(evidence["final_review_eligible"])
        self.assertIsNone(evidence["blocked_reason"])
        self.assertEqual(
            tuple(loaded["g1_body_names"][:3]),
            ("pelvis", "left_hip_roll_link", "left_knee_link"),
        )
        self.assertEqual(loaded["target_g1"].shape, (3, 14, 3))
        self.assertEqual(loaded["target_root_pos_w"].shape, (3, 3))
        self.assertEqual(loaded["target_root_rot_w"].shape, (3, 4))
        self.assertEqual(loaded["root_rot_format"], "wxyz")
        self.assertFalse(loaded["initial_root_xy_zeroed"])

    @unittest.skipUnless(RENDER_DEPS_AVAILABLE, "render dependencies are required")
    def test_readable_validation_render_writes_labeled_mp4(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "readable.mp4"
            report = render_readable_validation_video(
                trajectory=trajectory,
                video_path=video_path,
                target_fps=50,
                duration_sec=3 / 50,
                width=960,
                height=384,
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["review_mode"], "native_fps_contiguous_rollout")
            self.assertEqual(report["frame_count"], 3)
            self.assertEqual(report["source_frame_range"], [10, 12])
            self.assertIn("floor_contact_grid", report["readable_features"])
            self.assertIn("root_rotation_axes", report["readable_features"])
            self.assertIn("source_target_frame_counters", report["readable_features"])
            self.assertTrue(video_path.exists())
            self.assertGreater(video_path.stat().st_size, 0)

    @unittest.skipUnless(RENDER_DEPS_AVAILABLE, "render dependencies are required")
    def test_export_readable_validation_pack_finds_latest_raw_trajectories(self):
        trajectory = _dummy_trajectory(frames=2, joints=14)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in ("A1_concat", "A2_film_contact", "B1_adapter", "B2_expert"):
                raw_dir = (
                    root
                    / f"sonic_bones_seed_{variant}_group"
                    / "online_retarget_visual_validation"
                    / "step_00000002"
                    / "rank_000"
                )
                save_raw_validation_trajectory(
                    trajectory=trajectory,
                    output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                    target_fps=50,
                    duration_sec=2 / 50,
                )
            result = export_readable_validation_pack(
                search_root=root,
                run_group="group",
                output_dir=root / "pack",
                clips=(0,),
                width=960,
                height=384,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.videos_ok, 4)
        self.assertEqual(result.missing, ())
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["review_contract"]["mode"], "native_fps_contiguous_rollout")
        self.assertTrue(manifest["review_contract"]["final_review_eligible"])
        self.assertEqual(manifest["results"][0]["fps"], 50.0)
        self.assertEqual(manifest["results"][0]["frame_count"], 2)
        self.assertEqual(manifest["results"][0]["source_frame_range"], [10, 11])

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_export_readable_validation_pack_manifest_contract_without_render_deps(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)

        def fake_render(*, trajectory, video_path, target_fps, duration_sec, width, height):
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake")
            return {
                "status": "ok",
                "video_path": str(video_path),
                "fps": float(target_fps),
                "frame_count": len(trajectory["target_g1"]),
                "source_frame_range": [10, 12],
                "review_mode": "native_fps_contiguous_rollout",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = (
                root
                / "sonic_bones_seed_A1_concat_group"
                / "online_retarget_visual_validation"
                / "step_00005000"
                / "rank_000"
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            with mock.patch(
                "online_retarget.sonic_validation_export.render_readable_validation_video",
                side_effect=fake_render,
            ):
                result = export_readable_validation_pack(
                    search_root=root,
                    run_group="group",
                    output_dir=root / "pack",
                    clips=(0,),
                    variants=("A1_concat",),
                    width=960,
                    height=384,
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertTrue(manifest["review_contract"]["final_review_eligible"])
        self.assertEqual(manifest["review_contract"]["evidence"][0]["fps"], 50.0)
        self.assertEqual(manifest["review_contract"]["evidence"][0]["frame_count"], 3)
        self.assertEqual(manifest["review_contract"]["evidence"][0]["source_frame_range"], [10, 12])
        self.assertEqual(manifest["results"][0]["review_contract"]["mode"], "native_fps_contiguous_rollout")
        self.assertIsNone(manifest["results"][0]["review_contract"]["blocked_reason"])

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_export_readable_validation_pack_discovers_accepted_vertical_v2_raw_trajectory(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)

        def fake_render(*, trajectory, video_path, target_fps, duration_sec, width, height):
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake")
            return {
                "status": "ok",
                "video_path": str(video_path),
                "fps": float(target_fps),
                "frame_count": len(trajectory["target_g1"]),
                "source_frame_range": [10, 12],
                "review_mode": "native_fps_contiguous_rollout",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = (
                root
                / "sonic_bones_seed_A1_concat_group"
                / "visual_validation"
                / "step_00005000"
                / "accepted_vertical_v2"
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            with mock.patch(
                "online_retarget.sonic_validation_export.render_readable_validation_video",
                side_effect=fake_render,
            ):
                result = export_readable_validation_pack(
                    search_root=root,
                    run_group="group",
                    output_dir=root / "pack",
                    clips=(0,),
                    variants=("A1_concat",),
                    width=960,
                    height=384,
                )
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.videos_ok, 1)
        self.assertTrue(manifest["results"][0]["raw_trajectory_path"].endswith("clip_00_fixture_trajectory.npz"))

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_export_readable_validation_pack_auto_discovers_supervised_variant_and_clip_indices(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)

        def fake_render(*, trajectory, video_path, target_fps, duration_sec, width, height):
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake")
            return {
                "status": "ok",
                "video_path": str(video_path),
                "fps": float(target_fps),
                "frame_count": len(trajectory["target_g1"]),
                "source_frame_range": [10, 12],
                "review_mode": "native_fps_contiguous_rollout",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = (
                root
                / "lr342_nativefps_smoke"
                / "proportional"
                / "visual_validation"
                / "step_00005000"
                / "accepted_vertical_v2"
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_01_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            with mock.patch(
                "online_retarget.sonic_validation_export.render_readable_validation_video",
                side_effect=fake_render,
            ):
                result = export_readable_validation_pack(
                    search_root=root,
                    run_group="lr342_nativefps_smoke",
                    output_dir=root / "pack",
                    width=960,
                    height=384,
                )
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.videos_ok, 2)
        self.assertEqual(manifest["variants"], ["proportional"])
        self.assertEqual(manifest["clips"], [0, 1])
        self.assertEqual(
            manifest["review_contract"]["selection_mode"],
            {"variants": "auto_discovered", "clips": "auto_discovered"},
        )
        self.assertEqual(
            [item["clip_index"] for item in manifest["results"]],
            [0, 1],
        )
        self.assertEqual(
            [item["variant"] for item in manifest["results"]],
            ["proportional", "proportional"],
        )

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_native_fps_review_evidence_requires_source_frame_provenance(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)
        trajectory["source_frame_indices"] = []

        evidence = native_fps_review_evidence(trajectory)

        self.assertFalse(evidence["final_review_eligible"])
        self.assertIsNone(evidence["source_frame_range"])
        self.assertEqual(evidence["source_frame_indices_count"], 0)
        self.assertEqual(evidence["source_frame_indices_covered"], 0)
        self.assertIn("source_frame_indices cover 0 of 3 rendered frames", evidence["blocked_reason"])
        self.assertIn("source_frame_range is required", evidence["blocked_reason"])

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_native_fps_review_evidence_requires_physical_time_alignment(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)
        trajectory["physical_time_aligned"] = False

        evidence = native_fps_review_evidence(trajectory)

        self.assertFalse(evidence["final_review_eligible"])
        self.assertEqual(evidence["source_frame_range"], [10, 12])
        self.assertFalse(evidence["physical_time_aligned"])
        self.assertIn("physical_time_aligned must be true", evidence["blocked_reason"])

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_export_readable_validation_pack_fails_without_source_frame_provenance(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)
        trajectory["source_frame_indices"] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = (
                root
                / "sonic_bones_seed_A1_concat_group"
                / "online_retarget_visual_validation"
                / "step_00005000"
                / "rank_000"
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            with mock.patch(
                "online_retarget.sonic_validation_export.render_readable_validation_video"
            ) as render_mock:
                result = export_readable_validation_pack(
                    search_root=root,
                    run_group="group",
                    output_dir=root / "pack",
                    clips=(0,),
                    variants=("A1_concat",),
                    width=960,
                    height=384,
                )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        render_mock.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertFalse(manifest["review_contract"]["final_review_eligible"])
        self.assertEqual(manifest["results"][0]["status"], "failed")
        self.assertIn("source_frame_indices cover 0 of 3 rendered frames", manifest["results"][0]["render"]["message"])
        self.assertFalse(manifest["results"][0]["review_contract"]["final_review_eligible"])
        self.assertIsNone(manifest["results"][0]["review_contract"]["source_frame_range"])

    @unittest.skipUnless(RAW_DEPS_AVAILABLE, "numpy is required")
    def test_export_readable_validation_pack_fails_without_physical_time_alignment(self):
        trajectory = _dummy_trajectory(frames=3, joints=14)
        trajectory["physical_time_aligned"] = False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = (
                root
                / "sonic_bones_seed_A1_concat_group"
                / "online_retarget_visual_validation"
                / "step_00005000"
                / "rank_000"
            )
            save_raw_validation_trajectory(
                trajectory=trajectory,
                output_path=raw_dir / "clip_00_fixture_trajectory.npz",
                target_fps=50,
                duration_sec=3 / 50,
            )
            with mock.patch(
                "online_retarget.sonic_validation_export.render_readable_validation_video"
            ) as render_mock:
                result = export_readable_validation_pack(
                    search_root=root,
                    run_group="group",
                    output_dir=root / "pack",
                    clips=(0,),
                    variants=("A1_concat",),
                    width=960,
                    height=384,
                )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        render_mock.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertFalse(manifest["review_contract"]["final_review_eligible"])
        self.assertEqual(manifest["results"][0]["status"], "failed")
        self.assertIn("physical_time_aligned must be true", manifest["results"][0]["render"]["message"])
        self.assertFalse(manifest["results"][0]["review_contract"]["final_review_eligible"])

    @unittest.skipUnless(RENDER_DEPS_AVAILABLE, "render dependencies are required")
    def test_render_triplet_video_writes_mp4_with_current_matplotlib(self):
        frames = 3
        joints = 14
        base = np.zeros((frames, joints, 3), dtype=np.float32)
        base[..., 0] = np.linspace(0.0, 1.0, joints)
        base[..., 2] = np.linspace(0.0, 0.5, joints)
        for frame_idx in range(frames):
            base[frame_idx, :, 1] = frame_idx * 0.02
        trajectory = {
            "source_soma": [base[frame_idx] for frame_idx in range(frames)],
            "target_g1": [
                base[frame_idx] + np.array([0.0, 0.5, 0.0])
                for frame_idx in range(frames)
            ],
            "inferred_g1": [
                base[frame_idx] + np.array([0.0, 1.0, 0.0])
                for frame_idx in range(frames)
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "triplet.mp4"
            _render_triplet_video(
                trajectory=trajectory,
                video_path=video_path,
                target_fps=50,
                duration_sec=frames / 50.0,
            )

            self.assertTrue(video_path.exists())
            self.assertGreater(video_path.stat().st_size, 0)


class _PolicyWithRoutes:
    def __init__(self, routes):
        self.actor_module = _ActorModule(routes)


class _PolicyWithAuxState:
    def __init__(self):
        self.aux_losses = {"loss": 1.0}
        self.init_rollout_calls = 0
        self.clear_rollout_calls = 0

    def init_rollout(self):
        self.init_rollout_calls += 1

    def clear_rollout(self):
        self.clear_rollout_calls += 1
        del self.aux_losses


class _ActorModule:
    def __init__(self, routes):
        self.encoders = {"soma": _Encoder(routes)}


class _Encoder:
    def __init__(self, routes):
        self.last_routes = routes


def _dummy_trajectory(frames: int, joints: int):
    base = np.zeros((frames, joints, 3), dtype=np.float32)
    base[..., 0] = np.linspace(0.0, 0.8, joints)
    base[..., 2] = np.linspace(0.1, 1.1, joints)
    for frame_idx in range(frames):
        base[frame_idx, :, 1] = frame_idx * 0.02
    root_quat = np.zeros((frames, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    root_pos = base[:, 0, :].astype(np.float32, copy=True)
    return {
        "clip_index": 0,
        "local_env_index": 0,
        "motion_id": 7,
        "motion_key": "fixture_motion",
        "source_soma": [base[frame_idx] for frame_idx in range(frames)],
        "target_g1": [base[frame_idx] + np.array([0.0, 0.3, 0.0]) for frame_idx in range(frames)],
        "inferred_g1": [base[frame_idx] + np.array([0.0, 0.6, 0.0]) for frame_idx in range(frames)],
        "target_root_pos_w": [root_pos[frame_idx] + np.array([0.0, 0.3, 0.0]) for frame_idx in range(frames)],
        "target_root_rot_w": [root_quat[frame_idx] for frame_idx in range(frames)],
        "pred_root_pos_w": [root_pos[frame_idx] + np.array([0.0, 0.6, 0.0]) for frame_idx in range(frames)],
        "pred_root_rot_w": [root_quat[frame_idx] for frame_idx in range(frames)],
        "root_rot_format": "wxyz",
        "initial_root_xy_zeroed": False,
        "source_frame_indices": [10 + index for index in range(frames)],
        "encoder_routes": [1 for _ in range(frames)],
        "source_fps": 50.0,
        "target_fps": 50.0,
        "physical_time_aligned": True,
        "source_soma_names": (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        "g1_body_names": (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
    }


if __name__ == "__main__":
    unittest.main()
