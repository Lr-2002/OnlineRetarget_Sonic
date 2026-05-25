import importlib.util
from pathlib import Path
import tempfile
import unittest

from online_retarget.sonic_validation_callback import (
    _clip_report,
    _current_soma_routes,
    _render_triplet_video,
    _reset_policy_rollout_buffer,
    rank_video_indices,
    should_run_visual_validation,
    validation_frame_count,
)

RENDER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("imageio", "matplotlib", "numpy")
)

if RENDER_DEPS_AVAILABLE:
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


if __name__ == "__main__":
    unittest.main()
