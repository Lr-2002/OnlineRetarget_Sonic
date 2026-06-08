import unittest
from pathlib import Path
import tempfile

import numpy as np

try:
    import scripts.train_sonic_kin_skeleton_ae as sonic_train
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    sonic_train = None


if sonic_train is not None:
    class _FixedPredictionModel(sonic_train.nn.Module):
        def __init__(self, output: np.ndarray):
            super().__init__()
            self.register_buffer(
                "_output",
                sonic_train.torch.as_tensor(output, dtype=sonic_train.torch.float32),
            )

        def forward(self, motion, skeleton):
            return self._output[: motion.shape[0]].to(motion.device)


@unittest.skipIf(sonic_train is None, "torch is required for sonic kin trainer tests")
class SonicKinTrainTimingTests(unittest.TestCase):
    def test_shared_evaluation_cohort_ignores_variant_name_and_keeps_visual_subset(self):
        rows = [
            {
                "relative_path": f"sample_{index:03d}.pkl",
                "filename": f"sample_{index:03d}",
                "robot_relative_path": f"robot/sample_{index:03d}.pkl",
                "soma_relative_path": f"soma/sample_{index:03d}.pkl",
                "source_soma_proportional_path": f"soma_proportional/sample_{index:03d}.bvh",
                "frame_count": 120 + index,
                "split": "validation",
            }
            for index in range(120)
        ]
        base_config = {
            "training": {"seed": 1234},
            "visual_validation": {"num_videos": 8},
            "evaluation_cohort": {
                "enabled": True,
                "id": "lr270_shared_eval_v1",
                "seed": 20260608,
                "include_run_group": True,
                "visual_num_samples": 8,
                "metric_num_samples": 100,
            },
            "variant": {"name": "treatment_variant"},
            "wandb": {"name": "treatment_run"},
        }
        treatment = sonic_train.build_evaluation_cohort(rows, base_config, run_group="shared_group")
        baseline_config = {
            **base_config,
            "variant": {"name": "loss_off_baseline_variant"},
            "wandb": {"name": "baseline_run"},
        }
        baseline = sonic_train.build_evaluation_cohort(rows, baseline_config, run_group="shared_group")

        treatment_metric_keys = [
            sonic_train.evaluation_row_stable_key(row) for row in treatment["metric_rows"]
        ]
        baseline_metric_keys = [
            sonic_train.evaluation_row_stable_key(row) for row in baseline["metric_rows"]
        ]
        treatment_visual_keys = [
            sonic_train.evaluation_row_stable_key(row) for row in treatment["visual_rows"]
        ]
        baseline_visual_keys = [
            sonic_train.evaluation_row_stable_key(row) for row in baseline["visual_rows"]
        ]

        self.assertEqual(len(treatment_metric_keys), 100)
        self.assertEqual(len(treatment_visual_keys), 8)
        self.assertEqual(treatment_metric_keys, baseline_metric_keys)
        self.assertEqual(treatment_visual_keys, baseline_visual_keys)
        self.assertEqual(treatment_visual_keys, treatment_metric_keys[:8])
        self.assertTrue(treatment["visual_subset_of_metric"])

        manifest = sonic_train.evaluation_cohort_manifest_payload(treatment, Path("eval_cohort_manifest.json"))
        baseline_manifest = sonic_train.evaluation_cohort_manifest_payload(
            baseline,
            Path("baseline_eval_cohort_manifest.json"),
        )
        self.assertEqual(manifest["metric_row_count"], 100)
        self.assertEqual(manifest["visual_row_count"], 8)
        self.assertEqual(
            [row["stable_key"] for row in manifest["visual_rows"]],
            [row["stable_key"] for row in manifest["metric_rows"][:8]],
        )
        self.assertEqual(len(manifest["metric_rows_sha256"]), 64)
        self.assertEqual(len(manifest["visual_rows_sha256"]), 64)
        self.assertEqual(manifest["metric_rows_sha256"], baseline_manifest["metric_rows_sha256"])
        self.assertEqual(manifest["visual_rows_sha256"], baseline_manifest["visual_rows_sha256"])
        self.assertIn("variant.name", manifest["sampling"]["excluded_config_fields"])
        self.assertIn("wandb.name", manifest["sampling"]["excluded_config_fields"])

    def _predict_state(
        self,
        config,
        pred,
        *,
        joint_dim=2,
        fallback_root_pos=None,
        fallback_root_quat=None,
    ):
        pred = np.asarray(pred, dtype=np.float32)[None, :]
        fallback_root_pos = (
            np.asarray(fallback_root_pos, dtype=np.float32)
            if fallback_root_pos is not None
            else np.zeros((1, 3), dtype=np.float32)
        )
        fallback_root_quat = (
            np.asarray(fallback_root_quat, dtype=np.float32)
            if fallback_root_quat is not None
            else np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        torch = sonic_train.torch
        stats = {
            "motion_mean": torch.zeros(4, dtype=torch.float32),
            "motion_std": torch.ones(4, dtype=torch.float32),
            "skeleton_mean": torch.zeros(2, dtype=torch.float32),
            "skeleton_std": torch.ones(2, dtype=torch.float32),
            "target_mean": torch.zeros(pred.shape[1], dtype=torch.float32),
            "target_std": torch.ones(pred.shape[1], dtype=torch.float32),
        }
        return sonic_train._predict_g1_state_from_features(
            model=_FixedPredictionModel(pred),
            motion=np.zeros((1, 4), dtype=np.float32),
            skeleton=np.zeros((1, 2), dtype=np.float32),
            stats=stats,
            device=torch.device("cpu"),
            config=config,
            joint_dim=joint_dim,
            fallback_root_pos=fallback_root_pos,
            fallback_root_quat=fallback_root_quat,
        )

    @staticmethod
    def _prediction_with_root_target(root_pos):
        joint_dim = 2
        window = 1
        command_dim = window * joint_dim * 2
        pred = np.zeros(command_dim + window * 3 + window * 6, dtype=np.float32)
        pred[command_dim : command_dim + 3] = np.asarray(root_pos, dtype=np.float32)
        pred[command_dim + 3 : command_dim + 9] = np.asarray(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return pred

    def test_robot_root_rot_to_wxyz_converts_gmr_xyzw_motionlib_quat(self):
        root_rot_xyzw = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        converted = sonic_train.robot_root_rot_to_wxyz(
            root_rot_xyzw,
            {"input_data": {"robot_root_rot_format": "xyzw"}},
        )

        np.testing.assert_allclose(converted, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    def test_time_align_frame_maps_maps_target_time_to_source_time(self):
        frames = [{"Hips": (float(index), 0.0, 0.0)} for index in range(20)]

        aligned, indices = sonic_train._time_align_frame_maps(
            frames,
            source_fps=120.0,
            target_fps=50.0,
            frame_count=5,
        )

        self.assertEqual(indices, [0, 2, 4, 7, 9])
        self.assertEqual([frame["Hips"][0] for frame in aligned], [0.0, 2.0, 4.0, 7.0, 9.0])

    def test_source_target_timing_summary_accepts_120hz_source_to_50hz_target(self):
        summary = sonic_train.source_target_timing_summary(
            {"move_duration_frames": "120", "fps": "50"},
            frame_count=50,
            indexing={"source_fps": 120.0, "max_duration_delta_sec": 0.02},
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["flags"], [])

    def test_source_target_timing_summary_rejects_target_that_is_not_slower(self):
        summary = sonic_train.source_target_timing_summary(
            {"move_duration_frames": "50", "fps": "120"},
            frame_count=50,
            indexing={"source_fps": 120.0, "max_duration_delta_sec": 0.02},
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertIn("target_fps_not_below_source_fps", summary["flags"])

    def test_raw_sonic_dataset_setting_only_accepts_kin_or_phy(self):
        sonic_train.validate_raw_sonic_dataset_config(
            {
                "input_data": {
                    "dataset": "kin",
                    "data_root": "/tmp/bones_sonic",
                    "indexing": {},
                }
            }
        )
        sonic_train.validate_raw_sonic_dataset_config(
            {
                "input_data": {
                    "dataset": "phy",
                    "data_root": "/tmp/phsical_bones_sonic",
                    "indexing": {},
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "input_data.dataset must be one of: kin, phy"):
            sonic_train.validate_raw_sonic_dataset_config(
                {
                    "input_data": {
                        "dataset": "physical",
                        "data_root": "/tmp/phsical_bones_sonic",
                        "indexing": {},
                    }
                }
            )

    def test_phy_dataset_remaps_existing_sonic_index_rows_to_physicalized_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kin_root = root / "bones_sonic"
            phy_root = root / "phsical_bones_sonic"
            kin_root.mkdir()
            phy_root.mkdir()
            (phy_root / "data.txt").write_text("231121/foo.npz\n", encoding="utf-8")
            index_csv = root / "sonic_index.csv"
            index_csv.write_text(
                "schema_status,sonic_path,frame_count,fps,filename\n"
                ",/home/user/data/motion_data/bones_sonic/231121/foo.npz,80,50,foo.npz\n"
                ",/home/user/data/motion_data/bones_sonic/231121/not_saved.npz,80,50,not_saved.npz\n",
                encoding="utf-8",
            )
            config = {
                "input_data": {
                    "dataset": "phy",
                    "data_root": str(kin_root),
                    "dataset_roots": {
                        "kin": str(kin_root),
                        "phy": str(phy_root),
                    },
                    "dataset_manifests": {
                        "phy": str(phy_root / "data.txt"),
                    },
                    "indexing": {
                        "index_csv": str(index_csv),
                        "source_path_prefix": "/home/user/data/motion_data/bones_sonic",
                    },
                }
            }

            rows, skipped = sonic_train.rows_from_csv_index(config, sonic_train.data_root_from_config(config))

            self.assertEqual(skipped, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["path"], str(phy_root / "231121/foo.npz"))
            self.assertEqual(rows[0]["relative_path"], "231121/foo.npz")

    def test_soma_resample_matches_sonic_target_timeline_length(self):
        source = np.arange(330, dtype=np.float32).reshape(330, 1)

        resampled = sonic_train.resample_soma_array(source, 120.0, 50.0)

        self.assertEqual(resampled.shape, (138, 1))
        self.assertEqual(float(resampled[0, 0]), 0.0)
        self.assertGreater(float(resampled[-1, 0]), 320.0)

    def test_soma_motionlib_feature_builder_uses_soma_source_and_g1_kin_target(self):
        frames = 8
        soma_joints = np.zeros((frames, 26, 3), dtype=np.float32)
        soma_joints[:, :, 0] = np.arange(26, dtype=np.float32)
        soma_joints[:, :, 1] = np.arange(frames, dtype=np.float32)[:, None]
        identity = np.zeros((frames, 4), dtype=np.float32)
        identity[:, 0] = 1.0
        dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.01
        arrays = {
            "soma_joints": soma_joints,
            "soma_root_quat": identity.copy(),
            "joint_pos": dof,
            "joint_vel": sonic_train.finite_difference_velocity(dof, 50.0),
            "root_rot": identity.copy(),
        }

        motion, skeleton, target = sonic_train.build_soma_motionlib_features(
            arrays,
            np.asarray([0, 1], dtype=np.int64),
            window=3,
            step=1,
        )

        self.assertEqual(motion.shape, (2, 3 * 26 * 3 + 3 * 6))
        self.assertEqual(skeleton.shape, (2, 26 * 3 + 26))
        self.assertEqual(target.shape, (2, 3 * (29 + 29) + 3 * 6))
        np.testing.assert_allclose(target[0, :29], dof[0])

    def test_soma_motionlib_feature_builder_can_target_root_pos_and_rot_w(self):
        frames = 8
        soma_joints = np.zeros((frames, 26, 3), dtype=np.float32)
        identity = np.zeros((frames, 4), dtype=np.float32)
        identity[:, 0] = 1.0
        dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.01
        root_pos = np.stack(
            [
                np.linspace(0.0, 0.7, frames, dtype=np.float32),
                np.zeros(frames, dtype=np.float32),
                np.ones(frames, dtype=np.float32),
            ],
            axis=-1,
        )
        arrays = {
            "soma_joints": soma_joints,
            "soma_root_quat": identity.copy(),
            "joint_pos": dof,
            "joint_vel": sonic_train.finite_difference_velocity(dof, 50.0),
            "root_pos": root_pos,
            "root_rot": identity.copy(),
        }
        config = {"features": {"include_root_pos_target": True}}

        _, _, target = sonic_train.build_soma_motionlib_features(
            arrays,
            np.asarray([0, 1], dtype=np.int64),
            window=3,
            step=1,
            config=config,
        )

        command_dim = 3 * (29 + 29)
        root_pos_start = command_dim
        root_rot_start = command_dim + 3 * 3
        self.assertEqual(target.shape, (2, command_dim + 3 * 3 + 3 * 6))
        np.testing.assert_allclose(target[0, root_pos_start : root_pos_start + 3], root_pos[0])
        np.testing.assert_allclose(
            target[1, root_pos_start : root_pos_start + 3],
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            target[1, root_pos_start + 3 : root_pos_start + 6],
            np.asarray([0.1, 0.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            target[0, root_rot_start : root_rot_start + 6],
            np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_predict_soma_root_target_composes_local_xy_to_world(self):
        config = {
            "input_data": {"format": "soma_motionlib"},
            "features": {"future_window_frames": 1, "include_root_pos_target": True},
        }

        state = self._predict_state(
            config,
            self._prediction_with_root_target([0.25, -0.5, 1.2]),
            fallback_root_pos=[[10.0, 20.0, 0.3]],
        )

        np.testing.assert_allclose(
            state["root_pos"],
            np.asarray([[10.25, 19.5, 1.2]], dtype=np.float32),
            atol=1e-6,
        )

    def test_predict_non_soma_root_target_keeps_predicted_root(self):
        config = {
            "input_data": {"format": "npz"},
            "features": {"future_window_frames": 1, "include_root_pos_target": True},
        }

        state = self._predict_state(
            config,
            self._prediction_with_root_target([0.25, -0.5, 1.2]),
            fallback_root_pos=[[10.0, 20.0, 0.3]],
        )

        np.testing.assert_allclose(
            state["root_pos"],
            np.asarray([[0.25, -0.5, 1.2]], dtype=np.float32),
            atol=1e-6,
        )

    def test_predict_soma_without_root_target_keeps_fallback_root(self):
        config = {
            "input_data": {"format": "soma_motionlib"},
            "features": {"include_root_pos_target": False},
        }
        fallback_root_pos = np.asarray([[10.0, 20.0, 0.3]], dtype=np.float32)
        fallback_root_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        state = self._predict_state(
            config,
            [0.25, -0.5],
            fallback_root_pos=fallback_root_pos,
            fallback_root_quat=fallback_root_quat,
        )

        np.testing.assert_allclose(state["root_pos"], fallback_root_pos)
        np.testing.assert_allclose(state["root_quat"], fallback_root_quat)

    def test_accepted_body_position_metric_report_uses_uniform_tracking_body_contract(self):
        def fake_fk(_model, _joints, root_position, root_euler, include_empty_body_origin=True):
            del root_euler, include_empty_body_origin
            return {
                name: ((float(root_position[0]), float(index), 0.0),)
                for index, name in enumerate(sonic_train.A0_TRACKING_BODY_NAMES)
            }

        frames = 2
        identity = np.zeros((frames, 4), dtype=np.float32)
        identity[:, 0] = 1.0
        zero_joints = np.zeros((frames, 29), dtype=np.float32)
        target_root = np.zeros((frames, 3), dtype=np.float32)
        predicted_root = np.zeros((frames, 3), dtype=np.float32)
        predicted_root[:, 0] = 0.5

        report = sonic_train._accepted_body_position_metric_report(
            target_joint_pos=zero_joints,
            target_root_pos=target_root,
            target_root_quat=identity,
            predicted_joint_pos=zero_joints,
            predicted_root_pos=predicted_root,
            predicted_root_quat=identity,
            fps=50.0,
            g1_model=object(),
            render_deps={"g1_fk_body_positions": fake_fk},
            target_motion_path=Path("row2_g1_target_motion.npz"),
            prediction_motion_path=Path("row3_g1_kinematics_motion.npz"),
        )

        self.assertEqual(report["status"], "available")
        self.assertEqual(report["body_names"], list(sonic_train.A0_TRACKING_BODY_NAMES))
        self.assertEqual(report["weight_policy"], "uniform_14_tracking_bodies")
        self.assertTrue(report["metric_contract"]["pinned"])
        self.assertEqual(
            report["metric_contract"]["root_alignment"],
            "world_g1_root_no_pelvis_subtraction",
        )
        self.assertEqual(report["frame_count"], 2)
        self.assertEqual(report["sample_count"], 28.0)
        self.assertAlmostEqual(report["metric_results"]["mpjpe"]["value"], 0.5)
        self.assertAlmostEqual(report["metric_results"]["w_mpjpe"]["value"], 0.5)

    def test_kin_visual_validation_due_accepts_wall_clock_cadence(self):
        config = {
            "visual_validation": {
                "enabled": True,
                "every_steps": 20000,
                "every_minutes": 60,
            }
        }

        self.assertFalse(
            sonic_train.visual_validation_due(config, 499, now=3599.0, last_time=0.0)
        )
        self.assertTrue(
            sonic_train.visual_validation_due(config, 500, now=3600.0, last_time=0.0)
        )

    def test_rank0_stage_status_round_trip_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            path = sonic_train.rank0_stage_status_path(output_dir, "visual_validation", step=2000)
            sonic_train.write_rank0_stage_status(
                path,
                {
                    "status": "ok",
                    "stage": "visual_validation",
                    "step": 2000,
                    "metrics": {"visual_validation/videos_ok": 4.0},
                },
            )

            payload = sonic_train.wait_for_rank0_stage_status(path, timeout_sec=0.2, poll_sec=0.01)

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["stage"], "visual_validation")
            self.assertEqual(payload["step"], 2000)
            missing = sonic_train.rank0_stage_status_path(output_dir, "training_finalize")
            with self.assertRaises(TimeoutError):
                sonic_train.wait_for_rank0_stage_status(missing, timeout_sec=0.01, poll_sec=0.01)

    def test_accepted_visual_metrics_failed_only_for_acceptance_backend(self):
        failed_metrics = {
            "visual_validation/videos_ok": 4.0,
            "visual_validation/videos_failed": 1.0,
        }
        ok_metrics = {
            "visual_validation/videos_ok": 4.0,
            "visual_validation/videos_failed": 0.0,
        }

        self.assertTrue(
            sonic_train.accepted_visual_metrics_failed(
                failed_metrics,
                {"acceptance_backend": True},
            )
        )
        self.assertFalse(
            sonic_train.accepted_visual_metrics_failed(
                failed_metrics,
                {"acceptance_backend": False},
            )
        )
        self.assertFalse(
            sonic_train.accepted_visual_metrics_failed(
                ok_metrics,
                {"acceptance_backend": True},
            )
        )

    def test_temporal_consistency_loss_weight_is_config_gated(self):
        self.assertEqual(
            sonic_train.temporal_consistency_loss_weight({"training": {}}),
            0.0,
        )
        self.assertEqual(
            sonic_train.temporal_consistency_loss_weight(
                {"training": {"temporal_consistency_loss_enabled": True}}
            ),
            0.01,
        )
        self.assertEqual(
            sonic_train.temporal_consistency_loss_weight(
                {
                    "training": {
                        "temporal_consistency_loss_enabled": True,
                        "temporal_consistency_loss_weight": 0.02,
                    }
                }
            ),
            0.02,
        )

    def test_loss_and_metrics_adds_temporal_command_delta_mse_when_enabled(self):
        torch = sonic_train.torch
        # Two future frames, one joint position + one joint velocity per frame.
        pred_command = torch.tensor([[0.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
        target_command = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        pred_anchor = torch.zeros(1, 18, dtype=torch.float32)
        target_anchor = torch.zeros(1, 18, dtype=torch.float32)
        pred = torch.cat([pred_command, pred_anchor], dim=-1)
        target = torch.cat([target_command, target_anchor], dim=-1)
        stats = {
            "target_mean": torch.zeros(pred.shape[-1], dtype=torch.float32),
            "target_std": torch.ones(pred.shape[-1], dtype=torch.float32),
        }

        loss, metrics = sonic_train.loss_and_metrics(
            pred,
            target,
            target,
            stats,
            4,
            1,
            18,
            True,
            1.0,
            0.25,
            0.5,
            0.01,
        )

        self.assertAlmostEqual(float(metrics["command_mse_norm"]), 0.25)
        self.assertAlmostEqual(float(metrics["temporal_consistency_mse_norm"]), 0.5)
        self.assertAlmostEqual(float(metrics["temporal_consistency_loss_weight"]), 0.01)
        self.assertAlmostEqual(float(loss), 0.255)
