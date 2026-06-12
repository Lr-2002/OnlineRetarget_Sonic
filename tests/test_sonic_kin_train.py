import copy
import json
import tempfile
import unittest
from pathlib import Path

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
    def test_shared_evaluation_cohort_ignores_variant_name_and_run_group(self):
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
        treatment = sonic_train.build_evaluation_cohort(rows, base_config, run_group="treatment_group")
        baseline_config = {
            **base_config,
            "variant": {"name": "loss_off_baseline_variant"},
            "wandb": {"name": "baseline_run"},
        }
        baseline = sonic_train.build_evaluation_cohort(rows, baseline_config, run_group="baseline_group")

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
        self.assertIs(treatment["include_run_group"], False)
        self.assertEqual(treatment["run_group"], "")
        self.assertEqual(treatment["salt_sha256"], baseline["salt_sha256"])

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
        self.assertIs(manifest["include_run_group"], False)
        self.assertEqual(manifest["run_group"], "")
        self.assertNotIn("run_group", manifest["sampling"]["salt_fields"])
        self.assertIn("variant.name", manifest["sampling"]["excluded_config_fields"])
        self.assertIn("wandb.name", manifest["sampling"]["excluded_config_fields"])

    def test_non_cohort_visual_selection_preserves_legacy_variant_seed_salt(self):
        rows = [
            {
                "relative_path": f"legacy_{index:03d}.pkl",
                "filename": f"legacy_{index:03d}",
                "frame_count": 60 + index,
                "split": "validation",
            }
            for index in range(40)
        ]
        config = {
            "training": {"seed": 77},
            "variant": {"name": "legacy_variant_a"},
            "visual_validation": {"num_videos": 10},
        }

        selected, cohort_summary = sonic_train.select_visual_validation_rows(rows, config, run_group="ignored")
        legacy = sonic_train._select_visual_rows(
            rows,
            count=10,
            salt="legacy_variant_a:77",
        )

        self.assertEqual(cohort_summary, {})
        self.assertEqual(
            [sonic_train.evaluation_row_stable_key(row) for row in selected],
            [sonic_train.evaluation_row_stable_key(row) for row in legacy],
        )

        disabled_config = {
            **config,
            "evaluation_cohort": {
                "enabled": False,
                "id": "should_not_affect_legacy_selection",
                "seed": 20260608,
                "visual_num_samples": 8,
                "metric_num_samples": 100,
            },
        }
        disabled_selected, disabled_summary = sonic_train.select_visual_validation_rows(
            rows,
            disabled_config,
            run_group="ignored",
        )

        self.assertEqual(disabled_summary, {})
        self.assertEqual(
            [sonic_train.evaluation_row_stable_key(row) for row in disabled_selected],
            [sonic_train.evaluation_row_stable_key(row) for row in legacy],
        )

        variant_b = {
            **config,
            "variant": {"name": "legacy_variant_b"},
        }
        variant_b_selected, _ = sonic_train.select_visual_validation_rows(rows, variant_b, run_group="ignored")
        self.assertNotEqual(
            [sonic_train.evaluation_row_stable_key(row) for row in selected],
            [sonic_train.evaluation_row_stable_key(row) for row in variant_b_selected],
        )

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

    def test_soma_motionlib_previous_g1_action_condition_uses_anchor_minus_one(self):
        frames = 60
        soma_joints = np.zeros((frames, 26, 3), dtype=np.float32)
        identity = np.zeros((frames, 4), dtype=np.float32)
        identity[:, 0] = 1.0
        dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
        arrays = {
            "soma_joints": soma_joints,
            "soma_root_quat": identity.copy(),
            "joint_pos": dof,
            "joint_vel": sonic_train.finite_difference_velocity(dof, 50.0),
            "root_pos": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": identity.copy(),
        }
        config = {
            "features": {
                "include_root_pos_target": True,
                "previous_g1_action_condition": True,
            }
        }

        motion, skeleton, target = sonic_train.build_soma_motionlib_features(
            arrays,
            np.asarray([0, 3], dtype=np.int64),
            window=10,
            step=5,
            config=config,
        )

        self.assertEqual(motion.shape, (2, 869))
        self.assertEqual(skeleton.shape, (2, 104))
        self.assertEqual(target.shape, (2, 670))
        np.testing.assert_allclose(motion[:, -29:], dof[[0, 2]])
        command = target[:, : 10 * (29 + 29)].reshape(2, 10, 58)
        expected_target_indices = np.asarray(
            [
                np.arange(0, 50, 5, dtype=np.int64),
                np.arange(3, 53, 5, dtype=np.int64),
            ]
        )
        np.testing.assert_allclose(command[..., :29], dof[expected_target_indices])

    def test_soma_motionlib_previous_root_roll_pitch_uses_anchor_minus_one_and_excludes_yaw(self):
        frames = 60
        soma_joints = np.zeros((frames, 26, 3), dtype=np.float32)
        soma_identity = np.zeros((frames, 4), dtype=np.float32)
        soma_identity[:, 0] = 1.0
        dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
        root_euler = np.zeros((frames, 3), dtype=np.float32)
        root_euler[0] = [0.1, -0.2, 0.9]
        root_euler[2] = [0.35, 0.15, -1.2]
        root_rot = np.asarray(
            [sonic_train._euler_xyz_to_quat_wxyz(row) for row in root_euler],
            dtype=np.float32,
        )
        yaw_only = np.asarray(
            [sonic_train._euler_xyz_to_quat_wxyz([0.0, 0.0, 1.3])],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            sonic_train.quat_to_roll_pitch(yaw_only),
            np.zeros((1, 2), dtype=np.float32),
            atol=1e-6,
        )
        arrays = {
            "soma_joints": soma_joints,
            "soma_root_quat": soma_identity.copy(),
            "joint_pos": dof,
            "joint_vel": sonic_train.finite_difference_velocity(dof, 50.0),
            "root_pos": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": root_rot,
        }
        config = {
            "features": {
                "include_root_pos_target": True,
                "previous_g1_action_condition": True,
                "previous_g1_root_roll_pitch_condition": True,
            }
        }

        motion, skeleton, target = sonic_train.build_soma_motionlib_features(
            arrays,
            np.asarray([0, 3], dtype=np.int64),
            window=10,
            step=5,
            config=config,
        )

        self.assertEqual(motion.shape, (2, 871))
        self.assertEqual(skeleton.shape, (2, 104))
        self.assertEqual(target.shape, (2, 670))
        np.testing.assert_allclose(motion[:, -31:-2], dof[[0, 2]])
        np.testing.assert_allclose(
            motion[:, -2:],
            sonic_train.quat_to_roll_pitch(root_rot[[0, 2]]),
            atol=1e-6,
        )

    def test_previous_root_roll_pitch_expected_dims_guard_rejects_stale_action_only_stats(self):
        config = {
            "features": {
                "previous_g1_action_condition": True,
                "previous_g1_root_roll_pitch_condition": True,
                "expected_dims": {
                    "motion_token": 871,
                    "x_skel": 104,
                    "z_skel": 104,
                    "model_input": 975,
                    "target": 670,
                },
            }
        }

        sonic_train.assert_expected_feature_dims(
            config,
            motion_dim=871,
            skeleton_dim=104,
            target_dim=670,
            skeleton_feature_lookup=None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "motion_token: expected 871, got 869; model_input: expected 975, got 973",
        ):
            sonic_train.assert_expected_feature_dims(
                config,
                motion_dim=869,
                skeleton_dim=104,
                target_dim=670,
                skeleton_feature_lookup=None,
            )

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

    def test_supervised_resume_checkpoint_restores_model_optimizer_and_step(self):
        torch = sonic_train.torch
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            model = sonic_train.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            saved_model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
            sonic_train.save_checkpoint(
                output_dir,
                model,
                optimizer,
                step=1234,
                metrics={"loss": 0.5},
                keep_last=2,
            )

            resumed_model = sonic_train.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01, weight_decay=0.1)
            resume_state = sonic_train.load_supervised_resume_checkpoint(
                output_dir / "checkpoints" / "latest.pt",
                resumed_model,
                resumed_optimizer,
                torch.device("cpu"),
            )

            self.assertEqual(resume_state["step"], 1234)
            self.assertEqual(resume_state["metrics_keys"], ["loss"])
            for key, expected in saved_model_state.items():
                torch.testing.assert_close(resumed_model.state_dict()[key], expected)
            self.assertEqual(
                resumed_optimizer.state_dict()["param_groups"],
                saved_optimizer_state["param_groups"],
            )
            for loaded_state, expected_state in zip(
                resumed_optimizer.state_dict()["state"].values(),
                saved_optimizer_state["state"].values(),
            ):
                self.assertEqual(loaded_state.keys(), expected_state.keys())
                for key, expected in expected_state.items():
                    loaded = loaded_state[key]
                    if torch.is_tensor(expected):
                        torch.testing.assert_close(loaded, expected)
                    else:
                        self.assertEqual(loaded, expected)

    def test_rows_from_index_cache_payload_records_data_package_identity(self):
        config = {
            "input_data": {
                "format": "soma_motionlib",
                "robot_motion_dir": "/robot",
                "soma_motion_dir": "/soma",
                "source_fps": 120.0,
                "target_fps": 50.0,
                "max_clips": 10,
                "max_duration_delta_sec": 0.05,
                "data_package": {
                    "spec": "kin",
                    "category": "walk",
                    "indicator": "/packages/kin/walk.txt",
                    "missing_policy": "error",
                    "expected_row_count": 2,
                    "package_rows_sha256": "a" * 64,
                },
            }
        }
        rows = [{"relative_path": "clip.pkl"}]

        payload = sonic_train.rows_from_index_cache_payload(config, Path("/data_root"), rows, skipped=3)

        self.assertEqual(payload["cache_version"], 2)
        self.assertEqual(payload["config"]["source_fps"], 120.0)
        self.assertEqual(payload["config"]["target_fps"], 50.0)
        self.assertEqual(payload["config"]["max_clips"], 10)
        self.assertEqual(payload["config"]["max_duration_delta_sec"], 0.05)
        self.assertEqual(payload["config"]["data_package"]["spec"], "kin")
        self.assertEqual(payload["config"]["data_package"]["package_rows_sha256"], "a" * 64)
        self.assertEqual(
            payload["config_sha256"],
            sonic_train.rows_from_index_cache_signature(payload["config"]),
        )

    def test_rows_from_index_cache_rejects_row_affecting_input_changes(self):
        base_input = {
            "format": "soma_motionlib",
            "robot_motion_dir": "/robot",
            "soma_motion_dir": "/soma",
            "source_fps": 120.0,
            "target_fps": 50.0,
            "max_clips": 10,
            "max_duration_delta_sec": 0.05,
            "data_package": {
                "spec": "kin",
                "category": "walk",
                "indicator": "/packages/kin/walk.txt",
                "missing_policy": "error",
                "expected_row_count": 2,
                "package_rows_sha256": "a" * 64,
            },
        }
        base_config = {"input_data": base_input}
        rows = [{"relative_path": "clip.pkl"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "rows_from_index_cache.json"
            payload = sonic_train.rows_from_index_cache_payload(
                base_config,
                Path("/data_root"),
                rows,
                skipped=0,
            )
            sonic_train.write_rows_from_index_cache(cache_path, payload)

            mutations = {
                "source_fps": {"source_fps": 100.0},
                "target_fps": {"target_fps": 60.0},
                "max_clips": {"max_clips": 5},
                "max_duration_delta_sec": {"max_duration_delta_sec": 0.01},
                "data_package_digest": {
                    "data_package": {
                        **base_input["data_package"],
                        "package_rows_sha256": "b" * 64,
                    }
                },
            }
            for field, update in mutations.items():
                mutated_input = copy.deepcopy(base_input)
                mutated_input.update(update)
                expected_config = sonic_train.rows_from_index_cache_config(
                    {"input_data": mutated_input},
                    Path("/data_root"),
                )
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "cache config mismatch"):
                        sonic_train.read_rows_from_index_cache(
                            cache_path,
                            expected_config=expected_config,
                        )

    def test_rows_from_index_ignores_stale_data_package_cache_before_barrier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cache_path = sonic_train.rows_from_index_cache_path(output_dir)
            old_config = {
                "input_data": {
                    "format": "soma_motionlib",
                    "robot_motion_dir": "/old_robot",
                    "soma_motion_dir": "/soma",
                    "max_clips": 10,
                    "max_duration_delta_sec": 0.05,
                    "data_package": {
                        "spec": "kin",
                        "category": "walk",
                        "indicator": "/packages/old_walk.txt",
                        "missing_policy": "error",
                    },
                }
            }
            old_payload = sonic_train.rows_from_index_cache_payload(
                old_config,
                Path("/data_root"),
                [{"relative_path": "old.pkl"}],
                skipped=0,
            )
            sonic_train.write_rows_from_index_cache(cache_path, old_payload)

            config = {
                "input_data": {
                    "format": "soma_motionlib",
                    "robot_motion_dir": "/new_robot",
                    "soma_motion_dir": "/soma",
                    "max_clips": 10,
                    "max_duration_delta_sec": 0.05,
                    "data_package": {
                        "spec": "kin",
                        "category": "walk",
                        "indicator": "/packages/new_walk.txt",
                        "missing_policy": "error",
                    },
                }
            }
            runtime = {"distributed": True, "rank": 0, "world_size": 4}
            rebuilt_rows = [{"relative_path": "new.pkl", "split": "train"}]
            calls = {"rebuilt": 0, "barrier": 0}
            original_rows_from_pair = sonic_train.rows_from_soma_motionlib_pair
            original_filter = sonic_train.filter_rows_by_data_package_config
            original_barrier = sonic_train.distributed_barrier
            try:
                sonic_train.rows_from_soma_motionlib_pair = lambda *_args, **_kwargs: (
                    calls.__setitem__("rebuilt", calls["rebuilt"] + 1) or rebuilt_rows,
                    7,
                )
                sonic_train.filter_rows_by_data_package_config = lambda rows, _input_data: (
                    [dict(row) for row in rows],
                    {"selected_row_count": len(rows)},
                )
                sonic_train.distributed_barrier = lambda _runtime: calls.__setitem__(
                    "barrier",
                    calls["barrier"] + 1,
                )

                rows, skipped, summary = sonic_train.rows_from_index(
                    config,
                    Path("/data_root"),
                    output_dir=output_dir,
                    runtime=runtime,
                    return_package_summary=True,
                )
            finally:
                sonic_train.rows_from_soma_motionlib_pair = original_rows_from_pair
                sonic_train.filter_rows_by_data_package_config = original_filter
                sonic_train.distributed_barrier = original_barrier

            self.assertEqual(rows, rebuilt_rows)
            self.assertEqual(skipped, 7)
            self.assertEqual(summary, {"selected_row_count": 1})
            self.assertEqual(calls, {"rebuilt": 1, "barrier": 1})
            current_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(current_payload["config"]["robot_motion_dir"], "/new_robot")
            self.assertEqual(current_payload["rows"], rebuilt_rows)

    def test_rows_from_index_reuses_valid_current_cache_on_rank0_and_non_main_once(self):
        config = {
            "input_data": {
                "format": "soma_motionlib",
                "robot_motion_dir": "/robot",
                "soma_motion_dir": "/soma",
                "source_fps": 120.0,
                "target_fps": 50.0,
                "max_clips": 10,
                "max_duration_delta_sec": 0.05,
            }
        }
        cached_rows = [{"relative_path": "current.pkl", "split": "train"}]

        for rank in (0, 2):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                cache_path = sonic_train.rows_from_index_cache_path(output_dir)
                payload = sonic_train.rows_from_index_cache_payload(
                    config,
                    Path("/data_root"),
                    cached_rows,
                    skipped=4,
                )
                sonic_train.write_rows_from_index_cache(cache_path, payload)

                runtime = {"distributed": True, "rank": rank, "world_size": 4}
                calls = {"rebuilt": 0, "barrier": 0}
                original_rows_from_pair = sonic_train.rows_from_soma_motionlib_pair
                original_barrier = sonic_train.distributed_barrier
                try:
                    sonic_train.rows_from_soma_motionlib_pair = lambda *_args, **_kwargs: (
                        calls.__setitem__("rebuilt", calls["rebuilt"] + 1) or [{"relative_path": "rebuilt.pkl"}],
                        0,
                    )
                    sonic_train.distributed_barrier = lambda _runtime: calls.__setitem__(
                        "barrier",
                        calls["barrier"] + 1,
                    )

                    rows, skipped, summary = sonic_train.rows_from_index(
                        config,
                        Path("/data_root"),
                        output_dir=output_dir,
                        runtime=runtime,
                        return_package_summary=True,
                    )
                finally:
                    sonic_train.rows_from_soma_motionlib_pair = original_rows_from_pair
                    sonic_train.distributed_barrier = original_barrier

                self.assertEqual(rows, cached_rows)
                self.assertEqual(skipped, 4)
                self.assertIsNone(summary)
                self.assertEqual(calls, {"rebuilt": 0, "barrier": 1})
                current_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                self.assertEqual(current_payload["rows"], cached_rows)

    def test_wait_for_rows_from_index_cache_rejects_wrong_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "rows_from_index_cache.json"
            old_config = {"format": "soma_motionlib", "data_package": None}
            old_payload = {
                "cache_version": 2,
                "created_at": "2026-06-09T00:00:00+00:00",
                "config": old_config,
                "config_sha256": sonic_train.rows_from_index_cache_signature(old_config),
                "rows": [{"relative_path": "old.pkl"}],
                "row_count": 1,
                "skipped_count": 0,
            }
            cache_path.write_text(json.dumps(old_payload), encoding="utf-8")

            with self.assertRaisesRegex(TimeoutError, "cache config mismatch"):
                sonic_train.wait_for_rows_from_index_cache(
                    cache_path,
                    expected_config={"format": "soma_motionlib", "data_package": {"spec": "kin"}},
                    timeout_sec=0.0,
                    poll_sec=0.001,
                )

    def test_ab_overlap_loss_weight_is_config_gated(self):
        self.assertEqual(
            sonic_train.command_ab_overlap_loss_weight({"training": {}}),
            0.0,
        )
        self.assertEqual(
            sonic_train.command_ab_overlap_loss_weight(
                {"training": {"ab_overlap_loss_enabled": True}}
            ),
            0.01,
        )
        self.assertEqual(
            sonic_train.command_ab_overlap_loss_weight(
                {
                    "training": {
                        "ab_overlap_loss_enabled": True,
                        "ab_overlap_loss_weight": 0.03,
                    }
                }
            ),
            0.03,
        )

    def test_ab_overlap_batch_offset_uses_future_step_over_frame_stride(self):
        config = {
            "features": {"future_step": 5},
            "training": {"frame_stride": 1},
        }
        self.assertEqual(sonic_train.command_ab_overlap_batch_offset(config), 5)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            sonic_train.command_ab_overlap_batch_offset(
                {"features": {"future_step": 5}, "training": {"frame_stride": 2}}
            )

    def test_command_ab_overlap_loss_detects_between_horizon_error_when_temporal_deltas_match(self):
        torch = sonic_train.torch
        # Each row has perfect within-window delta agreement, but sample B horizon 0
        # disagrees with sample A horizon 1.
        pred_frames = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[100.0, 0.0], [101.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        target_frames = torch.tensor(
            [
                [[10.0, 0.0], [11.0, 0.0]],
                [[20.0, 0.0], [21.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        pred_command = pred_frames.reshape(2, -1)
        target_command = target_frames.reshape(2, -1)

        temporal = sonic_train.command_temporal_consistency_loss(
            pred_command,
            target_command,
            joint_dim=1,
        )
        overlap = sonic_train.command_ab_overlap_loss(
            pred_command,
            joint_dim=1,
            batch_offset=1,
            overlap_horizon_frames=1,
        )

        self.assertAlmostEqual(float(temporal), 0.0)
        self.assertAlmostEqual(float(overlap), 4900.5)

    def test_loss_and_metrics_adds_ab_overlap_mse_when_enabled(self):
        torch = sonic_train.torch
        pred_command = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [100.0, 0.0, 101.0, 0.0],
            ],
            dtype=torch.float32,
        )
        target_command = pred_command.clone()
        pred_anchor = torch.zeros(2, 18, dtype=torch.float32)
        target_anchor = torch.zeros(2, 18, dtype=torch.float32)
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
            0.0,
            0.01,
            1,
            1,
        )

        self.assertAlmostEqual(float(metrics["command_mse_norm"]), 0.0)
        self.assertAlmostEqual(float(metrics["ab_overlap_mse_norm"]), 4900.5)
        self.assertAlmostEqual(float(metrics["ab_overlap_loss_weight"]), 0.01)
        self.assertAlmostEqual(float(loss), 49.005, delta=1e-5)

    def test_training_batch_selection_preserves_order_when_ab_overlap_is_active(self):
        torch = sonic_train.torch
        motion = torch.arange(10, dtype=torch.float32).reshape(10, 1)
        skeleton = motion + 100.0
        target = motion + 200.0
        rng = torch.Generator(device=motion.device)
        rng.manual_seed(1234)

        motion_sel, skeleton_sel, target_sel = sonic_train.select_training_batch_frames(
            motion,
            skeleton,
            target,
            4,
            rng,
            preserve_order=True,
        )

        self.assertEqual(motion_sel.shape[0], 4)
        torch.testing.assert_close(motion_sel[1:] - motion_sel[:-1], torch.ones(3, 1))
        torch.testing.assert_close(skeleton_sel, motion_sel + 100.0)
        torch.testing.assert_close(target_sel, motion_sel + 200.0)

    def test_kin_window_dataset_exposes_same_clip_ab_horizon_pairs(self):
        torch = sonic_train.torch
        frames = 8
        joint_pos = np.arange(frames, dtype=np.float32).reshape(frames, 1)
        joint_vel = (np.arange(frames, dtype=np.float32) + 100.0).reshape(frames, 1)
        body_pos = np.zeros((frames, 1, 3), dtype=np.float32)
        body_quat = np.zeros((frames, 1, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        arrays = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
        }
        config = {
            "input_data": {"data_root": "/tmp/unused"},
            "features": {"future_window_frames": 3, "future_step": 2},
            "training": {"frame_stride": 1, "loader_chunk_frames": 8},
        }
        rows = [{"relative_path": "clip.npz", "frame_count": frames, "split": "train"}]
        original_load_arrays = sonic_train.load_arrays
        sonic_train.load_arrays = lambda _path: arrays
        try:
            dataset = sonic_train.KinWindowDataset(rows, "train", config)
            _motion, _skeleton, target = dataset[0]
        finally:
            sonic_train.load_arrays = original_load_arrays

        command = target[:, : 3 * 2].reshape(frames, 3, 2)
        offset = sonic_train.command_ab_overlap_batch_offset(config)
        self.assertEqual(offset, 2)
        torch.testing.assert_close(command[0, 1], command[offset, 0])
        torch.testing.assert_close(command[1, 1], command[1 + offset, 0])
