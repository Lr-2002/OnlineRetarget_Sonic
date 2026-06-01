from pathlib import Path
import tempfile
import unittest

from online_retarget.training_time_logging import (
    a0_metric_registry_results,
    apply_remote_logging_visual_overrides,
    build_remote_logging_contract,
    build_remote_logging_probe_payload,
    remote_logging_settings,
    visual_wandb_artifact_specs,
    wandb_registry_payload,
)


class TrainingTimeLoggingTests(unittest.TestCase):
    def _config(self):
        return {
            "variant": {"name": "a0_fixture"},
            "visual_validation": {
                "enabled": True,
                "every_steps": 20000,
                "every_minutes": 60,
                "num_videos": 8,
                "duration_sec": 4.0,
                "wandb_upload": True,
            },
            "remote_logging": {
                "enabled": True,
                "log_scalars": True,
                "visual_upload": True,
                "wandb_artifact_prefix": "lr177-a0-visual-validation",
            },
            "wandb": {"enabled": True, "mode": "offline"},
        }

    def test_contract_honors_cadence_and_non_invasive_flags(self):
        contract = build_remote_logging_contract(
            self._config(),
            output_dir="/tmp/out",
            run_group="probe",
            config_path="configs/a0.json",
        )

        self.assertEqual(contract["schema_version"], "online_retarget.remote_training_logging.v1")
        self.assertFalse(contract["non_invasive"]["changes_training_loss"])
        self.assertFalse(contract["non_invasive"]["changes_ddp_collectives"])
        self.assertEqual(contract["visuals"]["visual_every_n_steps"], 20000)
        self.assertEqual(contract["visuals"]["num_visual_samples"], 8)
        self.assertEqual(contract["visuals"]["max_video_sec"], 4.0)
        self.assertTrue(contract["visuals"]["wandb_upload_enabled"])
        self.assertIn("validation/g1_joint_pos_rmse_rad", contract["scalars"]["stable_metric_keys"])

    def test_registry_results_separate_measured_missing_and_not_applicable(self):
        results = a0_metric_registry_results(
            {"validation/g1_joint_pos_rmse_rad": 0.125},
            source="validation",
            step=200,
        )
        by_metric = {result.metric_id: result for result in results}

        self.assertEqual(by_metric["g1_joint_pos_rmse_rad"].status, "measured")
        self.assertEqual(by_metric["g1_joint_pos_rmse_rad"].value, 0.125)
        self.assertEqual(by_metric["body_position_mpjpe"].status, "missing")
        self.assertIsNone(by_metric["body_position_mpjpe"].value)
        self.assertEqual(by_metric["policy_success"].status, "not_applicable")
        self.assertIsNone(by_metric["policy_success"].value)

    def test_wandb_payload_uses_registry_ids_and_status_counts(self):
        results = a0_metric_registry_results(
            {"g1_joint_pos_rmse_rad": 0.2},
            source="train",
            step=1,
        )
        payload = wandb_registry_payload(results)

        self.assertEqual(payload["metric_registry/train/g1_joint_pos_rmse_rad"], 0.2)
        self.assertEqual(payload["metric_registry/status_count/measured"], 1.0)
        self.assertEqual(payload["metric_registry/status_count/missing"], 1.0)
        self.assertEqual(payload["metric_registry/status_count/not_applicable"], 1.0)

    def test_dry_run_probe_payload_uses_real_metric_results(self):
        results = a0_metric_registry_results(
            {"validation/g1_joint_pos_rmse_rad": 0.123},
            source="validation",
            step=0,
            sequence_id="dry_run_validation",
        )

        payload = build_remote_logging_probe_payload(self._config(), results)

        self.assertEqual(payload["remote_logging/probe"], 1.0)
        self.assertEqual(payload["metric_registry/validation/g1_joint_pos_rmse_rad"], 0.123)
        self.assertEqual(payload["metric_registry/status_count/measured"], 1.0)
        self.assertEqual(payload["metric_registry/status_count/missing"], 1.0)
        self.assertEqual(payload["metric_registry/status_count/not_applicable"], 1.0)

    def test_remote_logging_visual_overrides_drive_visual_validation_runtime_config(self):
        config = self._config()
        config["visual_validation"].update(
            {
                "every_steps": 50000,
                "every_minutes": 90,
                "num_videos": 2,
                "duration_sec": 1.0,
                "wandb_upload": False,
            }
        )
        config["remote_logging"].update(
            {
                "visual_upload": True,
                "visual_every_n_steps": 1234,
                "visual_every_minutes": 12.5,
                "num_visual_samples": 5,
                "max_video_sec": 2.5,
            }
        )

        applied = apply_remote_logging_visual_overrides(config)

        self.assertEqual(applied["visual_validation"]["every_steps"], 1234)
        self.assertEqual(applied["visual_validation"]["every_minutes"], 12.5)
        self.assertEqual(applied["visual_validation"]["num_videos"], 5)
        self.assertEqual(applied["visual_validation"]["duration_sec"], 2.5)
        self.assertTrue(applied["visual_validation"]["wandb_upload"])
        self.assertEqual(remote_logging_settings(applied)["visual_every_n_steps"], 1234)
        self.assertEqual(remote_logging_settings(applied)["num_visual_samples"], 5)

    def test_visual_specs_include_metadata_and_label_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            metadata = root / "metadata.json"
            video.write_bytes(b"fake")
            metadata.write_text("{}", encoding="utf-8")

            specs = visual_wandb_artifact_specs(
                [
                    {
                        "index": 0,
                        "combined_status": "ok",
                        "combined_video": str(video),
                        "metadata": str(metadata),
                        "filename": "clip",
                        "relative_path": "soma/clip.pkl",
                        "fps": 50,
                        "frames": 200,
                        "active_backend_is_acceptance_backend": False,
                    }
                ],
                config=self._config(),
                manifest={
                    "run_id": "run",
                    "run_group": "group",
                    "control_revision_actual": "abc",
                    "source_revision_actual": "def",
                },
                step=20000,
                checkpoint_path=root / "latest.pt",
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["artifact_type"], "online_retarget_visual_validation_video")
        self.assertTrue(specs[0]["upload_enabled"])
        self.assertEqual(
            specs[0]["metadata"]["render_acceptance_state"],
            "fallback_not_final_somamesh",
        )
        self.assertFalse(specs[0]["metadata"]["soma_mesh_final_render"])
        self.assertIn("step-00020000", specs[0]["artifact_name"])

    def test_disabled_wandb_mode_disables_visual_upload_without_disabling_contract(self):
        config = self._config()
        config["wandb"]["mode"] = "disabled"

        settings = remote_logging_settings(config)

        self.assertTrue(settings["enabled"])
        self.assertFalse(settings["visual_wandb_upload_enabled"])


if __name__ == "__main__":
    unittest.main()
