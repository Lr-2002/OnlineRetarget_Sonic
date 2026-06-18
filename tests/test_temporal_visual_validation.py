import json
import tempfile
import unittest
from pathlib import Path

from online_retarget.temporal_visual_validation import (
    TemporalVisualValidationConfig,
    _future_source_indices,
    run_temporal_native_fps_visual_validation,
)


class TemporalVisualValidationTests(unittest.TestCase):
    def test_future_source_indices_align_target_time_to_source_fps(self):
        indices = _future_source_indices(
            target_index=10,
            horizon=3,
            future_step=5,
            source_fps=25.0,
            target_fps=50.0,
        )

        self.assertEqual(indices, [5, 7, 10])

    def test_run_temporal_native_fps_visual_validation_disabled_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"

            summary = run_temporal_native_fps_visual_validation(
                torch=None,
                model=None,
                config={},
                visual_validation=TemporalVisualValidationConfig(enabled=False),
                samples=[],
                output_dir=output_dir,
                step=20,
                device="cpu",
                wandb_run=None,
            )

            summary_path = (
                output_dir
                / "online_retarget_visual_validation"
                / "step_00000020"
                / "summary.json"
            )
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["status"], "disabled")
        self.assertEqual(persisted["status"], "disabled")

    def test_run_temporal_native_fps_visual_validation_blocks_without_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"

            summary = run_temporal_native_fps_visual_validation(
                torch=None,
                model=None,
                config={},
                visual_validation=TemporalVisualValidationConfig(enabled=True, num_videos=1),
                samples=[],
                output_dir=output_dir,
                step=20,
                device="cpu",
                wandb_run=None,
            )

            summary_path = (
                output_dir
                / "online_retarget_visual_validation"
                / "step_00000020"
                / "summary.json"
            )
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["status"], "blocked")
        self.assertIn("provided no samples", summary["message"])
        self.assertEqual(persisted["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
