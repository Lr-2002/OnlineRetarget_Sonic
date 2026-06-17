import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import upload_accepted_vertical_v2_wandb as uploader


class UploadAcceptedVerticalV2WandbTests(unittest.TestCase):
    def test_upload_records_discovers_accepted_clip_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_step(root, step=990000, sample_id="walk")

            records = uploader.upload_records(root, steps={990000})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], 990000)
        self.assertEqual(records[0]["sample_id"], "walk")
        self.assertEqual(records[0]["combined_video"], str(paths["combined_video"]))
        self.assertEqual(records[0]["row1_soma_somamesh_video"], str(paths["row1"]))
        self.assertEqual(records[0]["row2_g1_target_isaaclab_video"], str(paths["row2"]))
        self.assertEqual(records[0]["row3_g1_kinematics_isaaclab_video"], str(paths["row3"]))

    def test_upload_records_backfills_row_media_from_metadata_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_step(
                root,
                step=1000000,
                sample_id="walk",
                omit_clip_row_fields=True,
            )

            records = uploader.upload_records(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["combined_video"], str(paths["combined_video"]))
        self.assertEqual(records[0]["row1_soma_somamesh_video"], str(paths["row1"]))
        self.assertEqual(records[0]["row2_g1_target_isaaclab_video"], str(paths["row2"]))
        self.assertEqual(records[0]["row3_g1_kinematics_isaaclab_video"], str(paths["row3"]))

    def test_upload_to_wandb_uses_periodic_accepted_vertical_v2_media_keys(self):
        logged = []
        saved = []
        api_paths = []
        init_kwargs = {}

        class FakeVideo:
            def __init__(self, path):
                self.path = path

        class FakeApiRun:
            id = "4gosirw6"
            lastHistoryStep = 1000003

        class FakeApi:
            def run(self, path):
                api_paths.append(path)
                return FakeApiRun()

        class FakeRun:
            id = "4gosirw6"

            def save(self, path):
                saved.append(path)

            def log(self, payload, step=None):
                logged.append((payload, step))

            def finish(self):
                return None

        def fake_init(**kwargs):
            init_kwargs.update(kwargs)
            return FakeRun()

        fake_wandb = types.SimpleNamespace(
            Video=FakeVideo,
            Api=FakeApi,
            init=fake_init,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_step(root, step=1000000, sample_id="walk")
            records = uploader.upload_records(root)
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                uploaded = uploader.upload_to_wandb(
                    records,
                    entity="world_model_xh",
                    project="OnlineRetarget_DP",
                    resume_run_id="4gosirw6",
                    run_name="ignored",
                )

        self.assertEqual(uploaded, 1)
        payload, step = logged[0]
        self.assertEqual(step, 1000004)
        self.assertNotEqual(step, 1000000)
        self.assertEqual(api_paths, ["world_model_xh/OnlineRetarget_DP/4gosirw6"])
        self.assertEqual(init_kwargs["entity"], "world_model_xh")
        self.assertEqual(init_kwargs["project"], "OnlineRetarget_DP")
        self.assertEqual(init_kwargs["id"], "4gosirw6")
        self.assertEqual(init_kwargs["resume"], "must")
        self.assertIsNone(init_kwargs["name"])
        self.assertEqual(
            payload["periodic_eval/visualization/accepted_vertical_v2/backfill_step"],
            1000000,
        )
        self.assertEqual(
            payload["periodic_eval/visualization/accepted_vertical_v2/backfill_source_step"],
            1000000,
        )
        self.assertEqual(
            payload["periodic_eval/visualization/accepted_vertical_v2/backfill_wandb_history_step"],
            1000004,
        )
        for key in (
            "periodic_eval/visualization/accepted_vertical_v2/primary",
            "periodic_eval/visualization/accepted_vertical_v2/row1_soma_somamesh",
            "periodic_eval/visualization/accepted_vertical_v2/row2_g1_target",
            "periodic_eval/visualization/accepted_vertical_v2/row3_g1_kinematics",
        ):
            self.assertIsInstance(payload[key], FakeVideo)
        self.assertTrue(any(path.endswith("__vertical_somamesh_g1target_g1kinematics.mp4") for path in saved))
        self.assertTrue(any(path.endswith("__row1_soma_somamesh.mp4") for path in saved))

    def test_upload_to_wandb_resume_requires_entity(self):
        with self.assertRaisesRegex(ValueError, "--entity is required"):
            uploader.upload_to_wandb(
                [],
                entity=None,
                project="OnlineRetarget_DP",
                resume_run_id="4gosirw6",
                run_name="ignored",
            )

    def test_upload_to_wandb_resume_rejects_wrong_target_run(self):
        class FakeApiRun:
            id = "wrong-run"
            lastHistoryStep = 7

        class FakeApi:
            def run(self, _path):
                return FakeApiRun()

        fake_wandb = types.SimpleNamespace(
            Api=FakeApi,
            init=mock.Mock(),
        )
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            with self.assertRaisesRegex(RuntimeError, "expected '4gosirw6'"):
                uploader.upload_to_wandb(
                    [],
                    entity="world_model_xh",
                    project="OnlineRetarget_DP",
                    resume_run_id="4gosirw6",
                    run_name="ignored",
                )
        fake_wandb.init.assert_not_called()


def _write_step(
    root: Path,
    *,
    step: int,
    sample_id: str,
    omit_clip_row_fields: bool = False,
) -> dict[str, Path]:
    step_id = f"step_{step:08d}"
    accepted_root = (
        root
        / "periodic_eval"
        / step_id
        / "visualization"
        / "accepted_vertical_v2"
    )
    clip_dir = accepted_root / step_id / "accepted_vertical_v2"
    clip_dir.mkdir(parents=True)
    stem = f"{sample_id}__{step_id}"
    paths = {
        "summary": accepted_root / "lr310_dp_visual_validation_summary.json",
        "metadata": clip_dir / f"{stem}__vertical_somamesh_g1target_g1kinematics.json",
        "combined_video": clip_dir / f"{stem}__vertical_somamesh_g1target_g1kinematics.mp4",
        "row1": clip_dir / f"{stem}__row1_soma_somamesh.mp4",
        "row2": clip_dir / f"{stem}__row2_g1_target_isaaclab.mp4",
        "row3": clip_dir / f"{stem}__row3_g1_kinematics_isaaclab.mp4",
    }
    for key, path in paths.items():
        if key in {"summary", "metadata"}:
            continue
        path.write_bytes(b"mp4")
    paths["metadata"].write_text(
        json.dumps(
            {
                "combined_video": str(paths["combined_video"]),
                "accepted_visual_contract": {
                    "panels": [
                        {"artifact": str(paths["row1"])},
                        {"artifact": str(paths["row2"])},
                        {"artifact": str(paths["row3"])},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clip = {
        "sample_id": sample_id,
        "acceptance_ok": True,
        "metadata": str(paths["metadata"]),
        "combined_video": str(paths["combined_video"]),
    }
    if not omit_clip_row_fields:
        clip.update(
            {
                "row1_soma_somamesh_video": str(paths["row1"]),
                "row2_g1_target_isaaclab_video": str(paths["row2"]),
                "row3_g1_kinematics_isaaclab_video": str(paths["row3"]),
            }
        )
    paths["summary"].write_text(
        json.dumps(
            {
                "step": step,
                "clips": [clip],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
