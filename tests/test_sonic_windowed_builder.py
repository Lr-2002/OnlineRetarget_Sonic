from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import tarfile
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal stdlib envs.
    np = None

from online_retarget.data.bones_sonic import SONIC_JOINT_NAMES
from online_retarget.data.schema import ObservationSpec
from online_retarget.data.sonic_windowed_builder import (
    SonicWindowedBuildConfig,
    _flat_positions_to_body_positions,
    _rot6d,
    _run_name,
    _source_features_from_bvh,
    _source_features_from_sonic,
    build_sonic_windowed_jsonl,
)
from online_retarget.data.windowed_builder import parse_bvh_motion


class SonicWindowedBuilderValueTests(unittest.TestCase):
    def test_rot6d_matches_lr290_identity_and_nontrivial_rotation(self) -> None:
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        z90 = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

        self.assertEqual(_rot6d(identity), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(_rot6d(z90), [0.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    def test_soma_bvh_source_positions_are_root_orientation_local(self) -> None:
        motion = parse_bvh_motion(_bvh_text(frames=1, hips_z_rotation=90.0))
        source = _source_features_from_bvh(
            motion,
            config=SonicWindowedBuildConfig(
                source_body_names=("Hips", "LeftFoot"),
                root_body="Hips",
                position_scale=1.0,
            ),
        )

        positions = _flat_positions_to_body_positions(source.positions[0], 2)

        self.assertEqual(positions[0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(positions[1][0], 0.0, places=6)
        self.assertAlmostEqual(positions[1][1], -1.0, places=6)
        self.assertAlmostEqual(positions[1][2], 0.0, places=6)
        self.assertEqual(source.rot6d[0][0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(source.skeleton[3], 0.0, places=6)
        self.assertAlmostEqual(source.skeleton[4], -1.0, places=6)
        self.assertAlmostEqual(source.skeleton[5], 0.0, places=6)

    def test_run_name_encodes_nondefault_target_future_step(self) -> None:
        run_name = _run_name(
            SonicWindowedBuildConfig(
                task_query="walk",
                target_horizon_frames=10,
                target_future_step=5,
                limit=128,
            )
        )

        self.assertIn("_fh10_fs5_", run_name)


@unittest.skipUnless(np is not None, "numpy required for SONIC windowed builder tests")
class SonicWindowedBuilderTests(unittest.TestCase):
    def test_default_source_mode_is_soma_bvh(self) -> None:
        self.assertEqual(SonicWindowedBuildConfig().source_mode, "soma_bvh")

    def test_build_sonic_windowed_jsonl_filters_walk_and_matches_dims(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            sonic_root = root / "bones_sonic" / "230101"
            root.mkdir()
            sonic_root.mkdir(parents=True)
            _write_source_tar(root / "soma_proportional.tar")
            walk_npz = sonic_root / "walk_forward__A001.npz"
            jump_npz = sonic_root / "jump__A001.npz"
            _write_sonic_npz(walk_npz)
            _write_sonic_npz(jump_npz)
            index_csv = Path(tmp) / "sonic_index.csv"
            _write_index(index_csv, walk_npz=walk_npz, jump_npz=jump_npz)

            result = build_sonic_windowed_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SonicWindowedBuildConfig(
                    split="train",
                    task_query="walk",
                    source_mode="soma_bvh",
                    train_ratio=1.0,
                    val_ratio=0.0,
                    limit=3,
                    history_frames=2,
                    target_horizon_frames=2,
                    window_stride=1,
                    max_windows_per_clip=3,
                    source_body_names=("Hips", "LeftFoot"),
                ),
            )
            sample_rows = [
                json.loads(line)
                for line in result.samples_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))

        spec = ObservationSpec(history_frames=2, source_body_count=2)
        self.assertEqual(result.sample_count, 3)
        self.assertEqual(result.selected_clip_count, 1)
        self.assertEqual(result.input_dim, spec.flattened_dim())
        self.assertEqual(result.output_dim, len(SONIC_JOINT_NAMES) * 2)
        self.assertEqual(len(sample_rows[0]["observation"]), spec.flattened_dim())
        self.assertEqual(len(sample_rows[0]["target_joints"]), len(SONIC_JOINT_NAMES))
        self.assertEqual(len(sample_rows[0]["future_target_joints"]), 2)
        self.assertEqual(len(sample_rows[0]["future_target_joints"][0]), len(SONIC_JOINT_NAMES))
        self.assertEqual(len(sample_rows[0]["prev_target_joints"]), len(SONIC_JOINT_NAMES))
        self.assertEqual(len(sample_rows[0]["source_body_tokens"]), 2)
        self.assertEqual(len(sample_rows[0]["source_body_tokens"][0]), 2)
        self.assertEqual(len(sample_rows[0]["source_body_tokens"][0][0]), 15)
        self.assertEqual(len(sample_rows[0]["source_skeleton"]), 8)
        self.assertEqual(sample_rows[0]["target_horizon_frames"], 2)
        self.assertEqual(sample_rows[0]["target_frame_indices"], [1, 2])
        self.assertEqual(sample_rows[1]["prev_target_frame"], sample_rows[1]["target_frame"] - 1)
        self.assertIn("walk_forward", sample_rows[0]["sample_id"])
        self.assertEqual(manifest["builder"], "sonic_walk_soma_bvh_to_g1_joint_window_debug")
        self.assertEqual(manifest["source_format"], "soma_bvh")
        self.assertEqual(manifest["target_format"], "bones_sonic_joint_pos_future_window")
        self.assertEqual(manifest["source_body_token_dim"], 15)
        self.assertEqual(manifest["source_step_dim"], 30)
        self.assertEqual(manifest["source_skeleton_dim"], 8)
        self.assertEqual(manifest["source_rotation_representation"], "rot6d")
        self.assertEqual(manifest["target_future_step"], 1)
        self.assertEqual(manifest["candidate_clip_count"], 1)
        self.assertEqual(manifest["target_horizon_frames"], 2)

    def test_build_sonic_windowed_jsonl_emits_rot6d_body_tokens(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            sonic_root = root / "bones_sonic" / "230101"
            root.mkdir()
            sonic_root.mkdir(parents=True)
            _write_source_tar(root / "soma_proportional.tar", frames=12)
            walk_npz = sonic_root / "walk_forward__A001.npz"
            jump_npz = sonic_root / "jump__A001.npz"
            _write_sonic_npz(walk_npz, frames=12)
            _write_sonic_npz(jump_npz, frames=12)
            index_csv = Path(tmp) / "sonic_index.csv"
            _write_index(index_csv, walk_npz=walk_npz, jump_npz=jump_npz)

            result = build_sonic_windowed_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SonicWindowedBuildConfig(
                    split="train",
                    task_query="walk",
                    source_mode="soma_bvh",
                    train_ratio=1.0,
                    val_ratio=0.0,
                    limit=1,
                    history_frames=2,
                    target_horizon_frames=2,
                    target_future_step=5,
                    window_stride=1,
                    max_windows_per_clip=1,
                ),
            )
            sample = json.loads(result.samples_jsonl.read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))

        self.assertEqual(len(sample["source_body_tokens"]), 2)
        self.assertEqual(len(sample["source_body_tokens"][0]), 30)
        self.assertEqual(len(sample["source_body_tokens"][0][0]), 15)
        self.assertEqual(len(sample["source_skeleton"]), 120)
        self.assertEqual(sample["target_frame_indices"], [1, 6])
        self.assertEqual(sample["target_future_step"], 5)
        self.assertEqual(manifest["source_body_count"], 30)
        self.assertEqual(manifest["source_body_token_dim"], 15)
        self.assertEqual(manifest["source_step_dim"], 450)
        self.assertEqual(manifest["source_skeleton_dim"], 120)
        self.assertEqual(manifest["source_rotation_representation"], "rot6d")
        self.assertEqual(manifest["target_future_step"], 5)
        self.assertIn("_fs5_", str(result.samples_jsonl))

    def test_sonic_body_pos_source_positions_are_root_orientation_local(self) -> None:
        assert np is not None
        body_pos = np.zeros((2, 30, 3), dtype=np.float32)
        body_pos[0, 1, :] = [1.0, 0.0, 0.0]
        body_pos[1, 1, :] = [2.0, 0.0, 0.0]
        body_quat = np.zeros((2, 30, 4), dtype=np.float32)
        body_quat[:, :, 0] = 1.0
        z90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        body_quat[:, 0, :] = z90
        body_quat[:, 1, :] = z90
        body_ang_vel = np.zeros((2, 30, 3), dtype=np.float32)
        body_ang_vel[:, 1, :] = [0.0, 1.0, 0.0]

        source = _source_features_from_sonic(
            body_pos,
            body_quat,
            body_ang_vel,
            config=SonicWindowedBuildConfig(
                source_body_names=("Hips", "Spine1"),
                position_scale=1.0,
            ),
            np=np,
        )

        frame0 = _flat_positions_to_body_positions(source.positions[0], 2)
        frame1 = _flat_positions_to_body_positions(source.positions[1], 2)

        self.assertEqual(frame0[0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(frame0[1][0], 0.0, places=6)
        self.assertAlmostEqual(frame0[1][1], -1.0, places=6)
        self.assertAlmostEqual(frame1[1][0], 0.0, places=6)
        self.assertAlmostEqual(frame1[1][1], -2.0, places=6)
        self.assertAlmostEqual(source.linear_velocities[1][1][0], 0.0, places=6)
        self.assertAlmostEqual(source.linear_velocities[1][1][1], -1.0, places=6)
        for actual, expected in zip(
            source.rot6d[0][1],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(source.angular_velocities[0][1][0], 1.0, places=6)
        self.assertAlmostEqual(source.angular_velocities[0][1][1], 0.0, places=6)

    def test_soma_bvh_reads_only_needed_prefix_frames(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            sonic_root = root / "bones_sonic" / "230101"
            root.mkdir()
            sonic_root.mkdir(parents=True)
            _write_source_tar(root / "soma_proportional.tar", frames=20)
            walk_npz = sonic_root / "walk_forward__A001.npz"
            jump_npz = sonic_root / "jump__A001.npz"
            _write_sonic_npz(walk_npz, frames=20)
            _write_sonic_npz(jump_npz, frames=20)
            index_csv = Path(tmp) / "sonic_index.csv"
            _write_index(index_csv, walk_npz=walk_npz, jump_npz=jump_npz)

            result = build_sonic_windowed_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SonicWindowedBuildConfig(
                    split="train",
                    task_query="walk",
                    source_mode="soma_bvh",
                    train_ratio=1.0,
                    val_ratio=0.0,
                    limit=1,
                    history_frames=2,
                    window_stride=1,
                    max_windows_per_clip=1,
                    source_body_names=("Hips", "LeftFoot"),
                ),
            )

        self.assertEqual(result.sample_count, 1)


def _write_index(path: Path, *, walk_npz: Path, jump_npz: Path) -> None:
    rows = [
        {
            "sonic_relative_path": "230101/walk_forward__A001.npz",
            "sonic_path": str(walk_npz),
            "date": "230101",
            "filename": "walk_forward__A001",
            "actor_uid": "A001",
            "is_mirror": "False",
            "metadata_row_index": "1",
            "package": "Locomotion",
            "category": "Basic Locomotion",
            "source_soma_proportional_path": "soma_proportional/bvh/walk_forward__A001.bvh",
            "schema_status": "ok",
        },
        {
            "sonic_relative_path": "230101/jump__A001.npz",
            "sonic_path": str(jump_npz),
            "date": "230101",
            "filename": "jump__A001",
            "actor_uid": "A001",
            "is_mirror": "False",
            "metadata_row_index": "2",
            "package": "Locomotion",
            "category": "Jump",
            "source_soma_proportional_path": "soma_proportional/bvh/jump__A001.bvh",
            "schema_status": "ok",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_sonic_npz(path: Path, frames: int = 5) -> None:
    assert np is not None
    joint_pos = np.zeros((frames, len(SONIC_JOINT_NAMES)), dtype=np.float32)
    for frame in range(frames):
        joint_pos[frame, 0] = float(frame)
    body_quat_w = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    np.savez(
        path,
        fps=np.asarray([50], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_pos_w=np.zeros((frames, 30, 3), dtype=np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
    )


def _write_source_tar(path: Path, frames: int = 5) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/walk_forward__A001.bvh", _bvh_text(frames=frames))
        _add_member(tar, "soma_proportional/bvh/jump__A001.bvh", _bvh_text(frames=frames))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text(frames: int = 5, hips_z_rotation: float = 0.0) -> str:
    frame_rows = "\n".join(
        (
            f"{frame}.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "
            f"{frame}.000000 1.000000 0.000000 {hips_z_rotation:.6f} 0.000000 0.000000 "
            "0.000000 0.000000 0.000000"
        )
        for frame in range(frames)
    )
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips
  {{
    OFFSET 0.000000 1.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftFoot
    {{
      OFFSET 0.000000 -1.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }}
  }}
}}
MOTION
Frames: {frames}
Frame Time: 0.020000
{frame_rows}
"""


if __name__ == "__main__":
    unittest.main()
