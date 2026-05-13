import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS, SKELETON_MEASURE_COLUMNS
from online_retarget.data.schema import ObservationSpec
from online_retarget.data.windowed_builder import (
    WindowedBuildConfig,
    build_windowed_jsonl,
    body_positions_from_bvh,
    parse_bvh_motion,
)


class WindowedBuilderTests(unittest.TestCase):
    def test_parse_bvh_motion_and_fk_positions(self):
        motion = parse_bvh_motion(_bvh_text())

        positions = body_positions_from_bvh(
            motion,
            body_names=("Hips", "LeftFoot"),
            root_body="Hips",
            position_scale=1.0,
        )

        self.assertEqual(len(motion.frames), 3)
        self.assertEqual(len(positions), 3)
        self.assertEqual(positions[0][:3], [0.0, 0.0, 0.0])
        self.assertEqual(positions[0][3:6], [0.0, -1.0, 0.0])
        self.assertEqual(positions[2][:3], [0.0, 0.0, 0.0])

    def test_build_windowed_jsonl_matches_schema_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "curated_index.csv"
            _write_index(index_csv)

            result = build_windowed_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=WindowedBuildConfig(
                    limit=1,
                    history_frames=2,
                    source_body_names=("Hips", "LeftFoot"),
                ),
            )
            sample = json.loads(result.samples_jsonl.read_text().strip())
            manifest = json.loads(result.manifest_json.read_text())

        spec = ObservationSpec(history_frames=2, source_body_count=2)
        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.input_dim, spec.flattened_dim())
        self.assertEqual(len(sample["observation"]), spec.flattened_dim())
        self.assertEqual(len(sample["target_joints"]), len(G1_JOINT_COLUMNS))
        self.assertEqual(manifest["builder"], "bvh_fk_30body_window")

    def test_build_windowed_jsonl_uses_action_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "curated_index.csv"
            _write_index(index_csv, merged_action="quarantine")

            result = build_windowed_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=WindowedBuildConfig(
                    limit=1,
                    history_frames=2,
                    source_body_names=("Hips", "LeftFoot"),
                ),
            )

        self.assertEqual(result.sample_count, 0)


def _write_index(path: Path, merged_action: str = "keep") -> None:
    row = {
        "row_index": "1",
        "split": "train",
        "actor_uid": "A001",
        "move_name": "good",
        "filename": "good",
        "package": "Locomotion",
        "category": "Baseline",
        "is_mirror": "False",
        "move_soma_proportional_path": "soma_proportional/bvh/good.bvh",
        "move_soma_proportional_shape_path": "soma_shapes/A001.npz",
        "move_g1_path": "g1/csv/good.csv",
        "curation_action": "keep",
        "quality_flags": "",
        "merged_quality_action": merged_action,
        "actor_weight_kg": "70",
        "actor_age_yr": "30",
        "actor_gender": "M",
    }
    for column in SKELETON_MEASURE_COLUMNS:
        row[column] = "170" if column == "actor_height_cm" else "1.0"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/good.bvh", _bvh_text())


def _write_g1_tar(path: Path) -> None:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=G1_CSV_COLUMNS)
    writer.writeheader()
    for frame in range(3):
        row = {column: "0.0" for column in G1_CSV_COLUMNS}
        row["Frame"] = str(frame)
        row[G1_JOINT_COLUMNS[0]] = str(frame)
        writer.writerow(row)
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "g1/csv/good.csv", out.getvalue())


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text() -> str:
    return """HIERARCHY
ROOT Root
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hips
  {
    OFFSET 0.000000 1.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftFoot
    {
      OFFSET 0.000000 -1.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.100000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 2.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""


if __name__ == "__main__":
    unittest.main()
