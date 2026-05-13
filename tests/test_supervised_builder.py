import csv
import io
import json
import tarfile
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS, SKELETON_MEASURE_COLUMNS
from online_retarget.data.supervised_builder import SupervisedBuildConfig, build_supervised_jsonl


class SupervisedBuilderTests(unittest.TestCase):
    def test_build_supervised_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = build_supervised_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SupervisedBuildConfig(limit=1, history_frames=2),
            )

            sample = json.loads(result.samples_jsonl.read_text().strip())

        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.skipped_count, 0)
        self.assertEqual(result.output_dim, len(G1_JOINT_COLUMNS))
        self.assertEqual(len(sample["observation"]), 2 * 6 + 13)
        self.assertEqual(len(sample["target_joints"]), len(G1_JOINT_COLUMNS))
        self.assertEqual(sample["target_frame"], 1)

    def test_build_supervised_jsonl_uses_action_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "curated_index.csv"
            _write_index(index_csv, merged_action="quarantine")

            result = build_supervised_jsonl(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SupervisedBuildConfig(
                    limit=1,
                    history_frames=2,
                    action_column="merged_quality_action",
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
        "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
        "move_soma_proportional_shape_path": "soma_shapes/A001.npz",
        "move_g1_path": "g1/csv/240101/good.csv",
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
    text = """HIERARCHY
ROOT Root
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}
MOTION
Frames: 3
Frame Time: 0.100000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
1.000000 0.000000 0.000000 0.000000 0.000000 0.000000
2.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/240101/good.bvh", text)


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
        _add_member(tar, "g1/csv/240101/good.csv", out.getvalue())


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
