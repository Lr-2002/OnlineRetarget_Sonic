import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS
from online_retarget.data.pair_quality import (
    PairQualityConfig,
    scan_pair_quality_from_index,
    summarize_pair_meta,
)


class PairQualityTests(unittest.TestCase):
    def test_summarize_pair_meta_flags_frame_mismatch(self):
        row = summarize_pair_meta(
            {
                "row_index": "1",
                "move_duration_frames": "3",
                "target_provenance": "kinematic_g1_csv",
            },
            {
                "present": True,
                "flags": (),
                "declared_frames": 3,
                "frame_count": 3,
                "frame_time": 1.0 / 30.0,
            },
            {"present": True, "flags": (), "frame_count": 5},
            PairQualityConfig(g1_fps=30.0, max_frame_count_delta=0),
        )

        self.assertEqual(row["quality_action"], "quarantine")
        self.assertIn("pair_frame_count_mismatch", row["quality_flags"])
        self.assertIn("pair_metadata_g1_frame_mismatch", row["quality_flags"])
        self.assertEqual(row["abs_frame_count_delta"], 2)

    def test_summarize_pair_meta_defaults_to_bones_seed_120hz(self):
        row = summarize_pair_meta(
            {
                "row_index": "1",
                "move_duration_frames": "3",
                "target_provenance": "kinematic_g1_csv",
            },
            {
                "present": True,
                "flags": (),
                "declared_frames": 3,
                "frame_count": 3,
                "frame_time": 1.0 / 120.0,
            },
            {"present": True, "flags": (), "frame_count": 3},
            PairQualityConfig(),
        )

        self.assertEqual(row["quality_action"], "keep")
        self.assertEqual(row["g1_fps"], 120.0)
        self.assertEqual(row["abs_duration_delta_sec"], 0.0)

    def test_scan_pair_quality_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_pair_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=PairQualityConfig(
                    expected_source_frame_time=1.0 / 30.0,
                    g1_fps=30.0,
                    max_frame_count_delta=0,
                    max_duration_delta_sec=1e-3,
                ),
                limit=None,
            )
            rows = [
                json.loads(line)
                for line in result.stats_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual(result.scanned_rows, 3)
        self.assertEqual(result.action_counts["keep"], 1)
        self.assertEqual(result.action_counts["quarantine"], 1)
        self.assertEqual(result.action_counts["exclude"], 1)
        self.assertEqual(result.flag_counts["pair_frame_count_mismatch"], 1)
        self.assertEqual(result.flag_counts["missing_g1_csv_member"], 1)
        self.assertEqual(rows[0]["source_frame_count"], 3)
        self.assertEqual(rows[0]["g1_frame_count"], 3)
        self.assertEqual(rows[0]["target_provenance"], "kinematic_g1_csv")
        self.assertEqual(report["sampling"]["mode"], "first_n")
        self.assertIsNone(report["sampling"]["limit"])

    def test_scan_pair_quality_stratified_sample_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_pair_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=PairQualityConfig(),
                limit=2,
                sample_by=("category",),
            )
            rows = [
                json.loads(line)
                for line in result.stats_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(result.report_json.read_text(encoding="utf-8"))

        self.assertEqual(result.scanned_rows, 2)
        self.assertEqual([row["category"] for row in rows], ["Baseline", "Jump"])
        self.assertEqual(report["sampling"]["mode"], "stratified_round_robin")


def _write_index(path: Path) -> None:
    fieldnames = [
        "row_index",
        "split",
        "actor_uid",
        "move_name",
        "filename",
        "package",
        "category",
        "is_mirror",
        "actor_gender",
        "move_duration_frames",
        "move_soma_proportional_path",
        "move_g1_path",
        "curation_action",
    ]
    rows = [
        {
            "row_index": "1",
            "split": "train",
            "actor_uid": "A001",
            "move_name": "good",
            "filename": "good",
            "package": "Locomotion",
            "category": "Baseline",
            "is_mirror": "False",
            "actor_gender": "M",
            "move_duration_frames": "3",
            "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
            "move_g1_path": "g1/csv/240101/good.csv",
            "curation_action": "keep",
        },
        {
            "row_index": "2",
            "split": "train",
            "actor_uid": "A002",
            "move_name": "jump",
            "filename": "jump",
            "package": "Locomotion",
            "category": "Jump",
            "is_mirror": "False",
            "actor_gender": "F",
            "move_duration_frames": "3",
            "move_soma_proportional_path": "soma_proportional/bvh/240101/jump.bvh",
            "move_g1_path": "g1/csv/240101/jump.csv",
            "curation_action": "keep",
        },
        {
            "row_index": "3",
            "split": "train",
            "actor_uid": "A003",
            "move_name": "missing",
            "filename": "missing",
            "package": "Locomotion",
            "category": "Baseline",
            "is_mirror": "True",
            "actor_gender": "M",
            "move_duration_frames": "3",
            "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
            "move_g1_path": "g1/csv/240101/missing.csv",
            "curation_action": "keep",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/240101/good.bvh", _bvh_text(3))
        _add_member(tar, "soma_proportional/bvh/240101/jump.bvh", _bvh_text(3))


def _write_g1_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "g1/csv/240101/good.csv", _g1_csv(3))
        _add_member(tar, "g1/csv/240101/jump.csv", _g1_csv(4))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text(frame_count: int) -> str:
    rows = "\n".join(
        "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000"
        for _ in range(frame_count)
    )
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}}
MOTION
Frames: {frame_count}
Frame Time: 0.03333333333333333
{rows}
"""


def _g1_csv(frame_count: int) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=G1_CSV_COLUMNS)
    writer.writeheader()
    for frame in range(frame_count):
        row = {column: "0.0" for column in G1_CSV_COLUMNS}
        row["Frame"] = str(frame)
        writer.writerow(row)
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
