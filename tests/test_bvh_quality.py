import csv
import io
import json
import tarfile
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bvh_quality import (
    BVHQualityConfig,
    parse_bvh_text,
    scan_bvh_quality_from_index,
)


class BVHQualityTests(unittest.TestCase):
    def test_parse_bvh_text(self):
        parsed = parse_bvh_text(_bvh_text([0.0, 0.1, 0.2]))

        self.assertEqual(parsed["declared_frames"], 3)
        self.assertEqual(parsed["frame_time"], 0.1)
        self.assertEqual(len(parsed["channel_names"]), 6)
        self.assertEqual(len(parsed["position_groups"]), 1)

    def test_scan_bvh_quality_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_bvh_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=BVHQualityConfig(
                    max_channel_velocity=10.0,
                    max_root_speed=10.0,
                    expected_frame_time=0.1,
                ),
                limit=None,
            )

            self.assertTrue(result.stats_jsonl.exists())
            self.assertTrue(result.report_json.exists())
            self.assertEqual(result.scanned_rows, 3)
            self.assertEqual(result.action_counts["keep"], 1)
            self.assertEqual(result.action_counts["quarantine"], 1)
            self.assertEqual(result.action_counts["exclude"], 1)
            self.assertEqual(result.flag_counts["source_channel_jump"], 1)
            self.assertEqual(result.flag_counts["source_root_discontinuity"], 1)
            self.assertEqual(result.flag_counts["missing_source_bvh_member"], 1)
            rows = [
                json.loads(line)
                for line in result.stats_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["package"], "Locomotion")
            self.assertEqual(rows[0]["category"], "Baseline")
            self.assertEqual(rows[0]["is_mirror"], "False")
            self.assertEqual(rows[0]["actor_gender"], "M")
            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["sampling"]["mode"], "first_n")

    def test_scan_bvh_quality_stratified_sample_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_bvh_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=BVHQualityConfig(
                    max_channel_velocity=1000.0,
                    max_root_speed=1000.0,
                    expected_frame_time=0.1,
                ),
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

    def test_scan_bvh_quality_records_dynamic_metrics_without_default_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_bvh_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=BVHQualityConfig(
                    max_channel_velocity=1000.0,
                    max_root_speed=1000.0,
                    expected_frame_time=0.1,
                ),
                limit=1,
            )
            row = json.loads(result.stats_jsonl.read_text(encoding="utf-8").splitlines()[0])

        self.assertGreater(row["max_abs_channel_acceleration"], 0.0)
        self.assertGreater(row["max_root_acceleration"], 0.0)
        self.assertEqual(row["channel_acceleration_jump_rate"], 0.0)
        self.assertEqual(row["root_acceleration_jump_rate"], 0.0)
        self.assertEqual(row["root_jerk_jump_rate"], 0.0)

    def test_scan_bvh_quality_flags_dynamic_thresholds_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_bvh_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=BVHQualityConfig(
                    max_channel_velocity=1000.0,
                    max_root_speed=1000.0,
                    max_channel_acceleration=1.0,
                    max_root_acceleration=1.0,
                    expected_frame_time=0.1,
                ),
                limit=1,
            )

        self.assertEqual(result.action_counts["quarantine"], 1)
        self.assertEqual(result.flag_counts["source_channel_acceleration_jump"], 1)
        self.assertEqual(result.flag_counts["source_root_acceleration_jump"], 1)


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
        "move_soma_proportional_path",
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
            "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
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
            "move_soma_proportional_path": "soma_proportional/bvh/240101/jump.bvh",
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
            "move_soma_proportional_path": "soma_proportional/bvh/240101/missing.bvh",
            "curation_action": "keep",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(
            tar,
            "soma_proportional/bvh/240101/good.bvh",
            _bvh_text([0.0, 0.1, 0.4, 0.9]),
        )
        _add_member(tar, "soma_proportional/bvh/240101/jump.bvh", _bvh_text([0.0, 10.0, 10.2]))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text(root_x_values: list[float]) -> str:
    rows = "\n".join(
        f"{value:.6f} 0.000000 0.000000 0.000000 0.000000 0.000000"
        for value in root_x_values
    )
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}}
MOTION
Frames: {len(root_x_values)}
Frame Time: 0.100000
{rows}
"""


if __name__ == "__main__":
    unittest.main()
