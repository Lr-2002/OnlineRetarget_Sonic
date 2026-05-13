import csv
import io
import tarfile
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
from online_retarget.data.g1_quality import G1QualityConfig, scan_g1_quality_from_index


class G1QualityTests(unittest.TestCase):
    def test_scan_g1_quality_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_g1_tar(root / "g1.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_g1_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=G1QualityConfig(fps=30.0, max_joint_velocity=20.0, max_root_speed=8.0),
                limit=None,
            )

            self.assertTrue(result.stats_jsonl.exists())
            self.assertTrue(result.report_json.exists())
            self.assertEqual(result.scanned_rows, 3)
            self.assertEqual(result.action_counts["keep"], 1)
            self.assertEqual(result.action_counts["quarantine"], 1)
            self.assertEqual(result.action_counts["exclude"], 1)
            self.assertEqual(result.flag_counts["joint_velocity_jump"], 1)
            self.assertEqual(result.flag_counts["root_discontinuity"], 1)
            self.assertEqual(result.flag_counts["missing_g1_csv_member"], 1)


def _write_index(path: Path) -> None:
    fieldnames = [
        "row_index",
        "split",
        "actor_uid",
        "move_name",
        "filename",
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
            "move_g1_path": "g1/csv/240101/good.csv",
            "curation_action": "keep",
        },
        {
            "row_index": "2",
            "split": "train",
            "actor_uid": "A002",
            "move_name": "jump",
            "filename": "jump",
            "move_g1_path": "g1/csv/240101/jump.csv",
            "curation_action": "keep",
        },
        {
            "row_index": "3",
            "split": "train",
            "actor_uid": "A003",
            "move_name": "missing",
            "filename": "missing",
            "move_g1_path": "g1/csv/240101/missing.csv",
            "curation_action": "keep",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_g1_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "g1/csv/240101/good.csv", _csv_text([0.0, 0.01, 0.02], 0.01))
        _add_member(tar, "g1/csv/240101/jump.csv", _csv_text([0.0, 2.0, 2.1], 1.0))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _csv_text(first_joint_values: list[float], root_step: float) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=G1_CSV_COLUMNS)
    writer.writeheader()
    for frame, first_joint in enumerate(first_joint_values):
        row = {column: "0.0" for column in G1_CSV_COLUMNS}
        row.update(
            {
                "Frame": str(frame),
                "root_translateX": str(frame * root_step),
                "root_translateY": "0.0",
                "root_translateZ": "1.0",
            }
        )
        row[G1_JOINT_COLUMNS[0]] = str(first_joint)
        writer.writerow(row)
    return out.getvalue()


if __name__ == "__main__":
    unittest.main()
