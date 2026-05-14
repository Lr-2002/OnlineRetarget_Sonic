import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS
from online_retarget.data.review_clips import ReviewClipExportConfig, export_review_clips


class ReviewClipExportTests(unittest.TestCase):
    def test_export_review_clips_extracts_files_and_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_source_tar(data_root / "soma_proportional.tar")
            _write_g1_tar(data_root / "g1.tar")
            review_csv = root / "review.csv"
            _write_review_csv(review_csv)

            result = export_review_clips(
                data_root=data_root,
                input_csv=review_csv,
                output_root=root / "clips",
                run_name="fixture",
                label="quarantine",
                config=ReviewClipExportConfig(limit=1),
            )
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
            rows = _read_csv(result.summary_csv)
            metadata_path = result.output_dir / "00_quarantine_good" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_exists = Path(metadata["source_bvh"]).exists()
            target_exists = Path(metadata["target_g1_csv"]).exists()

        self.assertEqual(result.exported_rows, 1)
        self.assertEqual(result.render_counts, {"not_requested": 1})
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["render_status"], {"not_requested": 1})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "good")
        self.assertEqual(rows[0]["render_status"], "not_requested")
        self.assertEqual(metadata["quality_flags"], "pair_duration_mismatch")
        self.assertTrue(source_exists)
        self.assertTrue(target_exists)

    def test_render_requires_model_xml(self):
        with self.assertRaises(ValueError) as raised:
            export_review_clips(
                data_root=Path("/tmp"),
                input_csv=Path("/tmp/missing.csv"),
                output_root=Path("/tmp/out"),
                run_name="bad",
                label="review",
                config=ReviewClipExportConfig(render_g1=True),
            )

        self.assertIn("model_xml", str(raised.exception))

    def test_export_review_clips_accepts_merged_quality_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_source_tar(data_root / "soma_proportional.tar")
            _write_g1_tar(data_root / "g1.tar")
            review_csv = root / "worst_clips.csv"
            _write_merged_review_csv(review_csv)

            result = export_review_clips(
                data_root=data_root,
                input_csv=review_csv,
                output_root=root / "clips",
                run_name="merged",
                label="g1_quality",
                config=ReviewClipExportConfig(limit=1),
            )
            rows = _read_csv(result.summary_csv)
            metadata = json.loads(
                (result.output_dir / "00_g1_quality_good" / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rows[0]["quality_action"], "quarantine")
        self.assertEqual(rows[0]["quality_flags"], "g1:g1_ground_penetration")
        self.assertEqual(metadata["merged_quality_action"], "quarantine")
        self.assertEqual(metadata["merged_quality_flags"], "g1:g1_ground_penetration")


def _write_review_csv(path: Path) -> None:
    fieldnames = [
        "quality_action",
        "quality_flags",
        "row_index",
        "split",
        "category",
        "actor_uid",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
        "source_frame_count",
        "g1_frame_count",
        "abs_frame_count_delta",
        "abs_duration_delta_sec",
        "source_duration_sec",
        "g1_duration_sec",
    ]
    rows = [
        {
            "quality_action": "quarantine",
            "quality_flags": "pair_duration_mismatch",
            "row_index": "1",
            "split": "train",
            "category": "Baseline",
            "actor_uid": "A001",
            "filename": "good",
            "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
            "move_g1_path": "g1/csv/240101/good.csv",
            "source_frame_count": "3",
            "g1_frame_count": "3",
            "abs_frame_count_delta": "0",
            "abs_duration_delta_sec": "0.01",
            "source_duration_sec": "0.1",
            "g1_duration_sec": "0.11",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_merged_review_csv(path: Path) -> None:
    fieldnames = [
        "merged_quality_action",
        "merged_quality_flags",
        "row_index",
        "split",
        "category",
        "actor_uid",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
    ]
    rows = [
        {
            "merged_quality_action": "quarantine",
            "merged_quality_flags": "g1:g1_ground_penetration",
            "row_index": "1",
            "split": "train",
            "category": "Baseline",
            "actor_uid": "A001",
            "filename": "good",
            "move_soma_proportional_path": "soma_proportional/bvh/240101/good.bvh",
            "move_g1_path": "g1/csv/240101/good.csv",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/240101/good.bvh", _bvh_text())


def _write_g1_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "g1/csv/240101/good.csv", _g1_csv())


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
}
MOTION
Frames: 3
Frame Time: 0.03333333333333333
0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0
"""


def _g1_csv() -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=G1_CSV_COLUMNS)
    writer.writeheader()
    for frame in range(3):
        row = {column: "0.0" for column in G1_CSV_COLUMNS}
        row["Frame"] = str(frame)
        writer.writerow(row)
    return output.getvalue()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
