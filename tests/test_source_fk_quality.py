import csv
import io
import json
import tarfile
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.source_fk_quality import (
    SourceFKQualityConfig,
    scan_source_fk_quality_from_index,
    summarize_source_fk_motion,
)
from online_retarget.data.windowed_builder import parse_bvh_motion


class SourceFKQualityTests(unittest.TestCase):
    def test_summarize_source_fk_motion_keeps_grounded_motion(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[0.0, 0.0, 0.0]))

        row = summarize_source_fk_motion(
            {"row_index": "1"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                min_contact_frame_ratio=0.5,
            ),
        )

        self.assertEqual(row["quality_action"], "keep")
        self.assertEqual(row["quality_flags"], "")
        self.assertEqual(row["contact_frame_ratio"], 1.0)

    def test_summarize_source_fk_motion_flags_foot_slide(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 1.0, 2.0], left_foot_y=[0.0, 0.0, 0.0]))

        row = summarize_source_fk_motion(
            {"row_index": "1"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                max_contact_slide_speed=0.5,
            ),
        )

        self.assertEqual(row["quality_action"], "downweight")
        self.assertIn("source_foot_slide", row["quality_flags"])
        self.assertGreater(row["max_contact_slide_speed"], 0.5)

    def test_summarize_source_fk_motion_derives_fps_from_bvh_frame_time(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 1.0, 2.0], left_foot_y=[0.0, 0.0, 0.0]))

        row = summarize_source_fk_motion(
            {"row_index": "1"},
            motion,
            SourceFKQualityConfig(
                position_scale=1.0,
                ground_height=0.0,
                max_contact_slide_speed=0.5,
            ),
        )

        self.assertEqual(row["quality_action"], "downweight")
        self.assertAlmostEqual(row["max_contact_slide_speed"], 10.0)

    def test_summarize_source_fk_motion_records_root_height_and_support(self):
        motion = parse_bvh_motion(
            _bvh_text(
                left_foot_x=[0.0, 0.0, 0.0],
                left_foot_y=[1.0, 1.0, 1.0],
                foot_offset_y=-1.0,
            )
        )

        row = summarize_source_fk_motion(
            {"row_index": "support"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                min_contact_frame_ratio=0.0,
            ),
        )

        self.assertEqual(row["quality_action"], "keep")
        self.assertEqual(row["root_body"], "Hips")
        self.assertEqual(row["root_height_min"], 1.0)
        self.assertEqual(row["root_height_max"], 1.0)
        self.assertEqual(row["root_height_range"], 0.0)
        self.assertEqual(row["mean_root_height"], 1.0)
        self.assertEqual(row["support_frame_ratio"], 1.0)
        self.assertEqual(row["mean_root_support_distance"], 0.0)
        self.assertEqual(row["max_root_support_distance"], 0.0)

    def test_summarize_source_fk_motion_flags_float_and_penetration(self):
        floating = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[0.2, 0.2, 0.2]))
        penetrating = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[-0.1, -0.1, -0.1]))
        config = SourceFKQualityConfig(
            fps=10.0,
            position_scale=1.0,
            ground_height=0.0,
            max_mean_foot_clearance=0.1,
            max_penetration_depth=0.03,
        )

        floating_row = summarize_source_fk_motion({"row_index": "1"}, floating, config)
        penetrating_row = summarize_source_fk_motion({"row_index": "2"}, penetrating, config)

        self.assertIn("source_low_foot_contact", floating_row["quality_flags"])
        self.assertIn("source_foot_float", floating_row["quality_flags"])
        self.assertIn("source_ground_penetration", penetrating_row["quality_flags"])
        self.assertEqual(floating_row["quality_action"], "quarantine")
        self.assertEqual(penetrating_row["quality_action"], "quarantine")

    def test_summarize_source_fk_motion_marks_contact_correction_candidate(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[0.08, 0.08, 0.08]))

        row = summarize_source_fk_motion(
            {"row_index": "repairable-float"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                max_mean_foot_clearance=0.05,
                max_contact_correction_offset=0.10,
            ),
        )

        self.assertEqual(row["quality_action"], "quarantine")
        self.assertIn("source_foot_float", row["quality_flags"])
        self.assertEqual(row["contact_correction_candidate"], 1)
        self.assertEqual(row["contact_correction_reason"], "vertical_float_offset")
        self.assertEqual(row["contact_correction_offset"], -0.08)
        self.assertEqual(row["contact_correction_abs_offset"], 0.08)

    def test_summarize_source_fk_motion_rejects_large_contact_correction_candidate(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[0.2, 0.2, 0.2]))

        row = summarize_source_fk_motion(
            {"row_index": "too-far"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                max_mean_foot_clearance=0.05,
                max_contact_correction_offset=0.10,
            ),
        )

        self.assertEqual(row["contact_correction_candidate"], 0)
        self.assertEqual(row["contact_correction_reason"], "")
        self.assertEqual(row["contact_correction_abs_offset"], 0.0)

    def test_summarize_source_fk_motion_excludes_nonfinite_fk(self):
        motion = parse_bvh_motion(_bvh_text(left_foot_x=[0.0, float("nan"), 0.0], left_foot_y=[0.0, 0.0, 0.0]))

        row = summarize_source_fk_motion(
            {"row_index": "1"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
            ),
        )

        self.assertEqual(row["quality_action"], "exclude")
        self.assertEqual(row["quality_flags"], "nonfinite_fk_position")

    def test_summarize_source_fk_motion_honors_frame_budget(self):
        motion = parse_bvh_motion(
            _bvh_text(
                left_foot_x=[0.0, 0.0, 0.0, 0.0, 0.0],
                left_foot_y=[0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )

        row = summarize_source_fk_motion(
            {"row_index": "1"},
            motion,
            SourceFKQualityConfig(
                fps=10.0,
                position_scale=1.0,
                ground_height=0.0,
                frame_stride=2,
                max_frames=2,
            ),
        )

        self.assertEqual(row["original_frame_count"], 5)
        self.assertEqual(row["frame_count"], 2)

    def test_scan_source_fk_quality_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_source_fk_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SourceFKQualityConfig(
                    fps=10.0,
                    position_scale=1.0,
                    ground_height=0.0,
                    max_contact_slide_speed=0.5,
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
        self.assertEqual(result.action_counts["downweight"], 1)
        self.assertEqual(result.action_counts["exclude"], 1)
        self.assertEqual(result.flag_counts["source_foot_slide"], 1)
        self.assertEqual(result.flag_counts["missing_source_bvh_member"], 1)
        self.assertEqual(rows[0]["ground_source"], "fixed")
        self.assertEqual(rows[0]["is_mirror"], "False")
        self.assertEqual(rows[0]["actor_gender"], "M")
        self.assertEqual(report["sampling"]["mode"], "first_n")

    def test_scan_source_fk_quality_stratified_sample_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            root.mkdir()
            _write_source_tar(root / "soma_proportional.tar")
            index_csv = Path(tmp) / "split_index.csv"
            _write_index(index_csv)

            result = scan_source_fk_quality_from_index(
                data_root=root,
                index_csv=index_csv,
                output_root=output,
                config=SourceFKQualityConfig(
                    fps=1.0,
                    position_scale=1.0,
                    ground_height=0.0,
                    max_contact_slide_speed=1000.0,
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
            "move_soma_proportional_path": "soma_proportional/bvh/good.bvh",
            "curation_action": "keep",
        },
        {
            "row_index": "2",
            "split": "train",
            "actor_uid": "A002",
            "move_name": "slide",
            "filename": "slide",
            "package": "Locomotion",
            "category": "Jump",
            "is_mirror": "False",
            "actor_gender": "F",
            "move_soma_proportional_path": "soma_proportional/bvh/slide.bvh",
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
            "move_soma_proportional_path": "soma_proportional/bvh/missing.bvh",
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
            "soma_proportional/bvh/good.bvh",
            _bvh_text(left_foot_x=[0.0, 0.0, 0.0], left_foot_y=[0.0, 0.0, 0.0]),
        )
        _add_member(
            tar,
            "soma_proportional/bvh/slide.bvh",
            _bvh_text(left_foot_x=[0.0, 1.0, 2.0], left_foot_y=[0.0, 0.0, 0.0]),
        )


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text(
    left_foot_x: list[float],
    left_foot_y: list[float],
    foot_offset_y: float = 0.0,
) -> str:
    if len(left_foot_x) != len(left_foot_y):
        raise ValueError("left_foot_x and left_foot_y must have same length")
    rows = "\n".join(
        (
            "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "
            f"{_format_bvh_float(x)} {_format_bvh_float(y)} 0.000000 "
            "0.000000 0.000000 0.000000 "
            "0.000000 0.000000 0.000000"
        )
        for x, y in zip(left_foot_x, left_foot_y)
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
      OFFSET 0.000000 {_format_bvh_float(foot_offset_y)} 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }}
  }}
}}
MOTION
Frames: {len(left_foot_x)}
Frame Time: 0.100000
{rows}
"""


def _format_bvh_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    unittest.main()
