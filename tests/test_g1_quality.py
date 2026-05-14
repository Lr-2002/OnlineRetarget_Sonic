import csv
import io
import json
import tarfile
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
from online_retarget.data.g1_quality import (
    G1QualityConfig,
    load_g1_kinematic_model,
    scan_g1_quality_from_index,
    summarize_g1_rows,
)


class G1QualityTests(unittest.TestCase):
    def test_default_fps_matches_bones_seed_pair_rate(self):
        self.assertEqual(G1QualityConfig().fps, 120.0)

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
                config=G1QualityConfig(
                    fps=30.0,
                    max_joint_velocity=20.0,
                    max_root_speed=8.0,
                    root_position_scale=1.0,
                    joint_angle_scale=1.0,
                    root_rotation_scale=1.0,
                ),
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

            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["model"]["mode"], "csv_root_joint")
            self.assertEqual(report["sampling"]["mode"], "first_n")
            rows = [
                json.loads(line)
                for line in result.stats_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["package"], "Locomotion")
            self.assertEqual(rows[0]["category"], "Baseline")
            self.assertEqual(rows[0]["is_mirror"], "False")
            self.assertEqual(rows[0]["actor_gender"], "M")

    def test_scan_g1_quality_stratified_sample_by_category(self):
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
                config=G1QualityConfig(
                    fps=30.0,
                    max_joint_velocity=1000.0,
                    max_root_speed=1000.0,
                    root_position_scale=1.0,
                    joint_angle_scale=1.0,
                    root_rotation_scale=1.0,
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
            self.assertEqual(report["sampling"]["sample_by"], ["category"])

    def test_summarize_g1_rows_records_dynamic_metrics_without_default_flags(self):
        row = summarize_g1_rows(
            {"row_index": "dynamic"},
            _csv_rows(
                first_joint_values=[0.0, 0.1, 0.5, 1.5],
                root_x_values=[0.0, 0.1, 0.5, 1.5],
                root_z_values=[1.0, 1.0, 1.0, 1.0],
            ),
            G1QualityConfig(
                fps=10.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                max_start_end_root_speed=1000.0,
            ),
        )

        self.assertGreater(row["max_abs_joint_acceleration"], 0.0)
        self.assertGreater(row["max_root_acceleration"], 0.0)
        self.assertGreater(row["max_root_jerk"], 0.0)
        self.assertEqual(row["joint_acceleration_jump_rate"], 0.0)
        self.assertEqual(row["root_acceleration_jump_rate"], 0.0)
        self.assertEqual(row["root_jerk_jump_rate"], 0.0)
        self.assertNotIn("g1_root_jerk_jump", row["quality_flags"])

    def test_summarize_g1_rows_flags_dynamic_thresholds_when_configured(self):
        row = summarize_g1_rows(
            {"row_index": "dynamic-threshold"},
            _csv_rows(
                first_joint_values=[0.0, 0.1, 0.5, 1.5],
                root_x_values=[0.0, 0.1, 0.5, 1.5],
                root_z_values=[1.0, 1.0, 1.0, 1.0],
            ),
            G1QualityConfig(
                fps=10.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                max_joint_acceleration=1.0,
                max_root_acceleration=1.0,
                max_root_jerk=1.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                max_start_end_root_speed=1000.0,
            ),
        )

        self.assertEqual(row["quality_action"], "quarantine")
        self.assertIn("g1_joint_acceleration_jump", row["quality_flags"])
        self.assertIn("g1_root_acceleration_jump", row["quality_flags"])
        self.assertIn("g1_root_jerk_jump", row["quality_flags"])

    def test_summarize_g1_rows_with_mjcf_contact_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            model = load_g1_kinematic_model(model_xml)

            sliding_rows = _csv_rows(
                first_joint_values=[0.0, 2.0, 2.0],
                root_x_values=[0.0, 0.50, 1.00],
                root_z_values=[0.0, 0.0, 0.0],
            )
            config = G1QualityConfig(
                fps=30.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                model_xml=model_xml,
                max_contact_slide_speed=0.25,
                max_mean_foot_clearance=0.10,
                max_penetration_depth=0.03,
                min_contact_frame_ratio=0.05,
                max_start_end_root_speed=1000.0,
            )

            row = summarize_g1_rows({"row_index": "1"}, sliding_rows, config, model=model)

            self.assertEqual(row["quality_mode"], "mjcf_fk")
            self.assertIn("g1_foot_slide", row["quality_flags"])
            self.assertIn("g1_joint_limit_violation", row["quality_flags"])
            self.assertGreater(row["contact_slide_rate"], 0.0)
            self.assertGreater(row["joint_limit_violation_rate"], 0.0)

    def test_summarize_g1_rows_records_fk_support_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            model = load_g1_kinematic_model(model_xml)
            config = G1QualityConfig(
                fps=30.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                model_xml=model_xml,
                max_contact_slide_speed=1000.0,
                max_mean_foot_clearance=1.0,
                max_penetration_depth=1.0,
                min_contact_frame_ratio=0.0,
                max_start_end_root_speed=1000.0,
            )

            row = summarize_g1_rows(
                {"row_index": "support"},
                _csv_rows(
                    [0.0, 0.0, 0.0],
                    root_x_values=[0.0, 0.0, 0.0],
                    root_z_values=[0.0, 0.0, 0.0],
                ),
                config,
                model=model,
            )

            self.assertEqual(row["quality_mode"], "mjcf_fk")
            self.assertEqual(row["root_height_min"], 0.0)
            self.assertEqual(row["root_height_max"], 0.0)
            self.assertEqual(row["root_height_range"], 0.0)
            self.assertEqual(row["support_frame_ratio"], 1.0)
            self.assertEqual(row["mean_root_support_distance"], 0.0)
            self.assertEqual(row["max_root_support_distance"], 0.0)

    def test_summarize_g1_rows_flags_fk_float_and_penetration(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            model = load_g1_kinematic_model(model_xml)
            config = G1QualityConfig(
                fps=30.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                model_xml=model_xml,
                max_contact_slide_speed=1000.0,
                max_mean_foot_clearance=0.10,
                max_penetration_depth=0.03,
                min_contact_frame_ratio=0.05,
                max_start_end_root_speed=1000.0,
            )

            floating = summarize_g1_rows(
                {"row_index": "2"},
                _csv_rows([0.0, 0.0, 0.0], root_x_values=[0.0, 0.0, 0.0], root_z_values=[0.4, 0.4, 0.4]),
                config,
                model=model,
            )
            penetrating = summarize_g1_rows(
                {"row_index": "3"},
                _csv_rows([0.0, 0.0, 0.0], root_x_values=[0.0, 0.0, 0.0], root_z_values=[-0.1, -0.1, -0.1]),
                config,
                model=model,
            )

            self.assertIn("g1_low_foot_contact", floating["quality_flags"])
            self.assertIn("g1_foot_float", floating["quality_flags"])
            self.assertIn("g1_ground_penetration", penetrating["quality_flags"])

    def test_summarize_g1_rows_marks_contact_correction_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_minimal_g1_mjcf(), encoding="utf-8")
            model = load_g1_kinematic_model(model_xml)
            config = G1QualityConfig(
                fps=30.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                model_xml=model_xml,
                max_contact_slide_speed=1000.0,
                max_mean_foot_clearance=0.05,
                max_penetration_depth=0.03,
                max_contact_correction_offset=0.10,
                min_contact_frame_ratio=0.0,
                max_start_end_root_speed=1000.0,
            )

            row = summarize_g1_rows(
                {"row_index": "repairable-float"},
                _csv_rows([0.0, 0.0, 0.0], root_x_values=[0.0, 0.0, 0.0], root_z_values=[0.08, 0.08, 0.08]),
                config,
                model=model,
            )

            self.assertIn("g1_foot_float", row["quality_flags"])
            self.assertEqual(row["contact_correction_candidate"], 1.0)
            self.assertEqual(row["contact_correction_reason"], "vertical_float_offset")
            self.assertEqual(row["contact_correction_offset"], -0.08)
            self.assertEqual(row["contact_correction_abs_offset"], 0.08)

    def test_summarize_g1_rows_flags_self_collision_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / "g1.xml"
            model_xml.write_text(_self_collision_proxy_mjcf(), encoding="utf-8")
            model = load_g1_kinematic_model(model_xml, foot_body_names=("left_foot", "right_foot"))
            config = G1QualityConfig(
                fps=30.0,
                max_joint_velocity=1000.0,
                max_root_speed=1000.0,
                root_position_scale=1.0,
                joint_angle_scale=1.0,
                root_rotation_scale=1.0,
                model_xml=model_xml,
                max_contact_slide_speed=1000.0,
                max_mean_foot_clearance=1.0,
                max_penetration_depth=1.0,
                min_contact_frame_ratio=0.0,
                max_start_end_root_speed=1000.0,
                min_self_collision_distance=0.02,
                min_self_collision_kinematic_hops=2,
            )

            row = summarize_g1_rows(
                {"row_index": "4"},
                _csv_rows([0.0, 0.0, 0.0], root_x_values=[0.0, 0.0, 0.0], root_z_values=[0.0, 0.0, 0.0]),
                config,
                model=model,
            )

            self.assertIn("g1_self_collision_proxy", row["quality_flags"])
            self.assertEqual(row["quality_action"], "quarantine")
            self.assertGreater(row["self_collision_proxy_rate"], 0.0)
            self.assertLess(row["min_self_collision_distance"], 0.02)


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
        _add_member(tar, "g1/csv/240101/good.csv", _csv_text([0.0, 0.01, 0.02], 0.001))
        _add_member(tar, "g1/csv/240101/jump.csv", _csv_text([0.0, 2.0, 2.1], 1.0))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _csv_text(first_joint_values: list[float], root_step: float) -> str:
    rows = _csv_rows(
        first_joint_values=first_joint_values,
        root_x_values=[frame * root_step for frame in range(len(first_joint_values))],
        root_z_values=[1.0 for _ in first_joint_values],
    )
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=G1_CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _csv_rows(
    first_joint_values: list[float],
    root_x_values: list[float],
    root_z_values: list[float],
) -> list[dict[str, str]]:
    rows = []
    for frame, first_joint in enumerate(first_joint_values):
        row = {column: "0.0" for column in G1_CSV_COLUMNS}
        row.update(
            {
                "Frame": str(frame),
                "root_translateX": str(root_x_values[frame]),
                "root_translateY": "0.0",
                "root_translateZ": str(root_z_values[frame]),
            }
        )
        row[G1_JOINT_COLUMNS[0]] = str(first_joint)
        rows.append(row)
    return rows


def _minimal_g1_mjcf() -> str:
    return """<mujoco model="minimal_g1">
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <freejoint name="pelvis"/>
      <body name="left_ankle_roll_link" pos="0 0 0">
        <joint name="left_hip_pitch_joint" axis="0 1 0" range="-1 1"/>
        <geom pos="0 0 0"/>
        <body name="left_toe_link" pos="0.1 0 0"/>
      </body>
      <body name="right_ankle_roll_link" pos="0 -0.1 0">
        <geom pos="0 0 0"/>
        <body name="right_toe_link" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _self_collision_proxy_mjcf() -> str:
    return """<mujoco model="self_collision_proxy">
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <freejoint name="pelvis"/>
      <body name="left_upper" pos="0 0 0">
        <geom pos="0 0 0"/>
        <body name="left_mid" pos="0 0 0">
          <body name="left_foot" pos="0 0 0">
            <geom pos="0 0 0"/>
          </body>
        </body>
      </body>
      <body name="right_upper" pos="0 0 0">
        <geom pos="0 0 0"/>
        <body name="right_mid" pos="0 0 0">
          <body name="right_foot" pos="0 0 0">
            <geom pos="0 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    unittest.main()
