import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
from online_retarget.data.quality_smoke import QualitySmokeConfig, run_quality_smoke


HEADER = [
    "move_name",
    "filename",
    "move_duration_frames",
    "package",
    "category",
    "is_neutral",
    "is_mirror",
    "move_soma_uniform_path",
    "move_soma_uniform_shape_path",
    "move_soma_proportional_path",
    "move_soma_proportional_shape_path",
    "move_g1_path",
    "take_name",
    "take_actor",
    "take_org_name",
    "take_date",
    "take_day_part",
    "content_name",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
    "content_all_rigplay_styles",
    "content_uniform_style",
    "content_type_of_movement",
    "content_body_position",
    "content_horizontal_move",
    "content_vertical_move",
    "content_props",
    "content_complex_action",
    "content_repeated_action",
    "actor_uid",
    "actor_height",
    "actor_height_cm",
    "actor_foot_cm",
    "actor_collarbone_height_cm",
    "actor_collarbone_span_cm",
    "actor_elbow_span_cm",
    "actor_wrist_span_cm",
    "actor_shoulder_span_cm",
    "actor_hips_height_cm",
    "actor_hips_bones_span_cm",
    "actor_knee_height_cm",
    "actor_ankle_height_cm",
    "actor_weight_kg",
    "actor_age_yr",
    "actor_gender",
    "actor_profession",
]


class QualitySmokeTests(unittest.TestCase):
    def test_run_quality_smoke_writes_traceable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            _write_metadata(root)
            _write_source_tar(root / "soma_proportional.tar")
            _write_g1_tar(root / "g1.tar")

            result = run_quality_smoke(
                data_root=root,
                output_root=output,
                config=QualitySmokeConfig(
                    run_name="fixture_quality_smoke",
                    limit=2,
                    sample_by=("category",),
                    frame_stride=1,
                    max_frames=None,
                    review_max_per_family=1,
                ),
            )

            report = json.loads(result.smoke_report_json.read_text(encoding="utf-8"))
            curated = json.loads(result.curated_report_json.read_text(encoding="utf-8"))
            preflight = json.loads(result.policy_preflight_json.read_text(encoding="utf-8"))
            review_template = result.review_decision_template_csv.read_text(encoding="utf-8")
            artifacts_exist = {
                "source": result.source_stats_jsonl.exists(),
                "source_fk": result.source_fk_stats_jsonl.exists(),
                "g1": result.g1_stats_jsonl.exists(),
                "pair": result.pair_stats_jsonl.exists(),
                "curated": result.curated_index_csv.exists(),
                "review": result.review_manifest_jsonl.exists(),
                "thresholds": all(path.exists() for path in result.threshold_proposal_jsons),
            }

        self.assertTrue(artifacts_exist["source"])
        self.assertTrue(artifacts_exist["source_fk"])
        self.assertTrue(artifacts_exist["g1"])
        self.assertTrue(artifacts_exist["pair"])
        self.assertTrue(artifacts_exist["curated"])
        self.assertTrue(artifacts_exist["review"])
        self.assertEqual(len(result.threshold_proposal_jsons), 4)
        self.assertTrue(artifacts_exist["thresholds"])
        self.assertEqual(report["config"]["run_name"], "fixture_quality_smoke")
        self.assertEqual(report["source"]["scanned_rows"], 2)
        self.assertEqual(report["g1"]["scanned_rows"], 2)
        self.assertEqual(curated["merged_source_rows"], 2)
        self.assertEqual(curated["merged_g1_rows"], 2)
        self.assertEqual(curated["merged_pair_rows"], 2)
        self.assertFalse(result.promotable)
        self.assertIn("threshold proposals have not been explicitly accepted", "\n".join(result.blockers))
        self.assertEqual(preflight["audit"]["status"], "blocked")
        self.assertIn("review_id", review_template)


def _write_metadata(root: Path) -> None:
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    rows = [
        _metadata_row("good", "A001", category="Baseline"),
        _metadata_row("jump", "A002", category="Jump"),
        _metadata_row("mirror", "A003", category="Baseline", is_mirror="True"),
    ]
    with (metadata / "seed_metadata_v003.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_row(name: str, actor_uid: str, category: str, is_mirror: str = "False") -> dict[str, str]:
    row = {key: "" for key in HEADER}
    row.update(
        {
            "move_name": name,
            "filename": name,
            "move_duration_frames": "3",
            "package": "Locomotion",
            "category": category,
            "is_neutral": "1.0",
            "is_mirror": is_mirror,
            "move_soma_proportional_path": f"soma_proportional/bvh/{name}.bvh",
            "move_soma_proportional_shape_path": (
                f"soma_shapes/soma_proportion_fit_mhr_params/{actor_uid}.npz"
            ),
            "move_g1_path": f"g1/csv/{name}.csv",
            "actor_uid": actor_uid,
            "actor_height_cm": "170",
            "actor_foot_cm": "28",
            "actor_collarbone_height_cm": "140",
            "actor_collarbone_span_cm": "35",
            "actor_elbow_span_cm": "100",
            "actor_wrist_span_cm": "130",
            "actor_shoulder_span_cm": "170",
            "actor_hips_height_cm": "95",
            "actor_hips_bones_span_cm": "30",
            "actor_knee_height_cm": "50",
            "actor_ankle_height_cm": "10",
            "actor_weight_kg": "70",
            "actor_age_yr": "30",
            "actor_gender": "M",
        }
    )
    return row


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "soma_proportional/bvh/good.bvh", _bvh_text([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        _add_member(tar, "soma_proportional/bvh/jump.bvh", _bvh_text([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]))
        _add_member(tar, "soma_proportional/bvh/mirror.bvh", _bvh_text([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))


def _write_g1_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        _add_member(tar, "g1/csv/good.csv", _g1_csv([0.0, 0.01, 0.02], root_step=0.001))
        _add_member(tar, "g1/csv/jump.csv", _g1_csv([0.0, 2.0, 2.1], root_step=1.0))
        _add_member(tar, "g1/csv/mirror.csv", _g1_csv([0.0, 0.01, 0.02], root_step=0.001))


def _add_member(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _bvh_text(left_foot_x: list[float], left_foot_y: list[float]) -> str:
    rows = "\n".join(
        (
            "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 "
            f"{x:.6f} {y:.6f} 0.000000 "
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
      OFFSET 0.000000 0.000000 0.000000
      CHANNELS 3 Zrotation Yrotation Xrotation
    }}
  }}
}}
MOTION
Frames: {len(left_foot_x)}
Frame Time: 0.008333333333333333
{rows}
"""


def _g1_csv(first_joint_values: list[float], root_step: float) -> str:
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
