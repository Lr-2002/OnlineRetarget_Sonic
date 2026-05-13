import csv
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.curation import (
    QualityPolicy,
    SplitConfig,
    assess_row_quality,
    build_split_index,
)


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


class BonesSeedIndexTests(unittest.TestCase):
    def test_quality_policy_assigns_actions(self):
        policy = QualityPolicy(min_duration_frames=60)

        good = assess_row_quality(_row("good", "A001"), policy)
        mirror = assess_row_quality(_row("mirror", "A001", is_mirror="True"), policy)
        short = assess_row_quality(_row("short", "A002", duration="24"), policy)
        missing_target = assess_row_quality(_row("missing", "A003", move_g1_path=""), policy)

        self.assertEqual(good.action, "keep")
        self.assertEqual(mirror.action, "downweight")
        self.assertIn("mirror_variant", mirror.flags)
        self.assertEqual(short.action, "quarantine")
        self.assertIn("short_clip", short.flags)
        self.assertEqual(missing_target.action, "exclude")
        self.assertIn("missing_move_g1_path", missing_target.flags)

    def test_build_split_index_writes_traceable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "runs"
            metadata = root / "metadata"
            metadata.mkdir(parents=True)
            metadata_path = metadata / "seed_metadata_v003.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                writer.writerow(_row("a_good", "A001"))
                writer.writerow(_row("a_mirror", "A001", is_mirror="True"))
                writer.writerow(_row("b_missing", "A002", move_g1_path=""))
                writer.writerow(_row("c_short", "A003", duration="20"))
                writer.writerow(_row("d_good", "A004"))

            result = build_split_index(
                data_root=root,
                output_root=output,
                split_config=SplitConfig(train_ratio=0.5, val_ratio=0.25, seed=7),
                quality_policy=QualityPolicy(min_duration_frames=60),
            )

            self.assertTrue(result.index_csv.exists())
            self.assertTrue(result.report_json.exists())
            self.assertTrue(result.manifest_json.exists())
            self.assertEqual(result.row_count, 5)
            self.assertEqual(result.actor_count, 4)
            self.assertEqual(result.action_counts["keep"], 2)
            self.assertEqual(result.action_counts["downweight"], 1)
            self.assertEqual(result.action_counts["quarantine"], 1)
            self.assertEqual(result.action_counts["exclude"], 1)
            self.assertEqual(result.flag_counts["mirror_variant"], 1)
            self.assertEqual(result.flag_counts["short_clip"], 1)
            self.assertEqual(result.flag_counts["missing_move_g1_path"], 1)

            rows = _read_index(result.index_csv)
            splits_by_actor: dict[str, set[str]] = {}
            for row in rows:
                if row["split"] == "excluded":
                    continue
                splits_by_actor.setdefault(row["actor_uid"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_actor.values()))

    def test_rejects_outputs_inside_read_only_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "metadata").mkdir(parents=True)
            with (root / "metadata" / "seed_metadata_v003.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                writer.writerow(_row("good", "A001"))

            with self.assertRaises(ValueError):
                build_split_index(root, root / "derived", SplitConfig(), QualityPolicy())


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(
    name: str,
    actor_uid: str,
    is_mirror: str = "False",
    duration: str = "120",
    move_g1_path: str | None = None,
) -> dict[str, str]:
    row = {key: "" for key in HEADER}
    row.update(
        {
            "move_name": name,
            "filename": name,
            "move_duration_frames": duration,
            "package": "Locomotion",
            "category": "Baseline",
            "is_neutral": "1.0",
            "is_mirror": is_mirror,
            "move_soma_proportional_path": f"soma_proportional/bvh/240101/{name}.bvh",
            "move_soma_proportional_shape_path": (
                f"soma_shapes/soma_proportion_fit_mhr_params/{actor_uid}.npz"
            ),
            "move_g1_path": (
                f"g1/csv/240101/{name}.csv" if move_g1_path is None else move_g1_path
            ),
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


if __name__ == "__main__":
    unittest.main()
