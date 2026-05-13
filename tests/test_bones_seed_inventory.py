import csv
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_JOINT_COLUMNS, actor_skeletons, summarize_metadata


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


class BonesSeedInventoryTests(unittest.TestCase):
    def test_summary_and_actor_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata"
            metadata.mkdir()
            path = metadata / "seed_metadata_v003.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                writer.writerow(_row("motion_a", "A001", "False"))
                writer.writerow(_row("motion_a_M", "A001", "True"))

            summary = summarize_metadata(root)
            actors = actor_skeletons(root)

            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.actors, 1)
            self.assertEqual(summary.mirrored_rows, 1)
            self.assertEqual(summary.g1_missing, 0)
            self.assertEqual(actors[0].actor_uid, "A001")
            self.assertEqual(actors[0].measurements_cm["actor_height_cm"], 170.0)

    def test_g1_joint_width(self):
        self.assertEqual(len(G1_JOINT_COLUMNS), 29)


def _row(name: str, actor_uid: str, is_mirror: str) -> dict[str, str]:
    row = {key: "" for key in HEADER}
    row.update(
        {
            "move_name": name,
            "filename": name,
            "move_duration_frames": "120",
            "package": "Locomotion",
            "category": "Baseline",
            "is_neutral": "1.0",
            "is_mirror": is_mirror,
            "move_soma_proportional_path": f"soma_proportional/bvh/240101/{name}.bvh",
            "move_soma_proportional_shape_path": f"soma_shapes/soma_proportion_fit_mhr_params/{actor_uid}.npz",
            "move_g1_path": f"g1/csv/240101/{name}.csv",
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
