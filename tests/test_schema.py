import csv
from pathlib import Path
import tempfile
import unittest

from online_retarget.data.bones_seed import G1_JOINT_COLUMNS, SKELETON_MEASURE_COLUMNS
from online_retarget.data.schema import (
    MORPHOLOGY_NUMERIC_COLUMNS,
    ObservationSpec,
    OutputSpec,
    iter_motion_pair_refs,
    motion_pair_ref_from_index_row,
)


class SchemaTests(unittest.TestCase):
    def test_observation_and_output_dims(self):
        observation = ObservationSpec(history_frames=8, source_body_count=30)
        output = OutputSpec()

        self.assertEqual(observation.source_feature_dim(), 8 * 30 * 3 * 2)
        self.assertEqual(observation.morphology_dim(), len(MORPHOLOGY_NUMERIC_COLUMNS))
        self.assertEqual(observation.robot_state_dim(), len(G1_JOINT_COLUMNS) * 3 + 4 + 3)
        self.assertEqual(output.output_dim(), len(G1_JOINT_COLUMNS))
        self.assertEqual(output.target, "g1_joint_position")

    def test_motion_pair_ref_from_index_row(self):
        row = _index_row()
        ref = motion_pair_ref_from_index_row(row, index_csv=Path("runs/index.csv"))

        self.assertEqual(ref.sample_id, "train:A001:motion_a:7")
        self.assertEqual(ref.actor_uid, "A001")
        self.assertTrue(ref.is_mirror)
        self.assertEqual(ref.quality_flags, ("mirror_variant", "joint_velocity_jump"))
        self.assertEqual(ref.morphology["actor_height_cm"], 170.0)
        self.assertEqual(ref.provenance["index_csv"], "runs/index.csv")

    def test_iter_motion_pair_refs_filters_by_split_and_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.csv"
            rows = [
                _index_row(row_index="1", split="train", action="keep"),
                _index_row(row_index="2", split="val", action="keep"),
                _index_row(row_index="3", split="train", action="quarantine"),
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            refs = list(iter_motion_pair_refs(path, splits=("train",), actions=("keep",)))

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].provenance["row_index"], "1")

    def test_iter_motion_pair_refs_supports_action_column_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.csv"
            row = _index_row(row_index="1", split="train", action="keep")
            row["merged_quality_action"] = "quarantine"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            refs = list(
                iter_motion_pair_refs(
                    path,
                    splits=("train",),
                    actions=("keep",),
                    action_column="merged_quality_action",
                )
            )

        self.assertEqual(refs, [])


def _index_row(row_index: str = "7", split: str = "train", action: str = "downweight") -> dict[str, str]:
    row = {
        "row_index": row_index,
        "split": split,
        "actor_uid": "A001",
        "move_name": "motion_a",
        "filename": "motion_a",
        "package": "Locomotion",
        "category": "Baseline",
        "is_mirror": "True",
        "move_soma_proportional_path": "soma_proportional/bvh/240101/motion_a.bvh",
        "move_soma_proportional_shape_path": "soma_shapes/A001.npz",
        "move_g1_path": "g1/csv/240101/motion_a.csv",
        "curation_action": action,
        "quality_flags": "mirror_variant|joint_velocity_jump",
        "actor_weight_kg": "70",
        "actor_age_yr": "30",
        "actor_gender": "M",
    }
    for column in SKELETON_MEASURE_COLUMNS:
        row[column] = "170" if column == "actor_height_cm" else "1.0"
    return row


if __name__ == "__main__":
    unittest.main()
