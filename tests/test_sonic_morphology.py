import csv
import tempfile
import unittest
from pathlib import Path

from online_retarget.sonic_morphology import (
    MEASUREMENT_COLUMNS,
    MORPHOLOGY_VECTOR_DIM,
    MorphologyError,
    load_morphology_table,
    morphology_from_registry_row,
    stable_skeleton_cluster,
)


class SonicMorphologyTests(unittest.TestCase):
    def test_morphology_row_exports_formal_source_feature_keys(self):
        morphology = morphology_from_registry_row(_row(actor_uid="A001", height_cm=170.0))
        features = morphology.as_source_features(num_clusters=4)

        self.assertEqual(features["actor_uid"], "A001")
        self.assertEqual(features["skeleton_id"], "A001")
        self.assertEqual(features["height"], 1.7)
        self.assertEqual(len(features["bone_lengths"]), len(MEASUREMENT_COLUMNS))
        self.assertEqual(len(features["body_proportions"]), len(MEASUREMENT_COLUMNS))
        self.assertEqual(len(features["soma_morphology"]), MORPHOLOGY_VECTOR_DIM)
        self.assertIn(features["skeleton_cluster_id"], {0, 1, 2, 3})

    def test_stable_skeleton_cluster_is_deterministic(self):
        self.assertEqual(
            stable_skeleton_cluster("A123", 8),
            stable_skeleton_cluster("A123", 8),
        )

    def test_load_morphology_table_from_registry_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.csv"
            rows = [_row(actor_uid="A001"), _row(actor_uid="A002")]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            table = load_morphology_table(path)

        self.assertEqual(set(table), {"A001", "A002"})
        self.assertEqual(table["A002"].skeleton_id, "A002")

    def test_missing_measurement_is_rejected(self):
        row = _row()
        row["actor_foot_cm"] = ""

        with self.assertRaisesRegex(MorphologyError, "actor_foot_cm"):
            morphology_from_registry_row(row)


def _row(actor_uid: str = "A001", height_cm: float = 170.0) -> dict[str, object]:
    row: dict[str, object] = {
        "actor_uid": actor_uid,
        "encoder_id": actor_uid,
        "shape_path": f"soma_shapes/soma_proportion_fit_mhr_params/{actor_uid}.npz",
    }
    for index, key in enumerate(MEASUREMENT_COLUMNS):
        row[key] = height_cm if key == "actor_height_cm" else 20.0 + index
    return row
