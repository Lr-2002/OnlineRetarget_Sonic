from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from online_retarget.data.bones_seed import SKELETON_MEASURE_COLUMNS
from online_retarget.data.skeleton_registry import build_skeleton_registry


def _npy_bytes(dtype: str, shape: tuple[int, ...], fill: int | float = 0) -> bytes:
    header = {
        "descr": dtype,
        "fortran_order": False,
        "shape": shape,
    }
    header_text = repr(header)
    prefix_len = 10
    padding = 16 - ((prefix_len + len(header_text) + 1) % 16)
    header_payload = (header_text + " " * padding + "\n").encode("latin1")
    if dtype == "<f4":
        item = struct.pack("<f", float(fill))
        item_size = 4
    elif dtype == "<i8":
        item = struct.pack("<q", int(fill))
        item_size = 8
    else:
        raise ValueError(dtype)
    count = 1
    for dim in shape:
        count *= dim
    return (
        b"\x93NUMPY"
        + b"\x01\x00"
        + struct.pack("<H", len(header_payload))
        + header_payload
        + item * count
    )


class SkeletonRegistryTests(unittest.TestCase):
    def test_build_registry_groups_by_actor_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape_path = root / "soma_shapes" / "soma_proportion_fit_mhr_params" / "A001.npz"
            shape_path.parent.mkdir(parents=True)
            with ZipFile(shape_path, "w", compression=ZIP_DEFLATED) as zip_file:
                zip_file.writestr("pose.npy", _npy_bytes("<f4", (52, 3), 0.0))
                zip_file.writestr("betas.npy", _npy_bytes("<f4", (16,), 0.1))

            index_csv = root / "curated_index.csv"
            rows = [
                _row("A001", split="train", action="keep"),
                _row("A001", split="val", action="downweight", mirror="True"),
                _row("A002", split="train", action="quarantine"),
            ]
            with index_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = build_skeleton_registry(
                index_csv=index_csv,
                data_root=root,
                output_root=root / "runs",
                run_name="test_registry",
            )

            self.assertEqual(result.actor_count, 1)
            self.assertEqual(result.clip_count, 2)
            self.assertEqual(result.shape_file_missing_count, 0)
            with result.registry_csv.open(newline="", encoding="utf-8") as handle:
                registry_rows = list(csv.DictReader(handle))
            self.assertEqual(registry_rows[0]["actor_uid"], "A001")
            self.assertEqual(registry_rows[0]["clip_count"], "2")
            self.assertEqual(registry_rows[0]["train_clip_count"], "1")
            self.assertEqual(registry_rows[0]["val_clip_count"], "1")
            self.assertEqual(registry_rows[0]["mirror_count"], "1")
            self.assertIn("betas:<f4:16", registry_rows[0]["shape_header_signature"])
            report = json.loads(result.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["all_actor_count"], 2)
            self.assertEqual(report["actor_count_excluded_by_action"], 1)
            self.assertEqual(report["rows_excluded_by_action"], 1)
            self.assertEqual(report["action_actor_counts"]["quarantine"], 1)


def _row(actor_uid: str, *, split: str, action: str, mirror: str = "False") -> dict[str, str]:
    row = {
        "row_index": "1",
        "split": split,
        "actor_uid": actor_uid,
        "move_name": f"walk__{actor_uid}",
        "filename": f"walk__{actor_uid}",
        "package": "Locomotion",
        "category": "Baseline",
        "is_mirror": mirror,
        "move_soma_proportional_path": f"soma_proportional/bvh/240101/walk__{actor_uid}.bvh",
        "move_soma_proportional_shape_path": (
            f"soma_shapes/soma_proportion_fit_mhr_params/{actor_uid}.npz"
        ),
        "move_g1_path": f"g1/csv/240101/walk__{actor_uid}.csv",
        "merged_quality_action": action,
        "curation_action": action,
        "quality_flags": "",
        "actor_weight_kg": "70",
        "actor_age_yr": "30",
        "actor_gender": "M",
    }
    for column in SKELETON_MEASURE_COLUMNS:
        row[column] = "170" if column == "actor_height_cm" else "1.0"
    return row


if __name__ == "__main__":
    unittest.main()
