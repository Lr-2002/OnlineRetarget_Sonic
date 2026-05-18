from __future__ import annotations

import csv
import struct
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from online_retarget.data.bones_sonic import (
    SONIC_BODY_NAMES,
    SONIC_JOINT_NAMES,
    build_sonic_index,
    inspect_sonic_npz,
)


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
    if dtype == "<i8":
        item = struct.pack("<q", int(fill))
        item_size = 8
    elif dtype == "<f4":
        item = struct.pack("<f", float(fill))
        item_size = 4
    else:
        raise ValueError(dtype)
    count = 1
    for dim in shape:
        count *= dim
    return b"\x93NUMPY" + b"\x01\x00" + struct.pack("<H", len(header_payload)) + header_payload + item * count


def _write_npz(path: Path, frames: int = 5) -> None:
    arrays = {
        "fps.npy": _npy_bytes("<i8", (1,), 50),
        "joint_pos.npy": _npy_bytes("<f4", (frames, 29), 0.1),
        "joint_vel.npy": _npy_bytes("<f4", (frames, 29), 0.0),
        "body_pos_w.npy": _npy_bytes("<f4", (frames, 30, 3), 0.0),
        "body_quat_w.npy": _npy_bytes("<f4", (frames, 30, 4), 0.0),
        "body_lin_vel_w.npy": _npy_bytes("<f4", (frames, 30, 3), 0.0),
        "body_ang_vel_w.npy": _npy_bytes("<f4", (frames, 30, 3), 0.0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zip_file:
        for name, payload in arrays.items():
            zip_file.writestr(name, payload)


class BonesSonicTest(unittest.TestCase):
    def test_sonic_orders_match_isaaclab_g1_reference(self) -> None:
        self.assertEqual(SONIC_BODY_NAMES[0], "pelvis")
        self.assertEqual(SONIC_BODY_NAMES[1], "left_hip_pitch_link")
        self.assertEqual(SONIC_BODY_NAMES[2], "right_hip_pitch_link")
        self.assertEqual(SONIC_BODY_NAMES[3], "waist_yaw_link")
        self.assertEqual(SONIC_BODY_NAMES[9], "torso_link")
        self.assertEqual(SONIC_BODY_NAMES[18], "left_ankle_roll_link")
        self.assertEqual(SONIC_BODY_NAMES[19], "right_ankle_roll_link")
        self.assertEqual(SONIC_BODY_NAMES[28], "left_wrist_yaw_link")
        self.assertEqual(SONIC_BODY_NAMES[29], "right_wrist_yaw_link")

        self.assertEqual(SONIC_JOINT_NAMES[0], "left_hip_pitch_joint")
        self.assertEqual(SONIC_JOINT_NAMES[1], "right_hip_pitch_joint")
        self.assertEqual(SONIC_JOINT_NAMES[2], "waist_yaw_joint")
        self.assertEqual(SONIC_JOINT_NAMES[17], "left_ankle_roll_joint")
        self.assertEqual(SONIC_JOINT_NAMES[18], "right_ankle_roll_joint")
        self.assertEqual(SONIC_JOINT_NAMES[27], "left_wrist_yaw_joint")
        self.assertEqual(SONIC_JOINT_NAMES[28], "right_wrist_yaw_joint")

    def test_inspect_sonic_npz_reads_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "230101" / "walk__A001.npz"
            _write_npz(path, frames=7)

            arrays = inspect_sonic_npz(path)

            self.assertEqual(arrays["fps"].scalar_preview, 50)
            self.assertEqual(arrays["joint_pos"].shape, (7, 29))
            self.assertEqual(arrays["body_pos_w"].shape, (7, 30, 3))

    def test_build_sonic_index_maps_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sonic_root = root / "bones_sonic"
            npz_path = sonic_root / "230101" / "walk__A001_M.npz"
            _write_npz(npz_path, frames=9)
            metadata_csv = root / "metadata.csv"
            with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "move_g1_path",
                        "actor_uid",
                        "is_mirror",
                        "package",
                        "category",
                        "move_duration_frames",
                        "move_soma_proportional_path",
                        "move_soma_proportional_shape_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "move_g1_path": "g1/csv/230101/walk__A001_M.csv",
                        "actor_uid": "A001",
                        "is_mirror": "True",
                        "package": "Locomotion",
                        "category": "Baseline",
                        "move_duration_frames": "9",
                        "move_soma_proportional_path": "soma_proportional/bvh/230101/walk__A001_M.bvh",
                        "move_soma_proportional_shape_path": "soma_shapes/A001.npz",
                    }
                )

            result = build_sonic_index(
                sonic_root=sonic_root,
                metadata_csv=metadata_csv,
                output_root=root / "runs",
                run_name="test_sonic",
            )

            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.schema_status_counts, {"ok": 1})
            with result.index_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sonic_relative_path"], "230101/walk__A001_M.npz")
            self.assertEqual(rows[0]["metadata_found"], "True")
            self.assertEqual(rows[0]["frame_count"], "9")
            self.assertEqual(rows[0]["fps"], "50")


if __name__ == "__main__":
    unittest.main()
