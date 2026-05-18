"""Read-only BONES-SONIC NPZ inventory helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import ast
import csv
import json
from pathlib import Path
import re
import struct
import subprocess
from typing import Any, Mapping, Sequence
from zipfile import ZipFile


SONIC_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

SONIC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

SONIC_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

SONIC_ORDER_NOTE = (
    "BONES-SONIC NPZ joint_pos, joint_vel, body_pos_w, body_quat_w, "
    "body_lin_vel_w, and body_ang_vel_w use SONIC/IsaacLab G1 order. This was "
    "verified against the official gear_sonic G1_ISAACLAB_JOINTS/mapping tables "
    "and an FK-to-body_pos_w sanity check; it is not the legacy BONES-SEED CSV "
    "or MJCF pre-order."
)


@dataclass(frozen=True)
class NpyHeader:
    dtype: str
    shape: tuple[int, ...]
    fortran_order: bool
    scalar_preview: int | float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class SonicIndexResult:
    output_dir: Path
    index_csv: Path
    report_json: Path
    scanned_files: int
    schema_status_counts: dict[str, int]
    actor_count: int
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["index_csv"] = str(self.index_csv)
        payload["report_json"] = str(self.report_json)
        return payload


def build_sonic_index(
    *,
    sonic_root: Path,
    metadata_csv: Path,
    output_root: Path,
    run_name: str = "bones_sonic_index_v0",
    limit: int | None = None,
) -> SonicIndexResult:
    """Build a traceable BONES-SONIC NPZ index without reading raw tensors."""

    sonic_root = sonic_root.expanduser()
    metadata_csv = metadata_csv.expanduser()
    output_dir = output_root.expanduser() / "indices" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_sonic_rel = _metadata_by_sonic_relative_path(metadata_csv)
    files = sorted(path for path in sonic_root.glob("*/*.npz") if path.is_file())
    if limit is not None:
        files = files[:limit]

    rows = [
        _row_for_npz(path, sonic_root=sonic_root, metadata_by_sonic_rel=metadata_by_sonic_rel)
        for path in files
    ]
    index_csv = output_dir / "sonic_index.csv"
    report_json = output_dir / "sonic_index_report.json"
    _write_csv(index_csv, rows)

    schema_status_counts = Counter(str(row["schema_status"]) for row in rows)
    actor_uids = {str(row["actor_uid"]) for row in rows if row.get("actor_uid")}
    fps_counts = Counter(str(row["fps"]) for row in rows)
    report = {
        "sonic_root": str(sonic_root),
        "metadata_csv": str(metadata_csv),
        "index_csv": str(index_csv),
        "run_name": run_name,
        "limit": limit,
        "scanned_files": len(rows),
        "source_file_count": len(files) if limit is None else "",
        "required_keys": list(SONIC_REQUIRED_KEYS),
        "joint_names": list(SONIC_JOINT_NAMES),
        "body_names": list(SONIC_BODY_NAMES),
        "body_order_note": SONIC_ORDER_NOTE,
        "joint_order_note": SONIC_ORDER_NOTE,
        "schema_status_counts": dict(sorted(schema_status_counts.items())),
        "fps_counts": dict(sorted(fps_counts.items())),
        "actor_count": len(actor_uids),
        "mirror_count": sum(_is_true(row.get("is_mirror")) for row in rows),
        "metadata_found_count": sum(_is_true(row.get("metadata_found")) for row in rows),
        "frame_count_summary": _metric_summary(rows, "frame_count"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return SonicIndexResult(
        output_dir=output_dir,
        index_csv=index_csv,
        report_json=report_json,
        scanned_files=len(rows),
        schema_status_counts=dict(sorted(schema_status_counts.items())),
        actor_count=len(actor_uids),
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def inspect_sonic_npz(path: Path) -> dict[str, NpyHeader]:
    """Return array headers for a BONES-SONIC NPZ file."""

    arrays: dict[str, NpyHeader] = {}
    with ZipFile(path) as zip_file:
        for member in sorted(name for name in zip_file.namelist() if name.endswith(".npy")):
            key = member[:-4]
            with zip_file.open(member) as handle:
                arrays[key] = _read_npy_header(handle)
    return arrays


def _row_for_npz(
    path: Path,
    *,
    sonic_root: Path,
    metadata_by_sonic_rel: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    rel_path = path.relative_to(sonic_root).as_posix()
    date = path.parent.name
    filename = path.stem
    actor_uid = _actor_uid_from_filename(filename)
    metadata = metadata_by_sonic_rel.get(rel_path, {})
    try:
        arrays = inspect_sonic_npz(path)
        flags = _schema_flags(arrays)
    except Exception as exc:
        arrays = {}
        flags = [f"npz_header_error:{type(exc).__name__}"]

    fps = _scalar(arrays.get("fps"))
    joint_shape = _shape(arrays.get("joint_pos"))
    body_shape = _shape(arrays.get("body_pos_w"))
    frame_count = joint_shape[0] if joint_shape else (body_shape[0] if body_shape else "")
    return {
        "sonic_relative_path": rel_path,
        "sonic_path": str(path),
        "date": date,
        "filename": filename,
        "actor_uid": metadata.get("actor_uid") or actor_uid,
        "is_mirror": metadata.get("is_mirror") or str(filename.endswith("_M")),
        "metadata_found": str(bool(metadata)),
        "metadata_row_index": metadata.get("_metadata_row_index", ""),
        "package": metadata.get("package", ""),
        "category": metadata.get("category", ""),
        "move_duration_frames": metadata.get("move_duration_frames", ""),
        "source_soma_proportional_path": metadata.get("move_soma_proportional_path", ""),
        "source_soma_proportional_shape_path": metadata.get("move_soma_proportional_shape_path", ""),
        "legacy_g1_csv_path": metadata.get("move_g1_path", ""),
        "fps": fps if fps is not None else "",
        "frame_count": frame_count,
        "joint_count": joint_shape[1] if len(joint_shape) >= 2 else "",
        "body_count": body_shape[1] if len(body_shape) >= 2 else "",
        "joint_pos_shape": _shape_string(joint_shape),
        "joint_vel_shape": _shape_string(_shape(arrays.get("joint_vel"))),
        "body_pos_w_shape": _shape_string(body_shape),
        "body_quat_w_shape": _shape_string(_shape(arrays.get("body_quat_w"))),
        "body_lin_vel_w_shape": _shape_string(_shape(arrays.get("body_lin_vel_w"))),
        "body_ang_vel_w_shape": _shape_string(_shape(arrays.get("body_ang_vel_w"))),
        "schema_status": "ok" if not flags else "invalid",
        "schema_flags": "|".join(flags),
    }


def _metadata_by_sonic_relative_path(metadata_csv: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not metadata_csv.exists():
        return rows
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            sonic_rel = _sonic_relative_from_metadata_row(row)
            if sonic_rel:
                payload = dict(row)
                payload["_metadata_row_index"] = str(index)
                rows[sonic_rel] = payload
    return rows


def _sonic_relative_from_metadata_row(row: Mapping[str, str]) -> str:
    g1_path = row.get("move_g1_path", "")
    if not g1_path:
        return ""
    path = Path(g1_path)
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "g1" and parts[1] == "csv":
        return (Path(parts[2]) / Path(parts[3]).with_suffix(".npz")).as_posix()
    return ""


def _read_npy_header(handle: Any) -> NpyHeader:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not an NPY payload")
    major = handle.read(1)[0]
    handle.read(1)
    if major == 1:
        header_len = struct.unpack("<H", handle.read(2))[0]
    elif major in {2, 3}:
        header_len = struct.unpack("<I", handle.read(4))[0]
    else:
        raise ValueError(f"unsupported NPY version: {major}")
    header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
    if not isinstance(header, dict):
        raise ValueError("NPY header is not a dict")
    dtype = str(header.get("descr", ""))
    shape = tuple(int(dim) for dim in header.get("shape", ()) or ())
    preview = handle.read(_dtype_size(dtype))
    return NpyHeader(
        dtype=dtype,
        shape=shape,
        fortran_order=bool(header.get("fortran_order")),
        scalar_preview=_decode_scalar_preview(dtype, preview) if shape in {(), (1,)} else None,
    )


def _schema_flags(arrays: Mapping[str, NpyHeader]) -> list[str]:
    flags: list[str] = []
    for key in SONIC_REQUIRED_KEYS:
        if key not in arrays:
            flags.append(f"missing_{key}")
    if _shape(arrays.get("joint_pos"))[-1:] != (29,):
        flags.append("joint_pos_not_29dof")
    if _shape(arrays.get("joint_vel"))[-1:] != (29,):
        flags.append("joint_vel_not_29dof")
    if _shape(arrays.get("body_pos_w"))[-2:] != (30, 3):
        flags.append("body_pos_w_not_30x3")
    if _shape(arrays.get("body_quat_w"))[-2:] != (30, 4):
        flags.append("body_quat_w_not_30x4")
    frame_counts = {
        shape[0]
        for key in SONIC_REQUIRED_KEYS
        for shape in (_shape(arrays.get(key)),)
        if key != "fps" and shape
    }
    if len(frame_counts) > 1:
        flags.append("frame_count_mismatch")
    return flags


def _actor_uid_from_filename(filename: str) -> str:
    match = re.search(r"__(A\d+)(?:_M)?$", filename)
    return match.group(1) if match else ""


def _shape(header: NpyHeader | None) -> tuple[int, ...]:
    return header.shape if header is not None else ()


def _scalar(header: NpyHeader | None) -> int | float | None:
    return header.scalar_preview if header is not None else None


def _shape_string(shape: Sequence[int]) -> str:
    return "x".join(str(item) for item in shape)


def _dtype_size(dtype: str) -> int:
    return {"<i8": 8, "|i8": 8, "<u8": 8, "<i4": 4, "|i4": 4, "<u4": 4, "<f8": 8, "<f4": 4}.get(dtype, 0)


def _decode_scalar_preview(dtype: str, preview: bytes) -> int | float | None:
    formats = {
        "<i8": "<q",
        "|i8": "<q",
        "<u8": "<Q",
        "<i4": "<i",
        "|i4": "<i",
        "<u4": "<I",
        "<f8": "<d",
        "<f4": "<f",
    }
    fmt = formats.get(dtype)
    if not fmt:
        return None
    size = struct.calcsize(fmt)
    if len(preview) < size:
        return None
    value = struct.unpack(fmt, preview[:size])[0]
    return round(float(value), 6) if isinstance(value, float) else int(value)


def _metric_summary(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, float]:
    values = sorted(_float(row.get(metric)) for row in rows if row.get(metric) not in (None, ""))
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(values[0], 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(values[-1], 6),
    }


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except Exception:
        return False
