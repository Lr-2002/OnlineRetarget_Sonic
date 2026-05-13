"""Typed dataset records and observation/output contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .bones_seed import G1_JOINT_COLUMNS, SKELETON_MEASURE_COLUMNS


MORPHOLOGY_NUMERIC_COLUMNS = (
    *SKELETON_MEASURE_COLUMNS,
    "actor_weight_kg",
    "actor_age_yr",
)


@dataclass(frozen=True)
class MotionPairRef:
    """Row-level source/target pair reference from a split index."""

    sample_id: str
    split: str
    actor_uid: str
    package: str
    category: str
    is_mirror: bool
    source_motion_path: str
    source_shape_path: str
    target_g1_path: str
    morphology: dict[str, float | None]
    actor_gender: str
    quality_action: str
    quality_flags: tuple[str, ...]
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobotStateSpec:
    """Robot-state side channel expected by online inference."""

    joint_dim: int = len(G1_JOINT_COLUMNS)
    include_joint_position: bool = True
    include_joint_velocity: bool = True
    include_previous_action: bool = True
    include_imu_orientation: bool = True
    include_base_angular_velocity: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationSpec:
    """First trainable observation contract."""

    history_frames: int = 8
    source_body_count: int = 30
    source_position_dim: int = 3
    include_source_velocity: bool = True
    include_morphology: bool = True
    robot_state: RobotStateSpec = RobotStateSpec()

    def source_feature_dim(self) -> int:
        per_frame = self.source_body_count * self.source_position_dim
        if self.include_source_velocity:
            per_frame *= 2
        return self.history_frames * per_frame

    def morphology_dim(self) -> int:
        return len(MORPHOLOGY_NUMERIC_COLUMNS) if self.include_morphology else 0

    def robot_state_dim(self) -> int:
        spec = self.robot_state
        total = 0
        if spec.include_joint_position:
            total += spec.joint_dim
        if spec.include_joint_velocity:
            total += spec.joint_dim
        if spec.include_previous_action:
            total += spec.joint_dim
        if spec.include_imu_orientation:
            total += 4
        if spec.include_base_angular_velocity:
            total += 3
        return total

    def flattened_dim(self) -> int:
        return self.source_feature_dim() + self.morphology_dim() + self.robot_state_dim()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_feature_dim"] = self.source_feature_dim()
        payload["morphology_dim"] = self.morphology_dim()
        payload["robot_state_dim"] = self.robot_state_dim()
        payload["flattened_dim"] = self.flattened_dim()
        return payload


@dataclass(frozen=True)
class OutputSpec:
    """First baseline output contract."""

    target: str = "g1_joint_position_delta"
    joint_dim: int = len(G1_JOINT_COLUMNS)
    include_root: bool = False

    def output_dim(self) -> int:
        return self.joint_dim + (6 if self.include_root else 0)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dim"] = self.output_dim()
        return payload


def motion_pair_ref_from_index_row(row: Mapping[str, str], index_csv: Path | None = None) -> MotionPairRef:
    """Convert a split-index CSV row to a typed motion pair reference."""

    row_index = row.get("row_index", "")
    actor_uid = row.get("actor_uid", "")
    filename = row.get("filename", "")
    sample_id = f"{row.get('split', 'unknown')}:{actor_uid}:{filename}:{row_index}"
    return MotionPairRef(
        sample_id=sample_id,
        split=row.get("split", ""),
        actor_uid=actor_uid,
        package=row.get("package", ""),
        category=row.get("category", ""),
        is_mirror=_is_true(row.get("is_mirror")),
        source_motion_path=row.get("move_soma_proportional_path", ""),
        source_shape_path=row.get("move_soma_proportional_shape_path", ""),
        target_g1_path=row.get("move_g1_path", ""),
        morphology={column: _maybe_float(row.get(column)) for column in MORPHOLOGY_NUMERIC_COLUMNS},
        actor_gender=row.get("actor_gender", ""),
        quality_action=row.get("curation_action", ""),
        quality_flags=tuple(flag for flag in row.get("quality_flags", "").split("|") if flag),
        provenance={
            "index_csv": str(index_csv) if index_csv else "",
            "row_index": row_index,
            "filename": filename,
        },
    )


def iter_motion_pair_refs(
    index_csv: Path,
    splits: Sequence[str] = (),
    actions: Sequence[str] = ("keep", "downweight"),
    action_column: str = "curation_action",
) -> Iterable[MotionPairRef]:
    """Yield motion pair refs from a split index with split/action filters."""

    split_filter = set(splits)
    action_filter = set(actions)
    with index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split_filter and row.get("split") not in split_filter:
                continue
            if action_filter and row.get(action_column) not in action_filter:
                continue
            yield motion_pair_ref_from_index_row(row, index_csv=index_csv)


def _maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
