"""Read-only BONES-SEED metadata inventory helpers.

This module intentionally uses only the Python standard library so the repo can
inspect dataset structure before the full training environment is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable, Iterator


METADATA_RELATIVE_PATH = Path("metadata/seed_metadata_v003.csv")

G1_CSV_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]

G1_JOINT_COLUMNS = G1_CSV_COLUMNS[7:]

SKELETON_MEASURE_COLUMNS = [
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
]


@dataclass(frozen=True)
class ActorSkeleton:
    actor_uid: str
    shape_path: str
    gender: str
    weight_kg: float | None
    age_yr: float | None
    measurements_cm: dict[str, float | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InventorySummary:
    metadata_csv: str
    rows: int
    actors: int
    proportional_shape_paths: int
    mirrored_rows: int
    g1_missing: int
    soma_proportional_missing: int
    height_min_cm: float | None
    height_max_cm: float | None
    packages: dict[str, int]
    categories: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def metadata_csv_path(data_root: Path) -> Path:
    return data_root / METADATA_RELATIVE_PATH


def iter_metadata_rows(data_root: Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    path = metadata_csv_path(data_root)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def summarize_metadata(data_root: Path) -> InventorySummary:
    actors: set[str] = set()
    shape_paths: set[str] = set()
    packages: dict[str, int] = {}
    categories: dict[str, int] = {}
    heights: list[float] = []
    rows = 0
    mirrored_rows = 0
    g1_missing = 0
    soma_proportional_missing = 0

    for row in iter_metadata_rows(data_root):
        rows += 1
        actors.add(row["actor_uid"])
        shape_paths.add(row["move_soma_proportional_shape_path"])
        mirrored_rows += row["is_mirror"].lower() == "true"
        g1_missing += not row.get("move_g1_path")
        soma_proportional_missing += not row.get("move_soma_proportional_path")
        _increment(packages, row["package"])
        _increment(categories, row["category"])
        height = _maybe_float(row.get("actor_height_cm"))
        if height is not None:
            heights.append(height)

    return InventorySummary(
        metadata_csv=str(metadata_csv_path(data_root)),
        rows=rows,
        actors=len(actors),
        proportional_shape_paths=len(shape_paths),
        mirrored_rows=mirrored_rows,
        g1_missing=g1_missing,
        soma_proportional_missing=soma_proportional_missing,
        height_min_cm=min(heights) if heights else None,
        height_max_cm=max(heights) if heights else None,
        packages=dict(sorted(packages.items(), key=lambda item: (-item[1], item[0]))),
        categories=dict(sorted(categories.items(), key=lambda item: (-item[1], item[0]))),
    )


def actor_skeletons(data_root: Path) -> list[ActorSkeleton]:
    by_actor: dict[str, ActorSkeleton] = {}
    for row in iter_metadata_rows(data_root):
        actor_uid = row["actor_uid"]
        if actor_uid in by_actor:
            continue
        by_actor[actor_uid] = ActorSkeleton(
            actor_uid=actor_uid,
            shape_path=row["move_soma_proportional_shape_path"],
            gender=row["actor_gender"],
            weight_kg=_maybe_float(row.get("actor_weight_kg")),
            age_yr=_maybe_float(row.get("actor_age_yr")),
            measurements_cm={key: _maybe_float(row.get(key)) for key in SKELETON_MEASURE_COLUMNS},
        )
    return sorted(by_actor.values(), key=lambda item: item.actor_uid)


def split_actor_uids(actor_items: Iterable[ActorSkeleton], train: float, val: float) -> dict[str, list[str]]:
    actor_uids = [item.actor_uid for item in actor_items]
    train_end = int(len(actor_uids) * train)
    val_end = train_end + int(len(actor_uids) * val)
    return {
        "train": actor_uids[:train_end],
        "val": actor_uids[train_end:val_end],
        "test": actor_uids[val_end:],
    }


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
