"""Actor/skeleton morphology features for SONIC-native retargeting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping


MEASUREMENT_COLUMNS = (
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
)
MORPHOLOGY_VECTOR_DIM = 1 + len(MEASUREMENT_COLUMNS) * 2 + 1


class MorphologyError(ValueError):
    """Raised when morphology lookup or parsing fails."""


@dataclass(frozen=True)
class ActorMorphology:
    actor_uid: str
    skeleton_id: str
    height_m: float
    measurements_m: tuple[float, ...]
    proportions: tuple[float, ...]
    shape_path: str

    def as_source_features(self, *, num_clusters: int = 4) -> dict[str, Any]:
        """Return actor and skeleton morphology features.

        ``num_clusters`` is the source skeleton/morphology bucket count, not
        actuator grouping. It remains the legacy public key for compatibility.
        """

        return {
            "actor_uid": self.actor_uid,
            "skeleton_id": self.skeleton_id,
            "skeleton_cluster_id": stable_skeleton_cluster(self.skeleton_id, num_clusters),
            "height": self.height_m,
            "bone_lengths": self.measurements_m,
            "body_proportions": self.proportions,
            "foot_leg_arm_torso_measurements": self.measurements_m,
            "soma_morphology": self.as_vector(num_clusters=num_clusters),
            "shape_path": self.shape_path,
        }

    def as_vector(self, *, num_clusters: int = 4) -> tuple[float, ...]:
        """Return the fixed numeric vector consumed by the SONIC encoder.

        Layout:
        ``height_m`` + 11 absolute measurements in meters + 11 height-normalized
        proportions + normalized deterministic skeleton cluster.
        """

        cluster = stable_skeleton_cluster(self.skeleton_id, num_clusters)
        cluster_norm = 0.0 if num_clusters <= 1 else cluster / float(num_clusters - 1)
        return (self.height_m, *self.measurements_m, *self.proportions, cluster_norm)


def load_morphology_table(registry_csv: str | Path) -> dict[str, ActorMorphology]:
    """Load actor morphology rows from a skeleton registry CSV."""

    table: dict[str, ActorMorphology] = {}
    with Path(registry_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            actor_uid = str(row.get("actor_uid") or "").strip()
            if not actor_uid:
                continue
            table[actor_uid] = morphology_from_registry_row(row)
    if not table:
        raise MorphologyError(f"no actor morphology rows loaded from {registry_csv}")
    return table


def morphology_from_registry_row(row: Mapping[str, Any]) -> ActorMorphology:
    actor_uid = _required_text(row, "actor_uid")
    skeleton_id = str(row.get("encoder_id") or actor_uid)
    measurements_m = tuple(_cm_to_m(_required_float(row, key)) for key in MEASUREMENT_COLUMNS)
    height_m = _cm_to_m(_required_float(row, "actor_height_cm"))
    if height_m <= 0:
        raise MorphologyError(f"invalid actor height for {actor_uid}: {height_m}")
    proportions = tuple(value / height_m for value in measurements_m)
    return ActorMorphology(
        actor_uid=actor_uid,
        skeleton_id=skeleton_id,
        height_m=height_m,
        measurements_m=measurements_m,
        proportions=proportions,
        shape_path=str(row.get("shape_path") or row.get("shape_abs_path") or ""),
    )


def stable_skeleton_cluster(skeleton_id: str, num_clusters: int) -> int:
    """Hash a source skeleton id into a deterministic morphology bucket.

    ``num_clusters`` is the source skeleton/morphology bucket count, not actuator grouping.
    The bucket id is used only as a morphology feature.
    """

    if num_clusters <= 0:
        raise MorphologyError("num_clusters must be positive")
    digest = hashlib.sha256(skeleton_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_clusters


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise MorphologyError(f"missing required text field: {key}")
    return value


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MorphologyError(f"missing or invalid numeric field: {key}") from exc
    if result <= 0:
        raise MorphologyError(f"non-positive numeric field: {key}={result}")
    return result


def _cm_to_m(value: float) -> float:
    return value / 100.0
