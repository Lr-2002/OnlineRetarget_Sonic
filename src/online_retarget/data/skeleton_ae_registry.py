"""All-identity skeleton geometry registry for the Skeleton AE gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
from typing import Mapping, Sequence

from .windowed_builder import BVHMotion, parse_bvh_motion


SOMA_AE_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandMiddle1",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandMiddle1",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
SKELETON_GEOMETRY_DIM = len(SOMA_AE_JOINT_NAMES) * 4


@dataclass(frozen=True)
class SkeletonAERegistryResult:
    output_dir: Path
    registry_csv: Path
    report_json: Path
    skeleton_count: int
    train_skeleton_count: int
    validation_skeleton_count: int
    geometry_failure_count: int
    split_leakage_count: int
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["registry_csv"] = str(self.registry_csv)
        payload["report_json"] = str(self.report_json)
        return payload


def build_all_skeleton_ae_registry(
    *,
    index_csv: Path,
    data_root: Path,
    output_root: Path,
    run_name: str = "bones_sonic_all_skeleton_ae_registry_v0",
    source_tar: Path | None = None,
    skeleton_id_column: str = "actor_uid",
    validation_ratio: float = 0.1,
    seed: int = 2026053001,
    position_scale: float = 0.01,
    limit: int | None = None,
) -> SkeletonAERegistryResult:
    """Build one continuous 104D geometry row per available skeleton identity."""

    if not 0.0 < float(validation_ratio) < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    index_csv = index_csv.expanduser()
    data_root = data_root.expanduser()
    output_dir = output_root.expanduser() / "skeleton_ae_registry" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_csv = output_dir / "skeleton_ae_registry.csv"
    report_json = output_dir / "skeleton_ae_registry_report.json"

    groups, source_column_counts, skipped_source_rows = _group_index_rows(
        index_csv=index_csv,
        skeleton_id_column=skeleton_id_column,
        limit=limit,
    )
    split_by_id = deterministic_skeleton_split(
        sorted(groups),
        validation_ratio=validation_ratio,
        seed=seed,
    )

    rows: list[dict[str, object]] = []
    geometry_failures: list[dict[str, str]] = []
    tar_path = source_tar.expanduser() if source_tar is not None else data_root / "soma_proportional.tar"
    source_archive = tarfile.open(tar_path, "r:*") if tar_path.exists() else None
    try:
        for skeleton_id in sorted(groups):
            payload = groups[skeleton_id]
            geometry = None
            source_path = ""
            last_error = ""
            for candidate, _count in payload["source_paths"].most_common():
                source_path = candidate
                try:
                    bvh_text = _read_bvh_text(
                        source_path,
                        data_root=data_root,
                        source_archive=source_archive,
                    )
                    geometry = skeleton_geometry_from_bvh_text(
                        bvh_text,
                        position_scale=position_scale,
                    )
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
            if geometry is None:
                geometry_failures.append(
                    {
                        "skeleton_id": skeleton_id,
                        "source_path": source_path,
                        "error": last_error,
                    }
                )
                continue
            split = split_by_id[skeleton_id]
            rows.append(
                {
                    "actor_uid": skeleton_id,
                    "encoder_id": skeleton_id,
                    "split": split,
                    "clip_count": int(payload["clip_count"]),
                    "source_soma_proportional_path": source_path,
                    "source_path_count": len(payload["source_paths"]),
                    "source_shape_path": _top_key(payload["shape_paths"]),
                    "package_counts_json": json.dumps(
                        dict(sorted(payload["package_counts"].items())),
                        sort_keys=True,
                    ),
                    "category_counts_json": json.dumps(
                        dict(sorted(payload["category_counts"].items())),
                        sort_keys=True,
                    ),
                    "filename_examples_json": json.dumps(
                        sorted(payload["filenames"])[:5],
                        sort_keys=True,
                    ),
                    "move_name_examples_json": json.dumps(
                        sorted(payload["move_names"])[:5],
                        sort_keys=True,
                    ),
                    "geometry_source": "soma_proportional_bvh_hierarchy_rest_offsets",
                    "geometry_position_scale": position_scale,
                    "geometry_joint_count": len(SOMA_AE_JOINT_NAMES),
                    "geometry_joint_names_json": json.dumps(list(SOMA_AE_JOINT_NAMES)),
                    "geometry_offset_shape": "[26, 3]",
                    "geometry_length_shape": "[26]",
                    "geometry_shape": f"[{SKELETON_GEOMETRY_DIM}]",
                    "geometry_dim": SKELETON_GEOMETRY_DIM,
                    "geometry_json": json.dumps([round(value, 10) for value in geometry]),
                }
            )
    finally:
        if source_archive is not None:
            source_archive.close()

    train_ids = {str(row["actor_uid"]) for row in rows if row["split"] == "train"}
    val_ids = {str(row["actor_uid"]) for row in rows if row["split"] == "validation"}
    leakage = sorted(train_ids & val_ids)
    rows.sort(key=lambda row: str(row["actor_uid"]))
    _write_csv(registry_csv, rows)

    report = {
        "index_csv": str(index_csv),
        "data_root": str(data_root),
        "source_tar": str(tar_path) if tar_path.exists() else "",
        "output_dir": str(output_dir),
        "registry_csv": str(registry_csv),
        "report_json": str(report_json),
        "run_name": run_name,
        "skeleton_id_column": skeleton_id_column,
        "source_path_columns": list(_SOURCE_PATH_COLUMNS),
        "source_column_counts": dict(sorted(source_column_counts.items())),
        "rows_without_skeleton_or_source": skipped_source_rows,
        "unique_skeleton_count_seen": len(groups),
        "unique_skeleton_count": len(rows),
        "train_skeleton_count": len(train_ids),
        "validation_skeleton_count": len(val_ids),
        "validation_ratio": validation_ratio,
        "split_seed": seed,
        "split_key": "actor_uid/encoder_id",
        "split_leakage_count": len(leakage),
        "split_leakage_ids": leakage[:20],
        "geometry": {
            "shape": [SKELETON_GEOMETRY_DIM],
            "joint_count": len(SOMA_AE_JOINT_NAMES),
            "offset_shape": [26, 3],
            "length_shape": [26],
            "position_scale": position_scale,
            "source": "SOMA proportional BVH hierarchy rest offsets",
            "joint_names": list(SOMA_AE_JOINT_NAMES),
        },
        "geometry_failure_count": len(geometry_failures),
        "geometry_failure_examples": geometry_failures[:20],
        "contract": {
            "model_input": "continuous x_skel geometry only",
            "normalization": "trainer must fit normalization on train split only",
            "target": "same x_skel geometry for reconstruction",
            "not_model_inputs": ["actor_uid", "encoder_id", "source metadata", "split"],
        },
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return SkeletonAERegistryResult(
        output_dir=output_dir,
        registry_csv=registry_csv,
        report_json=report_json,
        skeleton_count=len(rows),
        train_skeleton_count=len(train_ids),
        validation_skeleton_count=len(val_ids),
        geometry_failure_count=len(geometry_failures),
        split_leakage_count=len(leakage),
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def deterministic_skeleton_split(
    skeleton_ids: Sequence[str],
    *,
    validation_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique = sorted({skeleton_id for skeleton_id in skeleton_ids if skeleton_id})
    if not unique:
        return {}
    if len(unique) == 1:
        return {unique[0]: "train"}
    val_count = int(round(len(unique) * float(validation_ratio)))
    val_count = max(1, min(len(unique) - 1, val_count))
    ranked = sorted(unique, key=lambda item: _stable_hash_int(f"{seed}:{item}"))
    val_ids = set(ranked[:val_count])
    return {skeleton_id: ("validation" if skeleton_id in val_ids else "train") for skeleton_id in unique}


def skeleton_geometry_from_bvh_text(text: str, *, position_scale: float = 0.01) -> list[float]:
    motion = parse_bvh_motion(text, max_frames=1)
    return skeleton_geometry_from_bvh_motion(motion, position_scale=position_scale)


def skeleton_geometry_from_bvh_motion(
    motion: BVHMotion,
    *,
    position_scale: float = 0.01,
) -> list[float]:
    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    missing = [name for name in SOMA_AE_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise ValueError(f"BVH missing required SOMA joints: {', '.join(missing)}")
    root_index = name_to_index["Hips"]
    rest_positions: list[tuple[float, float, float]] = []
    for joint in motion.joints:
        parent = rest_positions[joint.parent] if joint.parent is not None else (0.0, 0.0, 0.0)
        rest_positions.append(
            (
                parent[0] + joint.offset[0] * position_scale,
                parent[1] + joint.offset[1] * position_scale,
                parent[2] + joint.offset[2] * position_scale,
            )
        )
    root = rest_positions[root_index]
    offsets: list[float] = []
    lengths: list[float] = []
    for name in SOMA_AE_JOINT_NAMES:
        pos = rest_positions[name_to_index[name]]
        rel = (pos[0] - root[0], pos[1] - root[1], pos[2] - root[2])
        offsets.extend(rel)
        lengths.append(math.sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]))
    geometry = offsets + lengths
    if len(geometry) != SKELETON_GEOMETRY_DIM:
        raise ValueError(f"expected geometry dim {SKELETON_GEOMETRY_DIM}, got {len(geometry)}")
    return geometry


_SOURCE_PATH_COLUMNS = (
    "source_soma_proportional_path",
    "move_soma_proportional_path",
)


def _group_index_rows(
    *,
    index_csv: Path,
    skeleton_id_column: str,
    limit: int | None,
) -> tuple[dict[str, dict[str, object]], Counter[str], int]:
    groups: dict[str, dict[str, object]] = {}
    source_column_counts: Counter[str] = Counter()
    skipped = 0
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and row_index >= limit:
                break
            skeleton_id = str(row.get(skeleton_id_column) or row.get("encoder_id") or "").strip()
            source_path, source_column = _source_path_from_row(row)
            if not skeleton_id or not source_path:
                skipped += 1
                continue
            source_column_counts[source_column] += 1
            payload = groups.setdefault(
                skeleton_id,
                {
                    "clip_count": 0,
                    "source_paths": Counter(),
                    "shape_paths": Counter(),
                    "package_counts": Counter(),
                    "category_counts": Counter(),
                    "filenames": set(),
                    "move_names": set(),
                },
            )
            payload["clip_count"] = int(payload["clip_count"]) + 1
            payload["source_paths"][source_path] += 1
            shape_path = str(
                row.get("source_soma_proportional_shape_path")
                or row.get("move_soma_proportional_shape_path")
                or ""
            ).strip()
            if shape_path:
                payload["shape_paths"][shape_path] += 1
            for key, target in (
                ("package", "package_counts"),
                ("category", "category_counts"),
            ):
                value = str(row.get(key) or "").strip()
                if value:
                    payload[target][value] += 1
            for key, target in (
                ("filename", "filenames"),
                ("move_name", "move_names"),
            ):
                value = str(row.get(key) or "").strip()
                if value:
                    payload[target].add(value)
    return groups, source_column_counts, skipped


def _source_path_from_row(row: Mapping[str, str]) -> tuple[str, str]:
    for column in _SOURCE_PATH_COLUMNS:
        value = str(row.get(column) or "").strip()
        if value:
            return value, column
    return "", ""


def _read_bvh_text(
    path_text: str,
    *,
    data_root: Path,
    source_archive: tarfile.TarFile | None,
) -> str:
    path = Path(path_text)
    file_candidates = [path] if path.is_absolute() else [data_root / path]
    if path_text.startswith("soma_proportional/"):
        file_candidates.append(data_root / path_text[len("soma_proportional/") :])
    for candidate in file_candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    if source_archive is None:
        raise FileNotFoundError(path_text)
    member_candidates = [path_text.lstrip("/")]
    if path_text.startswith("soma_proportional/"):
        member_candidates.append(path_text[len("soma_proportional/") :])
    for member in member_candidates:
        try:
            extracted = source_archive.extractfile(member)
        except KeyError:
            extracted = None
        if extracted is None:
            continue
        with extracted:
            return extracted.read().decode("utf-8", errors="replace").replace("\x00", "")
    raise FileNotFoundError(path_text)


def _stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _top_key(counter: Counter[str]) -> str:
    return counter.most_common(1)[0][0] if counter else ""


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
