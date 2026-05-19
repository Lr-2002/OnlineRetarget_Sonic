"""Actor/proportional-skeleton registry builders."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .bones_sonic import inspect_sonic_npz
from .schema import MORPHOLOGY_NUMERIC_COLUMNS


@dataclass(frozen=True)
class SkeletonRegistryResult:
    output_dir: Path
    registry_csv: Path
    report_json: Path
    actor_count: int
    clip_count: int
    shape_file_missing_count: int
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["registry_csv"] = str(self.registry_csv)
        payload["report_json"] = str(self.report_json)
        return payload


def build_skeleton_registry(
    *,
    index_csv: Path,
    data_root: Path,
    output_root: Path,
    run_name: str = "bones_sonic_txt_filtered_skeleton_registry_v0",
    action_column: str = "merged_quality_action",
    allowed_actions: Sequence[str] = ("keep", "downweight"),
) -> SkeletonRegistryResult:
    """Build an actor-level registry for SOMA proportional skeleton experiments.

    The curated row index is the source of truth. It already links each clip to
    an actor id, source SOMA-proportional BVH path, proportional shape file, and
    G1 target path. This function aggregates those rows into the minimum artifact
    needed to route examples to actor-specific encoders.
    """

    index_csv = index_csv.expanduser()
    data_root = data_root.expanduser()
    output_dir = output_root.expanduser() / "skeleton_registry" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_csv = output_dir / "skeleton_registry.csv"
    report_json = output_dir / "skeleton_registry_report.json"

    allowed_action_set = set(allowed_actions)
    accum: dict[str, dict[str, object]] = {}
    rows_seen = 0
    rows_used = 0
    action_counts: Counter[str] = Counter()
    action_actor_sets: dict[str, set[str]] = defaultdict(set)
    all_actor_set: set[str] = set()
    used_actor_set: set[str] = set()
    split_clip_counts: Counter[str] = Counter()
    split_actor_sets: dict[str, set[str]] = defaultdict(set)

    with index_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_seen += 1
            action = row.get(action_column, "")
            action_counts[action] += 1
            actor_uid = row.get("actor_uid", "")
            if not actor_uid:
                continue
            all_actor_set.add(actor_uid)
            action_actor_sets[action].add(actor_uid)
            if allowed_action_set and action not in allowed_action_set:
                continue
            used_actor_set.add(actor_uid)
            rows_used += 1
            split = row.get("split", "")
            split_clip_counts[split] += 1
            split_actor_sets[split].add(actor_uid)
            _accumulate_actor(accum, actor_uid, row, data_root=data_root)

    registry_rows = [_finalize_actor(actor_uid, payload) for actor_uid, payload in accum.items()]
    registry_rows.sort(key=lambda row: row["actor_uid"])
    _write_csv(registry_csv, registry_rows)

    shape_missing = sum(row["shape_file_exists"] != "True" for row in registry_rows)
    clips_per_actor = [int(row["clip_count"]) for row in registry_rows]
    report = {
        "index_csv": str(index_csv),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "registry_csv": str(registry_csv),
        "report_json": str(report_json),
        "run_name": run_name,
        "action_column": action_column,
        "allowed_actions": list(allowed_actions),
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "rows_excluded_by_action": rows_seen - rows_used,
        "all_actor_count": len(all_actor_set),
        "actor_count": len(registry_rows),
        "actor_count_excluded_by_action": len(all_actor_set - used_actor_set),
        "clip_count": sum(clips_per_actor),
        "shape_file_missing_count": shape_missing,
        "action_counts": dict(sorted(action_counts.items())),
        "action_actor_counts": {
            action: len(actors) for action, actors in sorted(action_actor_sets.items())
        },
        "split_clip_counts": dict(sorted(split_clip_counts.items())),
        "split_actor_counts": {
            split: len(actors) for split, actors in sorted(split_actor_sets.items())
        },
        "clip_count_summary": _summary(clips_per_actor),
        "top_actors_by_clip_count": registry_rows[:0]
        + sorted(registry_rows, key=lambda row: int(row["clip_count"]), reverse=True)[:20],
        "bottom_actors_by_clip_count": sorted(
            registry_rows, key=lambda row: int(row["clip_count"])
        )[:20],
        "shape_header_signatures": dict(
            sorted(Counter(row["shape_header_signature"] for row in registry_rows).items())
        ),
        "contract": {
            "unit": "actor_uid is the first skeleton id for proportional-SOMA experiments",
            "source_skeleton": "move_soma_proportional_path",
            "source_shape": "move_soma_proportional_shape_path",
            "target": "G1 motion paired in the same curated row",
            "next_use": "route samples by actor_uid to a per-skeleton encoder bank",
        },
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return SkeletonRegistryResult(
        output_dir=output_dir,
        registry_csv=registry_csv,
        report_json=report_json,
        actor_count=len(registry_rows),
        clip_count=sum(clips_per_actor),
        shape_file_missing_count=shape_missing,
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def _accumulate_actor(
    accum: dict[str, dict[str, object]],
    actor_uid: str,
    row: Mapping[str, str],
    *,
    data_root: Path,
) -> None:
    payload = accum.setdefault(
        actor_uid,
        {
            "clip_count": 0,
            "split_counts": Counter(),
            "package_counts": Counter(),
            "category_counts": Counter(),
            "mirror_count": 0,
            "shape_paths": Counter(),
            "source_paths": Counter(),
            "morphology_values": defaultdict(list),
        },
    )
    payload["clip_count"] = int(payload["clip_count"]) + 1
    payload["split_counts"][row.get("split", "")] += 1
    payload["package_counts"][row.get("package", "")] += 1
    payload["category_counts"][row.get("category", "")] += 1
    if _is_true(row.get("is_mirror", "")):
        payload["mirror_count"] = int(payload["mirror_count"]) + 1
    shape_path = row.get("move_soma_proportional_shape_path", "")
    source_path = row.get("move_soma_proportional_path", "")
    if shape_path:
        payload["shape_paths"][shape_path] += 1
    if source_path:
        payload["source_paths"][source_path] += 1
    for column in MORPHOLOGY_NUMERIC_COLUMNS:
        value = _maybe_float(row.get(column))
        if value is not None:
            payload["morphology_values"][column].append(value)
    payload["data_root"] = data_root


def _finalize_actor(actor_uid: str, payload: Mapping[str, object]) -> dict[str, object]:
    shape_paths: Counter[str] = payload["shape_paths"]
    shape_path = shape_paths.most_common(1)[0][0] if shape_paths else ""
    data_root = payload["data_root"]
    shape_abs = data_root / shape_path if shape_path else Path("")
    headers = _shape_headers(shape_abs) if shape_path and shape_abs.exists() else {}
    morphology = payload["morphology_values"]
    split_counts: Counter[str] = payload["split_counts"]
    package_counts: Counter[str] = payload["package_counts"]
    category_counts: Counter[str] = payload["category_counts"]
    row = {
        "actor_uid": actor_uid,
        "encoder_id": actor_uid,
        "clip_count": int(payload["clip_count"]),
        "train_clip_count": split_counts.get("train", 0),
        "val_clip_count": split_counts.get("val", 0),
        "test_clip_count": split_counts.get("test", 0),
        "mirror_count": int(payload["mirror_count"]),
        "shape_path": shape_path,
        "shape_abs_path": str(shape_abs) if shape_path else "",
        "shape_file_exists": str(bool(shape_path and shape_abs.exists())),
        "shape_header_signature": _header_signature(headers),
        "shape_header_json": json.dumps(headers, sort_keys=True),
        "source_path_count": len(payload["source_paths"]),
        "top_package": _top_key(package_counts),
        "top_category": _top_key(category_counts),
    }
    for column in MORPHOLOGY_NUMERIC_COLUMNS:
        values = morphology.get(column, [])
        row[column] = _mean(values) if values else ""
    return row


def _shape_headers(path: Path) -> dict[str, object]:
    try:
        headers = inspect_sonic_npz(path)
    except Exception as exc:
        return {"error": type(exc).__name__}
    return {
        key: {
            "dtype": header.dtype,
            "shape": list(header.shape),
            "fortran_order": header.fortran_order,
            "scalar_preview": header.scalar_preview,
        }
        for key, header in sorted(headers.items())
    }


def _header_signature(headers: Mapping[str, object]) -> str:
    if not headers:
        return ""
    if "error" in headers:
        return f"error:{headers['error']}"
    parts = []
    for key, header in sorted(headers.items()):
        if isinstance(header, Mapping):
            shape = "x".join(str(item) for item in header.get("shape", []))
            parts.append(f"{key}:{header.get('dtype')}:{shape}")
    return "|".join(parts)


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


def _summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "mean": round(sum(sorted_values) / len(sorted_values), 6),
        "max": float(sorted_values[-1]),
    }


def _top_key(counter: Counter[str]) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6)


def _maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
