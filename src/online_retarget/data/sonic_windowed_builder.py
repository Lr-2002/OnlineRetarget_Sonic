"""Build supervised walk samples from BONES-SONIC targets.

This builder is intentionally SONIC-native on the target side: it reads
``joint_pos`` from ``bones_sonic/*.npz`` instead of legacy G1 CSV files. The
source side still uses the SOMA-proportional BVH path recorded in the SONIC
index, so the resulting samples are a small source-skeleton to G1-joint
baseline rather than a target-only autoencoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tarfile
from typing import Any, Mapping, Sequence

from .bones_sonic import SONIC_JOINT_NAMES
from .schema import MORPHOLOGY_NUMERIC_COLUMNS, ObservationSpec
from .windowed_builder import (
    DEFAULT_SOURCE_BODY_NAMES,
    body_positions_from_bvh,
    parse_bvh_motion,
)


@dataclass(frozen=True)
class SonicWindowedBuildConfig:
    split: str = "train"
    task_query: str = "walk"
    source_mode: str = "soma_bvh"
    include_mirrors: bool = False
    limit: int = 512
    clip_limit: int | None = None
    history_frames: int = 8
    target_frame_offset: int = 0
    window_stride: int = 10
    max_windows_per_clip: int = 8
    split_seed: int = 17
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    position_scale: float = 0.01
    root_body: str = "Hips"
    source_body_names: tuple[str, ...] = DEFAULT_SOURCE_BODY_NAMES

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_body_names"] = list(self.source_body_names)
        return payload


@dataclass(frozen=True)
class SonicWindowedBuildResult:
    output_dir: Path
    samples_jsonl: Path
    manifest_json: Path
    sample_count: int
    selected_clip_count: int
    skipped_clip_count: int
    input_dim: int
    output_dim: int
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["samples_jsonl"] = str(self.samples_jsonl)
        payload["manifest_json"] = str(self.manifest_json)
        return payload


def build_sonic_windowed_jsonl(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: SonicWindowedBuildConfig | None = None,
) -> SonicWindowedBuildResult:
    """Build fixed-window observations with SONIC G1 joint targets."""

    np = _require_numpy()
    config = config or SonicWindowedBuildConfig()
    _validate_config(config)
    spec = ObservationSpec(
        history_frames=config.history_frames,
        source_body_count=len(config.source_body_names),
    )
    rows = _load_rows(index_csv)
    task_rows = [row for row in rows if _matches_task(row, config.task_query)]
    if not config.include_mirrors:
        task_rows = [row for row in task_rows if not _is_true(row.get("is_mirror"))]
    split_by_actor = _actor_splits(task_rows, config)
    split_rows = [
        row for row in task_rows if split_by_actor.get(row.get("actor_uid", "")) == config.split
    ]
    if config.clip_limit is not None:
        split_rows = split_rows[: config.clip_limit]

    output_dir = output_root.expanduser() / "supervised" / _run_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_jsonl = output_dir / "samples.jsonl"
    manifest_json = output_dir / "manifest.json"

    sample_count = 0
    selected_clip_count = 0
    skipped_clip_count = 0
    output_dim = len(SONIC_JOINT_NAMES)
    input_dim = spec.flattened_dim()
    source_tar = None
    try:
        if config.source_mode == "soma_bvh":
            source_tar = tarfile.open(data_root.expanduser() / "soma_proportional.tar", "r:*")
        with samples_jsonl.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                if sample_count >= config.limit:
                    break
                samples = _samples_for_row(row, source_tar, config, spec, np=np)
                if not samples:
                    skipped_clip_count += 1
                    continue
                selected_clip_count += 1
                for sample in samples:
                    if sample_count >= config.limit:
                        break
                    handle.write(json.dumps(sample, sort_keys=True))
                    handle.write("\n")
                    sample_count += 1
    finally:
        if source_tar is not None:
            source_tar.close()

    split_counts = _split_counts(task_rows, split_by_actor)
    manifest = {
        "data_root": str(data_root),
        "index_csv": str(index_csv),
        "samples_jsonl": str(samples_jsonl),
        "builder": "sonic_walk_soma_bvh_to_g1_joint_window_debug",
        "contract_note": (
            "Debug walk baseline. Targets are BONES-SONIC NPZ joint_pos in "
            "SONIC/IsaacLab G1 order. source_mode=soma_bvh uses SOMA-proportional "
            "BVH FK body-position windows; source_mode=sonic_body_pos uses the "
            "same NPZ body_pos_w as a fast target-state baseline and must not be "
            "reported as retargeting. This is not a promoted M2Q-gated dataset."
        ),
        "source_format": config.source_mode,
        "target_format": "bones_sonic_joint_pos",
        "observation_spec": spec.to_dict(),
        "config": config.to_dict(),
        "candidate_clip_count": len(task_rows),
        "split_clip_count": len(split_rows),
        "selected_clip_count": selected_clip_count,
        "skipped_clip_count": skipped_clip_count,
        "sample_count": sample_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "split_policy": {
            "group_by": "actor_uid",
            "seed": config.split_seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": max(0.0, 1.0 - config.train_ratio - config.val_ratio),
            "clip_counts": split_counts,
            "actor_counts": _actor_count_by_split(split_by_actor),
        },
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SonicWindowedBuildResult(
        output_dir=output_dir,
        samples_jsonl=samples_jsonl,
        manifest_json=manifest_json,
        sample_count=sample_count,
        selected_clip_count=selected_clip_count,
        skipped_clip_count=skipped_clip_count,
        input_dim=input_dim,
        output_dim=output_dim,
        git_sha=manifest["git_sha"],
        git_dirty=manifest["git_dirty"],
    )


def _samples_for_row(
    row: Mapping[str, str],
    source_tar: tarfile.TarFile | None,
    config: SonicWindowedBuildConfig,
    spec: ObservationSpec,
    *,
    np: Any,
) -> list[dict[str, object]]:
    target_joints, fps, body_positions = _read_sonic_motion(row, np=np)
    if target_joints is None:
        return []
    if config.source_mode == "soma_bvh":
        if source_tar is None:
            return []
        source_positions = _read_source_positions(source_tar, row, config, max_frames=_needed_source_frames(config))
        source_path = row.get("source_soma_proportional_path", "")
    else:
        source_positions = body_positions
        source_path = row.get("sonic_path", "")
    if source_positions is None:
        return []
    usable_frames = min(len(source_positions), len(target_joints))
    last_target = usable_frames - 1 - config.target_frame_offset
    max_start = last_target - config.history_frames + 1
    if max_start < 0:
        return []

    samples = []
    windows_for_clip = 0
    for start in range(0, max_start + 1, config.window_stride):
        target_index = start + config.history_frames - 1 + config.target_frame_offset
        history = source_positions[start : start + config.history_frames]
        observation = _observation_from_history(history, row, spec)
        sample = {
            "sample_id": _sample_id(row, config.split, start),
            "actor_uid": row.get("actor_uid", ""),
            "encoder_id": row.get("actor_uid", ""),
            "category": row.get("category", ""),
            "package": row.get("package", ""),
            "quality_action": "debug_unfiltered",
            "quality_flags": [],
            "source_motion_path": source_path,
            "source_mode": config.source_mode,
            "target_g1_path": row.get("sonic_path", ""),
            "sonic_relative_path": row.get("sonic_relative_path", ""),
            "history_frames": config.history_frames,
            "source_body_names": list(config.source_body_names),
            "target_joint_names": list(SONIC_JOINT_NAMES),
            "target_frame": target_index,
            "prev_target_frame": max(0, target_index - 1),
            "fps": fps,
            "observation": observation,
            "prev_target_joints": [
                float(value) for value in target_joints[max(0, target_index - 1)]
            ],
            "target_joints": [float(value) for value in target_joints[target_index]],
        }
        samples.append(sample)
        windows_for_clip += 1
        if windows_for_clip >= config.max_windows_per_clip:
            break
    return samples


def _read_source_positions(
    tar: tarfile.TarFile,
    row: Mapping[str, str],
    config: SonicWindowedBuildConfig,
    *,
    max_frames: int | None = None,
) -> list[list[float]] | None:
    member_path = row.get("source_soma_proportional_path", "")
    if not member_path:
        return None
    try:
        extracted = tar.extractfile(member_path)
    except (KeyError, tarfile.TarError):
        return None
    if extracted is None:
        return None
    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8").read()
        except UnicodeDecodeError:
            return None
    try:
        motion = parse_bvh_motion(text, max_frames=max_frames)
    except ValueError:
        return None
    return body_positions_from_bvh(
        motion,
        body_names=config.source_body_names,
        root_body=config.root_body,
        position_scale=config.position_scale,
    )


def _needed_source_frames(config: SonicWindowedBuildConfig) -> int | None:
    if config.max_windows_per_clip <= 0:
        return None
    last_start = (config.max_windows_per_clip - 1) * config.window_stride
    return last_start + config.history_frames + max(0, config.target_frame_offset)


def _read_sonic_motion(
    row: Mapping[str, str],
    *,
    np: Any,
) -> tuple[list[list[float]] | None, float, list[list[float]] | None]:
    path_text = row.get("sonic_path", "")
    if not path_text:
        return None, 0.0, None
    try:
        with np.load(Path(path_text)) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joint_pos = np.asarray(data["joint_pos"], dtype=float)
            body_pos = np.asarray(data["body_pos_w"], dtype=float)
    except Exception:
        return None, 0.0, None
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(SONIC_JOINT_NAMES):
        return None, fps, None
    if body_pos.ndim != 3 or body_pos.shape[1:] != (30, 3):
        return None, fps, None
    if not bool(np.isfinite(joint_pos).all()) or not bool(np.isfinite(body_pos).all()):
        return None, fps, None
    return joint_pos.tolist(), fps, _body_positions_from_sonic(body_pos, np=np)


def _body_positions_from_sonic(body_pos: Any, *, np: Any) -> list[list[float]]:
    pelvis = body_pos[:, :1, :]
    relative = body_pos - pelvis
    return relative.reshape((body_pos.shape[0], body_pos.shape[1] * body_pos.shape[2])).tolist()


def _observation_from_history(
    history: Sequence[Sequence[float]],
    row: Mapping[str, str],
    spec: ObservationSpec,
) -> list[float]:
    velocities = _velocities(history)
    source_features: list[float] = []
    for positions, velocity in zip(history, velocities):
        source_features.extend(float(value) for value in positions)
        source_features.extend(float(value) for value in velocity)
    return (
        source_features
        + _morphology_vector(row)
        + [0.0] * spec.robot_state_dim()
    )


def _velocities(frames: Sequence[Sequence[float]]) -> list[list[float]]:
    if not frames:
        return []
    velocities = [[0.0] * len(frames[0])]
    for prev, cur in zip(frames, frames[1:]):
        velocities.append([float(cur_value) - float(prev_value) for prev_value, cur_value in zip(prev, cur)])
    return velocities


def _morphology_vector(row: Mapping[str, str]) -> list[float]:
    return [_maybe_float(row.get(column)) for column in MORPHOLOGY_NUMERIC_COLUMNS]


def _load_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matches_task(row: Mapping[str, str], query: str) -> bool:
    tokens = [token for token in query.lower().split() if token]
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("filename", "category", "package", "sonic_relative_path")
    ).lower()
    return all(token in haystack for token in tokens)


def _actor_splits(
    rows: Sequence[Mapping[str, str]],
    config: SonicWindowedBuildConfig,
) -> dict[str, str]:
    actors = sorted({row.get("actor_uid", "") for row in rows if row.get("actor_uid", "")})
    rng = random.Random(config.split_seed)
    rng.shuffle(actors)
    train_count = int(len(actors) * config.train_ratio)
    val_count = int(len(actors) * config.val_ratio)
    if actors and train_count == 0:
        train_count = 1
    val_start = train_count
    test_start = min(len(actors), train_count + val_count)
    split_by_actor: dict[str, str] = {}
    for actor in actors[:val_start]:
        split_by_actor[actor] = "train"
    for actor in actors[val_start:test_start]:
        split_by_actor[actor] = "val"
    for actor in actors[test_start:]:
        split_by_actor[actor] = "test"
    return split_by_actor


def _split_counts(
    rows: Sequence[Mapping[str, str]],
    split_by_actor: Mapping[str, str],
) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split = split_by_actor.get(row.get("actor_uid", ""), "")
        if split in counts:
            counts[split] += 1
    return counts


def _actor_count_by_split(split_by_actor: Mapping[str, str]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split in split_by_actor.values():
        if split in counts:
            counts[split] += 1
    return counts


def _sample_id(row: Mapping[str, str], split: str, start: int) -> str:
    actor = row.get("actor_uid", "")
    filename = row.get("filename", "")
    row_id = row.get("metadata_row_index", row.get("sonic_relative_path", ""))
    return f"{split}:{actor}:{filename}:{row_id}:{start}"


def _run_name(config: SonicWindowedBuildConfig) -> str:
    task = re.sub(r"[^a-z0-9]+", "-", config.task_query.lower()).strip("-") or "task"
    source = "sonicbody" if config.source_mode == "sonic_body_pos" else "somabvh"
    return (
        f"{source}_{task}_{config.split}_h{config.history_frames}"
        f"_stride{config.window_stride}_limit{config.limit}"
    )


def _validate_config(config: SonicWindowedBuildConfig) -> None:
    if config.split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if config.source_mode not in {"sonic_body_pos", "soma_bvh"}:
        raise ValueError("source_mode must be sonic_body_pos or soma_bvh")
    if config.history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if config.window_stride <= 0:
        raise ValueError("window_stride must be positive")
    if config.max_windows_per_clip <= 0:
        raise ValueError("max_windows_per_clip must be positive")
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.clip_limit is not None and config.clip_limit <= 0:
        raise ValueError("clip_limit must be positive when provided")
    if config.train_ratio < 0 or config.val_ratio < 0 or config.train_ratio + config.val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to <= 1")


def _maybe_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment blocker path.
        raise RuntimeError("SONIC windowed sample building requires numpy") from exc
    return np


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
