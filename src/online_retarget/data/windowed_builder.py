"""Build schema-compatible fixed-window supervised samples.

This builder is the first step beyond the raw-BVH-channel debug path. It uses a
small standard-library BVH forward-kinematics implementation to emit the M3
observation contract: source body positions, source body velocities,
morphology, and zero-filled robot-state placeholders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from typing import Mapping, Sequence

from .bones_seed import G1_JOINT_COLUMNS
from .schema import (
    MORPHOLOGY_NUMERIC_COLUMNS,
    MotionPairRef,
    ObservationSpec,
    iter_motion_pair_refs,
)


DEFAULT_SOURCE_BODY_NAMES = (
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
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "LeftHandThumb1",
    "LeftHandIndex1",
    "LeftHandMiddle1",
    "LeftHandRing1",
    "LeftHandPinky1",
    "RightHandThumb1",
    "RightHandIndex1",
    "RightHandMiddle1",
)


@dataclass(frozen=True)
class WindowedBuildConfig:
    split: str = "train"
    actions: tuple[str, ...] = ("keep", "downweight")
    action_column: str = "merged_quality_action"
    limit: int = 16
    history_frames: int = 8
    target_frame_offset: int = 0
    window_stride: int = 10
    max_windows_per_clip: int = 1
    position_scale: float = 0.01
    root_body: str = "Hips"
    source_body_names: tuple[str, ...] = DEFAULT_SOURCE_BODY_NAMES

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_body_names"] = list(self.source_body_names)
        return payload


@dataclass(frozen=True)
class WindowedBuildResult:
    output_dir: Path
    samples_jsonl: Path
    manifest_json: Path
    sample_count: int
    skipped_count: int
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


@dataclass(frozen=True)
class BVHJoint:
    name: str
    parent: int | None
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_start: int


@dataclass(frozen=True)
class BVHMotion:
    joints: tuple[BVHJoint, ...]
    frames: tuple[tuple[float, ...], ...]
    frame_time: float
    channel_count: int


def build_windowed_jsonl(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: WindowedBuildConfig | None = None,
) -> WindowedBuildResult:
    """Build fixed-window 30-body observations and G1-joint targets."""

    config = config or WindowedBuildConfig()
    _validate_config(config)
    spec = ObservationSpec(
        history_frames=config.history_frames,
        source_body_count=len(config.source_body_names),
    )
    refs = iter_motion_pair_refs(
        index_csv,
        splits=(config.split,),
        actions=config.actions,
        action_column=config.action_column,
    )
    output_dir = output_root.expanduser() / "supervised" / _run_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_jsonl = output_dir / "samples.jsonl"
    manifest_json = output_dir / "manifest.json"

    sample_count = 0
    skipped_count = 0
    input_dim = spec.flattened_dim()
    output_dim = len(G1_JOINT_COLUMNS)
    with tarfile.open(data_root.expanduser() / "soma_proportional.tar", "r:*") as source_tar:
        with tarfile.open(data_root.expanduser() / "g1.tar", "r:*") as target_tar:
            with samples_jsonl.open("w", encoding="utf-8") as f:
                for ref in refs:
                    if sample_count >= config.limit:
                        break
                    samples = _build_samples(source_tar, target_tar, ref, config, spec)
                    if not samples:
                        skipped_count += 1
                        continue
                    for sample in samples:
                        if sample_count >= config.limit:
                            break
                        f.write(json.dumps(sample, sort_keys=True))
                        f.write("\n")
                        sample_count += 1

    manifest = {
        "data_root": str(data_root),
        "index_csv": str(index_csv),
        "samples_jsonl": str(samples_jsonl),
        "builder": "bvh_fk_30body_window",
        "contract_note": (
            "Samples use BVH FK source body positions/velocities plus morphology. "
            "Robot-state fields are zero-filled placeholders until online state is wired."
        ),
        "observation_spec": spec.to_dict(),
        "config": config.to_dict(),
        "sample_count": sample_count,
        "skipped_count": skipped_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return WindowedBuildResult(
        output_dir=output_dir,
        samples_jsonl=samples_jsonl,
        manifest_json=manifest_json,
        sample_count=sample_count,
        skipped_count=skipped_count,
        input_dim=input_dim,
        output_dim=output_dim,
        git_sha=manifest["git_sha"],
        git_dirty=manifest["git_dirty"],
    )


def parse_bvh_motion(text: str, max_frames: int | None = None) -> BVHMotion:
    """Parse BVH hierarchy and motion frames needed for FK."""

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        motion_idx = raw_lines.index("MOTION")
    except ValueError as exc:
        raise ValueError("BVH missing MOTION section") from exc

    joints: list[BVHJoint] = []
    stack: list[int] = []
    current: int | None = None
    channel_count = 0
    pending_offset: tuple[float, float, float] | None = None
    pending_channels: tuple[str, ...] | None = None
    pending_channel_start = 0

    for line in raw_lines[:motion_idx]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"} and len(parts) >= 2:
            parent = stack[-1] if stack else None
            joints.append(
                BVHJoint(
                    name=parts[1],
                    parent=parent,
                    offset=(0.0, 0.0, 0.0),
                    channels=(),
                    channel_start=channel_count,
                )
            )
            current = len(joints) - 1
            pending_offset = None
            pending_channels = None
        elif parts[0] == "{":
            if current is not None:
                stack.append(current)
        elif parts[0] == "}":
            if stack:
                stack.pop()
        elif parts[0] == "OFFSET" and current is not None and len(parts) >= 4:
            pending_offset = (float(parts[1]), float(parts[2]), float(parts[3]))
            joints[current] = _replace_joint(joints[current], offset=pending_offset)
        elif parts[0] == "CHANNELS" and current is not None and len(parts) >= 3:
            count = int(parts[1])
            pending_channels = tuple(parts[2 : 2 + count])
            pending_channel_start = channel_count
            joints[current] = _replace_joint(
                joints[current],
                channels=pending_channels,
                channel_start=pending_channel_start,
            )
            channel_count += count

    if motion_idx + 2 >= len(raw_lines):
        raise ValueError("BVH MOTION section is incomplete")
    declared_frames = int(_parse_header_value(raw_lines[motion_idx + 1], "Frames"))
    frame_time = _parse_header_value(raw_lines[motion_idx + 2], "Frame Time")
    frames: list[tuple[float, ...]] = []
    for line in raw_lines[motion_idx + 3 :]:
        if max_frames is not None and len(frames) >= max_frames:
            break
        values = tuple(float(item) for item in line.split())
        if len(values) == channel_count:
            frames.append(values)
    expected_frames = min(declared_frames, max_frames) if max_frames is not None else declared_frames
    if len(frames) != expected_frames:
        raise ValueError("BVH frame count or channel width mismatch")
    return BVHMotion(
        joints=tuple(joints),
        frames=tuple(frames),
        frame_time=frame_time,
        channel_count=channel_count,
    )


def body_positions_from_bvh(
    motion: BVHMotion,
    body_names: Sequence[str],
    root_body: str = "Hips",
    position_scale: float = 0.01,
) -> list[list[float]]:
    """Return flattened root-local body positions for every frame."""

    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    body_indices = [name_to_index.get(name) for name in body_names]
    root_index = name_to_index.get(root_body)
    frames = []
    for row in motion.frames:
        global_positions, _ = _forward_kinematics(motion, row)
        root = (
            global_positions[root_index]
            if root_index is not None
            else (0.0, 0.0, 0.0)
        )
        flattened = []
        for index in body_indices:
            if index is None:
                flattened.extend((0.0, 0.0, 0.0))
                continue
            position = global_positions[index]
            flattened.extend((position[axis] - root[axis]) * position_scale for axis in range(3))
        frames.append(flattened)
    return frames


def global_body_position_maps_from_bvh(
    motion: BVHMotion,
    body_names: Sequence[str],
    position_scale: float = 0.01,
) -> list[dict[str, tuple[float, float, float]]]:
    """Return global body positions by body name for every frame."""

    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    body_indices = [(name, name_to_index.get(name)) for name in body_names]
    frames = []
    for row in motion.frames:
        global_positions, _ = _forward_kinematics(motion, row)
        frame_positions: dict[str, tuple[float, float, float]] = {}
        for name, index in body_indices:
            if index is None:
                continue
            position = global_positions[index]
            frame_positions[name] = (
                position[0] * position_scale,
                position[1] * position_scale,
                position[2] * position_scale,
            )
        frames.append(frame_positions)
    return frames


def _build_samples(
    source_tar: tarfile.TarFile,
    target_tar: tarfile.TarFile,
    ref: MotionPairRef,
    config: WindowedBuildConfig,
    spec: ObservationSpec,
) -> list[dict[str, object]]:
    source_positions = _read_source_positions(source_tar, ref, config)
    target_values = _read_g1_joints(
        target_tar,
        ref.target_g1_path,
        max_frames=_needed_target_frames(config),
    )
    if source_positions is None or target_values is None:
        return []
    usable_frames = min(len(source_positions), len(target_values))
    last_target = usable_frames - 1 - config.target_frame_offset
    max_start = last_target - config.history_frames + 1
    if max_start < 0:
        return []

    samples = []
    windows_for_clip = 0
    for start in range(0, max_start + 1, config.window_stride):
        target_index = start + config.history_frames - 1 + config.target_frame_offset
        history = source_positions[start : start + config.history_frames]
        velocities = _velocities(history)
        source_features = []
        for positions, velocity in zip(history, velocities):
            source_features.extend(positions)
            source_features.extend(velocity)
        observation = (
            source_features
            + _morphology_vector(ref.morphology)
            + [0.0] * spec.robot_state_dim()
        )
        samples.append(
            {
                "sample_id": f"{ref.sample_id}:{start}",
                "actor_uid": ref.actor_uid,
                "category": ref.category,
                "package": ref.package,
                "quality_action": ref.quality_action,
                "quality_flags": list(ref.quality_flags),
                "source_motion_path": ref.source_motion_path,
                "target_g1_path": ref.target_g1_path,
                "history_frames": config.history_frames,
                "source_body_names": list(config.source_body_names),
                "window_start": start,
                "target_frame": target_index,
                "prev_target_frame": max(0, target_index - 1),
                "observation": observation,
                "prev_target_joints": target_values[max(0, target_index - 1)],
                "target_joints": target_values[target_index],
            }
        )
        windows_for_clip += 1
        if windows_for_clip >= config.max_windows_per_clip:
            break
    return samples


def _read_source_positions(
    tar: tarfile.TarFile,
    ref: MotionPairRef,
    config: WindowedBuildConfig,
) -> list[list[float]] | None:
    try:
        extracted = tar.extractfile(ref.source_motion_path)
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
        motion = parse_bvh_motion(text, max_frames=_needed_source_frames(config))
    except ValueError:
        return None
    return body_positions_from_bvh(
        motion,
        body_names=config.source_body_names,
        root_body=config.root_body,
        position_scale=config.position_scale,
    )


def _read_g1_joints(
    tar: tarfile.TarFile,
    member_path: str,
    *,
    max_frames: int | None = None,
) -> list[list[float]] | None:
    try:
        extracted = tar.extractfile(member_path)
    except (KeyError, tarfile.TarError):
        return None
    if extracted is None:
        return None
    values = []
    with extracted:
        try:
            rows = csv.DictReader(io.TextIOWrapper(extracted, encoding="utf-8", newline=""))
            for row in rows:
                if max_frames is not None and len(values) >= max_frames:
                    break
                frame = [_maybe_float(row.get(column)) for column in G1_JOINT_COLUMNS]
                if any(value is None for value in frame):
                    continue
                values.append([float(value) for value in frame if value is not None])
        except UnicodeDecodeError:
            return None
    return values


def _forward_kinematics(
    motion: BVHMotion,
    row: Sequence[float],
) -> tuple[list[tuple[float, float, float]], list[tuple[tuple[float, float, float], ...]]]:
    positions: list[tuple[float, float, float]] = []
    rotations: list[tuple[tuple[float, float, float], ...]] = []
    identity = _identity()
    for joint in motion.joints:
        parent_position = positions[joint.parent] if joint.parent is not None else (0.0, 0.0, 0.0)
        parent_rotation = rotations[joint.parent] if joint.parent is not None else identity
        local_translation = _local_translation(joint, row)
        local_rotation = _local_rotation(joint, row)
        world_translation = _mat_vec(parent_rotation, local_translation)
        world_position = _vec_add(parent_position, world_translation)
        world_rotation = _mat_mul(parent_rotation, local_rotation)
        positions.append(world_position)
        rotations.append(world_rotation)
    return positions, rotations


def _local_translation(joint: BVHJoint, row: Sequence[float]) -> tuple[float, float, float]:
    values = list(joint.offset)
    channel_map = {channel: joint.channel_start + offset for offset, channel in enumerate(joint.channels)}
    for axis, channel in enumerate(("Xposition", "Yposition", "Zposition")):
        index = channel_map.get(channel)
        if index is not None:
            values[axis] = row[index]
    return (values[0], values[1], values[2])


def _local_rotation(
    joint: BVHJoint,
    row: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    rotation = _identity()
    for offset, channel in enumerate(joint.channels):
        if not channel.endswith("rotation"):
            continue
        angle = math.radians(row[joint.channel_start + offset])
        axis = channel[0].upper()
        rotation = _mat_mul(rotation, _axis_rotation(axis, angle))
    return rotation


def _axis_rotation(axis: str, angle: float) -> tuple[tuple[float, float, float], ...]:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "Y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    if axis == "Z":
        return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return _identity()


def _velocities(frames: Sequence[Sequence[float]]) -> list[list[float]]:
    if not frames:
        return []
    velocities = [[0.0] * len(frames[0])]
    for prev, cur in zip(frames, frames[1:]):
        velocities.append([float(cur_value) - float(prev_value) for prev_value, cur_value in zip(prev, cur)])
    return velocities


def _morphology_vector(morphology: Mapping[str, float | None]) -> list[float]:
    return [0.0 if morphology.get(column) is None else float(morphology[column]) for column in MORPHOLOGY_NUMERIC_COLUMNS]


def _replace_joint(joint: BVHJoint, **updates: object) -> BVHJoint:
    return BVHJoint(
        name=str(updates.get("name", joint.name)),
        parent=updates.get("parent", joint.parent),  # type: ignore[arg-type]
        offset=updates.get("offset", joint.offset),  # type: ignore[arg-type]
        channels=updates.get("channels", joint.channels),  # type: ignore[arg-type]
        channel_start=int(updates.get("channel_start", joint.channel_start)),
    )


def _parse_header_value(line: str, name: str) -> float:
    prefix = f"{name}:"
    if not line.startswith(prefix):
        raise ValueError(f"expected BVH header line {prefix}")
    return float(line[len(prefix) :].strip())


def _identity() -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _mat_mul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _mat_vec(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _vec_add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _maybe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _run_name(config: WindowedBuildConfig) -> str:
    action_tag = config.action_column.replace("_", "-")
    body_tag = f"{len(config.source_body_names)}b"
    return (
        f"{config.split}_{action_tag}_{body_tag}_h{config.history_frames}"
        f"_stride{config.window_stride}_limit{config.limit}"
    )


def _needed_source_frames(config: WindowedBuildConfig) -> int | None:
    if config.max_windows_per_clip <= 0:
        return None
    last_start = (config.max_windows_per_clip - 1) * config.window_stride
    return last_start + config.history_frames


def _needed_target_frames(config: WindowedBuildConfig) -> int | None:
    source_frames = _needed_source_frames(config)
    if source_frames is None:
        return None
    return source_frames + config.target_frame_offset


def _validate_config(config: WindowedBuildConfig) -> None:
    if config.split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if config.window_stride <= 0:
        raise ValueError("window_stride must be positive")
    if config.max_windows_per_clip <= 0:
        raise ValueError("max_windows_per_clip must be positive")
    if config.target_frame_offset < 0:
        raise ValueError("target_frame_offset must be non-negative")


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
