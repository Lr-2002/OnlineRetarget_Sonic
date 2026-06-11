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
    _forward_kinematics,
    _mat_mul,
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
    target_horizon_frames: int = 1
    target_future_step: int = 1
    source_rotation: str = "rot6d"
    include_source_angular_velocity: bool = True
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


@dataclass(frozen=True)
class SourceMotionFeatures:
    positions: list[list[float]]
    rot6d: list[list[list[float]]]
    linear_velocities: list[list[list[float]]]
    angular_velocities: list[list[list[float]]]
    skeleton: list[float]


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
    output_dim = len(SONIC_JOINT_NAMES) * config.target_horizon_frames
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
        "target_format": _target_format(config),
        "observation_spec": spec.to_dict(),
        "config": config.to_dict(),
        "candidate_clip_count": len(task_rows),
        "split_clip_count": len(split_rows),
        "selected_clip_count": selected_clip_count,
        "skipped_clip_count": skipped_clip_count,
        "sample_count": sample_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "source_body_count": len(config.source_body_names),
        "source_body_token_dim": 15,
        "source_step_dim": len(config.source_body_names) * 15,
        "source_skeleton_dim": len(config.source_body_names) * 4,
        "source_rotation_representation": config.source_rotation,
        "rotation_representation": config.source_rotation,
        "source_body_token_fields": [
            "pos_x",
            "pos_y",
            "pos_z",
            "rot6d_0",
            "rot6d_1",
            "rot6d_2",
            "rot6d_3",
            "rot6d_4",
            "rot6d_5",
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
        ],
        "target_joint_dim": len(SONIC_JOINT_NAMES),
        "target_horizon_frames": config.target_horizon_frames,
        "target_future_step": config.target_future_step,
        "action_horizon": config.target_horizon_frames,
        "action_dim": len(SONIC_JOINT_NAMES),
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
    target_joints, fps, sonic_source = _read_sonic_motion(row, config, np=np)
    if target_joints is None:
        return []
    if config.source_mode == "soma_bvh":
        if source_tar is None:
            return []
        source_motion = _read_source_motion_features(
            source_tar,
            row,
            config,
            max_frames=_needed_source_frames(config),
        )
        source_path = row.get("source_soma_proportional_path", "")
    else:
        source_motion = sonic_source
        source_path = row.get("sonic_path", "")
    if source_motion is None:
        return []
    max_start_by_source = (
        len(source_motion.positions)
        - config.history_frames
        - config.target_frame_offset
        - (config.target_horizon_frames - 1) * config.target_future_step
    )
    max_start_by_target = (
        len(target_joints)
        - config.history_frames
        - config.target_frame_offset
        - (config.target_horizon_frames - 1) * config.target_future_step
    )
    max_start = min(max_start_by_source, max_start_by_target)
    if max_start < 0:
        return []

    samples = []
    windows_for_clip = 0
    for start in range(0, max_start + 1, config.window_stride):
        target_index = start + config.history_frames - 1 + config.target_frame_offset
        frame_indices = [
            target_index + offset * config.target_future_step
            for offset in range(config.target_horizon_frames)
        ]
        future_targets = [target_joints[index] for index in frame_indices]
        history = source_motion.positions[start : start + config.history_frames]
        observation = _observation_from_history(history, row, spec)
        sample = {
            "sample_id": _sample_id(row, config.split, start),
            "actor_uid": row.get("actor_uid", ""),
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
            "source_body_tokens": _source_body_tokens(source_motion, frame_indices),
            "source_body_token_dim": 15,
            "source_skeleton": source_motion.skeleton,
            "morphology": _morphology_vector(row),
            "robot_state": [0.0] * spec.robot_state_dim(),
            "target_joint_names": list(SONIC_JOINT_NAMES),
            "target_frame": target_index,
            "target_frame_indices": frame_indices,
            "target_horizon_frames": config.target_horizon_frames,
            "target_future_step": config.target_future_step,
            "prev_target_frame": max(0, target_index - 1),
            "fps": fps,
            "observation": observation,
            "prev_target_joints": [
                float(value) for value in target_joints[max(0, target_index - 1)]
            ],
            "target_joints": [float(value) for value in target_joints[target_index]],
            "future_target_joints": [
                [float(value) for value in frame] for frame in future_targets
            ],
        }
        samples.append(sample)
        windows_for_clip += 1
        if windows_for_clip >= config.max_windows_per_clip:
            break
    return samples


def _read_source_motion_features(
    tar: tarfile.TarFile,
    row: Mapping[str, str],
    config: SonicWindowedBuildConfig,
    *,
    max_frames: int | None = None,
) -> SourceMotionFeatures | None:
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
    return _source_features_from_bvh(
        motion,
        config=config,
    )


def _needed_source_frames(config: SonicWindowedBuildConfig) -> int | None:
    if config.max_windows_per_clip <= 0:
        return None
    last_start = (config.max_windows_per_clip - 1) * config.window_stride
    return (
        last_start
        + config.history_frames
        + config.target_frame_offset
        + (config.target_horizon_frames - 1) * config.target_future_step
    )


def _read_sonic_motion(
    row: Mapping[str, str],
    config: SonicWindowedBuildConfig,
    *,
    np: Any,
) -> tuple[list[list[float]] | None, float, SourceMotionFeatures | None]:
    path_text = row.get("sonic_path", "")
    if not path_text:
        return None, 0.0, None
    try:
        with np.load(Path(path_text)) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            joint_pos = np.asarray(data["joint_pos"], dtype=float)
            body_pos = np.asarray(data["body_pos_w"], dtype=float)
            body_quat = np.asarray(data["body_quat_w"], dtype=float)
            if "body_ang_vel_w" in data.files:
                body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=float)
            else:
                body_ang_vel = np.zeros_like(body_pos)
    except Exception:
        return None, 0.0, None
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(SONIC_JOINT_NAMES):
        return None, fps, None
    if body_pos.ndim != 3 or body_pos.shape[1:] != (30, 3):
        return None, fps, None
    if body_quat.ndim != 3 or body_quat.shape[1:] != (30, 4):
        return None, fps, None
    if body_ang_vel.ndim != 3 or body_ang_vel.shape[1:] != (30, 3):
        body_ang_vel = np.zeros_like(body_pos)
    if (
        not bool(np.isfinite(joint_pos).all())
        or not bool(np.isfinite(body_pos).all())
        or not bool(np.isfinite(body_quat).all())
        or not bool(np.isfinite(body_ang_vel).all())
    ):
        return None, fps, None
    return (
        joint_pos.tolist(),
        fps,
        _source_features_from_sonic(body_pos, body_quat, body_ang_vel, config=config, np=np),
    )


def _body_positions_from_sonic(body_pos: Any, *, np: Any) -> list[list[float]]:
    pelvis = body_pos[:, :1, :]
    relative = body_pos - pelvis
    return relative.reshape((body_pos.shape[0], body_pos.shape[1] * body_pos.shape[2])).tolist()


def _source_features_from_sonic(
    body_pos: Any,
    body_quat: Any,
    body_ang_vel: Any,
    *,
    config: SonicWindowedBuildConfig,
    np: Any,
) -> SourceMotionFeatures:
    selected = _selected_sonic_body_indices(config.source_body_names)
    pelvis = body_pos[:, :1, :]
    relative = (body_pos - pelvis) * config.position_scale
    positions = relative[:, selected, :].reshape((body_pos.shape[0], len(selected) * 3)).tolist()
    linear_velocities = _body_linear_velocities(
        [
            [[float(value) for value in relative[frame, body, :]] for body in selected]
            for frame in range(body_pos.shape[0])
        ]
    )
    root_rot = [_quat_to_matrix(body_quat[frame, 0, :]) for frame in range(body_quat.shape[0])]
    rot6d_frames = []
    for frame in range(body_quat.shape[0]):
        root_inv = _transpose(root_rot[frame])
        rot6d_frames.append(
            [
                _rot6d(_mat_mul(root_inv, _quat_to_matrix(body_quat[frame, body, :])))
                for body in selected
            ]
        )
    if config.include_source_angular_velocity:
        angular_velocities = [
            [
                _mat_vec(_transpose(root_rot[frame]), body_ang_vel[frame, body, :])
                for body in selected
            ]
            for frame in range(body_ang_vel.shape[0])
        ]
    else:
        angular_velocities = [
            [[0.0, 0.0, 0.0] for _ in selected]
            for _ in range(body_ang_vel.shape[0])
        ]
    return SourceMotionFeatures(
        positions=positions,
        rot6d=rot6d_frames,
        linear_velocities=linear_velocities,
        angular_velocities=angular_velocities,
        skeleton=_source_skeleton(positions, len(selected)),
    )


def _source_features_from_bvh(
    motion,
    *,
    config: SonicWindowedBuildConfig,
) -> SourceMotionFeatures:
    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    body_indices = [name_to_index.get(name) for name in config.source_body_names]
    root_index = name_to_index.get(config.root_body)
    positions: list[list[float]] = []
    rot6d_frames: list[list[list[float]]] = []
    rotation_frames: list[list[tuple[tuple[float, float, float], ...]]] = []
    for row in motion.frames:
        global_positions, global_rotations = _forward_kinematics(motion, row)
        root = global_positions[root_index] if root_index is not None else (0.0, 0.0, 0.0)
        root_rotation = global_rotations[root_index] if root_index is not None else _identity()
        root_inv = _transpose(root_rotation)
        flattened_positions: list[float] = []
        frame_rotations: list[tuple[tuple[float, float, float], ...]] = []
        frame_rot6d: list[list[float]] = []
        for index in body_indices:
            if index is None:
                relative_rotation = _identity()
                flattened_positions.extend((0.0, 0.0, 0.0))
            else:
                position = global_positions[index]
                flattened_positions.extend(
                    (position[axis] - root[axis]) * config.position_scale for axis in range(3)
                )
                relative_rotation = _mat_mul(root_inv, global_rotations[index])
            frame_rotations.append(relative_rotation)
            frame_rot6d.append(_rot6d(relative_rotation))
        positions.append(flattened_positions)
        rotation_frames.append(frame_rotations)
        rot6d_frames.append(frame_rot6d)
    body_positions = _position_frames_to_body_positions(positions, len(body_indices))
    return SourceMotionFeatures(
        positions=positions,
        rot6d=rot6d_frames,
        linear_velocities=_body_linear_velocities(body_positions),
        angular_velocities=(
            _body_angular_velocities(rotation_frames)
            if config.include_source_angular_velocity
            else [[[0.0, 0.0, 0.0] for _ in body_indices] for _ in positions]
        ),
        skeleton=_source_skeleton(positions, len(body_indices)),
    )


def _source_body_tokens(
    source: SourceMotionFeatures,
    frame_indices: Sequence[int],
) -> list[list[list[float]]]:
    tokens = []
    body_count = len(source.rot6d[0]) if source.rot6d else 0
    for frame_index in frame_indices:
        positions = _flat_positions_to_body_positions(source.positions[frame_index], body_count)
        frame_tokens = []
        for body_index in range(body_count):
            frame_tokens.append(
                positions[body_index]
                + source.rot6d[frame_index][body_index]
                + source.linear_velocities[frame_index][body_index]
                + source.angular_velocities[frame_index][body_index]
            )
        tokens.append(frame_tokens)
    return tokens


def _source_skeleton(positions: Sequence[Sequence[float]], body_count: int) -> list[float]:
    if not positions:
        return [0.0] * body_count * 4
    anchor = _flat_positions_to_body_positions(positions[0], body_count)
    distances = [
        math.sqrt(sum(float(value) * float(value) for value in body_position))
        for body_position in anchor
    ]
    return [value for body_position in anchor for value in body_position] + distances


def _selected_sonic_body_indices(body_names: Sequence[str]) -> list[int]:
    # SONIC NPZ body arrays use the same 30-body order as DEFAULT_SOURCE_BODY_NAMES.
    index_by_name = {name: index for index, name in enumerate(DEFAULT_SOURCE_BODY_NAMES)}
    return [index_by_name.get(name, 0) for name in body_names]


def _position_frames_to_body_positions(
    positions: Sequence[Sequence[float]],
    body_count: int,
) -> list[list[list[float]]]:
    return [_flat_positions_to_body_positions(frame, body_count) for frame in positions]


def _flat_positions_to_body_positions(
    positions: Sequence[float],
    body_count: int,
) -> list[list[float]]:
    return [
        [float(value) for value in positions[index * 3 : (index + 1) * 3]]
        for index in range(body_count)
    ]


def _body_linear_velocities(frames: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    if not frames:
        return []
    velocities = [[[0.0, 0.0, 0.0] for _ in frames[0]]]
    for prev, cur in zip(frames, frames[1:]):
        velocities.append(
            [
                [float(cur[body][axis]) - float(prev[body][axis]) for axis in range(3)]
                for body in range(len(cur))
            ]
        )
    return velocities


def _body_angular_velocities(
    frames: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> list[list[list[float]]]:
    if not frames:
        return []
    velocities = [[[0.0, 0.0, 0.0] for _ in frames[0]]]
    for prev, cur in zip(frames, frames[1:]):
        frame_velocities = []
        for prev_rot, cur_rot in zip(prev, cur):
            delta = _mat_mul(cur_rot, _transpose(prev_rot))
            frame_velocities.append(_rotation_vector(delta))
        velocities.append(frame_velocities)
    return velocities


def _rotation_vector(matrix: Sequence[Sequence[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle = math.acos(cosine)
    if abs(angle) < 1.0e-8:
        return [0.0, 0.0, 0.0]
    scale = angle / max(1.0e-8, 2.0 * math.sin(angle))
    return [
        (matrix[2][1] - matrix[1][2]) * scale,
        (matrix[0][2] - matrix[2][0]) * scale,
        (matrix[1][0] - matrix[0][1]) * scale,
    ]


def _quat_to_matrix(quat: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = (float(value) for value in quat[:4])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-8:
        return _identity()
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _identity() -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _transpose(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(matrix[col][row]) for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    ]


def _rot6d(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(matrix[0][0]),
        float(matrix[1][0]),
        float(matrix[2][0]),
        float(matrix[0][1]),
        float(matrix[1][1]),
        float(matrix[2][1]),
    ]


def _target_format(config: SonicWindowedBuildConfig) -> str:
    if config.target_horizon_frames > 1:
        return "bones_sonic_joint_pos_future_window"
    return "bones_sonic_joint_pos"


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
        f"{f'_fh{config.target_horizon_frames}' if config.target_horizon_frames > 1 else ''}"
        f"_stride{config.window_stride}_limit{config.limit}"
    )


def _validate_config(config: SonicWindowedBuildConfig) -> None:
    if config.split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if config.source_mode not in {"sonic_body_pos", "soma_bvh"}:
        raise ValueError("source_mode must be sonic_body_pos or soma_bvh")
    if config.history_frames <= 0:
        raise ValueError("history_frames must be positive")
    if config.target_horizon_frames <= 0:
        raise ValueError("target_horizon_frames must be positive")
    if config.target_future_step <= 0:
        raise ValueError("target_future_step must be positive")
    if config.source_rotation != "rot6d":
        raise ValueError("source_rotation must be rot6d")
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
