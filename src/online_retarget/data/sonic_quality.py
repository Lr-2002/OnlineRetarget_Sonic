"""BONES-SONIC NPZ target quality scanning."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .bones_sonic import SONIC_BODY_NAMES, SONIC_JOINT_NAMES
from .g1_quality import G1KinematicModel, load_g1_kinematic_model
from .row_sampling import sampling_run_tag, scan_sampling_report, select_rows_for_scan


DEFAULT_SONIC_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
SONIC_BODY_PARENTS: tuple[int | None, ...] = (
    None,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
)


@dataclass(frozen=True)
class SonicQualityConfig:
    """Thresholds for provisional SONIC NPZ quality flags."""

    max_joint_velocity: float = 20.0
    max_joint_step_velocity: float = 20.0
    max_joint_acceleration: float | None = None
    max_root_speed: float = 8.0
    max_root_step_speed: float = 8.0
    max_root_acceleration: float | None = None
    frame_stride: int = 1
    max_frames: int | None = None
    model_xml: Path | None = None
    ground_height: float = 0.0
    contact_height_threshold: float = 0.04
    max_contact_slide_speed: float = 0.25
    max_contact_skate_distance: float = 0.02
    max_mean_foot_clearance: float = 0.10
    max_penetration_depth: float = 0.03
    min_contact_frame_ratio: float = 0.05
    max_joint_limit_violation_rate: float = 0.0
    enable_joint_limit_flags: bool = False
    start_end_frames: int = 10
    max_start_end_root_speed: float = 0.20
    min_self_collision_distance: float = 0.015
    max_self_collision_proxy_rate: float = 0.0
    min_self_collision_kinematic_hops: int = 4
    enable_body_origin_contact_flags: bool = False
    enable_body_origin_self_collision_flags: bool = False
    foot_bodies: tuple[str, ...] = DEFAULT_SONIC_FOOT_BODIES

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_xml"] = str(self.model_xml) if self.model_xml is not None else None
        payload["foot_bodies"] = list(self.foot_bodies)
        return payload


@dataclass(frozen=True)
class SonicQualityScanResult:
    output_dir: Path
    stats_jsonl: Path
    report_json: Path
    scanned_rows: int
    skipped_rows: int
    action_counts: dict[str, int]
    flag_counts: dict[str, int]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["stats_jsonl"] = str(self.stats_jsonl)
        payload["report_json"] = str(self.report_json)
        return payload


def scan_sonic_quality_from_index(
    index_csv: Path,
    output_root: Path,
    config: SonicQualityConfig | None = None,
    limit: int | None = 100,
    sample_by: Sequence[str] = (),
) -> SonicQualityScanResult:
    """Scan BONES-SONIC NPZ tensors referenced by a SONIC index."""

    np = _require_numpy()
    config = config or SonicQualityConfig()
    _validate_config(config)
    model = load_g1_kinematic_model(config.model_xml) if config.model_xml else None
    candidate_rows = list(_iter_index_rows(index_csv))
    rows = select_rows_for_scan(candidate_rows, limit=limit, sample_by=sample_by)

    output_dir = (
        output_root.expanduser()
        / "quality"
        / f"{index_csv.parent.name}_sonic_{sampling_run_tag(limit, sample_by)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = output_dir / "sonic_quality_stats.jsonl"
    report_json = output_dir / "sonic_quality_report.json"

    scanned: list[dict[str, object]] = []
    skipped_rows = 0
    with stats_jsonl.open("w", encoding="utf-8") as stats_file:
        for row in rows:
            sonic_path = row.get("sonic_path", "")
            if not sonic_path:
                skipped_rows += 1
                continue
            result = scan_sonic_npz(Path(sonic_path), row, config, np=np, model=model)
            scanned.append(result)
            _write_jsonl_row(stats_file, result)

    action_counts = Counter(str(row["quality_action"]) for row in scanned)
    flag_counts = Counter(
        flag
        for row in scanned
        for flag in _split_flags(str(row.get("quality_flags", "")))
    )
    report = {
        "index_csv": str(index_csv),
        "stats_jsonl": str(stats_jsonl),
        "config": config.to_dict(),
        "model": _model_report(model),
        "limit": limit,
        "sampling": scan_sampling_report(candidate_rows, rows, limit=limit, sample_by=sample_by),
        "scanned_rows": len(scanned),
        "skipped_rows": skipped_rows,
        "action_counts": dict(sorted(action_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "metric_summary": _metric_summary(scanned),
        "body_names": list(SONIC_BODY_NAMES),
        "joint_names": list(SONIC_JOINT_NAMES),
        "metric_note": (
            "Provisional SONIC quality metrics read NPZ body origins and velocities directly. "
            "The SONIC NPZ tensors are interpreted in IsaacLab G1 order, not legacy "
            "BONES-SEED CSV/MJCF pre-order. "
            "Body-origin contact, joint-limit, and self-collision proxy metrics are metric-only "
            "unless their corresponding enable_* flag is set, because body origins are not sole "
            "contact points or collision geometry and the XML limit source may differ from the "
            "SONIC exporter."
        ),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return SonicQualityScanResult(
        output_dir=output_dir,
        stats_jsonl=stats_jsonl,
        report_json=report_json,
        scanned_rows=len(scanned),
        skipped_rows=skipped_rows,
        action_counts=dict(sorted(action_counts.items())),
        flag_counts=dict(sorted(flag_counts.items())),
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def scan_sonic_npz(
    path: Path,
    index_row: Mapping[str, str],
    config: SonicQualityConfig,
    *,
    np: Any | None = None,
    model: G1KinematicModel | None = None,
) -> dict[str, object]:
    """Scan one BONES-SONIC NPZ file."""

    np = np or _require_numpy()
    base = {
        "sonic_relative_path": index_row.get("sonic_relative_path", ""),
        "sonic_path": str(path),
        "date": index_row.get("date", ""),
        "filename": index_row.get("filename", ""),
        "actor_uid": index_row.get("actor_uid", ""),
        "package": index_row.get("package", ""),
        "category": index_row.get("category", ""),
        "is_mirror": index_row.get("is_mirror", ""),
        "legacy_g1_csv_path": index_row.get("legacy_g1_csv_path", ""),
        "source_soma_proportional_path": index_row.get("source_soma_proportional_path", ""),
        "quality_mode": "sonic_npz_body_state",
    }
    try:
        with np.load(path) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            arrays = {
                "joint_pos": np.asarray(data["joint_pos"], dtype=float),
                "joint_vel": np.asarray(data["joint_vel"], dtype=float),
                "body_pos_w": np.asarray(data["body_pos_w"], dtype=float),
                "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=float),
            }
    except Exception as exc:
        return _error_result(base, f"sonic_npz_load_error:{type(exc).__name__}")

    return summarize_sonic_arrays(base, arrays, fps=fps, config=config, np=np, model=model)


def summarize_sonic_arrays(
    base: Mapping[str, object],
    arrays: Mapping[str, Any],
    *,
    fps: float,
    config: SonicQualityConfig,
    np: Any | None = None,
    model: G1KinematicModel | None = None,
) -> dict[str, object]:
    """Summarize quality metrics from already-loaded SONIC arrays."""

    np = np or _require_numpy()
    flags: list[str] = []
    joint_pos = np.asarray(arrays["joint_pos"], dtype=float)
    joint_vel = np.asarray(arrays["joint_vel"], dtype=float)
    body_pos = np.asarray(arrays["body_pos_w"], dtype=float)
    body_lin_vel = np.asarray(arrays["body_lin_vel_w"], dtype=float)
    original_frame_count = int(joint_pos.shape[0]) if joint_pos.ndim >= 1 else 0

    if config.frame_stride > 1:
        joint_pos = joint_pos[:: config.frame_stride]
        joint_vel = joint_vel[:: config.frame_stride]
        body_pos = body_pos[:: config.frame_stride]
        body_lin_vel = body_lin_vel[:: config.frame_stride]
    if config.max_frames is not None:
        joint_pos = joint_pos[: config.max_frames]
        joint_vel = joint_vel[: config.max_frames]
        body_pos = body_pos[: config.max_frames]
        body_lin_vel = body_lin_vel[: config.max_frames]

    effective_fps = fps / config.frame_stride
    frame_count = int(joint_pos.shape[0]) if joint_pos.ndim >= 1 else 0
    shape_ok = (
        joint_pos.ndim == 2
        and joint_pos.shape[-1] == len(SONIC_JOINT_NAMES)
        and joint_vel.shape == joint_pos.shape
        and body_pos.ndim == 3
        and body_pos.shape[1:] == (len(SONIC_BODY_NAMES), 3)
        and body_lin_vel.shape == body_pos.shape
        and body_pos.shape[0] == frame_count
    )
    if not shape_ok:
        return _error_result(
            base,
            "sonic_schema_mismatch",
            fps=fps,
            frame_count=frame_count,
            original_frame_count=original_frame_count,
        )

    nonfinite_values = int(
        np.size(joint_pos)
        - np.isfinite(joint_pos).sum()
        + np.size(joint_vel)
        - np.isfinite(joint_vel).sum()
        + np.size(body_pos)
        - np.isfinite(body_pos).sum()
        + np.size(body_lin_vel)
        - np.isfinite(body_lin_vel).sum()
    )
    if frame_count == 0:
        return _error_result(
            base,
            "empty_motion",
            fps=fps,
            frame_count=frame_count,
            original_frame_count=original_frame_count,
        )
    if nonfinite_values:
        flags.append("nonfinite_value")

    abs_joint_velocity = np.abs(joint_vel)
    joint_step_velocity = (
        np.abs(np.diff(joint_pos, axis=0)) * effective_fps
        if frame_count >= 2
        else np.zeros((0, len(SONIC_JOINT_NAMES)))
    )
    joint_acceleration = (
        np.abs(np.diff(joint_vel, axis=0)) * effective_fps
        if frame_count >= 2
        else np.zeros((0, len(SONIC_JOINT_NAMES)))
    )
    pelvis_pos = body_pos[:, 0, :]
    pelvis_vel = body_lin_vel[:, 0, :]
    root_speed = np.linalg.norm(pelvis_vel, axis=1)
    root_step_speed = (
        np.linalg.norm(np.diff(pelvis_pos, axis=0), axis=1) * effective_fps
        if frame_count >= 2
        else np.zeros((0,))
    )
    root_acceleration = (
        np.linalg.norm(np.diff(pelvis_vel, axis=0), axis=1) * effective_fps
        if frame_count >= 2
        else np.zeros((0,))
    )
    start_end_root_speed = _start_end_values(root_speed, config.start_end_frames)
    contact_stats = _contact_stats(body_pos, body_lin_vel, config, effective_fps, np)
    joint_limit_stats = _joint_limit_stats(joint_pos, model, np)
    self_collision_stats = _self_collision_proxy_stats(body_pos, config, np)

    joint_jump_rate = _rate_above_np(abs_joint_velocity, config.max_joint_velocity, np)
    joint_step_jump_rate = _rate_above_np(joint_step_velocity, config.max_joint_step_velocity, np)
    root_jump_rate = _rate_above_np(root_speed, config.max_root_speed, np)
    root_step_jump_rate = _rate_above_np(root_step_speed, config.max_root_step_speed, np)
    joint_acceleration_jump_rate = (
        _rate_above_np(joint_acceleration, config.max_joint_acceleration, np)
        if config.max_joint_acceleration is not None
        else 0.0
    )
    root_acceleration_jump_rate = (
        _rate_above_np(root_acceleration, config.max_root_acceleration, np)
        if config.max_root_acceleration is not None
        else 0.0
    )
    max_start_end_root_speed = _max_np(start_end_root_speed, np)

    action = "keep"
    if nonfinite_values:
        action = "exclude"
    if joint_jump_rate > 0.0:
        flags.append("sonic_joint_velocity_jump")
        action = _worse_action(action, "quarantine")
    if joint_step_jump_rate > 0.0:
        flags.append("sonic_joint_position_jump")
        action = _worse_action(action, "quarantine")
    if root_jump_rate > 0.0 or root_step_jump_rate > 0.0:
        flags.append("sonic_root_discontinuity")
        action = _worse_action(action, "quarantine")
    if joint_acceleration_jump_rate > 0.0:
        flags.append("sonic_joint_acceleration_jump")
        action = _worse_action(action, "quarantine")
    if root_acceleration_jump_rate > 0.0:
        flags.append("sonic_root_acceleration_jump")
        action = _worse_action(action, "quarantine")
    if max_start_end_root_speed > config.max_start_end_root_speed:
        flags.append("sonic_unstable_start_end")
        action = _worse_action(action, "downweight")
    if (
        config.enable_joint_limit_flags
        and joint_limit_stats.get("joint_limit_violation_rate", 0.0)
        > config.max_joint_limit_violation_rate
    ):
        flags.append("sonic_joint_limit_violation")
        action = _worse_action(action, "quarantine")
    if config.enable_body_origin_contact_flags:
        if contact_stats.get("contact_frame_ratio", 0.0) < config.min_contact_frame_ratio:
            flags.append("sonic_low_foot_contact")
            action = _worse_action(action, "quarantine")
        if contact_stats.get("mean_foot_clearance", 0.0) > config.max_mean_foot_clearance:
            flags.append("sonic_foot_float")
            action = _worse_action(action, "quarantine")
        if contact_stats.get("contact_slide_rate", 0.0) > 0.0:
            flags.append("sonic_foot_slide")
            action = _worse_action(action, "downweight")
    if contact_stats.get("penetration_depth", 0.0) > config.max_penetration_depth:
        flags.append("sonic_ground_penetration")
        action = _worse_action(action, "quarantine")
    if (
        config.enable_body_origin_self_collision_flags
        and self_collision_stats.get("self_collision_proxy_rate", 0.0)
        > config.max_self_collision_proxy_rate
    ):
        flags.append("sonic_self_collision_proxy")
        action = _worse_action(action, "quarantine")

    result = {
        **base,
        "fps": round(fps, 6),
        "effective_fps": round(effective_fps, 6),
        "frame_count": frame_count,
        "original_frame_count": original_frame_count,
        "joint_dim": int(joint_pos.shape[-1]),
        "body_count": int(body_pos.shape[1]),
        "nonfinite_values": nonfinite_values,
        "max_abs_joint_velocity": _round(_max_np(abs_joint_velocity, np)),
        "mean_abs_joint_velocity": _round(_mean_np(abs_joint_velocity, np)),
        "joint_jump_rate": _round(joint_jump_rate),
        "max_abs_joint_step_velocity": _round(_max_np(joint_step_velocity, np)),
        "joint_step_jump_rate": _round(joint_step_jump_rate),
        "max_abs_joint_acceleration": _round(_max_np(joint_acceleration, np)),
        "mean_abs_joint_acceleration": _round(_mean_np(joint_acceleration, np)),
        "joint_acceleration_jump_rate": _round(joint_acceleration_jump_rate),
        "max_root_speed": _round(_max_np(root_speed, np)),
        "root_jump_rate": _round(root_jump_rate),
        "max_root_step_speed": _round(_max_np(root_step_speed, np)),
        "root_step_jump_rate": _round(root_step_jump_rate),
        "max_root_acceleration": _round(_max_np(root_acceleration, np)),
        "mean_root_acceleration": _round(_mean_np(root_acceleration, np)),
        "root_acceleration_jump_rate": _round(root_acceleration_jump_rate),
        "max_start_end_root_speed": _round(max_start_end_root_speed),
        "quality_action": action,
        "quality_flags": "|".join(flags),
    }
    result.update(_rounded_dict(contact_stats))
    result.update(_rounded_dict(joint_limit_stats))
    result.update(_rounded_dict(self_collision_stats))
    return result


def _contact_stats(
    body_pos: Any,
    body_lin_vel: Any,
    config: SonicQualityConfig,
    fps: float,
    np: Any,
) -> dict[str, float]:
    foot_indices = _body_indices(config.foot_bodies)
    if not foot_indices:
        return _empty_contact_stats()
    foot_pos = body_pos[:, foot_indices, :]
    foot_vel = body_lin_vel[:, foot_indices, :]
    foot_clearance = foot_pos[:, :, 2] - config.ground_height
    frame_min_foot_clearance = np.min(foot_clearance, axis=1)
    contact_mask = foot_clearance <= config.contact_height_threshold
    contact_frame_mask = np.any(contact_mask, axis=1)
    horizontal_speed = np.linalg.norm(foot_vel[:, :, :2], axis=2)
    contact_slide_values = horizontal_speed[contact_mask]
    body_clearance = body_pos[:, :, 2] - config.ground_height
    skate_distances = _contact_skate_distances(foot_pos, contact_mask, config, np)
    support_distances = _support_distances(body_pos[:, 0, :], foot_pos, contact_mask, np)
    return {
        "min_foot_height": float(np.min(foot_pos[:, :, 2])),
        "mean_foot_clearance": float(np.mean(frame_min_foot_clearance)),
        "max_foot_clearance": float(np.max(frame_min_foot_clearance)),
        "min_body_height": float(np.min(body_pos[:, :, 2])),
        "penetration_depth": float(max(0.0, config.ground_height - float(np.min(body_pos[:, :, 2])))),
        "contact_frame_ratio": float(np.mean(contact_frame_mask)),
        "contact_slide_rate": _rate_above_np(
            contact_slide_values, config.max_contact_slide_speed, np
        ),
        "max_contact_slide_speed": _max_np(contact_slide_values, np),
        "contact_skate_rate": _rate_above_np(
            skate_distances, config.max_contact_skate_distance, np
        ),
        "max_contact_skate_distance": _max_np(skate_distances, np),
        "root_to_support_mean_distance": _mean_np(support_distances, np),
        "root_to_support_max_distance": _max_np(support_distances, np),
        "body_penetration_frame_ratio": float(
            np.mean(np.min(body_clearance, axis=1) < -config.max_penetration_depth)
        ),
    }


def _joint_limit_stats(joint_pos: Any, model: G1KinematicModel | None, np: Any) -> dict[str, float]:
    if model is None:
        return {
            "joint_limit_checked_values": 0.0,
            "joint_limit_violation_rate": 0.0,
            "max_joint_limit_violation": 0.0,
        }
    violations = []
    total = 0
    for joint_index, joint_name in enumerate(SONIC_JOINT_NAMES):
        joint_range = model.joint_ranges.get(joint_name)
        if joint_range is None:
            continue
        lower, upper = joint_range
        values = joint_pos[:, joint_index]
        lower_violation = np.maximum(lower - values, 0.0)
        upper_violation = np.maximum(values - upper, 0.0)
        margin = np.maximum(lower_violation, upper_violation)
        violations.append(margin)
        total += int(values.size)
    if not violations:
        return {
            "joint_limit_checked_values": 0.0,
            "joint_limit_violation_rate": 0.0,
            "max_joint_limit_violation": 0.0,
        }
    margins = np.concatenate(violations)
    return {
        "joint_limit_checked_values": float(total),
        "joint_limit_violation_rate": float(np.mean(margins > 0.0)),
        "max_joint_limit_violation": _max_np(margins, np),
    }


def _self_collision_proxy_stats(
    body_pos: Any,
    config: SonicQualityConfig,
    np: Any,
) -> dict[str, float]:
    pairs = _self_collision_body_pairs(config.min_self_collision_kinematic_hops)
    if not pairs:
        return {
            "self_collision_checked_pairs": 0.0,
            "self_collision_proxy_rate": 0.0,
            "min_self_collision_distance": 0.0,
            "mean_min_self_collision_distance": 0.0,
        }
    pair_distances = []
    for left, right in pairs:
        pair_distances.append(np.linalg.norm(body_pos[:, left, :] - body_pos[:, right, :], axis=1))
    distances = np.stack(pair_distances, axis=1)
    per_frame_min = np.min(distances, axis=1)
    return {
        "self_collision_checked_pairs": float(len(pairs)),
        "self_collision_proxy_rate": float(np.mean(per_frame_min < config.min_self_collision_distance)),
        "min_self_collision_distance": _max_np(-per_frame_min, np) * -1.0,
        "mean_min_self_collision_distance": _mean_np(per_frame_min, np),
    }


def _contact_skate_distances(
    foot_pos: Any,
    contact_mask: Any,
    config: SonicQualityConfig,
    np: Any,
) -> Any:
    distances: list[float] = []
    for foot_index in range(foot_pos.shape[1]):
        contact = contact_mask[:, foot_index]
        start: int | None = None
        for frame_index, is_contact in enumerate(contact):
            if bool(is_contact) and start is None:
                start = frame_index
            if (not bool(is_contact) or frame_index == len(contact) - 1) and start is not None:
                end = frame_index + 1 if bool(is_contact) and frame_index == len(contact) - 1 else frame_index
                segment = foot_pos[start:end, foot_index, :2]
                if len(segment) > 1:
                    offsets = np.linalg.norm(segment - segment[0], axis=1)
                    distances.append(float(np.max(offsets)))
                start = None
    return np.asarray(distances, dtype=float)


def _support_distances(root_pos: Any, foot_pos: Any, contact_mask: Any, np: Any) -> Any:
    distances: list[float] = []
    for frame_index in range(root_pos.shape[0]):
        contact_feet = foot_pos[frame_index, contact_mask[frame_index], :2]
        if contact_feet.size == 0:
            continue
        distances.append(float(np.min(np.linalg.norm(contact_feet - root_pos[frame_index, :2], axis=1))))
    return np.asarray(distances, dtype=float)


def _self_collision_body_pairs(min_hops: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(len(SONIC_BODY_NAMES)):
        for right in range(left + 1, len(SONIC_BODY_NAMES)):
            if _kinematic_hop_distance(left, right) >= min_hops:
                pairs.append((left, right))
    return pairs


def _kinematic_hop_distance(left: int, right: int) -> int:
    left_depths = _ancestor_depths(left)
    node: int | None = right
    depth = 0
    while node is not None:
        if node in left_depths:
            return depth + left_depths[node]
        node = SONIC_BODY_PARENTS[node]
        depth += 1
    return 999


def _ancestor_depths(index: int) -> dict[int, int]:
    depths: dict[int, int] = {}
    node: int | None = index
    depth = 0
    while node is not None:
        depths[node] = depth
        node = SONIC_BODY_PARENTS[node]
        depth += 1
    return depths


def _body_indices(names: Sequence[str]) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(SONIC_BODY_NAMES)}
    return [index_by_name[name] for name in names if name in index_by_name]


def _start_end_values(values: Any, count: int) -> Any:
    if count <= 0 or len(values) <= count * 2:
        return values
    return values.take(list(range(count)) + list(range(len(values) - count, len(values))), axis=0)


def _iter_index_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("schema_status", "ok") == "ok"
        ]


def _error_result(
    base: Mapping[str, object],
    flag: str,
    *,
    fps: float = 0.0,
    frame_count: int = 0,
    original_frame_count: int = 0,
) -> dict[str, object]:
    return {
        **base,
        "fps": round(fps, 6),
        "effective_fps": round(fps, 6),
        "frame_count": frame_count,
        "original_frame_count": original_frame_count,
        "joint_dim": 0,
        "body_count": 0,
        "nonfinite_values": 0,
        "quality_action": "exclude",
        "quality_flags": flag,
        **_zero_metrics(),
    }


def _empty_contact_stats() -> dict[str, float]:
    return {
        "min_foot_height": 0.0,
        "mean_foot_clearance": 0.0,
        "max_foot_clearance": 0.0,
        "min_body_height": 0.0,
        "penetration_depth": 0.0,
        "contact_frame_ratio": 0.0,
        "contact_slide_rate": 0.0,
        "max_contact_slide_speed": 0.0,
        "contact_skate_rate": 0.0,
        "max_contact_skate_distance": 0.0,
        "root_to_support_mean_distance": 0.0,
        "root_to_support_max_distance": 0.0,
        "body_penetration_frame_ratio": 0.0,
    }


def _zero_metrics() -> dict[str, float]:
    metrics = {
        "max_abs_joint_velocity": 0.0,
        "mean_abs_joint_velocity": 0.0,
        "joint_jump_rate": 0.0,
        "max_abs_joint_step_velocity": 0.0,
        "joint_step_jump_rate": 0.0,
        "max_abs_joint_acceleration": 0.0,
        "mean_abs_joint_acceleration": 0.0,
        "joint_acceleration_jump_rate": 0.0,
        "max_root_speed": 0.0,
        "root_jump_rate": 0.0,
        "max_root_step_speed": 0.0,
        "root_step_jump_rate": 0.0,
        "max_root_acceleration": 0.0,
        "mean_root_acceleration": 0.0,
        "root_acceleration_jump_rate": 0.0,
        "max_start_end_root_speed": 0.0,
    }
    metrics.update(_empty_contact_stats())
    metrics.update(
        {
            "joint_limit_checked_values": 0.0,
            "joint_limit_violation_rate": 0.0,
            "max_joint_limit_violation": 0.0,
            "self_collision_checked_pairs": 0.0,
            "self_collision_proxy_rate": 0.0,
            "min_self_collision_distance": 0.0,
            "mean_min_self_collision_distance": 0.0,
        }
    )
    return metrics


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = [
        "fps",
        "frame_count",
        "max_abs_joint_velocity",
        "joint_jump_rate",
        "max_abs_joint_step_velocity",
        "max_root_speed",
        "max_root_step_speed",
        "max_start_end_root_speed",
        "contact_frame_ratio",
        "mean_foot_clearance",
        "penetration_depth",
        "contact_slide_rate",
        "max_contact_slide_speed",
        "self_collision_proxy_rate",
        "min_self_collision_distance",
        "joint_limit_violation_rate",
    ]
    return {metric: _summary_for_metric(rows, metric) for metric in metrics}


def _summary_for_metric(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, float]:
    values = sorted(_float(row.get(metric)) for row in rows if row.get(metric) not in (None, ""))
    if not values:
        return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    p95_index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return {
        "min": _round(values[0]),
        "mean": _round(sum(values) / len(values)),
        "p95": _round(values[p95_index]),
        "max": _round(values[-1]),
    }


def _validate_config(config: SonicQualityConfig) -> None:
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive when set")
    if config.max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity must be positive")
    if config.max_joint_step_velocity <= 0:
        raise ValueError("max_joint_step_velocity must be positive")
    if config.max_root_speed <= 0:
        raise ValueError("max_root_speed must be positive")
    if config.max_root_step_speed <= 0:
        raise ValueError("max_root_step_speed must be positive")
    if not 0.0 <= config.min_contact_frame_ratio <= 1.0:
        raise ValueError("min_contact_frame_ratio must be within [0, 1]")
    if config.min_self_collision_kinematic_hops <= 0:
        raise ValueError("min_self_collision_kinematic_hops must be positive")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SONIC quality scanning requires numpy. Use the project conda environment or another "
            "Python environment with numpy installed."
        ) from exc
    return np


def _rate_above_np(values: Any, threshold: float | None, np: Any) -> float:
    if threshold is None or values.size == 0:
        return 0.0
    return float(np.mean(values > threshold))


def _max_np(values: Any, np: Any) -> float:
    return float(np.max(values)) if values.size else 0.0


def _mean_np(values: Any, np: Any) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _rounded_dict(payload: Mapping[str, float]) -> dict[str, float]:
    return {key: _round(value) for key, value in payload.items()}


def _round(value: float) -> float:
    return round(float(value), 6)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _split_flags(flags: str) -> list[str]:
    return [flag for flag in flags.split("|") if flag]


def _worse_action(left: str, right: str) -> str:
    order = {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}
    return left if order[left] >= order[right] else right


def _model_report(model: G1KinematicModel | None) -> dict[str, object]:
    if model is None:
        return {
            "source_xml": "",
            "joint_limit_count": 0,
            "note": "No model XML was supplied, so joint-limit metrics are zero-filled.",
        }
    return {"source_xml": model.source_xml, "joint_limit_count": len(model.joint_ranges)}


def _write_jsonl_row(handle: Any, row: Mapping[str, object]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


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
