"""G1 target motion quality scanning from BONES-SEED CSV targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from .bones_seed import G1_JOINT_COLUMNS
from .row_sampling import sampling_run_tag, scan_sampling_report, select_rows_for_scan


ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
DEFAULT_FOOT_BODIES = (
    "left_ankle_roll_link",
    "left_toe_link",
    "right_ankle_roll_link",
    "right_toe_link",
)
DEFAULT_SELF_COLLISION_IGNORE_BODIES = (
    "imu_in_torso",
    "head_mocap",
    "pelvis_contour_link",
)


@dataclass(frozen=True)
class G1QualityConfig:
    fps: float = 120.0
    max_joint_velocity: float = 20.0
    max_root_speed: float = 8.0
    max_joint_acceleration: float | None = None
    max_root_acceleration: float | None = None
    max_root_jerk: float | None = None
    root_position_scale: float = 0.01
    joint_angle_scale: float = math.pi / 180.0
    root_rotation_scale: float = math.pi / 180.0
    frame_stride: int = 1
    max_frames: int | None = None
    model_xml: Path | None = None
    ground_height: float = 0.0
    contact_height_threshold: float = 0.04
    max_contact_slide_speed: float = 0.25
    max_mean_foot_clearance: float = 0.10
    max_penetration_depth: float = 0.03
    max_contact_correction_offset: float = 0.15
    min_contact_frame_ratio: float = 0.05
    max_joint_limit_violation_rate: float = 0.0
    start_end_frames: int = 10
    max_start_end_root_speed: float = 0.20
    min_self_collision_distance: float = 0.015
    max_self_collision_proxy_rate: float = 0.0
    min_self_collision_kinematic_hops: int = 4
    foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES
    self_collision_ignore_bodies: tuple[str, ...] = DEFAULT_SELF_COLLISION_IGNORE_BODIES

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_xml"] = str(self.model_xml) if self.model_xml is not None else None
        payload["foot_bodies"] = list(self.foot_bodies)
        payload["self_collision_ignore_bodies"] = list(self.self_collision_ignore_bodies)
        return payload


@dataclass(frozen=True)
class G1MJCFBody:
    name: str
    parent: int | None
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    joint_name: str | None
    joint_axis: tuple[float, float, float] | None
    joint_range: tuple[float, float] | None
    has_freejoint: bool
    geom_points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class G1KinematicModel:
    bodies: tuple[G1MJCFBody, ...]
    joint_ranges: dict[str, tuple[float, float]]
    foot_body_names: tuple[str, ...]
    source_xml: str


@dataclass(frozen=True)
class G1QualityScanResult:
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


def scan_g1_quality_from_index(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: G1QualityConfig | None = None,
    limit: int | None = 100,
    splits: Sequence[str] = (),
    actions: Sequence[str] = ("keep", "downweight", "quarantine"),
    sample_by: Sequence[str] = (),
) -> G1QualityScanResult:
    """Scan G1 CSV targets referenced by a split index."""

    config = config or G1QualityConfig()
    _validate_config(config)
    model = load_g1_kinematic_model(config.model_xml, config.foot_bodies) if config.model_xml else None

    candidate_rows = list(_iter_index_rows(index_csv, splits=splits, actions=actions))
    rows = select_rows_for_scan(candidate_rows, limit=limit, sample_by=sample_by)

    output_dir = output_root.expanduser() / "quality" / _quality_run_name(index_csv, limit, sample_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = output_dir / "g1_quality_stats.jsonl"
    report_json = output_dir / "g1_quality_report.json"

    g1_tar = data_root.expanduser() / "g1.tar"
    scanned: list[dict[str, object]] = []
    skipped_rows = 0
    with tarfile.open(g1_tar, "r:*") as tar:
        for row in rows:
            target_path = row.get("move_g1_path", "")
            if not target_path:
                skipped_rows += 1
                continue
            scanned.append(scan_g1_csv_member(tar, row, config, model=model))

    _write_jsonl(stats_jsonl, scanned)
    action_counts = Counter(str(row["quality_action"]) for row in scanned)
    flag_counts = Counter()
    for row in scanned:
        for flag in _split_flags(str(row["quality_flags"])):
            flag_counts[flag] += 1

    report = {
        "data_root": str(data_root),
        "index_csv": str(index_csv),
        "stats_jsonl": str(stats_jsonl),
        "config": config.to_dict(),
        "model": _model_report(model),
        "limit": limit,
        "filters": {"splits": list(splits), "actions": list(actions)},
        "sampling": scan_sampling_report(candidate_rows, rows, limit=limit, sample_by=sample_by),
        "scanned_rows": len(scanned),
        "skipped_rows": skipped_rows,
        "action_counts": dict(sorted(action_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "metric_summary": _metric_summary(scanned),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return G1QualityScanResult(
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


def scan_g1_csv_member(
    tar: tarfile.TarFile,
    index_row: Mapping[str, str],
    config: G1QualityConfig,
    model: G1KinematicModel | None = None,
) -> dict[str, object]:
    """Scan one G1 CSV member from an open tar archive."""

    target_path = index_row.get("move_g1_path", "")
    base = {
        "row_index": index_row.get("row_index", ""),
        "split": index_row.get("split", ""),
        "actor_uid": index_row.get("actor_uid", ""),
        "move_name": index_row.get("move_name", ""),
        "filename": index_row.get("filename", ""),
        "package": index_row.get("package", ""),
        "category": index_row.get("category", ""),
        "is_mirror": index_row.get("is_mirror", ""),
        "actor_gender": index_row.get("actor_gender", ""),
        "move_g1_path": target_path,
    }
    try:
        member = tar.getmember(target_path)
        extracted = tar.extractfile(member)
    except (KeyError, tarfile.TarError):
        return _empty_result(base, "missing_g1_csv_member", model)
    if extracted is None:
        return _empty_result(base, "empty_g1_csv_member", model)

    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8", newline="")
            rows = list(csv.DictReader(text))
        except UnicodeDecodeError:
            return _empty_result(base, "g1_csv_decode_error", model)

    return summarize_g1_rows(base, rows, config, model=model)


def summarize_g1_rows(
    base: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    config: G1QualityConfig,
    model: G1KinematicModel | None = None,
) -> dict[str, object]:
    """Summarize target-side G1 trajectory quality from CSV rows."""

    flags: list[str] = []
    nonfinite_values = 0
    prev_joints: list[float] | None = None
    prev_root: list[float] | None = None
    joint_velocity_samples: list[float] = []
    joint_acceleration_samples: list[float] = []
    root_speed_samples: list[float] = []
    root_acceleration_samples: list[float] = []
    root_jerk_samples: list[float] = []
    start_end_root_speed_samples: list[float] = []
    root_heights: list[float] = []
    parsed_frames: list[tuple[list[float], list[float], list[float], int]] = []

    sampled_rows = rows[:: config.frame_stride]
    if config.max_frames is not None:
        sampled_rows = sampled_rows[: config.max_frames]
    effective_fps = config.fps / config.frame_stride

    for sampled_index, row in enumerate(sampled_rows):
        joints_raw = [_parse_float(row.get(column)) for column in G1_JOINT_COLUMNS]
        root_raw = [_parse_float(row.get(column)) for column in ROOT_TRANSLATE_COLUMNS]
        root_euler_raw = [_parse_float(row.get(column)) for column in ROOT_ROTATE_COLUMNS]
        if any(value is None for value in joints_raw + root_raw + root_euler_raw):
            nonfinite_values += sum(value is None for value in joints_raw + root_raw + root_euler_raw)
            continue
        typed_joints = [
            float(value) * config.joint_angle_scale for value in joints_raw if value is not None
        ]
        typed_root = [
            float(value) * config.root_position_scale for value in root_raw if value is not None
        ]
        typed_root_euler = [
            float(value) * config.root_rotation_scale for value in root_euler_raw if value is not None
        ]
        if prev_joints is not None:
            joint_velocity_samples.extend(
                abs(cur - prev) * effective_fps for cur, prev in zip(typed_joints, prev_joints)
            )
        if len(parsed_frames) >= 2:
            previous_joints = parsed_frames[-1][0]
            previous_previous_joints = parsed_frames[-2][0]
            joint_acceleration_samples.extend(
                abs(cur - (2.0 * prev) + prev_prev) * effective_fps * effective_fps
                for cur, prev, prev_prev in zip(
                    typed_joints, previous_joints, previous_previous_joints
                )
            )
        if prev_root is not None:
            root_speed = math.dist(typed_root, prev_root) * effective_fps
            root_speed_samples.append(root_speed)
            if sampled_index < config.start_end_frames or sampled_index >= len(sampled_rows) - config.start_end_frames:
                start_end_root_speed_samples.append(root_speed)
        if len(parsed_frames) >= 2:
            previous_root = parsed_frames[-1][1]
            previous_previous_root = parsed_frames[-2][1]
            root_acceleration_samples.append(
                _vector_second_difference_norm(typed_root, previous_root, previous_previous_root)
                * effective_fps
                * effective_fps
            )
        if len(parsed_frames) >= 3:
            previous_root = parsed_frames[-1][1]
            previous_previous_root = parsed_frames[-2][1]
            previous_previous_previous_root = parsed_frames[-3][1]
            root_jerk_samples.append(
                _vector_third_difference_norm(
                    typed_root,
                    previous_root,
                    previous_previous_root,
                    previous_previous_previous_root,
                )
                * effective_fps
                * effective_fps
                * effective_fps
            )
        root_heights.append(typed_root[2])
        parsed_frames.append((typed_joints, typed_root, typed_root_euler, sampled_index))
        prev_joints = typed_joints
        prev_root = typed_root

    frame_count = len(sampled_rows)
    max_abs_joint_velocity = max(joint_velocity_samples) if joint_velocity_samples else 0.0
    mean_abs_joint_velocity = (
        sum(joint_velocity_samples) / len(joint_velocity_samples)
        if joint_velocity_samples
        else 0.0
    )
    joint_jump_rate = _rate_above(joint_velocity_samples, config.max_joint_velocity)
    max_abs_joint_acceleration = (
        max(joint_acceleration_samples) if joint_acceleration_samples else 0.0
    )
    mean_abs_joint_acceleration = (
        sum(joint_acceleration_samples) / len(joint_acceleration_samples)
        if joint_acceleration_samples
        else 0.0
    )
    joint_acceleration_jump_rate = (
        _rate_above(joint_acceleration_samples, config.max_joint_acceleration)
        if config.max_joint_acceleration is not None
        else 0.0
    )
    max_root_speed = max(root_speed_samples) if root_speed_samples else 0.0
    root_jump_rate = _rate_above(root_speed_samples, config.max_root_speed)
    max_root_acceleration = max(root_acceleration_samples) if root_acceleration_samples else 0.0
    mean_root_acceleration = (
        sum(root_acceleration_samples) / len(root_acceleration_samples)
        if root_acceleration_samples
        else 0.0
    )
    root_acceleration_jump_rate = (
        _rate_above(root_acceleration_samples, config.max_root_acceleration)
        if config.max_root_acceleration is not None
        else 0.0
    )
    max_root_jerk = max(root_jerk_samples) if root_jerk_samples else 0.0
    mean_root_jerk = sum(root_jerk_samples) / len(root_jerk_samples) if root_jerk_samples else 0.0
    root_jerk_jump_rate = (
        _rate_above(root_jerk_samples, config.max_root_jerk)
        if config.max_root_jerk is not None
        else 0.0
    )
    max_start_end_root_speed = (
        max(start_end_root_speed_samples) if start_end_root_speed_samples else 0.0
    )
    root_height_min = min(root_heights) if root_heights else 0.0
    root_height_max = max(root_heights) if root_heights else 0.0
    root_height_range = root_height_max - root_height_min

    joint_limit_stats = _joint_limit_stats(parsed_frames, model) if model is not None else {}
    contact_stats = _contact_stats(parsed_frames, model, config, effective_fps) if model is not None else {}
    self_collision_stats = (
        _self_collision_proxy_stats(parsed_frames, model, config) if model is not None else {}
    )

    action = "keep"
    if frame_count == 0:
        flags.append("empty_motion")
        action = "exclude"
    if nonfinite_values > 0:
        flags.append("nonfinite_value")
        action = "exclude"
    if frame_count == 1:
        flags.append("single_frame_motion")
        action = _worse_action(action, "quarantine")
    if joint_jump_rate > 0.0:
        flags.append("joint_velocity_jump")
        action = _worse_action(action, "quarantine")
    if root_jump_rate > 0.0:
        flags.append("root_discontinuity")
        action = _worse_action(action, "quarantine")
    if joint_acceleration_jump_rate > 0.0:
        flags.append("g1_joint_acceleration_jump")
        action = _worse_action(action, "quarantine")
    if root_acceleration_jump_rate > 0.0:
        flags.append("g1_root_acceleration_jump")
        action = _worse_action(action, "quarantine")
    if root_jerk_jump_rate > 0.0:
        flags.append("g1_root_jerk_jump")
        action = _worse_action(action, "quarantine")
    if max_start_end_root_speed > config.max_start_end_root_speed:
        flags.append("g1_unstable_start_end")
        action = _worse_action(action, "downweight")
    if joint_limit_stats.get("joint_limit_violation_rate", 0.0) > config.max_joint_limit_violation_rate:
        flags.append("g1_joint_limit_violation")
        action = _worse_action(action, "quarantine")
    if contact_stats:
        if contact_stats.get("contact_frame_ratio", 0.0) < config.min_contact_frame_ratio:
            flags.append("g1_low_foot_contact")
            action = _worse_action(action, "quarantine")
        if contact_stats.get("mean_foot_clearance", 0.0) > config.max_mean_foot_clearance:
            flags.append("g1_foot_float")
            action = _worse_action(action, "quarantine")
        if contact_stats.get("penetration_depth", 0.0) > config.max_penetration_depth:
            flags.append("g1_ground_penetration")
            action = _worse_action(action, "quarantine")
        if contact_stats.get("contact_slide_rate", 0.0) > 0.0:
            flags.append("g1_foot_slide")
            action = _worse_action(action, "downweight")
    if self_collision_stats:
        if (
            self_collision_stats.get("self_collision_proxy_rate", 0.0)
            > config.max_self_collision_proxy_rate
        ):
            flags.append("g1_self_collision_proxy")
            action = _worse_action(action, "quarantine")

    result = {
        **base,
        "frame_count": frame_count,
        "original_frame_count": len(rows),
        "joint_dim": len(G1_JOINT_COLUMNS),
        "nonfinite_values": nonfinite_values,
        "root_position_scale": config.root_position_scale,
        "joint_angle_scale": config.joint_angle_scale,
        "max_abs_joint_velocity": round(max_abs_joint_velocity, 6),
        "mean_abs_joint_velocity": round(mean_abs_joint_velocity, 6),
        "joint_jump_rate": round(joint_jump_rate, 6),
        "max_abs_joint_acceleration": round(max_abs_joint_acceleration, 6),
        "mean_abs_joint_acceleration": round(mean_abs_joint_acceleration, 6),
        "joint_acceleration_jump_rate": round(joint_acceleration_jump_rate, 6),
        "max_root_speed": round(max_root_speed, 6),
        "root_jump_rate": round(root_jump_rate, 6),
        "max_root_acceleration": round(max_root_acceleration, 6),
        "mean_root_acceleration": round(mean_root_acceleration, 6),
        "root_acceleration_jump_rate": round(root_acceleration_jump_rate, 6),
        "max_root_jerk": round(max_root_jerk, 6),
        "mean_root_jerk": round(mean_root_jerk, 6),
        "root_jerk_jump_rate": round(root_jerk_jump_rate, 6),
        "max_start_end_root_speed": round(max_start_end_root_speed, 6),
        "root_height_min": round(root_height_min, 6),
        "root_height_max": round(root_height_max, 6),
        "root_height_range": round(root_height_range, 6),
        "model_xml": model.source_xml if model is not None else "",
        "foot_bodies": "|".join(model.foot_body_names) if model is not None else "",
        "quality_mode": "mjcf_fk" if model is not None else "csv_root_joint",
        "quality_action": action,
        "quality_flags": "|".join(flags),
    }
    if model is not None:
        result.update(_rounded_dict(joint_limit_stats))
        result.update(_rounded_contact_stats(contact_stats))
        result.update(_rounded_dict(self_collision_stats))
    else:
        result.update(_empty_model_metrics())
    return result


def _iter_index_rows(
    index_csv: Path,
    splits: Sequence[str],
    actions: Sequence[str],
) -> Iterable[dict[str, str]]:
    split_filter = set(splits)
    action_filter = set(actions)
    with index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split_filter and row.get("split") not in split_filter:
                continue
            if action_filter and row.get("curation_action") not in action_filter:
                continue
            yield row


def load_g1_kinematic_model(
    model_xml: Path | None,
    foot_body_names: Sequence[str] = DEFAULT_FOOT_BODIES,
) -> G1KinematicModel:
    """Load the small MJCF subset needed for standard-library FK quality scans."""

    if model_xml is None:
        raise ValueError("model_xml is required")
    root = ET.parse(model_xml).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF missing worldbody: {model_xml}")

    bodies: list[G1MJCFBody] = []
    joint_ranges: dict[str, tuple[float, float]] = {}

    def visit(element: ET.Element, parent: int | None) -> None:
        if element.tag != "body":
            return
        body_index = len(bodies)
        joint_element = element.find("joint")
        joint_name: str | None = None
        joint_axis: tuple[float, float, float] | None = None
        joint_range: tuple[float, float] | None = None
        if joint_element is not None:
            joint_name = joint_element.attrib.get("name")
            joint_axis = _parse_vec3(joint_element.attrib.get("axis", "0 0 1"))
            if "range" in joint_element.attrib:
                parsed_range = _parse_float_tuple(joint_element.attrib["range"])
                if len(parsed_range) >= 2:
                    joint_range = (parsed_range[0], parsed_range[1])
                    if joint_name:
                        joint_ranges[joint_name] = joint_range
        body = G1MJCFBody(
            name=element.attrib.get("name", f"body_{body_index}"),
            parent=parent,
            pos=_parse_vec3(element.attrib.get("pos", "0 0 0")),
            quat=_parse_quat(element.attrib.get("quat", "1 0 0 0")),
            joint_name=joint_name,
            joint_axis=joint_axis,
            joint_range=joint_range,
            has_freejoint=element.find("freejoint") is not None,
            geom_points=_local_geom_points(element),
        )
        bodies.append(body)
        for child in element:
            if child.tag == "body":
                visit(child, body_index)

    for child in worldbody:
        if child.tag == "body":
            visit(child, None)

    body_names = {body.name for body in bodies}
    present_feet = tuple(name for name in foot_body_names if name in body_names)
    if not present_feet:
        raise ValueError(f"None of the requested foot bodies are present in {model_xml}")
    return G1KinematicModel(
        bodies=tuple(bodies),
        joint_ranges=joint_ranges,
        foot_body_names=present_feet,
        source_xml=str(model_xml),
    )


def _quality_run_name(index_csv: Path, limit: int | None, sample_by: Sequence[str] = ()) -> str:
    return f"{index_csv.parent.name}_{sampling_run_tag(limit, sample_by)}"


def _joint_limit_stats(
    parsed_frames: Sequence[tuple[list[float], list[float], list[float], int]],
    model: G1KinematicModel | None,
) -> dict[str, float]:
    if model is None or not parsed_frames:
        return {}
    violations = 0
    margins: list[float] = []
    total = 0
    for joints, _, _, _ in parsed_frames:
        for column, value in zip(G1_JOINT_COLUMNS, joints):
            joint_name = _joint_name_from_column(column)
            joint_range = model.joint_ranges.get(joint_name)
            if joint_range is None:
                continue
            lower, upper = joint_range
            total += 1
            if value < lower:
                violations += 1
                margins.append(lower - value)
            elif value > upper:
                violations += 1
                margins.append(value - upper)
            else:
                margins.append(0.0)
    return {
        "joint_limit_checked_values": float(total),
        "joint_limit_violation_rate": violations / total if total else 0.0,
        "max_joint_limit_violation": max(margins) if margins else 0.0,
    }


def _contact_stats(
    parsed_frames: Sequence[tuple[list[float], list[float], list[float], int]],
    model: G1KinematicModel | None,
    config: G1QualityConfig,
    fps: float,
) -> dict[str, float]:
    if model is None or not parsed_frames:
        return {}

    fk_frames = [
        _g1_fk_positions(model, joints, root, root_euler)
        for joints, root, root_euler, _ in parsed_frames
    ]
    foot_heights: list[float] = []
    body_heights: list[float] = []
    for frame in fk_frames:
        foot_points = [
            point[2]
            for body in model.foot_body_names
            for point in frame.get(body, ())
        ]
        all_points = [point[2] for points in frame.values() for point in points]
        if foot_points:
            foot_heights.append(min(foot_points))
        if all_points:
            body_heights.append(min(all_points))
    if not foot_heights:
        return {}

    foot_clearances = [height - config.ground_height for height in foot_heights]
    body_clearances = [height - config.ground_height for height in body_heights]
    contact_flags = [clearance <= config.contact_height_threshold for clearance in foot_clearances]
    contact_frame_ratio = sum(contact_flags) / len(contact_flags) if contact_flags else 0.0
    contact_slide_speeds = _g1_contact_slide_speeds(fk_frames, model, config, fps)
    support_distances = _g1_root_support_distances(fk_frames, model, config)
    min_foot_clearance = min(foot_clearances) if foot_clearances else 0.0
    mean_foot_clearance = sum(foot_clearances) / len(foot_clearances)
    contact_correction = _contact_correction_candidate(
        min_foot_clearance=min_foot_clearance,
        mean_foot_clearance=mean_foot_clearance,
        config=config,
    )
    max_contact_slide_speed = max(contact_slide_speeds) if contact_slide_speeds else 0.0
    contact_slide_rate = _rate_above(contact_slide_speeds, config.max_contact_slide_speed)
    min_body_clearance = min(body_clearances) if body_clearances else 0.0
    return {
        "ground_height": config.ground_height,
        "min_foot_height": min(foot_heights),
        "mean_foot_clearance": mean_foot_clearance,
        "max_foot_clearance": max(foot_clearances),
        "min_body_clearance": min_body_clearance,
        "penetration_depth": max(0.0, -min_body_clearance),
        **contact_correction,
        "contact_frame_ratio": contact_frame_ratio,
        "max_contact_slide_speed": max_contact_slide_speed,
        "contact_slide_rate": contact_slide_rate,
        "support_frame_ratio": len(support_distances) / len(fk_frames) if fk_frames else 0.0,
        "mean_root_support_distance": (
            sum(support_distances) / len(support_distances) if support_distances else 0.0
        ),
        "max_root_support_distance": max(support_distances) if support_distances else 0.0,
    }


def _self_collision_proxy_stats(
    parsed_frames: Sequence[tuple[list[float], list[float], list[float], int]],
    model: G1KinematicModel | None,
    config: G1QualityConfig,
) -> dict[str, float]:
    if model is None or not parsed_frames:
        return {}

    body_pairs = _self_collision_body_pairs(model, config)
    if not body_pairs:
        return {
            "self_collision_checked_pairs": 0.0,
            "self_collision_proxy_rate": 0.0,
            "min_self_collision_distance": 0.0,
            "mean_min_self_collision_distance": 0.0,
        }

    min_distances: list[float] = []
    for joints, root, root_euler, _ in parsed_frames:
        frame = _g1_fk_positions(model, joints, root, root_euler, include_empty_body_origin=False)
        frame_min: float | None = None
        for left, right in body_pairs:
            for left_point in frame.get(left, ()):
                for right_point in frame.get(right, ()):
                    distance = math.dist(left_point, right_point)
                    if frame_min is None or distance < frame_min:
                        frame_min = distance
        if frame_min is not None:
            min_distances.append(frame_min)

    if not min_distances:
        return {
            "self_collision_checked_pairs": float(len(body_pairs)),
            "self_collision_proxy_rate": 0.0,
            "min_self_collision_distance": 0.0,
            "mean_min_self_collision_distance": 0.0,
        }
    return {
        "self_collision_checked_pairs": float(len(body_pairs)),
        "self_collision_proxy_rate": _rate_below(
            min_distances, config.min_self_collision_distance
        ),
        "min_self_collision_distance": min(min_distances),
        "mean_min_self_collision_distance": sum(min_distances) / len(min_distances),
    }


def _self_collision_body_pairs(
    model: G1KinematicModel,
    config: G1QualityConfig,
) -> tuple[tuple[str, str], ...]:
    ignored = set(config.self_collision_ignore_bodies)
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(model.bodies):
        if left.name in ignored:
            continue
        for right_index in range(left_index + 1, len(model.bodies)):
            right = model.bodies[right_index]
            if right.name in ignored:
                continue
            if _kinematic_hop_distance(model, left_index, right_index) < config.min_self_collision_kinematic_hops:
                continue
            pairs.append((left.name, right.name))
    return tuple(pairs)


def _kinematic_hop_distance(model: G1KinematicModel, left: int, right: int) -> int:
    left_depths = _ancestor_depths(model, left)
    current = right
    depth = 0
    while True:
        if current in left_depths:
            return depth + left_depths[current]
        parent = model.bodies[current].parent
        if parent is None:
            break
        current = parent
        depth += 1
    return len(model.bodies)


def _ancestor_depths(model: G1KinematicModel, body_index: int) -> dict[int, int]:
    depths = {body_index: 0}
    current = body_index
    depth = 0
    while model.bodies[current].parent is not None:
        current = model.bodies[current].parent  # type: ignore[assignment]
        depth += 1
        depths[current] = depth
    return depths


def _g1_contact_slide_speeds(
    fk_frames: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    model: G1KinematicModel,
    config: G1QualityConfig,
    fps: float,
) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(fk_frames, fk_frames[1:]):
        for body in model.foot_body_names:
            prev_points = previous.get(body, ())
            cur_points = current.get(body, ())
            if not prev_points or not cur_points:
                continue
            prev_low = min(prev_points, key=lambda point: point[2])
            cur_low = min(cur_points, key=lambda point: point[2])
            if (
                prev_low[2] - config.ground_height > config.contact_height_threshold
                or cur_low[2] - config.ground_height > config.contact_height_threshold
            ):
                continue
            horizontal_distance = math.dist((prev_low[0], prev_low[1]), (cur_low[0], cur_low[1]))
            speeds.append(horizontal_distance * fps)
    return speeds


def _g1_root_support_distances(
    fk_frames: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    model: G1KinematicModel,
    config: G1QualityConfig,
) -> list[float]:
    distances: list[float] = []
    for frame in fk_frames:
        pelvis_points = frame.get("pelvis", ())
        if not pelvis_points:
            continue
        root = pelvis_points[0]
        support_points = [
            (point[0], point[1])
            for body in model.foot_body_names
            for point in frame.get(body, ())
            if point[2] - config.ground_height <= config.contact_height_threshold
        ]
        if not support_points:
            continue
        distances.append(_point_to_support_distance((root[0], root[1]), support_points))
    return distances


def _contact_correction_candidate(
    min_foot_clearance: float,
    mean_foot_clearance: float,
    config: G1QualityConfig,
) -> dict[str, object]:
    reason = ""
    offset = 0.0
    if mean_foot_clearance > config.max_mean_foot_clearance:
        reason = "vertical_float_offset"
        offset = -mean_foot_clearance
    elif -min_foot_clearance > config.max_penetration_depth:
        reason = "vertical_penetration_offset"
        offset = -min_foot_clearance

    abs_offset = abs(offset)
    candidate = bool(reason) and abs_offset <= config.max_contact_correction_offset
    return {
        "contact_correction_candidate": float(int(candidate)),
        "contact_correction_reason": reason if candidate else "",
        "contact_correction_offset": offset if candidate else 0.0,
        "contact_correction_abs_offset": abs_offset if candidate else 0.0,
    }


def _point_to_support_distance(
    point: tuple[float, float],
    support_points: Sequence[tuple[float, float]],
) -> float:
    unique_points = tuple(dict.fromkeys(support_points))
    if len(unique_points) == 1:
        return math.dist(point, unique_points[0])
    if len(unique_points) == 2:
        return _point_to_segment_distance(point, unique_points[0], unique_points[1])
    hull = _convex_hull(unique_points)
    if _point_in_convex_polygon(point, hull):
        return 0.0
    return min(
        _point_to_segment_distance(point, hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )


def _convex_hull(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return tuple(ordered)

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _point_in_convex_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    if len(polygon) < 3:
        return False
    sign = 0
    for index in range(len(polygon)):
        left = polygon[index]
        right = polygon[(index + 1) % len(polygon)]
        cross = (right[0] - left[0]) * (point[1] - left[1]) - (
            right[1] - left[1]
        ) * (point[0] - left[0])
        if abs(cross) < 1e-9:
            continue
        current_sign = 1 if cross > 0 else -1
        if sign == 0:
            sign = current_sign
        elif sign != current_sign:
            return False
    return True


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_sq = segment_x * segment_x + segment_y * segment_y
    if length_sq == 0.0:
        return math.dist(point, start)
    t = ((point[0] - start[0]) * segment_x + (point[1] - start[1]) * segment_y) / length_sq
    t = max(0.0, min(1.0, t))
    projection = (start[0] + t * segment_x, start[1] + t * segment_y)
    return math.dist(point, projection)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _validate_config(config: G1QualityConfig) -> None:
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity must be positive")
    if config.max_root_speed <= 0:
        raise ValueError("max_root_speed must be positive")
    if config.max_joint_acceleration is not None and config.max_joint_acceleration <= 0:
        raise ValueError("max_joint_acceleration must be positive when set")
    if config.max_root_acceleration is not None and config.max_root_acceleration <= 0:
        raise ValueError("max_root_acceleration must be positive when set")
    if config.max_root_jerk is not None and config.max_root_jerk <= 0:
        raise ValueError("max_root_jerk must be positive when set")
    if config.root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    if config.joint_angle_scale <= 0:
        raise ValueError("joint_angle_scale must be positive")
    if config.root_rotation_scale <= 0:
        raise ValueError("root_rotation_scale must be positive")
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive when set")
    if config.contact_height_threshold < 0:
        raise ValueError("contact_height_threshold must be non-negative")
    if config.max_contact_slide_speed <= 0:
        raise ValueError("max_contact_slide_speed must be positive")
    if config.max_mean_foot_clearance < 0:
        raise ValueError("max_mean_foot_clearance must be non-negative")
    if config.max_penetration_depth < 0:
        raise ValueError("max_penetration_depth must be non-negative")
    if config.max_contact_correction_offset < 0:
        raise ValueError("max_contact_correction_offset must be non-negative")
    if not 0.0 <= config.min_contact_frame_ratio <= 1.0:
        raise ValueError("min_contact_frame_ratio must be within [0, 1]")
    if not 0.0 <= config.max_joint_limit_violation_rate <= 1.0:
        raise ValueError("max_joint_limit_violation_rate must be within [0, 1]")
    if config.start_end_frames <= 0:
        raise ValueError("start_end_frames must be positive")
    if config.max_start_end_root_speed < 0:
        raise ValueError("max_start_end_root_speed must be non-negative")
    if config.min_self_collision_distance < 0:
        raise ValueError("min_self_collision_distance must be non-negative")
    if not 0.0 <= config.max_self_collision_proxy_rate <= 1.0:
        raise ValueError("max_self_collision_proxy_rate must be within [0, 1]")
    if config.min_self_collision_kinematic_hops <= 0:
        raise ValueError("min_self_collision_kinematic_hops must be positive")


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _rate_above(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value > threshold for value in values) / len(values)


def _rate_below(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value < threshold for value in values) / len(values)


def _vector_second_difference_norm(
    current: Sequence[float],
    previous: Sequence[float],
    previous_previous: Sequence[float],
) -> float:
    return math.sqrt(
        sum(
            (cur - (2.0 * prev) + prev_prev) ** 2
            for cur, prev, prev_prev in zip(current, previous, previous_previous)
        )
    )


def _vector_third_difference_norm(
    current: Sequence[float],
    previous: Sequence[float],
    previous_previous: Sequence[float],
    previous_previous_previous: Sequence[float],
) -> float:
    return math.sqrt(
        sum(
            (cur - (3.0 * prev) + (3.0 * prev_prev) - prev_prev_prev) ** 2
            for cur, prev, prev_prev, prev_prev_prev in zip(
                current, previous, previous_previous, previous_previous_previous
            )
        )
    )


def _g1_fk_positions(
    model: G1KinematicModel,
    joints: Sequence[float],
    root_position: Sequence[float],
    root_euler: Sequence[float],
    include_empty_body_origin: bool = True,
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    joint_values = {
        _joint_name_from_column(column): value for column, value in zip(G1_JOINT_COLUMNS, joints)
    }
    body_transforms: list[tuple[list[list[float]], tuple[float, float, float]]] = []
    output: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for body in model.bodies:
        local_rotation = _quat_to_matrix(body.quat)
        local_position = body.pos
        if body.has_freejoint:
            local_rotation = _root_rotation_matrix(root_euler)
            local_position = (root_position[0], root_position[1], root_position[2])
        if body.joint_name and body.joint_axis is not None:
            local_rotation = _matmul(
                local_rotation,
                _axis_angle_matrix(body.joint_axis, joint_values.get(body.joint_name, 0.0)),
            )
        if body.parent is None:
            global_rotation = local_rotation
            global_position = local_position
        else:
            parent_rotation, parent_position = body_transforms[body.parent]
            global_rotation = _matmul(parent_rotation, local_rotation)
            rotated_pos = _matvec(parent_rotation, local_position)
            global_position = (
                parent_position[0] + rotated_pos[0],
                parent_position[1] + rotated_pos[1],
                parent_position[2] + rotated_pos[2],
            )
        body_transforms.append((global_rotation, global_position))
        points = body.geom_points or (((0.0, 0.0, 0.0),) if include_empty_body_origin else ())
        output[body.name] = tuple(
            _add(global_position, _matvec(global_rotation, point)) for point in points
        )
    return output


def _root_rotation_matrix(euler_xyz: Sequence[float]) -> list[list[float]]:
    if len(euler_xyz) < 3:
        return _identity()
    rx = _axis_angle_matrix((1.0, 0.0, 0.0), euler_xyz[0])
    ry = _axis_angle_matrix((0.0, 1.0, 0.0), euler_xyz[1])
    rz = _axis_angle_matrix((0.0, 0.0, 1.0), euler_xyz[2])
    return _matmul(_matmul(rx, ry), rz)


def _joint_name_from_column(column: str) -> str:
    return column[:-4] if column.endswith("_dof") else column


def _local_geom_points(element: ET.Element) -> tuple[tuple[float, float, float], ...]:
    points = []
    for geom in element.findall("geom"):
        if "pos" in geom.attrib:
            points.append(_parse_vec3(geom.attrib["pos"]))
        else:
            points.append((0.0, 0.0, 0.0))
    return tuple(points)


def _parse_vec3(value: str) -> tuple[float, float, float]:
    parsed = _parse_float_tuple(value)
    padded = (*parsed, 0.0, 0.0, 0.0)
    return (padded[0], padded[1], padded[2])


def _parse_quat(value: str) -> tuple[float, float, float, float]:
    parsed = _parse_float_tuple(value)
    padded = (*parsed, 1.0, 0.0, 0.0, 0.0)
    return (padded[0], padded[1], padded[2], padded[3])


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def _quat_to_matrix(quat: Sequence[float]) -> list[list[float]]:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return _identity()
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> list[list[float]]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        return _identity()
    x, y, z = x / norm, y / norm, z / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [
            sum(float(left[row][k]) * float(right[k][col]) for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return (
        sum(float(matrix[0][k]) * float(vector[k]) for k in range(3)),
        sum(float(matrix[1][k]) * float(vector[k]) for k in range(3)),
        sum(float(matrix[2][k]) * float(vector[k]) for k in range(3)),
    )


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _rounded_dict(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def _rounded_contact_stats(values: Mapping[str, object]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            rounded[key] = round(float(value), 6)
        else:
            rounded[key] = value
    return rounded


def _empty_model_metrics() -> dict[str, object]:
    return {
        "joint_limit_checked_values": 0.0,
        "joint_limit_violation_rate": 0.0,
        "max_joint_limit_violation": 0.0,
        "ground_height": 0.0,
        "min_foot_height": 0.0,
        "mean_foot_clearance": 0.0,
        "max_foot_clearance": 0.0,
        "min_body_clearance": 0.0,
        "penetration_depth": 0.0,
        "contact_correction_candidate": 0.0,
        "contact_correction_reason": "",
        "contact_correction_offset": 0.0,
        "contact_correction_abs_offset": 0.0,
        "contact_frame_ratio": 0.0,
        "max_contact_slide_speed": 0.0,
        "contact_slide_rate": 0.0,
        "support_frame_ratio": 0.0,
        "mean_root_support_distance": 0.0,
        "max_root_support_distance": 0.0,
        "self_collision_checked_pairs": 0.0,
        "self_collision_proxy_rate": 0.0,
        "min_self_collision_distance": 0.0,
        "mean_min_self_collision_distance": 0.0,
    }


def _empty_result(
    base: Mapping[str, object],
    flag: str,
    model: G1KinematicModel | None = None,
) -> dict[str, object]:
    return {
        **base,
        "frame_count": 0,
        "original_frame_count": 0,
        "joint_dim": 0,
        "nonfinite_values": 0,
        "root_position_scale": 0.0,
        "joint_angle_scale": 0.0,
        "max_abs_joint_velocity": 0.0,
        "mean_abs_joint_velocity": 0.0,
        "joint_jump_rate": 0.0,
        "max_abs_joint_acceleration": 0.0,
        "mean_abs_joint_acceleration": 0.0,
        "joint_acceleration_jump_rate": 0.0,
        "max_root_speed": 0.0,
        "root_jump_rate": 0.0,
        "max_root_acceleration": 0.0,
        "mean_root_acceleration": 0.0,
        "root_acceleration_jump_rate": 0.0,
        "max_root_jerk": 0.0,
        "mean_root_jerk": 0.0,
        "root_jerk_jump_rate": 0.0,
        "max_start_end_root_speed": 0.0,
        "root_height_min": 0.0,
        "root_height_max": 0.0,
        "root_height_range": 0.0,
        "model_xml": model.source_xml if model is not None else "",
        "foot_bodies": "|".join(model.foot_body_names) if model is not None else "",
        "quality_mode": "mjcf_fk" if model is not None else "csv_root_joint",
        **_empty_model_metrics(),
        "quality_action": "exclude",
        "quality_flags": flag,
    }


def _model_report(model: G1KinematicModel | None) -> dict[str, object]:
    if model is None:
        return {
            "mode": "csv_root_joint",
            "source_xml": None,
            "body_count": 0,
            "joint_limit_count": 0,
            "foot_bodies": [],
            "note": "No model XML was supplied, so FK/contact and joint-limit metrics are zero-filled.",
        }
    return {
        "mode": "mjcf_fk",
        "source_xml": model.source_xml,
        "body_count": len(model.bodies),
        "joint_limit_count": len(model.joint_ranges),
        "foot_bodies": list(model.foot_body_names),
    }


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "original_frame_count",
        "frame_count",
        "max_abs_joint_velocity",
        "mean_abs_joint_velocity",
        "joint_jump_rate",
        "max_abs_joint_acceleration",
        "mean_abs_joint_acceleration",
        "joint_acceleration_jump_rate",
        "max_root_speed",
        "root_jump_rate",
        "max_root_acceleration",
        "mean_root_acceleration",
        "root_acceleration_jump_rate",
        "max_root_jerk",
        "mean_root_jerk",
        "root_jerk_jump_rate",
        "max_start_end_root_speed",
        "root_height_min",
        "root_height_max",
        "root_height_range",
        "joint_limit_checked_values",
        "joint_limit_violation_rate",
        "max_joint_limit_violation",
        "min_foot_height",
        "mean_foot_clearance",
        "max_foot_clearance",
        "min_body_clearance",
        "penetration_depth",
        "contact_correction_candidate",
        "contact_correction_offset",
        "contact_correction_abs_offset",
        "contact_frame_ratio",
        "max_contact_slide_speed",
        "contact_slide_rate",
        "support_frame_ratio",
        "mean_root_support_distance",
        "max_root_support_distance",
        "self_collision_checked_pairs",
        "self_collision_proxy_rate",
        "min_self_collision_distance",
        "mean_min_self_collision_distance",
    )
    return {metric: _summarize_values(_numeric_values(rows, metric)) for metric in metrics}


def _numeric_values(rows: Sequence[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "min": round(sorted_values[0], 6),
        "mean": round(sum(sorted_values) / len(sorted_values), 6),
        "p50": round(_percentile(sorted_values, 0.50), 6),
        "p90": round(_percentile(sorted_values, 0.90), 6),
        "p95": round(_percentile(sorted_values, 0.95), 6),
        "p99": round(_percentile(sorted_values, 0.99), 6),
        "max": round(sorted_values[-1], 6),
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _worse_action(left: str, right: str) -> str:
    order = {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}
    return left if order[left] >= order[right] else right


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]


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
