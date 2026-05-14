"""Source BVH FK/contact quality scanning for M2Q calibration."""

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

from .row_sampling import sampling_run_tag, scan_sampling_report, select_rows_for_scan
from .windowed_builder import BVHMotion, global_body_position_maps_from_bvh, parse_bvh_motion


DEFAULT_FOOT_BODIES = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
DEFAULT_BODY_BODIES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Head",
    "LeftHand",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)


@dataclass(frozen=True)
class SourceFKQualityConfig:
    fps: float | None = None
    position_scale: float = 0.01
    frame_stride: int = 1
    max_frames: int | None = None
    ground_height: float | None = None
    ground_percentile: float = 0.05
    contact_height_threshold: float = 0.04
    max_contact_slide_speed: float = 0.25
    max_mean_foot_clearance: float = 0.10
    max_penetration_depth: float = 0.03
    max_contact_correction_offset: float = 0.15
    min_contact_frame_ratio: float = 0.05
    root_body: str = "Hips"
    foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES
    body_bodies: tuple[str, ...] = DEFAULT_BODY_BODIES

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["foot_bodies"] = list(self.foot_bodies)
        payload["body_bodies"] = list(self.body_bodies)
        return payload


@dataclass(frozen=True)
class SourceFKQualityScanResult:
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


def scan_source_fk_quality_from_index(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: SourceFKQualityConfig | None = None,
    limit: int | None = 100,
    splits: Sequence[str] = (),
    actions: Sequence[str] = ("keep", "downweight", "quarantine"),
    action_column: str = "curation_action",
    sample_by: Sequence[str] = (),
) -> SourceFKQualityScanResult:
    """Scan source BVH clips for FK/contact geometry quality signals."""

    config = config or SourceFKQualityConfig()
    _validate_config(config)
    candidate_rows = list(_iter_index_rows(index_csv, splits=splits, actions=actions, action_column=action_column))
    rows = select_rows_for_scan(candidate_rows, limit=limit, sample_by=sample_by)

    output_dir = output_root.expanduser() / "quality" / _quality_run_name(index_csv, limit, sample_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = output_dir / "source_fk_quality_stats.jsonl"
    report_json = output_dir / "source_fk_quality_report.json"

    source_tar = data_root.expanduser() / "soma_proportional.tar"
    scanned: list[dict[str, object]] = []
    skipped_rows = 0
    stats_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tar, "r:*") as tar, stats_jsonl.open("w", encoding="utf-8") as stats_file:
        member_by_name = {member.name: member for member in tar.getmembers()}
        for row in rows:
            source_path = row.get("move_soma_proportional_path", "")
            if not source_path:
                skipped_rows += 1
                continue
            result = scan_source_fk_member(tar, row, config, member_by_name=member_by_name)
            scanned.append(result)
            _write_jsonl_row(stats_file, result)

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
        "limit": limit,
        "filters": {
            "splits": list(splits),
            "actions": list(actions),
            "action_column": action_column,
        },
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
    return SourceFKQualityScanResult(
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


def scan_source_fk_member(
    tar: tarfile.TarFile,
    index_row: Mapping[str, str],
    config: SourceFKQualityConfig,
    member_by_name: Mapping[str, tarfile.TarInfo] | None = None,
) -> dict[str, object]:
    """Scan one source BVH member from an open tar archive."""

    source_path = index_row.get("move_soma_proportional_path", "")
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
        "move_soma_proportional_path": source_path,
    }
    try:
        member = member_by_name[source_path] if member_by_name is not None else source_path
        extracted = tar.extractfile(member)
    except (KeyError, tarfile.TarError):
        return _empty_result(base, "missing_source_bvh_member")
    if extracted is None:
        return _empty_result(base, "empty_source_bvh_member")

    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8").read()
        except UnicodeDecodeError:
            return _empty_result(base, "source_bvh_decode_error")
    try:
        motion = parse_bvh_motion(text)
    except ValueError:
        return _empty_result(base, "source_bvh_parse_error")
    return summarize_source_fk_motion(base, motion, config)


def summarize_source_fk_motion(
    base: Mapping[str, object],
    motion,
    config: SourceFKQualityConfig,
) -> dict[str, object]:
    """Summarize foot/ground/contact metrics from parsed BVH motion."""

    requested_bodies = tuple(
        dict.fromkeys((config.root_body, *config.body_bodies, *config.foot_bodies))
    )
    original_frame_count = len(motion.frames)
    sampled_motion = _sample_motion(motion, config)
    frames = global_body_position_maps_from_bvh(
        sampled_motion,
        body_names=requested_bodies,
        position_scale=config.position_scale,
    )
    frame_count = len(frames)
    present_feet = tuple(body for body in config.foot_bodies if any(body in frame for frame in frames))
    if frame_count == 0:
        return _empty_result(base, "empty_motion", original_frame_count=original_frame_count)
    if not present_feet:
        return _empty_result(
            base,
            "missing_foot_bodies",
            frame_count=frame_count,
            original_frame_count=original_frame_count,
        )

    foot_heights = [_frame_min_height(frame, present_feet) for frame in frames]
    body_heights = [_frame_min_height(frame, requested_bodies) for frame in frames]
    valid_foot_heights = [height for height in foot_heights if height is not None]
    valid_body_heights = [height for height in body_heights if height is not None]
    if not valid_foot_heights:
        return _empty_result(
            base,
            "missing_foot_heights",
            frame_count=frame_count,
            original_frame_count=original_frame_count,
        )
    if not _all_frame_positions_finite(frames):
        result = _empty_result(
            base,
            "nonfinite_fk_position",
            frame_count=frame_count,
            original_frame_count=original_frame_count,
        )
        return {**result, "quality_action": "exclude"}

    ground_height = (
        config.ground_height
        if config.ground_height is not None
        else _percentile(sorted(valid_foot_heights), config.ground_percentile)
    )
    foot_clearances = [height - ground_height for height in valid_foot_heights]
    body_clearances = [height - ground_height for height in valid_body_heights]
    root_heights = [
        position[1] - ground_height
        for frame in frames
        for position in (frame.get(config.root_body),)
        if position is not None
    ]
    contact_flags = [clearance <= config.contact_height_threshold for clearance in foot_clearances]
    contact_frame_ratio = sum(contact_flags) / len(contact_flags) if contact_flags else 0.0
    effective_fps = _effective_fps(motion, config)
    contact_slide_speeds = _contact_slide_speeds(
        frames,
        present_feet,
        ground_height,
        config,
        fps=effective_fps,
    )
    support_distances = _root_support_distances(frames, present_feet, ground_height, config)

    mean_foot_clearance = _mean(foot_clearances)
    max_foot_clearance = max(foot_clearances) if foot_clearances else 0.0
    min_foot_clearance = min(foot_clearances) if foot_clearances else 0.0
    min_body_clearance = min(body_clearances) if body_clearances else 0.0
    penetration_depth = max(0.0, -min_body_clearance)
    contact_correction = _contact_correction_candidate(
        min_foot_clearance=min_foot_clearance,
        mean_foot_clearance=mean_foot_clearance,
        config=config,
    )
    max_contact_slide_speed = max(contact_slide_speeds) if contact_slide_speeds else 0.0
    contact_slide_rate = _rate_above(contact_slide_speeds, config.max_contact_slide_speed)
    root_height_min = min(root_heights) if root_heights else 0.0
    root_height_max = max(root_heights) if root_heights else 0.0
    mean_root_height = _mean(root_heights)
    support_frame_ratio = len(support_distances) / frame_count if frame_count else 0.0
    mean_root_support_distance = _mean(support_distances)
    max_root_support_distance = max(support_distances) if support_distances else 0.0

    flags: list[str] = []
    action = "keep"
    if contact_frame_ratio < config.min_contact_frame_ratio:
        flags.append("source_low_foot_contact")
        action = _worse_action(action, "quarantine")
    if mean_foot_clearance > config.max_mean_foot_clearance:
        flags.append("source_foot_float")
        action = _worse_action(action, "quarantine")
    if penetration_depth > config.max_penetration_depth:
        flags.append("source_ground_penetration")
        action = _worse_action(action, "quarantine")
    if contact_slide_rate > 0.0:
        flags.append("source_foot_slide")
        action = _worse_action(action, "downweight")

    return {
        **base,
        "original_frame_count": original_frame_count,
        "frame_count": frame_count,
        "present_foot_bodies": "|".join(present_feet),
        "ground_source": "fixed" if config.ground_height is not None else "foot_percentile",
        "ground_height": round(ground_height, 6),
        "min_foot_height": round(min(valid_foot_heights), 6),
        "mean_foot_clearance": round(mean_foot_clearance, 6),
        "max_foot_clearance": round(max_foot_clearance, 6),
        "min_body_clearance": round(min_body_clearance, 6),
        "penetration_depth": round(penetration_depth, 6),
        **contact_correction,
        "root_body": config.root_body,
        "root_height_min": round(root_height_min, 6),
        "root_height_max": round(root_height_max, 6),
        "root_height_range": round(root_height_max - root_height_min, 6),
        "mean_root_height": round(mean_root_height, 6),
        "support_frame_ratio": round(support_frame_ratio, 6),
        "mean_root_support_distance": round(mean_root_support_distance, 6),
        "max_root_support_distance": round(max_root_support_distance, 6),
        "contact_frame_ratio": round(contact_frame_ratio, 6),
        "max_contact_slide_speed": round(max_contact_slide_speed, 6),
        "contact_slide_rate": round(contact_slide_rate, 6),
        "quality_action": action,
        "quality_flags": "|".join(flags),
    }


def _contact_slide_speeds(
    frames: Sequence[Mapping[str, Sequence[float]]],
    foot_bodies: Sequence[str],
    ground_height: float,
    config: SourceFKQualityConfig,
    fps: float,
) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        for body in foot_bodies:
            prev_position = previous.get(body)
            cur_position = current.get(body)
            if prev_position is None or cur_position is None:
                continue
            prev_clearance = prev_position[1] - ground_height
            cur_clearance = cur_position[1] - ground_height
            if (
                prev_clearance > config.contact_height_threshold
                or cur_clearance > config.contact_height_threshold
            ):
                continue
            horizontal_distance = math.dist(
                (prev_position[0], prev_position[2]),
                (cur_position[0], cur_position[2]),
            )
            speeds.append(horizontal_distance * fps)
    return speeds


def _root_support_distances(
    frames: Sequence[Mapping[str, Sequence[float]]],
    foot_bodies: Sequence[str],
    ground_height: float,
    config: SourceFKQualityConfig,
) -> list[float]:
    distances: list[float] = []
    for frame in frames:
        root = frame.get(config.root_body)
        if root is None:
            continue
        support_points = [
            (position[0], position[2])
            for body in foot_bodies
            for position in (frame.get(body),)
            if position is not None
            and position[1] - ground_height <= config.contact_height_threshold
        ]
        if not support_points:
            continue
        distances.append(_point_to_support_distance((root[0], root[2]), support_points))
    return distances


def _contact_correction_candidate(
    min_foot_clearance: float,
    mean_foot_clearance: float,
    config: SourceFKQualityConfig,
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
        "contact_correction_candidate": int(candidate),
        "contact_correction_reason": reason if candidate else "",
        "contact_correction_offset": round(offset if candidate else 0.0, 6),
        "contact_correction_abs_offset": round(abs_offset if candidate else 0.0, 6),
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


def _frame_min_height(frame: Mapping[str, Sequence[float]], bodies: Sequence[str]) -> float | None:
    heights = [frame[body][1] for body in bodies if body in frame]
    return min(heights) if heights else None


def _empty_result(
    base: Mapping[str, object],
    flag: str,
    frame_count: int = 0,
    original_frame_count: int | None = None,
) -> dict[str, object]:
    original = frame_count if original_frame_count is None else original_frame_count
    return {
        **base,
        "original_frame_count": original,
        "frame_count": frame_count,
        "present_foot_bodies": "",
        "ground_source": "",
        "ground_height": 0.0,
        "min_foot_height": 0.0,
        "mean_foot_clearance": 0.0,
        "max_foot_clearance": 0.0,
        "min_body_clearance": 0.0,
        "penetration_depth": 0.0,
        "contact_correction_candidate": 0,
        "contact_correction_reason": "",
        "contact_correction_offset": 0.0,
        "contact_correction_abs_offset": 0.0,
        "root_body": "",
        "root_height_min": 0.0,
        "root_height_max": 0.0,
        "root_height_range": 0.0,
        "mean_root_height": 0.0,
        "support_frame_ratio": 0.0,
        "mean_root_support_distance": 0.0,
        "max_root_support_distance": 0.0,
        "contact_frame_ratio": 0.0,
        "max_contact_slide_speed": 0.0,
        "contact_slide_rate": 0.0,
        "quality_action": "exclude",
        "quality_flags": flag,
    }


def _iter_index_rows(
    index_csv: Path,
    splits: Sequence[str],
    actions: Sequence[str],
    action_column: str,
) -> Iterable[dict[str, str]]:
    split_filter = set(splits)
    action_filter = set(actions)
    with index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split_filter and row.get("split") not in split_filter:
                continue
            if action_filter and row.get(action_column) not in action_filter:
                continue
            yield row


def _sample_motion(motion: BVHMotion, config: SourceFKQualityConfig) -> BVHMotion:
    frames = motion.frames[:: config.frame_stride]
    if config.max_frames is not None:
        frames = frames[: config.max_frames]
    return BVHMotion(
        joints=motion.joints,
        frames=tuple(frames),
        frame_time=motion.frame_time * config.frame_stride,
        channel_count=motion.channel_count,
    )


def _validate_config(config: SourceFKQualityConfig) -> None:
    if config.fps is not None and config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.position_scale <= 0:
        raise ValueError("position_scale must be positive")
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive when set")
    if config.ground_height is not None and not math.isfinite(config.ground_height):
        raise ValueError("ground_height must be finite when set")
    if not 0.0 <= config.ground_percentile <= 1.0:
        raise ValueError("ground_percentile must be within [0, 1]")
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


def _effective_fps(motion: BVHMotion, config: SourceFKQualityConfig) -> float:
    if config.fps is not None:
        return config.fps / config.frame_stride
    if motion.frame_time <= 0:
        raise ValueError("BVH frame_time must be positive when fps is not configured")
    return 1.0 / (motion.frame_time * config.frame_stride)


def _quality_run_name(index_csv: Path, limit: int | None, sample_by: Sequence[str] = ()) -> str:
    return f"{index_csv.parent.name}_source_fk_{sampling_run_tag(limit, sample_by)}"


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "original_frame_count",
        "frame_count",
        "ground_height",
        "mean_foot_clearance",
        "max_foot_clearance",
        "min_body_clearance",
        "penetration_depth",
        "contact_correction_candidate",
        "contact_correction_offset",
        "contact_correction_abs_offset",
        "root_height_min",
        "root_height_max",
        "root_height_range",
        "mean_root_height",
        "support_frame_ratio",
        "mean_root_support_distance",
        "max_root_support_distance",
        "contact_frame_ratio",
        "max_contact_slide_speed",
        "contact_slide_rate",
    )
    return {metric: _summarize_values(_numeric_values(rows, metric)) for metric in metrics}


def _numeric_values(rows: Sequence[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
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
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _all_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _all_frame_positions_finite(frames: Sequence[Mapping[str, Sequence[float]]]) -> bool:
    for frame in frames:
        for position in frame.values():
            if not _all_finite(position):
                return False
    return True


def _rate_above(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value > threshold for value in values) / len(values)


def _worse_action(left: str, right: str) -> str:
    order = {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}
    return left if order[left] >= order[right] else right


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _write_jsonl_row(f, row)


def _write_jsonl_row(file, row: Mapping[str, object]) -> None:
    file.write(json.dumps(row, sort_keys=True))
    file.write("\n")
    file.flush()


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
