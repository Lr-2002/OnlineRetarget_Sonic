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
    fps: float = 30.0
    position_scale: float = 0.01
    frame_stride: int = 1
    max_frames: int | None = None
    ground_height: float | None = None
    ground_percentile: float = 0.05
    contact_height_threshold: float = 0.04
    max_contact_slide_speed: float = 0.25
    max_mean_foot_clearance: float = 0.10
    max_penetration_depth: float = 0.03
    min_contact_frame_ratio: float = 0.05
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
    with tarfile.open(source_tar, "r:*") as tar:
        for row in rows:
            source_path = row.get("move_soma_proportional_path", "")
            if not source_path:
                skipped_rows += 1
                continue
            scanned.append(scan_source_fk_member(tar, row, config))

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
        extracted = tar.extractfile(source_path)
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

    requested_bodies = tuple(dict.fromkeys((*config.body_bodies, *config.foot_bodies)))
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
    contact_flags = [clearance <= config.contact_height_threshold for clearance in foot_clearances]
    contact_frame_ratio = sum(contact_flags) / len(contact_flags) if contact_flags else 0.0
    effective_fps = config.fps / config.frame_stride
    contact_slide_speeds = _contact_slide_speeds(
        frames,
        present_feet,
        ground_height,
        config,
        fps=effective_fps,
    )

    mean_foot_clearance = _mean(foot_clearances)
    max_foot_clearance = max(foot_clearances) if foot_clearances else 0.0
    min_body_clearance = min(body_clearances) if body_clearances else 0.0
    penetration_depth = max(0.0, -min_body_clearance)
    max_contact_slide_speed = max(contact_slide_speeds) if contact_slide_speeds else 0.0
    contact_slide_rate = _rate_above(contact_slide_speeds, config.max_contact_slide_speed)

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
    if config.fps <= 0:
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
    if not 0.0 <= config.min_contact_frame_ratio <= 1.0:
        raise ValueError("min_contact_frame_ratio must be within [0, 1]")


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
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


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
