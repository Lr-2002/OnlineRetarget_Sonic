"""BVH source-motion quality scanning for SOMA skeleton clips."""

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


@dataclass(frozen=True)
class BVHQualityConfig:
    max_channel_velocity: float = 3000.0
    max_root_speed: float = 500.0
    expected_frame_time: float | None = None
    frame_time_tolerance: float = 1e-4

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class BVHQualityScanResult:
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


def scan_bvh_quality_from_index(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: BVHQualityConfig | None = None,
    limit: int | None = 100,
    splits: Sequence[str] = (),
    actions: Sequence[str] = ("keep", "downweight", "quarantine"),
    sample_by: Sequence[str] = (),
) -> BVHQualityScanResult:
    """Scan source BVH clips referenced by a split index."""

    config = config or BVHQualityConfig()
    candidate_rows = list(_iter_index_rows(index_csv, splits=splits, actions=actions))
    rows = select_rows_for_scan(candidate_rows, limit=limit, sample_by=sample_by)

    output_dir = output_root.expanduser() / "quality" / _quality_run_name(index_csv, limit, sample_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = output_dir / "source_bvh_quality_stats.jsonl"
    report_json = output_dir / "source_bvh_quality_report.json"

    source_tar = data_root.expanduser() / "soma_proportional.tar"
    scanned: list[dict[str, object]] = []
    skipped_rows = 0
    with tarfile.open(source_tar, "r:*") as tar:
        for row in rows:
            source_path = row.get("move_soma_proportional_path", "")
            if not source_path:
                skipped_rows += 1
                continue
            scanned.append(scan_bvh_member(tar, row, config))

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
    return BVHQualityScanResult(
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


def scan_bvh_member(
    tar: tarfile.TarFile,
    index_row: Mapping[str, str],
    config: BVHQualityConfig,
) -> dict[str, object]:
    """Scan one BVH member from an open tar archive."""

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
        member = tar.getmember(source_path)
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
        parsed = parse_bvh_text(text)
    except ValueError:
        return _empty_result(base, "source_bvh_parse_error")
    return _summarize_bvh(base, parsed, config, index_row)


def parse_bvh_text(text: str) -> dict[str, object]:
    """Parse enough BVH structure for quality statistics."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        motion_idx = lines.index("MOTION")
    except ValueError as exc:
        raise ValueError("BVH missing MOTION section") from exc

    channel_names: list[str] = []
    position_groups: list[tuple[str, tuple[int, int, int]]] = []
    current_joint = ""
    for line in lines[:motion_idx]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"} and len(parts) >= 2:
            current_joint = parts[1]
        elif parts[0] == "CHANNELS" and len(parts) >= 3:
            channel_count = int(parts[1])
            channels = parts[2 : 2 + channel_count]
            start = len(channel_names)
            for channel in channels:
                channel_names.append(f"{current_joint}.{channel}")
            position_indices = []
            for offset, channel in enumerate(channels):
                if channel in {"Xposition", "Yposition", "Zposition"}:
                    position_indices.append(start + offset)
            if len(position_indices) == 3:
                position_groups.append((current_joint, tuple(position_indices)))

    if motion_idx + 2 >= len(lines):
        raise ValueError("BVH MOTION section is incomplete")
    frames = _parse_header_value(lines[motion_idx + 1], "Frames")
    frame_time = _parse_header_value(lines[motion_idx + 2], "Frame Time")
    motion_lines = lines[motion_idx + 3 :]
    values: list[list[float]] = []
    nonfinite_values = 0
    width_mismatch_rows = 0
    for line in motion_lines:
        row_values: list[float] = []
        for item in line.split():
            value = _parse_float(item)
            if value is None:
                nonfinite_values += 1
            else:
                row_values.append(value)
        if len(row_values) != len(channel_names):
            width_mismatch_rows += 1
        values.append(row_values)

    return {
        "declared_frames": int(frames),
        "frame_time": float(frame_time),
        "channel_names": channel_names,
        "position_groups": position_groups,
        "values": values,
        "nonfinite_values": nonfinite_values,
        "width_mismatch_rows": width_mismatch_rows,
    }


def _summarize_bvh(
    base: Mapping[str, object],
    parsed: Mapping[str, object],
    config: BVHQualityConfig,
    index_row: Mapping[str, str],
) -> dict[str, object]:
    values = parsed["values"]
    if not isinstance(values, list):
        raise ValueError("parsed BVH values must be a list")
    frame_time = float(parsed["frame_time"])
    fps = 1.0 / frame_time if frame_time > 0 else 0.0
    channel_names = parsed["channel_names"]
    position_groups = parsed["position_groups"]
    frame_count = len(values)
    declared_frames = int(parsed["declared_frames"])
    channel_width = len(channel_names)

    channel_velocity_samples: list[float] = []
    root_speed_samples: list[float] = []
    root_joint = ""
    root_group = _select_root_proxy(values, position_groups)
    if root_group is not None:
        root_joint, root_indices = root_group
    else:
        root_indices = ()

    prev_row: list[float] | None = None
    prev_root: list[float] | None = None
    for row in values:
        if len(row) != channel_width:
            continue
        if prev_row is not None:
            channel_velocity_samples.extend(abs(cur - prev) * fps for cur, prev in zip(row, prev_row))
        if root_indices:
            root = [row[index] for index in root_indices]
            if prev_root is not None:
                root_speed_samples.append(math.dist(root, prev_root) * fps)
            prev_root = root
        prev_row = row

    flags: list[str] = []
    action = "keep"
    if frame_count == 0:
        flags.append("empty_motion")
        action = "exclude"
    if frame_count != declared_frames:
        flags.append("frame_count_mismatch")
        action = _worse_action(action, "quarantine")
    if int(parsed["nonfinite_values"]) > 0:
        flags.append("nonfinite_value")
        action = "exclude"
    if int(parsed["width_mismatch_rows"]) > 0:
        flags.append("channel_width_mismatch")
        action = "exclude"
    if config.expected_frame_time is not None:
        if abs(frame_time - config.expected_frame_time) > config.frame_time_tolerance:
            flags.append("frame_time_mismatch")
            action = _worse_action(action, "quarantine")
    channel_jump_rate = _rate_above(channel_velocity_samples, config.max_channel_velocity)
    root_jump_rate = _rate_above(root_speed_samples, config.max_root_speed)
    if channel_jump_rate > 0.0:
        flags.append("source_channel_jump")
        action = _worse_action(action, "quarantine")
    if root_jump_rate > 0.0:
        flags.append("source_root_discontinuity")
        action = _worse_action(action, "quarantine")

    return {
        **base,
        "declared_frames": declared_frames,
        "frame_count": frame_count,
        "metadata_duration_frames": index_row.get("move_duration_frames", ""),
        "frame_time": round(frame_time, 8),
        "fps": round(fps, 6),
        "channel_width": channel_width,
        "position_group_count": len(position_groups),
        "root_proxy_joint": root_joint,
        "nonfinite_values": int(parsed["nonfinite_values"]),
        "width_mismatch_rows": int(parsed["width_mismatch_rows"]),
        "max_abs_channel_velocity": round(max(channel_velocity_samples), 6)
        if channel_velocity_samples
        else 0.0,
        "mean_abs_channel_velocity": round(_mean(channel_velocity_samples), 6),
        "channel_jump_rate": round(channel_jump_rate, 6),
        "max_root_speed": round(max(root_speed_samples), 6) if root_speed_samples else 0.0,
        "root_jump_rate": round(root_jump_rate, 6),
        "quality_action": action,
        "quality_flags": "|".join(flags),
    }


def _select_root_proxy(
    values: Sequence[Sequence[float]],
    position_groups: Sequence[tuple[str, tuple[int, int, int]]],
) -> tuple[str, tuple[int, int, int]] | None:
    if not values or not position_groups:
        return None
    best_group = None
    best_displacement = -1.0
    for joint_name, indices in position_groups:
        first = _row_position(values[0], indices)
        last = _row_position(values[-1], indices)
        if first is None or last is None:
            continue
        displacement = math.dist(first, last)
        if displacement > best_displacement:
            best_displacement = displacement
            best_group = (joint_name, indices)
    return best_group


def _row_position(row: Sequence[float], indices: Sequence[int]) -> list[float] | None:
    if any(index >= len(row) for index in indices):
        return None
    return [row[index] for index in indices]


def _empty_result(base: Mapping[str, object], flag: str) -> dict[str, object]:
    return {
        **base,
        "declared_frames": 0,
        "frame_count": 0,
        "metadata_duration_frames": "",
        "frame_time": 0.0,
        "fps": 0.0,
        "channel_width": 0,
        "position_group_count": 0,
        "root_proxy_joint": "",
        "nonfinite_values": 0,
        "width_mismatch_rows": 0,
        "max_abs_channel_velocity": 0.0,
        "mean_abs_channel_velocity": 0.0,
        "channel_jump_rate": 0.0,
        "max_root_speed": 0.0,
        "root_jump_rate": 0.0,
        "quality_action": "exclude",
        "quality_flags": flag,
    }


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


def _parse_header_value(line: str, name: str) -> float:
    prefix = f"{name}:"
    if not line.startswith(prefix):
        raise ValueError(f"expected BVH header line {prefix}")
    return float(line[len(prefix) :].strip())


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate_above(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value > threshold for value in values) / len(values)


def _quality_run_name(index_csv: Path, limit: int | None, sample_by: Sequence[str] = ()) -> str:
    return f"{index_csv.parent.name}_source_{sampling_run_tag(limit, sample_by)}"


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "frame_count",
        "fps",
        "channel_width",
        "max_abs_channel_velocity",
        "mean_abs_channel_velocity",
        "channel_jump_rate",
        "max_root_speed",
        "root_jump_rate",
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
