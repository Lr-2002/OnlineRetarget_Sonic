"""Source-target pair quality scanning for BONES-SEED motion rows."""

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
class PairQualityConfig:
    expected_source_frame_time: float | None = None
    g1_fps: float = 120.0
    frame_time_tolerance: float = 1e-4
    max_frame_count_delta: int = 0
    max_duration_delta_sec: float = 1e-3
    target_provenance: str = "kinematic_g1_csv"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairQualityScanResult:
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


def scan_pair_quality_from_index(
    data_root: Path,
    index_csv: Path,
    output_root: Path,
    config: PairQualityConfig | None = None,
    limit: int | None = 100,
    splits: Sequence[str] = (),
    actions: Sequence[str] = ("keep", "downweight", "quarantine"),
    sample_by: Sequence[str] = (),
) -> PairQualityScanResult:
    """Scan source/G1 pair consistency for rows referenced by a split index."""

    config = config or PairQualityConfig()
    _validate_config(config)
    candidate_rows = list(_iter_index_rows(index_csv, splits=splits, actions=actions))
    rows = select_rows_for_scan(candidate_rows, limit=limit, sample_by=sample_by)

    output_dir = output_root.expanduser() / "quality" / _quality_run_name(index_csv, limit, sample_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_jsonl = output_dir / "pair_quality_stats.jsonl"
    report_json = output_dir / "pair_quality_report.json"

    source_tar_path = data_root.expanduser() / "soma_proportional.tar"
    g1_tar_path = data_root.expanduser() / "g1.tar"
    scanned: list[dict[str, object]] = []
    skipped_rows = 0
    with tarfile.open(source_tar_path, "r:*") as source_tar, tarfile.open(g1_tar_path, "r:*") as g1_tar:
        for row in rows:
            if not row.get("move_soma_proportional_path") or not row.get("move_g1_path"):
                skipped_rows += 1
                scanned.append(scan_pair_members(source_tar, g1_tar, row, config))
                continue
            scanned.append(scan_pair_members(source_tar, g1_tar, row, config))

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
    return PairQualityScanResult(
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


def scan_pair_members(
    source_tar: tarfile.TarFile,
    g1_tar: tarfile.TarFile,
    index_row: Mapping[str, str],
    config: PairQualityConfig,
) -> dict[str, object]:
    """Scan source BVH and target G1 CSV metadata for one row."""

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
        "move_duration_frames": index_row.get("move_duration_frames", ""),
        "move_soma_proportional_path": index_row.get("move_soma_proportional_path", ""),
        "move_g1_path": index_row.get("move_g1_path", ""),
        "target_provenance": config.target_provenance,
        "source_provenance": "soma_proportional_bvh",
    }
    source_meta = _read_bvh_meta(source_tar, str(base["move_soma_proportional_path"]))
    g1_meta = _read_g1_meta(g1_tar, str(base["move_g1_path"]))
    return summarize_pair_meta(base, source_meta, g1_meta, config)


def summarize_pair_meta(
    base: Mapping[str, object],
    source_meta: Mapping[str, object],
    g1_meta: Mapping[str, object],
    config: PairQualityConfig,
) -> dict[str, object]:
    """Summarize pair-level consistency from parsed source and target metadata."""

    flags: list[str] = []
    action = "keep"
    flags.extend(str(flag) for flag in source_meta.get("flags", ()))
    flags.extend(str(flag) for flag in g1_meta.get("flags", ()))
    if source_meta.get("present") is False or g1_meta.get("present") is False:
        action = "exclude"

    source_frames = int(source_meta.get("frame_count", 0) or 0)
    source_declared = int(source_meta.get("declared_frames", 0) or 0)
    source_frame_time = float(source_meta.get("frame_time", 0.0) or 0.0)
    g1_frames = int(g1_meta.get("frame_count", 0) or 0)
    source_fps = 1.0 / source_frame_time if source_frame_time > 0 else 0.0
    source_duration = _duration_seconds(source_frames, source_fps)
    g1_duration = _duration_seconds(g1_frames, config.g1_fps)
    frame_count_delta = g1_frames - source_frames
    duration_delta_sec = g1_duration - source_duration
    metadata_duration_frames = _maybe_int(base.get("move_duration_frames"))

    if source_frames and source_declared and source_frames != source_declared:
        flags.append("pair_source_declared_frame_mismatch")
        action = _worse_action(action, "quarantine")
    if metadata_duration_frames is not None and source_frames:
        if metadata_duration_frames != source_frames:
            flags.append("pair_metadata_source_frame_mismatch")
            action = _worse_action(action, "quarantine")
    if metadata_duration_frames is not None and g1_frames:
        if metadata_duration_frames != g1_frames:
            flags.append("pair_metadata_g1_frame_mismatch")
            action = _worse_action(action, "quarantine")
    if source_frames and g1_frames:
        if abs(frame_count_delta) > config.max_frame_count_delta:
            flags.append("pair_frame_count_mismatch")
            action = _worse_action(action, "quarantine")
        if abs(duration_delta_sec) > config.max_duration_delta_sec:
            flags.append("pair_duration_mismatch")
            action = _worse_action(action, "quarantine")
    if config.expected_source_frame_time is not None and source_frame_time:
        if abs(source_frame_time - config.expected_source_frame_time) > config.frame_time_tolerance:
            flags.append("pair_source_frame_time_mismatch")
            action = _worse_action(action, "quarantine")
    if not config.target_provenance:
        flags.append("pair_missing_target_provenance")
        action = _worse_action(action, "quarantine")

    return {
        **base,
        "source_present": bool(source_meta.get("present", False)),
        "g1_present": bool(g1_meta.get("present", False)),
        "source_declared_frames": source_declared,
        "source_frame_count": source_frames,
        "g1_frame_count": g1_frames,
        "metadata_duration_frames": metadata_duration_frames if metadata_duration_frames is not None else "",
        "source_frame_time": round(source_frame_time, 8),
        "source_fps": round(source_fps, 6),
        "g1_fps": round(config.g1_fps, 6),
        "source_duration_sec": round(source_duration, 6),
        "g1_duration_sec": round(g1_duration, 6),
        "frame_count_delta": frame_count_delta,
        "abs_frame_count_delta": abs(frame_count_delta),
        "duration_delta_sec": round(duration_delta_sec, 6),
        "abs_duration_delta_sec": round(abs(duration_delta_sec), 6),
        "quality_action": action,
        "quality_flags": "|".join(dict.fromkeys(flags)),
    }


def _read_bvh_meta(tar: tarfile.TarFile, source_path: str) -> dict[str, object]:
    if not source_path:
        return {"present": False, "flags": ("missing_source_bvh_path",)}
    try:
        member = tar.getmember(source_path)
        extracted = tar.extractfile(member)
    except (KeyError, tarfile.TarError):
        return {"present": False, "flags": ("missing_source_bvh_member",)}
    if extracted is None:
        return {"present": False, "flags": ("empty_source_bvh_member",)}

    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8").read()
        except UnicodeDecodeError:
            return {"present": False, "flags": ("source_bvh_decode_error",)}
    try:
        return _parse_bvh_meta(text)
    except ValueError:
        return {"present": False, "flags": ("source_bvh_parse_error",)}


def _parse_bvh_meta(text: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        motion_idx = lines.index("MOTION")
    except ValueError as exc:
        raise ValueError("BVH missing MOTION section") from exc
    if motion_idx + 2 >= len(lines):
        raise ValueError("BVH MOTION section is incomplete")
    declared_frames = int(_parse_header_value(lines[motion_idx + 1], "Frames"))
    frame_time = float(_parse_header_value(lines[motion_idx + 2], "Frame Time"))
    motion_lines = lines[motion_idx + 3 :]
    return {
        "present": True,
        "flags": (),
        "declared_frames": declared_frames,
        "frame_count": len(motion_lines),
        "frame_time": frame_time,
    }


def _read_g1_meta(tar: tarfile.TarFile, target_path: str) -> dict[str, object]:
    if not target_path:
        return {"present": False, "flags": ("missing_g1_csv_path",)}
    try:
        member = tar.getmember(target_path)
        extracted = tar.extractfile(member)
    except (KeyError, tarfile.TarError):
        return {"present": False, "flags": ("missing_g1_csv_member",)}
    if extracted is None:
        return {"present": False, "flags": ("empty_g1_csv_member",)}

    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8", newline="")
            rows = list(csv.DictReader(text))
        except UnicodeDecodeError:
            return {"present": False, "flags": ("g1_csv_decode_error",)}
        except csv.Error:
            return {"present": False, "flags": ("g1_csv_parse_error",)}
    return {"present": True, "flags": (), "frame_count": len(rows)}


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


def _quality_run_name(index_csv: Path, limit: int | None, sample_by: Sequence[str] = ()) -> str:
    return f"{index_csv.parent.name}_pair_{sampling_run_tag(limit, sample_by)}"


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "source_frame_count",
        "g1_frame_count",
        "abs_frame_count_delta",
        "source_duration_sec",
        "g1_duration_sec",
        "abs_duration_delta_sec",
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


def _parse_header_value(line: str, name: str) -> float:
    prefix = f"{name}:"
    if not line.startswith(prefix):
        raise ValueError(f"expected BVH header line {prefix}")
    return float(line[len(prefix) :].strip())


def _duration_seconds(frame_count: int, fps: float) -> float:
    return frame_count / fps if frame_count > 0 and fps > 0 else 0.0


def _maybe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _worse_action(left: str, right: str) -> str:
    order = {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}
    return left if order[left] >= order[right] else right


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]


def _validate_config(config: PairQualityConfig) -> None:
    if config.g1_fps <= 0:
        raise ValueError("g1_fps must be positive")
    if config.frame_time_tolerance < 0:
        raise ValueError("frame_time_tolerance must be non-negative")
    if config.max_frame_count_delta < 0:
        raise ValueError("max_frame_count_delta must be non-negative")
    if config.max_duration_delta_sec < 0:
        raise ValueError("max_duration_delta_sec must be non-negative")


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
