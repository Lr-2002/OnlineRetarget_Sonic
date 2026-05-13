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

from .bones_seed import G1_JOINT_COLUMNS


ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")


@dataclass(frozen=True)
class G1QualityConfig:
    fps: float = 30.0
    max_joint_velocity: float = 20.0
    max_root_speed: float = 8.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


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
) -> G1QualityScanResult:
    """Scan G1 CSV targets referenced by a split index."""

    config = config or G1QualityConfig()
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity must be positive")
    if config.max_root_speed <= 0:
        raise ValueError("max_root_speed must be positive")

    rows = list(_iter_index_rows(index_csv, splits=splits, actions=actions))
    if limit is not None:
        rows = rows[:limit]

    output_dir = output_root.expanduser() / "quality" / _quality_run_name(index_csv, limit)
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
            scanned.append(scan_g1_csv_member(tar, row, config))

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
) -> dict[str, object]:
    """Scan one G1 CSV member from an open tar archive."""

    target_path = index_row.get("move_g1_path", "")
    base = {
        "row_index": index_row.get("row_index", ""),
        "split": index_row.get("split", ""),
        "actor_uid": index_row.get("actor_uid", ""),
        "move_name": index_row.get("move_name", ""),
        "filename": index_row.get("filename", ""),
        "move_g1_path": target_path,
    }
    try:
        member = tar.getmember(target_path)
        extracted = tar.extractfile(member)
    except (KeyError, tarfile.TarError):
        return {
            **base,
            "frame_count": 0,
            "joint_dim": 0,
            "nonfinite_values": 0,
            "max_abs_joint_velocity": 0.0,
            "mean_abs_joint_velocity": 0.0,
            "joint_jump_rate": 0.0,
            "max_root_speed": 0.0,
            "root_jump_rate": 0.0,
            "quality_action": "exclude",
            "quality_flags": "missing_g1_csv_member",
        }
    if extracted is None:
        return {
            **base,
            "frame_count": 0,
            "joint_dim": 0,
            "nonfinite_values": 0,
            "max_abs_joint_velocity": 0.0,
            "mean_abs_joint_velocity": 0.0,
            "joint_jump_rate": 0.0,
            "max_root_speed": 0.0,
            "root_jump_rate": 0.0,
            "quality_action": "exclude",
            "quality_flags": "empty_g1_csv_member",
        }

    with extracted:
        try:
            text = io.TextIOWrapper(extracted, encoding="utf-8", newline="")
            rows = list(csv.DictReader(text))
        except UnicodeDecodeError:
            return {
                **base,
                "frame_count": 0,
                "joint_dim": 0,
                "nonfinite_values": 0,
                "max_abs_joint_velocity": 0.0,
                "mean_abs_joint_velocity": 0.0,
                "joint_jump_rate": 0.0,
                "max_root_speed": 0.0,
                "root_jump_rate": 0.0,
                "quality_action": "exclude",
                "quality_flags": "g1_csv_decode_error",
            }

    return _summarize_g1_rows(base, rows, config)


def _summarize_g1_rows(
    base: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    config: G1QualityConfig,
) -> dict[str, object]:
    flags: list[str] = []
    nonfinite_values = 0
    prev_joints: list[float] | None = None
    prev_root: list[float] | None = None
    joint_velocity_samples: list[float] = []
    root_speed_samples: list[float] = []

    for row in rows:
        joints = [_parse_float(row.get(column)) for column in G1_JOINT_COLUMNS]
        root = [_parse_float(row.get(column)) for column in ROOT_TRANSLATE_COLUMNS]
        if any(value is None for value in joints + root):
            nonfinite_values += sum(value is None for value in joints + root)
            continue
        typed_joints = [float(value) for value in joints if value is not None]
        typed_root = [float(value) for value in root if value is not None]
        if prev_joints is not None:
            joint_velocity_samples.extend(
                abs(cur - prev) * config.fps for cur, prev in zip(typed_joints, prev_joints)
            )
        if prev_root is not None:
            root_speed_samples.append(math.dist(typed_root, prev_root) * config.fps)
        prev_joints = typed_joints
        prev_root = typed_root

    frame_count = len(rows)
    max_abs_joint_velocity = max(joint_velocity_samples) if joint_velocity_samples else 0.0
    mean_abs_joint_velocity = (
        sum(joint_velocity_samples) / len(joint_velocity_samples)
        if joint_velocity_samples
        else 0.0
    )
    joint_jump_rate = _rate_above(joint_velocity_samples, config.max_joint_velocity)
    max_root_speed = max(root_speed_samples) if root_speed_samples else 0.0
    root_jump_rate = _rate_above(root_speed_samples, config.max_root_speed)

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

    return {
        **base,
        "frame_count": frame_count,
        "joint_dim": len(G1_JOINT_COLUMNS),
        "nonfinite_values": nonfinite_values,
        "max_abs_joint_velocity": round(max_abs_joint_velocity, 6),
        "mean_abs_joint_velocity": round(mean_abs_joint_velocity, 6),
        "joint_jump_rate": round(joint_jump_rate, 6),
        "max_root_speed": round(max_root_speed, 6),
        "root_jump_rate": round(root_jump_rate, 6),
        "quality_action": action,
        "quality_flags": "|".join(flags),
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


def _quality_run_name(index_csv: Path, limit: int | None) -> str:
    limit_tag = "full" if limit is None else f"limit{limit}"
    return f"{index_csv.parent.name}_{limit_tag}"


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


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "frame_count",
        "max_abs_joint_velocity",
        "mean_abs_joint_velocity",
        "joint_jump_rate",
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
