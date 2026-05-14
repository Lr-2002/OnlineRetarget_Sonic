"""Build balanced review CSVs from quality scan JSONL artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


DEFAULT_REVIEW_COLUMNS = (
    "quality_action",
    "quality_flags",
    "row_index",
    "split",
    "category",
    "actor_uid",
    "filename",
    "move_soma_proportional_path",
    "move_g1_path",
)
DEFAULT_METRIC_COLUMNS = (
    "max_abs_joint_velocity",
    "joint_jump_rate",
    "max_abs_joint_acceleration",
    "max_root_acceleration",
    "max_root_jerk",
    "max_root_speed",
    "max_start_end_root_speed",
    "joint_limit_violation_rate",
    "max_joint_limit_violation",
    "mean_foot_clearance",
    "penetration_depth",
    "contact_frame_ratio",
    "contact_slide_rate",
    "max_contact_slide_speed",
    "self_collision_proxy_rate",
    "min_self_collision_distance",
)


@dataclass(frozen=True)
class BalancedReviewExportResult:
    output_csv: Path
    report_json: Path
    input_jsonl: Path
    exported_rows: int
    family_counts: dict[str, int]
    flag_counts: dict[str, int]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_csv"] = str(self.output_csv)
        payload["report_json"] = str(self.report_json)
        payload["input_jsonl"] = str(self.input_jsonl)
        return payload


def export_balanced_quality_review_csv(
    *,
    stats_jsonl: Path,
    output_csv: Path,
    split_index_csv: Path | None = None,
    output_report_json: Path | None = None,
    flags: Sequence[str] = (),
    max_per_flag: int = 2,
    action_min_rank: str = "quarantine",
    include_downweight: bool = False,
) -> BalancedReviewExportResult:
    """Write a flag-balanced review CSV from a scanner JSONL file."""

    if max_per_flag <= 0:
        raise ValueError("max_per_flag must be positive")
    requested_flags = tuple(flags) or _discover_flags(stats_jsonl)
    if not requested_flags:
        raise ValueError("no quality flags found")

    split_rows = _read_split_rows(split_index_csv) if split_index_csv is not None else {}
    rows = [_merge_split_row(row, split_rows) for row in _read_jsonl(stats_jsonl)]
    action_floor = "downweight" if include_downweight else action_min_rank
    selected: list[dict[str, object]] = []
    seen_row_flag: set[tuple[str, str]] = set()
    family_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for flag in requested_flags:
        candidates = [
            row
            for row in rows
            if flag in _split_flags(str(row.get("quality_flags", "")))
            and _action_rank(str(row.get("quality_action", ""))) >= _action_rank(action_floor)
        ]
        candidates.sort(key=lambda row: _quality_sort_key(row, flag), reverse=True)
        added = 0
        for row in candidates:
            row_key = str(row.get("row_index", ""))
            key = (flag, row_key)
            if key in seen_row_flag:
                continue
            selected_row = _review_row(row, flag)
            selected.append(selected_row)
            seen_row_flag.add(key)
            family_counts[flag] += 1
            for row_flag in _split_flags(str(row.get("quality_flags", ""))):
                flag_counts[row_flag] += 1
            added += 1
            if added >= max_per_flag:
                break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, selected)
    report_json = output_report_json or output_csv.with_suffix(".report.json")
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    report = {
        "input_jsonl": str(stats_jsonl),
        "split_index_csv": str(split_index_csv) if split_index_csv is not None else "",
        "output_csv": str(output_csv),
        "requested_flags": list(requested_flags),
        "max_per_flag": max_per_flag,
        "action_min_rank": action_min_rank,
        "include_downweight": include_downweight,
        "exported_rows": len(selected),
        "family_counts": dict(sorted(family_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BalancedReviewExportResult(
        output_csv=output_csv,
        report_json=report_json,
        input_jsonl=stats_jsonl,
        exported_rows=len(selected),
        family_counts=dict(sorted(family_counts.items())),
        flag_counts=dict(sorted(flag_counts.items())),
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


def _review_row(row: Mapping[str, object], family_flag: str) -> dict[str, object]:
    output = {column: row.get(column, "") for column in DEFAULT_REVIEW_COLUMNS}
    output["review_family"] = family_flag
    output["quality_action"] = row.get("quality_action", "")
    output["quality_flags"] = row.get("quality_flags", "")
    for column in DEFAULT_METRIC_COLUMNS:
        output[column] = row.get(column, "")
    return output


def _merge_split_row(
    row: Mapping[str, object],
    split_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    row_index = str(row.get("row_index", ""))
    merged: dict[str, object] = dict(row)
    for key, value in split_rows.get(row_index, {}).items():
        if merged.get(key) in (None, ""):
            merged[key] = value
    return merged


def _read_split_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_index = row.get("row_index", "")
            if row_index:
                rows[row_index] = row
    return rows


def _discover_flags(stats_jsonl: Path) -> tuple[str, ...]:
    flags: Counter[str] = Counter()
    for row in _read_jsonl(stats_jsonl):
        for flag in _split_flags(str(row.get("quality_flags", ""))):
            flags[flag] += 1
    return tuple(flag for flag, _ in flags.most_common())


def _quality_sort_key(row: Mapping[str, object], flag: str) -> tuple[float, ...]:
    if flag == "g1_ground_penetration":
        primary = _float(row.get("penetration_depth"))
    elif flag == "g1_foot_slide":
        primary = _float(row.get("contact_slide_rate"))
    elif flag == "g1_joint_limit_violation":
        primary = _float(row.get("joint_limit_violation_rate"))
    elif flag == "g1_self_collision_proxy":
        primary = _float(row.get("self_collision_proxy_rate"))
    elif flag == "g1_foot_float":
        primary = _float(row.get("mean_foot_clearance"))
    elif flag == "g1_low_foot_contact":
        primary = 1.0 - _float(row.get("contact_frame_ratio"))
    elif flag == "joint_velocity_jump":
        primary = _float(row.get("max_abs_joint_velocity"))
    elif flag == "g1_unstable_start_end":
        primary = _float(row.get("max_start_end_root_speed"))
    else:
        primary = 0.0
    return (
        primary,
        _float(row.get("penetration_depth")),
        _float(row.get("contact_slide_rate")),
        _float(row.get("joint_limit_violation_rate")),
        _float(row.get("self_collision_proxy_rate")),
        _float(row.get("max_abs_joint_velocity")),
        _float(row.get("max_root_speed")),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {path}") from exc
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [*DEFAULT_REVIEW_COLUMNS, "review_family", *DEFAULT_METRIC_COLUMNS]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_flags(flags: str) -> list[str]:
    return [flag for flag in flags.split("|") if flag]


def _action_rank(action: str) -> int:
    return {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}.get(action, 0)


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
