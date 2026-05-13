"""Merge split index with source and target quality scan outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


ACTION_ORDER = {"keep": 0, "downweight": 1, "quarantine": 2, "exclude": 3}
RETAIN_ACTIONS = {"keep", "downweight"}


@dataclass(frozen=True)
class QualityMergeResult:
    output_dir: Path
    curated_index_csv: Path
    report_json: Path
    worst_clips_csv: Path
    row_count: int
    merged_source_rows: int
    merged_source_fk_rows: int
    merged_g1_rows: int
    action_counts: dict[str, int]
    flag_counts: dict[str, int]
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["curated_index_csv"] = str(self.curated_index_csv)
        payload["report_json"] = str(self.report_json)
        payload["worst_clips_csv"] = str(self.worst_clips_csv)
        return payload


def merge_quality_stats(
    split_index_csv: Path,
    output_root: Path,
    source_stats_jsonl: Path | None = None,
    source_fk_stats_jsonl: Path | None = None,
    g1_stats_jsonl: Path | None = None,
    run_name: str = "merged_quality",
) -> QualityMergeResult:
    """Merge quality scan JSONL files into a curated index CSV."""

    source_stats = _read_stats_by_row(source_stats_jsonl) if source_stats_jsonl else {}
    source_fk_stats = _read_stats_by_row(source_fk_stats_jsonl) if source_fk_stats_jsonl else {}
    g1_stats = _read_stats_by_row(g1_stats_jsonl) if g1_stats_jsonl else {}
    output_dir = output_root.expanduser() / "curated" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    curated_index_csv = output_dir / "curated_index.csv"
    report_json = output_dir / "curated_report.json"
    worst_clips_csv = output_dir / "worst_clips.csv"

    merged_rows = []
    with split_index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            merged_rows.append(
                _merge_row(
                    row,
                    source_stats.get(row["row_index"]),
                    source_fk_stats.get(row["row_index"]),
                    g1_stats.get(row["row_index"]),
                )
            )

    _write_csv(curated_index_csv, merged_rows)
    _write_csv(worst_clips_csv, _worst_clip_rows(merged_rows, source_stats, source_fk_stats, g1_stats))
    action_counts = Counter(row["merged_quality_action"] for row in merged_rows)
    flag_counts = Counter()
    for row in merged_rows:
        for flag in _split_flags(row["merged_quality_flags"]):
            flag_counts[flag] += 1

    report = {
        "split_index_csv": str(split_index_csv),
        "source_stats_jsonl": str(source_stats_jsonl) if source_stats_jsonl else "",
        "source_fk_stats_jsonl": str(source_fk_stats_jsonl) if source_fk_stats_jsonl else "",
        "g1_stats_jsonl": str(g1_stats_jsonl) if g1_stats_jsonl else "",
        "curated_index_csv": str(curated_index_csv),
        "worst_clips_csv": str(worst_clips_csv),
        "row_count": len(merged_rows),
        "merged_source_rows": len(source_stats),
        "merged_source_fk_rows": len(source_fk_stats),
        "merged_g1_rows": len(g1_stats),
        "action_counts": dict(sorted(action_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "breakdown": _breakdown(merged_rows),
        "diversity_loss": _diversity_loss(merged_rows),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    _write_json(report_json, report)
    return QualityMergeResult(
        output_dir=output_dir,
        curated_index_csv=curated_index_csv,
        report_json=report_json,
        worst_clips_csv=worst_clips_csv,
        row_count=len(merged_rows),
        merged_source_rows=len(source_stats),
        merged_source_fk_rows=len(source_fk_stats),
        merged_g1_rows=len(g1_stats),
        action_counts=dict(sorted(action_counts.items())),
        flag_counts=dict(sorted(flag_counts.items())),
        git_sha=report["git_sha"],
        git_dirty=report["git_dirty"],
    )


def _merge_row(
    row: Mapping[str, str],
    source_stats: Mapping[str, object] | None,
    source_fk_stats: Mapping[str, object] | None,
    g1_stats: Mapping[str, object] | None,
) -> dict[str, str]:
    source_action = str(source_stats.get("quality_action", "")) if source_stats else ""
    source_fk_action = str(source_fk_stats.get("quality_action", "")) if source_fk_stats else ""
    g1_action = str(g1_stats.get("quality_action", "")) if g1_stats else ""
    source_flags = str(source_stats.get("quality_flags", "")) if source_stats else ""
    source_fk_flags = str(source_fk_stats.get("quality_flags", "")) if source_fk_stats else ""
    g1_flags = str(g1_stats.get("quality_flags", "")) if g1_stats else ""
    actions = [
        row.get("curation_action", "keep"),
        source_action or "keep",
        source_fk_action or "keep",
        g1_action or "keep",
    ]
    merged_action = max(actions, key=lambda action: ACTION_ORDER.get(action, 0))
    flags = []
    flags.extend(_split_flags(row.get("quality_flags", "")))
    flags.extend(f"source:{flag}" for flag in _split_flags(source_flags))
    flags.extend(f"source_fk:{flag}" for flag in _split_flags(source_fk_flags))
    flags.extend(f"g1:{flag}" for flag in _split_flags(g1_flags))
    merged = dict(row)
    merged.update(
        {
            "source_quality_action": source_action,
            "source_quality_flags": source_flags,
            "source_fk_quality_action": source_fk_action,
            "source_fk_quality_flags": source_fk_flags,
            "g1_quality_action": g1_action,
            "g1_quality_flags": g1_flags,
            "merged_quality_action": merged_action,
            "merged_quality_flags": "|".join(dict.fromkeys(flags)),
        }
    )
    return {key: str(value) for key, value in merged.items()}


def _breakdown(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, dict[str, int]]]:
    dimensions = ("split", "package", "category")
    breakdown: dict[str, dict[str, Counter[str]]] = {dimension: {} for dimension in dimensions}
    for row in rows:
        action = row.get("merged_quality_action", "")
        for dimension in dimensions:
            key = row.get(dimension, "") or "unknown"
            bucket = breakdown[dimension].setdefault(key, Counter())
            bucket[action] += 1
    return {
        dimension: {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(values.items())
        }
        for dimension, values in breakdown.items()
    }


def _diversity_loss(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    dimensions = {
        "actor_uid": lambda row: _bucket(row.get("actor_uid", "")),
        "source_skeleton": _source_skeleton,
        "actor_height_bin": lambda row: _height_bin(row.get("actor_height_cm", "")),
        "actor_gender": lambda row: _bucket(row.get("actor_gender", "")),
        "package": lambda row: _bucket(row.get("package", "")),
        "category": lambda row: _bucket(row.get("category", "")),
        "split": lambda row: _bucket(row.get("split", "")),
        "mirror_status": lambda row: "mirror" if _is_true(row.get("is_mirror", "")) else "original",
    }
    return {name: _dimension_loss(rows, getter) for name, getter in dimensions.items()}


def _dimension_loss(
    rows: Sequence[Mapping[str, str]],
    key_for_row,
) -> dict[str, object]:
    buckets: dict[str, Counter[str]] = {}
    for row in rows:
        key = key_for_row(row)
        buckets.setdefault(key, Counter())[row.get("merged_quality_action", "keep")] += 1

    groups_without_retained = []
    groups_with_any_loss = []
    retained_rows = 0
    quarantined_or_excluded_rows = 0
    for key, counter in sorted(buckets.items()):
        retained = sum(counter[action] for action in RETAIN_ACTIONS)
        quarantined_or_excluded = sum(
            count for action, count in counter.items() if ACTION_ORDER.get(action, 0) >= ACTION_ORDER["quarantine"]
        )
        total = sum(counter.values())
        retained_rows += retained
        quarantined_or_excluded_rows += quarantined_or_excluded
        group = {
            "key": key,
            "total_rows": total,
            "retained_rows": retained,
            "quarantined_or_excluded_rows": quarantined_or_excluded,
            "retained_fraction": round(retained / total, 6) if total else 0.0,
            "action_counts": dict(sorted(counter.items())),
        }
        if retained == 0:
            groups_without_retained.append(group)
        if quarantined_or_excluded:
            groups_with_any_loss.append(group)

    total_groups = len(buckets)
    groups_with_retained = total_groups - len(groups_without_retained)
    return {
        "total_groups": total_groups,
        "groups_with_retained": groups_with_retained,
        "groups_without_retained": len(groups_without_retained),
        "lost_group_fraction": round(len(groups_without_retained) / total_groups, 6) if total_groups else 0.0,
        "total_rows": len(rows),
        "retained_rows": retained_rows,
        "quarantined_or_excluded_rows": quarantined_or_excluded_rows,
        "groups_without_retained_examples": groups_without_retained[:100],
        "groups_with_any_loss_examples": groups_with_any_loss[:100],
    }


def _worst_clip_rows(
    rows: Sequence[Mapping[str, str]],
    source_stats: Mapping[str, Mapping[str, object]],
    source_fk_stats: Mapping[str, Mapping[str, object]],
    g1_stats: Mapping[str, Mapping[str, object]],
    limit: int = 100,
) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        action = row.get("merged_quality_action", "keep")
        if ACTION_ORDER.get(action, 0) < ACTION_ORDER["quarantine"]:
            continue
        row_index = row.get("row_index", "")
        source = source_stats.get(row_index, {})
        source_fk = source_fk_stats.get(row_index, {})
        g1 = g1_stats.get(row_index, {})
        candidates.append(
            {
                "row_index": row_index,
                "split": row.get("split", ""),
                "actor_uid": row.get("actor_uid", ""),
                "package": row.get("package", ""),
                "category": row.get("category", ""),
                "filename": row.get("filename", ""),
                "move_soma_proportional_path": row.get("move_soma_proportional_path", ""),
                "move_g1_path": row.get("move_g1_path", ""),
                "merged_quality_action": action,
                "merged_quality_flags": row.get("merged_quality_flags", ""),
                "source_max_abs_channel_velocity": _stat(source, "max_abs_channel_velocity"),
                "source_channel_jump_rate": _stat(source, "channel_jump_rate"),
                "source_max_root_speed": _stat(source, "max_root_speed"),
                "source_fk_mean_foot_clearance": _stat(source_fk, "mean_foot_clearance"),
                "source_fk_penetration_depth": _stat(source_fk, "penetration_depth"),
                "source_fk_contact_frame_ratio": _stat(source_fk, "contact_frame_ratio"),
                "source_fk_contact_slide_rate": _stat(source_fk, "contact_slide_rate"),
                "source_fk_max_contact_slide_speed": _stat(source_fk, "max_contact_slide_speed"),
                "g1_max_abs_joint_velocity": _stat(g1, "max_abs_joint_velocity"),
                "g1_joint_jump_rate": _stat(g1, "joint_jump_rate"),
                "g1_max_root_speed": _stat(g1, "max_root_speed"),
                "g1_joint_limit_violation_rate": _stat(g1, "joint_limit_violation_rate"),
                "g1_max_joint_limit_violation": _stat(g1, "max_joint_limit_violation"),
                "g1_mean_foot_clearance": _stat(g1, "mean_foot_clearance"),
                "g1_penetration_depth": _stat(g1, "penetration_depth"),
                "g1_contact_frame_ratio": _stat(g1, "contact_frame_ratio"),
                "g1_contact_slide_rate": _stat(g1, "contact_slide_rate"),
                "g1_max_contact_slide_speed": _stat(g1, "max_contact_slide_speed"),
            }
        )
    candidates.sort(key=_worst_sort_key, reverse=True)
    return candidates[:limit]


def _worst_sort_key(row: Mapping[str, str]) -> tuple[float, ...]:
    return (
        ACTION_ORDER.get(row.get("merged_quality_action", "keep"), 0),
        _float(row.get("g1_penetration_depth", "")),
        _float(row.get("g1_contact_slide_rate", "")),
        _float(row.get("source_fk_penetration_depth", "")),
        _float(row.get("source_fk_contact_slide_rate", "")),
        _float(row.get("g1_joint_limit_violation_rate", "")),
        _float(row.get("g1_joint_jump_rate", "")),
        _float(row.get("source_channel_jump_rate", "")),
        _float(row.get("g1_max_abs_joint_velocity", "")),
        _float(row.get("source_max_abs_channel_velocity", "")),
        _float(row.get("g1_max_root_speed", "")),
    )


def _stat(stats: Mapping[str, object], key: str) -> str:
    value = stats.get(key, "")
    return "" if value is None else str(value)


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _read_stats_by_row(path: Path) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {path}") from exc
            row_index = str(row.get("row_index", ""))
            if row_index:
                stats[row_index] = row
    return stats


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_flags(flags: str) -> list[str]:
    if not flags:
        return []
    return [flag for flag in flags.split("|") if flag]


def _source_skeleton(row: Mapping[str, str]) -> str:
    shape_path = row.get("move_soma_proportional_shape_path", "")
    if shape_path:
        return Path(shape_path).stem or _bucket(row.get("actor_uid", ""))
    return _bucket(row.get("actor_uid", ""))


def _height_bin(value: str) -> str:
    try:
        height = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if height < 160:
        return "<160cm"
    if height < 170:
        return "160-170cm"
    if height < 180:
        return "170-180cm"
    if height < 190:
        return "180-190cm"
    return ">=190cm"


def _bucket(value: str | None) -> str:
    return value or "unknown"


def _is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
